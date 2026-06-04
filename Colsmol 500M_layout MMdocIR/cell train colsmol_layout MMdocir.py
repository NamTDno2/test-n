print(">>> CELL 1: Load Training Corpus → Embed → Train RVQ (KMeans) → Save Codebook")
print(">>> Offline training — runs ONCE, zero test-set leakage")

import gc, os, io, time, pickle, random, subprocess, sys, types
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm.notebook import tqdm

# ── Install vector-quantize-pytorch ──────────────────────────────────────────
_rvq_wheel_dir = "/kaggle/input/datasets/thinam4/rvq-wheels/rvq_wheels"
if os.path.isdir(_rvq_wheel_dir):
    subprocess.run(
        ["pip", "install", "--quiet", "--no-index",
         "--find-links", _rvq_wheel_dir, "vector-quantize-pytorch"],
        check=True
    )
from vector_quantize_pytorch import ResidualVQ

# ==============================================================================
# CONFIG
# ==============================================================================
TRAIN_PARQUET_DIR = "/kaggle/input/datasets/nguyenducdung1107/training-dataset-mmdocir/MMDOC_TRAINING/parquet"
BASE_MODEL        = "/kaggle/input/datasets/nguyenducdung1107/model500m"
LORA_PATH         = "/kaggle/input/datasets/nguyenducdung1107/colsmol500-adapter/colsmol500_adapter"
WORKING_DIR       = "/kaggle/working"

# ── Output file ───────────────────────────────────────────────────────────────
CODEBOOK_SAVE_KMEANS_NQ16 = os.path.join(WORKING_DIR, "rvq_codebooks_kmeans_nq16.pt")
CODEBOOK_SAVE_KMEANS_NQ32 = os.path.join(WORKING_DIR, "rvq_codebooks_kmeans_nq32.pt")

# ── Dataset selection ─────────────────────────────────────────────────────────
ALL_DATASETS  = ["SlideVQA", "ArxivQA", "MP-DocVQA", "TAT-DQA",
                 "SciQAG", "Wiki-ss", "DUDE"]
SAMPLE_FRAC   = 0.5
MAX_PAGES     = 80_000
RANDOM_SEED   = 42
EMBED_BATCH   = 16

# ── RVQ architecture ──────────────────────────────────────────────────────────
RVQ_N_QUANTIZERS_COARSE = 16
RVQ_N_QUANTIZERS_FINE   = 32
RVQ_CODEBOOK_SIZE       = 256

# ── RVQ Training hyper-params ─────────────────────────────────────────────────
RVQ_TRAINING_EPOCHS     = 120
RVQ_EARLY_STOP_PATIENCE = 20
RVQ_TRAINING_BATCH_SIZE = 4096
RVQ_TRAINING_SAMPLES    = 200_000
RVQ_DROPOUT_P           = 0.5
RVQ_DROPOUT_WARMUP_EPS  = 10

# ── Norm filter ───────────────────────────────────────────────────────────────
TOP_K_NORM_FRAC = 0.5

# ── Ranking loss hyper-params ─────────────────────────────────────────────────
# Contrastive (triplet margin) loss weight and margin
RANK_LOSS_WEIGHT   = 1.0          # λ_rank in L = L_compress + λ_rank * L_rank
RANK_MARGIN        = 0.2          # margin γ in max(0, γ - MaxSim(q,pos) + MaxSim(q,neg))
RANK_NEG_SAMPLES   = 8            # number of in-batch negatives per anchor

# OT (Wasserstein) loss hyper-params
OT_LOSS_WEIGHT     = 0.3          # λ_OT in total loss
OT_SINKHORN_ITERS  = 20           # Sinkhorn iterations for entropic OT
OT_SINKHORN_EPS    = 0.05         # entropic regularisation ε (smaller = closer to exact OT)

_EMBED_KEYS = ['embedding', 'embeddings', 'patches', 'tokens', 'features', 'data']

META_COLS = ["file_name", "page"]

# ==============================================================================
# STEP 2: LOAD + SAMPLE TRAINING PARQUET (lazy, low-mem)
# ==============================================================================
print("\n" + "="*70)
print("STEP 2: Load training parquet (lazy sampling)")
print("="*70)

def _sample_parquet_meta(path, dataset_name, frac, seed, max_rows=None):
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    target = max(1, int(total * frac))
    if max_rows: target = min(target, max_rows)

    rng = random.Random(seed)
    group_ids = list(range(pf.num_row_groups))
    rng.shuffle(group_ids)

    frames = []; selected = 0
    for rg in group_ids:
        if selected >= target: break
        try:
            table = pf.read_row_group(rg, columns=META_COLS)
        except Exception as e:
            print(f"  [WARN] skip rg={rg}: {e}"); continue
        part = table.to_pandas().reset_index(drop=True); del table
        part["row_group_id"] = rg
        part["row_in_group"]  = range(len(part))
        left = target - selected
        if len(part) > left:
            part = part.sample(n=left, random_state=seed + rg).reset_index(drop=True)
        part["dataset"]        = dataset_name
        part["source_parquet"] = path
        frames.append(part); selected += len(part)
        del part; gc.collect()

    if not frames:
        return pd.DataFrame(), total
    return pd.concat(frames, ignore_index=True), total

dfs = []
for name in ALL_DATASETS:
    path = os.path.join(TRAIN_PARQUET_DIR, f"{name}_filter.parquet")
    if not os.path.exists(path):
        print(f"  [SKIP] {name}: not found"); continue
    try:
        df_tmp, n_full = _sample_parquet_meta(path, name, SAMPLE_FRAC, RANDOM_SEED)
        dfs.append(df_tmp)
        print(f"  {name}: {n_full} rows → sampled {len(df_tmp)}")
        del df_tmp; gc.collect()
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")

if not dfs:
    raise RuntimeError("No dataset loaded. Check parquet paths.")

layouts_df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
del dfs; gc.collect()

if MAX_PAGES and len(layouts_df) > MAX_PAGES:
    layouts_df = layouts_df.sample(n=MAX_PAGES, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"  Capped to {MAX_PAGES} pages")

layouts_df["join_doc_name"] = (
    layouts_df["dataset"] + "__" + layouts_df["file_name"].astype(str))
layouts_df = layouts_df.sort_values(
    ["source_parquet", "row_group_id", "row_in_group"]).reset_index(drop=True)

print(f"\n  Total pages: {len(layouts_df)}")
print(f"  Dataset distribution:\n{layouts_df['dataset'].value_counts().to_string()}")

# ==============================================================================
# STEP 3: EMBED TRAINING PAGES
# ==============================================================================
print("\n" + "="*70)
print("STEP 3: Embed training pages with ColSmolVLM")
print("="*70)

def _build_text_from_layouts(layouts):
    if not layouts: return "Document page."
    text_keys = ["text", "ocr_text", "content", "caption", "value"]
    texts = []
    for lay in layouts:
        if not isinstance(lay, dict): continue
        for tk in text_keys:
            v = lay.get(tk)
            if v and isinstance(v, str) and len(v.strip()) > 2:
                texts.append(v.strip()); break
    return " ".join(texts)[:1000] if texts else "Document page."

def embed_pages_from_parquet(layouts_df, model, proc, dev, batch_size=EMBED_BATCH):
    n = len(layouts_df)
    parquet_cache = {}
    fused_index   = [None] * n

    def _get_pf(path):
        if path not in parquet_cache:
            parquet_cache[path] = pq.ParquetFile(path)
        return parquet_cache[path]

    batch_imgs, batch_texts, batch_pos = [], [], []
    grouped = layouts_df.groupby(["source_parquet", "row_group_id"], sort=False)

    for (source_path, rg), group_df in tqdm(
            grouped, total=grouped.ngroups, desc="  Row groups"):
        try:
            pf    = _get_pf(source_path)
            table = pf.read_row_group(int(rg), columns=["layouts", "image"])
            rg_data = table.to_pydict(); del table
        except Exception as e:
            print(f"  [WARN] read error ({source_path}, rg={rg}): {e}"); continue

        layouts_col = rg_data.get("layouts", [])
        image_col   = rg_data.get("image",   [])

        for _, row in group_df.iterrows():
            rig = int(row["row_in_group"])
            pos = int(row.name)

            try:
                layouts = layouts_col[rig] if rig < len(layouts_col) else None
                img_obj = image_col[rig]   if rig < len(image_col)   else None
                raw = (img_obj if isinstance(img_obj, bytes)
                       else img_obj.get("bytes") if isinstance(img_obj, dict)
                       else None)
                img  = (Image.open(io.BytesIO(raw)).convert("RGB")
                        if raw and len(raw) > 0
                        else Image.new("RGB", (224, 224), "white"))
                if img.width < 14 or img.height < 14:
                    img = Image.new("RGB", (224, 224), "white")
                text = _build_text_from_layouts(layouts)
            except Exception:
                img  = Image.new("RGB", (224, 224), "white")
                text = "Document page."

            batch_imgs.append(img)
            batch_texts.append(str(text).replace("<image>", " ")[:1000])
            batch_pos.append(pos)

            if len(batch_imgs) >= batch_size:
                with torch.no_grad():
                    try:
                        vis_inp = proc.process_images(batch_imgs).to(dev)
                        out_vis = model(**vis_inp)
                        txt_inp = proc.process_queries(
                            batch_texts, max_length=192, suffix="").to(dev)
                        out_txt = model(**txt_inp)
                        for k in range(len(batch_imgs)):
                            vis_e = out_vis[k].cpu().float().numpy()
                            txt_e = out_txt[k].cpu().float().numpy()
                            fused = np.concatenate([vis_e, txt_e], axis=0).astype(np.float16)
                            fused_index[batch_pos[k]] = fused
                        del vis_inp, txt_inp, out_vis, out_txt
                    except Exception as e:
                        print(f"  [WARN] embed error: {e}")
                batch_imgs.clear(); batch_texts.clear(); batch_pos.clear()
                torch.cuda.empty_cache(); gc.collect()

        del rg_data; gc.collect()

    if batch_imgs:
        with torch.no_grad():
            try:
                vis_inp = proc.process_images(batch_imgs).to(dev)
                out_vis = model(**vis_inp)
                txt_inp = proc.process_queries(
                    batch_texts, max_length=192, suffix="").to(dev)
                out_txt = model(**txt_inp)
                for k in range(len(batch_imgs)):
                    vis_e = out_vis[k].cpu().float().numpy()
                    txt_e = out_txt[k].cpu().float().numpy()
                    fused = np.concatenate([vis_e, txt_e], axis=0).astype(np.float16)
                    fused_index[batch_pos[k]] = fused
            except Exception as e:
                print(f"  [WARN] final batch error: {e}")
        torch.cuda.empty_cache(); gc.collect()

    parquet_cache.clear()
    fused_index = [e for e in fused_index if e is not None]
    return fused_index

t0 = time.perf_counter()
fused_train = embed_pages_from_parquet(
    layouts_df, model, processor, device, EMBED_BATCH)
print(f"\n  Embedded {len(fused_train)} pages in {time.perf_counter()-t0:.1f}s")

if len(fused_train) == 0:
    raise RuntimeError("No pages embedded. Check parquet + model paths.")

emb_dim = fused_train[0].shape[1]
print(f"  Embedding dim per patch: {emb_dim}")

del model, processor
gc.collect(); torch.cuda.empty_cache()
print("  Model freed from VRAM")

# ==============================================================================
# STEP 4: NORM FILTER + PATCH SAMPLING
# ==============================================================================
print("\n" + "="*70)
print("STEP 4: Norm-filter patches + sample training pool")
print("="*70)

def top_norm_filter(arr: np.ndarray, frac: float = TOP_K_NORM_FRAC) -> np.ndarray:
    """Keep top-frac patches by L2 norm. Removes whitespace/background noise."""
    L = arr.shape[0]
    if L <= 1 or frac >= 1.0:
        return arr
    k = max(1, int(L * frac))
    norms   = np.linalg.norm(arr.astype(np.float32), axis=-1)
    top_idx = np.argpartition(norms, -k)[-k:]
    return arr[np.sort(top_idx)]

def sample_patches(emb_list, n_samples: int, seed: int = 42) -> torch.Tensor:
    rng      = np.random.default_rng(seed)
    filtered = [top_norm_filter(e.astype(np.float32)) for e in emb_list]
    all_lens = [a.shape[0] for a in filtered]
    total    = sum(all_lens)
    n_take   = min(n_samples, total)
    flat_idx = np.sort(rng.choice(total, n_take, replace=False))
    D        = filtered[0].shape[1]
    result   = np.empty((n_take, D), dtype=np.float32)
    cumsum = out = 0
    it = iter(flat_idx); nfi = next(it, None)
    for pi, L in enumerate(all_lens):
        pe = cumsum + L
        while nfi is not None and nfi < pe:
            result[out] = filtered[pi][nfi - cumsum]; out += 1; nfi = next(it, None)
        cumsum = pe
        if nfi is None: break
    print(f"  Sampled {out:,} / {total:,} patches "
          f"({out/max(total,1)*100:.1f}% of norm-filtered pool)")
    return torch.from_numpy(result[:out])

train_raw  = sample_patches(fused_train, RVQ_TRAINING_SAMPLES)
train_data = F.normalize(train_raw, dim=-1)
del train_raw, fused_train; gc.collect()
print(f"  train_data: {train_data.shape}  (L2-normalized, unit hypersphere)")

# ==============================================================================
# SHARED UTILITIES: RANKING LOSSES (used by KMeans pipeline)
# ==============================================================================

def maxsim_score(query_patches: torch.Tensor,
                 doc_patches:   torch.Tensor) -> torch.Tensor:
    """
    Compute ColPali-style MaxSim score.

    query_patches : (Q, D)  — query patch embeddings (L2-normalized)
    doc_patches   : (P, D)  — document patch embeddings (quantized or raw)

    MaxSim(q, d) = Σ_{j=1}^{Q}  max_{k=1}^{P}  cos(q_j, d_k)
                = Σ_j  max_k  (q_j · d_k)   [since both normalized]

    Returns: scalar score
    """
    # (Q, P) cosine similarity matrix
    sim = torch.mm(query_patches, doc_patches.t())
    # Max over document patches, sum over query patches
    return sim.max(dim=1).values.sum()


def batch_maxsim(queries: torch.Tensor,
                 docs:    torch.Tensor,
                 n_q_patches: int) -> torch.Tensor:
    """
    Vectorized MaxSim over a batch.

    queries : (B * n_q_patches, D)  — flattened query patches
    docs    : (B * n_d_patches, D)  — flattened doc patches
    n_q_patches: patches per query

    Returns: (B,) MaxSim scores
    """
    B = queries.shape[0] // n_q_patches
    n_d_patches = docs.shape[0] // B

    q = queries.view(B, n_q_patches, -1)   # (B, Q, D)
    d = docs.view(B, n_d_patches, -1)       # (B, P, D)

    # (B, Q, P) cosine similarity
    sim = torch.bmm(q, d.transpose(1, 2))   # (B, Q, P)
    # Max over P (doc patches), sum over Q (query patches)
    return sim.max(dim=2).values.sum(dim=1)  # (B,)


def ranking_contrastive_loss(x_orig: torch.Tensor,
                              z_q:    torch.Tensor,
                              margin: float = RANK_MARGIN,
                              n_neg:  int   = RANK_NEG_SAMPLES) -> torch.Tensor:
    """
    In-batch triplet ranking loss for MaxSim preservation.
    """
    N = x_orig.shape[0]

    # Anchor: x_orig[i], positive: z_q[i], negatives: z_q[rand j≠i]
    # Cosine similarity (both normalized → dot product)
    pos_sim = (x_orig * z_q).sum(dim=-1)          # (N,) anchor-positive scores

    # Sample n_neg negatives per anchor (in-batch)
    neg_idx = torch.zeros(N, n_neg, dtype=torch.long, device=x_orig.device)
    for i in range(N):
        pool = torch.arange(N, device=x_orig.device)
        pool = pool[pool != i]
        perm = torch.randperm(len(pool), device=x_orig.device)[:n_neg]
        neg_idx[i] = pool[perm]

    # (N, n_neg, D) negative quantized embeddings
    neg_z = z_q[neg_idx.view(-1)].view(N, n_neg, -1)

    # (N, n_neg) anchor-negative scores
    neg_sim = (x_orig.unsqueeze(1) * neg_z).sum(dim=-1)

    # Triplet hinge: max(0, margin - pos + neg)
    # Use hardest negative (highest neg_sim) for max-margin training
    hardest_neg_sim = neg_sim.max(dim=1).values   # (N,)
    loss = F.relu(margin - pos_sim + hardest_neg_sim).mean()
    return loss


def sinkhorn_log(log_alpha: torch.Tensor,
                 n_iters:   int   = OT_SINKHORN_ITERS,
                 eps:       float = OT_SINKHORN_EPS) -> torch.Tensor:
    """
    Log-domain Sinkhorn algorithm for numerical stability.
    """
    N, M = log_alpha.shape
    log_a = torch.full((N,), -np.log(N), device=log_alpha.device, dtype=log_alpha.dtype)
    log_b = torch.full((M,), -np.log(M), device=log_alpha.device, dtype=log_alpha.dtype)

    log_u = torch.zeros(N, device=log_alpha.device, dtype=log_alpha.dtype)
    log_v = torch.zeros(M, device=log_alpha.device, dtype=log_alpha.dtype)

    for _ in range(n_iters):
        # u update: log_u = log_a - logsumexp(log_alpha + log_v, dim=1)
        log_u = log_a - torch.logsumexp(
            log_alpha + log_v.unsqueeze(0), dim=1)
        # v update: log_v = log_b - logsumexp(log_alpha + log_u, dim=0)
        log_v = log_b - torch.logsumexp(
            log_alpha + log_u.unsqueeze(1), dim=0)

    # log transport plan: log_T[i,j] = log_alpha[i,j] + log_u[i] + log_v[j]
    log_T = log_alpha + log_u.unsqueeze(1) + log_v.unsqueeze(0)
    return log_T


def wasserstein_ot_loss(x_orig: torch.Tensor,
                         z_q:    torch.Tensor,
                         eps:    float = OT_SINKHORN_EPS,
                         n_iters: int  = OT_SINKHORN_ITERS) -> torch.Tensor:
    """
    Sinkhorn-Wasserstein loss between distributions of original and quantized embeddings.
    """
    N = x_orig.shape[0]

    # Subsample for efficiency
    ot_n = min(N, 512)
    idx  = torch.randperm(N, device=x_orig.device)[:ot_n]
    x    = x_orig[idx]  # (ot_n, D), no grad needed
    z    = z_q[idx]     # (ot_n, D), grad flows here

    # Angular cost matrix: C[i,j] = 1 - cos(x_i, z_j) ∈ [0, 2]
    cos_sim = torch.mm(x, z.t())         # (ot_n, ot_n)
    C = 1.0 - cos_sim                    # angular distance ∈ [0, 2]

    # Log-cost for Sinkhorn: log_alpha = -C / eps
    log_alpha = -C / eps                 # (ot_n, ot_n)

    # Sinkhorn iterations in log-domain
    with torch.no_grad():
        log_T = sinkhorn_log(log_alpha.detach(), n_iters=n_iters, eps=eps)

    # Transport plan (normalized)
    T = log_T.exp()                      # (ot_n, ot_n), no grad

    # OT loss = Σ_{i,j} C[i,j] * T[i,j]
    loss = (C * T).sum()
    return loss


# ==============================================================================
# STEP 5: KMEANS-BASED RVQ TRAINING
# ==============================================================================
print("\n" + "="*70)
print("STEP 5: KMeans-initialized RVQ")
print("="*70)

def rvq_forward_with_dropout_and_ranking(
    model:        ResidualVQ,
    x:            torch.Tensor,
    n_drop_suffix: int,
    use_rank_loss: bool = True,
    use_ot_loss:   bool = True,
) -> tuple:
    NQ       = len(model.layers)
    n_active = max(1, NQ - n_drop_suffix)

    residual   = x.clone()
    z_q_cumsum = torch.zeros_like(x)
    all_commits = []

    for k in range(n_active):
        z_q_k, _, commit_k = model.layers[k](residual.unsqueeze(1))
        z_q_k      = z_q_k.squeeze(1)
        z_q_cumsum = z_q_cumsum + z_q_k
        residual   = residual - z_q_k
        if commit_k is not None:
            all_commits.append(commit_k.sum())

    # ── Compression losses (MSE + cosine alignment) ───────────────────────
    l_recon  = F.mse_loss(z_q_cumsum.float(), x.float().detach())
    l_align  = (1.0 - F.cosine_similarity(
                    z_q_cumsum.float(), x.float(), dim=-1)).mean()
    l_commit = (torch.stack(all_commits).mean()
                if all_commits else torch.tensor(0., device=x.device))

    # ── Ranking loss: contrastive triplet on MaxSim ───────────────────────
    z_q_norm = F.normalize(z_q_cumsum.float(), dim=-1)
    x_norm   = F.normalize(x.float(), dim=-1)

    if use_rank_loss and RANK_LOSS_WEIGHT > 0:
        l_rank = ranking_contrastive_loss(
            x_norm, z_q_norm,
            margin=RANK_MARGIN,
            n_neg=RANK_NEG_SAMPLES,
        )
    else:
        l_rank = torch.tensor(0., device=x.device)

    # ── OT loss: Sinkhorn-Wasserstein distributional alignment ───────────
    if use_ot_loss and OT_LOSS_WEIGHT > 0:
        l_ot = wasserstein_ot_loss(
            x_norm.detach(), z_q_norm,
            eps=OT_SINKHORN_EPS,
            n_iters=OT_SINKHORN_ITERS,
        )
    else:
        l_ot = torch.tensor(0., device=x.device)

    total = (l_recon
             + 0.5  * l_align
             + 0.25 * l_commit
             + RANK_LOSS_WEIGHT * l_rank
             + OT_LOSS_WEIGHT   * l_ot)

    losses = {
        'recon':  l_recon.item(),
        'align':  l_align.item(),
        'commit': l_commit.item(),
        'rank':   l_rank.item(),
        'ot':     l_ot.item(),
    }
    return total, losses, n_active


def train_rvq_kmeans(NQ: int, train_data: torch.Tensor,
                     dev: str, emb_dim: int) -> ResidualVQ:
    model = ResidualVQ(
        dim                     = emb_dim,
        num_quantizers          = NQ,
        codebook_size           = RVQ_CODEBOOK_SIZE,
        kmeans_init             = True,
        kmeans_iters            = 10,
        threshold_ema_dead_code = 2,
        commitment_weight       = 0.5,
        learnable_codebook      = True,
        ema_update              = False,
    ).to(dev)

    print(f"  KMeans init: 10 iterations of Lloyd's algorithm")
    print(f"  + Quantizer dropout training")
    print(f"  + Contrastive ranking loss (λ={RANK_LOSS_WEIGHT}, margin={RANK_MARGIN})")
    print(f"  + Sinkhorn-OT loss (λ={OT_LOSS_WEIGHT}, ε={OT_SINKHORN_EPS})")

    n     = train_data.shape[0]
    n_val = max(RVQ_TRAINING_BATCH_SIZE, int(n * 0.1))
    perm  = torch.randperm(n)
    t_data = train_data[perm[n_val:]]
    v_data = train_data[perm[:n_val]]

    from torch.utils.data import DataLoader, TensorDataset
    import torch.amp as amp

    loader = DataLoader(
        TensorDataset(t_data),
        batch_size = RVQ_TRAINING_BATCH_SIZE,
        shuffle    = True, drop_last=True,
        pin_memory = (dev == 'cuda'))

    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=RVQ_TRAINING_EPOCHS, eta_min=1e-5)
    scaler = amp.GradScaler(device=dev, enabled=(dev == 'cuda'))

    best_val   = float('inf')
    patience   = 0
    best_state = None
    rng_drop   = random.Random(42)

    for epoch in range(RVQ_TRAINING_EPOCHS):
        model.train()

        if epoch < RVQ_DROPOUT_WARMUP_EPS:
            drop_p = 0.0
        else:
            ramp   = min(1.0, (epoch - RVQ_DROPOUT_WARMUP_EPS) / 20.0)
            drop_p = RVQ_DROPOUT_P * ramp

        ep_losses = {k: 0. for k in ['recon', 'align', 'commit', 'rank', 'ot']}
        ep_total  = 0.
        nb = 0

        for (batch,) in loader:
            batch = batch.to(dev, non_blocking=True)

            n_drop = 0
            if drop_p > 0 and rng_drop.random() < drop_p:
                n_drop = min(
                    NQ - 1,
                    int(np.random.geometric(p=1 - drop_p * 0.5)) - 1)

            opt.zero_grad(set_to_none=True)
            with amp.autocast(device_type=dev, enabled=(dev == 'cuda')):
                loss, comp_losses, n_act = rvq_forward_with_dropout_and_ranking(
                    model, batch, n_drop,
                    use_rank_loss=True, use_ot_loss=True)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()

            ep_total += loss.item()
            for k, v in comp_losses.items():
                ep_losses[k] += v
            nb += 1

        sched.step()

        model.eval()
        with torch.no_grad():
            vi = torch.randperm(v_data.shape[0])[:min(4096, v_data.shape[0])]
            vb = v_data[vi].to(dev)
            with amp.autocast(device_type=dev, enabled=(dev == 'cuda')):
                val_loss, val_comp, _ = rvq_forward_with_dropout_and_ranking(
                    model, vb, n_drop_suffix=0,
                    use_rank_loss=False, use_ot_loss=False)
            val_loss = val_loss.item()

        if epoch == 0 or (epoch + 1) % 10 == 0:
            lr_now = sched.get_last_lr()[0]
            el = {k: v/max(nb,1) for k, v in ep_losses.items()}
            print(f"  [KMeans NQ={NQ}] ep{epoch+1:3d}  "
                  f"total={ep_total/max(nb,1):.5f}  "
                  f"recon={el['recon']:.5f}  "
                  f"rank={el['rank']:.5f}  "
                  f"ot={el['ot']:.5f}  "
                  f"val={val_loss:.5f}  "
                  f"drop_p={drop_p:.2f}  lr={lr_now:.2e}")

        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            patience   = 0
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience += 1

        if patience >= RVQ_EARLY_STOP_PATIENCE:
            print(f"  [KMeans NQ={NQ}] early-stop ep{epoch+1}  best_val={best_val:.5f}")
            break

    if best_state:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

    model.eval()
    return model


# ==============================================================================
# STEP 6: CODEBOOK EXTRACTION + DIAGNOSTICS (shared)
# ==============================================================================

def extract_codebooks(model: ResidualVQ, NQ: int, dev: str) -> list:
    model.eval()
    cbs = []
    with torch.no_grad():
        for qi in range(NQ):
            vq = model.layers[qi]; cb = None
            for attr in ['_codebook.embed', '_codebook.embed_avg', 'codebook']:
                try:
                    obj = vq
                    for part in attr.split('.'): obj = getattr(obj, part)
                    obj = obj.detach().float()
                    if obj.dim() == 3: obj = obj[0]
                    if obj.dim() == 2 and obj.shape[0] == RVQ_CODEBOOK_SIZE:
                        cb = obj; break
                    if obj.dim() == 2 and obj.shape[1] == RVQ_CODEBOOK_SIZE:
                        cb = obj.t(); break
                except (AttributeError, IndexError):
                    continue
            if cb is None:
                for n, p in vq.named_parameters():
                    if RVQ_CODEBOOK_SIZE in p.shape:
                        obj = p.detach().float()
                        while obj.dim() > 2: obj = obj[0]
                        if obj.shape[0] == RVQ_CODEBOOK_SIZE: cb = obj; break
                        if obj.dim() == 2 and obj.shape[1] == RVQ_CODEBOOK_SIZE:
                            cb = obj.t(); break
            if cb is None:
                raise RuntimeError(f"Cannot extract codebook stage={qi}")
            cbs.append(cb.to(dev))
    return cbs


@torch.no_grad()
def codebook_diagnostics(codebooks: list, val_sample: torch.Tensor,
                          dev: str) -> dict:
    NQ = len(codebooks)
    val_sample = val_sample.to(dev)
    residual   = val_sample.clone()
    z_q_sum    = torch.zeros_like(val_sample)
    perplexities = []
    stage_mses   = []
    dead_codes   = []

    for k, cb in enumerate(codebooks):
        inner  = torch.mm(residual, cb.t())
        res_sq = (residual ** 2).sum(-1, keepdim=True)
        cb_sq  = (cb ** 2).sum(-1).unsqueeze(0)
        dist   = res_sq - 2 * inner + cb_sq
        idx    = dist.argmin(-1)
        z_q_k  = cb[idx]

        stage_mses.append(F.mse_loss(z_q_k, residual).item())

        counts = torch.zeros(RVQ_CODEBOOK_SIZE, device=dev)
        counts.scatter_add_(0, idx, torch.ones(len(idx), device=dev))
        dead_codes.append(int((counts == 0).sum().item()))
        p = counts / counts.sum().clamp(min=1e-9)
        m = p > 0
        H = -(p[m] * p[m].log()).sum().item()
        perplexities.append(2 ** H)

        z_q_sum  = z_q_sum + z_q_k
        residual = residual - z_q_k

    final_mse   = F.mse_loss(z_q_sum, val_sample).item()
    final_align = F.cosine_similarity(z_q_sum, val_sample, dim=-1).mean().item()

    # MaxSim preservation: compare MaxSim(z_q) vs MaxSim(x) on same pairs
    n_pairs = min(256, val_sample.shape[0] // 2)
    perm    = torch.randperm(val_sample.shape[0])
    anchors   = val_sample[perm[:n_pairs]]
    docs_raw  = val_sample[perm[n_pairs:2*n_pairs]]
    docs_q    = z_q_sum[perm[n_pairs:2*n_pairs]]
    maxsim_raw  = (anchors * docs_raw).sum(dim=-1).mean().item()
    maxsim_quant = (anchors * F.normalize(docs_q, dim=-1)).sum(dim=-1).mean().item()
    maxsim_ratio = maxsim_quant / max(abs(maxsim_raw), 1e-8)

    return {
        'perplexity_per_stage': [round(p, 1) for p in perplexities],
        'stage_recon_mse':      [round(m, 6) for m in stage_mses],
        'dead_codes_per_stage': dead_codes,
        'total_recon_mse':      round(final_mse, 6),
        'cosine_alignment':     round(final_align, 4),
        'maxsim_preservation':  round(maxsim_ratio, 4),
        'mean_perplexity':      float(np.mean(perplexities)),
        'min_perplexity':       float(np.min(perplexities)),
        'total_dead_codes':     sum(dead_codes),
    }


# ==============================================================================
# STEP 7: TRAIN KMEANS PIPELINE AND SAVE CODEBOOK
# ==============================================================================

nq_levels = sorted(set([RVQ_N_QUANTIZERS_COARSE, RVQ_N_QUANTIZERS_FINE]))

# ── Shared val set
val_size = min(8192, train_data.shape[0] // 5)
val_idx  = torch.randperm(train_data.shape[0])[:val_size]
val_data = train_data[val_idx]

all_codebooks_kmeans = {}
all_diag_kmeans      = {}

# ────────────────────────────────────────────────────────────────────────────
# PIPELINE A: KMeans + Ranking Losses
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "█"*70)
print("PIPELINE A: KMeans Init + Ranking Losses (Baseline++)")
print("█"*70)

for NQ in nq_levels:
    print(f"\n{'─'*70}")
    print(f"  [KMeans] NQ={NQ}")
    print(f"{'─'*70}")
    t0 = time.perf_counter()
    model_km = train_rvq_kmeans(NQ, train_data, device, emb_dim)
    print(f"  Total elapsed: {time.perf_counter()-t0:.1f}s")

    cbs_km = extract_codebooks(model_km, NQ, device)
    all_codebooks_kmeans[NQ] = [cb.cpu() for cb in cbs_km]

    diag_km = codebook_diagnostics(cbs_km, val_data, device)
    all_diag_kmeans[NQ] = diag_km

    print(f"\n  [KMeans] Diagnostics NQ={NQ}:")
    print(f"    Alignment:          {diag_km['cosine_alignment']:.4f}")
    print(f"    MaxSim preservation:{diag_km['maxsim_preservation']:.4f}  (target >0.90)")
    print(f"    Recon MSE:          {diag_km['total_recon_mse']:.6f}")
    print(f"    Mean perplexity:    {diag_km['mean_perplexity']:.1f} / {RVQ_CODEBOOK_SIZE} "
          f"({diag_km['mean_perplexity']/RVQ_CODEBOOK_SIZE*100:.0f}% effective)")
    print(f"    Min  perplexity:    {diag_km['min_perplexity']:.1f}")
    print(f"    Dead codes:         {diag_km['total_dead_codes']} / {NQ * RVQ_CODEBOOK_SIZE}")

    del model_km; gc.collect(); torch.cuda.empty_cache()


# ==============================================================================
# STEP 8: SAVE ARTIFACT FILE
# ==============================================================================
print("\n" + "="*70)
print("STEP 8: Save artifacts")
print("="*70)

# ── File: KMeans codebooks ─────────────────────────────────────────────────
for NQ in nq_levels:
    save_path = CODEBOOK_SAVE_KMEANS_NQ16 if NQ == 16 else CODEBOOK_SAVE_KMEANS_NQ32
    torch.save({
        'codebooks':         {NQ: all_codebooks_kmeans[NQ]},
        'diagnostics':       {NQ: all_diag_kmeans[NQ]},
        'emb_dim':           emb_dim,
        'codebook_size':     RVQ_CODEBOOK_SIZE,
        'nq_coarse':         RVQ_N_QUANTIZERS_COARSE,
        'nq_fine':           RVQ_N_QUANTIZERS_FINE,
        'norm_filter':       TOP_K_NORM_FRAC,
        'n_train_pages':     len(layouts_df),
        'n_train_patches':   int(train_data.shape[0]),
        'dropout_p':         RVQ_DROPOUT_P,
        'train_datasets':    ALL_DATASETS,
        'init_method':       'KMeans',
        'rank_loss_weight':  RANK_LOSS_WEIGHT,
        'rank_margin':       RANK_MARGIN,
        'ot_loss_weight':    OT_LOSS_WEIGHT,
        'ot_sinkhorn_eps':   OT_SINKHORN_EPS,
        'loss_components':   ['recon', 'align', 'commit', 'rank_contrastive', 'sinkhorn_ot'],
    }, save_path)
    sz_km = os.path.getsize(save_path) / 1024**2
    print(f"  [A] KMeans codebook NQ={NQ} → {save_path}  ({sz_km:.1f} MB)")


# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "="*70)
print("CELL 1 COMPLETE — KMeans codebook files saved")
print("="*70)
print(f"\n  File NQ=16: {CODEBOOK_SAVE_KMEANS_NQ16}")
print(f"  File NQ=32: {CODEBOOK_SAVE_KMEANS_NQ32}")
print(f"    Init:   KMeans (Lloyd's, 10 iters)")
print(f"    Train:  Dropout + Contrastive ranking (λ={RANK_LOSS_WEIGHT}) + Sinkhorn-OT (λ={OT_LOSS_WEIGHT})")
print(f"\n  Shared losses:")
print(f"    L_total = L_recon + 0.5*L_align + 0.25*L_commit")
print(f"            + {RANK_LOSS_WEIGHT}*L_rank  [contrastive triplet, in-batch negatives]")
print(f"            + {OT_LOSS_WEIGHT}*L_OT    [Sinkhorn-Wasserstein, ε={OT_SINKHORN_EPS}]")
print(f"\n  Diagnostics summary:")
for NQ in nq_levels:
    dk = all_diag_kmeans[NQ]
    print(f"  NQ={NQ:2d}  KMeans: align={dk['cosine_alignment']:.4f}  "
          f"maxsim={dk['maxsim_preservation']:.4f}  "
          f"dead={dk['total_dead_codes']}/{NQ*RVQ_CODEBOOK_SIZE}")
print(f"\n  Load in Cell 2:")
print(f"    cbs_nq16 = torch.load('{CODEBOOK_SAVE_KMEANS_NQ16}')")
print(f"    cbs_nq32 = torch.load('{CODEBOOK_SAVE_KMEANS_NQ32}')")
print("="*70)