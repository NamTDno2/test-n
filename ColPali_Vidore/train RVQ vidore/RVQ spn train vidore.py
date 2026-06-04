# ==============================================================================
# METHOD 7 - RVQ Training on ColPali Train Set
# Train RVQ codebooks from scratch using the official ColPali training dataset.
# ==============================================================================

# -- Install vector-quantize-pytorch (offline wheel preferred) -----------------
import subprocess, io, os, gc, glob, time, json, random
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm

_rvq_wheel_dir = "/kaggle/input/datasets/thinam4/rvq-wheels/rvq_wheels"
if os.path.isdir(_rvq_wheel_dir):
    subprocess.run(
        ["pip", "install", "--quiet", "--no-index",
         "--find-links", _rvq_wheel_dir, "vector-quantize-pytorch"],
        check=True
    )
else:
    subprocess.run(["pip", "install", "--quiet", "vector-quantize-pytorch"], check=True)

from vector_quantize_pytorch import ResidualVQ

# ==============================================================================
# CONFIG - RTX PRO 6000 (Blackwell, 96 GB VRAM)
# ==============================================================================

# Training data: ColPali official train split (82 shards)
TRAIN_DATA_ROOT = "/kaggle/input/datasets/namthi/colpali-train-set"

# Output codebooks go into /kaggle/working/rvq_codebooks
OUTPUT_DIR = os.path.join(WORKING_DIR, "rvq_codebooks")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMB_DIM = 128   # ColPali projection dimension (query_model.dim == 128)

# RVQ configs to train: (NQ, CB_SIZE, label)
RVQ_CONFIGS = [
    (32, 32, "c32_f32"),
]


# ── Cấu hình Siêu tham số nâng cao từ ColSmol ──────────────────────────────────
RVQ_TRAINING_EPOCHS     = 40       # Giữ nguyên 40 epoch (hoặc tăng lên 60-120 nếu muốn hội tụ sâu hơn)
RVQ_TRAINING_BATCH_SIZE = 8192     # Batch size huấn luyện (đã tối ưu cho RTX 6000 96GB)
RVQ_DROPOUT_P           = 0.5      # Xác suất Dropout Quantizer
RVQ_DROPOUT_WARMUP_EPS  = 10       # Số epoch khởi động trước khi áp dụng dropout

N_TRAIN_PATCHES         = 1_700_000  # tổng patches cần encode
PATCHES_PER_SHARD       = 20_000    # giới hạn tối đa mỗi shard tránh mất cân bằng domain
ENCODE_BATCH_SIZE       = 16       # Batch size khi encode ảnh qua ColPali

# Cấu hình cho Spherical Prototype Network (SPN)
SPN_EPOCHS         = 30
SPN_BATCH_SIZE     = 2048
SPN_LR             = 3e-3
SPN_HIDDEN_DIM     = 512
SPN_TEMP_START     = 1.0
SPN_TEMP_END       = 0.1
SPN_ENTROPY_WEIGHT = 0.5

# Cấu hình cho Contrastive Triplet Ranking Loss
RANK_LOSS_WEIGHT   = 1.0
RANK_MARGIN        = 0.2
RANK_NEG_SAMPLES   = 8

# Cấu hình cho Optimal Transport (OT) Loss
OT_LOSS_WEIGHT     = 0.3
OT_SINKHORN_ITERS  = 20
OT_SINKHORN_EPS    = 0.05

# Blackwell / RTX PRO 6000: enable TF32
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

ENCODE_DTYPE = torch.bfloat16

print(">>>> METHOD 7: Advanced RVQ Codebook Training (SPN + Ranking Losses)")
print(f"  Train data     : {TRAIN_DATA_ROOT}")
print(f"  Output         : {OUTPUT_DIR}")
print(f"  Configs        : {[c[2] for c in RVQ_CONFIGS]}")
print(f"  Target patches : {N_TRAIN_PATCHES:,}")
print()

# ==============================================================================
# DATA STREAMING - lazy Parquet shard loader
# ==============================================================================

def _list_train_shards(root):
    pattern = os.path.join(root, "train-*.parquet")
    shards  = sorted(glob.glob(pattern))
    if not shards:
        raise FileNotFoundError(f"No train shards found at {pattern}")
    return shards

def _load_shard_images(shard_path):
    table  = pq.read_table(shard_path, columns=["image"])
    images = []
    for row in table.to_pydict()["image"]:
        if isinstance(row, dict):
            img_bytes = row.get("bytes") or row.get("path")
            if isinstance(img_bytes, str):
                img = Image.open(img_bytes).convert("RGB")
            else:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        elif isinstance(row, (bytes, bytearray)):
            img = Image.open(io.BytesIO(row)).convert("RGB")
        else:
            img = row.convert("RGB")
        images.append(img)
    return images

@torch.no_grad()
def _encode_images_rvq(images, batch_size=ENCODE_BATCH_SIZE):
    all_embs = []
    for start in range(0, len(images), batch_size):
        batch_imgs = images[start : start + batch_size]
        inputs     = query_processor.process_images(batch_imgs)
        inputs     = {k: v.to(device) for k, v in inputs.items()}
        with torch.autocast(device_type="cuda", dtype=ENCODE_DTYPE):
            embs = query_model(**inputs)
        embs = embs.float().cpu().numpy()
        del inputs
        torch.cuda.empty_cache()
        gc.collect()
        for i in range(embs.shape[0]):
            all_embs.append(embs[i])
        del embs
    return all_embs

def top_norm_filter(patches: np.ndarray, keep_ratio: float = 0.5) -> np.ndarray:
    """Keep only the top-`keep_ratio` patches by L2 norm.
    Low-norm patches are typically blank/whitespace regions and hurt codebook quality.
    Operates in raw (un-normalised) Euclidean space — do NOT call after normalisation.
    """
    norms = np.linalg.norm(patches, axis=1)          # (N,)
    k     = max(1, int(len(norms) * keep_ratio))
    idx   = np.argpartition(norms, -k)[-k:]          # top-k indices (unordered)
    return patches[idx]


def collect_training_patches(shards, target=N_TRAIN_PATCHES,
                              patches_per_shard=PATCHES_PER_SHARD, seed=42):
    # Stream shards -> encode -> filter whitespace patches (top 50% by L2 norm)
    # -> random subsample -> globally shuffled patch array.
    # Per-shard cap avoids over-representation of any single document domain.
    # NOTE: patches are kept in raw Euclidean space (NOT L2-normalised) so that
    #       ResidualVQ can learn geometry-correct Euclidean codebooks.
    # Returns: np.ndarray (N, EMB_DIM) float32
    rng          = np.random.default_rng(seed)
    patch_buf    = []
    total_so_far = 0

    print(f"Collecting patches: {len(shards)} shards, "
          f"cap {patches_per_shard:,}/shard, target {target:,} ...\n")

    pbar = tqdm(shards, desc="Encoding shards", unit="shard")
    for shard_path in pbar:
        if total_so_far >= target:
            break
        try:
            images = _load_shard_images(shard_path)
        except Exception as e:
            print(f"  [WARN] Skip {os.path.basename(shard_path)}: {e}")
            continue

        embs = _encode_images_rvq(images)
        del images; gc.collect()

        shard_patches = np.concatenate(embs, axis=0).astype(np.float32)
        del embs

        # ── Step 1: filter out low-norm (whitespace/blank) patches ──────────
        shard_patches = top_norm_filter(shard_patches, keep_ratio=0.5)

        # ── Step 2: subsample to per-shard cap ──────────────────────────────
        n_shard = shard_patches.shape[0]
        n_take  = min(patches_per_shard, n_shard, target - total_so_far)
        if n_take < n_shard:
            idx           = rng.choice(n_shard, n_take, replace=False)
            shard_patches = shard_patches[idx]

        # ── NOTE: No L2-normalisation here. Euclidean space is preserved. ───

        patch_buf.append(shard_patches)
        total_so_far += shard_patches.shape[0]
        pbar.set_postfix({"patches": f"{total_so_far:,}"})

    all_patches = np.concatenate(patch_buf, axis=0)
    shuffle_idx = rng.permutation(all_patches.shape[0])
    print(f"\n  Total patches: {all_patches.shape[0]:,}  ({all_patches.nbytes/1e6:.1f} MB)")
    return all_patches[shuffle_idx]


# ==============================================================================
# LOSS FUNCTIONS (Huấn luyện Gradient Nâng cao)
# ==============================================================================

def ranking_contrastive_loss(x_orig: torch.Tensor, z_q: torch.Tensor, 
                             margin: float = RANK_MARGIN, 
                             n_neg: int = RANK_NEG_SAMPLES) -> torch.Tensor:
    """Hàm loss phân cấp thứ tự Triplet (MaxSim-aware)"""
    N = x_orig.shape[0]
    pos_sim = (x_orig * z_q).sum(dim=-1)

    neg_idx = torch.zeros(N, n_neg, dtype=torch.long, device=x_orig.device)
    for i in range(N):
        pool = torch.arange(N, device=x_orig.device)
        pool = pool[pool != i]
        perm = torch.randperm(len(pool), device=x_orig.device)[:n_neg]
        neg_idx[i] = pool[perm]

    neg_z = z_q[neg_idx.view(-1)].view(N, n_neg, -1)
    neg_sim = (x_orig.unsqueeze(1) * neg_z).sum(dim=-1)
    hardest_neg_sim = neg_sim.max(dim=1).values
    loss = F.relu(margin - pos_sim + hardest_neg_sim).mean()
    return loss


def sinkhorn_log(log_alpha: torch.Tensor, n_iters: int = OT_SINKHORN_ITERS, 
                 eps: float = OT_SINKHORN_EPS) -> torch.Tensor:
    """Giải thuật Sinkhorn ổn định số học trong miền log"""
    N, M = log_alpha.shape
    log_a = torch.full((N,), -np.log(N), device=log_alpha.device, dtype=log_alpha.dtype)
    log_b = torch.full((M,), -np.log(M), device=log_alpha.device, dtype=log_alpha.dtype)

    log_u = torch.zeros(N, device=log_alpha.device, dtype=log_alpha.dtype)
    log_v = torch.zeros(M, device=log_alpha.device, dtype=log_alpha.dtype)

    for _ in range(n_iters):
        log_u = log_a - torch.logsumexp(log_alpha + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_alpha + log_u.unsqueeze(1), dim=0)

    log_T = log_alpha + log_u.unsqueeze(1) + log_v.unsqueeze(0)
    return log_T


def wasserstein_ot_loss(x_orig: torch.Tensor, z_q: torch.Tensor, 
                       eps: float = OT_SINKHORN_EPS, 
                       n_iters: int = OT_SINKHORN_ITERS) -> torch.Tensor:
    """Optimal Transport loss đo độ lệch hình học phân phối"""
    N = x_orig.shape[0]
    ot_n = min(N, 512)
    idx  = torch.randperm(N, device=x_orig.device)[:ot_n]
    x    = x_orig[idx]
    z    = z_q[idx]

    cos_sim = torch.mm(x, z.t())
    C = 1.0 - cos_sim
    log_alpha = -C / eps

    with torch.no_grad():
        log_T = sinkhorn_log(log_alpha.detach(), n_iters=n_iters, eps=eps)

    T = log_T.exp()
    loss = (C * T).sum()
    return loss


def rvq_forward_with_dropout_and_ranking(
    model:         ResidualVQ,
    x:             torch.Tensor,
    n_drop_suffix: int,
    use_rank_loss: bool = True,
    use_ot_loss:   bool = True,
) -> tuple:
    """Hàm forward tích hợp lan truyền gradient và các hàm loss nâng cao"""
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

    l_recon  = F.mse_loss(z_q_cumsum.float(), x.float().detach())
    l_align  = (1.0 - F.cosine_similarity(z_q_cumsum.float(), x.float(), dim=-1)).mean()
    l_commit = (torch.stack(all_commits).mean() if all_commits else torch.tensor(0., device=x.device))

    z_q_norm = F.normalize(z_q_cumsum.float(), dim=-1)
    x_norm   = F.normalize(x.float(), dim=-1)

    if use_rank_loss and RANK_LOSS_WEIGHT > 0:
        l_rank = ranking_contrastive_loss(x_norm, z_q_norm, margin=RANK_MARGIN, n_neg=RANK_NEG_SAMPLES)
    else:
        l_rank = torch.tensor(0., device=x.device)

    if use_ot_loss and OT_LOSS_WEIGHT > 0:
        l_ot = wasserstein_ot_loss(x_norm.detach(), z_q_norm, eps=OT_SINKHORN_EPS, n_iters=OT_SINKHORN_ITERS)
    else:
        l_ot = torch.tensor(0., device=x.device)

    total = (l_recon + 0.5 * l_align + 0.25 * l_commit + RANK_LOSS_WEIGHT * l_rank + OT_LOSS_WEIGHT * l_ot)

    losses = {
        'recon':  l_recon.item(),
        'align':  l_align.item(),
        'commit': l_commit.item(),
        'rank':   l_rank.item(),
        'ot':     l_ot.item(),
    }
    return total, losses, n_active


# ==============================================================================
# SPN (Spherical Prototype Network) INITIALIZER
# ==============================================================================

class SphericalPrototypeNetwork(nn.Module):
    """Mạng MLP học phân bổ trọng tâm cụm tối ưu trên mặt cầu đơn vị"""
    def __init__(self, emb_dim: int, n_proto: int, hidden_dim: int = 512):
        super().__init__()
        self.n_proto  = n_proto
        self.emb_dim  = emb_dim

        self.encoder = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, emb_dim),
        )
        self.skip_weight = nn.Parameter(torch.tensor(0.1))

        proto_init = torch.randn(n_proto, emb_dim)
        proto_init = F.normalize(proto_init, dim=-1)
        self.prototypes = nn.Parameter(proto_init)

    @property
    def spherical_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototypes, dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return F.normalize(x + self.skip_weight.abs() * h, dim=-1)

    def forward(self, x: torch.Tensor, temperature: float = 1.0):
        z = self.encode(x)
        P = self.spherical_prototypes

        sim = torch.mm(z, P.t()) / temperature
        q   = F.softmax(sim, dim=-1)
        q_bar = q.mean(dim=0)

        eps = 1e-8
        H   = -(q_bar * (q_bar + eps).log()).sum()
        l_entropy = -H

        idx    = q.argmax(dim=-1)
        p_hard = P[idx]
        l_recon = F.mse_loss(p_hard, z.detach())

        sim_pp_grad = torch.mm(P, P.t())
        sim_pp_grad.fill_diagonal_(-1.0)
        l_spread = F.relu(sim_pp_grad).mean()

        loss = l_entropy + SPN_ENTROPY_WEIGHT * l_recon + 0.1 * l_spread

        return loss, q, idx, {
            'l_entropy':    l_entropy.item(),
            'l_recon':      l_recon.item(),
            'l_spread':     l_spread.item(),
            'entropy':      H.item(),
            'max_entropy':  float(np.log(self.n_proto)),
        }

    @torch.no_grad()
    def seed_from_data(self, data: torch.Tensor):
        n_samples = min(self.n_proto, len(data))
        perm = torch.randperm(len(data))[:n_samples]
        self.prototypes.data[:n_samples].copy_(F.normalize(data[perm], dim=-1))


def train_spn_for_stage(residual_data: torch.Tensor, n_proto: int, 
                        emb_dim: int, dev: str, stage_idx: int) -> torch.Tensor:
    """Huấn luyện SPN học trọng tâm cụm cho từng tầng Residual (phân cấp)"""
    import torch.amp as amp
    from torch.utils.data import DataLoader, TensorDataset

    spn = SphericalPrototypeNetwork(emb_dim=emb_dim, n_proto=n_proto, hidden_dim=SPN_HIDDEN_DIM).to(dev)
    spn.seed_from_data(residual_data.to(dev))

    loader = DataLoader(
        TensorDataset(residual_data),
        batch_size=SPN_BATCH_SIZE, shuffle=True, drop_last=True,
        pin_memory=(dev == 'cuda'),
    )

    opt   = torch.optim.AdamW(spn.parameters(), lr=SPN_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SPN_EPOCHS, eta_min=SPN_LR / 10)
    scaler = amp.GradScaler(device=dev, enabled=(dev == 'cuda'))

    best_entropy = -float('inf')
    best_proto   = None

    for epoch in range(SPN_EPOCHS):
        spn.train()
        frac = epoch / max(SPN_EPOCHS - 1, 1)
        tau  = SPN_TEMP_START + frac * (SPN_TEMP_END - SPN_TEMP_START)
        ep_loss = ep_H = nb = 0

        for (batch,) in loader:
            batch = batch.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with amp.autocast(device_type=dev, enabled=(dev == 'cuda')):
                loss, _, _, info = spn(batch, temperature=tau)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(spn.parameters(), 1.0)
            with torch.no_grad():
                spn.prototypes.data = F.normalize(spn.prototypes.data, dim=-1)
            scaler.step(opt); scaler.update()
            ep_loss += loss.item()
            ep_H    += info['entropy']
            nb      += 1

        sched.step()
        mean_H = ep_H / max(nb, 1)
        if mean_H > best_entropy:
            best_entropy = mean_H
            best_proto   = spn.spherical_prototypes.detach().cpu().clone()

    with torch.no_grad():
        spn.eval()
        val_batch = residual_data[:min(4096, len(residual_data))].to(dev)
        _, q_final, idx_final, _ = spn(val_batch, temperature=SPN_TEMP_END)
        used = idx_final.unique().numel()
        dead = n_proto - used
        h_pct = best_entropy / info['max_entropy'] * 100
    print(f"    [SPN stage={stage_idx}] done | best_H={best_entropy:.3f} ({h_pct:.0f}%) | dead_codes={dead}/{n_proto}")

    del spn, loader; gc.collect(); torch.cuda.empty_cache()
    return best_proto


def initialize_rvq_codebooks_with_spn(
    rvq_model:  ResidualVQ,
    train_data: torch.Tensor,
    NQ:         int,
    CB_SIZE:    int,  # Truyền CB_SIZE động thay thế biến toàn cục
    emb_dim:    int,
    dev:        str,
) -> ResidualVQ:
    """Khởi tạo trọng số codebook cho từng quantizer stage của RVQ bằng SPN"""
    print(f"\n  Per-stage SPN initialization for NQ={NQ} stages...")
    residual = train_data.clone()

    spn_n = min(len(residual), SPN_BATCH_SIZE * SPN_EPOCHS)
    perm  = torch.randperm(len(residual))[:spn_n]
    spn_residual = residual[perm]

    for k in range(NQ):
        print(f"  Stage {k}/{NQ-1}: SPN on residual_k distribution")
        residual_norm = F.normalize(spn_residual, dim=-1)
        proto_k = train_spn_for_stage(
            residual_data=residual_norm,
            n_proto=CB_SIZE,  # Áp dụng CB_SIZE động
            emb_dim=emb_dim, dev=dev, stage_idx=k,
        )

        vq_layer    = rvq_model.layers[k]
        proto_k_gpu = proto_k.to(dev)
        injected    = False

        for attr_path in ['_codebook.embed', '_codebook.embed_avg', 'codebook']:
            try:
                obj = vq_layer
                parts = attr_path.split('.')
                for part in parts[:-1]: obj = getattr(obj, part)
                param = getattr(obj, parts[-1])
                target = proto_k_gpu if param.data.shape == proto_k_gpu.shape \
                    else proto_k_gpu.t() if param.data.shape == proto_k_gpu.t().shape \
                    else None
                if target is not None:
                    if isinstance(param, nn.Parameter): param.data.copy_(target)
                    else: param.copy_(target)
                    injected = True
                    print(f"    Injected via {attr_path}")
                    break
            except (AttributeError, RuntimeError):
                continue

        if not injected:
            for n, p in vq_layer.named_parameters():
                if CB_SIZE in p.shape and p.data.dim() >= 2:
                    flat = p.data
                    while flat.dim() > 2: flat = flat[0]
                    if flat.shape[0] == CB_SIZE:
                        flat.copy_(proto_k_gpu); injected = True
                    elif flat.dim() == 2 and flat.shape[1] == CB_SIZE:
                        flat.copy_(proto_k_gpu.t()); injected = True
                    if injected:
                        print(f"    Injected via named_param {n} (fallback)")
                        break

        if not injected:
            print(f"    [WARN] Could not inject stage {k} — using random init")

        # Tính toán residual_k+1 cho stage tiếp theo
        with torch.no_grad():
            proto_gpu_norm = F.normalize(proto_k_gpu, dim=-1)
            chunk_size = 8192
            z_q_k_list = []
            for start in range(0, len(spn_residual), chunk_size):
                chunk = F.normalize(spn_residual[start:start+chunk_size].to(dev), dim=-1)
                sims  = torch.mm(chunk, proto_gpu_norm.t())
                idx   = sims.argmax(dim=-1)
                z_q_k_list.append(proto_gpu_norm[idx].cpu())
            z_q_k = torch.cat(z_q_k_list, dim=0)
            spn_residual = spn_residual - z_q_k

        del proto_k_gpu, z_q_k; gc.collect(); torch.cuda.empty_cache()

    print(f"  SPN initialization complete for all {NQ} stages\n")
    return rvq_model


# ==============================================================================
# HÀM HUẤN LUYỆN RVQ CHÍNH (SPN + Gradient Fine-tuning)
# ==============================================================================

def train_rvq_model_advanced(NQ: int, CB_SIZE: int, train_data: torch.Tensor, dev: str, emb_dim: int) -> ResidualVQ:
    """Hàm huấn luyện RVQ nâng cao thay thế cho train_rvq_model cũ"""
    # Khởi tạo mô hình RVQ với gradient descent (ema_update=False)
    model = ResidualVQ(
        dim                     = emb_dim,
        num_quantizers          = NQ,
        codebook_size           = CB_SIZE,
        kmeans_init             = False,  # Thay KMeans bằng SPN
        kmeans_iters            = 0,
        threshold_ema_dead_code = 2,
        commitment_weight       = 0.5,
        learnable_codebook      = True,
        ema_update              = False,  # Sử dụng Gradient-based AdamW
    ).to(dev)

    # ── Phase 1: SPN Initialization ──────────────────────────────────────────
    t0_init = time.perf_counter()
    model = initialize_rvq_codebooks_with_spn(model, train_data, NQ, CB_SIZE, emb_dim, dev)
    print(f"  [SPN Init Done] elapsed = {time.perf_counter()-t0_init:.1f}s")

    # ── Phase 2: Gradient training with Dropout & Multi-task Losses ───────────
    print(f"  Starting Gradient Fine-Tuning | epoch: {RVQ_TRAINING_EPOCHS}")
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
        pin_memory = (dev == 'cuda')
    )

    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=RVQ_TRAINING_EPOCHS, eta_min=1e-5)
    scaler = amp.GradScaler(device=dev, enabled=(dev == 'cuda'))

    best_val   = float('inf')
    no_improve = 0
    best_state = None
    rng_drop   = random.Random(42)

    for epoch in range(RVQ_TRAINING_EPOCHS):
        model.train()

        # Áp dụng Quantizer Dropout tăng dần sau giai đoạn Warmup
        if epoch < RVQ_DROPOUT_WARMUP_EPS:
            drop_p = 0.0
        else:
            ramp   = min(1.0, (epoch - RVQ_DROPOUT_WARMUP_EPS) / 20.0)
            drop_p = RVQ_DROPOUT_P * ramp

        ep_losses = {k: 0. for k in ['recon', 'align', 'commit', 'rank', 'ot']}
        nb = 0

        for (batch,) in loader:
            batch = batch.to(dev, non_blocking=True)

            n_drop = 0
            if drop_p > 0 and rng_drop.random() < drop_p:
                n_drop = min(NQ - 1, int(np.random.geometric(p=1 - drop_p * 0.5)) - 1)

            opt.zero_grad(set_to_none=True)
            with amp.autocast(device_type=dev, enabled=(dev == 'cuda')):
                loss, comp_losses, _ = rvq_forward_with_dropout_and_ranking(
                    model, batch, n_drop, use_rank_loss=True, use_ot_loss=True
                )

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            for k, v in comp_losses.items():
                ep_losses[k] += v
            nb += 1

        sched.step()

        # Validation set check
        model.eval()
        with torch.no_grad():
            v_batch = v_data[:min(8192, len(v_data))].to(dev)
            val_loss, _, _ = rvq_forward_with_dropout_and_ranking(
                model, v_batch, n_drop_suffix=0, use_rank_loss=True, use_ot_loss=True
            )
            val_val = val_loss.item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_losses = {k: v / nb for k, v in ep_losses.items()}
            print(f"    Epoch {epoch+1:>3}/{RVQ_TRAINING_EPOCHS} | val_loss={val_val:.5f} | "
                  f"recon={avg_losses['recon']:.5f}, rank={avg_losses['rank']:.4f}, ot={avg_losses['ot']:.4f}")

        # Early stopping check (Patience=10)
        if val_val < best_val - 1e-6:
            best_val = val_val
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= 10:
            print(f"    Early stop at epoch {epoch+1} (no validation improvement for 10 epochs)")
            break

    # Load lại checkpoint tốt nhất
    if best_state is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return model


def extract_rvq_codebooks(rvq_model, NQ, CB_SIZE):
    # Extract codebook tensors from a trained ResidualVQ.
    # Codebooks are kept in raw Euclidean space (NO L2-normalisation) so that
    # inference-time distance computations (Euclidean / ADC) remain correct.
    # Returns np.ndarray (NQ, CB_SIZE, EMB_DIM) float32.
    cbs = []
    for qi in range(NQ):
        vq = rvq_model.layers[qi]
        cb = None
        for attr_path in ['_codebook.embed', 'codebook', '_codebook.embed_avg']:
            try:
                obj = vq
                for part in attr_path.split('.'):
                    obj = getattr(obj, part)
                if isinstance(obj, torch.Tensor):
                    cb = obj[0] if obj.ndim == 3 else obj
                    cb = cb.detach().float().cpu()   # NO F.normalize — preserve Euclidean geometry
                    break
            except (AttributeError, IndexError):
                continue
        if cb is None:
            raise RuntimeError(f"Cannot extract codebook for quantizer {qi}.")
        cbs.append(cb.numpy())
    return np.stack(cbs, axis=0)


def save_rvq_codebook(cbs_np, NQ, CB_SIZE, label):
    npy_path  = os.path.join(OUTPUT_DIR, f"{label}_codebook.npy")
    meta_path = os.path.join(OUTPUT_DIR, f"{label}_meta.json")

    np.save(npy_path, cbs_np)

    meta = {
        "label"               : label,
        "NQ"                  : NQ,
        "CB_SIZE"             : CB_SIZE,
        "EMB_DIM"             : EMB_DIM,
        "codebook_shape"      : list(cbs_np.shape),
        "bytes_per_patch"     : NQ,
        "compression_vs_f32"  : round(EMB_DIM * 4 / NQ, 1),
        "trained_on"          : f"{N_TRAIN_PATCHES:,} patches - ColPali train set (SPN)",
        "model_base"          : COLPALI_BASE,
        "model_lora"          : COLPALI_LORA,
        "timestamp"           : time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    size_mb = os.path.getsize(npy_path) / 1e6
    print(f"  OK: {npy_path}  ({size_mb:.3f} MB)")
    print(f"     meta : {meta_path}")


# ==============================================================================
# MAIN - collect patches -> free encoder -> train each config -> save
# ==============================================================================

_t_total = time.time()

# Bắt đầu chạy luồng lấy dữ liệu và chạy train
_shards = _list_train_shards(TRAIN_DATA_ROOT)
print(f"Found {len(_shards)} train shards\n")

_patches_np = collect_training_patches(_shards)
_train_data = torch.from_numpy(_patches_np)
del _patches_np; gc.collect()

# Giải phóng VRAM của encoder sang CPU để chuẩn bị train RVQ
with torch.no_grad():
    pass
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
    print(f"VRAM free: {free_gb:.1f} GB — proceeding to RVQ training.\n")
gc.collect()

_rvq_results = {}
for _NQ, _CB_SIZE, _label in RVQ_CONFIGS:
    print(f"\n{'='*60}")
    print(f"CONFIG: {_label}  (NQ={_NQ}, CB_SIZE={_CB_SIZE})")
    print(f"{'='*60}")

    _t_cfg   = time.time()
    # Gọi hàm huấn luyện nâng cao mới định nghĩa
    _rvq     = train_rvq_model_advanced(_NQ, _CB_SIZE, _train_data, device, EMB_DIM)
    _elapsed = time.time() - _t_cfg
    print(f"  Training time: {_elapsed:.1f}s")

    _cbs_np = extract_rvq_codebooks(_rvq, _NQ, _CB_SIZE)
    del _rvq; gc.collect()

    save_rvq_codebook(_cbs_np, _NQ, _CB_SIZE, _label)
    del _cbs_np; gc.collect()

    _rvq_results[_label] = {"NQ": _NQ, "CB_SIZE": _CB_SIZE, "train_secs": round(_elapsed, 1)}

del _train_data; gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# In bảng tổng hợp
print(f"\n{'='*60}")
print(f"ALL DONE in {(time.time()-_t_total)/60:.1f} min")
print(f"{'='*60}")
print(f"{'Config':<15} {'NQ':>4} {'CB':>5} {'bytes/patch':>12} {'ratio':>8} {'secs':>8}")
print("-" * 55)
for _lbl, _r in _rvq_results.items():
    _ratio = round(EMB_DIM * 4 / _r["NQ"], 1)
    print(f"{_lbl:<15} {_r['NQ']:>4} {_r['CB_SIZE']:>5} {_r['NQ']:>12}  {_ratio:>6.1f}x  {_r['train_secs']:>7.1f}")
