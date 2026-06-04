# ==============================================================================
# METHOD 7: RVQ Retrieval — Pre-trained Codebook (NO re-training)
#
# Pre-trained codebook (upload lên kaggle dataset):
#   /kaggle/input/datasets/namthi/weight-train-rvq-colsmol-vidore/c32_f32_codebook_spn.npy
#
# Codebook format: np.ndarray (NQ, CB_SIZE, EMB_DIM) float32, L2-normalized
#   c32_f32_spn -> (32, 32, 128)   32 bytes/patch  (16× compression vs float32)
#
# ── PART A: Pure RVQ — Single-stage greedy ADC full scan ────────────────────
#   Full-scan ADC → top-10 directly (1 codebook, 1 pass)
#
# ── PART B: RVQ + Beam Search (b=5) ─────────────────────────────────────────
#   Full-scan greedy ADC → top-K candidates  (fast)
#   Re-rank candidates với BEAM SEARCH b=5 trên query tokens
#
#   Beam search: thay vì dùng raw query token vectors, sinh ra b=5 reconstructed
#   query vectors per token (qua RVQ codebook tree), rồi lấy MAX similarity.
#   Bù lỗi lượng hóa trong index bằng cách khám phá nhiều "query probes".
#
# Requires (from earlier cells):
#   all_page_embeddings  : list of np.ndarray (L, D)
#   qa_pairs, query_model, query_processor, device
#   hit_metrics, record, print_summary, _init_metric, WORKING_DIR
# ==============================================================================

import os, gc, time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
import pandas as pd

# ── GPU settings ───────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CODEBOOK_ROOT    = "/kaggle/input/datasets/namthi/weight-train-rvq-colsmol-vidore"
QUANT_PAGE_BATCH = 128    # pages per GPU batch during corpus quantisation
ADC_CHUNK_SIZE   = 4096   # documents per chunk during full-scan ADC
BEAM_WIDTH       = 5      # beam search width (Part B)

# Pre-trained codebook metadata (single stage: NQ=32, CB_SIZE=32)
CB_META = {
    "c32_f32_spn": {"NQ": 32, "CB_SIZE": 32, "file": "c32_f32_codebook_spn.npy"},
}

# Sweep TOP_K candidates for Part B beam search re-ranking
# Part A does a full scan → top-10 directly (no TOP_K needed)
TOP_K_LIST = [100]

print(">>>> METHOD 7: ColSmol RVQ Retrieval (Pre-trained Codebook, 1-Stage)")
print(f"  Codebook root : {CODEBOOK_ROOT}")
print(f"  Config        : c32_f32_spn  (NQ=32, CB_SIZE=32, 32 bytes/patch)")
print(f"  Beam width    : {BEAM_WIDTH}  (Part B only)")
print(f"  TOP_K sweep   : {TOP_K_LIST}  (Part B only)")
print()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load pre-trained codebook from .npy file
# ══════════════════════════════════════════════════════════════════════════════
def load_codebook(label):
    """
    Load .npy → list of NQ GPU tensors, each shape (CB_SIZE, EMB_DIM).
    File on disk: (NQ, CB_SIZE, EMB_DIM) float32.
    """
    cfg    = CB_META[label]
    path   = os.path.join(CODEBOOK_ROOT, cfg["file"])
    cbs_np = np.load(path)                              # (NQ, CB_SIZE, D)
    assert cbs_np.ndim == 3, f"Expected 3D array, got {cbs_np.shape}"
    assert cbs_np.shape[0] == cfg["NQ"],      f"NQ mismatch: {cbs_np.shape[0]} vs {cfg['NQ']}"
    assert cbs_np.shape[1] == cfg["CB_SIZE"], f"CB_SIZE mismatch: {cbs_np.shape[1]} vs {cfg['CB_SIZE']}"
    # Ensure L2-normalised (should already be, but safety re-norm)
    norms  = np.linalg.norm(cbs_np, axis=-1, keepdims=True).clip(min=1e-8)
    cbs_np = cbs_np / norms
    cbs    = [torch.from_numpy(cbs_np[qi].copy()).float().to(device)
              for qi in range(cbs_np.shape[0])]
    print(f"  {label}: {path}")
    print(f"    shape={cbs_np.shape}  |  {cbs_np.nbytes/1e6:.2f} MB")
    return cbs

print("Loading codebook...")
cb_cache = {}
for lbl in CB_META:
    cb_cache[lbl] = load_codebook(lbl)
print("✅ Codebook loaded.\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Sanity-check all_page_embeddings + compute index stats
# ══════════════════════════════════════════════════════════════════════════════
if 'all_page_embeddings' not in globals() or all_page_embeddings is None:
    raise RuntimeError(
        "all_page_embeddings not found — run the index-load cell first."
    )

n_pages_total       = len(all_page_embeddings)
EMB_DIM             = int(all_page_embeddings[0].shape[-1])
doc_lengths         = [e.shape[0] for e in all_page_embeddings]
max_doc_len         = int(max(doc_lengths))
total_patches       = int(sum(doc_lengths))
bytes_per_patch_f32 = EMB_DIM * 4

print(f"Document index:")
print(f"  Pages        : {n_pages_total:,}")
print(f"  Total patches: {total_patches:,}")
print(f"  EMB_DIM      : {EMB_DIM}   max_doc_len: {max_doc_len}")
print(f"  float32 size : {total_patches * bytes_per_patch_f32 / 1e6:.1f} MB")
mem_rvq = total_patches * CB_META["c32_f32_spn"]["NQ"] / 1e6
print(f"  uint8  size  : {mem_rvq:.1f} MB  (c32_f32_spn, {CB_META['c32_f32_spn']['NQ']} bytes/patch)\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Quantise corpus with codebook (greedy nearest-neighbour RVQ)
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def quantize_corpus(codebooks, label):
    """
    Encode every document patch using greedy RVQ:
      residual[0] = patch_vector
      for each level qi:
          best_idx   = argmax cosine_sim(residual, codebook[qi])
          residual  -= codebook[qi][best_idx]
    Returns:
      idx_arr  : (n_docs, max_doc_len, NQ)  uint8  — codebook indices
      mask_arr : (n_docs, max_doc_len)       bool   — valid-patch mask
    """
    NQ       = len(codebooks)
    n        = n_pages_total
    idx_arr  = np.zeros((n, max_doc_len, NQ), dtype=np.uint8)
    mask_arr = np.zeros((n, max_doc_len),      dtype=bool)

    for bs in tqdm(range(0, n, QUANT_PAGE_BATCH),
                   desc=f"  Quantise {label}", leave=False):
        be    = min(bs + QUANT_PAGE_BATCH, n)
        pages = all_page_embeddings[bs:be]
        B     = len(pages)

        bt = torch.zeros(B, max_doc_len, EMB_DIM, device=device)
        bm = torch.zeros(B, max_doc_len, dtype=torch.bool)
        for i, emb in enumerate(pages):
            L         = emb.shape[0]
            emb_t     = torch.from_numpy(emb.astype(np.float32)).to(device)
            bt[i, :L] = F.normalize(emb_t, dim=-1)
            bm[i, :L] = True

        flat     = bt.reshape(B * max_doc_len, EMB_DIM)   # (B*L, D)
        residual = flat.clone()
        idxs     = torch.zeros(B * max_doc_len, NQ, dtype=torch.long, device=device)

        for qi, cb in enumerate(codebooks):                # cb: (CB, D)
            sims       = torch.mm(residual, cb.t())        # (B*L, CB)
            best_idx   = sims.argmax(dim=-1)               # (B*L,)
            residual  -= cb[best_idx]
            idxs[:, qi] = best_idx

        idxs = idxs.reshape(B, max_doc_len, NQ)
        idx_arr[bs:be]  = idxs.cpu().numpy().astype(np.uint8)
        mask_arr[bs:be] = bm.numpy()

    return (torch.from_numpy(idx_arr).pin_memory(),
            torch.from_numpy(mask_arr).pin_memory())


print("Quantising corpus with c32_f32_spn codebook...")
rvq_idx = {}
for lbl in CB_META:
    t0 = time.time()
    rvq_idx[lbl] = quantize_corpus(cb_cache[lbl], lbl)
    mb = rvq_idx[lbl][0].element_size() * rvq_idx[lbl][0].nelement() / 1e6
    print(f"  {lbl}: {time.time()-t0:.1f}s | {mb:.1f} MB uint8  "
          f"(vs {total_patches * bytes_per_patch_f32 / 1e6:.0f} MB float32)")

# Free memory: only need the quantised indices from now on
if 'all_page_embeddings' in globals():
    del all_page_embeddings

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("✅ Corpus quantisation done (deleted all_page_embeddings to free RAM).\n")


# ══════════════════════════════════════════════════════════════════════════════
# ADC (Asymmetric Distance Computation) HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _build_lut(q_norm, codebooks):
    """LUT[qi, tok_i, cb_entry] = dot(query_token_i, codebook[qi][cb_entry])
    Shape: (NQ, Nq, CB_SIZE)"""
    return torch.stack([torch.mm(q_norm, cb.t()) for cb in codebooks])


@torch.no_grad()
def adc_full_scan(q_norm, lut, idx_cpu, mask_cpu):
    """
    Full-corpus ADC scan.
    Returns: (n_docs,) scores tensor on GPU.
    """
    n_docs = idx_cpu.shape[0]
    NQ_    = idx_cpu.shape[2]
    Nq     = q_norm.shape[0]
    scores = torch.zeros(n_docs, device=device)

    for s in range(0, n_docs, ADC_CHUNK_SIZE):
        e   = min(s + ADC_CHUNK_SIZE, n_docs)
        ci  = idx_cpu[s:e].to(device, non_blocking=True)   # (B, L, NQ)
        cm  = mask_cpu[s:e].to(device, non_blocking=True)  # (B, L)
        cs, ml = ci.shape[0], ci.shape[1]

        sim = torch.zeros(Nq, cs, ml, device=device)
        for q in range(NQ_):
            idx_e = ci[:, :, q].long().unsqueeze(0).expand(Nq, -1, -1)
            sim  += torch.gather(lut[q].unsqueeze(1).expand(-1, cs, -1), 2, idx_e)

        sim.masked_fill_(~cm.unsqueeze(0), float('-inf'))
        # MaxSim: max over patches, sum over query tokens
        scores[s:e] = sim.max(dim=-1).values.sum(dim=0)

    return scores


@torch.no_grad()
def adc_rerank(q_norm, lut, candidates, idx_cpu, mask_cpu):
    """
    ADC re-ranking on a candidate subset.
    Returns: top-10 global doc indices.
    """
    NQ_  = idx_cpu.shape[2]
    Nq   = q_norm.shape[0]
    K    = len(candidates)
    ct   = torch.tensor(candidates, dtype=torch.long)
    ci   = idx_cpu[ct].to(device)    # (K, L, NQ)
    cm   = mask_cpu[ct].to(device)   # (K, L)
    ml   = ci.shape[1]
    sim  = torch.zeros(Nq, K, ml, device=device)
    for q in range(NQ_):
        idx_e = ci[:, :, q].long().unsqueeze(0).expand(Nq, -1, -1)
        sim  += torch.gather(lut[q].unsqueeze(1).expand(-1, K, -1), 2, idx_e)
    sim.masked_fill_(~cm.unsqueeze(0), float('-inf'))
    fine = sim.max(dim=-1).values.sum(dim=0)   # (K,)
    top  = torch.topk(fine, min(10, K)).indices.cpu().tolist()
    return [candidates[i] for i in top]


# ══════════════════════════════════════════════════════════════════════════════
# BEAM SEARCH HELPERS (Part B)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _beam_reconstruct(q_norm, codebooks, beam_width):
    """
    True beam search through RVQ codebook tree for all query tokens.

    Algorithm (batched over all Nq tokens simultaneously):
      Level 0 : for each token, pick top-B codebook entries → B candidate paths
      Level 1+: for each of B paths, pick the BEST remaining entry among all CB
                entries → keep best B paths (by cumulative inner-product score)

    Returns: (Nq, B, D) float32 tensor — B L2-normalised reconstructed vectors
             per query token (on GPU).
    """
    Nq, D  = q_norm.shape
    B      = beam_width
    NQ     = len(codebooks)

    # ── Level 0: expand to B beams ──────────────────────────────────────────
    cb0     = codebooks[0]                                            # (CB, D)
    sims0   = torch.mm(q_norm, cb0.t())                               # (Nq, CB)
    tvals0, tidxs0 = torch.topk(sims0, min(B, cb0.shape[0]), dim=-1) # (Nq, B)

    residuals = q_norm.unsqueeze(1) - cb0[tidxs0]                    # (Nq, B, D)
    recons    = cb0[tidxs0].clone()                                   # (Nq, B, D)
    scores    = tvals0                                                # (Nq, B)

    arange_nq = torch.arange(Nq, device=device)

    # ── Levels 1 .. NQ-1: extend & prune ────────────────────────────────────
    for qi_level in range(1, NQ):
        cb  = codebooks[qi_level]                                     # (CB, D)
        CB  = cb.shape[0]

        res_flat = residuals.reshape(Nq * B, D)                       # (Nq*B, D)
        sims     = torch.mm(res_flat, cb.t()).reshape(Nq, B, CB)      # (Nq, B, CB)

        cand_scores = scores.unsqueeze(-1) + sims                     # (Nq, B, CB)
        cand_flat   = cand_scores.reshape(Nq, B * CB)                 # (Nq, B*CB)

        top_vals, top_flat_idxs = torch.topk(cand_flat, B, dim=-1)   # (Nq, B)
        beam_from = top_flat_idxs // CB
        cb_chosen = top_flat_idxs %  CB

        par_res   = residuals[arange_nq.unsqueeze(1), beam_from]      # (Nq, B, D)
        par_recon = recons[arange_nq.unsqueeze(1), beam_from]         # (Nq, B, D)
        cb_vecs   = cb[cb_chosen]                                     # (Nq, B, D)

        residuals = par_res - cb_vecs
        recons    = par_recon + cb_vecs
        scores    = top_vals

    return F.normalize(recons, dim=-1)   # (Nq, B, D)


@torch.no_grad()
def beam_adc_rerank(q_norm, candidates, idx_cpu, mask_cpu, codebooks,
                    beam_width=BEAM_WIDTH):
    """
    Beam-search ADC re-ranking on candidate subset.
    Returns: top-10 global doc indices.
    """
    Nq  = q_norm.shape[0]
    B   = beam_width
    NQ_ = idx_cpu.shape[2]
    K   = len(candidates)

    # Build B reconstructed query vectors per token: (Nq, B, D)
    beam_q      = _beam_reconstruct(q_norm, codebooks, B)   # (Nq, B, D)
    beam_q_flat = beam_q.reshape(Nq * B, -1)                # (Nq*B, D)

    lut = torch.stack([torch.mm(beam_q_flat, cb.t()) for cb in codebooks])  # (NQ, Nq*B, CB)

    ct  = torch.tensor(candidates, dtype=torch.long)
    ci  = idx_cpu[ct].to(device)    # (K, L, NQ)
    cm  = mask_cpu[ct].to(device)   # (K, L)
    ml  = ci.shape[1]

    CAND_CHUNK = 256
    fine = torch.zeros(K, device=device)
    for cs in range(0, K, CAND_CHUNK):
        ce   = min(cs + CAND_CHUNK, K)
        ci_c = ci[cs:ce]       # (Cc, L, NQ)
        cm_c = cm[cs:ce]       # (Cc, L)
        Cc   = ci_c.shape[0]

        sim_c = torch.zeros(Nq * B, Cc, ml, device=device)
        for q in range(NQ_):
            idx_e = ci_c[:, :, q].long().unsqueeze(0).expand(Nq * B, -1, -1)
            sim_c += torch.gather(
                lut[q].unsqueeze(1).expand(-1, Cc, -1), 2, idx_e
            )
        sim_c.masked_fill_(~cm_c.unsqueeze(0), float('-inf'))   # (Nq*B, Cc, ml)

        # Reshape → (Nq, B, Cc, ml), max over B → (Nq, Cc, ml)
        sim_c = sim_c.view(Nq, B, Cc, ml).max(dim=1).values

        # MaxSim: max over patches → (Nq, Cc), sum over tokens → (Cc,)
        fine[cs:ce] = sim_c.max(dim=-1).values.sum(dim=0)

    top = torch.topk(fine, min(10, K)).indices.cpu().tolist()
    return [candidates[i] for i in top]


# ══════════════════════════════════════════════════════════════════════════════
# PART A: PURE RVQ — Single-Stage Greedy ADC Full Scan
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART A: RVQ — Single-Stage Greedy ADC (c32_f32_spn)")
print("=" * 70)

cfg_label_a = "rvq_c32_f32_spn_greedy"
cb_32       = cb_cache["c32_f32_spn"]
idx_32, mask_32 = rvq_idx["c32_f32_spn"]
mem_32      = total_patches * CB_META["c32_f32_spn"]["NQ"] / 1e6

print(f"\n  {cfg_label_a}  ({mem_32:.0f} MB uint8 → direct top-10)")

rvq_met_a  = {}
rvq_dom_a  = {}
rvq_rows_a = []
_lat_a     = []

t_eval = time.time()
for qi, item in tqdm(enumerate(qa_pairs), total=len(qa_pairs),
                     desc=f"  {cfg_label_a}", leave=False):
    q_in = query_processor.process_queries([item['question']]).to(device)
    with torch.no_grad():
        q_proj = query_model(**q_in)
    tidx   = torch.where(q_in['attention_mask'][0] > 0)[0]
    q_norm = F.normalize(q_proj[0][tidx].float(), dim=-1)
    gt_set = item.get('gt_relevance', item['gt_embed_indices'])
    domain = item['domain']

    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Single-stage: full-scan ADC → top-10 directly
    lut    = _build_lut(q_norm, cb_32)
    scores = adc_full_scan(q_norm, lut, idx_32, mask_32)
    top10  = torch.topk(scores, min(10, n_pages_total)).indices.cpu().tolist()

    if torch.cuda.is_available(): torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    _lat_a.append(ms)

    m = hit_metrics(top10, gt_set)
    record(rvq_met_a, rvq_dom_a, cfg_label_a, m, domain)
    rvq_rows_a.append({
        'method': 'RVQ', 'part': 'A',
        'config': cfg_label_a, 'NQ': 32, 'CB_SIZE': 32,
        'beam_width': 1, 'top_K': 'full',
        'query_id': qi, 'doc_name': item['doc_name'],
        'domain': domain, 'question': item['question'],
        'r@1': m['r1'], 'r@5': m['r5'], 'r@10': m['r10'],
        'ndcg@10': round(m['n10'], 4), 'score_ms': round(ms, 3),
    })

avg_ms_a = np.mean(_lat_a)
print(f"  Done {time.time()-t_eval:.1f}s | avg {avg_ms_a:.1f} ms/query")
print_summary(rvq_met_a, rvq_dom_a, [cfg_label_a],
              title="Part A — RVQ Single-Stage Greedy ADC (c32_f32)")

pd.DataFrame(rvq_rows_a).to_csv(
    os.path.join(WORKING_DIR, "method7_partA_rvq.csv"), index=False)
print("✅ Saved: method7_partA_rvq.csv")


# ══════════════════════════════════════════════════════════════════════════════
# PART B: RVQ + BEAM SEARCH (b=5)
#   Full-scan greedy ADC → top-K candidates  (same index as Part A)
#   Re-rank candidates với beam-search width=5 trên query tokens
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"PART B: RVQ + Beam Search  (b={BEAM_WIDTH}, c32_f32_spn)")
print("=" * 70)

rvq_met_b  = {}
rvq_dom_b  = {}
rvq_rows_b = []
_lat_b     = {}

for TOP_K in TOP_K_LIST:
    cfg_lbl = f"rvq_beam{BEAM_WIDTH}_c32_f32_spn_k{TOP_K}"
    print(f"\n  {cfg_lbl}  ({mem_32:.0f}MB greedy → top-{TOP_K} → beam-{BEAM_WIDTH} re-rank)")

    t_eval = time.time()
    for qi, item in tqdm(enumerate(qa_pairs), total=len(qa_pairs),
                         desc=f"  {cfg_lbl}", leave=False):
        q_in = query_processor.process_queries([item['question']]).to(device)
        with torch.no_grad():
            q_proj = query_model(**q_in)
        tidx   = torch.where(q_in['attention_mask'][0] > 0)[0]
        q_norm = F.normalize(q_proj[0][tidx].float(), dim=-1)
        gt_set = item.get('gt_relevance', item['gt_embed_indices'])
        domain = item['domain']

        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Stage 1: greedy full-scan ADC → top-K candidates (fast)
        lut    = _build_lut(q_norm, cb_32)
        scores = adc_full_scan(q_norm, lut, idx_32, mask_32)
        cands  = torch.topk(scores, min(TOP_K, n_pages_total)).indices.cpu().tolist()

        # Stage 2: beam-search ADC re-ranking on candidates → top-10
        top10 = beam_adc_rerank(q_norm, cands, idx_32, mask_32, cb_32, BEAM_WIDTH)

        if torch.cuda.is_available(): torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        _lat_b.setdefault(cfg_lbl, []).append(ms)

        m = hit_metrics(top10, gt_set)
        record(rvq_met_b, rvq_dom_b, cfg_lbl, m, domain)
        rvq_rows_b.append({
            'method': 'RVQ+Beam', 'part': 'B',
            'config': cfg_lbl, 'NQ': 32, 'CB_SIZE': 32,
            'beam_width': BEAM_WIDTH, 'top_K': TOP_K,
            'query_id': qi, 'doc_name': item['doc_name'],
            'domain': domain, 'question': item['question'],
            'r@1': m['r1'], 'r@5': m['r5'], 'r@10': m['r10'],
            'ndcg@10': round(m['n10'], 4), 'score_ms': round(ms, 3),
        })

    avg_ms = np.mean(_lat_b.get(cfg_lbl, [0.0]))
    print(f"  Done {time.time()-t_eval:.1f}s | avg {avg_ms:.1f} ms/query")

print_summary(rvq_met_b, rvq_dom_b,
              [f"rvq_beam{BEAM_WIDTH}_c32_f32_spn_k{k}" for k in TOP_K_LIST],
              title=f"Part B — RVQ + Beam Search (b={BEAM_WIDTH}, c32_f32_spn)")

pd.DataFrame(rvq_rows_b).to_csv(
    os.path.join(WORKING_DIR, "method7_partB_rvq_beam.csv"), index=False)
print(f"✅ Saved: method7_partB_rvq_beam.csv")


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 105)
print(f"{'Config':<45} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'nDCG@10':>9} {'ms/q':>8}")
print("-" * 105)

def _print_row(cfg_lbl, met_dict, lat_dict):
    m   = met_dict.get(cfg_lbl, _init_metric())
    cnt = max(m['count'], 1)
    lat = np.mean(lat_dict.get(cfg_lbl, [0.0]))
    print(f"{cfg_lbl:<45} "
          f"{m['r1']/cnt*100:6.2f}%  "
          f"{m['r5']/cnt*100:6.2f}%  "
          f"{m['r10']/cnt*100:6.2f}%  "
          f"{m['n10']/cnt:8.4f}  "
          f"{lat:7.1f}ms")

_print_row(cfg_label_a, rvq_met_a, {cfg_label_a: _lat_a})
print()
for k in TOP_K_LIST:
    _print_row(f"rvq_beam{BEAM_WIDTH}_c32_f32_spn_k{k}", rvq_met_b, _lat_b)

# Save combined
all_rows = rvq_rows_a + rvq_rows_b
pd.DataFrame(all_rows).to_csv(
    os.path.join(WORKING_DIR, "method7_rvq_combined.csv"), index=False)
print("✅ Saved: method7_rvq_combined.csv")

# Cleanup
for lbl in list(rvq_idx.keys()): del rvq_idx[lbl]
del rvq_idx, cb_cache; gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
print("\n>>> Method 7 complete.")
