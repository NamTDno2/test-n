print(">>> CELL 2: RVQ Retrieval Evaluation on MMDocIR Test Set")
print(">>> Uses KMeans codebook from rvq_codebooks_kmeans.pt")
print(">>> Methods: (A) Pure ADC full-scan  (B) ADC + Beam-Search re-rank")

# ==============================================================================
# DEPENDENCIES — same as training cell
# ==============================================================================
import gc, os, io, time, pickle, random, glob, json, subprocess, sys, types
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm

# ── Install vector-quantize-pytorch (wheel dir) ──────────────────────────────
_rvq_wheel_dir = "/kaggle/input/datasets/thinam4/rvq-wheels/rvq_wheels"
if os.path.isdir(_rvq_wheel_dir):
    subprocess.run(
        ["pip", "install", "--quiet", "--no-index",
         "--find-links", _rvq_wheel_dir, "vector-quantize-pytorch"],
        check=True
    )

# ── Patch colpali_engine import conflicts ─────────────────────────────────────
def _patch_colpali_stub_all():
    problem_modules = [
        "colpali_engine.models.gemma3",
        "colpali_engine.models.gemma3.bigemma3",
        "colpali_engine.models.gemma3.colgemma3",
        "colpali_engine.models.modernvbert",
        "colpali_engine.models.modernvbert.bivbert",
        "colpali_engine.models.modernvbert.colvbert",
        "colpali_engine.models.paligemma",
        "colpali_engine.models.paligemma.bipali",
        "colpali_engine.models.paligemma.colpali",
        "colpali_engine.models.paligemma.bipali_proj",
        "colpali_engine.models.qwen2",
        "colpali_engine.models.qwen2.biqwen2",
        "colpali_engine.models.qwen2.colqwen2",
        "colpali_engine.models.qwen3",
        "colpali_engine.models.qwen3.biqwen3",
        "colpali_engine.models.qwen3.colqwen3",
        "colpali_engine.models.qwen3_5",
        "colpali_engine.models.qwen3_5.biqwen3_5",
        "colpali_engine.models.qwen3_5.colqwen3_5",
        "colpali_engine.models.qwen_omni",
        "colpali_engine.models.qwen_omni.colqwen2_5_omni",
    ]
    stub_class_map = {
        "colpali_engine.models.gemma3":       ["BiGemma3","BiGemmaProcessor3","ColGemma3","ColGemmaProcessor3"],
        "colpali_engine.models.modernvbert":  ["BiModernVBert","BiModernVBertProcessor","ColModernVBert","ColModernVBertProcessor"],
        "colpali_engine.models.paligemma":    ["BiPali","BiPaliProcessor","BiPaliProj","ColPali","ColPaliProcessor"],
        "colpali_engine.models.qwen2":        ["BiQwen2","BiQwen2Processor","ColQwen2","ColQwen2Processor"],
        "colpali_engine.models.qwen3":        ["BiQwen3","BiQwen3Processor","ColQwen3","ColQwen3Processor"],
        "colpali_engine.models.qwen3_5":      ["BiQwen3_5","BiQwen3_5Processor","ColQwen3_5","ColQwen3_5Processor"],
        "colpali_engine.models.qwen_omni":    ["ColQwen2_5Omni","ColQwen2_5OmniProcessor"],
    }
    for name in problem_modules:
        stub = types.ModuleType(name)
        sys.modules[name] = stub
    for mod_name, cls_list in stub_class_map.items():
        mod = sys.modules[mod_name]
        for cls_name in cls_list:
            setattr(mod, cls_name, type(cls_name, (), {}))
    stale = [k for k in sys.modules if k.startswith("colpali_engine") and k not in problem_modules]
    for k in stale:
        del sys.modules[k]
    print(f">>> Stubbed {len(problem_modules)} colpali_engine submodules")

_patch_colpali_stub_all()
from colpali_engine.models.idefics3 import ColIdefics3, ColIdefics3Processor

# ==============================================================================
# CONFIG
# ==============================================================================
BASE_MODEL         = "/kaggle/input/datasets/nguyenducdung1107/model500m"
LORA_PATH          = "/kaggle/input/datasets/nguyenducdung1107/colsmol500-adapter/colsmol500_adapter"
COLSMOL_DIR        = "/kaggle/input/datasets/nguyenducdung1107/colsmol500m-layoutmmdoc/colsmol500m-pkl"
ANNOTATIONS_PATH   = "/kaggle/input/datasets/namthi/mmdocir-eval-data/MMDocIR_annotations.jsonl"
PARQUET_PATH       = "/kaggle/input/datasets/namthi/mmdocir-eval-data/MMDocIR_layouts.parquet"
ENHANCED_JSONL_DIR = "/kaggle/input/datasets/cdnghnam/siglip-qwen-enhaced/SIGLIP_QWEN_ENHACED/LAYOUT_CONTENT_FINAL"
ENHANCED_IMG_DIR   = "/kaggle/input/datasets/cdnghnam/siglip-qwen-enhaced/SIGLIP_QWEN_ENHACED/IMAGE ENHACED"
WORKING_DIR        = "/kaggle/working"

# ── Codebooks from training cell ──────────────────────────────────────────────
CODEBOOK_PATH_KMEANS_NQ16 = os.path.join(WORKING_DIR, "rvq_codebooks_kmeans_nq16.pt")
CODEBOOK_PATH_KMEANS_NQ32 = os.path.join(WORKING_DIR, "rvq_codebooks_kmeans_nq32.pt")

# ── Evaluation settings ───────────────────────────────────────────────────────
QUERY_BATCH_SIZE = 50
ADC_DOC_CHUNK    = 4000
BEAM_WIDTH       = 5          # beam width for Method B re-rank
TOP_K_RETRIEVE   = 100        # candidates from ADC before beam re-rank
NQ_EVAL_LEVELS   = None       # None = use all NQ levels in checkpoint; or list e.g. [8, 16]

BATCH_RANGE_PKL_OVERRIDE = {
    (0, 25): os.path.join(COLSMOL_DIR, "0-25.pkl"),
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# STEP 1: LOAD MODEL + LORA
# ==============================================================================
print("\n" + "="*70)
print("STEP 1: Load ColSmolVLM-500M + LoRA")
print("="*70)

from peft import PeftModel

gc.collect(); torch.cuda.empty_cache()

base = ColIdefics3.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager"
)
model = PeftModel.from_pretrained(base, LORA_PATH)
model.eval()
processor = ColIdefics3Processor.from_pretrained(BASE_MODEL)

print("✅ Model + LoRA loaded")

# ==============================================================================
# STEP 2: LOAD CODEBOOK
# ==============================================================================
print("\n" + "="*70)
print("STEP 2: Load KMeans RVQ codebook")
print("="*70)

all_codebooks_kmeans = {}
all_diag_kmeans = {}
emb_dim = None
RVQ_CODEBOOK_SIZE = None
RVQ_N_QUANTIZERS_COARSE = None
RVQ_N_QUANTIZERS_FINE = None

checkpoints_loaded = 0
for path in (CODEBOOK_PATH_KMEANS_NQ16, CODEBOOK_PATH_KMEANS_NQ32):
    if os.path.exists(path):
        print(f"  Loading checkpoint: {os.path.basename(path)}")
        ckpt = torch.load(path, map_location="cpu")
        all_codebooks_kmeans.update(ckpt["codebooks"])
        if "diagnostics" in ckpt:
            all_diag_kmeans.update(ckpt["diagnostics"])
        emb_dim = ckpt["emb_dim"]
        RVQ_CODEBOOK_SIZE = ckpt["codebook_size"]
        RVQ_N_QUANTIZERS_COARSE = ckpt["nq_coarse"]
        RVQ_N_QUANTIZERS_FINE = ckpt["nq_fine"]
        checkpoints_loaded += 1

if checkpoints_loaded == 0:
    raise FileNotFoundError(
        f"❌ Codebook checkpoints not found at {CODEBOOK_PATH_KMEANS_NQ16} or {CODEBOOK_PATH_KMEANS_NQ32}\n"
        "   Run cell train colsmol_layout MMdocir.py first!"
    )

nq_levels = sorted(all_codebooks_kmeans.keys())
if NQ_EVAL_LEVELS is not None:
    nq_levels = [nq for nq in NQ_EVAL_LEVELS if nq in all_codebooks_kmeans]

print(f"  Embedding dim:    {emb_dim}")
print(f"  Codebook size:    {RVQ_CODEBOOK_SIZE}")
print(f"  NQ levels avail:  {sorted(all_codebooks_kmeans.keys())}")
print(f"  NQ levels to eval:{nq_levels}")
for NQ in nq_levels:
    if NQ in all_diag_kmeans:
        d = all_diag_kmeans[NQ]
        print(f"  NQ={NQ}: align={d['cosine_alignment']:.4f}  "
              f"maxsim={d['maxsim_preservation']:.4f}  "
              f"dead={d['total_dead_codes']}/{NQ*RVQ_CODEBOOK_SIZE}")

# Move codebooks to device once
codebooks_on_device = {
    NQ: [cb.to(device) for cb in cbs]
    for NQ, cbs in all_codebooks_kmeans.items()
    if NQ in nq_levels
}

# ==============================================================================
# STEP 3: INFRASTRUCTURE — scoring / metric helpers
# ==============================================================================

# ── MaxSim full-float (baseline) ─────────────────────────────────────────────
def uniform_maxsim_scores(q_norm, doc_matrix, doc_mask, chunk_size=ADC_DOC_CHUNK):
    """Standard MaxSim: sum over query tokens of max-over-doc-tokens similarity."""
    N   = doc_matrix.shape[0]
    dev = q_norm.device
    out = torch.zeros(N, device=dev)
    for s in range(0, N, chunk_size):
        e   = min(s + chunk_size, N)
        sim = torch.einsum("qd,nld->qnl", q_norm.float(), doc_matrix[s:e].float())
        sim.masked_fill_(~doc_mask[s:e].unsqueeze(0), float("-inf"))
        ms  = sim.max(dim=-1).values
        ms  = ms.masked_fill(ms == float("-inf"), 0.0)
        out[s:e] = ms.sum(0)
    return out


# ── ADC (Asymmetric Distance Computation) with RVQ codebooks ─────────────────
@torch.no_grad()
def adc_scores_rvq(q_norm: torch.Tensor,
                   doc_codes: torch.Tensor,
                   codebooks: list,
                   chunk_size: int = ADC_DOC_CHUNK) -> torch.Tensor:
    """
    Pure ADC retrieval using quantized document codes.

    q_norm     : (Q, D)  — normalized query token embeddings (float)
    doc_codes  : (N, NQ) — integer codes per document layout (one code per stage)
    codebooks  : list of (CB_SIZE, D) tensors — one per RVQ stage
    Returns    : (N,) MaxSim-ADC scores
    """
    NQ = len(codebooks)
    N  = doc_codes.shape[0]
    Q  = q_norm.shape[0]
    dev = q_norm.device
    out = torch.zeros(N, device=dev)

    # Precompute query–centroid similarities for each stage: (NQ, Q, CB_SIZE)
    # q_norm: (Q, D),  cb: (CB_SIZE, D) → sim: (Q, CB_SIZE)
    stage_q_cb = []
    for k in range(NQ):
        cb = codebooks[k].float()          # (CB_SIZE, D)
        qcb = torch.mm(q_norm.float(), cb.t())  # (Q, CB_SIZE)
        stage_q_cb.append(qcb)            # (Q, CB_SIZE)

    for s in range(0, N, chunk_size):
        e     = min(s + chunk_size, N)
        chunk = doc_codes[s:e]             # (chunk, NQ)

        # For each layout (n), reconstruct the quantized embedding via centroids
        # then compute MaxSim with query tokens.
        # Efficient: gather centroid-query sims by index, sum across stages.
        # Result shape: (Q, chunk)
        adc_sim = torch.zeros(Q, e - s, device=dev)
        for k in range(NQ):
            idx     = chunk[:, k].long()          # (chunk,)
            # stage_q_cb[k]: (Q, CB_SIZE) → gather col idx → (Q, chunk)
            adc_sim = adc_sim + stage_q_cb[k][:, idx]

        # MaxSim: max over doc-patch dimension is just the single reconstructed vec
        # (no patch dimension here — one code per layout = one reconstructed vec)
        # MaxSim = sum_q adc_sim[q, n]  (max trivially over 1 patch)
        out[s:e] = adc_sim.sum(0)

    return out


@torch.no_grad()
def adc_scores_rvq_patched(q_norm: torch.Tensor,
                            doc_patch_codes: list,
                            codebooks: list,
                            chunk_size: int = ADC_DOC_CHUNK) -> torch.Tensor:
    """
    ADC retrieval when each document has MULTIPLE patches (variable-length).

    doc_patch_codes : list of (L_i, NQ) int tensors — one per layout document
    Returns         : (N,) MaxSim-ADC scores
    """
    NQ  = len(codebooks)
    N   = len(doc_patch_codes)
    Q   = q_norm.shape[0]
    dev = q_norm.device
    out = torch.zeros(N, device=dev)

    # Precompute (Q, CB_SIZE) per stage
    stage_q_cb = [torch.mm(q_norm.float(), codebooks[k].float().t())
                  for k in range(NQ)]

    for i in range(N):
        codes = doc_patch_codes[i].to(dev)  # (L_i, NQ)
        L     = codes.shape[0]
        # Reconstruct: (L, Q)  ← sum over NQ stages of stage_q_cb[k][:, codes[:,k]]
        recon_sim = torch.zeros(Q, L, device=dev)
        for k in range(NQ):
            idx       = codes[:, k].long()           # (L_i,)
            recon_sim = recon_sim + stage_q_cb[k][:, idx]   # (Q, L_i)
        # MaxSim: max over L, sum over Q
        ms_val = recon_sim.max(dim=1).values.sum()
        out[i] = ms_val

    return out


# ── Build full doc matrix (float16) ──────────────────────────────────────────
def _coerce(item):
    if isinstance(item, torch.Tensor): return item
    if isinstance(item, dict):
        for k in ("embedding", "embeddings", "vector", "vectors", "token_embeddings"):
            if k in item and isinstance(item[k], torch.Tensor): return item[k]
        tvs = [(k, v) for k, v in item.items() if isinstance(v, torch.Tensor)]
        if tvs: return tvs[0][1]
    raise ValueError(f"Cannot coerce type: {type(item)}")


def build_full(docs_list, device):
    """Padded, normalized doc matrix (float16 on CUDA)."""
    ts   = [_coerce(x) for x in docs_list]
    ts   = [t.squeeze(0) if t.dim() > 2 else (t.unsqueeze(0) if t.dim() == 1 else t)
            for t in ts]
    N    = len(ts)
    Lmax = max(t.shape[0] for t in ts)
    D    = ts[0].shape[1]
    dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
    pad  = torch.zeros(N, Lmax, D, device=device, dtype=dtype)
    mask = torch.zeros(N, Lmax, device=device, dtype=torch.bool)
    for i, t in enumerate(ts):
        L = t.shape[0]
        pad[i, :L]  = F.normalize(t.float().to(device), dim=-1).to(dtype)
        mask[i, :L] = True
    return pad, mask


# ── Quantize doc embeddings to codes ─────────────────────────────────────────
@torch.no_grad()
def quantize_to_codes(patch_emb: torch.Tensor, codebooks: list) -> torch.Tensor:
    """
    Greedily assign each patch to NQ codebook entries.
    patch_emb : (L, D) float32, L2-normalized
    Returns   : (L, NQ) int32 codes
    """
    NQ       = len(codebooks)
    L, D     = patch_emb.shape
    dev      = patch_emb.device
    codes    = torch.zeros(L, NQ, dtype=torch.int32, device=dev)
    residual = patch_emb.clone().float()
    for k, cb in enumerate(codebooks):
        cb_f  = cb.float().to(dev)           # (CB_SIZE, D)
        inner = torch.mm(residual, cb_f.t()) # (L, CB_SIZE)
        res_sq = (residual ** 2).sum(-1, keepdim=True)
        cb_sq  = (cb_f ** 2).sum(-1).unsqueeze(0)
        dist   = res_sq - 2 * inner + cb_sq  # (L, CB_SIZE)
        idx    = dist.argmin(-1)             # (L,)
        codes[:, k] = idx.int()
        residual = residual - cb_f[idx]
    return codes


# ── Quantize doc embeddings to codes using Beam Search ────────────────────────
@torch.no_grad()
def quantize_to_codes_beam(patch_emb: torch.Tensor, codebooks: list, beam_width: int = 5) -> torch.Tensor:
    """
    Quantize document patches using Beam Search to keep the top-B reconstruction paths.
    patch_emb : (L, D) float32, L2-normalized
    codebooks : list of NQ tensors, each (CB_SIZE, D)
    beam_width: B (default: 5)
    Returns   : (L, B, NQ) int32 codes
    """
    NQ       = len(codebooks)
    L, D     = patch_emb.shape
    B        = beam_width
    dev      = patch_emb.device

    if L == 0:
        return torch.zeros(0, B, NQ, dtype=torch.int32, device=dev)

    # Level 0: Find top B closest centroids in the first codebook stage
    cb0     = codebooks[0].float().to(dev)           # (CB_SIZE, D)
    inner0  = torch.mm(patch_emb, cb0.t())           # (L, CB_SIZE)
    res_sq0 = (patch_emb ** 2).sum(-1, keepdim=True) # (L, 1)
    cb_sq0  = (cb0 ** 2).sum(-1).unsqueeze(0)        # (1, CB_SIZE)
    dist0   = res_sq0 - 2 * inner0 + cb_sq0          # (L, CB_SIZE)

    val0, idx0 = torch.topk(-dist0, min(B, cb0.shape[0]), dim=-1) # (L, B)
    errors = -val0                                   # (L, B)

    residuals = patch_emb.unsqueeze(1) - cb0[idx0]   # (L, B, D)

    paths = torch.zeros(L, B, NQ, dtype=torch.int32, device=dev)
    paths[:, :, 0] = idx0.int()

    arange_L = torch.arange(L, device=dev).unsqueeze(1) # (L, 1)

    for k in range(1, NQ):
        cb      = codebooks[k].float().to(dev)        # (CB_SIZE, D)
        CB_SIZE = cb.shape[0]

        res_flat = residuals.reshape(L * B, D)
        inner    = torch.mm(res_flat, cb.t())         # (L * B, CB_SIZE)
        res_sq   = (res_flat ** 2).sum(-1, keepdim=True) # (L * B, 1)
        cb_sq    = (cb ** 2).sum(-1).unsqueeze(0)     # (1, CB_SIZE)
        dist     = res_sq - 2 * inner + cb_sq         # (L * B, CB_SIZE)
        dist     = dist.reshape(L, B, CB_SIZE)        # (L, B, CB_SIZE)

        cand_errors = errors.unsqueeze(-1) + dist     # (L, B, CB_SIZE)
        cand_flat   = cand_errors.reshape(L, B * CB_SIZE) # (L, B * CB_SIZE)

        val, top_flat_idx = torch.topk(-cand_flat, B, dim=-1) # (L, B)
        errors = -val                                 # (L, B)

        beam_from = top_flat_idx // CB_SIZE           # (L, B)
        cb_chosen = top_flat_idx % CB_SIZE            # (L, B)

        paths     = paths[arange_L, beam_from]        # (L, B, NQ)
        paths[:, :, k] = cb_chosen.int()

        par_res   = residuals[arange_L, beam_from]    # (L, B, D)
        residuals = par_res - cb[cb_chosen]           # (L, B, D)

    return paths


# ── Query encoding ────────────────────────────────────────────────────────────
def build_content_mask(inputs, processor):
    attn_mask = inputs["attention_mask"]
    input_ids = inputs.get("input_ids", None)
    if input_ids is None:
        return attn_mask.float()
    tok = getattr(processor, "tokenizer", processor)
    special_ids = set()
    for attr in ["pad_token_id", "bos_token_id", "eos_token_id",
                 "unk_token_id", "sep_token_id", "cls_token_id"]:
        tid = getattr(tok, attr, None)
        if tid is not None: special_ids.add(int(tid))
    if hasattr(tok, "added_tokens_encoder"):
        for _, tid in tok.added_tokens_encoder.items():
            special_ids.add(int(tid))
    if not special_ids:
        return attn_mask.float()
    special_tensor = torch.tensor(list(special_ids), device=input_ids.device)
    is_special = (input_ids.unsqueeze(-1) == special_tensor).any(dim=-1)
    return attn_mask.float() * (~is_special).float()


def encode_all_queries(qa_pairs, processor, model, device):
    encoded = []
    for i in range(0, len(qa_pairs), QUERY_BATCH_SIZE):
        batch = qa_pairs[i:i + QUERY_BATCH_SIZE]
        q_in  = processor.process_queries([it["question"] for it in batch]).to(device)
        with torch.no_grad():
            q_out = model(**{k: v for k, v in q_in.items()})
            q_embs = q_out.float() if isinstance(q_out, torch.Tensor) \
                     else q_out.last_hidden_state.float()
        cmasks = build_content_mask(q_in, processor).float()
        for j, item in enumerate(batch):
            cidx        = torch.where(cmasks[j] > 0)[0]
            raw_content = q_embs[j][cidx]
            q_norm      = F.normalize(raw_content, dim=-1)
            encoded.append({
                "q_norm":   q_norm.cpu(),
                "gt_set":   item["gt_pos_set"],
                "lmr":      item["layout_mapping_raw"],
                "domain":   item["domain"],
                "doc_name": item["doc_name"],
                "question": item["question"],
            })
        del q_in, q_embs, cmasks
    return encoded


# ── Metric helpers ────────────────────────────────────────────────────────────
def _parse_bbox(raw):
    if raw is None: return None
    try:
        if isinstance(raw, np.ndarray):
            f = raw.flatten()
            return [float(x) for x in f] if len(f) == 4 else None
    except: pass
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        try: return [float(x) for x in raw]
        except: return None
    if isinstance(raw, dict):
        for keys in [("x1","y1","x2","y2"), ("top","left","bottom","right"),
                     ("left","top","right","bottom")]:
            if all(k in raw for k in keys):
                try: return [float(raw[k]) for k in keys]
                except: pass
    return None

def overlap_area(b1, b2):
    it = max(b1[0], b2[0]); il = max(b1[1], b2[1])
    ib = min(b1[2], b2[2]); ir = min(b1[3], b2[3])
    return (ib - it) * (ir - il) if it < ib and il < ir else 0.0

def recall_area(top_k, bbox_list, lmr):
    ra = 0.0
    for p in top_k:
        info = bbox_list[p] if 0 <= p < len(bbox_list) else None
        if info is None: continue
        pg, t, l, b, r = info
        for gt in lmr:
            if pg != gt["page"]: continue
            gb = _parse_bbox(gt.get("bbox"))
            if gb: ra += overlap_area([t, l, b, r], gb)
    ga = 0.0
    for gt in lmr:
        gb = _parse_bbox(gt.get("bbox"))
        if gb: t2, l2, b2, r2 = gb; ga += max(0, b2-t2) * max(0, r2-l2)
    return 0.0 if ga <= 0 else min(ra / ga, 1.0)

def ndcg(ranked, gt, k):
    dcg  = sum(1 / np.log2(r + 2) for r, i in enumerate(ranked[:k]) if i in gt)
    idcg = sum(1 / np.log2(r + 2) for r in range(min(len(gt), k)))
    return dcg / idcg if idcg > 0 else 0.0

def hit_metrics(top10, gt_set, bbox_list, lmr):
    if not lmr: return None
    h = next((r + 1 for r, i in enumerate(top10) if i in gt_set), -1)
    return {
        "r1":       int(h != -1 and h <= 1),
        "r5":       int(h != -1 and h <= 5),
        "r10":      int(h != -1 and h <= 10),
        "recall1":  recall_area(top10[:1],  bbox_list, lmr),
        "recall5":  recall_area(top10[:5],  bbox_list, lmr),
        "recall10": recall_area(top10[:10], bbox_list, lmr),
        "n1":       ndcg(top10, gt_set, 1),
        "n5":       ndcg(top10, gt_set, 5),
        "n10":      ndcg(top10, gt_set, 10),
    }

# ── Metric accumulator ────────────────────────────────────────────────────────
def _init_m():
    return {"r1": 0, "r5": 0, "r10": 0,
            "n1": 0., "n5": 0., "n10": 0.,
            "recall1": 0., "recall5": 0., "recall10": 0., "count": 0}

def _add(d, s):
    for f in ("r1", "r5", "r10"): d[f] += int(s[f])
    for f in ("n1", "n5", "n10", "recall1", "recall5", "recall10"): d[f] += float(s[f])
    d["count"] += 1

def _ens(store, k):
    if k not in store: store[k] = _init_m()
    return store[k]

def record(key, m, domain, all_metrics, all_domain_metrics):
    _add(_ens(all_metrics, key), m)
    if domain not in all_domain_metrics: all_domain_metrics[domain] = {}
    _add(_ens(all_domain_metrics[domain], key), m)

def print_metrics(store, label=""):
    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"{'─'*70}")
    for key, m in sorted(store.items()):
        c = max(m["count"], 1)
        print(f"  [{key}]  N={m['count']}")
        print(f"    Hit@1={m['r1']/c:.4f}  Hit@5={m['r5']/c:.4f}  Hit@10={m['r10']/c:.4f}")
        print(f"    NDCG@1={m['n1']/c:.4f}  NDCG@5={m['n5']/c:.4f}  NDCG@10={m['n10']/c:.4f}")
        print(f"    Recall@1={m['recall1']/c:.4f}  Recall@5={m['recall5']/c:.4f}  Recall@10={m['recall10']/c:.4f}")

# ==============================================================================
# STEP 4: LOAD PKL FILES + BUILD BATCH RANGES
# ==============================================================================
print("\n" + "="*70)
print("STEP 4: Discover PKL batches")
print("="*70)

pkl_files = sorted(glob.glob(os.path.join(COLSMOL_DIR, "*.pkl")))
print(f"  Found {len(pkl_files)} PKL files")

if len(pkl_files) == 0:
    raise ValueError("❌ No PKL files found. Check COLSMOL_DIR path!")

BATCH_RANGES = []
for p in pkl_files:
    base = os.path.basename(p).replace(".pkl", "")
    try:
        s, e = map(int, base.split("-"))
        BATCH_RANGES.append((s, e))
    except:
        try:
            s, e = map(int, base.split(" ")[-1].split("-"))
            BATCH_RANGES.append((s, e))
        except:
            pass

BATCH_RANGES = sorted(BATCH_RANGES) or [(i, i + 1) for i in range(len(pkl_files))]
for r, p in BATCH_RANGE_PKL_OVERRIDE.items():
    if r not in BATCH_RANGES:
        BATCH_RANGES = sorted(set(BATCH_RANGES + [r]))

print(f"  Batch ranges: {BATCH_RANGES}")

# ==============================================================================
# STEP 5: LOAD ANNOTATIONS + BUILD DOC / QA STRUCTURES
# ==============================================================================
print("\n" + "="*70)
print("STEP 5: Load annotations + intersection docs")
print("="*70)

valid_docs = set()
with open(ANNOTATIONS_PATH) as f:
    for line in f:
        try: valid_docs.add(json.loads(line)["doc_name"].replace(".pdf", ""))
        except: pass

jsonl_map = {
    os.path.basename(p).replace("_layout.jsonl", ""): p
    for p in glob.glob(os.path.join(ENHANCED_JSONL_DIR, "*.jsonl"))
}
intersection_docs = sorted(valid_docs.intersection(jsonl_map.keys()))
print(f"  Intersection docs: {len(intersection_docs)}")

# Pre-load all annotations grouped by doc
print("  Loading QA annotations...")
qa_by_doc: dict[str, list] = {}
with open(ANNOTATIONS_PATH) as f:
    for line in f:
        try:
            d = json.loads(line)
            doc = d["doc_name"].replace(".pdf", "")
            if doc not in qa_by_doc:
                qa_by_doc[doc] = []
            qa_by_doc[doc].append(d)
        except:
            pass

# Parquet for bbox
print("  Loading Parquet for bbox...")
df_parquet_all = pd.read_parquet(PARQUET_PATH)
df_parquet_all["join_doc_name"] = df_parquet_all["doc_name"].str.replace(".pdf", "", regex=False)

# ==============================================================================
# STEP 6: MAIN EVALUATION LOOP — iterate over batches
# ==============================================================================
print("\n" + "="*70)
print("STEP 6: Evaluation loop")
print("="*70)

# Global accumulators {method_key: metric_dict}
all_metrics        = {}
all_domain_metrics = {}

for batch_idx, (START_IDX, END_IDX) in enumerate(BATCH_RANGES):
    print(f"\n{'═'*70}")
    print(f"  BATCH [{START_IDX}:{END_IDX}]  ({batch_idx+1}/{len(BATCH_RANGES)})")
    print(f"{'═'*70}")

    # ── Find PKL file for this batch ──────────────────────────────────────────
    pkl_path = BATCH_RANGE_PKL_OVERRIDE.get((START_IDX, END_IDX))
    if pkl_path is None:
        candidates = [p for p in pkl_files
                      if f"{START_IDX}-{END_IDX}" in os.path.basename(p)]
        pkl_path = candidates[0] if candidates else None

    if pkl_path is None or not os.path.exists(pkl_path):
        print(f"  ⚠️  No PKL found for batch [{START_IDX}:{END_IDX}], skipping")
        continue

    # ── Load PKL ──────────────────────────────────────────────────────────────
    print(f"  Loading PKL: {os.path.basename(pkl_path)}")
    with open(pkl_path, "rb") as f:
        pkl_data = pickle.load(f)

    # pkl_data expected format:
    #   list of dicts with keys like 'embedding', 'doc_name', 'layout_id', 'page_id', ...
    #   OR dict mapping doc_name -> list of layout embeddings
    # Handle both cases:
    if isinstance(pkl_data, dict):
        # dict: {doc_name: [embedding_tensor, ...]} or {doc_name: {layout_id: tensor}}
        layouts_list = []
        for doc_name, entries in pkl_data.items():
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict):
                        entry["doc_name"] = doc_name
                        layouts_list.append(entry)
                    elif isinstance(entry, torch.Tensor):
                        layouts_list.append({"doc_name": doc_name, "embedding": entry})
    elif isinstance(pkl_data, list):
        layouts_list = pkl_data
    else:
        print(f"  ❌ Unknown PKL format: {type(pkl_data)}, skipping")
        continue

    print(f"  PKL entries: {len(layouts_list)}")
    if len(layouts_list) == 0:
        print(f"  ⚠️  Empty PKL, skipping")
        continue

    # ── Select docs for this batch ────────────────────────────────────────────
    batch_docs = intersection_docs[START_IDX:END_IDX]
    batch_docs_set = set(batch_docs)
    print(f"  Target docs: {len(batch_docs)}")

    # Filter layouts to batch docs
    def _get_doc_name(entry):
        for k in ("doc_name", "document", "doc"):
            if k in entry: return str(entry[k]).replace(".pdf", "")
        return None

    layouts_in_batch = [e for e in layouts_list
                        if _get_doc_name(e) in batch_docs_set]
    print(f"  Layouts in batch: {len(layouts_in_batch)}")

    if len(layouts_in_batch) == 0:
        print("  ⚠️  No layouts match batch docs, skipping")
        del pkl_data, layouts_list; gc.collect()
        continue

    # ── Build doc-level index ─────────────────────────────────────────────────
    # Group layouts by doc, assign global layout indices
    doc_to_layout_idx: dict[str, list] = {}
    layout_embeddings: list            = []   # (L_i, D) tensors in order
    bbox_list: list                    = []   # (page, t, l, b, r) or None

    # Parquet subset for bbox
    df_batch = df_parquet_all[df_parquet_all["join_doc_name"].isin(batch_docs_set)].copy()
    df_batch = df_batch.sort_values(["join_doc_name", "page_id", "layout_id"])

    # Index parquet rows for fast bbox lookup
    parquet_idx = {}
    for _, row in df_batch.iterrows():
        key = (str(row["join_doc_name"]).replace(".pdf",""), int(row.get("page_id", -1)),
               int(row.get("layout_id", -1)))
        bbox = None
        for bc in ("bbox", "bounding_box", "coords"):
            if bc in row and pd.notna(row[bc]):
                bbox = row[bc]; break
        parquet_idx[key] = {
            "page":  int(row.get("page_id", -1)),
            "bbox":  bbox,
        }

    for entry in layouts_in_batch:
        doc = _get_doc_name(entry)
        if doc is None: continue

        emb = _coerce(entry) if isinstance(entry, dict) and any(
            k in entry for k in ("embedding","embeddings","vector","vectors","token_embeddings")
        ) else None
        if emb is None:
            # Try direct tensor
            for k, v in entry.items():
                if isinstance(v, torch.Tensor):
                    emb = v; break
        if emb is None: continue

        emb = emb.squeeze(0) if emb.dim() > 2 else emb
        if emb.dim() == 1: emb = emb.unsqueeze(0)  # (1, D)

        global_idx = len(layout_embeddings)
        if doc not in doc_to_layout_idx:
            doc_to_layout_idx[doc] = []
        doc_to_layout_idx[doc].append(global_idx)
        layout_embeddings.append(emb.float())

        page  = int(entry.get("page_id", entry.get("page", -1)))
        lid   = int(entry.get("layout_id", entry.get("layout", -1)))
        bbox_info = parquet_idx.get((doc, page, lid))
        if bbox_info:
            bb = _parse_bbox(bbox_info["bbox"])
            if bb:
                bbox_list.append((page, bb[0], bb[1], bb[2], bb[3]))
            else:
                bbox_list.append(None)
        else:
            bbox_list.append(None)

    N_layouts = len(layout_embeddings)
    print(f"  Total layout embeddings: {N_layouts}")
    del pkl_data, layouts_list, layouts_in_batch; gc.collect()

    if N_layouts == 0:
        print("  ⚠️  No usable embeddings, skipping")
        continue

    # ── Build QA pairs for batch docs ────────────────────────────────────────
    qa_pairs = []
    for doc in batch_docs:
        if doc not in qa_by_doc: continue
        layout_idx_for_doc = doc_to_layout_idx.get(doc, [])
        if not layout_idx_for_doc: continue

        for ann in qa_by_doc[doc]:
            question = ann.get("question", "")
            if not question: continue
            # Ground truth: set of global layout indices
            gt_layout_ids = set()
            lmr = []
            for gt_entry in ann.get("layout_mapping", []):
                page = gt_entry.get("page", -1)
                lid  = gt_entry.get("layout_id", gt_entry.get("layout", -1))
                lmr.append(gt_entry)
                # Find matching global index
                for gi in layout_idx_for_doc:
                    entry_page = -1
                    entry_lid  = -1
                    # We need to trace back - store metadata alongside
                    # Using parquet_idx lookup
                    bb_info = bbox_list[gi]
                    if bb_info is not None:
                        ep = bb_info[0] if isinstance(bb_info, tuple) else -1
                        if ep == page:
                            gt_layout_ids.add(gi)
            if not gt_layout_ids: continue
            qa_pairs.append({
                "question":          question,
                "doc_name":          doc,
                "gt_pos_set":        gt_layout_ids,
                "layout_mapping_raw": lmr,
                "domain":            ann.get("domain", "unknown"),
            })

    print(f"  QA pairs: {len(qa_pairs)}")
    if len(qa_pairs) == 0:
        print("  ⚠️  No QA pairs found for batch, skipping")
        del layout_embeddings, doc_to_layout_idx; gc.collect()
        continue

    # ── Encode all queries ────────────────────────────────────────────────────
    print("  Encoding queries...")
    t0 = time.perf_counter()
    encoded_queries = encode_all_queries(qa_pairs, processor, model, device)
    print(f"  Query encoding: {time.perf_counter()-t0:.1f}s  ({len(encoded_queries)} queries)")

    # ── Build full doc matrix (for baseline MaxSim) ───────────────────────────
    print("  Building full doc matrix (float16)...")
    doc_matrix, doc_mask = build_full(layout_embeddings, device)
    print(f"  doc_matrix: {doc_matrix.shape}")

    # ── Quantize all doc embeddings for ADC ──────────────────────────────────
    print("  Quantizing doc embeddings to codes (greedy + beam)...")
    doc_codes_by_nq: dict[int, list] = {}        # {NQ: list of (L_i, NQ) tensors}
    doc_codes_beam_by_nq: dict[int, list] = {}   # {NQ: list of (L_i * B, NQ) tensors}
    for NQ in nq_levels:
        cbs = codebooks_on_device[NQ]
        doc_codes_nq = []
        doc_codes_beam_nq = []
        for emb in tqdm(layout_embeddings, desc=f"  Quantize NQ={NQ}", leave=False):
            emb_n = F.normalize(emb.to(device), dim=-1)
            # Greedy quantization (Method A)
            codes = quantize_to_codes(emb_n, cbs)  # (L_i, NQ)
            doc_codes_nq.append(codes.cpu())
            # Beam search quantization (Method B)
            codes_beam = quantize_to_codes_beam(emb_n, cbs, beam_width=BEAM_WIDTH) # (L_i, B, NQ)
            doc_codes_beam_nq.append(codes_beam.reshape(-1, NQ).cpu())
        doc_codes_by_nq[NQ] = doc_codes_nq
        doc_codes_beam_by_nq[NQ] = doc_codes_beam_nq
        print(f"  NQ={NQ}: {len(doc_codes_nq)} docs quantized (greedy & beam)")

    # ──────────────────────────────────────────────────────────────────────────
    # EVALUATION — per query
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n  Evaluating {len(encoded_queries)} queries...")

    for qi, qdata in enumerate(tqdm(encoded_queries, desc="  Queries")):
        q_norm    = qdata["q_norm"].to(device)         # (Q_tok, D)
        gt_set    = qdata["gt_set"]
        lmr       = qdata["lmr"]
        domain    = qdata["domain"]

        if not gt_set: continue

        # ── Method 0: Full-float MaxSim (baseline) ───────────────────────────
        scores_full = uniform_maxsim_scores(q_norm, doc_matrix, doc_mask)
        ranked_full = scores_full.argsort(descending=True).cpu().tolist()[:10]
        m0 = hit_metrics(ranked_full, gt_set, bbox_list, lmr)
        if m0: record("baseline_maxsim", m0, domain, all_metrics, all_domain_metrics)

        # ── Method A + B per NQ level ─────────────────────────────────────────
        for NQ in nq_levels:
            cbs       = codebooks_on_device[NQ]
            doc_codes = doc_codes_by_nq[NQ]

            # ── Method A: Pure ADC full-scan ─────────────────────────────────
            scores_adc = adc_scores_rvq_patched(q_norm, doc_codes, cbs)
            ranked_adc = scores_adc.argsort(descending=True).cpu().tolist()[:10]
            mA = hit_metrics(ranked_adc, gt_set, bbox_list, lmr)
            if mA: record(f"A_adc_NQ{NQ}", mA, domain, all_metrics, all_domain_metrics)

            # ── Method B: RVQ + Beam Search (b=5) full-scan ──────────────────
            # Không cần vector gốc (float) — chỉ dùng codes (int) + codebook.
            doc_codes_beam = doc_codes_beam_by_nq[NQ]
            scores_beam = adc_scores_rvq_patched(q_norm, doc_codes_beam, cbs)
            ranked_beam = scores_beam.argsort(descending=True).cpu().tolist()[:10]
            mB = hit_metrics(ranked_beam, gt_set, bbox_list, lmr)
            if mB: record(f"B_rvq_beam_NQ{NQ}_b{BEAM_WIDTH}", mB, domain,
                          all_metrics, all_domain_metrics)

        del q_norm, scores_full

    gc.collect(); torch.cuda.empty_cache()

    # ── Print batch results ───────────────────────────────────────────────────
    print_metrics(all_metrics, label=f"CUMULATIVE after batch [{START_IDX}:{END_IDX}]")

    del doc_matrix, doc_mask, doc_codes_by_nq, doc_codes_beam_by_nq, layout_embeddings; gc.collect()

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "█"*70)
print("FINAL EVALUATION RESULTS — MMDocIR Test Set")
print("█"*70)
print_metrics(all_metrics, label="ALL METHODS — OVERALL")

# Per-domain breakdown
print("\n" + "="*70)
print("  Per-domain breakdown:")
for domain, dom_store in sorted(all_domain_metrics.items()):
    print(f"\n  Domain: {domain}")
    print_metrics(dom_store, label=f"  domain={domain}")

# ── Save results ───────────────────────────────────────────────────────────────
save_path = os.path.join(WORKING_DIR, "eval_results_kmeans_rvq.json")
results_out = {
    "overall":    {k: dict(v) for k, v in all_metrics.items()},
    "per_domain": {d: {k: dict(v) for k, v in m.items()}
                   for d, m in all_domain_metrics.items()},
    "config": {
        "nq_levels":     nq_levels,
        "top_k_retrieve": TOP_K_RETRIEVE,
        "beam_width":    BEAM_WIDTH,
        "codebook_size": RVQ_CODEBOOK_SIZE,
        "emb_dim":       emb_dim,
    }
}
with open(save_path, "w") as f:
    json.dump(results_out, f, indent=2)
print(f"\n✅ Results saved → {save_path}")
print("="*70)
