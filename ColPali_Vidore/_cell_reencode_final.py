# ==============================================================================
# BƯỚC 2.5: RE-ENCODE CORPUS (IMAGE-ONLY) TRONG SESSION HIỆN TẠI
# Copy nguyên logic từ ColPali-v1.3 encode.ipynb
# Chạy SAU cell 2 (model loaded), THAY THẾ cell load pkl
#
# Ước tính: ~13 phút trên RTX Pro 6000 (batch=12, 19252 pages)
# ==============================================================================
import os, gc, io, pickle, time, re
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm.notebook import tqdm

WORKING_DIR = "/kaggle/working"
SAVE_PATH = os.path.join(WORKING_DIR, "colpali13_vidore_v3_LIVE.pkl")
CKPT_PATH = os.path.join(WORKING_DIR, "colpali13_vidore_v3_LIVE_ckpt.pkl")

# Giống hệt encode notebook
BATCH_SIZE = 12
SAVE_EVERY = 900
_autocast_dtype = torch.bfloat16

VIDORE_DATASET_ROOT = "/kaggle/input/datasets/namthi/vidore-v3"
VIDORE_DOMAINS = [
    "vidore_v3_computer_science", "vidore_v3_energy", "vidore_v3_finance_en",
    "vidore_v3_hr", "vidore_v3_industrial", "vidore_v3_pharmaceuticals", "vidore_v3_physics",
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f">>> Device: {device}, Batch: {BATCH_SIZE}")

# ============================================================
# BƯỚC 1: Load tất cả corpus pages vào DataFrame
# (Copy từ encode notebook BƯỚC 1)
# ============================================================
print(">>> Loading corpus pages from ViDoRe V3...")
root = Path(VIDORE_DATASET_ROOT)

def _extract_img(image_value):
    if image_value is None: return None
    if isinstance(image_value, dict):
        b = image_value.get("bytes")
        if isinstance(b, (bytes, bytearray)) and len(b) > 0: return bytes(b)
    if isinstance(image_value, (bytes, bytearray)) and len(image_value) > 0:
        return bytes(image_value)
    return None

rows = []
for domain_name in VIDORE_DOMAINS:
    domain_dir = root / domain_name
    if not domain_dir.exists(): continue
    corpus_files = sorted(domain_dir.rglob("corpus/*.parquet"))
    for fp in corpus_files:
        try:
            df = pd.read_parquet(fp)
        except: continue
        for _, row in df.iterrows():
            img_bytes = _extract_img(row.get("image"))
            if img_bytes is None: continue
            cid = str(row.get("corpus_id", "")).strip()
            doc_id = str(row.get("doc_id", "")).strip()
            page = int(float(row.get("page_number_in_doc", 0)))
            pk = f"{domain_name}__{doc_id}__p{page}__{cid}"
            rows.append({
                "domain": domain_name, "doc_id": doc_id, "page": page,
                "corpus_id": cid, "page_key": pk, "img_bytes": img_bytes,
            })
        del df; gc.collect()
    print(f"  {domain_name}: {sum(1 for r in rows if r['domain']==domain_name)} pages")

pages_df = pd.DataFrame(rows)
pages_df = pages_df.drop_duplicates(subset=["page_key"], keep="first")
pages_df = pages_df.sort_values(by=["domain","doc_id","page"]).reset_index(drop=True)
del rows; gc.collect()

print(f">>> Total pages to encode: {len(pages_df)}")

# ============================================================
# BƯỚC 2: Encode (copy từ encode notebook BƯỚC 3, image-only)
# ============================================================
fused_index = []
start_idx = 0

# Resume từ checkpoint nếu có
if os.path.exists(CKPT_PATH):
    print(">>> Resuming from checkpoint...")
    with open(CKPT_PATH, "rb") as f:
        ckpt = pickle.load(f)
    fused_index = ckpt.get("fused_index", [])
    start_idx = int(ckpt.get("next_row", len(fused_index)))
    print(f"  Resumed {len(fused_index)} pages, continue from row {start_idx}")
    del ckpt; gc.collect()

def load_img_safe(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if img.width < 14 or img.height < 14:
            return Image.new("RGB", (224, 224), "white")
        return img
    except:
        return Image.new("RGB", (224, 224), "white")

remaining = pages_df.iloc[start_idx:]
total = len(pages_df)
batch_imgs = []

print(f">>> Encoding {len(remaining)} remaining pages (image-only)...")
t0 = time.time()

for i, (_, row) in enumerate(tqdm(remaining.iterrows(), total=len(remaining), desc="Encoding")):
    batch_imgs.append(load_img_safe(row["img_bytes"]))
    
    if len(batch_imgs) >= BATCH_SIZE or i == len(remaining) - 1:
        with torch.no_grad():
            # Giống hệt encode notebook: autocast context
            with torch.autocast(device_type="cuda", dtype=_autocast_dtype, enabled=True):
                vis_inputs = processor.process_images(batch_imgs).to(device)
                vis_out = model(**vis_inputs)
                vis_list = list(torch.unbind(vis_out.float().cpu()))
            
            for v in vis_list:
                fused_index.append(v.to(torch.float16).contiguous())
        
        del vis_inputs, vis_out, vis_list
        batch_imgs = []
        
        # Checkpoint
        if len(fused_index) > 0 and len(fused_index) % SAVE_EVERY == 0:
            with open(CKPT_PATH, "wb") as f:
                pickle.dump({"fused_index": fused_index, "next_row": start_idx + i + 1}, f)
            elapsed = time.time() - t0
            speed = (i+1) / elapsed
            eta = (len(remaining) - i - 1) / speed
            print(f"  Checkpoint: {len(fused_index)}/{total} pages ({elapsed:.0f}s, ETA {eta:.0f}s)")

elapsed = time.time() - t0
print(f">>> Encoding done: {len(fused_index)} pages in {elapsed:.0f}s ({elapsed/60:.1f} min)")

# ============================================================
# BƯỚC 3: Save index
# ============================================================
page_keys = pages_df["page_key"].tolist()[:len(fused_index)]
meta_cols = ["domain","doc_id","page","corpus_id","page_key"]
metadata_list = pages_df[meta_cols].iloc[:len(fused_index)].to_dict("records")

payload = {
    "model_name": "vidore/colpali-v1.3",
    "dataset": "vidore-v3",
    "index_level": "page",
    "index_type": "image_only",
    "num_pages": len(fused_index),
    "fused_index": fused_index,
    "page_keys": page_keys,
    "page_key_to_row": {k: i for i, k in enumerate(page_keys)},
    "metadata": metadata_list,
}
with open(SAVE_PATH, "wb") as f:
    pickle.dump(payload, f)

if os.path.exists(CKPT_PATH):
    os.remove(CKPT_PATH)

print(f">>> Saved: {SAVE_PATH} ({os.path.getsize(SAVE_PATH)/1e6:.0f} MB)")

# ============================================================
# BƯỚC 4: Chuẩn bị all_page_embeddings + QA pairs cho method cells
# ============================================================

# Free img_bytes (lớn nhất)
del pages_df; gc.collect()

all_page_embeddings = []
for emb in fused_index:
    all_page_embeddings.append(emb.float().cpu().numpy())

print(f">>> all_page_embeddings: {len(all_page_embeddings)}")
print(f">>> Embedding shape: {all_page_embeddings[0].shape}")

# Build QA pairs
def _normalize_corpus_id(v):
    if v is None: return None
    if isinstance(v, float) and np.isnan(v): return None
    if isinstance(v, (int, np.integer)): return str(int(v))
    if isinstance(v, (float, np.floating)):
        if np.isnan(v): return None
        return str(int(v)) if float(v).is_integer() else str(v)
    s = str(v).strip()
    if not s: return None
    if re.fullmatch(r"\d+\.0+", s): return s.split(".")[0]
    return s

domain_cid_to_idx = {}
global_cid_to_idx = {}
for idx, m in enumerate(metadata_list):
    d = m['domain']
    c = _normalize_corpus_id(m['corpus_id'])
    if c is None: continue
    domain_cid_to_idx[(d, c)] = idx
    global_cid_to_idx[c] = idx

print(f">>> Lookup: {len(domain_cid_to_idx)} domain keys")

qa_pairs = []
match_stats = {"matched_domain_key": 0, "matched_global_fallback": 0, "missed": 0}

for domain_name in VIDORE_DOMAINS:
    domain_dir = root / domain_name
    if not domain_dir.exists(): continue
    qrel_files = sorted(domain_dir.rglob("qrels/*.parquet"))
    query_files = sorted(domain_dir.rglob("queries/*.parquet"))
    if not qrel_files or not query_files: continue
    
    qrels_df = pd.concat([pd.read_parquet(f) for f in qrel_files], ignore_index=True)
    queries_df = pd.concat([pd.read_parquet(f) for f in query_files], ignore_index=True)
    if 'score' in qrels_df.columns:
        qrels_df = qrels_df[qrels_df['score'] > 0]
    qrels_df['corpus_id_norm'] = qrels_df['corpus_id'].map(_normalize_corpus_id)
    merged = queries_df[['query_id','query']].merge(
        qrels_df[['query_id','corpus_id_norm']], on='query_id', how='inner')
    
    domain_qa = 0
    for qid, grp in merged.groupby('query_id'):
        q_text = str(grp.iloc[0]['query']).strip()
        if len(q_text) < 3: continue
        gt_indices = []
        for _, row in grp.iterrows():
            cid = row['corpus_id_norm']
            idx = domain_cid_to_idx.get((domain_name, cid))
            if idx is not None:
                gt_indices.append(idx); match_stats['matched_domain_key'] += 1
            else:
                idx = global_cid_to_idx.get(cid)
                if idx is not None:
                    gt_indices.append(idx); match_stats['matched_global_fallback'] += 1
                else:
                    match_stats['missed'] += 1
        if gt_indices:
            qa_pairs.append({
                'question': q_text,
                'gt_embed_indices': sorted(set(gt_indices)),
                'doc_name': domain_name,
                'domain': domain_name,
            })
            domain_qa += 1
    print(f"  {domain_name}: {domain_qa} queries")

print(f"\n>>> QA pairs: {len(qa_pairs)}")
print(f">>> Match stats: {match_stats}")

# Alias cho method cells
query_model = model
query_processor = processor
print(">>> READY — chạy method cells tiếp theo!")
