# ==============================================================================
# CHẨN ĐOÁN — viết GIỐNG HỆT encode notebook, KHÔNG gc/empty_cache giữa chừng
# Chạy NGAY SAU cell 2 (model), TRƯỚC cell load pkl
# ==============================================================================
import torch, io, os, pickle
import torch.nn.functional as F
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

device = "cuda" if torch.cuda.is_available() else "cpu"
_autocast_dtype = torch.bfloat16

print("="*70)
print("CHẨN ĐOÁN — encode giống hệt encode notebook")
print("="*70)

# 1. Load 1 ảnh từ corpus
corpus_file = "/kaggle/input/datasets/namthi/vidore-v3/vidore_v3_computer_science/vidore_v3_computer_science/corpus/test-00000-of-00002.parquet"
table = pq.read_table(corpus_file, columns=['corpus_id','image']).slice(0, 1)
cid = int(table.column('corpus_id')[0].as_py())
img_data = table.column('image')[0].as_py()
page_img = Image.open(io.BytesIO(img_data['bytes'] if isinstance(img_data, dict) else img_data)).convert("RGB")
del table, img_data
print(f"[1] corpus_id={cid}, img={page_img.size}")

# 2. Encode image + query TRONG CÙNG 1 autocast context (giống encode notebook)
q_text = "What is Introduction to Python Programming?"

with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=_autocast_dtype, enabled=True):
        # Image — giống hệt encode notebook
        vis_inputs = processor.process_images([page_img]).to(device)
        vis_out = model(**vis_inputs)
        live_emb = vis_out[0].float().cpu() if isinstance(vis_out, torch.Tensor) else vis_out.last_hidden_state[0].float().cpu()
        
        # Query — giống hệt encode notebook dùng process_queries
        txt_inputs = processor.process_queries([q_text]).to(device)
        txt_out = model(**txt_inputs)
        q_raw = txt_out[0].float().cpu() if isinstance(txt_out, torch.Tensor) else txt_out.last_hidden_state[0].float().cpu()

del vis_inputs, vis_out, txt_inputs, txt_out, page_img

print(f"[2] LIVE image: {live_emb.shape}")
print(f"[3] Query raw:  {q_raw.shape}")

# 3. MaxSim (trên CPU, không cần GPU)
live_norm = F.normalize(live_emb, dim=-1)
q_norm = F.normalize(q_raw, dim=-1)
sim = torch.einsum('qd,ld->ql', q_norm, live_norm)
maxsim = sim.max(dim=-1).values
score_live = maxsim.sum().item()
print(f"\n[4] MaxSim query vs LIVE image: {score_live:.3f}")

# 4. Load pkl, tìm corpus_id, so sánh
INDEX_PATH = "/kaggle/input/datasets/namthi/vidore-encoded/colpali13_page_index_vidore_v3_pagelevel.pkl"
print(f"\n[5] Loading pkl...")
with open(INDEX_PATH, "rb") as f:
    payload = pickle.load(f)

pkl_idx = payload.get("fused_index", payload.get("embeddings", []))
pkl_keys = payload.get("page_keys", [])
print(f"    pkl: {len(pkl_idx)} pages")

# Tìm page với corpus_id
stored_pos = None
for i, pk in enumerate(pkl_keys):
    if f"__{cid}" in pk and "computer_science" in pk:
        stored_pos = i
        break

if stored_pos is not None:
    stored_raw = pkl_idx[stored_pos]
    stored = stored_raw.float() if isinstance(stored_raw, torch.Tensor) else torch.from_numpy(np.array(stored_raw, dtype=np.float32))
    n_vis = live_emb.shape[0]  # 1031
    
    print(f"    Found: idx={stored_pos}, key={pkl_keys[stored_pos]}")
    print(f"    STORED: {stored.shape}, LIVE: {live_emb.shape}")
    
    # So sánh image tokens
    cos = F.cosine_similarity(live_emb, stored[:n_vis], dim=-1)
    print(f"\n    === SO SÁNH IMAGE TOKENS ===")
    print(f"    Cosine: mean={cos.mean():.6f}, min={cos.min():.6f}, max={cos.max():.6f}")
    
    # MaxSim query vs stored
    stored_img_norm = F.normalize(stored[:n_vis], dim=-1)
    sim_s = torch.einsum('qd,ld->ql', q_norm, stored_img_norm)
    score_stored = sim_s.max(dim=-1).values.sum().item()
    
    stored_full_norm = F.normalize(stored, dim=-1)
    sim_f = torch.einsum('qd,ld->ql', q_norm, stored_full_norm)
    score_full = sim_f.max(dim=-1).values.sum().item()
    
    print(f"\n    === MAXSIM ===")
    print(f"    Query vs LIVE image:          {score_live:.3f}")
    print(f"    Query vs STORED image part:   {score_stored:.3f}")
    print(f"    Query vs STORED full (fused): {score_full:.3f}")
    
    if cos.mean() > 0.99:
        print(f"\n    ✅ pkl KHỚP model hiện tại (cosine={cos.mean():.6f})")
    elif cos.mean() > 0.90:
        print(f"\n    ⚠️ Khác nhẹ (cosine={cos.mean():.6f}) — float16?")
    else:
        print(f"\n    ❌ pkl KHÔNG KHỚP (cosine={cos.mean():.6f})")
else:
    print(f"    ❌ corpus_id={cid} KHÔNG có trong pkl!")
    print(f"    Sample keys: {pkl_keys[:3]}")

del pkl_idx, pkl_keys, payload
print(f"\n{'='*70}")
print("XONG — paste output cho tôi!")
