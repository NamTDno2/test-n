# ==============================================================================
# PHÂN TÍCH SÂU — KHÔNG encode image, KHÔNG crash
# Chạy SAU cell load pkl + QA pairs (sau cell 3,4)
# Dùng QUERY encoding (đã hoạt động) + stored embeddings
# ==============================================================================
import torch, gc, time
import torch.nn.functional as F
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print("="*70)
print("PHÂN TÍCH SÂU — tại sao R@10 = 0%")
print("="*70)

# Kiểm tra biến từ cell trước
assert 'all_page_embeddings' in dir(), "Chạy cell load pkl trước!"
assert 'qa_pairs' in dir(), "Chạy cell build QA trước!"

n_docs = len(all_page_embeddings)
print(f"Docs: {n_docs}, Queries: {len(qa_pairs)}")

# Lấy 5 query đầu
for qi in range(min(5, len(qa_pairs))):
    item = qa_pairs[qi]
    gt_set = set(item['gt_embed_indices'])
    domain = item['domain']
    q_text = item['question'][:100]
    
    # Encode query (dùng model — ĐÃ CHẠY OK với 200 queries)
    q_inp = query_processor.process_queries([item['question']]).to(device)
    with torch.no_grad():
        q_out = query_model(**q_inp)
    
    if isinstance(q_out, torch.Tensor):
        q_proj = q_out
    else:
        q_proj = q_out[0]
    
    am = q_inp['attention_mask'][0] > 0
    q_emb = F.normalize(q_proj[0][am].float(), dim=-1)  # (Q, 128) on GPU
    
    print(f"\n{'='*70}")
    print(f"Q{qi}: {q_text}")
    print(f"  Domain: {domain}")
    print(f"  GT indices: {sorted(gt_set)[:5]}{'...' if len(gt_set)>5 else ''}")
    print(f"  GT page_keys: {[page_keys[i] for i in sorted(gt_set)[:3]]}")
    print(f"  Query tokens: {q_emb.shape[0]}, dim: {q_emb.shape[1]}")
    
    # Lấy GT doc embedding
    gt_idx = sorted(gt_set)[0]
    gt_emb_raw = all_page_embeddings[gt_idx]
    gt_emb = torch.from_numpy(gt_emb_raw).float() if isinstance(gt_emb_raw, np.ndarray) else gt_emb_raw.float()
    gt_emb_norm = F.normalize(gt_emb, dim=-1).to(device)
    
    # MaxSim: query vs GT doc
    sim_gt = torch.einsum('qd,ld->ql', q_emb, gt_emb_norm)
    maxsim_gt = sim_gt.max(dim=-1).values
    score_gt = maxsim_gt.sum().item()
    
    print(f"\n  --- GT doc (idx={gt_idx}) ---")
    print(f"  GT emb shape: {gt_emb.shape}, norm[0]={gt_emb[0].norm():.4f}")
    print(f"  GT MaxSim score: {score_gt:.3f} (sum of max-per-token)")
    print(f"  GT per-token MaxSim: mean={maxsim_gt.mean():.4f}, min={maxsim_gt.min():.4f}, max={maxsim_gt.max():.4f}")
    
    # Lấy random non-GT doc embedding
    rand_idx = 0 if 0 not in gt_set else 1
    rand_emb_raw = all_page_embeddings[rand_idx]
    rand_emb = torch.from_numpy(rand_emb_raw).float() if isinstance(rand_emb_raw, np.ndarray) else rand_emb_raw.float()
    rand_emb_norm = F.normalize(rand_emb, dim=-1).to(device)
    
    sim_rand = torch.einsum('qd,ld->ql', q_emb, rand_emb_norm)
    maxsim_rand = sim_rand.max(dim=-1).values
    score_rand = maxsim_rand.sum().item()
    
    print(f"\n  --- Random doc (idx={rand_idx}) ---")
    print(f"  Random emb shape: {rand_emb.shape}")
    print(f"  Random MaxSim score: {score_rand:.3f}")
    print(f"  Random per-token MaxSim: mean={maxsim_rand.mean():.4f}")
    
    # Score difference
    margin = score_gt - score_rand
    print(f"\n  Margin (GT - Random): {margin:.3f}")
    if margin > 1.0:
        print(f"  ✅ GT scores higher — model CAN distinguish")
    elif margin > 0:
        print(f"  ⚠️ GT slightly higher — weak signal")
    else:
        print(f"  ❌ Random scores higher — embeddings NOT aligned")
    
    # Full ranking: query vs all docs (dùng doc_matrix nếu đã build)
    if 'doc_matrix' in dir():
        M = torch.einsum('qd,nld->qnl', q_emb, doc_matrix)
        M.masked_fill_(~doc_mask.unsqueeze(0), float('-inf'))
        M_max = M.max(dim=-1).values  # (Q, N)
        scores = M_max.sum(dim=0)  # (N,)
        
        sorted_idx = torch.argsort(scores, descending=True).cpu().tolist()
        gt_rank = None
        for r, idx in enumerate(sorted_idx):
            if idx in gt_set:
                gt_rank = r + 1
                break
        
        top5_scores = [(sorted_idx[r], scores[sorted_idx[r]].item()) for r in range(5)]
        gt_score_val = scores[gt_idx].item()
        
        print(f"\n  --- FULL RANKING ---")
        print(f"  GT rank: {gt_rank}/{n_docs}")
        print(f"  GT score: {gt_score_val:.3f}")
        print(f"  Top 5 docs:")
        for r, (didx, sc) in enumerate(top5_scores):
            is_gt = "★GT" if didx in gt_set else "   "
            pk = page_keys[didx] if didx < len(page_keys) else "?"
            d_emb = all_page_embeddings[didx]
            d_len = d_emb.shape[0] if hasattr(d_emb, 'shape') else len(d_emb)
            print(f"    [{r+1}] {is_gt} idx={didx:5d}, score={sc:.3f}, len={d_len}, key={pk[:60]}")
        
        # Check: do top docs share a pattern?
        top_lens = [all_page_embeddings[sorted_idx[r]].shape[0] for r in range(20)]
        gt_lens = [all_page_embeddings[g].shape[0] for g in sorted(gt_set)[:5]]
        print(f"\n  Top-20 embedding lengths: mean={np.mean(top_lens):.0f}, max={max(top_lens)}")
        print(f"  GT embedding lengths: {gt_lens}")
        if max(top_lens) > 2000 and all(l < max(top_lens) for l in gt_lens):
            print(f"  ⚠️ LENGTH BIAS: Top docs are LONGER → more tokens → higher MaxSim sum")
    
    del q_inp, q_out
    gc.collect()

print(f"\n{'='*70}")
print(">>> Paste output cho tôi!")
