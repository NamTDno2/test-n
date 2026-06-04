import pandas as pd
import ast

def calc_recall_at_k(retrieved_str, gt_str, k):
    """Tính recall@K cho một dòng cụ thể."""
    try:
        # Chuyển chuỗi thành list
        retrieved = ast.literal_eval(retrieved_str)
        if not isinstance(retrieved, list):
            retrieved = [retrieved]
    except:
        retrieved = []
        
    try:
        # Chuyển chuỗi thành list
        gt = ast.literal_eval(gt_str)
        if not isinstance(gt, list):
            gt = [gt]
    except:
        gt = []
        
    if not gt:
        return 0.0
        
    # Lấy top K phần tử được truy xuất
    retrieved_k = retrieved[:k]
    # Số lượng phần tử đúng (giao của tập retrieved_k và tập ground truth)
    hits = set(gt).intersection(set(retrieved_k))
    
    # Tính recall
    return len(hits) / len(set(gt))

# 1. Đọc dữ liệu
df = pd.read_csv('ColSmol original.csv')

# 2. Tính recall cho từng mức K = 1, 5, 10 trên từng dòng
for k in [1, 5, 10]:
    df[f'recall@{k}'] = df.apply(lambda row: calc_recall_at_k(row['layout_retrieved'], row['gt'], k), axis=1)

# 3. Tính Micro Recall (trung bình toàn bộ tập dữ liệu)
print("=== Micro Recall ===")
micro_recall = {k: df[f'recall@{k}'].mean() for k in [1, 5, 10]}
for k, val in micro_recall.items():
    print(f"Recall@{k}: {val:.4f}")

# 4. Tính Macro Recall (trung bình của Recall theo từng domain)
print("\n=== Macro Recall (theo domain) ===")
macro_recall = {}
for k in [1, 5, 10]:
    # Lấy trung bình recall@K cho từng domain
    domain_recalls = df.groupby('domain')[f'recall@{k}'].mean()
    # Tính trung bình của các domain
    macro_recall[k] = domain_recalls.mean()

for k, val in macro_recall.items():
    print(f"Recall@{k}: {val:.4f}")

# (Tùy chọn) In chi tiết recall@10 của từng domain để tham khảo
print("\n=== Chi tiết Recall@10 theo Domain ===")
print(df.groupby('domain')['recall@10'].mean())