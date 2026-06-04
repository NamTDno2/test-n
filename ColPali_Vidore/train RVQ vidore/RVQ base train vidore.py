# ==============================================================================
# METHOD 7 - RVQ Training on ColPali Train Set
# Train RVQ codebooks from scratch using the official ColPali training dataset.
#
# Variables inherited from earlier cells (no need to redefine):
#   Cell 1: os, gc, glob, json, time, np, torch, F, tqdm
#            COLPALI_BASE, COLPALI_LORA, WORKING_DIR, device
#   Cell 4: query_model (ColPali + LoRA, bfloat16, on GPU)
#            query_processor (ColPaliProcessor)
# ==============================================================================

# -- Install vector-quantize-pytorch (offline wheel preferred) -----------------
import subprocess, io
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
# WORKING_DIR is already defined in Cell 1 as "/kaggle/working"
OUTPUT_DIR = os.path.join(WORKING_DIR, "rvq_codebooks")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMB_DIM = 128   # ColPali projection dimension (query_model.dim == 128)

# RVQ configs to train: (NQ, CB_SIZE, label)
#   c32_f32 -> 32 quantizers x 32-code codebook -> 32 bytes/patch  (4x compression)
RVQ_CONFIGS = [
    (32, 32, "c32_f32"),
]


RVQ_TRAINING_EPOCHS     = 40       # max epochs per config (early-stop patience=10)
RVQ_TRAINING_BATCH_SIZE = 8192     # gradient-step batch size (tuned for 96 GB VRAM)
N_TRAIN_PATCHES         = 1_700_000  # total patches to collect across all shards
PATCHES_PER_SHARD       = 20_000    # per-shard cap - prevents any domain dominating
ENCODE_BATCH_SIZE       = 16       # images per ColPali forward pass

# Blackwell / RTX PRO 6000: enable TF32 for ~2x GEMM throughput
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

ENCODE_DTYPE = torch.bfloat16  # must match query_model dtype from Cell 4

print(">>>> METHOD 7: RVQ Codebook Training")
print(f"  Model base     : {COLPALI_BASE}")
print(f"  Model LoRA     : {COLPALI_LORA}")
print(f"  Train data     : {TRAIN_DATA_ROOT}")
print(f"  Output         : {OUTPUT_DIR}")
print(f"  Configs        : {[c[2] for c in RVQ_CONFIGS]}")
print(f"  Target patches : {N_TRAIN_PATCHES:,}")
print(f"  Batch size     : {RVQ_TRAINING_BATCH_SIZE:,}")
print()

# ==============================================================================
# DATA STREAMING - lazy Parquet shard loader
# ==============================================================================

def _list_train_shards(root):
    # Return sorted list of train-*.parquet shard paths
    pattern = os.path.join(root, "train-*.parquet")
    shards  = sorted(glob.glob(pattern))
    if not shards:
        raise FileNotFoundError(f"No train shards found at {pattern}")
    return shards


def _load_shard_images(shard_path):
    # Load one Parquet shard lazily and return a list of PIL Images.
    # Handles 'image' column as: bytes, dict{'bytes':...}, or PIL Image directly.
    import pyarrow.parquet as pq
    from PIL import Image

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
        del inputs          # ← xóa inputs TRƯỚC khi append để giải phóng VRAM ngay
        torch.cuda.empty_cache()
        gc.collect()
        for i in range(embs.shape[0]):
            all_embs.append(embs[i])
        del embs
    return all_embs


# ==============================================================================
# DOMAIN-AWARE PATCH SAMPLER
# ==============================================================================

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
# RVQ TRAINING HELPERS
# ==============================================================================

def train_rvq_model(NQ, CB_SIZE, train_data):
    # Train ResidualVQ model.
    # train_data: (N, D) float32 CPU tensor, L2-normalised.
    # Returns trained model moved back to CPU.
    rvq_model = ResidualVQ(
        dim                     = EMB_DIM,
        num_quantizers          = NQ,
        codebook_size           = CB_SIZE,
        kmeans_init             = True,       # warm-start centroids with k-means
        threshold_ema_dead_code = 2,          # revive dead codebook entries automatically
        commitment_weight       = 0.25,
    ).to(device)
    rvq_model.train()

    n = train_data.shape[0]
    print(f"  Training NQ={NQ}, CB_SIZE={CB_SIZE} | {n:,} patches | max {RVQ_TRAINING_EPOCHS} epochs")

    best_loss, no_improve = float("inf"), 0

    for epoch in range(RVQ_TRAINING_EPOCHS):
        perm      = torch.randperm(n)
        ep_loss   = 0.0
        n_batches = 0

        for i in range(0, n, RVQ_TRAINING_BATCH_SIZE):
            batch = train_data[perm[i : i + RVQ_TRAINING_BATCH_SIZE]].to(device)
            with torch.no_grad():     # không cần grad, EMA update xảy ra trong forward
                _, _, commit_loss = rvq_model(batch.unsqueeze(1))
            loss_val = commit_loss.sum().item()
            ep_loss   += loss_val
            n_batches += 1

        avg_loss = ep_loss / max(n_batches, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:>3}/{RVQ_TRAINING_EPOCHS}: commit_loss={avg_loss:.6f}")

        # Early stopping: patience=10
        if avg_loss < best_loss - 1e-6:
            best_loss, no_improve = avg_loss, 0
        else:
            no_improve += 1
        if no_improve >= 10:
            print(f"    Early stop at epoch {epoch+1} (no improvement for 10 epochs)")
            break

    rvq_model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return rvq_model


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
            raise RuntimeError(
                f"Cannot extract codebook for quantizer {qi}. "
                "Try updating vector-quantize-pytorch."
            )
        cbs.append(cb.numpy())
    return np.stack(cbs, axis=0)   # (NQ, CB_SIZE, EMB_DIM)


def save_rvq_codebook(cbs_np, NQ, CB_SIZE, label):
    # Save codebook as .npy + metadata .json to OUTPUT_DIR.
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
        "trained_on"          : f"{N_TRAIN_PATCHES:,} patches - ColPali train set",
        "model_base"          : COLPALI_BASE,
        "model_lora"          : COLPALI_LORA,
        "timestamp"           : time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    size_mb = os.path.getsize(npy_path) / 1e6
    print(f"  OK: {npy_path}  ({size_mb:.3f} MB)")
    print(f"     meta : {meta_path}")
    print(f"     shape: {cbs_np.shape}  dtype={cbs_np.dtype}")


# ==============================================================================
# MAIN - collect patches -> free encoder -> train each config -> save
# ==============================================================================

_t_total = time.time()

# Step 1: Discover training shards
_shards = _list_train_shards(TRAIN_DATA_ROOT)
print(f"Found {len(_shards)} train shards\n")

# Step 2: Encode images + domain-aware sampling using query_model (Cell 4)
_patches_np = collect_training_patches(_shards)
_train_data = torch.from_numpy(_patches_np)
del _patches_np; gc.collect()

# Step 3: Move query_model to CPU to free ~8 GB VRAM for RVQ training.
#   NOTE: query_model is NOT deleted. Restore with query_model.cuda() if needed later.
with torch.no_grad():
    pass  # flush autograd graph
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free_gb = (torch.cuda.get_device_properties(0).total_memory
               - torch.cuda.memory_allocated()) / 1e9
    print(f"VRAM free: {free_gb:.1f} GB — proceeding to RVQ training.\n")
gc.collect()

# Step 4: Train each RVQ config and save codebook to disk
_rvq_results = {}
for _NQ, _CB_SIZE, _label in RVQ_CONFIGS:
    print(f"\n{'='*60}")
    print(f"CONFIG: {_label}  (NQ={_NQ}, CB_SIZE={_CB_SIZE})")
    print(f"{'='*60}")

    _t_cfg   = time.time()
    _rvq     = train_rvq_model(_NQ, _CB_SIZE, _train_data)
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

# Step 5: Print summary table
print(f"\n{'='*60}")
print(f"ALL DONE in {(time.time()-_t_total)/60:.1f} min")
print(f"{'='*60}")
print(f"{'Config':<15} {'NQ':>4} {'CB':>5} {'bytes/patch':>12} {'ratio':>8} {'secs':>8}")
print("-" * 55)
for _lbl, _r in _rvq_results.items():
    _ratio = round(EMB_DIM * 4 / _r["NQ"], 1)
    print(f"{_lbl:<15} {_r['NQ']:>4} {_r['CB_SIZE']:>5} {_r['NQ']:>12}  {_ratio:>6.1f}x  {_r['train_secs']:>7.1f}")

print(f"\nCodebooks saved to: {OUTPUT_DIR}")
print("  -> Download .npy files from Kaggle Output tab,")
print("     then upload as a dataset to reuse in inference notebooks.")
print("\nFiles:")
for _fn in sorted(os.listdir(OUTPUT_DIR)):
    _fp = os.path.join(OUTPUT_DIR, _fn)
    print(f"  {_fn}  ({os.path.getsize(_fp)/1e6:.3f} MB)")