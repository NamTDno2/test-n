# --- BƯỚC 0: SETUP FOR RTX PRO 6000 BLACKWELL (KAGGLE OFFLINE) ---
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

OFFLINE_MODE = True
OFFLINE_DATASET_CANDIDATES = [
    "/kaggle/input/datasets/namthi/offline-kaggle-deps",
    "/kaggle/input/datasets/namthi/offline-kaggle-deps/offline_kaggle_deps",
    "/kaggle/input/offline-kaggle-deps",
    "/kaggle/input/offline-kaggle-deps/offline_kaggle_deps",
]

# torch 2.7.0+cu128 — hỗ trợ SM 12.0 (RTX PRO 6000 Blackwell)
TORCH_TARGET_VERSION = "2.7.0"

def _resolve_offline_dataset_root(candidates):
    for base in candidates:
        probe_dirs = [base, os.path.join(base, "offline_kaggle_deps"), os.path.join(base, "offline-kaggle-deps")]
        for d in probe_dirs:
            if not os.path.isdir(d):
                continue
            if os.path.isdir(os.path.join(d, "wheels")) and os.path.isfile(os.path.join(d, "requirements-runtime.txt")):
                return d
    return None

OFFLINE_DATASET_DIR = _resolve_offline_dataset_root(OFFLINE_DATASET_CANDIDATES)
OFFLINE_WHEEL_DIR = os.path.join(OFFLINE_DATASET_DIR, "wheels") if OFFLINE_DATASET_DIR else ""
OFFLINE_REQ_FILE = os.path.join(OFFLINE_DATASET_DIR, "requirements-runtime.txt") if OFFLINE_DATASET_DIR else ""

def _run(cmd):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)

def _current_py_tag():
    return f"cp{sys.version_info.major}{sys.version_info.minor}"

def _wheel_matches_python_tag(filename):
    low = filename.lower()
    return ("py3-none-any" in low) or ("py3-none-manylinux" in low) or (_current_py_tag() in low)

def _find_wheel(package_name, version_prefix=None, require_python_match=False):
    if not os.path.isdir(OFFLINE_WHEEL_DIR):
        return None
    prefix = f"{package_name.lower()}-"
    cands = []
    for fn in os.listdir(OFFLINE_WHEEL_DIR):
        low = fn.lower()
        if not low.endswith(".whl"):
            continue
        if not low.startswith(prefix):
            continue
        if version_prefix and f"-{version_prefix}" not in low:
            continue
        if require_python_match and (not _wheel_matches_python_tag(fn)):
            continue
        cands.append(fn)
    if not cands:
        return None
    return os.path.join(OFFLINE_WHEEL_DIR, sorted(cands)[-1])

if OFFLINE_MODE and not OFFLINE_DATASET_DIR:
    raise FileNotFoundError(f"Offline dataset not found. Checked: {OFFLINE_DATASET_CANDIDATES}")

print(f">>> Using torch target: {TORCH_TARGET_VERSION}")
print(f">>> Runtime Python tag: {_current_py_tag()}")

torch_was_pinned = False

if OFFLINE_MODE:
    print(">>> OFFLINE MODE: install dependencies from dataset")
    print(f">>> OFFLINE_DATASET_DIR = {OFFLINE_DATASET_DIR}")

    if not os.path.isdir(OFFLINE_WHEEL_DIR):
        raise FileNotFoundError(f"Wheels folder not found: {OFFLINE_WHEEL_DIR}")
    if not os.path.isfile(OFFLINE_REQ_FILE):
        raise FileNotFoundError(f"requirements-runtime.txt not found: {OFFLINE_REQ_FILE}")

    # ── 1. Filter requirements: skip packages that should use Kaggle's preinstalled ──
    filtered_req = "/kaggle/working/requirements-runtime-filtered.txt"
    # QUAN TRỌNG: skip numpy/scipy để tránh "numpy.dtype size changed" error
    skip_prefixes = ("pyarrow", "torch", "torchvision", "torchaudio", "triton", "numpy", "scipy")
    with open(OFFLINE_REQ_FILE, "r", encoding="utf-8") as rf, open(filtered_req, "w", encoding="utf-8") as wf:
        for line in rf:
            s = line.strip().lower()
            if (not s) or s.startswith("#"):
                wf.write(line)
                continue
            normalized = s.replace(" ", "")
            if normalized.startswith(skip_prefixes):
                continue
            wf.write(line)

    # ── 2. Install non-torch deps (--no-deps tránh pip resolver conflict) ──
    print(f">>> Installing non-torch deps (filtered, --no-deps)...")
    _run([
        sys.executable, "-m", "pip", "install",
        "--no-index", "--find-links", OFFLINE_WHEEL_DIR,
        "--no-deps",
        "-r", filtered_req,
    ])

    # ── 3. Install nvidia-cusparselt TRƯỚC torch (provides libcusparseLt.so.0) ──
    cusparselt_whls = [
        f for f in os.listdir(OFFLINE_WHEEL_DIR)
        if f.endswith(".whl") and "cusparselt" in f.lower()
    ]
    for whl in cusparselt_whls:
        whl_path = os.path.join(OFFLINE_WHEEL_DIR, whl)
        print(f">>> Installing NVIDIA lib: {whl}")
        _run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", whl_path])

    # ── 4. Install torch cu128 (SM 12.0 support) ──
    torch_whl = _find_wheel("torch", TORCH_TARGET_VERSION, require_python_match=True)
    if not torch_whl:
        available = [f for f in os.listdir(OFFLINE_WHEEL_DIR) if f.lower().startswith("torch-") and f.endswith(".whl")]
        print(f"WARN: No torch {TORCH_TARGET_VERSION} wheel found. Available: {available}")
        print("WARN: Keep using runtime torch preinstalled by Kaggle.")
    else:
        print(f">>> Installing torch: {os.path.basename(torch_whl)}")
        _run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", torch_whl])
        torch_was_pinned = True

    # ── 5. Install torchvision/torchaudio cu128 ──
    tv_whl = _find_wheel("torchvision", require_python_match=True)
    ta_whl = _find_wheel("torchaudio", require_python_match=True)
    if tv_whl:
        print(f">>> Installing torchvision: {os.path.basename(tv_whl)}")
        _run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", tv_whl])
    if ta_whl:
        print(f">>> Installing torchaudio: {os.path.basename(ta_whl)}")
        _run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", ta_whl])

    # ── 6. Install colpali-engine ──
    colpali_whls = [f for f in os.listdir(OFFLINE_WHEEL_DIR) if f.endswith(".whl") and "colpali" in f.lower()]
    if colpali_whls:
        colpali_path = os.path.join(OFFLINE_WHEEL_DIR, sorted(colpali_whls)[0])
        print(f">>> Installing colpali-engine (--no-deps): {os.path.basename(colpali_path)}")
        _run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", OFFLINE_WHEEL_DIR, "--no-deps", colpali_path])
    else:
        raise FileNotFoundError("No colpali-engine wheel found.")

else:
    print(">>> ONLINE MODE")
    _run([sys.executable, "-m", "pip", "uninstall", "-y", "tensorflow", "pyarrow"])
    _run([sys.executable, "-m", "pip", "install", "pyarrow<20.0.0"])
    _run([sys.executable, "-m", "pip", "install", "-U", "git+https://github.com/illuin-tech/colpali"])
    _run([sys.executable, "-m", "pip", "install", "-U", f"torch=={TORCH_TARGET_VERSION}", "transformers", "accelerate", "peft", "bitsandbytes"])
    _run([sys.executable, "-m", "pip", "install", "-U", "torchvision", "torchaudio"])

# ── Preload NVIDIA libs (torch cu128 cần libcusparseLt.so.0) ──
# nvidia pip packages đặt .so files trong path riêng, torch không tự tìm được.
# Phải dùng ctypes preload trước khi import torch.
import ctypes
import glob

nvidia_lib_dirs = glob.glob("/usr/local/lib/python*/dist-packages/nvidia/*/lib")
if nvidia_lib_dirs:
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(nvidia_lib_dirs) + (":" + existing_ld if existing_ld else "")
    print(f">>> Added {len(nvidia_lib_dirs)} NVIDIA lib dirs to LD_LIBRARY_PATH")

# Preload specific libraries that torch needs
for pattern in [
    "/usr/local/lib/python*/dist-packages/nvidia/cusparselt/lib/libcusparseLt.so*",
    "/usr/local/lib/python*/dist-packages/nvidia/*/lib/lib*.so*",
]:
    for lib_path in sorted(glob.glob(pattern)):
        try:
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass  # Không phải tất cả .so đều cần preload

cusparselt_found = glob.glob("/usr/local/lib/python*/dist-packages/nvidia/cusparselt/lib/libcusparseLt.so*")
if cusparselt_found:
    print(f">>> Preloaded libcusparseLt: {cusparselt_found[0]}")
else:
    print("WARN: libcusparseLt.so not found in nvidia packages!")

import torch

print(f">>> torch version: {torch.__version__}")
if torch_was_pinned and (not torch.__version__.startswith(TORCH_TARGET_VERSION)):
    raise RuntimeError(
        f"Expected torch {TORCH_TARGET_VERSION}, but got {torch.__version__}. "
        "Please verify offline wheel pack and restart kernel."
    )
if not torch_was_pinned:
    print(">>> Using runtime torch (wheel pin skipped).")

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name
    vram_gb = props.total_memory / (1024 ** 3)
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    print(f">>> GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB | CC={cc_major}.{cc_minor}")
else:
    raise RuntimeError("CUDA is not available.")

PERF_CFG = {
    "GPU_NAME": gpu_name,
    "VRAM_GB": round(vram_gb, 1),
    "BATCH_SIZE_ENCODE": 12,
    "BATCH_Q_EVAL": 12,
    "DOC_CHUNK_SIZE": 1024,
    "SAVE_EVERY": 900,
}
print(
    ">>> RTX6000 config: "
    f"BATCH_SIZE_ENCODE={PERF_CFG['BATCH_SIZE_ENCODE']}, "
    f"BATCH_Q_EVAL={PERF_CFG['BATCH_Q_EVAL']}, "
    f"DOC_CHUNK_SIZE={PERF_CFG['DOC_CHUNK_SIZE']}"
)

print(">>> Setup dependencies done")
