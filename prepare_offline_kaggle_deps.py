"""
prepare_offline_kaggle_deps.py
================================
Build offline wheel pack cho Kaggle:
  - GPU  : RTX PRO 6000 Blackwell Server Edition (SM 12.0)
  - CUDA : torch cu128 (Kaggle chỉ có CUDA 12.6 libs, nhưng cu128 wheel
            bundles riêng, cần thêm nvidia-cusparselt-cu12)
  - Torch: 2.7.0+cu128
  - Python: 3.12

Chạy script trên máy local có internet:
    python prepare_offline_kaggle_deps.py

Upload lên Kaggle:
    kaggle datasets version -p offline_kaggle_deps -m "torch270 cu128 cusparselt"
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ─── OUTPUT ──────────────────────────────────────────────────────────────────
OUT_DIR     = Path("offline_kaggle_deps")
WHEELS_DIR  = OUT_DIR / "wheels"
REQ_FILE    = OUT_DIR / "requirements-runtime.txt"
INSTALL_SH  = OUT_DIR / "install_offline.sh"
README_FILE = OUT_DIR / "README_offline.txt"
META_FILE   = OUT_DIR / "dataset-metadata.json"

# ─── TARGET PLATFORM ─────────────────────────────────────────────────────────
TARGET_PYTHON_VERSION = "312"
TARGET_ABI            = "cp312"
TORCH_PLATFORM        = "manylinux_2_28_x86_64"
NON_TORCH_PLATFORMS   = [
    "manylinux_2_17_x86_64",
    "manylinux2014_x86_64",
    "manylinux_2_28_x86_64",
]

# ─── INDEX URLS ───────────────────────────────────────────────────────────────
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
PYPI_INDEX_URL  = "https://pypi.org/simple"

# ─── TORCH STACK ──────────────────────────────────────────────────────────────
TORCH_STACK = {
    "torch":       "2.7.0",
    "torchvision": "0.22.0",
    "torchaudio":  "2.7.0",
}

# ─── NVIDIA CUDA LIBRARIES ───────────────────────────────────────────────────
# torch cu128 cần libcusparseLt.so.0 mà Kaggle (cu126 runtime) không có.
# Package nvidia-cusparselt-cu12 cung cấp library này.
NVIDIA_PACKAGES = [
    "nvidia-cusparselt-cu12",
]

# ─── NON-TORCH PACKAGES ──────────────────────────────────────────────────────
NON_TORCH_PACKAGES = [
    "colpali-engine==0.3.14",
    "transformers==4.51.3",
    "accelerate==1.6.0",
    "peft==0.18.1",
    "huggingface_hub==0.30.2",
    "safetensors==0.4.5",
    "tokenizers==0.21.1",
    "einops==0.8.1",
    "pillow==11.2.1",
    "pandas==2.2.3",
    "pyarrow==19.0.1",
    "numpy==1.26.4",
    "scipy==1.13.1",
    "tqdm==4.67.1",
    "regex==2024.11.6",
    "typing_extensions==4.12.2",
    "packaging==24.2",
    "filelock==3.18.0",
    "pyyaml==6.0.2",
    "psutil==7.0.0",
    "requests==2.32.3",
    "urllib3==2.4.0",
    "certifi==2025.1.31",
    "charset-normalizer==3.4.1",
    "idna==3.10",
]


def run(cmd: List[str]):
    print("$", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def pip_download_single(
    package: str,
    dest: Path,
    index_url: str,
    platform_tag: str,
    extra_index: Optional[str] = None,
    no_deps: bool = True,
) -> bool:
    cmd = [
        sys.executable, "-m", "pip", "download",
        package,
        "-d", str(dest),
        "--only-binary=:all:",
        "--index-url", index_url,
        "--platform",  platform_tag,
        "--implementation", "cp",
        "--python-version", TARGET_PYTHON_VERSION,
        "--abi", TARGET_ABI,
    ]
    if no_deps:
        cmd.append("--no-deps")
    if extra_index:
        cmd += ["--extra-index-url", extra_index]
    try:
        run(cmd)
        return True
    except subprocess.CalledProcessError:
        return False


def download_torch_stack() -> bool:
    print(f"\n{'='*70}")
    print(f"  STEP 1: Torch stack (cu128, {TORCH_PLATFORM})")
    print(f"{'='*70}")
    pkgs = [
        f"torch=={TORCH_STACK['torch']}",
        f"torchvision=={TORCH_STACK['torchvision']}",
        f"torchaudio=={TORCH_STACK['torchaudio']}",
    ]
    cmd = [
        sys.executable, "-m", "pip", "download",
        *pkgs,
        "-d", str(WHEELS_DIR),
        "--only-binary=:all:",
        "--index-url", TORCH_INDEX_URL,
        "--platform", TORCH_PLATFORM,
        "--implementation", "cp",
        "--python-version", TARGET_PYTHON_VERSION,
        "--abi", TARGET_ABI,
        "--extra-index-url", PYPI_INDEX_URL,
    ]
    try:
        run(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Torch download FAILED: {e}")
        return False


def download_nvidia_packages() -> bool:
    print(f"\n{'='*70}")
    print(f"  STEP 2: NVIDIA CUDA libraries ({len(NVIDIA_PACKAGES)} packages)")
    print(f"  (Required: libcusparseLt.so.0 for torch cu128)")
    print(f"{'='*70}")

    failed = []
    for pkg in NVIDIA_PACKAGES:
        success = False
        for platform in NON_TORCH_PLATFORMS:
            ok = pip_download_single(
                pkg, WHEELS_DIR,
                index_url=PYPI_INDEX_URL,
                platform_tag=platform,
                no_deps=True,
            )
            if ok:
                print(f"  ✓ {pkg} [{platform}]")
                success = True
                break
        if not success:
            print(f"  ✗ FAILED: {pkg}")
            failed.append(pkg)

    if failed:
        print(f"  ✗ {len(failed)} NVIDIA packages failed: {failed}")
        return False
    return True


def download_non_torch_packages() -> bool:
    print(f"\n{'='*70}")
    print(f"  STEP 3: Non-torch packages ({len(NON_TORCH_PACKAGES)} packages)")
    print(f"{'='*70}")

    REQ_FILE.write_text("\n".join(NON_TORCH_PACKAGES) + "\n", encoding="utf-8")

    failed = []
    for pkg in NON_TORCH_PACKAGES:
        success = False
        for platform in NON_TORCH_PLATFORMS:
            ok = pip_download_single(
                pkg, WHEELS_DIR,
                index_url=PYPI_INDEX_URL,
                platform_tag=platform,
                no_deps=True,
            )
            if ok:
                print(f"  ✓ {pkg} [{platform}]")
                success = True
                break
        if not success:
            ok = pip_download_single(
                pkg, WHEELS_DIR,
                index_url=PYPI_INDEX_URL,
                platform_tag="any",
                no_deps=True,
            )
            if ok:
                print(f"  ✓ {pkg} [pure-python]")
                success = True
        if not success:
            print(f"  ✗ FAILED: {pkg}")
            failed.append(pkg)

    if failed:
        print(f"\n  ✗ {len(failed)} packages failed: {failed}")
        return False
    return True


def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    WHEELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Offline Kaggle Dependency Builder")
    print("  GPU  : RTX PRO 6000 Blackwell Server Edition (SM 12.0)")
    print("  CUDA : torch cu128 + nvidia-cusparselt-cu12")
    print(f"  Torch: {TORCH_STACK['torch']}+cu128")
    print("=" * 70)

    # Step 1: Torch
    if not download_torch_stack():
        raise RuntimeError("Torch download failed")

    # Step 2: NVIDIA libs (cusparselt)
    if not download_nvidia_packages():
        raise RuntimeError("NVIDIA packages download failed")

    # Step 3: Non-torch deps
    if not download_non_torch_packages():
        raise RuntimeError("Non-torch packages download failed")

    # ── Metadata files ───────────────────────────────────────────────────
    def _ver(prefix):
        for r in NON_TORCH_PACKAGES:
            if r.lower().startswith(prefix.lower()):
                return r.split("==", 1)[-1]
        return "?"

    INSTALL_SH.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "WHEEL_DIR=${1:-/kaggle/input/offline-kaggle-deps-colsmol/wheels}\n"
        "# Install nvidia-cusparselt FIRST (needed by torch cu128)\n"
        "pip install --no-index --find-links \"$WHEEL_DIR\" --no-deps nvidia-cusparselt-cu12\n"
        "# Install torch stack\n"
        "pip install --no-index --find-links \"$WHEEL_DIR\" --no-deps torch torchvision torchaudio\n"
        "# Install remaining deps\n"
        "pip install --no-index --find-links \"$WHEEL_DIR\" --no-deps -r \"$WHEEL_DIR/../requirements-runtime.txt\"\n",
        encoding="utf-8",
    )

    whl_files  = sorted(WHEELS_DIR.glob("*.whl"))
    total_size = sum(w.stat().st_size for w in whl_files)
    size_gb    = total_size / (1024**3)

    README_FILE.write_text(
        "Offline Dependency Pack — RTX PRO 6000 Blackwell (SM 12.0)\n"
        "=" * 60 + "\n\n"
        "KEY POINT: torch cu128 + nvidia-cusparselt-cu12\n"
        "  Kaggle runtime has CUDA 12.6 libs. torch cu126 does NOT support SM 12.0.\n"
        "  torch cu128 supports SM 12.0 but needs libcusparseLt.so.0.\n"
        "  nvidia-cusparselt-cu12 provides this library.\n\n"
        f"  torch          == {TORCH_STACK['torch']}+cu128\n"
        f"  torchvision    == {TORCH_STACK['torchvision']}+cu128\n"
        f"  torchaudio     == {TORCH_STACK['torchaudio']}+cu128\n"
        f"  colpali-engine == {_ver('colpali-engine')}\n"
        f"  transformers   == {_ver('transformers')}\n"
        f"  peft           == {_ver('peft')}\n\n"
        "INSTALL ORDER (critical):\n"
        "  1. nvidia-cusparselt-cu12  (provides libcusparseLt.so.0)\n"
        "  2. torch + torchvision + torchaudio\n"
        "  3. remaining deps\n\n"
        f"Total wheels: {len(whl_files)} ({size_gb:.2f} GB)\n",
        encoding="utf-8",
    )

    META_FILE.write_text(
        json.dumps({
            "title": "offline-kaggle-deps-colsmol",
            "id":    "namthi/offline-kaggle-deps-colsmol",
            "licenses": [{"name": "CC0-1.0"}],
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("  ✅ BUILD THÀNH CÔNG!")
    print("=" * 70)
    print(f"  torch        : {TORCH_STACK['torch']}+cu128")
    print(f"  nvidia-cusparselt-cu12: included ✓")
    print(f"  Total wheels : {len(whl_files)} ({size_gb:.2f} GB)")
    print()
    print("  Bước tiếp theo:")
    print("  1) kaggle datasets version -p offline_kaggle_deps -m \"cu128 cusparselt\"")
    print("  2) Notebook BƯỚC 0: TORCH_TARGET_VERSION = '2.7.0'")
    print("     + Thêm install nvidia-cusparselt TRƯỚC torch (xem README)")
    print("  3) New session → chạy từ BƯỚC 0")
    print("=" * 70)


if __name__ == "__main__":
    main()
