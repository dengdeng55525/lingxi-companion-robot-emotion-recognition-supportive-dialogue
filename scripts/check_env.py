#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lingxi.io_utils import load_yaml, resolve_path


REQUIRED_PACKAGES = [
    "transformers",
    "datasets",
    "peft",
    "accelerate",
    "bitsandbytes",
    "huggingface_hub",
    "modelscope",
    "pandas",
    "yaml",
    "sklearn",
    "tqdm",
]


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen25_1p5b_qlora.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    missing = [pkg for pkg in REQUIRED_PACKAGES if not package_available(pkg)]

    print(f"Python: {sys.version.split()[0]}")
    print(f"Torch: {torch.__version__}")
    print(f"Torch CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
    print(f"nvidia-smi: {shutil.which('nvidia-smi') or 'not found'}")

    model_path = resolve_path(config["model"]["base_model"])
    train_file = resolve_path(config["data"]["train_file"])
    valid_file = resolve_path(config["data"]["valid_file"])
    print(f"Model path: {model_path} exists={model_path.exists()}")
    print(f"Train file: {train_file} exists={train_file.exists()}")
    print(f"Valid file: {valid_file} exists={valid_file.exists()}")

    if missing:
        print("Missing packages: " + ", ".join(missing))
    else:
        print("All required Python packages are importable.")

    if missing:
        return 2
    if not torch.cuda.is_available():
        print("CUDA is not available. Formal QLoRA training is blocked.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
