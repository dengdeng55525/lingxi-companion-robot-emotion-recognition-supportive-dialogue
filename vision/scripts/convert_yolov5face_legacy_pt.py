from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_FACE_DIR = PROJECT_ROOT / "yolov5-face"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-save a YOLO5Face .pt checkpoint with the legacy PyTorch serialization format."
    )
    parser.add_argument("--input", required=True, help="Input .pt checkpoint saved by newer PyTorch.")
    parser.add_argument("--output", required=True, help="Output .pt checkpoint for older PyTorch loaders.")
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(YOLO_FACE_DIR))

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import torch

    checkpoint = torch.load(str(input_path), map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        try:
            checkpoint["model"] = checkpoint["model"].float().eval()
        except AttributeError:
            pass

    torch.save(checkpoint, str(output_path), _use_new_zipfile_serialization=False)
    print(f"[INFO] legacy checkpoint written: {output_path}")


if __name__ == "__main__":
    main()
