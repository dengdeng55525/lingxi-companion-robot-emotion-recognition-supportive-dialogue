from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_FACE_DIR = PROJECT_ROOT / "yolov5-face"


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO5Face .pt weights to a static-shape ONNX model.")
    parser.add_argument("--weights", required=True, help="YOLO5Face .pt weights path.")
    parser.add_argument("--output", required=True, help="Output ONNX path.")
    parser.add_argument("--img-size", type=int, default=640, help="Static square input size.")
    parser.add_argument("--opset", type=int, default=10, help="ONNX opset. JetPack 4.2 should start with 10.")
    parser.add_argument("--simplify", action="store_true", help="Run onnxsim after export if it is installed.")
    return parser.parse_args()


def main():
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(YOLO_FACE_DIR))

    import torch
    from models.experimental import attempt_load

    model = attempt_load(str(weights_path), map_location="cpu")
    model.float().eval()

    # YOLO5Face has an ONNX-friendly Detect.export_cat path that returns one tensor.
    for module in model.modules():
        if hasattr(module, "export_cat"):
            module.export_cat = True

    dummy = torch.zeros(1, 3, args.img_size, args.img_size)
    with torch.no_grad():
        model(dummy)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        verbose=False,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["output"],
    )
    print(f"[INFO] ONNX model written: {output_path}")

    if args.simplify:
        try:
            import onnx
            from onnxsim import simplify
        except ImportError:
            print("[WARN] onnx or onnxsim is not installed; skip simplification.")
            return

        onnx_model = onnx.load(str(output_path))
        simplified_model, ok = simplify(onnx_model)
        if not ok:
            raise RuntimeError("onnxsim reported that the simplified model is invalid.")
        onnx.save(simplified_model, str(output_path))
        print(f"[INFO] simplified ONNX model written: {output_path}")


if __name__ == "__main__":
    main()
