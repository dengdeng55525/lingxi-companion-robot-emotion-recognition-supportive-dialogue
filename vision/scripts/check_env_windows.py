import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main():
    print("Python:", sys.executable)
    print("Python version:", sys.version.replace("\n", " "))

    if has_module("torch"):
        import torch

        print("torch:", torch.__version__)
        print("torch.cuda.is_available():", torch.cuda.is_available())
    else:
        print("torch: MISSING")

    print("opencv(cv2):", "OK" if has_module("cv2") else "MISSING")
    print("emotiefflib:", "OK" if has_module("emotiefflib") else "MISSING")

    weights = ROOT / "yolov5-face" / "weights" / "yolov5s-face.pt"
    print("YOLO weights:", weights)
    print("YOLO weights exists:", weights.exists())


if __name__ == "__main__":
    main()
