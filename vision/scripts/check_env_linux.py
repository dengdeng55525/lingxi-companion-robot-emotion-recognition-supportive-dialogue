from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def print_item(name: str, ok: bool, detail: str = "") -> None:
    mark = "OK" if ok else "WARN"
    suffix = f" - {detail}" if detail else ""
    print(f"[{mark}] {name}{suffix}")


def run_command(args: list[str]) -> tuple[bool, str]:
    try:
        proc = subprocess.run(args, check=False, text=True, capture_output=True, timeout=8)
    except Exception as exc:
        return False, str(exc)
    output = (proc.stdout or proc.stderr or "").strip()
    return proc.returncode == 0, output


def import_module(name: str):
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        print_item(name, False, str(exc))
        return None
    version = getattr(module, "__version__", "unknown")
    print_item(name, True, f"version={version}")
    return module


def check_python() -> None:
    print(f"[INFO] platform: {platform.platform()}")
    print(f"[INFO] python: {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info < (3, 8):
        print_item("python>=3.8", False, "Ubuntu 20.04 / ROS Noetic expects Python 3.8")
    else:
        print_item("python>=3.8", True)


def check_gpu(torch_module) -> None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        print_item("nvidia-smi", False, "not found; the VM currently does not expose an NVIDIA GPU")
    else:
        ok, output = run_command([nvidia_smi])
        first_line = output.splitlines()[0] if output else "no output"
        print_item("nvidia-smi", ok, first_line)

    if torch_module is None:
        return

    cuda_available = bool(torch_module.cuda.is_available())
    cuda_version = getattr(torch_module.version, "cuda", None)
    print_item("torch.cuda.is_available()", cuda_available, f"torch cuda={cuda_version}")
    if cuda_available:
        count = torch_module.cuda.device_count()
        names = [torch_module.cuda.get_device_name(i) for i in range(count)]
        print_item("visible CUDA devices", True, ", ".join(names))
    else:
        print("[HINT] Use --device cpu for a VM CPU smoke test. Use --device cuda only after nvidia-smi works in the VM.")


def check_opencv(cv2_module) -> None:
    if cv2_module is None:
        return
    build = cv2_module.getBuildInformation()
    ffmpeg_ok = "FFMPEG:                      YES" in build or "FFMPEG: YES" in build
    gstreamer_ok = "GStreamer:                   YES" in build or "GStreamer: YES" in build
    print_item("OpenCV FFmpeg backend", ffmpeg_ok)
    print_item("OpenCV GStreamer backend", gstreamer_ok)


def check_ros() -> None:
    print(f"[INFO] ROS_MASTER_URI={os.environ.get('ROS_MASTER_URI', '')}")
    print(f"[INFO] ROS_IP={os.environ.get('ROS_IP', '')}")
    print(f"[INFO] ROS_HOSTNAME={os.environ.get('ROS_HOSTNAME', '')}")
    rospy = import_module("rospy")
    cv_bridge = import_module("cv_bridge")
    sensor_msgs = import_module("sensor_msgs.msg")
    if rospy and cv_bridge and sensor_msgs:
        print_item("ROS Python image stack", True)
    else:
        print("[HINT] Run: source /opt/ros/noetic/setup.bash")
        print("[HINT] If cv_bridge is missing: sudo apt install ros-noetic-cv-bridge")


def check_files() -> None:
    weights = PROJECT_ROOT / "yolov5-face" / "weights" / "yolov5s-face.pt"
    print_item("YOLO5Face weights", weights.exists(), str(weights))
    script = PROJECT_ROOT / "vm_ros_emotion_to_json.py"
    print_item("VM ROS entry script", script.exists(), str(script))


def main() -> int:
    check_python()
    numpy_module = import_module("numpy")
    cv2_module = import_module("cv2")
    torch_module = import_module("torch")
    import_module("torchvision")
    import_module("emotiefflib")
    if numpy_module is not None:
        pass
    check_gpu(torch_module)
    check_opencv(cv2_module)
    check_ros()
    check_files()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
