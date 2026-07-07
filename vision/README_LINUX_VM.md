# Linux VM ROS 视频流检测说明

这个入口用于在 Ubuntu 20.04 虚拟机中订阅小车发布的 ROS 图像话题，在虚拟机本地运行 YOLO5Face + EmotiEffLib，并持续输出 `outputs/latest_emotion.json`。

根据你提供的环境信息，当前状态是：

- 虚拟机：Ubuntu 20.04.6 LTS，Python 3.8.10，8 vCPU，7.7 GiB RAM。
- ROS 链路：小车 IP `192.168.79.91`，虚拟机 IP `192.168.79.199`，图像话题 `/camera/rgb/image_raw` 已可通过 `rqt_image_view` 查看。
- 推理依赖：`torch 1.13.1+cu117`、`torchvision 0.14.1+cu117`、`cv2 4.13.0`、`numpy 1.24.4`。
- GPU 状态：`nvidia-smi` 不存在，`torch.cuda.is_available()` 为 `False`。也就是说当前虚拟机还不能使用 NVIDIA GPU。

## GPU 结论

实时检测建议使用 GPU，尤其是希望接近摄像头原始 24 FPS 时。但你当前这台 VMware 虚拟机没有暴露 CUDA 设备，所以项目会自动回退 CPU。

判断 GPU 是否真正可用，只看这两个命令：

```bash
nvidia-smi
python3 - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

如果 `nvidia-smi` 不能运行，`--device cuda` 一定不可用。可以先用 `--device cpu` 或默认 `auto` 做链路验证；需要实时性能时，再考虑 GPU 直通、裸机 Linux、WSL2 GPU 或把推理放回有 GPU 的主机。

## 一次性环境检查

在虚拟机中进入项目目录后运行：

```bash
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://192.168.79.91:11311
export ROS_IP=192.168.79.199
export ROS_HOSTNAME=192.168.79.199

python3 scripts/check_env_linux.py
```

如果 `cv_bridge` 缺失：

```bash
sudo apt update
sudo apt install ros-noetic-cv-bridge
```

Python 依赖建议在不覆盖已有 torch 的前提下安装：

```bash
python3 -m pip install -r requirements_linux.txt
```

## 运行前确认 ROS 图像

小车端先启动 ASTRA：

```bash
ROS_MASTER_URI=http://192.168.79.91:11311 ROS_IP=192.168.79.91 ROS_HOSTNAME=192.168.79.91 roslaunch clbrobot astra_laser.launch
```

虚拟机端确认图像频率：

```bash
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://192.168.79.91:11311
export ROS_IP=192.168.79.199
export ROS_HOSTNAME=192.168.79.199

rostopic hz /camera/rgb/image_raw
```

能看到接近 `24.x` Hz 后再启动检测。

## 方式 A：直接订阅 ROS 图像话题

这是当前推荐方式，少一层视频转码，延迟更低。

```bash
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://192.168.79.91:11311
export ROS_IP=192.168.79.199
export ROS_HOSTNAME=192.168.79.199

python3 vm_ros_emotion_to_json.py \
  --ros-image-topic /camera/rgb/image_raw \
  --device auto \
  --face-weights yolov5-face/weights/yolov5s-face.pt \
  --emotieff-engine torch \
  --crop-margin 0.30 \
  --sample-step 3 \
  --max-infer-fps 8 \
  --json-output outputs/latest_emotion.json
```

如果 CPU 压力太大，可以把 `--sample-step` 调到 `5` 或 `8`，也可以把 `--max-infer-fps` 调低。GPU 真正可用后，把 `--device auto` 保持不变即可自动走 CUDA，或显式使用 `--device cuda`。

## 方式 B：一键启动检测和 API

在项目目录中运行：

```bash
ROBOT_IP=192.168.79.91 VM_IP=192.168.79.199 bash scripts/start_vm_ros_vision_stack.sh
```

启动后会运行：

- `web_video_server`：把 `/camera/rgb/image_raw` 转成浏览器可看的 HTTP 视频预览。如果未安装会跳过并提示。
- `vm_ros_emotion_to_json.py`：订阅 `/camera/rgb/image_raw` 并写 `outputs/latest_emotion.json`。
- `vision_result_api.py`：提供 `http://虚拟机IP:8090/vision/latest`。
- `python3 -m http.server`：提供前端页面 `http://虚拟机IP:9000/vision_dashboard.html`。

查看日志：

```bash
tail -f logs/vm_ros_vision_*/emotion_json.log
tail -f logs/vm_ros_vision_*/vision_api.log
tail -f logs/vm_ros_vision_*/web_video_server.log
tail -f logs/vm_ros_vision_*/frontend_http.log
```

如果要在虚拟机本机浏览器打开，推荐使用：

```text
http://127.0.0.1:9000/vision_dashboard.html
```

前端默认 API 地址是：

```text
http://127.0.0.1:8090
```

默认视频预览地址是：

```text
http://127.0.0.1:8080/stream?topic=/camera/rgb/image_raw
```

## 方式 C：读取 HTTP/RTSP 视频流

如果后续你把 ROS 图像转成 HTTP/RTSP 流，也可以继续使用原来的流入口：

```bash
python3 stream_emotion_to_json.py \
  --source "http://192.168.79.91:8080/stream?topic=/camera/rgb/image_raw" \
  --opencv-backend ffmpeg \
  --ffmpeg-low-latency \
  --device auto \
  --json-output outputs/latest_emotion.json
```

RTSP 示例：

```bash
python3 stream_emotion_to_json.py \
  --source "rtsp://192.168.79.91:8554/live" \
  --opencv-backend ffmpeg \
  --ffmpeg-low-latency \
  --rtsp-transport tcp \
  --device auto \
  --json-output outputs/latest_emotion.json
```
