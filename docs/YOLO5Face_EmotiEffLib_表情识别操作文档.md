# YOLO5Face + EmotiEffLib 表情识别操作文档

## 1. 项目简介

本项目用于智能陪护机器人场景中的人脸表情识别。主流程使用 YOLO5Face 检测人脸，再把扩展后的人脸 crop 区域送入 EmotiEffLib 的 `enet_b0_8_best_vgaf` 模型进行表情分类。

最终输出 7 类表情：

```text
neutral, happy, sad, surprise, fear, disgust, angry
```

如果 EmotiEffLib 模型输出 8 类，其中 `contempt` 会默认合并到 `neutral`。

## 2. 两个版本区别

### Windows 摄像头验证版

目录：`yolov5_emotieff_vgaf_windows/`

用途：在 Windows 本机上打开电脑摄像头，实时显示人脸框、crop 框、表情标签、置信度和 FPS，适合调试、演示和参数验证。

### Jetson 部署版

目录：`yolov5_emotieff_vgaf_jetson/`

用途：部署到 Jetson 小车或机器人端。既可以运行本地摄像头检测，也可以启动 FastAPI HTTP 服务，给 GPT API、语音交互、ROS 控制模块或其他业务模块调用。

## 3. Windows 版使用方法

### 解压

解压 `yolov5_emotieff_vgaf_windows.zip`，进入目录：

```powershell
cd yolov5_emotieff_vgaf_windows
```

### 创建或激活环境

建议使用已有环境：

```powershell
conda activate faceemo310
```

### 安装依赖

```powershell
pip install -r requirements_windows.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

依赖文件没有固定 torch / torchvision / CUDA 版本，避免覆盖已有环境。

### 设置代理，若需要

EmotiEffLib 的 `enet_b0_8_best_vgaf` 可能首次运行时自动下载。如果 GitHub 下载慢，可以设置代理：

```powershell
$env:HTTP_PROXY="http://127.0.0.1:7892"
$env:HTTPS_PROXY="http://127.0.0.1:7892"
```

### 检查环境

```powershell
python .\scripts\check_env_windows.py
```

### 运行摄像头实时检测

```powershell
python .\windows_camera_realtime.py --camera-index 0 --face-weights .\yolov5-face\weights\yolov5s-face.pt --emotion-backend emotieff --emotieff-model-name enet_b0_8_best_vgaf --emotieff-engine torch --crop-margin 0.30 --sample-step 3
```

按 `q` 退出窗口。

### 保存检测视频

```powershell
python .\windows_camera_realtime.py --camera-index 0 --save-output outputs\windows_camera_test.mp4 --face-weights .\yolov5-face\weights\yolov5s-face.pt --emotion-backend emotieff --emotieff-model-name enet_b0_8_best_vgaf --emotieff-engine torch --crop-margin 0.30 --sample-step 3
```

### 调整 crop_margin

`--crop-margin` 控制在人脸框外扩多少区域再送入表情分类器。

- 表情不稳定、嘴角或额头被裁掉：可以调大，例如 `0.35`、`0.40`。
- 背景太多、多人靠近时误判：可以调小，例如 `0.20`、`0.25`。

### CPU 模式运行说明

Windows 版不会强制使用 CUDA。独显关闭或没有 CUDA 时会自动使用 CPU，只是速度可能较慢。也可以显式指定：

```powershell
python .\windows_camera_realtime.py --device cpu
```

## 4. Jetson 版使用方法

### 解压

```bash
unzip yolov5_emotieff_vgaf_jetson.zip
cd yolov5_emotieff_vgaf_jetson
```

### PyTorch / JetPack 注意事项

Jetson 上的 PyTorch / torchvision 通常要根据 JetPack 版本安装 NVIDIA 提供的 wheel。不建议直接 `pip install torch torchvision` 覆盖系统环境，否则可能导致 CUDA 不可用。

### 安装依赖

```bash
pip3 install -r requirements_jetson.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果 OpenCV 与 Jetson 系统环境冲突，可以使用系统自带 OpenCV 或按 JetPack 版本安装。

### 检查环境

```bash
python3 scripts/check_env_jetson.py
```

### 运行摄像头检测

默认不弹窗，只输出命令行日志：

```bash
python3 jetson_camera_realtime.py --source 0 --face-weights yolov5-face/weights/yolov5s-face.pt --emotieff-model-name enet_b0_8_best_vgaf --emotieff-engine torch --crop-margin 0.30 --sample-step 5
```

如果 Jetson 接了显示器，可以加 `--show`：

```bash
python3 jetson_camera_realtime.py --source 0 --show
```

### 启动 HTTP API 服务

```bash
python3 emotion_api_server.py --host 0.0.0.0 --port 8000 --source 0 --face-weights yolov5-face/weights/yolov5s-face.pt --emotieff-model-name enet_b0_8_best_vgaf --emotieff-engine torch --crop-margin 0.30 --sample-step 5
```

### 调用 /health

```bash
curl http://127.0.0.1:8000/health
```

### 调用 /emotion/latest

```bash
curl http://127.0.0.1:8000/emotion/latest
```

### 调用 /emotion/image

```bash
curl -X POST http://127.0.0.1:8000/emotion/image -F "file=@test.jpg"
```

## 5. 参数说明

- `--crop-margin`：控制 crop 框相对 YOLO 人脸框的外扩比例。默认 `0.30`。
- `--sample-step`：每隔多少帧做一次完整检测和分类。值越大越省算力，但结果更新更慢。
- `--img-size`：YOLO 输入尺寸。默认 `640`，越大可能越准但更慢。
- `--face-conf`：人脸检测置信度阈值。调高可减少误检，调低可增加召回。
- `--device`：推理设备。留空自动选择；可手动设为 `cpu` 或 `cuda`。
- `--emotieff-engine`：EmotiEffLib 推理后端，默认 `torch`。

## 6. 常见问题

### No module named pandas

YOLO5Face 的部分工具文件会导入 `pandas`。运行：

```bash
pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple
```

或直接安装对应 requirements。

### PyTorch 2.6 weights_only load failed

PyTorch 2.6+ 修改了 `torch.load` 的默认行为。本交付包中的 `yolov5-face/models/experimental.py` 已兼容：

```python
try:
    ckpt = torch.load(w, map_location=map_location, weights_only=False)
except TypeError:
    ckpt = torch.load(w, map_location=map_location)
```

### GitHub 下载 EmotiEffLib 模型慢

首次运行 `enet_b0_8_best_vgaf` 可能需要下载模型。可设置代理，或提前在网络较好的环境运行一次。

### 摄像头打不开

Windows 尝试更换 `--camera-index 1` 或 `--camera-index 2`。Jetson 检查 `/dev/video*`，并确认用户有摄像头访问权限。

### CPU 模式很卡

可以增大 `--sample-step`，例如 `5` 或 `8`；也可以降低 `--img-size`，例如 `480`。

### 黄色框和绿色框分别代表什么

绿色框是 YOLO5Face 原始检测框。黄色框是扩展后实际送入 EmotiEffLib 的 crop 框。

### 表情识别不稳定怎么办

优先调整 `--crop-margin`。如果脸部裁剪太紧，调大；如果背景太多或旁边有人干扰，调小。算力不足时增大 `--sample-step`。

## 7. 输出结果说明

- `emotion`：最终表情标签。
- `confidence`：最终表情置信度。
- `probabilities`：7 类表情概率字典。
- `bbox`：YOLO5Face 原始人脸框 `[x1, y1, x2, y2]`。
- `crop_bbox`：实际送入 EmotiEffLib 的裁剪框。
- `face_confidence`：YOLO5Face 人脸检测置信度。

## 8. 给其他模块调用的建议

Jetson API 可以作为机器人情绪感知模块。其他模块可以周期性请求 `/emotion/latest`，把当前人的表情传给：

- GPT API：作为对话情绪上下文。
- 语音交互模块：根据用户情绪调整语气。
- ROS 控制模块：根据表情触发陪护动作。
- 小车行为模块：例如检测到开心时靠近，检测到难过时播放安慰语音。
