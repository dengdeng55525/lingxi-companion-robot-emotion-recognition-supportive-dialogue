# YOLO5Face + EmotiEffLib-vgaf Windows 版

当前项目保留了原有的本地摄像头实时验证能力，并新增了一个适合小车视频流接入的入口脚本：

```text
视频流 / 本地摄像头 -> YOLO5Face 检测人脸 -> 扩展 crop_bbox -> EmotiEffLib 表情分类 -> 持续输出 latest_emotion.json
```

## 目录说明

- `windows_camera_realtime.py`：原有 Windows 本地摄像头实时验证脚本。
- `stream_emotion_to_json.py`：新增统一入口，支持 `--camera-url` 或 `--camera-index`，并持续输出 JSON。
- `pipeline_face_emotion.py`：主推理流程，保留 YOLO5Face + EmotiEffLib 逻辑。
- `utils/json_writer.py`：JSON 原子写入工具，避免读取到半截文件。
- `emotion_backends/`：EmotiEffLib 分类器封装。
- `yolov5-face/`：YOLO5Face 推理代码和 `yolov5s-face.pt` 权重。
- `scripts/check_env_windows.py`：环境检查脚本。
- `scripts/run_windows_camera.ps1`：本地摄像头快速运行示例。
- `docs/参数说明.md`：原有本地实时脚本参数说明。
- `docs/video_stream_json_usage.md`：视频流 JSON 输出脚本说明。

## 安装依赖

建议使用已有 Conda 环境，不要覆盖已有 torch / torchvision / CUDA 版本。

```powershell
conda activate faceemo310
pip install -r requirements_windows.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

`requirements_windows.txt` 没有固定 torch 版本，避免破坏已有深度学习环境。

## 检查环境

```powershell
python .\scripts\check_env_windows.py
```

需要看到：

- OpenCV 可导入
- EmotiEffLib 可导入
- `yolov5-face/weights/yolov5s-face.pt` 存在

## 首次运行模型下载慢

EmotiEffLib 的 `enet_b0_8_best_vgaf` 可能首次运行时自动下载。如果 GitHub 下载慢，可以在 PowerShell 中设置代理：

```powershell
$env:HTTP_PROXY="http://127.0.0.1:7892"
$env:HTTPS_PROXY="http://127.0.0.1:7892"
```

## 运行本地摄像头实时验证

```powershell
python .\windows_camera_realtime.py --camera-index 0 --face-weights .\yolov5-face\weights\yolov5s-face.pt --emotion-backend emotieff --emotieff-model-name enet_b0_8_best_vgaf --emotieff-engine torch --crop-margin 0.30 --sample-step 3
```

也可以运行：

```powershell
.\scripts\run_windows_camera.ps1
```

按 `q` 退出窗口。

## 运行视频流分析并输出 JSON

HTTP/RTSP 视频流示例：

```powershell
python .\stream_emotion_to_json.py --camera-url http://192.168.x.x:xxxx/video --face-weights .\yolov5-face\weights\yolov5s-face.pt --emotion-backend emotieff --emotieff-model-name enet_b0_8_best_vgaf --emotieff-engine torch --crop-margin 0.30 --sample-step 3 --json-output .\outputs\latest_emotion.json --json-update-interval 0.5 --show
```

本地摄像头示例：

```powershell
python .\stream_emotion_to_json.py --camera-index 0 --face-weights .\yolov5-face\weights\yolov5s-face.pt --emotion-backend emotieff --emotieff-model-name enet_b0_8_best_vgaf --emotieff-engine torch --crop-margin 0.30 --sample-step 3 --json-output .\outputs\latest_emotion.json --show
```

说明：

- 传入 `--camera-url` 时，会优先读取视频流地址。
- 不传 `--camera-url` 时，会回退到 `--camera-index` 本地摄像头模式。
- 不加 `--show` 时，脚本只在后台运行并持续写 JSON。
- 下游程序只需要读取 `outputs/latest_emotion.json`，即可获取当前最新视觉识别结果。

## 画面说明

- 绿色框：YOLO5Face 原始人脸检测框。
- 黄色框：实际送入 EmotiEffLib 的 crop 框。
- 标签示例：`happy 0.82 face:0.91`。
- FPS：当前显示帧率。

更多 JSON 字段和接入说明见 [docs/video_stream_json_usage.md](D:\desktop\release_build\yolov5_emotieff_vgaf_windows\docs\video_stream_json_usage.md)。
