# 视频流 JSON 输出使用说明

## 目标

`stream_emotion_to_json.py` 用于持续读取视频流或本地摄像头画面，复用现有 `FaceEmotionPipeline` 完成人脸检测和表情识别，并把最新结果稳定写入 `JSON` 文件。

默认输出文件：

```text
outputs/latest_emotion.json
```

## 运行示例

### 1. 分析 HTTP 或 RTSP 视频流

```powershell
python .\stream_emotion_to_json.py --camera-url http://192.168.x.x:xxxx/video --face-weights .\yolov5-face\weights\yolov5s-face.pt --emotion-backend emotieff --emotieff-model-name enet_b0_8_best_vgaf --emotieff-engine torch --crop-margin 0.30 --sample-step 3 --json-output .\outputs\latest_emotion.json --json-update-interval 0.5 --show
```

也可以改成 RTSP：

```powershell
python .\stream_emotion_to_json.py --camera-url rtsp://192.168.x.x:8554/live --face-weights .\yolov5-face\weights\yolov5s-face.pt --emotion-backend emotieff --emotieff-model-name enet_b0_8_best_vgaf --emotieff-engine torch --crop-margin 0.30 --sample-step 3 --json-output .\outputs\latest_emotion.json --json-update-interval 0.5 --show
```

### 2. 分析本地摄像头

```powershell
python .\stream_emotion_to_json.py --camera-index 0 --face-weights .\yolov5-face\weights\yolov5s-face.pt --emotion-backend emotieff --emotieff-model-name enet_b0_8_best_vgaf --emotieff-engine torch --crop-margin 0.30 --sample-step 3 --json-output .\outputs\latest_emotion.json --show
```

## 参数说明

- `--camera-url`：视频流地址，例如 HTTP、RTSP。传入后优先于 `--camera-index`。
- `--camera-index`：本地摄像头编号，默认 `0`。
- `--face-weights`：YOLO5Face 权重路径。
- `--device`：推理设备，留空时自动选择 CUDA 或 CPU。
- `--crop-margin`：在人脸框外扩多少再送入 EmotiEffLib。
- `--sample-step`：每隔多少帧执行一次完整推理，默认 `3`。
- `--json-output`：输出 JSON 文件路径，默认 `outputs/latest_emotion.json`。
- `--json-update-interval`：JSON 最小写入间隔，默认 `0.5` 秒。
- `--reconnect-interval`：视频流断开后的重连等待时间，默认 `2.0` 秒。
- `--show`：显示实时画面、检测框和情绪标签。

## JSON 输出结构

检测到人脸时：

```json
{
  "status": "ok",
  "timestamp": 1710000000.123,
  "source": "http://192.168.x.x:xxxx/video",
  "frame_index": 120,
  "face_count": 1,
  "valid": true,
  "main_person": {
    "emotion": "happy",
    "confidence": 0.82,
    "probabilities": {
      "neutral": 0.10,
      "happy": 0.82,
      "sad": 0.03,
      "surprise": 0.02,
      "fear": 0.01,
      "disgust": 0.01,
      "angry": 0.01
    },
    "bbox": [120, 80, 260, 260]
  },
  "faces": [
    {
      "emotion": "happy",
      "confidence": 0.82,
      "probabilities": {
        "neutral": 0.10,
        "happy": 0.82,
        "sad": 0.03,
        "surprise": 0.02,
        "fear": 0.01,
        "disgust": 0.01,
        "angry": 0.01
      },
      "bbox": [120, 80, 260, 260]
    }
  ]
}
```

没有检测到人脸时：

```json
{
  "status": "ok",
  "timestamp": 1710000001.456,
  "source": "http://192.168.x.x:xxxx/video",
  "frame_index": 121,
  "face_count": 0,
  "valid": false,
  "main_person": null,
  "faces": [],
  "message": "no face detected"
}
```

视频流打开失败、读取失败或推理异常时：

```json
{
  "status": "error",
  "timestamp": 1710000002.789,
  "source": "http://192.168.x.x:xxxx/video",
  "frame_index": 121,
  "face_count": 0,
  "valid": false,
  "main_person": null,
  "faces": [],
  "message": "failed to read video stream"
}
```

## 字段说明

- `status`：`ok` 表示脚本正常运行，`error` 表示当前源读取或推理存在异常。
- `timestamp`：写入 JSON 时的 Unix 时间戳，单位秒。
- `source`：输入源。视频流时为 URL，本地摄像头时为 `camera_index:0` 这类字符串。
- `frame_index`：最近一次完成分析的帧编号。
- `face_count`：当前帧中识别到的人脸数量。
- `valid`：当前结果是否存在可用主目标。
- `main_person`：面积最大的那张脸，会作为主目标输出。
- `faces`：当前帧全部人脸结果数组。
- `message`：无脸或异常时的补充说明。

## 原子写入说明

JSON 写入流程如下：

```text
outputs/latest_emotion.json.tmp -> rename / replace -> outputs/latest_emotion.json
```

这样下游程序在读取时，不容易拿到半截 JSON 内容。

## 下游程序如何读取

后续推理程序只需要反复读取：

```text
outputs/latest_emotion.json
```

Python 示例：

```python
import json
from pathlib import Path

json_path = Path("outputs/latest_emotion.json")
data = json.loads(json_path.read_text(encoding="utf-8"))

if data["status"] == "ok" and data["valid"]:
    main_person = data["main_person"]
    emotion = main_person["emotion"]
    confidence = main_person["confidence"]
```

建议读取策略：

- 先判断 `status == "ok"`。
- 再判断 `valid == true`。
- 需要主目标时读取 `main_person`。
- 需要多人脸时遍历 `faces`。

## 运行日志

脚本启动和运行时会持续输出关键信息，包括：

- 当前视频流地址或本地摄像头编号
- 模型加载状态
- 当前处理 FPS
- 当前主情绪
- 是否检测到人脸
- JSON 输出路径
- 视频流断开后的重连提示
