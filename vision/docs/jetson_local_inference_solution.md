# Jetson 本地推理解决方案

## 判断

当前报错来自 `yolov5s-face.pt`：JetPack 4.2 上的 torch 1.4.0 读不了新版 PyTorch 保存格式。`--emotieff-engine onnx` 只影响 EmotiEffLib 表情分类器，不能解决 YOLO5Face `.pt` 加载失败。

因此本地推理建议分两步验证：

1. 先尝试旧格式 `.pt`，用最小改动确认 torch 1.4.0 是否能跑 YOLO5Face。
2. 如果旧格式 `.pt` 仍有结构兼容问题，再把 YOLO5Face 也转成 ONNX/TensorRT。

## 路线 A：快速尝试旧格式 .pt

在有新版 torch 的电脑或虚拟机上运行：

```powershell
python .\scripts\convert_yolov5face_legacy_pt.py --input .\yolov5-face\weights\yolov5s-face.pt --output .\yolov5-face\weights\yolov5s-face_torch14.pt
```

把 `yolov5s-face_torch14.pt` 传到 Jetson 后，先只测加载：

```bash
python3 - <<'PY'
import torch
ckpt = torch.load("yolov5s-face_torch14.pt", map_location="cpu")
print(type(ckpt))
print("load ok")
PY
```

如果加载通过，再用现有脚本替换权重路径测试：

```bash
python3 stream_emotion_to_json.py \
  --camera-index 0 \
  --face-weights yolov5-face/weights/yolov5s-face_torch14.pt \
  --emotion-backend emotieff \
  --emotieff-engine torch \
  --crop-margin 0.30 \
  --sample-step 3 \
  --json-output outputs/latest_emotion.json \
  --show
```

如果这里能跑，先保留这个方案。它改动最小，适合作为本地推理第一轮验证。

## 路线 B：YOLO5Face 转 ONNX

如果路线 A 失败，说明不仅是文件格式问题，还可能有模型结构、算子或 torchvision 兼容问题。下一步把 YOLO5Face 导出成 ONNX。

在有新版 torch 的电脑或虚拟机上运行：

```powershell
python .\scripts\export_yolov5face_onnx.py --weights .\yolov5-face\weights\yolov5s-face.pt --output .\yolov5-face\weights\yolov5s-face.onnx --img-size 640 --opset 10 --simplify
```

如果没有安装 `onnxsim`，可以去掉 `--simplify`：

```powershell
python .\scripts\export_yolov5face_onnx.py --weights .\yolov5-face\weights\yolov5s-face.pt --output .\yolov5-face\weights\yolov5s-face.onnx --img-size 640 --opset 10
```

JetPack 4.2 上建议优先走 TensorRT，因为 JetPack 4.2 自带 TensorRT Python API。先在 Jetson 上测试能不能解析：

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=yolov5-face/weights/yolov5s-face.onnx \
  --saveEngine=yolov5-face/weights/yolov5s-face_fp16.engine \
  --fp16
```

如果 `trtexec` 不在这个路径，先找一下：

```bash
find /usr -name trtexec 2>/dev/null
```

## 验证顺序

建议不要一上来跑视频流，先按这个顺序排错：

1. 权重加载或 ONNX/TensorRT 解析通过。
2. 单张图片人脸检测输出 bbox。
3. 单张 face crop 表情分类输出 emotion。
4. 摄像头单帧推理。
5. 视频循环 + JSON 输出。

这样每一步失败都能清楚定位，不会把摄像头、模型、JSON、性能问题混在一起。

## 预期结果

本地推理第一轮优先做路线 A，因为它最快。只要旧格式 `.pt` 能在 Jetson 上加载并完成一次 `process_frame()`，就可以直接复用当前 `stream_emotion_to_json.py` 输出 `outputs/latest_emotion.json`。

如果路线 A 卡在模型结构或算子兼容，切到路线 B。路线 B 的最终工程改动是新增一个 YOLO5Face ONNX/TensorRT detector 后端，让 `FaceEmotionPipeline` 可以选择 `torch` 或 `onnx/trt` 人脸检测后端。
