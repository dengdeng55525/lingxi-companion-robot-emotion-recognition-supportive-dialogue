import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from emotion_backends import EmotiEffClassifier


PROJECT_ROOT = Path(__file__).resolve().parent
YOLO_FACE_DIR = PROJECT_ROOT / "yolov5-face"
sys.path.insert(0, str(YOLO_FACE_DIR))

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression_face, scale_coords


def select_torch_device(device: str = "") -> torch.device:
    requested = str(device or "").strip()
    cuda_available = torch.cuda.is_available()

    if requested.lower() in ("", "auto"):
        return torch.device("cuda:0" if cuda_available else "cpu")

    selected = torch.device(requested)
    if selected.type != "cuda":
        return selected

    if not cuda_available:
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Inside the Linux VM this usually means the NVIDIA GPU is not passed "
            "through to the guest, the NVIDIA driver is missing, or nvidia-smi is "
            "not available. Use --device cpu for a CPU test, or fix GPU passthrough "
            "before running with --device cuda."
        )

    cuda_count = torch.cuda.device_count()
    cuda_index = selected.index if selected.index is not None else 0
    if cuda_index >= cuda_count:
        raise RuntimeError(
            f"CUDA device {selected} was requested, but only {cuda_count} CUDA device(s) are visible."
        )
    return selected


class YOLOFaceDetector:
    def __init__(
        self,
        weights: str,
        device: torch.device,
        img_size: int = 640,
        conf_thres: float = 0.4,
        iou_thres: float = 0.5,
    ):
        self.device = device
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.model = attempt_load(weights, map_location=device)
        self.model.to(device).eval()

    @torch.no_grad()
    def detect(self, img0: np.ndarray):
        img = letterbox(img0, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(self.device).float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        pred = self.model(img_tensor)[0]
        pred = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)
        results = []
        for det in pred:
            if det is None or len(det) == 0:
                continue
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img0.shape).round()
            h, w = img0.shape[:2]
            for row in det:
                x1, y1, x2, y2 = row[:4].detach().cpu().numpy().astype(int)
                face_conf = float(row[4].detach().cpu().item())
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))
                if x2 <= x1 or y2 <= y1:
                    continue
                results.append({"bbox": [int(x1), int(y1), int(x2), int(y2)], "face_confidence": face_conf})
        return results


class FaceEmotionPipeline:
    def __init__(
        self,
        face_weights: str = "yolov5-face/weights/yolov5s-face.pt",
        device: str = "",
        img_size: int = 640,
        face_conf: float = 0.4,
        iou_thres: float = 0.5,
        crop_margin: float = 0.30,
        emotion_backend: str = "emotieff",
        emotieff_model_name: str = "enet_b0_8_best_vgaf",
        emotieff_engine: str = "torch",
    ):
        if emotion_backend != "emotieff":
            raise ValueError("This release package only keeps YOLO5Face + EmotiEffLib.")

        self.device = select_torch_device(device)
        self.crop_margin = float(crop_margin)
        face_weights_path = Path(face_weights)
        if not face_weights_path.is_absolute():
            face_weights_path = PROJECT_ROOT / face_weights_path

        print(f"[INFO] device: {self.device}")
        print(f"[INFO] torch cuda available: {torch.cuda.is_available()}")
        if self.device.type == "cuda":
            print(f"[INFO] cuda device name: {torch.cuda.get_device_name(self.device)}")
        print(f"[INFO] face weights: {face_weights_path}")
        self.detector = YOLOFaceDetector(
            weights=str(face_weights_path),
            device=self.device,
            img_size=img_size,
            conf_thres=face_conf,
            iou_thres=iou_thres,
        )
        self.classifier = EmotiEffClassifier(
            model_name=emotieff_model_name,
            engine=emotieff_engine,
            device=self.device,
            merge_to_7=True,
        )

    def expand_bbox(self, bbox, img_shape):
        x1, y1, x2, y2 = bbox
        h, w = img_shape[:2]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        # EmotiEffLib expects a face-like crop, not a thin detector box.  On the
        # ASTRA stream YOLO can sometimes return a narrow face box at the image
        # edge; using a square crop around the detection center keeps the whole
        # face in the emotion model input.
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        side = max(bw, bh) * (1.0 + 2.0 * self.crop_margin)
        side = max(side, 112.0)
        half = side / 2.0

        nx1 = max(0, int(round(cx - half)))
        ny1 = max(0, int(round(cy - half)))
        nx2 = min(w - 1, int(round(cx + half)))
        ny2 = min(h - 1, int(round(cy + half)))

        if nx2 <= nx1:
            nx1 = max(0, min(w - 2, nx1))
            nx2 = min(w - 1, nx1 + 1)
        if ny2 <= ny1:
            ny1 = max(0, min(h - 2, ny1))
            ny2 = min(h - 1, ny1 + 1)
        return [int(nx1), int(ny1), int(nx2), int(ny2)]

    def process_frame(self, frame: np.ndarray):
        detections = self.detector.detect(frame)
        outputs = []
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = self.expand_bbox(bbox, frame.shape)
            face_crop = frame[y1:y2, x1:x2]
            emotion_result = self.classifier.predict(face_crop)
            if emotion_result is None:
                continue
            outputs.append(
                {
                    "bbox": [int(v) for v in bbox],
                    "crop_bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "face_confidence": float(det["face_confidence"]),
                    "emotion": str(emotion_result["emotion"]),
                    "confidence": float(emotion_result["confidence"]),
                    "model_emotion": str(emotion_result.get("model_emotion", emotion_result["emotion"])),
                    "top_probability_label": str(
                        emotion_result.get("top_probability_label", emotion_result["emotion"])
                    ),
                    "top_probability": float(
                        emotion_result.get("top_probability", emotion_result["confidence"])
                    ),
                    "label_consistent": bool(emotion_result.get("label_consistent", True)),
                    "emotion_classes": [
                        str(item) for item in emotion_result.get("emotion_classes", [])
                    ],
                    "raw_scores": [
                        float(item) for item in emotion_result.get("raw_scores", [])
                    ],
                    "softmax_by_class": {
                        str(k): float(v)
                        for k, v in emotion_result.get("softmax_by_class", {}).items()
                    },
                    "score_debug": dict(emotion_result.get("score_debug", {})),
                    "input_debug": dict(emotion_result.get("input_debug", {})),
                    "probabilities": {str(k): float(v) for k, v in emotion_result["probabilities"].items()},
                }
            )
        return outputs

    @staticmethod
    def draw_label(vis: np.ndarray, x: int, y: int, label: str, color=(0, 255, 0)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2
        padding = 5
        frame_h, frame_w = vis.shape[:2]
        text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_w, text_h = text_size
        box_w = text_w + padding * 2
        box_h = text_h + baseline + padding * 2
        x = max(0, min(int(x), frame_w - box_w))
        if y - box_h >= 0:
            y1, y2 = y - box_h, y
            text_y = y2 - baseline - padding
        else:
            y1 = max(0, min(int(y), frame_h - box_h))
            y2 = y1 + box_h
            text_y = y1 + padding + text_h
        cv2.rectangle(vis, (x, y1), (min(frame_w - 1, x + box_w), min(frame_h - 1, y2)), color, -1)
        cv2.putText(vis, label, (x + padding, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    def draw_results(self, frame: np.ndarray, results, fps: Optional[float] = None):
        vis = frame.copy()
        for item in results:
            x1, y1, x2, y2 = item["bbox"]
            cx1, cy1, cx2, cy2 = item.get("crop_bbox", item["bbox"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
            label = f"{item['emotion']} {item['confidence']:.2f} face:{item['face_confidence']:.2f}"
            self.draw_label(vis, x1, y1, label, color=(0, 255, 0))
        if fps is not None:
            self.draw_label(vis, 10, 28, f"FPS {fps:.1f}", color=(40, 40, 40))
        return vis
