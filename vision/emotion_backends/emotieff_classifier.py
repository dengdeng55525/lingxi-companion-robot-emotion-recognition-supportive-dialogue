from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from emotiefflib.facial_analysis import EmotiEffLibRecognizer


class EmotiEffClassifier:
    """EmotiEffLib wrapper with a stable dict output used by the release code."""

    def __init__(
        self,
        model_name: str = "enet_b0_8_best_vgaf",
        engine: str = "torch",
        device: Optional[Union[torch.device, str]] = None,
        merge_to_7: bool = True,
    ):
        self.model_name = model_name
        self.engine = engine
        self.device = str(device) if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.merge_to_7 = merge_to_7
        self.standard_7_classes = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "angry"]

        self.recognizer = EmotiEffLibRecognizer(
            engine=self.engine,
            model_name=self.model_name,
            device=self.device,
        )

        # EmotiEffLib uses this index order internally:
        # 7 classes: Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
        # 8 classes: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
        # Keeping this order is critical, otherwise the displayed label can disagree with
        # the probability vector.
        self.fallback_7_classes = [
            "angry",
            "disgust",
            "fear",
            "happy",
            "neutral",
            "sad",
            "surprise",
        ]
        self.fallback_8_classes = [
            "angry",
            "contempt",
            "disgust",
            "fear",
            "happy",
            "neutral",
            "sad",
            "surprise",
        ]

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    def _normalize_label(self, label: str) -> str:
        label = str(label).strip().lower()
        alias = {
            "anger": "angry",
            "angry": "angry",
            "happiness": "happy",
            "happy": "happy",
            "sadness": "sad",
            "sad": "sad",
            "surprise": "surprise",
            "surprised": "surprise",
            "fear": "fear",
            "afraid": "fear",
            "disgust": "disgust",
            "disgusted": "disgust",
            "neutral": "neutral",
            "contempt": "contempt",
        }
        label = alias.get(label, label)
        if self.merge_to_7 and label == "contempt":
            label = "neutral"
        return label

    def _read_class_names_from_obj(self, obj, score_len: int):
        if obj is None:
            return None
        for attr in [
            "classes",
            "class_names",
            "emotion_classes",
            "emotion_names",
            "emotions",
            "idx_to_class",
            "idx_to_emotion",
            "idx_to_emotion_class",
            "idx_to_label",
        ]:
            if not hasattr(obj, attr):
                continue
            value = getattr(obj, attr)
            if isinstance(value, dict):
                try:
                    names = [value[i] for i in range(score_len)]
                    return [self._normalize_label(x) for x in names]
                except Exception:
                    continue
            if isinstance(value, (list, tuple)) and len(value) == score_len:
                return [self._normalize_label(x) for x in value]
        return None

    def _get_class_names(self, score_len: int) -> List[str]:
        names = self._read_class_names_from_obj(self.recognizer, score_len)
        if names is None:
            names = self._read_class_names_from_obj(getattr(self.recognizer, "model", None), score_len)
        if names is None:
            names = self.fallback_8_classes if score_len == 8 else self.fallback_7_classes
        return [self._normalize_label(x) for x in names]

    def _build_probabilities(self, score: np.ndarray) -> Dict[str, float]:
        prob = self._softmax(score)
        class_names = self._get_class_names(len(prob))
        probabilities: Dict[str, float] = {}
        for name, p in zip(class_names, prob):
            name = self._normalize_label(name)
            probabilities[name] = probabilities.get(name, 0.0) + float(p)

        if self.merge_to_7:
            probabilities = {name: float(probabilities.get(name, 0.0)) for name in self.standard_7_classes}

        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: float(v / total) for k, v in probabilities.items()}
        return probabilities

    @staticmethod
    def _top_probability(probabilities: Dict[str, float]):
        if not probabilities:
            return None, 0.0
        label = max(probabilities, key=probabilities.get)
        return str(label), float(probabilities[label])

    @staticmethod
    def _prepare_face_rgb(face_img: np.ndarray, input_format: str = "bgr") -> np.ndarray:
        if face_img is None or face_img.size == 0:
            raise ValueError("Empty face image received by EmotiEffClassifier.")
        if len(face_img.shape) != 3 or face_img.shape[2] != 3:
            raise ValueError(f"Invalid face image shape: {face_img.shape}")
        if face_img.dtype != np.uint8:
            face_img = np.clip(face_img, 0, 255).astype(np.uint8)
        if input_format.lower() == "bgr":
            return cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        if input_format.lower() == "rgb":
            return face_img
        raise ValueError(f"Unsupported input_format: {input_format}")

    @staticmethod
    def _image_debug(face_img: np.ndarray):
        arr = np.asarray(face_img)
        if arr.size == 0:
            return {
                "shape": [0, 0, 0],
                "mean": 0.0,
                "std": 0.0,
                "min": 0,
                "max": 0,
                "checksum": 0,
            }
        # A cheap fingerprint. If the person moves or expression/crop changes,
        # this should usually change too.
        sample = arr[:: max(1, arr.shape[0] // 16), :: max(1, arr.shape[1] // 16), :]
        return {
            "shape": [int(x) for x in arr.shape],
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
            "checksum": int(np.sum(sample.astype(np.uint32)) % 1000000007),
        }

    def predict(self, face_img: np.ndarray, input_format: str = "bgr"):
        results = self.predict_batch([face_img], input_format=input_format)
        return results[0] if results else None

    @torch.no_grad()
    def predict_batch(self, face_imgs: List[np.ndarray], input_format: str = "bgr"):
        if not face_imgs:
            return []

        valid_faces = []
        valid_debug = []
        valid_indices = []
        for idx, face_img in enumerate(face_imgs):
            if face_img is None or face_img.size == 0:
                continue
            prepared = self._prepare_face_rgb(face_img, input_format=input_format)
            valid_faces.append(prepared)
            valid_debug.append(self._image_debug(prepared))
            valid_indices.append(idx)

        output = [None for _ in face_imgs]
        if not valid_faces:
            return output

        emotions, scores = self.recognizer.predict_emotions(valid_faces, logits=True)
        for local_idx, original_idx in enumerate(valid_indices):
            model_emotion = self._normalize_label(emotions[local_idx])
            score = np.asarray(scores[local_idx], dtype=np.float32)
            class_names = self._get_class_names(len(score))
            probabilities = self._build_probabilities(score)
            softmax_score = self._softmax(score)
            emotion, confidence = self._top_probability(probabilities)
            if emotion is None:
                emotion = model_emotion
                confidence = float(np.max(softmax_score))
            output[original_idx] = {
                "emotion": str(emotion),
                "confidence": confidence,
                "model_emotion": str(model_emotion),
                "top_probability_label": str(emotion),
                "top_probability": confidence,
                "label_consistent": bool(model_emotion == emotion),
                "emotion_classes": [str(x) for x in class_names],
                "raw_scores": [float(x) for x in score.tolist()],
                "softmax_by_class": {
                    str(name): float(value) for name, value in zip(class_names, softmax_score.tolist())
                },
                "score_debug": {
                    "min": float(np.min(score)),
                    "max": float(np.max(score)),
                    "std": float(np.std(score)),
                    "checksum": int(np.sum(np.round(score * 1000000).astype(np.int64)) % 1000000007),
                },
                "input_debug": valid_debug[local_idx],
                "probabilities": {str(k): float(v) for k, v in probabilities.items()},
            }
        return output
