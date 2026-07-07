from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


LATEST_EMOTION_PATH = Path("/home/lzc/.ros/latest_emotion.json")
REQUIRED_SCORE_LABELS = ("happy", "sad", "angry", "neutral")
KNOWN_OUTPUT_LABELS = {"happy", "sad", "angry", "anxious", "neutral", "excited", "unknown"}


def clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def bbox_area(item: dict[str, Any]) -> int:
    bbox = item.get("bbox") or []
    if len(bbox) != 4:
        return 0
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return max(0, x2 - x1) * max(0, y2 - y1)


def normalize_emotion(label: Any) -> str:
    value = str(label or "").strip().lower()
    aliases = {
        "happiness": "happy",
        "happy": "happy",
        "sadness": "sad",
        "sad": "sad",
        "anger": "angry",
        "angry": "angry",
        "neutral": "neutral",
        "contempt": "neutral",
        "fear": "anxious",
        "afraid": "anxious",
        "anxiety": "anxious",
        "anxious": "anxious",
        "disgust": "anxious",
        "disgusted": "anxious",
        "surprise": "excited",
        "surprised": "excited",
        "excitement": "excited",
        "excited": "excited",
    }
    return aliases.get(value, value if value in KNOWN_OUTPUT_LABELS else "unknown")


def normalize_scores(probabilities: dict[str, Any] | None) -> dict[str, float]:
    scores: dict[str, float] = {label: 0.0 for label in REQUIRED_SCORE_LABELS}
    if not probabilities:
        return scores

    for raw_label, raw_score in probabilities.items():
        label = normalize_emotion(raw_label)
        score = clamp01(raw_score)
        if label == "unknown":
            label = str(raw_label or "unknown").strip().lower() or "unknown"
        scores[label] = clamp01(scores.get(label, 0.0) + score)

    for label in REQUIRED_SCORE_LABELS:
        scores.setdefault(label, 0.0)
    return scores


def build_latest_emotion_payload(results: list[dict[str, Any]] | None) -> dict[str, Any]:
    face_count = len(results or [])
    if face_count <= 0:
        return {
            "timestamp": float(time.time()),
            "face_count": 0,
            "dominant_emotion": "unknown",
            "confidence": 0.0,
            "scores": {},
        }

    main_face = max(results or [], key=bbox_area)
    probabilities = main_face.get("probabilities") or {}
    scores = normalize_scores(probabilities)
    dominant = normalize_emotion(
        main_face.get("emotion")
        or main_face.get("top_probability_label")
        or main_face.get("model_emotion")
    )

    if dominant == "unknown" and scores:
        dominant = max(scores, key=scores.get)

    confidence = scores.get(dominant)
    if confidence is None or confidence <= 0:
        confidence = clamp01(
            main_face.get("confidence")
            or main_face.get("top_probability")
            or main_face.get("face_confidence")
        )
        if dominant in REQUIRED_SCORE_LABELS or dominant in ("anxious", "excited"):
            scores[dominant] = max(scores.get(dominant, 0.0), confidence)

    return {
        "timestamp": float(time.time()),
        "face_count": int(face_count),
        "dominant_emotion": str(dominant),
        "confidence": clamp01(confidence),
        "scores": {str(key): float(value) for key, value in scores.items()},
    }


def atomic_write_latest_emotion(payload: dict[str, Any], target_path: str | Path = LATEST_EMOTION_PATH) -> Path:
    target = Path(target_path)
    temp = target.with_name(f"{target.name}.tmp")
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
    except Exception:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    return target


def write_latest_emotion(results: list[dict[str, Any]] | None, target_path: str | Path = LATEST_EMOTION_PATH) -> Path:
    return atomic_write_latest_emotion(build_latest_emotion_payload(results), target_path)
