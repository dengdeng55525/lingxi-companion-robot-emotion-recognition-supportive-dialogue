from __future__ import print_function

import argparse
import json
import os
import time
from pathlib import Path

import cv2

from simple_emotion_writer import LATEST_EMOTION_PATH, write_latest_emotion


EMOTION_ORDER = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "angry"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Jetson video/stream emotion inference with atomic JSON output."
    )
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--face-weights", type=str, default="yolov5-face/weights/yolov5s-face.pt")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--emotion-backend", type=str, default="emotieff", choices=["emotieff"])
    parser.add_argument("--emotieff-model-name", type=str, default="enet_b0_8_best_vgaf")
    parser.add_argument("--emotieff-engine", type=str, default="torch", choices=["torch", "onnx"])
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--face-conf", type=float, default=0.4)
    parser.add_argument("--iou-thres", type=float, default=0.5)
    parser.add_argument("--crop-margin", type=float, default=0.30)
    parser.add_argument("--sample-step", type=int, default=3)
    parser.add_argument("--json-output", type=str, default="outputs/latest_emotion.json")
    parser.add_argument("--json-update-interval", type=float, default=0.5)
    parser.add_argument("--latest-emotion-output", type=str, default=str(LATEST_EMOTION_PATH))
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def atomic_write_json(path, payload):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".tmp")

    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temp_path), str(output_path))
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise


def open_source(source):
    capture_source = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(capture_source)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def ordered_probabilities(probabilities):
    ordered = {}
    for label in EMOTION_ORDER:
        if label in probabilities:
            ordered[label] = float(probabilities[label])
    for label, value in probabilities.items():
        if label not in ordered:
            ordered[str(label)] = float(value)
    return ordered


def serialize_face(result, frame_shape):
    frame_h, frame_w = frame_shape[:2]
    bbox = [int(v) for v in result["bbox"]]
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    area = bbox_area(bbox)
    frame_area = max(1, int(frame_w) * int(frame_h))

    return {
        "emotion": str(result["emotion"]),
        "confidence": float(result["confidence"]),
        "face_confidence": float(result.get("face_confidence", 0.0)),
        "model_emotion": str(result.get("model_emotion", result["emotion"])),
        "top_probability_label": str(result.get("top_probability_label", result["emotion"])),
        "top_probability": float(result.get("top_probability", result["confidence"])),
        "label_consistent": bool(result.get("label_consistent", True)),
        "emotion_classes": [str(item) for item in result.get("emotion_classes", [])],
        "raw_scores": [float(item) for item in result.get("raw_scores", [])],
        "softmax_by_class": {
            str(k): float(v) for k, v in result.get("softmax_by_class", {}).items()
        },
        "score_debug": dict(result.get("score_debug", {})),
        "input_debug": dict(result.get("input_debug", {})),
        "probabilities": ordered_probabilities(result.get("probabilities", {})),
        "bbox": bbox,
        "crop_bbox": [int(v) for v in result.get("crop_bbox", bbox)],
        "center": [float(cx), float(cy)],
        "center_norm": [float(cx / max(1, frame_w)), float(cy / max(1, frame_h))],
        "area": int(area),
        "area_ratio": float(area / float(frame_area)),
    }


def build_payload(status, source, frame_index, frame, results, fps, message=""):
    frame_h, frame_w = frame.shape[:2] if frame is not None else (0, 0)
    faces = []
    if results:
        faces = [serialize_face(item, frame.shape) for item in results]
        faces.sort(key=lambda item: item["area"], reverse=True)

    payload = {
        "status": status,
        "timestamp": time.time(),
        "source": str(source),
        "frame_index": int(frame_index),
        "fps": float(fps),
        "frame": {
            "width": int(frame_w),
            "height": int(frame_h),
        },
        "face_count": len(faces),
        "valid": bool(status == "ok" and faces),
        "main_person": faces[0] if faces else None,
        "faces": faces,
    }

    if message:
        payload["message"] = str(message)
    elif status == "ok" and not faces:
        payload["message"] = "no face detected"

    return payload


def write_latest_emotion_state(results, status, output_path):
    write_latest_emotion(results if status == "ok" else [], output_path)


def main():
    args = parse_args()
    sample_step = max(1, int(args.sample_step))
    json_update_interval = max(0.0, float(args.json_update_interval))
    latest_emotion_output_path = Path(args.latest_emotion_output)

    print("[INFO] source: {}".format(args.source))
    print("[INFO] json output: {}".format(Path(args.json_output).resolve()))
    print("[INFO] latest emotion output: {}".format(latest_emotion_output_path))
    print("[INFO] loading pipeline...")
    from pipeline_face_emotion import FaceEmotionPipeline

    pipeline = FaceEmotionPipeline(
        face_weights=args.face_weights,
        device=args.device,
        img_size=args.img_size,
        face_conf=args.face_conf,
        iou_thres=args.iou_thres,
        crop_margin=args.crop_margin,
        emotion_backend=args.emotion_backend,
        emotieff_model_name=args.emotieff_model_name,
        emotieff_engine=args.emotieff_engine,
    )
    print("[INFO] pipeline ready.")

    cap = open_source(args.source)
    if not cap.isOpened():
        payload = build_payload(
            "error", args.source, 0, None, [], 0.0, "failed to open video source"
        )
        atomic_write_json(args.json_output, payload)
        write_latest_emotion_state([], "error", latest_emotion_output_path)
        raise RuntimeError("Failed to open camera/source: {}".format(args.source))

    frame_index = 0
    last_results = []
    fps = 0.0
    last_tick = time.time()
    last_json_write = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                payload = build_payload(
                    "error", args.source, frame_index, None, [], fps, "failed to read video source"
                )
                atomic_write_json(args.json_output, payload)
                write_latest_emotion_state([], "error", latest_emotion_output_path)
                print("[WARN] failed to read source frame.")
                break

            now = time.time()
            current_fps = 1.0 / max(1e-6, now - last_tick)
            fps = current_fps if fps <= 0 else fps * 0.9 + current_fps * 0.1
            last_tick = now

            status = "ok"
            message = ""
            if frame_index % sample_step == 0:
                try:
                    last_results = pipeline.process_frame(frame)
                    if not last_results:
                        message = "no face detected"
                except Exception as exc:
                    status = "error"
                    last_results = []
                    message = "inference failed: {}".format(exc)
                    print("[ERROR] {}".format(message))

            now = time.time()
            if now - last_json_write >= json_update_interval:
                payload = build_payload(
                    status, args.source, frame_index, frame, last_results, fps, message
                )
                atomic_write_json(args.json_output, payload)
                write_latest_emotion_state(last_results, status, latest_emotion_output_path)
                last_json_write = now

            if frame_index % 30 == 0:
                main = "none"
                debug = ""
                if last_results:
                    largest = max(last_results, key=lambda item: bbox_area(item["bbox"]))
                    main = "{} {:.2f}".format(largest["emotion"], largest["confidence"])
                    input_debug = largest.get("input_debug", {})
                    score_debug = largest.get("score_debug", {})
                    debug = " input_checksum={} score_checksum={} score_std={:.6f}".format(
                        input_debug.get("checksum", "n/a"),
                        score_debug.get("checksum", "n/a"),
                        float(score_debug.get("std", 0.0) or 0.0),
                    )
                print(
                    "frame={} faces={} fps={:.1f} main={}{} json={}".format(
                        frame_index, len(last_results), fps, main, debug, args.json_output
                    )
                )

            if args.show:
                vis = pipeline.draw_results(frame, last_results, fps=fps)
                cv2.imshow("Jetson Emotion JSON", vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            frame_index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] released.")


if __name__ == "__main__":
    main()
