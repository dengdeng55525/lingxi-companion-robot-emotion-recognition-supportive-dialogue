from __future__ import annotations

import argparse
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from json_writer import atomic_write_json
from simple_emotion_writer import LATEST_EMOTION_PATH, write_latest_emotion


EMOTION_ORDER = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "angry"]


@dataclass
class AnalysisState:
    status: str = "error"
    frame_index: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    message: str = "waiting to read video stream"
    fps: float = 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read a local camera or video stream, run YOLO5Face + EmotiEffLib, and keep a JSON result file updated."
    )
    parser.add_argument("--source", type=str, default="", help="Unified source. Can be a URL/path or a camera index. Overrides --camera-url/--camera-index.")
    parser.add_argument("--camera-url", type=str, default="", help="HTTP/RTSP stream URL. Takes precedence over --camera-index.")
    parser.add_argument("--camera-index", type=int, default=0, help="Local camera index used when --camera-url is empty.")
    parser.add_argument(
        "--opencv-backend",
        type=str,
        default="auto",
        choices=["auto", "default", "ffmpeg", "gstreamer", "dshow", "v4l2"],
        help="OpenCV capture backend. auto uses FFmpeg for URLs, DSHOW on Windows cameras, and V4L2 on Linux cameras.",
    )
    parser.add_argument("--ffmpeg-low-latency", action="store_true", help="Set low-latency FFmpeg capture options before opening URL streams.")
    parser.add_argument("--rtsp-transport", type=str, default="tcp", choices=["tcp", "udp"], help="RTSP transport used when --ffmpeg-low-latency is enabled.")
    parser.add_argument("--open-timeout-ms", type=int, default=5000, help="OpenCV open timeout for supported backends.")
    parser.add_argument("--read-timeout-ms", type=int, default=5000, help="OpenCV read timeout for supported backends.")
    parser.add_argument("--face-weights", type=str, default="yolov5-face/weights/yolov5s-face.pt")
    parser.add_argument("--device", type=str, default="", help="Empty or auto means auto-select cuda/cpu.")
    parser.add_argument("--emotion-backend", type=str, default="emotieff", choices=["emotieff"])
    parser.add_argument("--emotieff-model-name", type=str, default="enet_b0_8_best_vgaf")
    parser.add_argument("--emotieff-engine", type=str, default="torch", choices=["torch", "onnx"])
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--face-conf", type=float, default=0.4)
    parser.add_argument("--iou-thres", type=float, default=0.5)
    parser.add_argument("--crop-margin", type=float, default=0.30)
    parser.add_argument("--sample-step", type=int, default=3)
    parser.add_argument("--json-output", type=str, default="outputs/latest_emotion.json")
    parser.add_argument("--json-update-interval", type=float, default=0.5, help="Minimum seconds between JSON writes.")
    parser.add_argument("--latest-emotion-output", type=str, default=str(LATEST_EMOTION_PATH))
    parser.add_argument("--reconnect-interval", type=float, default=2.0, help="Seconds to wait before reopening the source.")
    parser.add_argument("--show", action="store_true", help="Show the realtime visualization window.")
    return parser.parse_args()


def resolve_source(args):
    source = str(args.source or "").strip()
    if source:
        if source.isdigit():
            index = int(source)
            return index, f"camera_index:{index}", False
        return source, source, True

    camera_url = str(args.camera_url or "").strip()
    if camera_url:
        return camera_url, camera_url, True

    return int(args.camera_index), f"camera_index:{int(args.camera_index)}", False


def opencv_backend_id(name: str, is_stream: bool):
    backend = str(name or "auto").lower()
    if backend == "auto":
        if is_stream:
            backend = "ffmpeg"
        elif platform.system().lower() == "windows":
            backend = "dshow"
        elif platform.system().lower() == "linux":
            backend = "v4l2"
        else:
            backend = "default"

    mapping = {
        "default": None,
        "ffmpeg": getattr(cv2, "CAP_FFMPEG", None),
        "gstreamer": getattr(cv2, "CAP_GSTREAMER", None),
        "dshow": getattr(cv2, "CAP_DSHOW", None),
        "v4l2": getattr(cv2, "CAP_V4L2", None),
    }
    return backend, mapping.get(backend)


def configure_ffmpeg_options(args, is_stream: bool) -> None:
    if not is_stream or not args.ffmpeg_low_latency:
        return

    options = [
        ("fflags", "nobuffer"),
        ("flags", "low_delay"),
        ("probesize", "32"),
        ("analyzeduration", "0"),
        ("max_delay", "500000"),
    ]
    source = str(args.source or args.camera_url or "").lower()
    if source.startswith("rtsp://"):
        options.insert(0, ("rtsp_transport", args.rtsp_transport))

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(f"{key};{value}" for key, value in options)
    print(f"[INFO] OPENCV_FFMPEG_CAPTURE_OPTIONS={os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']}")


def capture_params(args):
    params = []
    prop_pairs = [
        ("CAP_PROP_OPEN_TIMEOUT_MSEC", int(args.open_timeout_ms)),
        ("CAP_PROP_READ_TIMEOUT_MSEC", int(args.read_timeout_ms)),
    ]
    for prop_name, value in prop_pairs:
        prop = getattr(cv2, prop_name, None)
        if prop is not None and value > 0:
            params.extend([prop, value])
    return params


def make_capture(capture_source, backend_id, params):
    if backend_id is None:
        try:
            return cv2.VideoCapture(capture_source, params)
        except Exception:
            return cv2.VideoCapture(capture_source)

    try:
        return cv2.VideoCapture(capture_source, backend_id, params)
    except Exception:
        return cv2.VideoCapture(capture_source, backend_id)


def open_video_source(capture_source, is_stream: bool, args):
    configure_ffmpeg_options(args, is_stream)
    backend_name, backend_id = opencv_backend_id(args.opencv_backend, is_stream)
    params = capture_params(args)
    cap = make_capture(capture_source, backend_id, params)

    if not cap.isOpened() and backend_name != "default":
        cap.release()
        print(f"[WARN] OpenCV backend {backend_name} failed, retrying default backend.")
        cap = make_capture(capture_source, None, params)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[INFO] video source opened with backend: {backend_name}")
    return cap


def bbox_area(bbox: list[int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def reorder_probabilities(probabilities: dict[str, Any]) -> dict[str, float]:
    ordered: dict[str, float] = {}
    for label in EMOTION_ORDER:
        if label in probabilities:
            ordered[label] = float(probabilities[label])
    for label, value in probabilities.items():
        if label not in ordered:
            ordered[str(label)] = float(value)
    return ordered


def serialize_face(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "emotion": str(result["emotion"]),
        "confidence": float(result["confidence"]),
        "model_emotion": str(result.get("model_emotion", result["emotion"])),
        "top_probability_label": str(result.get("top_probability_label", result["emotion"])),
        "top_probability": float(result.get("top_probability", result["confidence"])),
        "label_consistent": bool(result.get("label_consistent", True)),
        "probabilities": reorder_probabilities(result.get("probabilities", {})),
        "bbox": [int(v) for v in result["bbox"]],
    }


def build_result_payload(source: str, state: AnalysisState) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": state.status,
        "timestamp": time.time(),
        "source": source,
        "frame_index": int(state.frame_index),
        "face_count": 0,
        "valid": False,
        "main_person": None,
        "faces": [],
    }

    if state.status == "error":
        payload["message"] = state.message
        return payload

    faces = [serialize_face(item) for item in state.results]
    faces.sort(key=lambda item: bbox_area(item["bbox"]), reverse=True)
    payload["face_count"] = len(faces)
    payload["faces"] = faces
    if faces:
        payload["valid"] = True
        payload["main_person"] = faces[0]
    else:
        payload["message"] = state.message or "no face detected"
    return payload


def write_state_json(source: str, state: AnalysisState, output_path: Path) -> Path:
    return atomic_write_json(output_path, build_result_payload(source, state))


def write_latest_emotion_state(state: AnalysisState, output_path: Path) -> Path:
    results = state.results if state.status == "ok" else []
    return write_latest_emotion(results, output_path)


def update_fps(previous_fps: float, last_tick: float):
    now = time.perf_counter()
    current_fps = 1.0 / max(1e-6, now - last_tick)
    if previous_fps <= 0:
        return current_fps, now
    return previous_fps * 0.9 + current_fps * 0.1, now


def describe_main_person(results: list[dict[str, Any]]) -> str:
    if not results:
        return "none"
    main_face = max(results, key=lambda item: bbox_area(item["bbox"]))
    return f"{main_face['emotion']} ({main_face['confidence']:.2f})"


def log_runtime_status(source: str, state: AnalysisState, output_path: Path):
    if state.status == "error":
        print(
            f"[WARN] source={source} frame={state.frame_index} valid=False "
            f"message={state.message} json={output_path}"
        )
        return

    print(
        f"[INFO] source={source} frame={state.frame_index} fps={state.fps:.1f} "
        f"face_count={len(state.results)} valid={bool(state.results)} "
        f"main={describe_main_person(state.results)} json={output_path}"
    )


def main():
    args = parse_args()
    sample_step = max(1, int(args.sample_step))
    json_update_interval = max(0.0, float(args.json_update_interval))
    reconnect_interval = max(0.2, float(args.reconnect_interval))
    json_output_path = Path(args.json_output)
    latest_emotion_output_path = Path(args.latest_emotion_output)
    capture_source, source, is_stream = resolve_source(args)
    window_name = "YOLO5Face + EmotiEffLib Stream"

    print(f"[INFO] source: {source}")
    print(f"[INFO] capture source: {capture_source}")
    print(f"[INFO] json output: {json_output_path.resolve()}")
    print(f"[INFO] latest emotion output: {latest_emotion_output_path}")
    print(f"[INFO] sample step: {sample_step}")
    print(f"[INFO] json update interval: {json_update_interval:.2f}s")
    print(f"[INFO] visualization: {'on' if args.show else 'off'}")
    print("[INFO] loading YOLO5Face + EmotiEffLib pipeline...")
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

    state = AnalysisState(message="waiting to read video stream")
    frame_index = 0
    last_json_write_time = 0.0
    last_log_time = 0.0
    stop_requested = False

    try:
        while not stop_requested:
            cap = open_video_source(capture_source, is_stream, args)
            if not cap.isOpened():
                state.status = "error"
                state.results = []
                state.frame_index = frame_index
                state.message = "failed to open video stream"
                try:
                    write_state_json(source, state, json_output_path)
                    write_latest_emotion_state(state, latest_emotion_output_path)
                    last_json_write_time = time.monotonic()
                except Exception as exc:
                    print(f"[ERROR] failed to write JSON after open error: {exc}")
                print(f"[WARN] failed to open source, retrying in {reconnect_interval:.1f}s...")
                time.sleep(reconnect_interval)
                continue

            print("[INFO] video source opened.")
            last_tick = time.perf_counter()

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        state.status = "error"
                        state.results = []
                        state.frame_index = frame_index
                        state.message = "failed to read video stream"
                        try:
                            write_state_json(source, state, json_output_path)
                            write_latest_emotion_state(state, latest_emotion_output_path)
                            last_json_write_time = time.monotonic()
                        except Exception as exc:
                            print(f"[ERROR] failed to write JSON after read error: {exc}")
                        print(f"[WARN] video stream disconnected, retrying in {reconnect_interval:.1f}s...")
                        break

                    state.fps, last_tick = update_fps(state.fps, last_tick)

                    if frame_index % sample_step == 0:
                        try:
                            state.results = pipeline.process_frame(frame)
                            state.status = "ok"
                            state.frame_index = frame_index
                            state.message = "no face detected" if not state.results else ""
                        except Exception as exc:
                            print(f"[ERROR] inference failed on frame {frame_index}: {exc}")
                            state.status = "error"
                            state.results = []
                            state.frame_index = frame_index
                            state.message = "emotion inference failed"
                            try:
                                write_state_json(source, state, json_output_path)
                                write_latest_emotion_state(state, latest_emotion_output_path)
                                last_json_write_time = time.monotonic()
                            except Exception as write_exc:
                                print(f"[ERROR] failed to write JSON after inference error: {write_exc}")

                    now = time.monotonic()
                    if now - last_json_write_time >= json_update_interval:
                        try:
                            write_state_json(source, state, json_output_path)
                            write_latest_emotion_state(state, latest_emotion_output_path)
                            last_json_write_time = now
                        except Exception as exc:
                            print(f"[ERROR] failed to write JSON: {exc}")
                            state.status = "error"
                            state.results = []
                            state.message = "failed to write json output"

                    if now - last_log_time >= 1.0:
                        log_runtime_status(source, state, json_output_path)
                        last_log_time = now

                    if args.show:
                        vis_results = state.results if state.status == "ok" else []
                        vis = pipeline.draw_results(frame, vis_results, fps=state.fps)
                        if state.status == "ok" and not state.results:
                            FaceEmotionPipeline.draw_label(vis, 10, 60, "no face detected", color=(0, 215, 255))
                        elif state.status == "error":
                            FaceEmotionPipeline.draw_label(vis, 10, 60, state.message, color=(0, 0, 255))

                        cv2.imshow(window_name, vis)
                        if (cv2.waitKey(1) & 0xFF) == ord("q"):
                            stop_requested = True
                            break

                    frame_index += 1
            finally:
                cap.release()

            if not stop_requested:
                time.sleep(reconnect_interval)
    except KeyboardInterrupt:
        print("[INFO] interrupted by user.")
    finally:
        cv2.destroyAllWindows()
        print("[INFO] stream released.")


if __name__ == "__main__":
    main()
