from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from json_writer import atomic_write_json
from simple_emotion_writer import LATEST_EMOTION_PATH, build_latest_emotion_payload


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a local webcam and upload JPEG frames to AutoDL over HTTP."
    )
    parser.add_argument("--server-url", required=True, help="AutoDL public HTTP service base URL.")
    parser.add_argument("--token", default=os.environ.get("ROBOTVISION_STREAM_TOKEN", ""))
    parser.add_argument("--camera-url", default="", help="HTTP/MJPEG or RTSP camera stream URL. Takes precedence over --camera-index.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--upload-fps", type=float, default=5.0)
    parser.add_argument("--jpeg-quality", type=int, default=78)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--result-hz", type=float, default=0.5)
    parser.add_argument("--json-output", type=str, default="outputs/latest_emotion.json")
    parser.add_argument("--latest-emotion-output", type=str, default=str(LATEST_EMOTION_PATH))
    parser.add_argument("--reconnect-interval", type=float, default=2.0)
    parser.add_argument("--show-local", action="store_true")
    parser.add_argument("--backend", choices=["auto", "dshow"], default="dshow")
    return parser.parse_args(argv)


@dataclass(frozen=True)
class FrameSnapshot:
    sequence: int
    frame: Any
    capture_timestamp: float


class LatestFrameSlot:
    def __init__(self, clock: Callable[[], float] = time.time):
        self._clock = clock
        self._lock = threading.Lock()
        self._sequence = 0
        self._frame = None
        self._capture_timestamp = 0.0

    def update(self, frame: Any, capture_timestamp: float | None = None) -> FrameSnapshot:
        timestamp = self._clock() if capture_timestamp is None else float(capture_timestamp)
        with self._lock:
            self._sequence += 1
            self._frame = frame
            self._capture_timestamp = timestamp
            return FrameSnapshot(self._sequence, self._frame, self._capture_timestamp)

    def get(self) -> FrameSnapshot | None:
        with self._lock:
            if self._frame is None:
                return None
            return FrameSnapshot(self._sequence, self._frame, self._capture_timestamp)


def frame_endpoint(server_url: str) -> str:
    return f"{server_url.rstrip('/')}/api/frame"


def result_endpoint(server_url: str) -> str:
    return f"{server_url.rstrip('/')}/api/latest-result"


def result_summary(payload: dict) -> str:
    faces = payload.get("faces", {}).get("items", [])
    objects = payload.get("objects", {}).get("items", [])
    face_labels = ",".join(item.get("emotion", "face") for item in faces) or "-"
    object_labels = ",".join(item.get("class", "object") for item in objects) or "-"
    latency = payload.get("transport", {}).get("end_to_end_ms")
    latency_text = "--" if latency is None else f"{latency:.0f}ms"
    return (
        f"status={payload.get('status')} faces={face_labels} "
        f"objects={object_labels} latency={latency_text}"
    )


def clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def bbox_to_xyxy(value: Any) -> list[int]:
    if isinstance(value, dict):
        if all(key in value for key in ("x1", "y1", "x2", "y2")):
            return [int(value[key]) for key in ("x1", "y1", "x2", "y2")]
        if all(key in value for key in ("left", "top", "right", "bottom")):
            return [int(value[key]) for key in ("left", "top", "right", "bottom")]
        if all(key in value for key in ("x", "y", "width", "height")):
            x1 = int(value["x"])
            y1 = int(value["y"])
            return [x1, y1, x1 + int(value["width"]), y1 + int(value["height"])]

    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return [int(float(item)) for item in value[:4]]

    return [0, 0, 0, 0]


def bbox_area(bbox: list[int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def first_present(mapping: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def extract_items(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    container = payload.get(key)
    if isinstance(container, dict):
        items = container.get("items", [])
    elif isinstance(container, list):
        items = container
    else:
        items = []
    return [item for item in items if isinstance(item, dict)]


def normalize_face_item(item: dict[str, Any]) -> dict[str, Any]:
    probabilities = first_present(
        item,
        ("probabilities", "scores", "emotion_scores", "emotion_probabilities"),
        {},
    )
    if not isinstance(probabilities, dict):
        probabilities = {}

    raw_emotion = first_present(
        item,
        (
            "emotion",
            "dominant_emotion",
            "expression",
            "top_probability_label",
            "label",
            "class",
        ),
        "unknown",
    )
    confidence = first_present(
        item,
        (
            "confidence",
            "emotion_confidence",
            "top_probability",
            "score",
            "probability",
        ),
        0.0,
    )
    bbox = bbox_to_xyxy(first_present(item, ("bbox", "xyxy", "box", "rect"), [0, 0, 0, 0]))

    return {
        "emotion": str(raw_emotion or "unknown").strip().lower() or "unknown",
        "confidence": clamp01(confidence),
        "model_emotion": str(item.get("model_emotion", raw_emotion or "unknown")),
        "top_probability_label": str(item.get("top_probability_label", raw_emotion or "unknown")),
        "top_probability": clamp01(item.get("top_probability", confidence)),
        "probabilities": {str(key): clamp01(value) for key, value in probabilities.items()},
        "bbox": bbox,
        "area": bbox_area(bbox),
        "autodl": item,
    }


def convert_autodl_result(payload: dict[str, Any]) -> dict[str, Any]:
    faces = [normalize_face_item(item) for item in extract_items(payload, "faces")]
    faces.sort(key=lambda item: item["area"], reverse=True)

    face_container = payload.get("faces") if isinstance(payload.get("faces"), dict) else {}
    faces_stale = bool(face_container.get("stale")) if isinstance(face_container, dict) else False
    remote_status = str(payload.get("status", "") or "").lower()
    local_status = "ok" if remote_status in {"ok", "ready", "running", "success"} and not faces_stale else "error"
    if faces and not faces_stale and remote_status not in {"error", "failed", "waiting"}:
        local_status = "ok"

    return {
        "status": local_status,
        "timestamp": float(time.time()),
        "source": str(payload.get("source") or "autodl-http-camera-upload"),
        "frame_index": int(payload.get("frame_index") or 0),
        "fps": float((payload.get("performance") or {}).get("fps") or 0.0),
        "face_count": len(faces) if not faces_stale else 0,
        "valid": bool(local_status == "ok" and faces and not faces_stale),
        "main_person": faces[0] if faces and not faces_stale else None,
        "faces": faces if not faces_stale else [],
        "objects": extract_items(payload, "objects"),
        "message": str(payload.get("message") or ""),
        "autodl": payload,
    }


def write_result_files(remote_payload: dict[str, Any], json_output: str | Path, latest_emotion_output: str | Path) -> tuple[Path, Path]:
    vision_payload = convert_autodl_result(remote_payload)
    vision_path = atomic_write_json(json_output, vision_payload)
    latest_payload = build_latest_emotion_payload(vision_payload.get("faces", []))
    objects = vision_payload.get("objects", [])
    latest_payload["object_count"] = len(objects)
    latest_payload["objects"] = objects
    latest_path = atomic_write_json(latest_emotion_output, latest_payload)
    return vision_path, latest_path


def validate_stream_token(token: str) -> None:
    if len(token) < 8:
        raise ValueError("Set --token or ROBOTVISION_STREAM_TOKEN to at least 8 characters.")
    try:
        token.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "--token / ROBOTVISION_STREAM_TOKEN must contain only ASCII characters "
            "because it is sent in the X-Stream-Token HTTP header."
        ) from exc
    if any(char in token for char in "\r\n"):
        raise ValueError("--token / ROBOTVISION_STREAM_TOKEN must not contain newline characters.")


def is_loopback_url(url: str) -> bool:
    host = urlparse(str(url)).hostname
    if not host:
        return False
    return host == "localhost" or host == "::1" or host.startswith("127.")


def disable_proxy_for_loopback_url(url: str, environ: dict[str, str] | None = None) -> bool:
    if not is_loopback_url(url):
        return False
    env = os.environ if environ is None else environ
    for name in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        env.pop(name, None)
    entries = ["localhost", "127.0.0.1", "127.0.0.0/8", "::1"]
    existing = env.get("NO_PROXY") or env.get("no_proxy") or ""
    merged = []
    for item in existing.split(",") + entries:
        item = item.strip()
        if item and item not in merged:
            merged.append(item)
    no_proxy_value = ",".join(merged)
    env["NO_PROXY"] = no_proxy_value
    env["no_proxy"] = no_proxy_value
    return True


def selected_source(args: argparse.Namespace) -> tuple[str | int, str, bool]:
    camera_url = str(args.camera_url or "").strip()
    if camera_url:
        return camera_url, camera_url, True
    camera_index = int(args.camera_index)
    return camera_index, f"camera:{camera_index}", False


def open_video_capture(args: argparse.Namespace, cv2_module: Any, os_name: str | None = None) -> Any:
    capture_source, _source_label, is_url = selected_source(args)
    if is_url:
        disable_proxy_for_loopback_url(str(capture_source))
        capture = cv2_module.VideoCapture(capture_source)
    else:
        current_os_name = os.name if os_name is None else os_name
        backend = cv2_module.CAP_DSHOW if args.backend == "dshow" and current_os_name == "nt" else 0
        capture = cv2_module.VideoCapture(capture_source, backend)
        if int(args.width) > 0:
            capture.set(cv2_module.CAP_PROP_FRAME_WIDTH, int(args.width))
        if int(args.height) > 0:
            capture.set(cv2_module.CAP_PROP_FRAME_HEIGHT, int(args.height))

    capture.set(cv2_module.CAP_PROP_BUFFERSIZE, 1)
    return capture


class LatestFrameCapture:
    def __init__(
        self,
        args: argparse.Namespace,
        cv2_module: Any,
        *,
        open_capture: Callable[[argparse.Namespace, Any], Any] = open_video_capture,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
        log: Callable[[str], None] = print,
    ):
        self.args = args
        self.cv2 = cv2_module
        self.open_capture = open_capture
        self.sleep = sleep
        self.log = log
        self.slot = LatestFrameSlot(clock=clock)
        self.stop_event = threading.Event()
        self._capture_lock = threading.Lock()
        self._capture = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self.run, name="latest-frame-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self._release_current_capture()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, float(self.args.reconnect_interval) + 1.0))

    def get_latest(self) -> FrameSnapshot | None:
        return self.slot.get()

    def run(self, max_reconnects: int | None = None) -> None:
        reconnects = 0
        try:
            while not self.stop_event.is_set():
                capture = self.open_capture(self.args, self.cv2)
                with self._capture_lock:
                    self._capture = capture

                if not capture.isOpened():
                    self._release_capture(capture)
                    reconnects = self._sleep_before_reconnect(reconnects)
                    if max_reconnects is not None and reconnects >= max_reconnects:
                        break
                    continue

                while not self.stop_event.is_set():
                    success, frame = capture.read()
                    if not success or frame is None:
                        self._release_capture(capture)
                        reconnects = self._sleep_before_reconnect(reconnects)
                        break
                    self.slot.update(frame)

                if max_reconnects is not None and reconnects >= max_reconnects:
                    break
        finally:
            self._release_current_capture()

    def _sleep_before_reconnect(self, reconnects: int) -> int:
        reconnect_interval = float(self.args.reconnect_interval)
        self.log(f"[WARN] video source disconnected; reconnecting in {reconnect_interval:.1f}s")
        if not self.stop_event.is_set():
            self.sleep(reconnect_interval)
        return reconnects + 1

    def _release_current_capture(self) -> None:
        with self._capture_lock:
            capture = self._capture
            self._capture = None
        self._release_capture(capture)

    def _release_capture(self, capture: Any) -> None:
        if capture is None:
            return
        try:
            capture.release()
        except Exception:
            pass
        with self._capture_lock:
            if self._capture is capture:
                self._capture = None


def resize_for_upload(frame: Any, args: argparse.Namespace, cv2_module: Any) -> Any:
    width = int(args.width)
    height = int(args.height)
    if width > 0 and height > 0:
        return cv2_module.resize(frame, (width, height))
    return frame


def encode_jpeg(frame: Any, args: argparse.Namespace, cv2_module: Any) -> bytes | None:
    encoded_ok, encoded = cv2_module.imencode(
        ".jpg",
        frame,
        [int(cv2_module.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
    )
    if not encoded_ok:
        return None
    return encoded.tobytes()


def upload_snapshot(
    *,
    args: argparse.Namespace,
    cv2_module: Any,
    session: Any,
    headers: dict[str, str],
    snapshot: FrameSnapshot | None,
    last_uploaded_sequence: int,
    request_exception: type[BaseException] = Exception,
    log: Callable[[str], None] = print,
) -> tuple[int, bool]:
    if snapshot is None or snapshot.sequence == last_uploaded_sequence:
        return last_uploaded_sequence, False

    upload_frame = resize_for_upload(snapshot.frame, args, cv2_module)
    encoded = encode_jpeg(upload_frame, args, cv2_module)
    if encoded is None:
        return snapshot.sequence, False

    try:
        response = session.post(
            frame_endpoint(args.server_url),
            data=encoded,
            headers={
                **headers,
                "Content-Type": "image/jpeg",
                "X-Capture-Timestamp": f"{snapshot.capture_timestamp:.6f}",
            },
            timeout=args.timeout,
        )
        response.raise_for_status()
        return snapshot.sequence, True
    except (request_exception, UnicodeEncodeError) as exc:
        log(f"[WARN] upload failed: {exc}")
        return snapshot.sequence, False


def main(argv=None) -> int:
    args = parse_args(argv)
    validate_stream_token(args.token)
    if args.upload_fps <= 0:
        raise ValueError("--upload-fps must be greater than zero.")
    if not 20 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 20 and 100.")
    if args.reconnect_interval <= 0:
        raise ValueError("--reconnect-interval must be greater than zero.")

    import cv2
    import requests

    _capture_source, source_label, _is_url = selected_source(args)
    capture_reader = LatestFrameCapture(args, cv2)
    session = requests.Session()
    session.trust_env = False
    headers = {"X-Stream-Token": args.token}
    json_output_path = Path(args.json_output)
    latest_emotion_output_path = Path(args.latest_emotion_output)
    upload_interval = 1.0 / args.upload_fps
    result_interval = 1.0 / args.result_hz if args.result_hz > 0 else None
    next_upload = time.perf_counter()
    next_result = time.perf_counter()
    last_uploaded_sequence = 0
    uploaded = 0
    failed = 0
    print(f"[INFO] source={source_label}")
    print(f"[INFO] target={frame_endpoint(args.server_url)}")
    print(f"[INFO] result json={json_output_path.resolve()}")
    print(f"[INFO] latest emotion json={latest_emotion_output_path}")
    print(f"[INFO] upload={args.upload_fps:g} FPS jpeg_quality={args.jpeg_quality}")

    capture_reader.start()
    try:
        while True:
            now = time.perf_counter()
            snapshot = capture_reader.get_latest()
            if now >= next_upload:
                previous_sequence = last_uploaded_sequence
                last_uploaded_sequence, success = upload_snapshot(
                    args=args,
                    cv2_module=cv2,
                    session=session,
                    headers=headers,
                    snapshot=snapshot,
                    last_uploaded_sequence=last_uploaded_sequence,
                    request_exception=requests.RequestException,
                )
                if last_uploaded_sequence != previous_sequence and success:
                    uploaded += 1
                    next_upload = max(next_upload + upload_interval, now + 0.001)
                elif last_uploaded_sequence != previous_sequence:
                    failed += 1
                    next_upload = max(next_upload + upload_interval, now + 0.001)

            if result_interval is not None and now >= next_result:
                try:
                    response = session.get(
                        result_endpoint(args.server_url),
                        headers=headers,
                        timeout=args.timeout,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    write_result_files(payload, json_output_path, latest_emotion_output_path)
                    print(
                        f"[INFO] sent={uploaded} failed={failed} "
                        f"{result_summary(payload)}"
                    )
                except (requests.RequestException, UnicodeEncodeError, ValueError, OSError) as exc:
                    print(f"[WARN] result request failed: {exc}")
                next_result = now + result_interval

            if args.show_local and snapshot is not None:
                cv2.imshow("Local camera upload", snapshot.frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            time.sleep(0.002)
    except KeyboardInterrupt:
        print("\n[INFO] sender interrupted")
    finally:
        capture_reader.stop()
        if args.show_local:
            cv2.destroyAllWindows()
        session.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(2)
