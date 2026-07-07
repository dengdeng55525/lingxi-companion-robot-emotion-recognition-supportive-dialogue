from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from json_writer import atomic_write_json
from simple_emotion_writer import LATEST_EMOTION_PATH, write_latest_emotion


EMOTION_ORDER = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "angry"]


@dataclass
class FrameSnapshot:
    seq: int = 0
    frame: Any = None
    received_at: float = 0.0
    message_stamp: float = 0.0


class LatestFrameBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot = FrameSnapshot()

    def update(self, frame, message_stamp: float) -> None:
        with self._lock:
            self._snapshot = FrameSnapshot(
                seq=self._snapshot.seq + 1,
                frame=frame,
                received_at=time.time(),
                message_stamp=float(message_stamp or 0.0),
            )

    def snapshot(self) -> FrameSnapshot:
        with self._lock:
            frame = None if self._snapshot.frame is None else self._snapshot.frame.copy()
            return FrameSnapshot(
                seq=self._snapshot.seq,
                frame=frame,
                received_at=self._snapshot.received_at,
                message_stamp=self._snapshot.message_stamp,
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read a ROS Image topic in a Linux VM, run YOLO5Face + EmotiEffLib, and keep a JSON result file updated."
    )
    parser.add_argument("--ros-image-topic", type=str, default="/camera/rgb/image_raw")
    parser.add_argument("--ros-master-uri", type=str, default="", help="Optional ROS_MASTER_URI override, e.g. http://192.168.79.91:11311.")
    parser.add_argument("--ros-ip", type=str, default="", help="Optional ROS_IP override for this VM.")
    parser.add_argument("--ros-hostname", type=str, default="", help="Optional ROS_HOSTNAME override for this VM.")
    parser.add_argument("--face-weights", type=str, default="yolov5-face/weights/yolov5s-face.pt")
    parser.add_argument("--device", type=str, default="", help="Empty or auto means auto-select cuda/cpu.")
    parser.add_argument("--emotion-backend", type=str, default="emotieff", choices=["emotieff"])
    parser.add_argument("--emotieff-model-name", type=str, default="enet_b0_8_best_vgaf")
    parser.add_argument("--emotieff-engine", type=str, default="torch", choices=["torch", "onnx"])
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--face-conf", type=float, default=0.4)
    parser.add_argument("--iou-thres", type=float, default=0.5)
    parser.add_argument("--crop-margin", type=float, default=0.30)
    parser.add_argument("--sample-step", type=int, default=3, help="Run inference once every N received ROS frames.")
    parser.add_argument("--max-infer-fps", type=float, default=8.0, help="Upper limit for inference work; useful when running on CPU.")
    parser.add_argument("--json-output", type=str, default="outputs/latest_emotion.json")
    parser.add_argument("--json-update-interval", type=float, default=0.5)
    parser.add_argument("--latest-emotion-output", type=str, default=str(LATEST_EMOTION_PATH))
    parser.add_argument("--stale-after", type=float, default=3.0, help="Seconds without ROS frames before JSON is marked stale.")
    parser.add_argument("--queue-size", type=int, default=1)
    parser.add_argument("--buff-size", type=int, default=2**24)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def apply_ros_env(args) -> None:
    overrides = {
        "ROS_MASTER_URI": args.ros_master_uri,
        "ROS_IP": args.ros_ip,
        "ROS_HOSTNAME": args.ros_hostname,
    }
    for name, value in overrides.items():
        value = str(value or "").strip()
        if value:
            os.environ[name] = value


def bbox_area(bbox: list[int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def ordered_probabilities(probabilities: dict[str, Any]) -> dict[str, float]:
    ordered: dict[str, float] = {}
    for label in EMOTION_ORDER:
        if label in probabilities:
            ordered[label] = float(probabilities[label])
    for label, value in probabilities.items():
        if label not in ordered:
            ordered[str(label)] = float(value)
    return ordered


def serialize_face(result: dict[str, Any], frame_shape) -> dict[str, Any]:
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
        "probabilities": ordered_probabilities(result.get("probabilities", {})),
        "bbox": bbox,
        "crop_bbox": [int(v) for v in result.get("crop_bbox", bbox)],
        "center": [float(cx), float(cy)],
        "center_norm": [float(cx / max(1, frame_w)), float(cy / max(1, frame_h))],
        "area": int(area),
        "area_ratio": float(area / float(frame_area)),
    }


def build_payload(
    status: str,
    source: str,
    frame_index: int,
    frame,
    results: list[dict[str, Any]],
    fps: float,
    args,
    snapshot: FrameSnapshot | None = None,
    message: str = "",
) -> dict[str, Any]:
    frame_h, frame_w = frame.shape[:2] if frame is not None else (0, 0)
    faces = []
    if results and frame is not None:
        faces = [serialize_face(item, frame.shape) for item in results]
        faces.sort(key=lambda item: item["area"], reverse=True)

    payload: dict[str, Any] = {
        "status": status,
        "timestamp": time.time(),
        "source": source,
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
        "ros": {
            "topic": args.ros_image_topic,
            "master_uri": os.environ.get("ROS_MASTER_URI", ""),
            "ros_ip": os.environ.get("ROS_IP", ""),
            "ros_hostname": os.environ.get("ROS_HOSTNAME", ""),
            "message_stamp": float(snapshot.message_stamp) if snapshot else 0.0,
            "received_at": float(snapshot.received_at) if snapshot else 0.0,
        },
    }

    if message:
        payload["message"] = str(message)
    elif status == "ok" and not faces:
        payload["message"] = "no face detected"

    return payload


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def write_latest_emotion_state(results: list[dict[str, Any]], output_path: Path) -> None:
    write_latest_emotion(results, output_path)


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


def main():
    args = parse_args()
    apply_ros_env(args)

    try:
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image
    except Exception as exc:
        raise RuntimeError(
            "Failed to import ROS Python packages. Source /opt/ros/noetic/setup.bash "
            "before running this script and make sure ros-noetic-cv-bridge is installed."
        ) from exc

    source = f"ros:{args.ros_image_topic}"
    output_path = Path(args.json_output)
    latest_emotion_output_path = Path(args.latest_emotion_output)
    sample_step = max(1, int(args.sample_step))
    json_update_interval = max(0.0, float(args.json_update_interval))
    stale_after = max(0.5, float(args.stale_after))
    max_infer_fps = max(0.0, float(args.max_infer_fps))
    min_infer_interval = 0.0 if max_infer_fps <= 0 else 1.0 / max_infer_fps

    print(f"[INFO] ROS_MASTER_URI={os.environ.get('ROS_MASTER_URI', '')}")
    print(f"[INFO] ROS_IP={os.environ.get('ROS_IP', '')}")
    print(f"[INFO] ROS_HOSTNAME={os.environ.get('ROS_HOSTNAME', '')}")
    print(f"[INFO] source: {source}")
    print(f"[INFO] json output: {output_path.resolve()}")
    print(f"[INFO] latest emotion output: {latest_emotion_output_path}")
    print(f"[INFO] sample step: {sample_step}")
    print(f"[INFO] max infer fps: {max_infer_fps if max_infer_fps > 0 else 'unlimited'}")
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

    rospy.init_node("vm_ros_emotion_to_json", anonymous=True)
    bridge = CvBridge()
    buffer = LatestFrameBuffer()

    def on_image(msg):
        try:
            frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "failed to convert ROS image: %s", exc)
            return
        stamp = msg.header.stamp.to_sec() if msg.header and msg.header.stamp else 0.0
        buffer.update(frame, stamp)

    subscriber = rospy.Subscriber(
        args.ros_image_topic,
        Image,
        on_image,
        queue_size=max(1, int(args.queue_size)),
        buff_size=max(1024, int(args.buff_size)),
    )
    print(f"[INFO] subscribed: {subscriber.resolved_name}")

    frame_index = 0
    last_processed_seq = 0
    last_results: list[dict[str, Any]] = []
    fps = 0.0
    last_tick = time.perf_counter()
    last_json_write = 0.0
    last_log = 0.0
    last_infer_time = 0.0
    window_name = "YOLO5Face + EmotiEffLib ROS"

    try:
        while not rospy.is_shutdown():
            snapshot = buffer.snapshot()
            now_wall = time.time()
            now_mono = time.monotonic()

            if snapshot.frame is None:
                if now_mono - last_json_write >= json_update_interval:
                    payload = build_payload(
                        "error",
                        source,
                        frame_index,
                        None,
                        [],
                        fps,
                        args,
                        snapshot,
                        "waiting for ROS image topic",
                    )
                    write_payload(output_path, payload)
                    write_latest_emotion_state([], latest_emotion_output_path)
                    last_json_write = now_mono
                time.sleep(0.05)
                continue

            if now_wall - snapshot.received_at > stale_after:
                if now_mono - last_json_write >= json_update_interval:
                    payload = build_payload(
                        "error",
                        source,
                        frame_index,
                        snapshot.frame,
                        last_results,
                        fps,
                        args,
                        snapshot,
                        "ROS image topic is stale",
                    )
                    write_payload(output_path, payload)
                    write_latest_emotion_state([], latest_emotion_output_path)
                    last_json_write = now_mono
                time.sleep(0.05)
                continue

            if snapshot.seq == last_processed_seq:
                time.sleep(0.005)
                continue

            last_processed_seq = snapshot.seq
            should_infer = snapshot.seq % sample_step == 0 and now_mono - last_infer_time >= min_infer_interval
            status = "ok"
            message = ""

            if should_infer:
                try:
                    last_results = pipeline.process_frame(snapshot.frame)
                    fps, last_tick = update_fps(fps, last_tick)
                    last_infer_time = now_mono
                    if not last_results:
                        message = "no face detected"
                except Exception as exc:
                    status = "error"
                    last_results = []
                    message = f"inference failed: {exc}"
                    print(f"[ERROR] {message}")

            if now_mono - last_json_write >= json_update_interval:
                payload = build_payload(
                    status,
                    source,
                    frame_index,
                    snapshot.frame,
                    last_results,
                    fps,
                    args,
                    snapshot,
                    message,
                )
                write_payload(output_path, payload)
                write_latest_emotion_state(last_results if status == "ok" else [], latest_emotion_output_path)
                last_json_write = now_mono

            if now_mono - last_log >= 1.0:
                age = now_wall - snapshot.received_at
                print(
                    f"[INFO] frame={frame_index} ros_seq={snapshot.seq} fps={fps:.1f} "
                    f"age={age:.2f}s faces={len(last_results)} main={describe_main_person(last_results)} "
                    f"json={output_path}"
                )
                last_log = now_mono

            if args.show:
                vis = pipeline.draw_results(snapshot.frame, last_results, fps=fps)
                if not last_results:
                    FaceEmotionPipeline.draw_label(vis, 10, 60, "no face detected", color=(0, 215, 255))
                cv2.imshow(window_name, vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    rospy.signal_shutdown("user requested quit")
                    break

            frame_index += 1
    except KeyboardInterrupt:
        print("[INFO] interrupted by user.")
    finally:
        cv2.destroyAllWindows()
        print("[INFO] released.")


if __name__ == "__main__":
    main()
