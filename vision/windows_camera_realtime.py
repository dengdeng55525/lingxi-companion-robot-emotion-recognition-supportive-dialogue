import argparse
import time
from pathlib import Path

import cv2

from pipeline_face_emotion import FaceEmotionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Windows camera realtime test for YOLO5Face + EmotiEffLib.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--face-weights", type=str, default="yolov5-face/weights/yolov5s-face.pt")
    parser.add_argument("--device", type=str, default="", help="Empty means auto-select cuda/cpu.")
    parser.add_argument("--emotion-backend", type=str, default="emotieff", choices=["emotieff"])
    parser.add_argument("--emotieff-model-name", type=str, default="enet_b0_8_best_vgaf")
    parser.add_argument("--emotieff-engine", type=str, default="torch", choices=["torch", "onnx"])
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--face-conf", type=float, default=0.4)
    parser.add_argument("--iou-thres", type=float, default=0.5)
    parser.add_argument("--crop-margin", type=float, default=0.30)
    parser.add_argument("--sample-step", type=int, default=3)
    parser.add_argument("--save-output", type=str, default="")
    return parser.parse_args()


def open_camera(camera_index: int):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if cap.isOpened():
        return cap
    cap.release()
    print("[WARN] CAP_DSHOW failed, retrying default OpenCV backend.")
    return cv2.VideoCapture(camera_index)


def create_writer(output_path: str, fps: float, frame_size):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    print(f"[INFO] saving video to: {path}")
    return writer


def main():
    args = parse_args()
    sample_step = max(1, int(args.sample_step))
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

    cap = open_camera(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {args.camera_index}")

    writer = None
    frame_idx = 0
    last_results = []
    fps = 0.0
    last_time = time.perf_counter()
    window_name = "YOLO5Face + EmotiEffLib"

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] failed to read camera frame.")
                break

            if frame_idx % sample_step == 0:
                last_results = pipeline.process_frame(frame)

            now = time.perf_counter()
            current_fps = 1.0 / max(1e-6, now - last_time)
            fps = current_fps if fps <= 0 else fps * 0.9 + current_fps * 0.1
            last_time = now

            vis = pipeline.draw_results(frame, last_results, fps=fps)

            if writer is None and args.save_output:
                cap_fps = cap.get(cv2.CAP_PROP_FPS)
                if cap_fps <= 1:
                    cap_fps = 25.0
                height, width = vis.shape[:2]
                writer = create_writer(args.save_output, cap_fps, (width, height))
            if writer is not None:
                writer.write(vis)

            cv2.imshow(window_name, vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("[INFO] camera released.")


if __name__ == "__main__":
    main()
