from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

import send_webcam_http as sender


class FakeEncoded:
    def __init__(self, payload=b"jpeg"):
        self.payload = payload

    def tobytes(self):
        return self.payload


class FakeCapture:
    def __init__(self, opened=True, read_results=None):
        self.opened = opened
        self.read_results = list(read_results or [])
        self.set_calls = []
        self.release_count = 0

    def isOpened(self):
        return self.opened

    def read(self):
        if self.read_results:
            return self.read_results.pop(0)
        return False, None

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return True

    def release(self):
        self.release_count += 1


class FakeCv2:
    CAP_DSHOW = 700
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_BUFFERSIZE = 5
    IMWRITE_JPEG_QUALITY = 1

    def __init__(self, captures=None):
        self.captures = list(captures or [])
        self.video_capture_calls = []
        self.resize_calls = []
        self.imencode_calls = []

    def VideoCapture(self, *args):
        self.video_capture_calls.append(args)
        if self.captures:
            return self.captures.pop(0)
        return FakeCapture()

    def resize(self, frame, size):
        self.resize_calls.append((frame, size))
        return ("resized", frame, size)

    def imencode(self, ext, frame, params):
        self.imencode_calls.append((ext, frame, params))
        return True, FakeEncoded()


class FakeResponse:
    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self):
        self.posts = []

    def post(self, *args, **kwargs):
        self.posts.append((args, kwargs))
        return FakeResponse()


def make_args(extra=None):
    argv = ["--server-url", "http://127.0.0.1:6008", "--token", "abcdefgh"]
    if extra:
        argv.extend(extra)
    return sender.parse_args(argv)


class SendWebcamHttpTests(unittest.TestCase):
    def test_camera_url_argument_is_parsed(self):
        url = "http://127.0.0.1:8080/stream?topic=/camera/rgb/image_raw"
        args = make_args(["--camera-url", url])

        self.assertEqual(args.camera_url, url)
        self.assertEqual(args.reconnect_interval, 2.0)

    def test_camera_url_takes_precedence_over_camera_index(self):
        url = "rtsp://example.test/live"
        args = make_args(["--camera-url", url, "--camera-index", "9"])

        capture_source, source_label, is_url = sender.selected_source(args)

        self.assertEqual(capture_source, url)
        self.assertEqual(source_label, url)
        self.assertTrue(is_url)

    def test_url_source_does_not_use_cap_dshow(self):
        url = "http://example.test/stream"
        fake_capture = FakeCapture()
        fake_cv2 = FakeCv2([fake_capture])
        args = make_args(["--camera-url", url, "--backend", "dshow"])

        capture = sender.open_video_capture(args, fake_cv2, os_name="nt")

        self.assertIs(capture, fake_capture)
        self.assertEqual(fake_cv2.video_capture_calls, [(url,)])
        self.assertEqual(fake_capture.set_calls, [(fake_cv2.CAP_PROP_BUFFERSIZE, 1)])

    def test_loopback_url_clears_proxy_env_for_opencv(self):
        env = {
            "HTTP_PROXY": "http://proxy.example:8080",
            "http_proxy": "http://proxy.example:8080",
            "NO_PROXY": "localhost",
        }

        changed = sender.disable_proxy_for_loopback_url(
            "http://127.0.0.1:8080/stream?topic=/camera/rgb/image_raw",
            environ=env,
        )

        self.assertTrue(changed)
        self.assertNotIn("HTTP_PROXY", env)
        self.assertNotIn("http_proxy", env)
        self.assertIn("127.0.0.1", env["NO_PROXY"])

    def test_latest_frame_overwrites_old_frame(self):
        slot = sender.LatestFrameSlot(clock=lambda: 123.0)

        slot.update("old-frame", capture_timestamp=1.0)
        slot.update("new-frame", capture_timestamp=2.0)
        snapshot = slot.get()

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.sequence, 2)
        self.assertEqual(snapshot.frame, "new-frame")
        self.assertEqual(snapshot.capture_timestamp, 2.0)

    def test_same_sequence_is_not_uploaded_twice(self):
        fake_session = FakeSession()
        fake_cv2 = FakeCv2()
        args = make_args()
        snapshot = sender.FrameSnapshot(sequence=7, frame="frame", capture_timestamp=10.0)

        last_sequence, uploaded = sender.upload_snapshot(
            args=args,
            cv2_module=fake_cv2,
            session=fake_session,
            headers={"X-Stream-Token": args.token},
            snapshot=snapshot,
            last_uploaded_sequence=7,
        )

        self.assertEqual(last_sequence, 7)
        self.assertFalse(uploaded)
        self.assertEqual(fake_session.posts, [])
        self.assertEqual(fake_cv2.imencode_calls, [])

    def test_non_ascii_token_is_rejected_before_http_headers(self):
        with self.assertRaisesRegex(ValueError, "ASCII"):
            sender.validate_stream_token("中文token123")

    def test_read_failure_enters_reconnect_flow(self):
        fake_capture = FakeCapture(opened=True, read_results=[(False, None)])
        fake_cv2 = FakeCv2([fake_capture])
        sleeps = []
        logs = []
        args = make_args(["--reconnect-interval", "2.0"])
        reader = sender.LatestFrameCapture(
            args,
            fake_cv2,
            sleep=sleeps.append,
            log=logs.append,
        )

        reader.run(max_reconnects=1)

        self.assertEqual(fake_capture.release_count, 1)
        self.assertEqual(sleeps, [2.0])
        self.assertIn("[WARN] video source disconnected; reconnecting in 2.0s", logs)

    def test_camera_index_remains_compatible(self):
        fake_capture = FakeCapture()
        fake_cv2 = FakeCv2([fake_capture])
        args = make_args(["--camera-index", "3", "--width", "640", "--height", "480"])

        capture_source, source_label, is_url = sender.selected_source(args)
        capture = sender.open_video_capture(args, fake_cv2, os_name="nt")

        self.assertEqual(capture_source, 3)
        self.assertEqual(source_label, "camera:3")
        self.assertFalse(is_url)
        self.assertIs(capture, fake_capture)
        self.assertEqual(fake_cv2.video_capture_calls, [(3, fake_cv2.CAP_DSHOW)])
        self.assertIn((fake_cv2.CAP_PROP_FRAME_WIDTH, 640), fake_capture.set_calls)
        self.assertIn((fake_cv2.CAP_PROP_FRAME_HEIGHT, 480), fake_capture.set_calls)
        self.assertIn((fake_cv2.CAP_PROP_BUFFERSIZE, 1), fake_capture.set_calls)

    def test_autodl_result_is_converted_to_project_emotion_schema(self):
        payload = {
            "schema": "vision.v2",
            "status": "ok",
            "source": "http-camera-upload",
            "frame_index": 12,
            "faces": {
                "stale": False,
                "items": [
                    {
                        "emotion": "happy",
                        "confidence": 0.91,
                        "probabilities": {"happy": 0.91, "sad": 0.02, "angry": 0.01, "neutral": 0.06},
                        "bbox": [10, 20, 110, 220],
                    }
                ],
            },
            "objects": {"stale": False, "items": []},
            "performance": {"fps": 8.5},
        }

        converted = sender.convert_autodl_result(payload)

        self.assertEqual(converted["status"], "ok")
        self.assertEqual(converted["source"], "http-camera-upload")
        self.assertEqual(converted["frame_index"], 12)
        self.assertEqual(converted["face_count"], 1)
        self.assertTrue(converted["valid"])
        self.assertEqual(converted["main_person"]["emotion"], "happy")
        self.assertEqual(converted["main_person"]["bbox"], [10, 20, 110, 220])

    def test_autodl_result_files_feed_latest_emotion_file(self):
        payload = {
            "status": "ok",
            "frame_index": 13,
            "faces": {
                "stale": False,
                "items": [
                    {
                        "emotion": "angry",
                        "confidence": 0.82,
                        "probabilities": {"angry": 0.82, "neutral": 0.18},
                        "bbox": [0, 0, 50, 60],
                    }
                ],
            },
            "objects": {
                "stale": False,
                "items": [
                    {
                        "class": "bench",
                        "confidence": 0.57,
                        "bbox": [128, 98, 512, 383],
                    }
                ],
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_output = Path(tmp_dir) / "outputs" / "latest_emotion.json"
            latest_output = Path(tmp_dir) / ".ros" / "latest_emotion.json"

            sender.write_result_files(payload, json_output, latest_output)

            full_payload = json.loads(json_output.read_text(encoding="utf-8"))
            latest_payload = json.loads(latest_output.read_text(encoding="utf-8"))

        self.assertEqual(full_payload["face_count"], 1)
        self.assertEqual(full_payload["main_person"]["emotion"], "angry")
        self.assertEqual(latest_payload["face_count"], 1)
        self.assertEqual(latest_payload["dominant_emotion"], "angry")
        self.assertGreater(latest_payload["confidence"], 0.8)
        self.assertEqual(latest_payload["object_count"], 1)
        self.assertEqual(latest_payload["objects"][0]["class"], "bench")


if __name__ == "__main__":
    unittest.main()
