import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse


def parse_args():
    parser = argparse.ArgumentParser(description="Serve latest emotion JSON and optionally push it to another model API.")
    parser.add_argument("--json-path", type=str, default="outputs/latest_emotion.json")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--push-url", type=str, default="", help="Optional downstream inference-model URL to receive POST payloads.")
    parser.add_argument("--push-interval", type=float, default=0.5)
    parser.add_argument("--stale-after", type=float, default=3.0, help="Seconds after which data is marked stale.")
    return parser.parse_args()


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def add_api_metadata(payload, stale_after):
    data = dict(payload)
    timestamp = float(data.get("timestamp", 0.0) or 0.0)
    age = max(0.0, time.time() - timestamp) if timestamp > 0 else None
    data["api"] = {
        "served_at": time.time(),
        "data_age_seconds": age,
        "stale": bool(age is None or age > stale_after),
    }
    return data


def make_push_payload(payload, stale_after):
    data = add_api_metadata(payload, stale_after)
    main_person = data.get("main_person")
    decision_input = {
        "valid": bool(data.get("valid")),
        "status": data.get("status"),
        "timestamp": data.get("timestamp"),
        "frame_index": data.get("frame_index"),
        "face_count": data.get("face_count"),
        "main_emotion": main_person.get("emotion") if main_person else None,
        "main_confidence": main_person.get("confidence") if main_person else 0.0,
        "main_center_norm": main_person.get("center_norm") if main_person else None,
        "main_area_ratio": main_person.get("area_ratio") if main_person else 0.0,
    }
    return {
        "type": "vision_emotion_update",
        "decision_input": decision_input,
        "vision": data,
    }


def write_json_response(handler, status_code, payload):
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def start_push_loop(json_path, push_url, push_interval, stale_after):
    import requests

    def worker():
        last_frame_index = None
        while True:
            try:
                payload = read_json(json_path)
                frame_index = payload.get("frame_index")
                if frame_index != last_frame_index:
                    requests.post(push_url, json=make_push_payload(payload, stale_after), timeout=1.5)
                    last_frame_index = frame_index
            except Exception as exc:
                print("[WARN] failed to push vision payload: {}".format(exc))
            time.sleep(max(0.1, push_interval))

    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()


def make_handler(json_path, stale_after):
    class VisionResultHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print("[HTTP] " + fmt % args)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"

            if path == "/health":
                write_json_response(
                    self,
                    200,
                    {
                        "status": "ok" if json_path.exists() else "waiting",
                        "json_path": str(json_path),
                        "json_exists": json_path.exists(),
                    },
                )
                return

            if path == "/vision/latest":
                try:
                    payload = add_api_metadata(read_json(json_path), stale_after)
                    write_json_response(self, 200, payload)
                except FileNotFoundError:
                    write_json_response(self, 404, {"detail": "vision json file not found"})
                except ValueError as exc:
                    write_json_response(self, 503, {"detail": "vision json file is not valid json: {}".format(exc)})
                return

            if path == "/vision/decision-input":
                try:
                    payload = read_json(json_path)
                    write_json_response(self, 200, make_push_payload(payload, stale_after)["decision_input"])
                except FileNotFoundError:
                    write_json_response(self, 404, {"detail": "vision json file not found"})
                except ValueError as exc:
                    write_json_response(self, 503, {"detail": "vision json file is not valid json: {}".format(exc)})
                return

            write_json_response(
                self,
                404,
                {
                    "detail": "not found",
                    "available_paths": ["/health", "/vision/latest", "/vision/decision-input"],
                },
            )

    return VisionResultHandler


def main():
    args = parse_args()
    json_path = Path(args.json_path)

    if args.push_url:
        print("[INFO] pushing vision updates to: {}".format(args.push_url))
        start_push_loop(json_path, args.push_url, args.push_interval, args.stale_after)

    print("[INFO] serving json: {}".format(json_path.resolve()))
    print("[INFO] listen: {}:{}".format(args.host, args.port))
    server = HTTPServer((args.host, args.port), make_handler(json_path, args.stale_after))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[INFO] stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
