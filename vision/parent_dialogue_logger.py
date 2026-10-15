from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_LOG_FILE = str(Path.home() / ".ros" / "ai_companion_dialogue.jsonl")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log ROS dialogue topics for the parent dashboard.")
    parser.add_argument("--user-topic", type=str, default="/iat_text")
    parser.add_argument("--assistant-topic", type=str, default="/ai_reply")
    parser.add_argument("--log-file", type=str, default=DEFAULT_LOG_FILE)
    parser.add_argument("--max-text-length", type=int, default=500)
    return parser.parse_args(argv)


def iso_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")


def normalize_role(role: Any) -> str:
    value = str(role or "").strip().lower()
    if value in {"user", "child", "kid"}:
        return "child"
    if value in {"assistant", "robot", "ai"}:
        return "robot"
    return value or "unknown"


def make_dialogue_record(
    role: str,
    text: str,
    *,
    timestamp: float | None = None,
    source_topic: str = "",
    max_text_length: int = 500,
) -> dict[str, Any] | None:
    clean_text = str(text or "").strip()
    if not clean_text:
        return None
    if max_text_length > 0:
        clean_text = clean_text[:max_text_length]
    ts = float(timestamp if timestamp is not None else time.time())
    return {
        "timestamp": ts,
        "iso_time": iso_time(ts),
        "role": normalize_role(role),
        "text": clean_text,
        "source_topic": source_topic,
    }


class JsonlDialogueLogger:
    def __init__(self, log_file: str | Path, *, clock=time.time, max_text_length: int = 500):
        self.log_file = Path(log_file).expanduser()
        self.clock = clock
        self.max_text_length = max_text_length

    def append(self, role: str, text: str, *, source_topic: str = "") -> bool:
        record = make_dialogue_record(
            role,
            text,
            timestamp=self.clock(),
            source_topic=source_topic,
            max_text_length=self.max_text_length,
        )
        if record is None:
            return False
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import rospy
    from std_msgs.msg import String

    logger = JsonlDialogueLogger(args.log_file, max_text_length=args.max_text_length)

    def on_user(msg):
        if logger.append("child", msg.data, source_topic=args.user_topic):
            rospy.loginfo("Parent dialogue log child: %s", msg.data)

    def on_assistant(msg):
        if logger.append("robot", msg.data, source_topic=args.assistant_topic):
            rospy.loginfo("Parent dialogue log robot: %s", msg.data)

    rospy.init_node("parent_dialogue_logger")
    rospy.Subscriber(args.user_topic, String, on_user, queue_size=20)
    rospy.Subscriber(args.assistant_topic, String, on_assistant, queue_size=20)
    rospy.loginfo(
        "Parent dialogue logger ready: user_topic=%s assistant_topic=%s log_file=%s",
        args.user_topic,
        args.assistant_topic,
        args.log_file,
    )
    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
