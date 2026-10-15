from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import ProxyHandler, Request, build_opener, urlopen


SCHEMA = "parent-dashboard.v1"
DEFAULT_DB_PATH = "outputs/parent_dashboard.sqlite3"
DEFAULT_DIALOGUE_LOG = str(Path.home() / ".ros" / "ai_companion_dialogue.jsonl")
DEFAULT_LEARNING_LOG = str(Path.home() / ".ros" / "ai_learning_events.jsonl")
DEFAULT_VISION_URL = "http://127.0.0.1:8090/vision/latest"

NEGATIVE_EMOTIONS = {"sad", "angry", "fear", "disgust"}
POSITIVE_EMOTIONS = {"happy", "surprise"}
EMOTION_ATTENTION_ORDER = ["angry", "sad", "disgust", "fear", "happy", "surprise", "neutral"]
EMOTION_LABELS = {
    "happy": "开心",
    "sad": "低落",
    "angry": "生气",
    "fear": "紧张",
    "disgust": "抗拒",
    "surprise": "惊讶",
    "neutral": "平静",
    "unknown": "未知",
}
DANGEROUS_OBJECT_DISPLAY_NAMES = {
    "knife": "刀具",
    "scissors": "剪刀",
    "fork": "叉子",
    "baseball bat": "球棒",
    "hammer": "锤子",
    "cutter": "美工刀",
    "blade": "刀片",
    "saw": "锯子",
    "axe": "斧头",
    "gun": "枪形物品",
    "pistol": "枪形物品",
    "刀": "刀具",
    "刀具": "刀具",
    "菜刀": "菜刀",
    "剪刀": "剪刀",
    "锤子": "锤子",
}
DANGEROUS_OBJECT_KEYWORDS = (
    ("knife", "刀具"),
    ("scissors", "剪刀"),
    ("cutter", "美工刀"),
    ("blade", "刀片"),
    ("hammer", "锤子"),
    ("saw", "锯子"),
    ("axe", "斧头"),
    ("刀", "刀具"),
    ("剪刀", "剪刀"),
    ("锤", "锤子"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve parent dashboard history collected from the latest vision API.")
    parser.add_argument("--vision-url", type=str, default=DEFAULT_VISION_URL)
    parser.add_argument("--db", type=str, default=DEFAULT_DB_PATH)
    parser.add_argument("--dialogue-log", type=str, default=DEFAULT_DIALOGUE_LOG)
    parser.add_argument("--learning-log", type=str, default=DEFAULT_LEARNING_LOG)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--fetch-timeout", type=float, default=1.5)
    parser.add_argument("--no-mock-on-empty", action="store_true")
    return parser.parse_args(argv)


def local_date_key(timestamp: float | None = None) -> str:
    return datetime.fromtimestamp(timestamp or time.time()).strftime("%Y-%m-%d")


def iso_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")


def time_label(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%H:%M")


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def init_db(db_path: str | Path) -> Path:
    path = Path(db_path).expanduser()
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS emotion_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sampled_at REAL NOT NULL,
                source_timestamp REAL,
                date_key TEXT NOT NULL,
                iso_time TEXT NOT NULL,
                status TEXT,
                valid INTEGER NOT NULL,
                face_count INTEGER NOT NULL,
                dominant_emotion TEXT NOT NULL,
                confidence REAL NOT NULL,
                scores_json TEXT NOT NULL,
                object_count INTEGER NOT NULL,
                objects_json TEXT NOT NULL,
                raw_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learned_objects (
                date_key TEXT NOT NULL,
                class_name TEXT NOT NULL,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                count INTEGER NOT NULL,
                max_confidence REAL NOT NULL,
                PRIMARY KEY (date_key, class_name)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_events (
                event_id TEXT PRIMARY KEY,
                date_key TEXT NOT NULL,
                timestamp REAL NOT NULL,
                iso_time TEXT NOT NULL,
                user_text TEXT NOT NULL,
                reply TEXT NOT NULL,
                intent TEXT NOT NULL,
                objects_json TEXT NOT NULL,
                raw_json TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_emotion_samples_date ON emotion_samples(date_key, sampled_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_learning_events_date ON learning_events(date_key, timestamp)")
        conn.commit()
    return path


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_emotion(value: Any) -> str:
    label = str(value or "unknown").strip().lower()
    if label in {"nature", "natural"}:
        return "neutral"
    return label or "unknown"


def normalize_object_name(value: Any) -> str:
    name = str(value or "").strip()
    return name


def normalize_object_key(value: Any) -> str:
    return normalize_object_name(value).lower().replace("_", " ")


def dangerous_object_display_name(class_name: Any) -> str:
    key = normalize_object_key(class_name)
    if not key:
        return ""
    if key in DANGEROUS_OBJECT_DISPLAY_NAMES:
        return DANGEROUS_OBJECT_DISPLAY_NAMES[key]
    for keyword, display_name in DANGEROUS_OBJECT_KEYWORDS:
        if keyword in key:
            return display_name
    return ""


def extract_dangerous_objects(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dangerous = []
    seen = set()
    for item in objects:
        if not isinstance(item, dict):
            continue
        class_name = normalize_object_name(item.get("class_name") or item.get("class") or item.get("name") or item.get("label"))
        display_name = dangerous_object_display_name(class_name)
        if not class_name or not display_name:
            continue
        key = normalize_object_key(class_name)
        if key in seen:
            continue
        seen.add(key)
        dangerous.append(
            {
                "class_name": class_name,
                "display_name": display_name,
                "confidence": safe_float(item.get("confidence") or item.get("score")),
            }
        )
    return dangerous


def extract_scores(payload: dict[str, Any], main_person: dict[str, Any] | None) -> dict[str, float]:
    raw_scores = {}
    if main_person:
        raw_scores = (
            main_person.get("probabilities")
            or main_person.get("scores")
            or main_person.get("emotion_scores")
            or {}
        )
    if not raw_scores:
        raw_scores = payload.get("scores") or payload.get("probabilities") or {}
    if not isinstance(raw_scores, dict):
        return {}
    return {str(key): safe_float(value) for key, value in raw_scores.items()}


def extract_main_emotion(payload: dict[str, Any]) -> dict[str, Any]:
    main_person = payload.get("main_person")
    if not isinstance(main_person, dict):
        faces = payload.get("faces") or []
        main_person = faces[0] if isinstance(faces, list) and faces and isinstance(faces[0], dict) else None

    if main_person:
        emotion = normalize_emotion(main_person.get("emotion") or main_person.get("dominant_emotion"))
        confidence = safe_float(main_person.get("confidence") or main_person.get("emotion_confidence"))
        scores = extract_scores(payload, main_person)
    else:
        emotion = normalize_emotion(payload.get("dominant_emotion") or payload.get("emotion"))
        confidence = safe_float(payload.get("confidence") or payload.get("emotion_confidence"))
        scores = extract_scores(payload, None)

    return {
        "emotion": emotion,
        "confidence": confidence,
        "scores": scores,
    }


def extract_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_objects = payload.get("objects") or []
    if isinstance(raw_objects, dict):
        raw_objects = raw_objects.get("items") or []
    if not isinstance(raw_objects, list):
        return []

    objects = []
    for item in raw_objects:
        if not isinstance(item, dict):
            continue
        class_name = normalize_object_name(item.get("class") or item.get("class_name") or item.get("name") or item.get("label"))
        if not class_name:
            continue
        objects.append(
            {
                "class_name": class_name,
                "confidence": safe_float(item.get("confidence") or item.get("score")),
            }
        )
    return objects


def record_vision_payload(db_path: str | Path, payload: dict[str, Any], now: float | None = None) -> dict[str, Any]:
    db_file = init_db(db_path)
    sampled_at = float(now if now is not None else time.time())
    source_timestamp = safe_float(payload.get("timestamp"), sampled_at)
    date_key = local_date_key(sampled_at)
    main = extract_main_emotion(payload)
    objects = extract_objects(payload)

    record = {
        "sampled_at": sampled_at,
        "source_timestamp": source_timestamp,
        "date_key": date_key,
        "status": str(payload.get("status") or ""),
        "valid": bool(payload.get("valid")),
        "face_count": safe_int(payload.get("face_count")),
        "dominant_emotion": main["emotion"],
        "confidence": main["confidence"],
        "scores": main["scores"],
        "objects": objects,
        "object_count": len(objects),
    }

    with sqlite3.connect(db_file) as conn:
        conn.execute(
            """
            INSERT INTO emotion_samples (
                sampled_at, source_timestamp, date_key, iso_time, status, valid, face_count,
                dominant_emotion, confidence, scores_json, object_count, objects_json, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["sampled_at"],
                record["source_timestamp"],
                record["date_key"],
                iso_time(record["sampled_at"]),
                record["status"],
                1 if record["valid"] else 0,
                record["face_count"],
                record["dominant_emotion"],
                record["confidence"],
                json_dumps(record["scores"]),
                record["object_count"],
                json_dumps(record["objects"]),
                json_dumps(payload),
            ),
        )

        conn.commit()
    return record


def upsert_learned_object(conn: sqlite3.Connection, date_key: str, class_name: str, timestamp: float, confidence: float) -> None:
    row = conn.execute(
        "SELECT count, max_confidence FROM learned_objects WHERE date_key = ? AND class_name = ?",
        (date_key, class_name),
    ).fetchone()
    if row is None:
        conn.execute(
            """
            INSERT INTO learned_objects (date_key, class_name, first_seen, last_seen, count, max_confidence)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (date_key, class_name, timestamp, timestamp, 1, confidence),
        )
        return

    count, max_confidence = row
    conn.execute(
        """
        UPDATE learned_objects
        SET last_seen = ?, count = ?, max_confidence = ?
        WHERE date_key = ? AND class_name = ?
        """,
        (timestamp, int(count) + 1, max(float(max_confidence), confidence), date_key, class_name),
    )


def normalize_learning_event_objects(raw_objects: Any) -> list[dict[str, Any]]:
    if isinstance(raw_objects, dict):
        raw_objects = raw_objects.get("items") or []
    if not isinstance(raw_objects, list):
        return []

    objects = []
    seen = set()
    for item in raw_objects:
        if not isinstance(item, dict):
            continue
        class_name = normalize_object_name(item.get("class_name") or item.get("class") or item.get("name") or item.get("label"))
        if not class_name or class_name in seen:
            continue
        seen.add(class_name)
        objects.append({"class_name": class_name, "confidence": safe_float(item.get("confidence") or item.get("score"))})
    return objects


def make_learning_event_id(raw: dict[str, Any], objects: list[dict[str, Any]]) -> str:
    existing = str(raw.get("event_id") or "").strip()
    if existing:
        return existing
    timestamp = safe_float(raw.get("timestamp"), 0.0)
    key = "{}|{}|{}|{}".format(
        "%.6f" % timestamp,
        raw.get("user_text") or raw.get("text") or "",
        raw.get("reply") or "",
        ",".join(item["class_name"] for item in objects),
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def ingest_learning_events(db_path: str | Path, learning_log: str | Path, date_key: str) -> int:
    db_file = init_db(db_path)
    path = Path(learning_log).expanduser()
    if not path.exists():
        return 0

    inserted = 0
    with sqlite3.connect(db_file) as conn, path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except ValueError:
                continue
            if not isinstance(raw, dict):
                continue
            timestamp = safe_float(raw.get("timestamp"), 0.0)
            if timestamp <= 0 or local_date_key(timestamp) != date_key:
                continue
            intent = str(raw.get("intent") or "").strip()
            if intent != "object_learning_query":
                continue
            objects = normalize_learning_event_objects(raw.get("objects"))
            if not objects:
                continue
            event_id = make_learning_event_id(raw, objects)
            result = conn.execute(
                """
                INSERT OR IGNORE INTO learning_events (
                    event_id, date_key, timestamp, iso_time, user_text, reply, intent, objects_json, raw_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    date_key,
                    timestamp,
                    str(raw.get("iso_time") or iso_time(timestamp)),
                    str(raw.get("user_text") or raw.get("text") or ""),
                    str(raw.get("reply") or ""),
                    intent,
                    json_dumps(objects),
                    json_dumps(raw),
                ),
            )
            if result.rowcount:
                inserted += 1
        conn.commit()
    return inserted


def is_loopback_url(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return host in {"localhost", "127.0.0.1", "::1"} or host.startswith("127.")


def fetch_json(url: str, timeout: float = 1.5) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json"})
    opener = build_opener(ProxyHandler({})) if is_loopback_url(url) else None
    open_func = opener.open if opener is not None else urlopen
    with open_func(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return json.loads(response.read().decode(charset))


def sample_once(
    db_path: str | Path,
    vision_url: str,
    *,
    fetcher=fetch_json,
    fetch_timeout: float = 1.5,
    now: float | None = None,
) -> dict[str, Any]:
    payload = fetcher(vision_url, timeout=fetch_timeout)
    if not isinstance(payload, dict):
        raise ValueError("vision API did not return a JSON object")
    return record_vision_payload(db_path, payload, now=now)


def run_sampler(
    db_path: str | Path,
    vision_url: str,
    sample_interval: float,
    stop_event: threading.Event,
    *,
    fetcher=fetch_json,
    fetch_timeout: float = 1.5,
    log=print,
) -> None:
    interval = max(0.2, float(sample_interval))
    while not stop_event.is_set():
        try:
            record = sample_once(db_path, vision_url, fetcher=fetcher, fetch_timeout=fetch_timeout)
            log(
                "[INFO] parent sample emotion={} confidence={:.2f} objects={}".format(
                    record["dominant_emotion"],
                    record["confidence"],
                    record["object_count"],
                )
            )
        except (OSError, URLError, ValueError, json.JSONDecodeError) as exc:
            log("[WARN] parent dashboard sample failed: {}".format(exc))
        stop_event.wait(interval)


def load_emotion_samples(db_path: str | Path, date_key: str, limit: int = 2000) -> list[dict[str, Any]]:
    path = Path(db_path).expanduser()
    if not path.exists():
        return []
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT sampled_at, iso_time, status, valid, face_count, dominant_emotion, confidence,
                   scores_json, object_count, objects_json
            FROM emotion_samples
            WHERE date_key = ?
            ORDER BY sampled_at ASC
            LIMIT ?
            """,
            (date_key, int(limit)),
        ).fetchall()

    samples = []
    for row in rows:
        timestamp = float(row["sampled_at"])
        samples.append(
            {
                "timestamp": timestamp,
                "iso_time": row["iso_time"],
                "time_label": time_label(timestamp),
                "status": row["status"],
                "valid": bool(row["valid"]),
                "face_count": int(row["face_count"]),
                "emotion": normalize_emotion(row["dominant_emotion"]),
                "confidence": float(row["confidence"]),
                "scores": safe_json_loads(row["scores_json"], {}),
                "object_count": int(row["object_count"]),
                "objects": safe_json_loads(row["objects_json"], []),
            }
        )
    return samples


def load_learned_objects(db_path: str | Path, date_key: str) -> list[dict[str, Any]]:
    path = Path(db_path).expanduser()
    if not path.exists():
        return []
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT timestamp, objects_json
            FROM learning_events
            WHERE date_key = ?
            ORDER BY timestamp ASC
            """,
            (date_key,),
        ).fetchall()

    aggregate: dict[str, dict[str, Any]] = {}
    for row in rows:
        timestamp = float(row["timestamp"])
        for obj in safe_json_loads(row["objects_json"], []):
            if not isinstance(obj, dict):
                continue
            class_name = normalize_object_name(obj.get("class_name") or obj.get("class"))
            if not class_name:
                continue
            confidence = safe_float(obj.get("confidence"))
            item = aggregate.get(class_name)
            if item is None:
                aggregate[class_name] = {
                    "class_name": class_name,
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "count": 1,
                    "max_confidence": confidence,
                }
            else:
                item["last_seen"] = timestamp
                item["count"] += 1
                item["max_confidence"] = max(item["max_confidence"], confidence)

    objects = []
    for item in aggregate.values():
        objects.append(
            {
                "class_name": item["class_name"],
                "first_seen": item["first_seen"],
                "first_seen_label": time_label(item["first_seen"]),
                "last_seen": item["last_seen"],
                "last_seen_label": time_label(item["last_seen"]),
                "count": item["count"],
                "max_confidence": item["max_confidence"],
            }
        )
    return sorted(objects, key=lambda item: (item["first_seen"], item["class_name"]))


def load_dangerous_objects(db_path: str | Path, date_key: str) -> list[dict[str, Any]]:
    path = Path(db_path).expanduser()
    if not path.exists():
        return []
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT sampled_at, objects_json
            FROM emotion_samples
            WHERE date_key = ?
            ORDER BY sampled_at ASC
            """,
            (date_key,),
        ).fetchall()

    aggregate: dict[str, dict[str, Any]] = {}
    for row in rows:
        timestamp = float(row["sampled_at"])
        raw_objects = safe_json_loads(row["objects_json"], [])
        if not isinstance(raw_objects, list):
            continue
        for obj in extract_dangerous_objects(raw_objects):
            class_name = obj["class_name"]
            key = normalize_object_key(class_name)
            confidence = safe_float(obj.get("confidence"))
            item = aggregate.get(key)
            if item is None:
                aggregate[key] = {
                    "class_name": class_name,
                    "display_name": obj["display_name"],
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "count": 1,
                    "max_confidence": confidence,
                }
            else:
                item["last_seen"] = timestamp
                item["count"] += 1
                item["max_confidence"] = max(item["max_confidence"], confidence)

    objects = []
    for item in aggregate.values():
        objects.append(
            {
                "class_name": item["class_name"],
                "display_name": item["display_name"],
                "first_seen": item["first_seen"],
                "first_seen_label": time_label(item["first_seen"]),
                "last_seen": item["last_seen"],
                "last_seen_label": time_label(item["last_seen"]),
                "count": item["count"],
                "max_confidence": item["max_confidence"],
            }
        )
    return sorted(objects, key=lambda item: (-item["last_seen"], item["display_name"], item["class_name"]))


def safe_json_loads(value: str, fallback: Any) -> Any:
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


def load_dialogue_entries(log_path: str | Path, date_key: str, limit: int = 500) -> list[dict[str, Any]]:
    path = Path(log_path).expanduser()
    if not path.exists():
        return []

    entries = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except ValueError:
                continue
            if not isinstance(raw, dict):
                continue
            timestamp = safe_float(raw.get("timestamp"), 0.0)
            text = str(raw.get("text") or "").strip()
            if timestamp <= 0 or not text or local_date_key(timestamp) != date_key:
                continue
            role = normalize_role(raw.get("role") or raw.get("speaker"))
            entries.append(
                {
                    "timestamp": timestamp,
                    "iso_time": iso_time(timestamp),
                    "time_label": time_label(timestamp),
                    "role": role,
                    "text": text,
                }
            )
    entries.sort(key=lambda item: item["timestamp"])
    return entries[-int(limit):]


def normalize_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    if role in {"assistant", "robot", "ai"}:
        return "robot"
    if role in {"user", "child", "kid"}:
        return "child"
    return role or "unknown"


def build_trend_summary(samples: list[dict[str, Any]]) -> str:
    if not samples:
        return "今天还没有形成足够的情绪记录，页面会在互动过程中持续更新。"
    if len(samples) < 3:
        return "今天已有少量记录，暂时只能作为实时观察参考。"

    emotions = [normalize_emotion(item.get("emotion")) for item in samples]
    known = [emotion for emotion in emotions if emotion != "unknown"]
    if not known:
        return "今天的画面里暂时没有稳定的人脸情绪记录。"

    changes = sum(1 for prev, current in zip(known, known[1:]) if prev != current)
    change_ratio = changes / max(1, len(known) - 1)
    negative_ratio = sum(1 for item in known if item in NEGATIVE_EMOTIONS) / len(known)
    positive_ratio = sum(1 for item in known if item in POSITIVE_EMOTIONS) / len(known)
    counts = emotion_counts(samples)
    focus_emotion = next((emotion for emotion in EMOTION_ATTENTION_ORDER if counts.get(emotion, 0) > 0), "unknown")
    focus_label = EMOTION_LABELS.get(focus_emotion, focus_emotion)

    if negative_ratio >= 0.45 and len(known) >= 5:
        return "今天较多时段识别到{}等需要优先关注的表情，建议家长多陪伴和关注；必要时咨询专业人士。".format(focus_label)
    if focus_emotion in NEGATIVE_EMOTIONS:
        return "最近记录里出现过{}表情，虽然不一定持续，但建议家长优先结合当时对话和活动场景看一看。".format(focus_label)
    if positive_ratio >= 0.55:
        return "今天多数记录呈现{}等轻松或积极表情，整体互动状态较平稳。".format(focus_label)
    if focus_emotion in POSITIVE_EMOTIONS:
        return "最近记录里能看到{}表情，建议继续结合聊天记录观察孩子对活动的兴趣。".format(focus_label)
    if change_ratio >= 0.55:
        return "今天情绪变化比较频繁，建议结合当时对话内容和活动场景一起观察。"
    return "今天情绪整体比较平稳，建议继续结合聊天记录和实际陪伴场景观察。"


def emotion_counts(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in samples:
        emotion = normalize_emotion(item.get("emotion"))
        counts[emotion] = counts.get(emotion, 0) + 1
    return counts


def build_dashboard_payload(
    db_path: str | Path,
    dialogue_log: str | Path,
    learning_log: str | Path = DEFAULT_LEARNING_LOG,
    *,
    date_key: str | None = None,
    mock_on_empty: bool = True,
    now: float | None = None,
) -> dict[str, Any]:
    date = date_key or local_date_key(now)
    ingest_learning_events(db_path, learning_log, date)
    samples = load_emotion_samples(db_path, date)
    objects = load_learned_objects(db_path, date)
    dangerous_objects = load_dangerous_objects(db_path, date)
    dialogue = load_dialogue_entries(dialogue_log, date)

    if mock_on_empty and not samples and not objects and not dangerous_objects and not dialogue:
        return mock_dashboard_payload(date, now=now)

    latest = samples[-1] if samples else None
    summary = {
        "sample_count": len(samples),
        "current_emotion": latest["emotion"] if latest else "unknown",
        "current_emotion_label": EMOTION_LABELS.get(latest["emotion"], latest["emotion"]) if latest else "未知",
        "current_confidence": latest["confidence"] if latest else 0.0,
        "face_count": latest["face_count"] if latest else 0,
        "dialogue_count": len(dialogue),
        "learned_object_count": len(objects),
        "dangerous_object_count": len(dangerous_objects),
        "emotion_counts": emotion_counts(samples),
        "trend_text": build_trend_summary(samples),
        "last_updated": latest["timestamp"] if latest else None,
    }
    return {
        "schema": SCHEMA,
        "date": date,
        "mock": False,
        "summary": summary,
        "emotions": samples,
        "dialogue": dialogue,
        "objects": objects,
        "dangerous_objects": dangerous_objects,
    }


def mock_dashboard_payload(date_key: str, now: float | None = None) -> dict[str, Any]:
    base = now if now is not None else time.time()
    start = base - 3600
    raw_samples = [
        (start + 0, "neutral", 0.62, 1),
        (start + 600, "happy", 0.81, 1),
        (start + 1200, "happy", 0.76, 1),
        (start + 1800, "surprise", 0.58, 1),
        (start + 2400, "neutral", 0.67, 1),
        (start + 3000, "happy", 0.74, 1),
    ]
    samples = [
        {
            "timestamp": ts,
            "iso_time": iso_time(ts),
            "time_label": time_label(ts),
            "status": "mock",
            "valid": True,
            "face_count": faces,
            "emotion": emotion,
            "confidence": confidence,
            "scores": {emotion: confidence},
            "object_count": 1,
            "objects": [],
        }
        for ts, emotion, confidence, faces in raw_samples
    ]
    dialogue = [
        {
            "timestamp": start + 420,
            "iso_time": iso_time(start + 420),
            "time_label": time_label(start + 420),
            "role": "child",
            "text": "灵犀，这是什么？",
        },
        {
            "timestamp": start + 430,
            "iso_time": iso_time(start + 430),
            "time_label": time_label(start + 430),
            "role": "robot",
            "text": "我看到一个玩具车，我们可以一起说一说它的颜色。",
        },
        {
            "timestamp": start + 1980,
            "iso_time": iso_time(start + 1980),
            "time_label": time_label(start + 1980),
            "role": "child",
            "text": "我还想看绘本。",
        },
        {
            "timestamp": start + 1990,
            "iso_time": iso_time(start + 1990),
            "time_label": time_label(start + 1990),
            "role": "robot",
            "text": "好呀，我们慢慢看，看到新的东西就一起认识。",
        },
    ]
    objects = [
        {
            "class_name": "toy car",
            "first_seen": start + 390,
            "first_seen_label": time_label(start + 390),
            "last_seen": start + 1260,
            "last_seen_label": time_label(start + 1260),
            "count": 4,
            "max_confidence": 0.88,
        },
        {
            "class_name": "book",
            "first_seen": start + 1920,
            "first_seen_label": time_label(start + 1920),
            "last_seen": start + 3060,
            "last_seen_label": time_label(start + 3060),
            "count": 3,
            "max_confidence": 0.79,
        },
    ]
    dangerous_objects = []
    latest = samples[-1]
    return {
        "schema": SCHEMA,
        "date": date_key,
        "mock": True,
        "summary": {
            "sample_count": len(samples),
            "current_emotion": latest["emotion"],
            "current_emotion_label": EMOTION_LABELS.get(latest["emotion"], latest["emotion"]),
            "current_confidence": latest["confidence"],
            "face_count": latest["face_count"],
            "dialogue_count": len(dialogue),
            "learned_object_count": len(objects),
            "dangerous_object_count": len(dangerous_objects),
            "emotion_counts": emotion_counts(samples),
            "trend_text": build_trend_summary(samples),
            "last_updated": latest["timestamp"],
        },
        "emotions": samples,
        "dialogue": dialogue,
        "objects": objects,
        "dangerous_objects": dangerous_objects,
    }


def write_json_response(handler: BaseHTTPRequestHandler, status_code: int, payload: Any) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(db_path: str | Path, dialogue_log: str | Path, learning_log: str | Path, mock_on_empty: bool = True):
    class ParentDashboardHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print("[HTTP] " + fmt % args)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            query = parse_qs(parsed.query)
            date_key = query.get("date", [None])[0]

            if path == "/health":
                write_json_response(
                    self,
                    200,
                    {
                        "status": "ok",
                        "schema": SCHEMA,
                        "db": str(Path(db_path).expanduser()),
                        "dialogue_log": str(Path(dialogue_log).expanduser()),
                        "learning_log": str(Path(learning_log).expanduser()),
                    },
                )
                return

            if path in {"/parent/today", "/parent/summary", "/parent"}:
                payload = build_dashboard_payload(
                    db_path,
                    dialogue_log,
                    learning_log,
                    date_key=date_key,
                    mock_on_empty=mock_on_empty,
                )
                write_json_response(self, 200, payload)
                return

            write_json_response(
                self,
                404,
                {
                    "detail": "not found",
                    "available_paths": ["/health", "/parent/today"],
                },
            )

    return ParentDashboardHandler


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    db_path = init_db(args.db)
    dialogue_log = Path(args.dialogue_log).expanduser()
    learning_log = Path(args.learning_log).expanduser()
    sample_interval = max(0.2, float(args.sample_interval))
    mock_on_empty = not bool(args.no_mock_on_empty)

    stop_event = threading.Event()
    sampler = threading.Thread(
        target=run_sampler,
        args=(db_path, args.vision_url, sample_interval, stop_event),
        kwargs={"fetch_timeout": args.fetch_timeout},
        name="parent-dashboard-sampler",
        daemon=True,
    )
    sampler.start()

    print("[INFO] parent dashboard db={}".format(db_path.resolve()))
    print("[INFO] vision source={}".format(args.vision_url))
    print("[INFO] dialogue log={}".format(dialogue_log))
    print("[INFO] learning log={}".format(learning_log))
    print("[INFO] listen: {}:{}".format(args.host, args.port))
    server = HTTPServer((args.host, args.port), make_handler(db_path, dialogue_log, learning_log, mock_on_empty=mock_on_empty))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[INFO] stopped.")
    finally:
        stop_event.set()
        server.server_close()
        sampler.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
