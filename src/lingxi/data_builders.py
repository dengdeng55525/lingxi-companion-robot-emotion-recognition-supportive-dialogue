from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from .constants import (
    EMOTION_KEYWORDS,
    SCENES,
    SUPPORT_TEMPLATES,
    SYSTEM_PROMPT,
)
from .io_utils import resolve_path


TEXT_KEYS = (
    "content",
    "text",
    "value",
    "utterance",
    "Utterance",
    "sentence",
    "query",
    "question",
    "answer",
    "response",
    "回复",
    "问题",
)

EMOTION_KEYS = ("emotion", "Emotion", "label", "情绪")

ROLE_USER = {"user", "human", "用户", "speaker1", "client", "seeker"}
ROLE_ASSISTANT = {"assistant", "gpt", "bot", "supporter", "helper"}


def infer_emotion(text: str, fallback: str | None = None) -> str:
    normalized = text or ""
    for emotion, keywords in EMOTION_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return emotion
    if fallback:
        return normalize_emotion(fallback)
    return "平静"


def normalize_emotion(value: str | None) -> str:
    if not value:
        return "平静"
    text = str(value).strip()
    mapping = {
        "happy": "开心",
        "neutral": "平静",
        "sad": "悲伤",
        "sadness": "悲伤",
        "angry": "愤怒",
        "anger": "愤怒",
        "fear": "焦虑",
        "anxiety": "焦虑",
        "surprise": "平静",
        "disgust": "愤怒",
        "like": "开心",
        "happiness": "开心",
        "depress": "悲伤",
        "depression": "悲伤",
    }
    lower = text.lower()
    if lower in mapping:
        return mapping[lower]
    for emotion in SUPPORT_TEMPLATES:
        if emotion in text:
            return emotion
    return "平静"


def make_sample(
    user_text: str,
    assistant_text: str,
    *,
    source: str,
    emotion: str | None = None,
    scene: str = "日常陪伴",
    include_system: bool = True,
) -> dict[str, Any] | None:
    user_text = clean_text(user_text)
    assistant_text = clean_text(assistant_text)
    if not user_text or not assistant_text:
        return None
    emotion = infer_emotion(user_text + assistant_text, emotion)
    messages = []
    if include_system:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.extend(
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    )
    return {
        "messages": messages,
        "source": source,
        "emotion": emotion,
        "scene": scene,
    }


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).replace("\u3000", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def build_soulchat_samples(root: str | Path, max_samples: int, include_system: bool) -> list[dict[str, Any]]:
    root_path = resolve_path(root)
    if not root_path.exists():
        return []
    rng = random.Random(17)
    samples: list[dict[str, Any]] = []
    for record in iter_records(root_path):
        for messages in extract_conversations(record):
            for user_text, assistant_text in pair_user_assistant(messages):
                emotion = infer_emotion(user_text)
                sample = make_sample(
                    user_text,
                    assistant_text,
                    source="SoulChatCorpus",
                    emotion=emotion,
                    scene="情绪陪伴",
                    include_system=include_system,
                )
                if sample:
                    samples.append(sample)
                    if len(samples) >= max_samples:
                        rng.shuffle(samples)
                        return samples
    rng.shuffle(samples)
    return samples[:max_samples]


def build_psyqa_samples(root: str | Path, max_samples: int, include_system: bool) -> list[dict[str, Any]]:
    root_path = resolve_path(root)
    if not root_path.exists():
        return []
    samples: list[dict[str, Any]] = []
    for record in iter_records(root_path):
        if not isinstance(record, dict):
            continue
        question = first_value(record, ("question", "query", "title", "description", "问题"))
        answer = first_value(record, ("answer", "response", "reply", "回复"))
        if not answer and isinstance(record.get("answers"), list) and record["answers"]:
            first_answer = record["answers"][0]
            if isinstance(first_answer, dict):
                answer = first_answer.get("answer_text") or first_answer.get("text") or ""
        if not question or not answer:
            continue
        user_text = f"用户希望获得日常情绪支持，但不需要医学诊断。用户说：{question}"
        emotion = infer_emotion(question)
        sample = make_sample(
            user_text,
            answer,
            source="PsyQA_example",
            emotion=emotion,
            scene="心理支持样例参考",
            include_system=include_system,
        )
        if sample:
            samples.append(sample)
            if len(samples) >= max_samples:
                break
    return samples[:max_samples]


def deduplicate_samples(samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique = []
    for sample in samples:
        messages = sample.get("messages", [])
        user_text = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        assistant_text = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), "")
        key = f"{user_text[:160]}\n{assistant_text[:160]}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(sample)
    return unique


def split_samples(
    samples: list[dict[str, Any]],
    *,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    total = len(shuffled)
    test_size = max(1, int(total * test_ratio)) if total >= 10 else max(0, int(total * test_ratio))
    valid_size = max(1, int(total * valid_ratio)) if total >= 10 else max(0, int(total * valid_ratio))
    test = shuffled[:test_size]
    valid = shuffled[test_size : test_size + valid_size]
    train = shuffled[test_size + valid_size :]
    return train, valid, test


def iter_records(root: Path) -> Iterable[Any]:
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".csv", ".tsv", ".txt"}
    )
    for path in files:
        yield from read_records(path)


def read_records(path: Path) -> Iterable[Any]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        elif suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                yield from flatten_json(json.load(f))
        elif suffix in {".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                yield from csv.DictReader(f, delimiter=delimiter)
        elif suffix == ".txt":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = clean_text(line)
                    if line:
                        yield {"text": line}
    except UnicodeDecodeError:
        return
    except (json.JSONDecodeError, csv.Error):
        return


def flatten_json(obj: Any) -> Iterable[Any]:
    if isinstance(obj, list):
        for item in obj:
            yield from flatten_json(item)
    elif isinstance(obj, dict):
        if any(key in obj for key in ("messages", "conversation", "conversations", "dialog", "dialogue")):
            yield obj
        elif all(isinstance(v, (dict, list)) for v in obj.values()):
            for value in obj.values():
                yield from flatten_json(value)
        else:
            yield obj


def extract_conversations(record: Any) -> Iterable[list[dict[str, str]]]:
    if not isinstance(record, dict):
        return
    for key in ("messages", "conversation", "conversations", "dialog", "dialogue", "history"):
        value = record.get(key)
        if isinstance(value, list):
            messages = normalize_messages(value)
            if messages:
                yield messages
    if "instruction" in record and ("output" in record or "response" in record):
        user_text = " ".join(filter(None, [clean_text(record.get("instruction")), clean_text(record.get("input"))]))
        assistant_text = clean_text(record.get("output") or record.get("response"))
        if user_text and assistant_text:
            yield [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]


def normalize_messages(items: list[Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for idx, item in enumerate(items):
        if isinstance(item, str):
            content = clean_text(item)
            role = "user" if idx % 2 == 0 else "assistant"
        elif isinstance(item, dict):
            content = first_text(item)
            role_value = item.get("role") or item.get("from") or item.get("speaker") or item.get("Speaker")
            role = normalize_role(role_value, idx)
        else:
            continue
        if content:
            messages.append({"role": role, "content": content})
    return messages


def normalize_role(value: Any, idx: int) -> str:
    text = str(value or "").strip().lower()
    if text in ROLE_USER:
        return "user"
    if text in ROLE_ASSISTANT:
        return "assistant"
    return "user" if idx % 2 == 0 else "assistant"


def pair_user_assistant(messages: list[dict[str, str]]) -> Iterable[tuple[str, str]]:
    last_user = ""
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "user":
            last_user = content
        elif role == "assistant" and last_user:
            yield last_user, content
            last_user = ""


def first_text(record: dict[str, Any]) -> str:
    return clean_text(first_value(record, TEXT_KEYS))


def first_value(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    for key, value in record.items():
        if str(key).lower() in {k.lower() for k in keys} and value not in (None, ""):
            return value
    return ""


def infer_scene(row: dict[str, Any]) -> str:
    for key in ("scene", "Scene", "topic", "Topic", "场景"):
        value = clean_text(row.get(key))
        if value:
            return value
    return "家庭陪伴"
