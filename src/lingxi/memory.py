from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import EMOTION_KEYWORDS, SYSTEM_PROMPT
from .io_utils import resolve_path
from .safety import is_high_risk


EMOTION_LABELS = ("危机", "焦虑", "悲伤", "孤独", "愤怒", "疲惫", "开心", "平静")

EMOTION_ALIASES = {
    "neutral": "平静",
    "calm": "平静",
    "happy": "开心",
    "happiness": "开心",
    "joy": "开心",
    "joyful": "开心",
    "like": "开心",
    "positive": "开心",
    "sad": "悲伤",
    "sadness": "悲伤",
    "depressed": "悲伤",
    "angry": "愤怒",
    "anger": "愤怒",
    "anxious": "焦虑",
    "anxiety": "焦虑",
    "fear": "焦虑",
    "lonely": "孤独",
    "tired": "疲惫",
    "crisis": "危机",
    "risk": "危机",
}

def normalize_emotion(value: str | None) -> str:
    if not value:
        return "平静"
    text = str(value).strip()
    if not text:
        return "平静"
    lower = text.lower()
    if lower in EMOTION_ALIASES:
        return EMOTION_ALIASES[lower]
    for emotion in EMOTION_LABELS:
        if emotion in text:
            return emotion
    return "平静"


def infer_emotion(text: str, fallback: str | None = None) -> str:
    normalized = text or ""
    if is_high_risk(normalized):
        return "危机"
    for emotion, keywords in EMOTION_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return emotion
    return normalize_emotion(fallback)


def load_memory(path: str | Path) -> list[dict[str, Any]]:
    memory_path = resolve_path(path)
    if not memory_path.exists():
        return []
    with memory_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Memory file must contain a JSON list: {memory_path}")
    return [record for record in data if isinstance(record, dict)]


def save_memory(records: list[dict[str, Any]], path: str | Path) -> None:
    memory_path = resolve_path(path)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    with memory_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")


def next_round(records: list[dict[str, Any]]) -> int:
    rounds = [int(record.get("round", 0)) for record in records if str(record.get("round", "")).isdigit()]
    return max(rounds, default=0) + 1


def append_memory(
    path: str | Path,
    *,
    emotion: str,
    user_text: str,
    robot_reply: str,
    keep_records: int = 50,
) -> dict[str, Any]:
    records = load_memory(path)
    record = {
        "round": next_round(records),
        "emotion": normalize_emotion(emotion),
        "user_text": user_text,
        "robot_reply": robot_reply,
    }
    records.append(record)
    if keep_records > 0:
        records = records[-keep_records:]
    save_memory(records, path)
    return record


def recent_memory(records: list[dict[str, Any]], window: int = 3) -> list[dict[str, Any]]:
    if window <= 0:
        return []
    return records[-window:]


def emotion_history(records: list[dict[str, Any]], window: int = 3) -> str:
    recent = recent_memory(records, window)
    if not recent:
        return "无"
    return " → ".join(normalize_emotion(str(record.get("emotion", ""))) for record in recent)


def emotion_trend(records: list[dict[str, Any]], current_emotion: str, window: int = 3) -> str:
    recent = recent_memory(records, window)
    emotions = [normalize_emotion(str(record.get("emotion", ""))) for record in recent]
    current = normalize_emotion(current_emotion)
    if current == "危机":
        return "当前表达包含安全风险信号，需要优先关注用户安全，并建议联系身边可信任的人或当地紧急支持。"
    if current == "开心":
        if emotions and emotions[-1] != current:
            return "当前表达是正向分享，历史情绪只作为背景；回复应回应当前开心，不要延续上一轮压力或负面话题。"
        return "当前表达偏正向，可以自然回应这份开心，并邀请用户继续分享。"
    if len(emotions) >= 2 and emotions[-2:] == [current, current] and current != "平静":
        return f"最近多轮持续出现{current}情绪，回复应更温和，减少追问，先稳定情绪。"
    if emotions and emotions[-1] == current and current != "平静":
        return f"上一轮和当前都偏向{current}，需要承接前文感受，避免像第一次听到一样重新开始。"
    if emotions:
        return "历史情绪有变化，当前表达优先；回复应结合当前表达，不机械延续上一轮话题或情绪判断。"
    return "暂无历史情绪，仅根据当前表达进行陪伴回应。"


def format_recent_records(records: list[dict[str, Any]], window: int = 3) -> str:
    recent = recent_memory(records, window)
    if not recent:
        return "无"
    lines = []
    for record in recent:
        round_id = record.get("round", "?")
        emotion = normalize_emotion(str(record.get("emotion", "")))
        user_text = str(record.get("user_text", "")).strip()
        lines.append(f"第{round_id}轮：情绪={emotion}；用户表达={user_text}")
    return "\n".join(lines)


def build_memory_prompt(
    *,
    user_text: str,
    scene: str,
    current_emotion: str,
    records: list[dict[str, Any]],
    window: int = 3,
) -> str:
    emotion = normalize_emotion(current_emotion)
    return "\n".join(
        [
            f"当前情绪：{emotion}",
            f"历史情绪：{emotion_history(records, window)}",
            f"情绪趋势：{emotion_trend(records, emotion, window)}",
            f"用户表达：{user_text}",
            f"场景：{scene}",
            "优先级：当前用户表达最高；短期记忆只作为背景，不能覆盖或改写当前表达。",
            "",
            "最近记忆（仅包含历史用户表达和情绪，不包含历史回复原文）：",
            format_recent_records(records, window),
            "",
            "请结合短期情绪记忆，生成一段温和、尊重、非评判的家庭陪伴回复。",
            "不要做医学诊断，不要给药物建议，不要输出机器人动作建议。",
        ]
    )


def build_memory_messages(
    *,
    user_text: str,
    scene: str,
    current_emotion: str,
    records: list[dict[str, Any]],
    window: int = 3,
) -> list[dict[str, str]]:
    system = (
        SYSTEM_PROMPT
        + "\n你可以使用短期情绪记忆理解用户最近的情绪变化，但不要把记忆当作医学诊断依据。"
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": build_memory_prompt(
                user_text=user_text,
                scene=scene,
                current_emotion=current_emotion,
                records=records,
                window=window,
            ),
        },
    ]
