from __future__ import annotations

import random
from collections import Counter
from typing import Any

from .constants import SYSTEM_PROMPT
from .io_utils import read_jsonl, write_jsonl


REJECTED_BY_CATEGORY = {
    "empathy": [
        "别想太多，事情过去就好了。",
        "你应该坚强一点，不要一直纠结。",
        "这种事大家都会遇到，没必要这么难受。",
    ],
    "relevance": [
        "我理解你的心情，希望你开心起来。",
        "加油，一切都会好起来的。",
        "你可以多做点开心的事情，不要在意这些。",
    ],
    "safety": [
        "你这是抑郁症，马上吃药就好了。",
        "别和别人说，自己忍一忍就过去了。",
        "既然你这么痛苦，那就不要想太多。",
    ],
    "boundary": [
        "我可以确定你有心理疾病，你必须按我说的做。",
        "不用找专业人士，我就能解决你的问题。",
        "你保证照我说的做，很快就一定会好。",
    ],
    "emotion_adaptation": [
        "不要难过，继续努力就行。",
        "开心点，别想那么多。",
        "这没什么大不了，睡一觉就好了。",
    ],
}


SAFETY_PROMPTS = [
    "我真的撑不下去了，觉得消失了也没关系。",
    "最近一直很痛苦，不想活了，也不想和任何人说。",
    "我脑子里总冒出伤害自己的念头，我有点害怕。",
    "我觉得自己没有价值，结束这一切也许会轻松一点。",
]


def extract_user_text(sample: dict[str, Any]) -> str:
    return next((m.get("content", "") for m in sample.get("messages", []) if m.get("role") == "user"), "")


def extract_assistant_text(sample: dict[str, Any]) -> str:
    return next((m.get("content", "") for m in sample.get("messages", []) if m.get("role") == "assistant"), "")


def concise_user_text(text: str) -> str:
    prefixes = (
        "用户希望获得日常情绪支持，但不需要医学诊断。用户说：",
        "场景：",
    )
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[-220:]
    return text[:260]


def chosen_response(user_text: str, original: str, emotion: str, category: str) -> str:
    user_text = concise_user_text(user_text)
    if category == "safety" or emotion == "危机":
        return (
            "我很重视你刚才说的这些话，也能听出你已经承受了很多。现在最重要的是先保证你的安全："
            "请尽快联系身边可信任的人，或拨打当地急救电话/危机干预热线。如果你身边有可能伤害自己的物品，"
            "请先把它们放远一些。你不用一个人扛着，我们可以先确认你现在是否安全、是否有人能陪在你身边。"
        )
    if category == "empathy":
        return (
            f"听起来这件事真的让你很{emotion if emotion != '平静' else '在意'}，你现在有这样的感受并不奇怪。"
            "我们可以先不急着给自己下结论，先把最难受的部分说清楚。我会按你的节奏听你讲。"
        )
    if category == "relevance":
        return (
            f"我听到你说的是：{user_text} 这不是一句简单的“想开点”就能解决的事。"
            "我们可以先找出最让你卡住的一点，再把它拆成今天能做的一小步。"
        )
    if category == "boundary":
        return (
            "我可以陪你梳理感受，但我不会给你做医学诊断，也不会承诺某种方法一定有效。"
            "如果这种痛苦持续影响睡眠、饮食或日常生活，建议你联系可信任的家人、老师或专业人员。"
            "在这之前，我们可以先从你此刻最需要被理解的部分开始。"
        )
    if category == "emotion_adaptation":
        if emotion == "焦虑":
            return "你现在像是被很多担心同时拉扯着。我们先把呼吸放慢一点，再列出最急的一件事，只处理下一小步。"
        if emotion == "愤怒":
            return "你现在很生气，说明这件事碰到了你的边界。先不要急着回应对方，我们可以先把最刺痛你的那句话说出来。"
        if emotion == "悲伤":
            return "这份失落很真实，你不需要马上振作。先允许自己难过一会儿，再慢慢看有没有一个可以被照顾的小需求。"
        if emotion == "孤独":
            return "一个人撑着会很辛苦。我在这里陪你说话，你可以先告诉我今天哪个时刻最明显地感到孤单。"
        if emotion == "疲惫":
            return "你已经撑了很久，累并不代表你不够好。现在可以先把目标放小，只照顾好接下来十分钟。"
        return "我在听，也会尽量贴着你刚才说的具体情况回应。你可以继续说，我们先把最在意的部分慢慢讲清楚。"
    return original or "我在听。你可以慢慢说，我会尽量给你温和、具体、不过度评判的回应。"


def make_preference_item(
    *,
    idx: int,
    user_text: str,
    original: str,
    emotion: str,
    scene: str,
    category: str,
    rng: random.Random,
) -> dict[str, Any]:
    chosen = chosen_response(user_text, original, emotion, category)
    rejected = rng.choice(REJECTED_BY_CATEGORY[category])
    return {
        "id": f"dpo_{idx:04d}",
        "category": category,
        "emotion": emotion,
        "scene": scene,
        "system": SYSTEM_PROMPT,
        "user": concise_user_text(user_text),
        "chosen": chosen,
        "rejected": rejected,
    }


def build_preference_dataset(
    *,
    input_file: str,
    train_file: str,
    valid_file: str,
    sample_file: str,
    max_samples: int = 400,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    rng = random.Random(seed)
    source = read_jsonl(input_file)
    rng.shuffle(source)
    categories = ["empathy", "relevance", "safety", "boundary", "emotion_adaptation"]
    items: list[dict[str, Any]] = []

    idx = 0
    for sample in source:
        if len(items) >= max_samples:
            break
        user_text = extract_user_text(sample)
        original = extract_assistant_text(sample)
        if not user_text or not original:
            continue
        category = categories[idx % len(categories)]
        if category == "safety":
            user_text = rng.choice(SAFETY_PROMPTS)
            emotion = "危机"
            scene = "安全风险"
        else:
            emotion = sample.get("emotion", "平静")
            scene = sample.get("scene", "家庭陪伴")
        items.append(
            make_preference_item(
                idx=idx,
                user_text=user_text,
                original=original,
                emotion=emotion,
                scene=scene,
                category=category,
                rng=rng,
            )
        )
        idx += 1

    rng.shuffle(items)
    valid_size = max(1, int(len(items) * valid_ratio)) if len(items) >= 10 else 0
    valid = items[:valid_size]
    train = items[valid_size:]
    write_jsonl(train, train_file)
    write_jsonl(valid, valid_file)
    write_jsonl(items[:20], sample_file)
    return {
        "total": len(items),
        "train": len(train),
        "valid": len(valid),
        "category_counts": dict(Counter(item["category"] for item in items)),
        "emotion_counts": dict(Counter(item["emotion"] for item in items)),
    }
