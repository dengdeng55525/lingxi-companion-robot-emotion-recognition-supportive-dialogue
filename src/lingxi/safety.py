from __future__ import annotations

from dataclasses import dataclass


SAFETY_KEYWORDS = {
    "self_harm": (
        "自伤",
        "自杀",
        "伤害自己",
        "不想活",
        "结束生命",
        "轻生",
        "割腕",
        "跳楼",
        "消失了也没关系",
        "撑不下去",
    ),
    "harm_others": (
        "伤害别人",
        "杀了他",
        "杀了她",
        "报复",
        "弄死",
        "打死",
        "想伤人",
    ),
    "extreme_distress": (
        "极端痛苦",
        "痛苦到受不了",
        "活着没意思",
        "没有希望",
        "彻底崩溃",
        "撑不住",
    ),
    "sleep_and_health": (
        "长期失眠",
        "好多天没睡",
        "整晚睡不着",
        "睡眠障碍",
    ),
    "medication": (
        "吃药",
        "药量",
        "停药",
        "抗抑郁药",
        "安眠药",
        "镇静剂",
    ),
    "diagnosis": (
        "诊断",
        "确诊",
        "是不是抑郁症",
        "是不是焦虑症",
        "抑郁症",
        "焦虑症",
        "双相",
        "精神疾病",
    ),
}

HIGH_RISK_CATEGORIES = {"self_harm", "harm_others", "extreme_distress"}
BOUNDARY_CATEGORIES = {"sleep_and_health", "medication", "diagnosis"}

SAFETY_REPLY = (
    "我能感受到你现在很难受。这个问题不适合只由陪伴机器人判断或处理。"
    "建议你尽快联系身边可信任的人，或者寻求专业帮助。"
    "如果你现在有伤害自己或伤害别人的冲动，请优先保证安全，并联系当地急救电话或危机干预热线。"
    "我可以先陪你做几次缓慢呼吸，或者帮你整理想对身边人或专业人士说的话。"
)

BOUNDARY_REPLY = (
    "我能理解你希望尽快弄清楚自己的状态。涉及诊断、用药或长期失眠的问题，"
    "不适合只由陪伴机器人判断，也不应该由我给出医学结论或药物建议。"
    "建议你联系专业医生或心理健康专业人士。此刻我可以先陪你梳理症状、情绪和想咨询的问题，"
    "帮助你更清楚地表达给专业人士。"
)


@dataclass(frozen=True)
class SafetyResult:
    triggered: bool
    level: str
    categories: tuple[str, ...]
    matched_keywords: tuple[str, ...]
    reply: str


def check_safety(text: str) -> SafetyResult:
    normalized = text or ""
    categories = []
    matched_keywords = []
    for category, keywords in SAFETY_KEYWORDS.items():
        hits = [keyword for keyword in keywords if keyword in normalized]
        if hits:
            categories.append(category)
            matched_keywords.extend(hits)

    if not categories:
        return SafetyResult(
            triggered=False,
            level="safe",
            categories=(),
            matched_keywords=(),
            reply="",
        )

    category_set = set(categories)
    if category_set & HIGH_RISK_CATEGORIES:
        level = "high"
        reply = SAFETY_REPLY
    else:
        level = "boundary"
        reply = BOUNDARY_REPLY

    return SafetyResult(
        triggered=True,
        level=level,
        categories=tuple(categories),
        matched_keywords=tuple(dict.fromkeys(matched_keywords)),
        reply=reply,
    )


def is_high_risk(text: str) -> bool:
    return check_safety(text).level == "high"
