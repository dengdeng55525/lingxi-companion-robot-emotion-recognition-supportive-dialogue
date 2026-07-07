#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lingxi.io_utils import read_jsonl, resolve_path


def read_json(path: str | Path) -> dict[str, Any]:
    target = resolve_path(path)
    if not target.exists():
        return {}
    with target.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_domain_scores(path: str | Path) -> dict[str, dict[str, Any]]:
    data = read_json(path)
    return {target.get("label", ""): target.get("summary", {}) for target in data.get("targets", [])}


def metric(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpo-train", default="data/processed/dpo_train.jsonl")
    parser.add_argument("--dpo-valid", default="data/processed/dpo_valid.jsonl")
    parser.add_argument("--sft-metrics", default="outputs/lingxi-qwen25-1p5b-lora-30min/train_metrics.json")
    parser.add_argument("--dpo-metrics", default="outputs/lingxi-qwen25-1p5b-dpo/dpo_metrics.json")
    parser.add_argument("--domain-eval", default="reports/dpo_domain_qa_eval.json")
    parser.add_argument("--out", default="reports/dpo_alignment_report.md")
    args = parser.parse_args()

    dpo_train = read_jsonl(args.dpo_train)
    dpo_valid = read_jsonl(args.dpo_valid)
    category_counts = Counter(item.get("category", "") for item in dpo_train + dpo_valid)
    emotion_counts = Counter(item.get("emotion", "") for item in dpo_train + dpo_valid)
    sft_metrics = read_json(args.sft_metrics)
    dpo_metrics = read_json(args.dpo_metrics)
    domain_scores = collect_domain_scores(args.domain_eval)

    lines = [
        "# 二阶段 DPO 偏好对齐实验报告",
        "",
        "## 方法概述",
        "",
        "本项目采用两阶段训练：第一阶段使用 LoRA-SFT 学习家庭陪伴情绪支持回复风格；第二阶段使用 DPO 偏好对齐，让模型进一步偏向更共情、更安全、更具体、不过度诊断的回答。",
        "",
        "DPO 数据每条包含 `prompt/chosen/rejected`。`chosen` 表示更符合陪伴场景的回答，`rejected` 表示说教、泛泛安慰、医学越界或安全性不足的回答。",
        "",
        "本项目当前不输出机器人动作建议，因此 DPO 偏好维度聚焦于：共情性、相关性、安全性、边界意识、情绪适配。",
        "",
        "## 数据统计",
        "",
        f"- DPO train: {len(dpo_train)}",
        f"- DPO valid: {len(dpo_valid)}",
        f"- DPO total: {len(dpo_train) + len(dpo_valid)}",
        "",
        "| 类别 | 数量 |",
        "| --- | ---: |",
    ]
    for category, count in sorted(category_counts.items()):
        lines.append(f"| {category} | {count} |")

    lines.extend(["", "| 情绪 | 数量 |", "| --- | ---: |"])
    for emotion, count in sorted(emotion_counts.items()):
        lines.append(f"| {emotion} | {count} |")

    lines.extend(
        [
            "",
            "## 训练指标",
            "",
            "| 阶段 | Train loss | Eval loss | PPL/Reward acc | Runtime |",
            "| --- | ---: | ---: | ---: | ---: |",
            "| SFT | {train_loss} | {eval_loss} | {ppl} | {runtime} |".format(
                train_loss=metric(sft_metrics.get("train_loss")),
                eval_loss=metric(sft_metrics.get("eval_loss")),
                ppl=metric(sft_metrics.get("perplexity")),
                runtime=metric(sft_metrics.get("train_runtime")),
            ),
            "| DPO | {train_loss} | {eval_loss} | {reward_acc} | {runtime} |".format(
                train_loss=metric(dpo_metrics.get("train_loss")),
                eval_loss=metric(dpo_metrics.get("eval_loss")),
                reward_acc=metric(
                    dpo_metrics.get("eval_rewards/accuracies")
                    or dpo_metrics.get("rewards/accuracies")
                    or dpo_metrics.get("eval_reward_accuracies")
                ),
                runtime=metric(dpo_metrics.get("train_runtime")),
            ),
            "",
            "## 领域问答对比",
            "",
            "| 模型 | 平均分 | 平均长度 | 危机样本通过率 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for label in ("base", "sft", "dpo"):
        summary = domain_scores.get(label, {})
        safety = summary.get("safety_pass_rate")
        safety_text = "" if safety is None else f"{safety:.2%}"
        lines.append(
            f"| {label} | {metric(summary.get('avg_score'))} | {metric(summary.get('avg_length_chars'))} | {safety_text} |"
        )

    lines.extend(
        [
            "",
            "## 报告结论写法建议",
            "",
            "- 如果 DPO 的领域问答平均分高于 SFT，可以说明偏好对齐提升了回答的共情性、安全性或具体性。",
            "- 如果 DPO 的危机样本通过率高于 SFT，可以强调 DPO 对安全边界有正向作用。",
            "- 如果 DPO 指标没有明显提升，应解释为偏好数据规模较小、规则构造数据仍有限，但方法流程完整，可作为后续人工标注偏好数据的基础。",
            "- 最终论文/课程报告中应引用 `reports/dpo_domain_qa_eval.md` 的具体案例，而不是只引用自动分数。",
        ]
    )

    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
