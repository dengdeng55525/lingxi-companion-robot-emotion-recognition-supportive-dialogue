#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lingxi.io_utils import load_yaml, resolve_path


COLORS = {
    "blue": "#2563eb",
    "orange": "#f97316",
    "green": "#16a34a",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "teal": "#0d9488",
    "gray": "#64748b",
    "yellow": "#ca8a04",
}


def read_json(path: str | Path) -> dict[str, Any]:
    target = resolve_path(path)
    if not target.exists():
        return {}
    with target.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def label_from_output_dir(output_dir: str | Path) -> str:
    return Path(output_dir).name


def expand_patterns(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(resolve_path(pattern))))
        paths.extend(Path(item) for item in matches)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def method_name(row: dict[str, Any]) -> str:
    if row.get("use_rslora"):
        return "rsLoRA"
    if row.get("neftune_noise_alpha"):
        return "NEFTune"
    return "LoRA"


def load_domain_eval(path: str | Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, float]]]:
    data = read_json(path)
    summaries: dict[str, dict[str, Any]] = {}
    item_scores: dict[str, dict[str, float]] = {}
    for target in data.get("targets", []):
        label = target.get("label", "")
        if not label:
            continue
        summaries[label] = target.get("summary", {})
        per_prompt: dict[str, float] = {}
        for item in target.get("items", []):
            prompt_id = item.get("prompt", {}).get("id", "")
            score = metric(item.get("metrics", {}).get("score"))
            if prompt_id and score is not None:
                per_prompt[prompt_id] = score
        item_scores[label] = per_prompt
    return summaries, item_scores


def collect_experiment_rows(
    config_patterns: list[str],
    domain_summaries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config_path in expand_patterns(config_patterns):
        config = load_yaml(config_path)
        training = config.get("training", {})
        lora = config.get("lora", {})
        output_dir = training.get("output_dir", "")
        label = label_from_output_dir(output_dir)
        metrics = read_json(resolve_path(output_dir) / "train_metrics.json")
        summary = domain_summaries.get(label, {})
        row = {
            "config": str(config_path.relative_to(ROOT)),
            "label": label,
            "output_dir": output_dir,
            "rank": lora.get("r"),
            "alpha": lora.get("alpha"),
            "use_rslora": bool(lora.get("use_rslora", False)),
            "max_steps": training.get("max_steps"),
            "epochs": training.get("num_train_epochs"),
            "neftune_noise_alpha": training.get("neftune_noise_alpha"),
            "learning_rate": training.get("learning_rate"),
            "train_loss": metric(metrics.get("train_loss")),
            "eval_loss": metric(metrics.get("eval_loss")),
            "perplexity": metric(metrics.get("perplexity")),
            "train_runtime": metric(metrics.get("train_runtime")),
            "train_samples_per_second": metric(metrics.get("train_samples_per_second")),
            "train_steps_per_second": metric(metrics.get("train_steps_per_second")),
            "eval_runtime": metric(metrics.get("eval_runtime")),
            "domain_score": metric(summary.get("avg_score")),
            "avg_length_chars": metric(summary.get("avg_length_chars")),
            "safety_pass_rate": metric(summary.get("safety_pass_rate")),
            "metrics_found": bool(metrics),
        }
        row["method"] = method_name(row)
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row.get("rank") or 0,
            row.get("max_steps") or 0,
            row.get("method") or "",
            row.get("label") or "",
        )
    )
    return rows


def collect_step_history(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, float]]]:
    histories: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        path = resolve_path(row["output_dir"]) / "trainer_state.json"
        data = read_json(path)
        history: list[dict[str, float]] = []
        for item in data.get("log_history", []):
            step = metric(item.get("step"))
            if step is None:
                continue
            point: dict[str, float] = {"step": step}
            for key in ("loss", "eval_loss", "learning_rate", "grad_norm"):
                value = metric(item.get(key))
                if value is not None:
                    point[key] = value
            if len(point) > 1:
                history.append(point)
        if history:
            histories[row["label"]] = history
    return histories


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#0f172a",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, name: str, saved: list[Path]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)


def annotate_bars(ax: plt.Axes, bars: Any, fmt: str = "{:.3f}") -> None:
    for bar in bars:
        height = bar.get_height()
        if math.isnan(height):
            continue
        offset = 3 if height >= 0 else -10
        va = "bottom" if height >= 0 else "top"
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
            color="#334155",
        )


def labels(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["label"]).replace("_", "\n") for row in rows]


def plot_loss_comparison(rows: list[dict[str, Any]], out_dir: Path, saved: list[Path]) -> None:
    valid = [row for row in rows if row.get("train_loss") is not None and row.get("eval_loss") is not None]
    if not valid:
        return
    x = list(range(len(valid)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(10, len(valid) * 1.2), 5.6))
    train = [row["train_loss"] for row in valid]
    evals = [row["eval_loss"] for row in valid]
    bars1 = ax.bar([i - width / 2 for i in x], train, width, label="Train loss", color=COLORS["blue"])
    bars2 = ax.bar([i + width / 2 for i in x], evals, width, label="Eval loss", color=COLORS["orange"])
    ax.set_title("SFT Loss by Experiment")
    ax.set_ylabel("Loss")
    ax.set_xticks(x, labels(valid), rotation=35, ha="right")
    lower = min(train + evals) - 0.04
    upper = max(train + evals) + 0.05
    ax.set_ylim(max(0, lower), upper)
    ax.legend(frameon=False, ncols=2)
    annotate_bars(ax, bars1)
    annotate_bars(ax, bars2)
    save_figure(fig, out_dir, "training_loss_comparison.png", saved)


def plot_perplexity(rows: list[dict[str, Any]], out_dir: Path, saved: list[Path]) -> None:
    valid = [row for row in rows if row.get("perplexity") is not None]
    if not valid:
        return
    x = list(range(len(valid)))
    values = [row["perplexity"] for row in valid]
    fig, ax = plt.subplots(figsize=(max(10, len(valid) * 1.15), 5.4))
    bars = ax.bar(x, values, color=COLORS["green"])
    best = min(values)
    ax.axhline(best, color=COLORS["red"], linestyle="--", linewidth=1.2, label=f"Best PPL: {best:.3f}")
    ax.set_title("Validation Perplexity by Experiment")
    ax.set_ylabel("Perplexity")
    ax.set_xticks(x, labels(valid), rotation=35, ha="right")
    ax.set_ylim(min(values) - 0.08, max(values) + 0.12)
    ax.legend(frameon=False)
    annotate_bars(ax, bars, "{:.2f}")
    save_figure(fig, out_dir, "perplexity_comparison.png", saved)


def plot_runtime(rows: list[dict[str, Any]], out_dir: Path, saved: list[Path]) -> None:
    valid = [
        row
        for row in rows
        if row.get("train_runtime") is not None and row.get("train_steps_per_second") is not None
    ]
    if not valid:
        return
    x = list(range(len(valid)))
    runtime = [row["train_runtime"] for row in valid]
    steps_s = [row["train_steps_per_second"] for row in valid]
    fig, ax1 = plt.subplots(figsize=(max(10, len(valid) * 1.15), 5.4))
    bars = ax1.bar(x, runtime, color=COLORS["teal"], alpha=0.82, label="Runtime (s)")
    ax1.set_ylabel("Runtime (seconds)")
    ax1.set_xticks(x, labels(valid), rotation=35, ha="right")
    ax1.set_title("Training Runtime and Throughput")
    ax2 = ax1.twinx()
    ax2.plot(x, steps_s, marker="o", color=COLORS["red"], linewidth=2.2, label="Train steps/s")
    ax2.set_ylabel("Train steps per second")
    annotate_bars(ax1, bars, "{:.0f}")
    lines, line_labels = ax1.get_legend_handles_labels()
    lines2, line_labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, line_labels + line_labels2, frameon=False, loc="upper left")
    save_figure(fig, out_dir, "runtime_throughput.png", saved)


def plot_rank_ablation(rows: list[dict[str, Any]], out_dir: Path, saved: list[Path]) -> None:
    valid = [
        row
        for row in rows
        if row.get("max_steps") == 175
        and row.get("method") == "LoRA"
        and row.get("rank") is not None
        and row.get("eval_loss") is not None
    ]
    valid.sort(key=lambda row: row["rank"])
    if len(valid) < 2:
        return
    x = [row["rank"] for row in valid]
    fig, ax1 = plt.subplots(figsize=(8.5, 5.2))
    ax1.plot(x, [row["train_loss"] for row in valid], marker="o", linewidth=2.2, label="Train loss", color=COLORS["blue"])
    ax1.plot(x, [row["eval_loss"] for row in valid], marker="o", linewidth=2.2, label="Eval loss", color=COLORS["orange"])
    ax1.set_title("Rank Ablation at 175 Steps")
    ax1.set_xlabel("LoRA rank")
    ax1.set_ylabel("Loss")
    ax1.set_xticks(x)
    ax2 = ax1.twinx()
    ax2.plot(x, [row["perplexity"] for row in valid], marker="s", linewidth=2, color=COLORS["green"], label="Perplexity")
    ax2.set_ylabel("Perplexity")
    lines, line_labels = ax1.get_legend_handles_labels()
    lines2, line_labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, line_labels + line_labels2, frameon=False, loc="best")
    save_figure(fig, out_dir, "rank_ablation_loss_ppl.png", saved)


def plot_step_ablation(rows: list[dict[str, Any]], out_dir: Path, saved: list[Path]) -> None:
    valid = [
        row
        for row in rows
        if row.get("rank") == 8
        and row.get("method") == "LoRA"
        and row.get("max_steps") is not None
        and row.get("eval_loss") is not None
    ]
    valid.sort(key=lambda row: row["max_steps"])
    if len(valid) < 2:
        return
    x = [row["max_steps"] for row in valid]
    fig, ax1 = plt.subplots(figsize=(8.5, 5.2))
    ax1.plot(x, [row["train_loss"] for row in valid], marker="o", linewidth=2.2, label="Train loss", color=COLORS["blue"])
    ax1.plot(x, [row["eval_loss"] for row in valid], marker="o", linewidth=2.2, label="Eval loss", color=COLORS["orange"])
    ax1.set_title("Step Ablation at Rank 8")
    ax1.set_xlabel("Max train steps")
    ax1.set_ylabel("Loss")
    ax1.set_xticks(x)
    ax2 = ax1.twinx()
    ax2.plot(x, [row["domain_score"] for row in valid], marker="s", linewidth=2, color=COLORS["purple"], label="Domain score")
    ax2.set_ylabel("Domain score")
    lines, line_labels = ax1.get_legend_handles_labels()
    lines2, line_labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, line_labels + line_labels2, frameon=False, loc="best")
    save_figure(fig, out_dir, "steps_ablation_loss_domain.png", saved)


def plot_method_ablation(rows: list[dict[str, Any]], out_dir: Path, saved: list[Path]) -> None:
    pairs = [
        ("Rank 8 / 175 steps", [row for row in rows if row.get("rank") == 8 and row.get("max_steps") == 175]),
        ("Rank 16 / 175 steps", [row for row in rows if row.get("rank") == 16 and row.get("max_steps") == 175]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    plotted = False
    for ax, (title, group) in zip(axes, pairs):
        group = [row for row in group if row.get("eval_loss") is not None]
        group.sort(key=lambda row: row.get("method") != "LoRA")
        if not group:
            ax.axis("off")
            continue
        plotted = True
        x = list(range(len(group)))
        values = [row["eval_loss"] for row in group]
        bar_colors = [COLORS["blue"] if row.get("method") == "LoRA" else COLORS["purple"] for row in group]
        bars = ax.bar(x, values, color=bar_colors)
        ax.set_title(title)
        ax.set_xticks(x, [row["method"] for row in group])
        ax.set_ylabel("Eval loss")
        ax.set_ylim(min(values) - 0.02, max(values) + 0.025)
        annotate_bars(ax, bars)
    if plotted:
        fig.suptitle("Training Method Ablation", fontweight="bold")
        save_figure(fig, out_dir, "method_ablation_eval_loss.png", saved)
    else:
        plt.close(fig)


def plot_domain_summary(
    domain_summaries: dict[str, dict[str, Any]],
    out_dir: Path,
    saved: list[Path],
) -> None:
    preferred = [
        "base",
        "rank4_steps175",
        "rank8_steps90",
        "rank8_steps175",
        "rank8_steps175_neftune",
        "rank8_steps350",
        "rank16_steps175",
        "rank16_steps175_rslora",
    ]
    ordered = [label for label in preferred if label in domain_summaries]
    ordered.extend(label for label in domain_summaries if label not in ordered)
    if not ordered:
        return
    x = list(range(len(ordered)))
    scores = [metric(domain_summaries[label].get("avg_score")) or 0.0 for label in ordered]
    safety = [metric(domain_summaries[label].get("safety_pass_rate")) or 0.0 for label in ordered]
    fig, ax1 = plt.subplots(figsize=(max(11, len(ordered) * 1.12), 5.5))
    bars = ax1.bar(x, scores, color=COLORS["blue"], alpha=0.86, label="Avg score")
    ax1.set_title("Domain QA Score and Safety Pass Rate")
    ax1.set_ylabel("Avg score")
    ax1.set_ylim(0, 105)
    ax1.set_xticks(x, [label.replace("_", "\n") for label in ordered], rotation=35, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, [value * 100 for value in safety], marker="o", color=COLORS["red"], linewidth=2.2, label="Safety pass (%)")
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("Safety pass rate (%)")
    annotate_bars(ax1, bars, "{:.1f}")
    lines, line_labels = ax1.get_legend_handles_labels()
    lines2, line_labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, line_labels + line_labels2, frameon=False, loc="upper left")
    save_figure(fig, out_dir, "domain_score_safety.png", saved)


def plot_heatmap(
    domain_scores: dict[str, dict[str, float]],
    out_dir: Path,
    saved: list[Path],
) -> None:
    if not domain_scores:
        return
    preferred = [
        "base",
        "rank4_steps175",
        "rank8_steps90",
        "rank8_steps175",
        "rank8_steps175_neftune",
        "rank8_steps350",
        "rank16_steps175",
        "rank16_steps175_rslora",
    ]
    models = [label for label in preferred if label in domain_scores]
    models.extend(label for label in domain_scores if label not in models)
    prompt_ids = sorted({pid for scores in domain_scores.values() for pid in scores})
    if not models or not prompt_ids:
        return
    matrix = []
    for prompt_id in prompt_ids:
        matrix.append([domain_scores.get(model, {}).get(prompt_id, math.nan) for model in models])
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.18), max(4.2, len(prompt_ids) * 0.75)))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=50, vmax=100, aspect="auto")
    ax.set_title("Domain QA Score Heatmap")
    ax.set_xticks(range(len(models)), [model.replace("_", "\n") for model in models], rotation=35, ha="right")
    ax.set_yticks(range(len(prompt_ids)), prompt_ids)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if math.isnan(value):
                continue
            color = "white" if value < 70 else "#0f172a"
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", color=color, fontsize=8)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Score")
    save_figure(fig, out_dir, "qa_score_heatmap.png", saved)


def plot_loss_vs_domain(rows: list[dict[str, Any]], out_dir: Path, saved: list[Path]) -> None:
    valid = [row for row in rows if row.get("eval_loss") is not None and row.get("domain_score") is not None]
    if len(valid) < 2:
        return
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    method_colors = {"LoRA": COLORS["blue"], "NEFTune": COLORS["purple"], "rsLoRA": COLORS["green"]}
    for row in valid:
        ax.scatter(
            row["eval_loss"],
            row["domain_score"],
            s=85,
            color=method_colors.get(row["method"], COLORS["gray"]),
            edgecolor="white",
            linewidth=0.8,
            label=row["method"],
        )
        ax.annotate(row["label"], (row["eval_loss"], row["domain_score"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    handles, handle_labels = ax.get_legend_handles_labels()
    by_label = dict(zip(handle_labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=False, title="Method")
    ax.set_title("Eval Loss vs Domain QA Score")
    ax.set_xlabel("Eval loss (lower is better)")
    ax.set_ylabel("Domain QA avg score (higher is better)")
    ax.invert_xaxis()
    save_figure(fig, out_dir, "loss_vs_domain_score.png", saved)


def plot_step_histories(histories: dict[str, list[dict[str, float]]], out_dir: Path, saved: list[Path]) -> None:
    if not histories:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    plotted_loss = False
    plotted_lr = False
    for label, history in histories.items():
        loss_points = [(item["step"], item["loss"]) for item in history if "loss" in item]
        eval_points = [(item["step"], item["eval_loss"]) for item in history if "eval_loss" in item]
        lr_points = [(item["step"], item["learning_rate"]) for item in history if "learning_rate" in item]
        if loss_points:
            plotted_loss = True
            axes[0].plot([x for x, _ in loss_points], [y for _, y in loss_points], linewidth=1.8, label=f"{label} train")
        if eval_points:
            plotted_loss = True
            axes[0].plot([x for x, _ in eval_points], [y for _, y in eval_points], linewidth=1.8, linestyle="--", label=f"{label} eval")
        if lr_points:
            plotted_lr = True
            axes[1].plot([x for x, _ in lr_points], [y for _, y in lr_points], linewidth=1.8, label=label)
    axes[0].set_title("Step Loss Curves")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Learning Rate Schedule")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Learning rate")
    if plotted_loss:
        axes[0].legend(frameon=False, fontsize=8)
    else:
        axes[0].axis("off")
    if plotted_lr:
        axes[1].legend(frameon=False, fontsize=8)
    else:
        axes[1].axis("off")
    save_figure(fig, out_dir, "step_loss_learning_rate_curves.png", saved)


def plot_dpo_rewards(dpo_metrics: dict[str, Any], out_dir: Path, saved: list[Path]) -> None:
    if not dpo_metrics:
        return
    chosen = metric(dpo_metrics.get("eval_rewards/chosen"))
    rejected = metric(dpo_metrics.get("eval_rewards/rejected"))
    margin = metric(dpo_metrics.get("eval_rewards/margins"))
    accuracy = metric(dpo_metrics.get("eval_rewards/accuracies"))
    train_loss = metric(dpo_metrics.get("train_loss"))
    eval_loss = metric(dpo_metrics.get("eval_loss"))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    if chosen is not None and rejected is not None:
        bars = axes[0].bar(["Chosen", "Rejected"], [chosen, rejected], color=[COLORS["green"], COLORS["red"]])
        axes[0].axhline(0, color="#0f172a", linewidth=0.9)
        axes[0].set_title("DPO Eval Rewards")
        axes[0].set_ylabel("Reward")
        annotate_bars(axes[0], bars)
    else:
        axes[0].axis("off")
    right_values = []
    right_labels = []
    if margin is not None:
        right_labels.append("Margin")
        right_values.append(margin)
    if accuracy is not None:
        right_labels.append("Accuracy x10")
        right_values.append(accuracy * 10)
    if right_values:
        bars = axes[1].bar(right_labels, right_values, color=[COLORS["purple"], COLORS["yellow"]][: len(right_values)])
        axes[1].set_title("Reward Margin and Accuracy")
        axes[1].set_ylabel("Scaled value")
        annotate_bars(axes[1], bars)
    else:
        axes[1].axis("off")
    loss_labels = []
    loss_values = []
    if train_loss is not None:
        loss_labels.append("Train")
        loss_values.append(train_loss)
    if eval_loss is not None:
        loss_labels.append("Eval")
        loss_values.append(eval_loss)
    if loss_values:
        bars = axes[2].bar(loss_labels, loss_values, color=[COLORS["blue"], COLORS["orange"]][: len(loss_values)])
        axes[2].set_title("DPO Loss")
        axes[2].set_ylabel("Loss")
        annotate_bars(axes[2], bars, "{:.4f}")
    else:
        axes[2].axis("off")
    fig.suptitle("DPO Training Metrics", fontweight="bold")
    save_figure(fig, out_dir, "dpo_reward_metrics.png", saved)


def plot_sft_dpo_domain(
    dpo_domain_summaries: dict[str, dict[str, Any]],
    out_dir: Path,
    saved: list[Path],
) -> None:
    ordered = [label for label in ("base", "sft", "dpo") if label in dpo_domain_summaries]
    if not ordered:
        return
    x = list(range(len(ordered)))
    score = [metric(dpo_domain_summaries[label].get("avg_score")) or 0.0 for label in ordered]
    length = [metric(dpo_domain_summaries[label].get("avg_length_chars")) or 0.0 for label in ordered]
    safety = [metric(dpo_domain_summaries[label].get("safety_pass_rate")) or 0.0 for label in ordered]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))
    bars0 = axes[0].bar(x, score, color=COLORS["blue"])
    axes[0].set_title("Avg Score")
    axes[0].set_xticks(x, ordered)
    axes[0].set_ylim(0, 105)
    annotate_bars(axes[0], bars0, "{:.1f}")
    bars1 = axes[1].bar(x, [item * 100 for item in safety], color=COLORS["red"])
    axes[1].set_title("Safety Pass Rate")
    axes[1].set_xticks(x, ordered)
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("%")
    annotate_bars(axes[1], bars1, "{:.0f}")
    bars2 = axes[2].bar(x, length, color=COLORS["teal"])
    axes[2].set_title("Avg Response Length")
    axes[2].set_xticks(x, ordered)
    annotate_bars(axes[2], bars2, "{:.0f}")
    fig.suptitle("Base vs SFT vs DPO Domain Evaluation", fontweight="bold")
    save_figure(fig, out_dir, "sft_dpo_domain_eval.png", saved)


def write_overview_csv(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "metrics_overview.csv"
    fields = [
        "label",
        "method",
        "rank",
        "alpha",
        "max_steps",
        "epochs",
        "neftune_noise_alpha",
        "learning_rate",
        "train_loss",
        "eval_loss",
        "perplexity",
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
        "domain_score",
        "avg_length_chars",
        "safety_pass_rate",
        "output_dir",
        "config",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def write_manifest(saved: list[Path], csv_path: Path, out_dir: Path, histories: dict[str, list[dict[str, float]]]) -> Path:
    path = out_dir / "figures_manifest.md"
    lines = [
        "# Training Visualization Outputs",
        "",
        "Generated figures:",
        "",
    ]
    for item in saved:
        lines.append(f"- `{item.relative_to(ROOT)}`")
    lines.extend(["", "Data table:", "", f"- `{csv_path.relative_to(ROOT)}`", ""])
    if histories:
        lines.append("Per-step curves were generated from `trainer_state.json` files.")
    else:
        lines.append(
            "No `trainer_state.json` files were found, so the available plots use final train/eval metrics and evaluation summaries."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-configs",
        nargs="+",
        default=["configs/experiments/qwen25_1p5b_lora_rank*_steps*.yaml"],
        help="Config globs for SFT ablation experiments.",
    )
    parser.add_argument("--domain-eval", default="reports/domain_qa_eval.json")
    parser.add_argument("--dpo-domain-eval", default="reports/dpo_domain_qa_eval.json")
    parser.add_argument("--dpo-metrics", default="outputs/lingxi-qwen25-1p5b-dpo/dpo_metrics.json")
    parser.add_argument("--out-dir", default="reports/figures")
    args = parser.parse_args()

    setup_style()
    out_dir = resolve_path(args.out_dir)
    saved: list[Path] = []

    domain_summaries, domain_scores = load_domain_eval(args.domain_eval)
    dpo_domain_summaries, _ = load_domain_eval(args.dpo_domain_eval)
    rows = collect_experiment_rows(args.experiment_configs, domain_summaries)
    histories = collect_step_history(rows)

    plot_loss_comparison(rows, out_dir, saved)
    plot_perplexity(rows, out_dir, saved)
    plot_runtime(rows, out_dir, saved)
    plot_rank_ablation(rows, out_dir, saved)
    plot_step_ablation(rows, out_dir, saved)
    plot_method_ablation(rows, out_dir, saved)
    plot_domain_summary(domain_summaries, out_dir, saved)
    plot_heatmap(domain_scores, out_dir, saved)
    plot_loss_vs_domain(rows, out_dir, saved)
    plot_step_histories(histories, out_dir, saved)
    plot_dpo_rewards(read_json(args.dpo_metrics), out_dir, saved)
    plot_sft_dpo_domain(dpo_domain_summaries, out_dir, saved)

    csv_path = write_overview_csv(rows, out_dir)
    manifest_path = write_manifest(saved, csv_path, out_dir, histories)
    print(f"Wrote {len(saved)} figures to {out_dir}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
