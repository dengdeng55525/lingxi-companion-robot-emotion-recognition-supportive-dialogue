#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lingxi.data_builders import (
    build_psyqa_samples,
    build_soulchat_samples,
    deduplicate_samples,
    split_samples,
)
from lingxi.io_utils import load_yaml, set_seed, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_lingxi.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    set_seed(config.get("seed", 42))
    paths = config["paths"]
    sampling = config["sampling"]
    include_system = config.get("format", {}).get("include_system_prompt", True)

    samples = []
    samples.extend(
        build_soulchat_samples(paths["soulchat_dir"], sampling["soulchat_max_samples"], include_system)
    )
    samples.extend(build_psyqa_samples(paths["psyqa_dir"], sampling["psyqa_max_samples"], include_system))

    samples = deduplicate_samples(samples)
    if args.limit is not None:
        samples = samples[: args.limit]

    train, valid, test = split_samples(
        samples,
        valid_ratio=sampling["valid_ratio"],
        test_ratio=sampling["test_ratio"],
        seed=config.get("seed", 42),
    )
    write_jsonl(train, paths["train_file"])
    write_jsonl(valid, paths["valid_file"])
    write_jsonl(test, paths["test_file"])
    write_jsonl(samples[:20], paths["sample_file"])

    source_counts = Counter(sample["source"] for sample in samples)
    emotion_counts = Counter(sample["emotion"] for sample in samples)
    summary = {
        "total": len(samples),
        "train": len(train),
        "valid": len(valid),
        "test": len(test),
        "source_counts": dict(source_counts),
        "emotion_counts": dict(emotion_counts),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not samples:
        print("No samples were built. Check raw dataset paths.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
