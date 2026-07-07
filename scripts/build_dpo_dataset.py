#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lingxi.preference_data import build_preference_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/lingxi_train.jsonl")
    parser.add_argument("--train-file", default="data/processed/dpo_train.jsonl")
    parser.add_argument("--valid-file", default="data/processed/dpo_valid.jsonl")
    parser.add_argument("--sample-file", default="examples/dpo_sample.jsonl")
    parser.add_argument("--max-samples", type=int, default=400)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = build_preference_dataset(
        input_file=args.input,
        train_file=args.train_file,
        valid_file=args.valid_file,
        sample_file=args.sample_file,
        max_samples=args.max_samples,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["total"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
