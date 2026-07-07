#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

NUM_PROCESSES="${NUM_PROCESSES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-220}"
SFT_ADAPTER="${SFT_ADAPTER:-outputs/lingxi-qwen25-1p5b-lora-30min}"
DPO_OUTPUT_ROOT="${DPO_OUTPUT_ROOT:-outputs/lingxi-qwen25-1p5b-dpo}"
DPO_ADAPTER="${DPO_ADAPTER:-${DPO_OUTPUT_ROOT}/policy}"

echo "[1/5] Building DPO preference dataset"
python scripts/build_dpo_dataset.py \
  --input data/processed/lingxi_train.jsonl \
  --train-file data/processed/dpo_train.jsonl \
  --valid-file data/processed/dpo_valid.jsonl \
  --sample-file examples/dpo_sample.jsonl \
  --max-samples 400 \
  --valid-ratio 0.1

if [[ ! -f "${SFT_ADAPTER}/adapter_model.safetensors" ]]; then
  echo "Missing SFT adapter: ${SFT_ADAPTER}/adapter_model.safetensors" >&2
  echo "Run first-stage SFT before DPO:" >&2
  echo "  accelerate launch --num_processes ${NUM_PROCESSES} --mixed_precision ${MIXED_PRECISION} scripts/train_lora.py --config configs/qwen25_1p5b_qlora_30min.yaml" >&2
  exit 2
fi

echo "[2/5] Training second-stage DPO adapter on GPU"
accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --main_process_port 0 \
  scripts/train_dpo.py --config configs/dpo/qwen25_1p5b_dpo.yaml

echo "[3/5] Evaluating base vs SFT vs DPO on domain QA prompts"
python scripts/evaluate_domain_qa.py \
  --prompts examples/eval_prompts.jsonl \
  --adapter "sft=${SFT_ADAPTER}" \
  --adapter "dpo=${DPO_ADAPTER}" \
  --out-json reports/dpo_domain_qa_eval.json \
  --out-md reports/dpo_domain_qa_eval.md \
  --max-new-tokens "${MAX_NEW_TOKENS}"

echo "[4/5] Summarizing DPO alignment report"
python scripts/summarize_dpo_alignment.py \
  --dpo-train data/processed/dpo_train.jsonl \
  --dpo-valid data/processed/dpo_valid.jsonl \
  --sft-metrics "${SFT_ADAPTER}/train_metrics.json" \
  --dpo-metrics "${DPO_OUTPUT_ROOT}/dpo_metrics.json" \
  --domain-eval reports/dpo_domain_qa_eval.json \
  --out reports/dpo_alignment_report.md

echo "[5/5] Done"
echo "Reports:"
echo "  reports/dpo_domain_qa_eval.md"
echo "  reports/dpo_alignment_report.md"
