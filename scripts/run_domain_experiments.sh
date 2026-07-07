#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

NUM_PROCESSES="${NUM_PROCESSES:-2}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-220}"

CONFIGS=(
  "configs/experiments/qwen25_1p5b_lora_rank4_steps175.yaml"
  "configs/experiments/qwen25_1p5b_lora_rank8_steps90.yaml"
  "configs/experiments/qwen25_1p5b_lora_rank8_steps175.yaml"
  "configs/experiments/qwen25_1p5b_lora_rank8_steps175_neftune.yaml"
  "configs/experiments/qwen25_1p5b_lora_rank8_steps350.yaml"
  "configs/experiments/qwen25_1p5b_lora_rank16_steps175.yaml"
  "configs/experiments/qwen25_1p5b_lora_rank16_steps175_rslora.yaml"
)

ADAPTER_ARGS=(
  --adapter "rank4_steps175=outputs/experiments/rank4_steps175"
  --adapter "rank8_steps90=outputs/experiments/rank8_steps90"
  --adapter "rank8_steps175=outputs/experiments/rank8_steps175"
  --adapter "rank8_steps175_neftune=outputs/experiments/rank8_steps175_neftune"
  --adapter "rank8_steps350=outputs/experiments/rank8_steps350"
  --adapter "rank16_steps175=outputs/experiments/rank16_steps175"
  --adapter "rank16_steps175_rslora=outputs/experiments/rank16_steps175_rslora"
)

echo "[1/4] Rebuilding processed dataset"
python scripts/prepare_dataset.py --config configs/data_lingxi.yaml

echo "[2/4] Training LoRA ablation adapters on GPU"
for config in "${CONFIGS[@]}"; do
  echo "Training ${config}"
  accelerate launch \
    --num_processes "${NUM_PROCESSES}" \
    --mixed_precision "${MIXED_PRECISION}" \
    --main_process_port 0 \
    scripts/train_lora.py --config "${config}" --resume-from-checkpoint auto
done

echo "[3/4] Evaluating base model and LoRA adapters on domain QA prompts"
python scripts/evaluate_domain_qa.py \
  --prompts examples/eval_prompts.jsonl \
  --out-json reports/domain_qa_eval.json \
  --out-md reports/domain_qa_eval.md \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  "${ADAPTER_ARGS[@]}"

echo "[4/4] Summarizing hyperparameter analysis"
python scripts/summarize_experiments.py \
  --configs "configs/experiments/qwen25_1p5b_lora_rank*_steps*.yaml" \
  --eval-json reports/domain_qa_eval.json \
  --out reports/hyperparameter_analysis.md

echo "Done."
echo "Reports:"
echo "  reports/domain_qa_eval.md"
echo "  reports/hyperparameter_analysis.md"
