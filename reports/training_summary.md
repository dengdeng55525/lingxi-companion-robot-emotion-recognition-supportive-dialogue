# Lingxi First-Stage Training Summary

## Environment

- Base model: `Qwen2.5-1.5B-Instruct`
- Framework: Hugging Face Transformers + PEFT QLoRA
- GPU: 2 x NVIDIA GeForce RTX 4090 D, 24GB
- Torch: `2.1.2+cu118`

## Dataset

The processed instruction dataset uses only the requested sources after reducing local raw data:

| Dataset | Language | Use |
| --- | --- | --- |
| SoulChatCorpus | Chinese | Empathy, listening, comfort, multi-turn emotional support |
| PsyQA example | Chinese | Long-form psychological support QA examples |

Generated split:

- Train: 2,790
- Valid: 155
- Test: 155
- Total: 3,100

Source counts:

- SoulChatCorpus: 3,000
- PsyQA example: 100

The processed samples contain only dialogue messages and dataset metadata; no auxiliary behavior-control field is used.

## Smoke Training

- LoRA rank: 8
- LoRA alpha: 16
- Trainable params: 9,232,384
- Max steps: 2
- Train loss: 1.5330
- Eval loss: 2.4257
- Eval perplexity: 11.3105

This run verifies the full pipeline only. The official training setup for this project is `configs/qwen25_1p5b_qlora_30min.yaml`; the file directly defines `max_steps=175`, `max_train_samples=2790`, and `max_eval_samples=155`. This keeps one formal run lightweight on dual RTX 4090 D while still producing a valid LoRA adapter and training metrics.

The official config is optimized for dual RTX 4090 D training by using per-device micro-batch size 8, gradient accumulation 1, gradient checkpointing, pre-tokenized datasets, and multi-worker DataLoader prefetching. With two GPUs, the global effective batch size is 16, so the 175-step formal run approximately covers the current 2,790-sample training split once.

Local GPU pressure test on the RTX 4090 D:

- `batch_size=8`, no gradient checkpointing: reached 100% GPU utilization but failed with CUDA OOM near the logits allocation.
- `batch_size=6`, no gradient checkpointing: completed, peak GPU utilization 84%, peak memory 23,637 MiB.
- `batch_size=8`, gradient checkpointing enabled: completed, peak GPU utilization 100%, peak memory 22,763 MiB, peak power 304.8 W, and 10-step smoke throughput around 16.6 samples/s.

Dual-GPU benchmark:

- Launch command: `accelerate launch --num_processes 2 --mixed_precision bf16 scripts/train_lora.py --config configs/qwen25_1p5b_qlora_30min.yaml`
- 100-step benchmark: 56.67 seconds training runtime, 1.765 steps/s, 28.23 samples/s.
- GPU 0: peak utilization 100%, peak memory 20,603 MiB, peak power 288.6 W.
- GPU 1: peak utilization 100%, peak memory 17,993 MiB, peak power 266.7 W.
- Current one-epoch estimate for 2,790 training samples: `ceil(2790 / (8 * 2)) = 175` steps. At 1.765 steps/s, training compute is about 1.7 minutes; including model loading, pre-tokenization, evaluation, and checkpoint saving, the practical wall-clock estimate is about 3-5 minutes.
