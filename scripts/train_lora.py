#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lingxi.io_utils import load_yaml, resolve_path, set_seed
from lingxi.train_utils import ChatSFTDataset, DataCollatorForCausalChat, perplexity


def get_distributed_state() -> tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return local_rank, world_size


def rank_log(local_rank: int, message: str) -> None:
    print(f"[rank {local_rank}] {message}", flush=True)


def find_resume_checkpoint(output_dir: str) -> str | None:
    if not Path(output_dir).exists():
        return None
    return get_last_checkpoint(output_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen25_1p5b_qlora.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    set_seed(config.get("seed", 42))
    training = config["training"]
    local_rank, world_size = get_distributed_state()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if config["model"].get("load_in_4bit", True) and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. QLoRA 4-bit training requires a visible GPU.")

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model_path = str(resolve_path(config["model"]["base_model"]))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=config["model"].get("trust_remote_code", True))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if config["model"].get("load_in_4bit", True):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config["model"].get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, config["model"].get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=config["model"].get("bnb_4bit_use_double_quant", True),
        )

    device_map = None
    if torch.cuda.is_available():
        device_map = {"": local_rank} if world_size > 1 else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=config["model"].get("trust_remote_code", True),
        torch_dtype="auto",
        device_map=device_map,
        quantization_config=quantization_config,
    )
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config["training"].get("gradient_checkpointing", True),
        )

    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config["lora"]["target_modules"],
        use_rslora=config["lora"].get("use_rslora", False),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    max_train_samples = (
        args.max_train_samples if args.max_train_samples is not None else training.get("max_train_samples")
    )
    max_eval_samples = args.max_eval_samples if args.max_eval_samples is not None else training.get("max_eval_samples")
    pretokenize = config["data"].get("pretokenize", False)

    rank_log(local_rank, "building train dataset")
    train_dataset = ChatSFTDataset(
        config["data"]["train_file"],
        tokenizer,
        max_length=config["data"]["max_seq_length"],
        limit=max_train_samples,
        pretokenize=pretokenize,
    )
    rank_log(local_rank, f"train dataset ready: {len(train_dataset)}")
    rank_log(local_rank, "building eval dataset")
    eval_dataset = ChatSFTDataset(
        config["data"]["valid_file"],
        tokenizer,
        max_length=config["data"]["max_seq_length"],
        limit=max_eval_samples,
        pretokenize=pretokenize,
    )
    rank_log(local_rank, f"eval dataset ready: {len(eval_dataset)}")

    output_dir = str(resolve_path(args.output_dir or training["output_dir"]))
    max_steps = args.max_steps if args.max_steps is not None else training.get("max_steps", -1)
    dataloader_num_workers = training.get("dataloader_num_workers", 0)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training["num_train_epochs"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        per_device_eval_batch_size=training["per_device_eval_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"],
        warmup_ratio=training["warmup_ratio"],
        weight_decay=training["weight_decay"],
        logging_steps=training["logging_steps"],
        eval_steps=training["eval_steps"],
        save_steps=training["save_steps"],
        save_total_limit=training["save_total_limit"],
        gradient_checkpointing=training.get("gradient_checkpointing", True),
        optim=training.get("optim", "paged_adamw_8bit"),
        lr_scheduler_type=training.get("lr_scheduler_type", "cosine"),
        report_to=training.get("report_to", ["tensorboard"]),
        eval_strategy="steps",
        save_strategy="steps",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        max_steps=max_steps,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=training.get("dataloader_pin_memory", True),
        dataloader_persistent_workers=training.get("dataloader_persistent_workers", False)
        and dataloader_num_workers > 0,
        dataloader_prefetch_factor=training.get("dataloader_prefetch_factor")
        if dataloader_num_workers > 0
        else None,
        torch_empty_cache_steps=training.get("torch_empty_cache_steps"),
        eval_accumulation_steps=training.get("eval_accumulation_steps"),
        gradient_checkpointing_kwargs=training.get("gradient_checkpointing_kwargs"),
        neftune_noise_alpha=training.get("neftune_noise_alpha"),
        ddp_find_unused_parameters=False if world_size > 1 else None,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForCausalChat(tokenizer),
        tokenizer=tokenizer,
    )
    resume_from_checkpoint = args.resume_from_checkpoint or training.get("resume_from_checkpoint")
    if resume_from_checkpoint == "auto":
        resume_from_checkpoint = find_resume_checkpoint(output_dir)
        if resume_from_checkpoint is None and trainer.is_world_process_zero():
            print(f"No checkpoint found in {output_dir}; starting from scratch.")
    rank_log(local_rank, f"starting trainer.train(resume_from_checkpoint={resume_from_checkpoint})")
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    metrics = result.metrics
    eval_metrics = trainer.evaluate()
    metrics.update(eval_metrics)
    metrics["perplexity"] = perplexity(eval_metrics.get("eval_loss"))

    trainer.save_model(output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with (Path(output_dir) / "train_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        with (Path(output_dir) / "training_config_snapshot.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
