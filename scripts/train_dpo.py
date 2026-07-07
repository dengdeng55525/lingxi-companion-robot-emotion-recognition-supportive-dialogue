#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lingxi.io_utils import load_yaml, read_jsonl, resolve_path, set_seed


def get_distributed_state() -> tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return local_rank, world_size


def rank_log(local_rank: int, message: str) -> None:
    print(f"[rank {local_rank}] {message}", flush=True)


def quantization_config(config: dict[str, Any]) -> BitsAndBytesConfig | None:
    if not config["model"].get("load_in_4bit", True):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config["model"].get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=getattr(torch, config["model"].get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=config["model"].get("bnb_4bit_use_double_quant", True),
    )


def load_base_model(config: dict[str, Any], local_rank: int, world_size: int):
    device_map = None
    if torch.cuda.is_available():
        device_map = {"": local_rank} if world_size > 1 else "auto"
    return AutoModelForCausalLM.from_pretrained(
        str(resolve_path(config["model"]["base_model"])),
        trust_remote_code=config["model"].get("trust_remote_code", True),
        torch_dtype="auto",
        device_map=device_map,
        quantization_config=quantization_config(config),
    )


def load_policy_model(config: dict[str, Any], local_rank: int, world_size: int):
    from peft import PeftModel, prepare_model_for_kbit_training

    sft_adapter = resolve_path(config["model"]["sft_adapter"])
    if not sft_adapter.exists():
        raise FileNotFoundError(
            f"SFT adapter not found: {sft_adapter}. Train stage 1 first or update model.sft_adapter."
        )

    policy = load_base_model(config, local_rank, world_size)
    if config["model"].get("load_in_4bit", True):
        policy = prepare_model_for_kbit_training(
            policy,
            use_gradient_checkpointing=config["training"].get("gradient_checkpointing", True),
        )
    policy = PeftModel.from_pretrained(policy, str(sft_adapter), adapter_name="policy", is_trainable=True)
    policy.load_adapter(str(sft_adapter), adapter_name="reference", is_trainable=False)
    policy.set_adapter("policy")
    policy.config.use_cache = False
    return policy


def split_assistant_suffix(tokenizer: Any, system: str, user: str, assistant: str) -> tuple[str, str]:
    prompt_messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    full_messages = prompt_messages + [{"role": "assistant", "content": assistant}]
    prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    if full.startswith(prompt):
        return prompt, full[len(prompt) :]
    return prompt, assistant + (tokenizer.eos_token or "")


def load_dpo_dataset(path: str, tokenizer: Any) -> Dataset:
    rows = []
    for record in read_jsonl(path):
        prompt, chosen = split_assistant_suffix(tokenizer, record["system"], record["user"], record["chosen"])
        _, rejected = split_assistant_suffix(tokenizer, record["system"], record["user"], record["rejected"])
        rows.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    return Dataset.from_list(rows)


def normalize_tokenized_rows(
    dataset: Dataset,
    pad_token_id: int,
    label_pad_token_id: int = -100,
) -> tuple[Dataset, int]:
    tokenized_columns = [
        column
        for column in dataset.column_names
        if column.endswith(("_input_ids", "_attention_mask", "_labels", "_pixel_values"))
    ]
    if not tokenized_columns:
        return dataset, 0

    def fill_value_for_column(column: str) -> int | float:
        if column.endswith("_labels"):
            return label_pad_token_id
        if column.endswith("_attention_mask"):
            return 0
        if column.endswith("_pixel_values"):
            return 0.0
        return pad_token_id

    def replace_none(value: Any, fill_value: int | float) -> tuple[Any, int]:
        if value is None:
            return fill_value, 1
        if isinstance(value, (list, tuple)):
            replaced_items = [replace_none(item, fill_value) for item in value]
            return [item for item, _ in replaced_items], sum(count for _, count in replaced_items)
        return value, 0

    rows: list[dict[str, Any]] = []
    replaced = 0
    for record in dataset:
        row = dict(record)
        for column in tokenized_columns:
            row[column], count = replace_none(row.get(column), fill_value_for_column(column))
            replaced += count
        rows.append(row)
    return Dataset.from_list(rows), replaced


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo/qwen25_1p5b_dpo.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    set_seed(config.get("seed", 42))
    training = config["training"]
    local_rank, world_size = get_distributed_state()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    elif config["model"].get("load_in_4bit", True):
        raise RuntimeError("CUDA is unavailable. DPO QLoRA training requires a visible GPU.")

    tokenizer = AutoTokenizer.from_pretrained(
        str(resolve_path(config["model"]["base_model"])),
        trust_remote_code=config["model"].get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rank_log(local_rank, "tokenizer ready")

    model = load_policy_model(config, local_rank, world_size)
    model.print_trainable_parameters()
    rank_log(local_rank, "policy/reference adapters ready")

    rank_log(local_rank, "building DPO train dataset")
    train_dataset = load_dpo_dataset(config["data"]["train_file"], tokenizer)
    rank_log(local_rank, f"DPO train dataset ready: {len(train_dataset)}")
    rank_log(local_rank, "building DPO eval dataset")
    eval_dataset = load_dpo_dataset(config["data"]["valid_file"], tokenizer)
    rank_log(local_rank, f"DPO eval dataset ready: {len(eval_dataset)}")

    output_dir = str(resolve_path(args.output_dir or training["output_dir"]))
    max_steps = args.max_steps if args.max_steps is not None else training.get("max_steps", -1)
    dataloader_num_workers = training.get("dataloader_num_workers", 0)
    dpo_args = DPOConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        num_train_epochs=training.get("num_train_epochs", 1),
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
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=training.get("dataloader_pin_memory", True),
        dataloader_persistent_workers=training.get("dataloader_persistent_workers", False)
        and dataloader_num_workers > 0,
        dataloader_prefetch_factor=training.get("dataloader_prefetch_factor")
        if dataloader_num_workers > 0
        else None,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        remove_unused_columns=False,
        beta=config["dpo"].get("beta", 0.1),
        loss_type=config["dpo"].get("loss_type", "sigmoid"),
        max_length=config["data"].get("max_length", 1024),
        max_prompt_length=config["data"].get("max_prompt_length", 512),
        max_target_length=config["data"].get("max_target_length", 512),
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        model_adapter_name="policy",
        ref_adapter_name="reference",
    )
    text_columns = [
        column
        for column in ("prompt", "chosen", "rejected")
        if column in getattr(trainer.train_dataset, "column_names", [])
    ]
    if text_columns:
        trainer.train_dataset = trainer.train_dataset.remove_columns(text_columns)
        trainer.eval_dataset = trainer.eval_dataset.remove_columns(text_columns)
    trainer.train_dataset, normalized_train = normalize_tokenized_rows(
        trainer.train_dataset,
        pad_token_id=tokenizer.pad_token_id,
    )
    trainer.eval_dataset, normalized_eval = normalize_tokenized_rows(
        trainer.eval_dataset,
        pad_token_id=tokenizer.pad_token_id,
    )
    rank_log(local_rank, "DPO trainer ready")
    rank_log(local_rank, f"DPO normalized None token values: train={normalized_train}, eval={normalized_eval}")
    rank_log(local_rank, "starting DPO trainer.train()")
    result = trainer.train()
    metrics = result.metrics
    eval_metrics = trainer.evaluate()
    metrics.update(eval_metrics)

    trainer.save_model(output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with (Path(output_dir) / "dpo_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        with (Path(output_dir) / "dpo_config_snapshot.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
