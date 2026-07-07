from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from .io_utils import read_jsonl


class ChatSFTDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer: Any,
        max_length: int,
        limit: int | None = None,
        pretokenize: bool = False,
    ) -> None:
        self.records = read_jsonl(path)
        if limit is not None:
            self.records = self.records[:limit]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.features = [self.encode(record) for record in self.records] if pretokenize else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        if self.features is not None:
            return self.features[index]
        return self.encode(self.records[index])

    def encode(self, record: dict[str, Any]) -> dict[str, list[int]]:
        messages = record["messages"]
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        full = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]
        labels = list(input_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        if all(label == -100 for label in labels):
            labels[-1] = input_ids[-1]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@dataclass
class DataCollatorForCausalChat:
    tokenizer: Any

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(item["input_ids"]) for item in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad_len = max_len - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(item["attention_mask"] + [0] * pad_len)
            batch["labels"].append(item["labels"] + [-100] * pad_len)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def perplexity(eval_loss: float | None) -> float | None:
    if eval_loss is None:
        return None
    try:
        return math.exp(eval_loss)
    except OverflowError:
        return float("inf")
