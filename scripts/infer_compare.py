#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lingxi.constants import SYSTEM_PROMPT
from lingxi.io_utils import read_jsonl, resolve_path


def generate(model, tokenizer, user_text: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def load_model(model_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    return model, tokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default="outputs/lingxi-qwen25-1p5b-lora")
    parser.add_argument("--prompts", default="examples/eval_prompts.jsonl")
    parser.add_argument("--out", default="reports/before_after_compare.md")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--skip-adapter", action="store_true")
    args = parser.parse_args()

    prompts = read_jsonl(args.prompts)
    model_path = resolve_path(args.model)
    adapter_path = resolve_path(args.adapter)
    report_path = resolve_path(args.out)

    base_model, tokenizer = load_model(model_path)
    base_model.eval()
    base_outputs = {
        item["id"]: generate(base_model, tokenizer, item["user"], args.max_new_tokens)
        for item in prompts
    }
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    adapter_outputs = {}
    if not args.skip_adapter and adapter_path.exists():
        from peft import PeftModel

        adapter_base_model, tokenizer = load_model(model_path)
        adapter_model = PeftModel.from_pretrained(adapter_base_model, str(adapter_path))
        adapter_model.eval()
        adapter_outputs = {
            item["id"]: generate(adapter_model, tokenizer, item["user"], args.max_new_tokens)
            for item in prompts
        }

    lines = ["# Lingxi Before/After Comparison", ""]
    for item in prompts:
        user_text = item["user"]
        base_response = base_outputs[item["id"]]
        lines.extend(
            [
                f"## {item['id']}",
                "",
                f"**Emotion**: {item.get('emotion', '')}",
                "",
                f"**User**: {user_text}",
                "",
                "**Base model**",
                "",
                base_response,
                "",
            ]
        )
        if item["id"] in adapter_outputs:
            lines.extend(["**LoRA adapter**", "", adapter_outputs[item["id"]], ""])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
