#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lingxi.io_utils import clear_proxy_env, ensure_dir, proxy_free_env, resolve_path


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SOULCHAT_ID = "YIRONGCHEN/SoulChatCorpus"
PSYQA_REPO = "https://github.com/thu-coai/PsyQA.git"


def run(cmd: list[str], *, dry_run: bool) -> bool:
    printable = " ".join(cmd)
    print(f"$ {printable}")
    if dry_run:
        return True
    result = subprocess.run(cmd, cwd=ROOT, env=proxy_free_env(), check=False)
    return result.returncode == 0


def path_ready(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def download_model(target: Path, dry_run: bool, hf_endpoint: str) -> None:
    if path_ready(target):
        print(f"Model already exists: {target}")
        return
    ensure_dir(target)
    print("Downloading model with ModelScope first.")
    if not dry_run:
        try:
            clear_proxy_env()
            from modelscope import snapshot_download

            snapshot_download(MODEL_ID, local_dir=str(target))
            if path_ready(target):
                return
        except Exception as exc:
            print(f"ModelScope model download failed: {exc}")
    else:
        print(f"dry-run: modelscope snapshot_download({MODEL_ID}, local_dir={target})")
        print(f"dry-run fallback: HF_ENDPOINT={hf_endpoint} huggingface_hub snapshot_download")
        return

    print("Falling back to Hugging Face mirror.")
    os.environ["HF_ENDPOINT"] = hf_endpoint
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def download_soulchat(target: Path, dry_run: bool) -> None:
    if path_ready(target):
        print(f"SoulChatCorpus already exists: {target}")
        return
    ensure_dir(target)
    cmd = [
        sys.executable,
        "-m",
        "modelscope.cli.cli",
        "download",
        "--dataset",
        SOULCHAT_ID,
        "--local_dir",
        str(target),
    ]
    if run(cmd, dry_run=dry_run) and path_ready(target):
        return
    print("Trying ModelScope git endpoint for SoulChatCorpus.")
    cmd = [
        "git",
        "-c",
        "http.proxy=",
        "-c",
        "https.proxy=",
        "clone",
        "--depth",
        "1",
        f"https://www.modelscope.cn/datasets/{SOULCHAT_ID}.git",
        str(target),
    ]
    run(cmd, dry_run=dry_run)


def reduce_soulchat_json(target: Path, keep_every: int, dry_run: bool) -> None:
    if keep_every <= 1:
        return
    path = target / "SoulChatCorpus-sft-multi-Turn.json"
    if not path.exists():
        return
    if dry_run:
        print(f"dry-run: keep every {keep_every}th SoulChat record in {path}")
        return

    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        print(f"Skip SoulChat reduction because {path} is not a JSON list.")
        return
    if len(records) <= 30000:
        print(f"SoulChatCorpus already reduced: {len(records)} records.")
        return

    reduced = records[::keep_every]
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(reduced, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    tmp.replace(path)
    print(f"Reduced SoulChatCorpus from {len(records)} to {len(reduced)} records.")


def clone_repo(repo: str, target: Path, dry_run: bool) -> None:
    if path_ready(target):
        print(f"Repository already exists: {target}")
        return
    ensure_dir(target.parent)
    cmd = [
        "git",
        "-c",
        "http.proxy=",
        "-c",
        "https.proxy=",
        "clone",
        "--depth",
        "1",
        repo,
        str(target),
    ]
    run(cmd, dry_run=dry_run)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--model-dir", default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--soulchat-keep-every", type=int, default=10)
    args = parser.parse_args()

    clear_proxy_env()
    print("Proxy environment variables cleared for this process.")
    print(f"HF mirror endpoint for fallback: {args.hf_endpoint}")

    if not args.skip_model:
        download_model(resolve_path(args.model_dir), args.dry_run, args.hf_endpoint)
    if not args.skip_data:
        soulchat_dir = resolve_path("data/raw/soulchat")
        download_soulchat(soulchat_dir, args.dry_run)
        reduce_soulchat_json(soulchat_dir, args.soulchat_keep_every, args.dry_run)
        clone_repo(PSYQA_REPO, resolve_path("data/raw/psyqa_repo"), args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
