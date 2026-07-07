#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
LOG_DIR = Path(__file__).resolve().parent / "logs"

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lingxi.memory import (  # noqa: E402
    append_memory,
    build_memory_prompt,
    infer_emotion,
    load_memory,
    normalize_emotion,
    save_memory,
)
from lingxi.safety import check_safety  # noqa: E402


JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


EXPERIMENT_DIRS = {
    "SFT 正式版": "outputs/lingxi-qwen25-1p5b-lora-30min",
    "rank4_steps175": "outputs/experiments/rank4_steps175",
    "rank8_steps90": "outputs/experiments/rank8_steps90",
    "rank8_steps175": "outputs/experiments/rank8_steps175",
    "rank8_steps175_neftune": "outputs/experiments/rank8_steps175_neftune",
    "rank8_steps350": "outputs/experiments/rank8_steps350",
    "rank16_steps175": "outputs/experiments/rank16_steps175",
    "rank16_steps175_rslora": "outputs/experiments/rank16_steps175_rslora",
    "DPO policy": "outputs/lingxi-qwen25-1p5b-dpo/policy",
}


REPORT_FILES = [
    "reports/training_summary.md",
    "reports/domain_qa_eval.md",
    "reports/hyperparameter_analysis.md",
    "reports/dpo_domain_qa_eval.md",
    "reports/dpo_alignment_report.md",
    "reports/before_after_compare.md",
    "reports/memory_module_design.md",
    "reports/safety_boundary_design.md",
    "reports/neftune_experiment_design.md",
    "reports/rslora_experiment_design.md",
]


CONFIG_FILES = [
    "configs/data_lingxi.yaml",
    "configs/qwen25_1p5b_qlora_30min.yaml",
    "configs/experiments/qwen25_1p5b_lora_base.yaml",
    "configs/experiments/qwen25_1p5b_lora_rank4_steps175.yaml",
    "configs/experiments/qwen25_1p5b_lora_rank8_steps90.yaml",
    "configs/experiments/qwen25_1p5b_lora_rank8_steps175.yaml",
    "configs/experiments/qwen25_1p5b_lora_rank8_steps175_neftune.yaml",
    "configs/experiments/qwen25_1p5b_lora_rank8_steps350.yaml",
    "configs/experiments/qwen25_1p5b_lora_rank16_steps175.yaml",
    "configs/experiments/qwen25_1p5b_lora_rank16_steps175_rslora.yaml",
    "configs/dpo/qwen25_1p5b_dpo.yaml",
]


COMMANDS: dict[str, dict[str, Any]] = {
    "check_env": {
        "label": "环境检查",
        "command": ["python", "scripts/check_env.py"],
        "description": "检查 CUDA、模型路径、数据文件和核心依赖。",
        "kind": "diagnostic",
    },
    "download_assets_dry": {
        "label": "下载预检",
        "command": ["python", "scripts/download_assets.py", "--dry-run"],
        "description": "只检查模型和数据集下载目标，不写入大文件。",
        "kind": "data",
    },
    "prepare_dataset": {
        "label": "构建指令数据",
        "command": ["python", "scripts/prepare_dataset.py", "--config", "configs/data_lingxi.yaml"],
        "description": "重新生成 train/valid/test 指令数据集。",
        "kind": "data",
    },
    "train_formal": {
        "label": "正式 SFT 训练",
        "command": [
            "accelerate",
            "launch",
            "--num_processes",
            "2",
            "--mixed_precision",
            "bf16",
            "--main_process_port",
            "0",
            "scripts/train_lora.py",
            "--config",
            "configs/qwen25_1p5b_qlora_30min.yaml",
            "--resume-from-checkpoint",
            "auto",
        ],
        "description": "双卡运行正式 QLoRA-SFT 配置。",
        "kind": "train",
    },
    "smoke_train": {
        "label": "训练冒烟",
        "command": [
            "accelerate",
            "launch",
            "scripts/train_lora.py",
            "--config",
            "configs/qwen25_1p5b_qlora_30min.yaml",
            "--max-steps",
            "2",
            "--max-train-samples",
            "8",
            "--max-eval-samples",
            "4",
        ],
        "description": "极小步数验证训练脚本可启动。",
        "kind": "train",
    },
    "run_domain_experiments": {
        "label": "LoRA 消融全套",
        "command": ["bash", "scripts/run_domain_experiments.sh"],
        "description": "训练 rank/steps/NEFTune/rsLoRA 并生成对比报告。",
        "kind": "train",
    },
    "evaluate_domain": {
        "label": "领域问答评测",
        "command": [
            "python",
            "scripts/evaluate_domain_qa.py",
            "--prompts",
            "examples/eval_prompts.jsonl",
            "--out-json",
            "reports/domain_qa_eval.json",
            "--out-md",
            "reports/domain_qa_eval.md",
            "--max-new-tokens",
            "220",
            "--adapter",
            "rank4_steps175=outputs/experiments/rank4_steps175",
            "--adapter",
            "rank8_steps90=outputs/experiments/rank8_steps90",
            "--adapter",
            "rank8_steps175=outputs/experiments/rank8_steps175",
            "--adapter",
            "rank8_steps175_neftune=outputs/experiments/rank8_steps175_neftune",
            "--adapter",
            "rank8_steps350=outputs/experiments/rank8_steps350",
            "--adapter",
            "rank16_steps175=outputs/experiments/rank16_steps175",
            "--adapter",
            "rank16_steps175_rslora=outputs/experiments/rank16_steps175_rslora",
        ],
        "description": "重新生成基础模型与各 LoRA adapter 的同题回复对比。",
        "kind": "eval",
    },
    "summarize_experiments": {
        "label": "汇总超参数报告",
        "command": [
            "python",
            "scripts/summarize_experiments.py",
            "--configs",
            "configs/experiments/qwen25_1p5b_lora_rank*_steps*.yaml",
            "--eval-json",
            "reports/domain_qa_eval.json",
            "--out",
            "reports/hyperparameter_analysis.md",
        ],
        "description": "汇总 rank、轮数、NEFTune、rsLoRA 分析。",
        "kind": "report",
    },
    "build_dpo_dataset": {
        "label": "构造 DPO 数据",
        "command": ["python", "scripts/build_dpo_dataset.py", "--max-samples", "400"],
        "description": "生成 400 条 prompt/chosen/rejected 偏好数据。",
        "kind": "data",
    },
    "run_dpo_pipeline": {
        "label": "DPO 二阶段全套",
        "command": ["bash", "scripts/run_dpo_pipeline.sh"],
        "description": "构造 DPO 数据、训练 policy adapter、生成 DPO 对比报告。",
        "kind": "train",
    },
    "infer_compare": {
        "label": "微调前后对比",
        "command": [
            "python",
            "scripts/infer_compare.py",
            "--model",
            "models/Qwen2.5-1.5B-Instruct",
            "--adapter",
            "outputs/lingxi-qwen25-1p5b-lora-30min",
            "--prompts",
            "examples/eval_prompts.jsonl",
            "--out",
            "reports/before_after_compare.md",
        ],
        "description": "生成正式 SFT adapter 的前后回复对比。",
        "kind": "eval",
    },
    "compile_check": {
        "label": "脚本语法检查",
        "command": [
            "python",
            "-m",
            "py_compile",
            "scripts/train_lora.py",
            "scripts/train_dpo.py",
            "scripts/evaluate_domain_qa.py",
            "scripts/summarize_experiments.py",
            "scripts/summarize_dpo_alignment.py",
            "webapp/server.py",
        ],
        "description": "检查关键 Python 脚本语法。",
        "kind": "diagnostic",
    },
}


def now() -> float:
    return time.time()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def safe_repo_path(raw_path: str) -> Path:
    normalized = unquote(raw_path).strip().lstrip("/")
    candidate = (ROOT / normalized).resolve()
    if ROOT not in candidate.parents and candidate != ROOT:
        raise ValueError("path escapes project root")
    return candidate


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for count, _ in enumerate(f, start=1):
            pass
    return count


def file_info(path: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": rel(path) if path.is_absolute() and (ROOT in path.parents or path == ROOT) else str(path),
        "exists": exists,
        "size": path.stat().st_size if exists and path.is_file() else 0,
        "mtime": path.stat().st_mtime if exists else None,
    }


def run_text(command: list[str], timeout: int = 8) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return result.returncode, result.stdout
    except Exception as exc:
        return 1, str(exc)


def run_split(command: list[str], timeout: int = 8) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as exc:
        return 1, "", str(exc)


def gpu_status() -> list[dict[str, Any]]:
    code, output = run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=4,
    )
    if code != 0:
        return []
    gpus = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        gpus.append(
            {
                "index": parts[0],
                "name": parts[1],
                "memory_used": int(float(parts[2])),
                "memory_total": int(float(parts[3])),
                "utilization": int(float(parts[4])),
                "temperature": int(float(parts[5])),
            }
        )
    return gpus


def load_metrics_file(root: Path, kind: str = "train") -> dict[str, Any]:
    metric_path = root / ("dpo_metrics.json" if kind == "dpo" else "train_metrics.json")
    if not metric_path.exists():
        return {}
    try:
        return read_json(metric_path)
    except Exception:
        return {}


def collect_experiments() -> list[dict[str, Any]]:
    rows = []
    for label, root_text in EXPERIMENT_DIRS.items():
        root_path = ROOT / root_text
        is_dpo = label == "DPO policy"
        metrics_root = ROOT / "outputs/lingxi-qwen25-1p5b-dpo" if is_dpo else root_path
        metrics = load_metrics_file(metrics_root, "dpo" if is_dpo else "train")
        adapter_path = root_path / "adapter_model.safetensors"
        rows.append(
            {
                "label": label,
                "path": root_text,
                "adapter_exists": adapter_path.exists(),
                "metrics_exists": bool(metrics),
                "train_loss": metrics.get("train_loss"),
                "eval_loss": metrics.get("eval_loss"),
                "perplexity": metrics.get("perplexity"),
                "reward_accuracy": metrics.get("eval_rewards/accuracies"),
                "reward_margin": metrics.get("eval_rewards/margins"),
                "runtime": metrics.get("train_runtime"),
                "step_per_second": metrics.get("train_steps_per_second"),
            }
        )
    return rows


def collect_reports() -> list[dict[str, Any]]:
    reports = []
    for report in REPORT_FILES:
        path = ROOT / report
        title = report
        if path.exists():
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as f:
                    first = f.readline().strip()
                if first.startswith("#"):
                    title = first.lstrip("#").strip()
            except Exception:
                pass
        item = file_info(path)
        item["title"] = title
        reports.append(item)
    return reports


def collect_data_status() -> dict[str, Any]:
    files = {
        "train": ROOT / "data/processed/lingxi_train.jsonl",
        "valid": ROOT / "data/processed/lingxi_valid.jsonl",
        "test": ROOT / "data/processed/lingxi_test.jsonl",
        "dpo_train": ROOT / "data/processed/dpo_train.jsonl",
        "dpo_valid": ROOT / "data/processed/dpo_valid.jsonl",
        "eval_prompts": ROOT / "examples/eval_prompts.jsonl",
        "lingxi_sample": ROOT / "examples/lingxi_sample.jsonl",
        "dpo_sample": ROOT / "examples/dpo_sample.jsonl",
    }
    return {
        key: {
            **file_info(path),
            "lines": line_count(path),
        }
        for key, path in files.items()
    }


def sample_jsonl(path: Path, limit: int = 5) -> list[Any]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append(line)
            if len(rows) >= limit:
                break
    return rows


def job_snapshot(job: dict[str, Any]) -> dict[str, Any]:
    proc = job.get("process")
    running = proc is not None and proc.poll() is None
    status = "running" if running else job.get("status", "unknown")
    if proc is not None and not running and status == "running":
        status = "success" if proc.returncode == 0 else "failed"
        job["status"] = status
        job["returncode"] = proc.returncode
        job["ended_at"] = job.get("ended_at") or now()
    return {
        "id": job["id"],
        "key": job["key"],
        "label": job["label"],
        "command": job["command"],
        "status": status,
        "returncode": job.get("returncode"),
        "started_at": job.get("started_at"),
        "ended_at": job.get("ended_at"),
        "log_path": rel(job["log_path"]),
    }


def watch_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return
    proc = job["process"]
    returncode = proc.wait()
    try:
        job.get("log_file").close()
    except Exception:
        pass
    with JOBS_LOCK:
        job["returncode"] = returncode
        job["ended_at"] = now()
        job["status"] = "success" if returncode == 0 else "failed"


def start_job(key: str) -> dict[str, Any]:
    if key not in COMMANDS:
        raise KeyError(f"unknown command: {key}")
    spec = COMMANDS[key]
    job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{job_id}-{key}.log"
    command = list(spec["command"])
    log_file = log_path.open("w", encoding="utf-8")
    log_file.write("$ " + " ".join(command) + "\n\n")
    log_file.flush()
    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )
    job = {
        "id": job_id,
        "key": key,
        "label": spec["label"],
        "command": command,
        "status": "running",
        "returncode": None,
        "started_at": now(),
        "ended_at": None,
        "log_path": log_path,
        "process": proc,
        "log_file": log_file,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    thread = threading.Thread(target=watch_job, args=(job_id,), daemon=True)
    thread.start()
    return job_snapshot(job)


def stop_job(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise KeyError(job_id)
    proc = job.get("process")
    if proc is not None and proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        job["status"] = "stopping"
    return job_snapshot(job)


def build_status() -> dict[str, Any]:
    experiments = collect_experiments()
    reports = collect_reports()
    memory_records = load_memory(ROOT / "data/memory/memory.json")
    return {
        "project": "灵犀：基于 LoRA 微调的家庭陪伴情绪支持对话助手",
        "root": str(ROOT),
        "time": now(),
        "gpus": gpu_status(),
        "model": file_info(ROOT / "models/Qwen2.5-1.5B-Instruct/model.safetensors"),
        "data": collect_data_status(),
        "experiments": experiments,
        "reports": reports,
        "memory_count": len(memory_records),
        "latest_memory": memory_records[-3:],
        "jobs": [job_snapshot(job) for job in sorted(JOBS.values(), key=lambda item: item["started_at"], reverse=True)[:8]],
    }


class LingxiHandler(BaseHTTPRequestHandler):
    server_version = "LingxiFrontend/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw.strip() else {}

    def write_json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def write_text(self, text: str, status: int = 200, content_type: str = "text/plain; charset=utf-8") -> None:
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def write_error(self, status: int, message: str) -> None:
        self.write_json({"error": message}, status=status)

    def write_sse_headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True

    def send_sse(self, event: str, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False)
        packet = f"event: {event}\ndata: {data}\n\n".encode("utf-8")
        self.wfile.write(packet)
        self.wfile.flush()

    def send_static(self, request_path: str) -> None:
        if request_path == "/":
            request_path = "/index.html"
        raw = request_path.removeprefix("/static/")
        if request_path.startswith("/static/"):
            path = (STATIC_DIR / raw).resolve()
        else:
            path = (STATIC_DIR / request_path.lstrip("/")).resolve()
        if STATIC_DIR not in path.parents and path != STATIC_DIR:
            self.write_error(HTTPStatus.FORBIDDEN, "invalid static path")
            return
        if not path.exists() or not path.is_file():
            self.write_error(HTTPStatus.NOT_FOUND, "static file not found")
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            if path == "/api/status":
                self.write_json(build_status())
            elif path == "/api/metrics":
                self.write_json({"experiments": collect_experiments()})
            elif path == "/api/reports":
                self.write_json({"reports": collect_reports()})
            elif path == "/api/report":
                report_path = query.get("path", [""])[0]
                path_obj = safe_repo_path(report_path)
                if not path_obj.exists():
                    self.write_error(404, "report not found")
                else:
                    self.write_json({"path": rel(path_obj), "content": path_obj.read_text(encoding="utf-8", errors="ignore")})
            elif path == "/api/configs":
                self.write_json({"configs": [file_info(ROOT / item) for item in CONFIG_FILES]})
            elif path == "/api/config":
                config_path = query.get("path", [""])[0]
                path_obj = safe_repo_path(config_path)
                self.write_json({"path": rel(path_obj), "content": path_obj.read_text(encoding="utf-8", errors="ignore")})
            elif path == "/api/data":
                self.write_json({"data": collect_data_status()})
            elif path == "/api/sample":
                sample_path = query.get("path", ["examples/eval_prompts.jsonl"])[0]
                limit = int(query.get("limit", ["5"])[0])
                path_obj = safe_repo_path(sample_path)
                self.write_json({"path": rel(path_obj), "items": sample_jsonl(path_obj, limit)})
            elif path == "/api/memory":
                records = load_memory(ROOT / "data/memory/memory.json")
                self.write_json({"records": records})
            elif path == "/api/commands":
                self.write_json(
                    {
                        "commands": [
                            {
                                "key": key,
                                "label": spec["label"],
                                "description": spec["description"],
                                "kind": spec["kind"],
                                "command": spec["command"],
                            }
                            for key, spec in COMMANDS.items()
                        ]
                    }
                )
            elif path == "/api/jobs":
                with JOBS_LOCK:
                    jobs = [job_snapshot(job) for job in sorted(JOBS.values(), key=lambda item: item["started_at"], reverse=True)]
                self.write_json({"jobs": jobs})
            elif path.startswith("/api/jobs/"):
                self.handle_job_get(path)
            elif path.startswith("/static/") or path == "/":
                self.send_static(path)
            else:
                self.send_static(path)
        except Exception as exc:
            self.write_error(500, str(exc))

    def handle_job_get(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) < 3:
            self.write_error(404, "job not found")
            return
        job_id = parts[2]
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if not job:
            self.write_error(404, "job not found")
            return
        if len(parts) == 4 and parts[3] == "log":
            text = job["log_path"].read_text(encoding="utf-8", errors="ignore") if job["log_path"].exists() else ""
            self.write_json({"id": job_id, "log": text})
            return
        self.write_json({"job": job_snapshot(job)})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/safety":
                body = self.read_json_body()
                result = check_safety(str(body.get("text", "")))
                self.write_json(
                    {
                        "triggered": result.triggered,
                        "level": result.level,
                        "categories": result.categories,
                        "matched_keywords": result.matched_keywords,
                        "reply": result.reply,
                    }
                )
            elif path == "/api/memory":
                body = self.read_json_body()
                record = append_memory(
                    ROOT / "data/memory/memory.json",
                    emotion=normalize_emotion(body.get("emotion")),
                    user_text=str(body.get("user_text", "")),
                    robot_reply=str(body.get("robot_reply", "")),
                )
                self.write_json({"record": record, "records": load_memory(ROOT / "data/memory/memory.json")})
            elif path == "/api/memory/prompt":
                body = self.read_json_body()
                records = load_memory(ROOT / "data/memory/memory.json")
                user_text = str(body.get("user_text", ""))
                emotion = normalize_emotion(body.get("emotion")) if body.get("emotion") else infer_emotion(user_text)
                prompt = build_memory_prompt(
                    user_text=user_text,
                    scene=str(body.get("scene", "家庭陪伴")),
                    current_emotion=emotion,
                    records=records,
                    window=int(body.get("window", 3)),
                )
                safety = check_safety(user_text)
                self.write_json(
                    {
                        "emotion": emotion,
                        "prompt": prompt,
                        "safety": {
                            "triggered": safety.triggered,
                            "level": safety.level,
                            "categories": safety.categories,
                            "matched_keywords": safety.matched_keywords,
                            "reply": safety.reply,
                        },
                    }
                )
            elif path == "/api/chat":
                self.handle_chat()
            elif path == "/api/chat/stream":
                self.handle_chat_stream()
            elif path == "/api/jobs":
                body = self.read_json_body()
                job = start_job(str(body.get("command", "")))
                self.write_json({"job": job}, status=201)
            elif path.startswith("/api/jobs/") and path.endswith("/stop"):
                job_id = path.strip("/").split("/")[2]
                self.write_json({"job": stop_job(job_id)})
            else:
                self.write_error(404, "unknown endpoint")
        except KeyError as exc:
            self.write_error(404, str(exc))
        except Exception as exc:
            self.write_error(500, str(exc))

    def handle_chat(self) -> None:
        body = self.read_json_body()
        user_text = str(body.get("user_text", "")).strip()
        if not user_text:
            self.write_error(400, "user_text is required")
            return
        scene = str(body.get("scene", "家庭陪伴"))
        emotion = normalize_emotion(body.get("emotion")) if body.get("emotion") else infer_emotion(user_text)
        window = int(body.get("window", 3))
        no_save = bool(body.get("no_save", False))
        disable_safety = bool(body.get("disable_safety", False))
        mode = str(body.get("mode", "dry"))
        safety = check_safety(user_text)

        if safety.triggered and not disable_safety:
            if not no_save:
                append_memory(
                    ROOT / "data/memory/memory.json",
                    emotion="危机" if safety.level == "high" else "焦虑",
                    user_text=user_text,
                    robot_reply=safety.reply,
                )
            self.write_json(
                {
                    "mode": "safety",
                    "emotion": "危机" if safety.level == "high" else "焦虑",
                    "reply": safety.reply,
                    "safety": {
                        "triggered": safety.triggered,
                        "level": safety.level,
                        "categories": safety.categories,
                        "matched_keywords": safety.matched_keywords,
                    },
                    "records": load_memory(ROOT / "data/memory/memory.json"),
                }
            )
            return

        records = load_memory(ROOT / "data/memory/memory.json")
        prompt = build_memory_prompt(
            user_text=user_text,
            scene=scene,
            current_emotion=emotion,
            records=records,
            window=window,
        )
        if mode == "dry":
            self.write_json({"mode": "dry", "emotion": emotion, "prompt": prompt, "records": records})
            return

        if mode == "manual":
            reply = str(body.get("manual_reply", "")).strip()
            if not reply:
                self.write_error(400, "manual_reply is required for manual mode")
                return
            if not no_save:
                append_memory(ROOT / "data/memory/memory.json", emotion=emotion, user_text=user_text, robot_reply=reply)
            self.write_json(
                {
                    "mode": "manual",
                    "emotion": emotion,
                    "prompt": prompt,
                    "reply": reply,
                    "records": load_memory(ROOT / "data/memory/memory.json"),
                }
            )
            return

        adapter = str(body.get("adapter", "")).strip()
        command = [
            "python",
            "scripts/memory_chat.py",
            "--user",
            user_text,
            "--scene",
            scene,
            "--emotion",
            emotion,
            "--window",
            str(window),
            "--max-new-tokens",
            str(int(body.get("max_new_tokens", 220))),
        ]
        if adapter:
            command.extend(["--adapter", adapter])
        if no_save:
            command.append("--no-save")
        if disable_safety:
            command.append("--disable-safety")
        code, output, diagnostics = run_split(command, timeout=int(body.get("timeout", 600)))
        self.write_json(
            {
                "mode": "generate",
                "emotion": emotion,
                "prompt": prompt,
                "reply": output.strip(),
                "diagnostics": diagnostics.strip(),
                "returncode": code,
                "records": load_memory(ROOT / "data/memory/memory.json"),
            },
            status=200 if code == 0 else 500,
        )

    def handle_chat_stream(self) -> None:
        body = self.read_json_body()
        user_text = str(body.get("user_text", "")).strip()
        if not user_text:
            self.write_error(400, "user_text is required")
            return

        scene = str(body.get("scene", "家庭陪伴"))
        emotion = normalize_emotion(body.get("emotion")) if body.get("emotion") else infer_emotion(user_text)
        window = int(body.get("window", 3))
        no_save = bool(body.get("no_save", False))
        disable_safety = bool(body.get("disable_safety", False))
        mode = str(body.get("mode", "generate"))

        self.write_sse_headers()
        self.send_sse("stage", {"key": "receive", "title": "接收用户输入", "detail": f"{len(user_text)} 字"})
        self.send_sse("stage", {"key": "emotion", "title": "情绪识别", "detail": emotion})

        safety = check_safety(user_text)
        safety_payload = {
            "triggered": safety.triggered,
            "level": safety.level,
            "categories": safety.categories,
            "matched_keywords": safety.matched_keywords,
        }
        self.send_sse("safety", safety_payload)
        if safety.triggered and not disable_safety:
            reply = safety.reply
            if not no_save:
                self.send_sse("stage", {"key": "memory_write", "title": "写入安全边界记忆", "detail": safety.level})
                append_memory(
                    ROOT / "data/memory/memory.json",
                    emotion="危机" if safety.level == "high" else "焦虑",
                    user_text=user_text,
                    robot_reply=reply,
                )
            self.send_sse("final", {
                "mode": "safety",
                "emotion": "危机" if safety.level == "high" else "焦虑",
                "reply": reply,
                "records": load_memory(ROOT / "data/memory/memory.json"),
                "safety": safety_payload,
            })
            return

        records = load_memory(ROOT / "data/memory/memory.json")
        self.send_sse("stage", {"key": "memory_read", "title": "检索短期记忆", "detail": f"最近 {min(window, len(records))} / 共 {len(records)} 条"})
        prompt = build_memory_prompt(
            user_text=user_text,
            scene=scene,
            current_emotion=emotion,
            records=records,
            window=window,
        )
        self.send_sse("prompt", {"emotion": emotion, "prompt": prompt})
        self.send_sse("stage", {"key": "prompt", "title": "构建 Prompt", "detail": f"{len(prompt)} 字"})

        if mode == "dry":
            self.send_sse("final", {"mode": "dry", "emotion": emotion, "reply": prompt, "prompt": prompt, "records": records})
            return

        if mode == "manual":
            reply = str(body.get("manual_reply", "")).strip()
            if not reply:
                self.send_sse("error", {"error": "manual_reply is required for manual mode"})
                return
            if not no_save:
                self.send_sse("stage", {"key": "memory_write", "title": "写入手动回复记忆", "detail": emotion})
                append_memory(ROOT / "data/memory/memory.json", emotion=emotion, user_text=user_text, robot_reply=reply)
            self.send_sse("final", {
                "mode": "manual",
                "emotion": emotion,
                "prompt": prompt,
                "reply": reply,
                "records": load_memory(ROOT / "data/memory/memory.json"),
            })
            return

        adapter = str(body.get("adapter", "")).strip()
        command = [
            "python",
            "-u",
            "scripts/memory_chat.py",
            "--stream-jsonl",
            "--user",
            user_text,
            "--scene",
            scene,
            "--emotion",
            emotion,
            "--window",
            str(window),
            "--max-new-tokens",
            str(int(body.get("max_new_tokens", 220))),
        ]
        if adapter:
            command.extend(["--adapter", adapter])
        if no_save:
            command.append("--no-save")
        if disable_safety:
            command.append("--disable-safety")

        self.send_sse("stage", {"key": "subprocess", "title": "启动推理进程", "detail": "stream-jsonl"})
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stderr_queue: queue.Queue[str] = queue.Queue()

        def read_stderr() -> None:
            assert proc.stderr is not None
            for err_line in proc.stderr:
                stderr_queue.put(err_line.rstrip())

        threading.Thread(target=read_stderr, daemon=True).start()

        reply_chunks: list[str] = []
        final_payload: dict[str, Any] | None = None
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            while not stderr_queue.empty():
                self.send_sse("diagnostic", {"text": stderr_queue.get()})
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                self.send_sse("diagnostic", {"text": line})
                continue
            event = str(payload.pop("event", "message"))
            if event == "token":
                chunk = str(payload.get("text", ""))
                reply_chunks.append(chunk)
            elif event == "final":
                final_payload = payload
            self.send_sse(event, payload)

        returncode = proc.wait()
        while not stderr_queue.empty():
            self.send_sse("diagnostic", {"text": stderr_queue.get()})
        if returncode != 0:
            self.send_sse("error", {"error": f"推理进程退出码 {returncode}", "returncode": returncode})
            return

        reply = str((final_payload or {}).get("reply") or "".join(reply_chunks)).strip()
        self.send_sse("stage", {"key": "memory_refresh", "title": "刷新记忆状态", "detail": "完成"})
        self.send_sse("final", {
            "mode": "generate",
            "emotion": emotion,
            "prompt": prompt,
            "reply": reply,
            "returncode": returncode,
            "records": load_memory(ROOT / "data/memory/memory.json"),
        })

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/memory":
            save_memory([], ROOT / "data/memory/memory.json")
            self.write_json({"records": []})
        else:
            self.write_error(404, "unknown endpoint")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), LingxiHandler)
    print(f"Lingxi frontend running at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
