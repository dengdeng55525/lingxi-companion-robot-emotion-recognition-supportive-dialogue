# Lingxi: Emotion Recognition and Supportive Dialogue Module for Companion Robots

> 灵犀：面向陪伴机器人的情绪识别与支持对话模块

# AUTHOR: JunzeDing

>作者：丁俊泽

Lingxi is a full companion-robot engineering project. Beyond the emotion recognition and supportive-dialogue module, it integrates robot vision (face-emotion + object detection), voice input/output, LLM-driven companion dialogue with safe motion planning, RoboEyes expression display, and a parent-facing dashboard. This repository is the complete engineering backup of the Lingxi early-education companion robot.

本仓库面向比赛展示与复现实验，包含视觉感知（人脸情绪 + 物体识别）、语音交互、LLM 对话决策、底盘控制、RoboEyes 表情、家长端仪表盘、数据构建、QLoRA 微调、DPO 偏好对齐、情绪记忆、安全边界、Web 控制台和实验报告。系统定位为早教陪护机器人整机工程，情绪支持模块不用于医学诊断、心理治疗或药物建议。

## Robot Integration Overview

This repository also contains a sanitized backup of the current Lingxi companion robot integration code:

- `vision/`: VM-side vision bridge, AutoDL HTTP webcam uploader, dashboard API, and emotion JSON tools.
- `vision/frontend/`: real-time vision dashboard and parent-facing dashboard HTML pages.
- `vision/parent_dashboard_api.py`: parent dashboard API for emotion history, dialogue log, learning concepts, and dangerous-object alerts.
- `vision/parent_dialogue_logger.py`: ROS topic logger for child/robot dialogue history.
- `ros_ws/src/robot_voice/`: ROS voice input, AI companion dialogue, TTS, and motion-command integration.
- `ros_ws/src/robot_voice/scripts/ai_companion_node.py`: LLM-based companion decision node with safety action whitelist.
- `ros_ws/src/robot_voice/scripts/ssh_voice_control.py`: SSH-based chassis motion executor (`/ai_motion_text` -> `/cmd_vel`).
- `ros_ws/src/robot_voice/scripts/ssh_arecord_vosk_iat.py`: Vosk-based offline ASR from Jetson microphone.
- `ros_ws/src/robot_voice/scripts/ssh_reply_speaker.py`: Edge-TTS based speech output.
- `ros_ws/src/RoboEyes/`: RoboEyes display library/source used by the robot expression stack.
- `desktop_tools/`: launch and sync helper scripts used from the VM desktop.
- `desktop_tools/sync_emotion_to_jetson.py`: VM -> Jetson emotion sync for RoboEyes.
- `docs/`: runbooks and troubleshooting notes.
- `docs/灵犀早教陪护机器人技术报告.md`: full-system technical report.
- `docs/ROBOEYES_启动与排障手册.md`: RoboEyes launch and troubleshooting manual.
- `docs/YOLO5Face_EmotiEffLib_表情识别操作文档.md`: vision-model operations guide.

The current robot vision flow uses AutoDL/SeetaCloud for inference:

1. Jetson publishes camera frames through ROS.
2. The VM `web_video_server` exposes the ROS image as HTTP/MJPEG.
3. `vision/send_webcam_http.py` uploads JPEG frames to AutoDL.
4. AutoDL returns emotion and object detection JSON.
5. The VM writes `vision/outputs/latest_emotion.json` and `$HOME/.ros/latest_emotion.json`.
6. ROS AI dialogue reads the latest JSON for supportive replies, object questions, and RoboEyes emotion sync.

The full companion loop is:

```text
Camera frames (Jetson ROS)
  -> HTTP/MJPEG video stream (VM web_video_server)
  -> AutoDL emotion + object inference
  -> latest_emotion.json (full + lightweight)
  -> vision_result_api.py (dashboard)
  -> parent_dashboard_api.py (parent dashboard)
  -> ai_companion_node.py (LLM decision, safety, learning events)
  -> /ai_reply (Edge-TTS playback) + /ai_motion_text (SSH chassis control)
  -> sync_emotion_to_jetson.py (RoboEyes expression sync)
```

This GitHub backup intentionally does not contain real SSH passwords, API keys, or stream tokens. Keep values such as `ROBOTVISION_STREAM_TOKEN`, `OPENAI_API_KEY`, and SeetaCloud/AutoDL SSH credentials in local private notes or environment variables.


## Competition Positioning

陪伴机器人不能只完成普通问答，还需要理解用户情绪、记住近期状态，并在用户处于压力、焦虑或低落时给出更合适的回应。灵犀围绕这一需求构建了一个可接入机器人本体的情绪支持模块。

The core competition value is:

- Emotional understanding: detect emotional categories and intensity cues from user input.
- Supportive dialogue: generate responses with empathy, validation, comfort, and gentle guidance.
- Short-term memory: track recent emotional states and avoid isolated single-turn replies.
- Safety boundary: identify high-risk expressions and avoid unsafe medical or crisis advice.
- Robot-ready interface: expose a dialogue pipeline that can be connected to speech, expression, and motion modules.
- Reproducible training: provide SFT, DPO, ablation experiments, reports, and a Web console.

## Key Features

### 1. Emotion Recognition

The module infers the user's current emotional state from natural language input, including common states such as sadness, anxiety, stress, anger, loneliness, and neutral mood. The detected emotion is used as an explicit condition for response generation.

### 2. Supportive Dialogue Generation

The dialogue agent is fine-tuned from `Qwen2.5-1.5B-Instruct` with LoRA/QLoRA. It is optimized for warm, respectful, non-judgmental emotional support in daily companionship scenarios.

### 3. Short-Term Emotional Memory

The system stores recent conversation rounds in `data/memory/memory.json` during local runtime. Before generating a reply, it retrieves recent emotional context and builds a memory-aware prompt, so the assistant can respond with continuity instead of treating every message as an isolated input.

### 4. Safety Boundary

The safety module detects self-harm, harm to others, extreme distress, diagnosis, medication, and long-term sleep-related expressions. For high-risk or medical-boundary inputs, the system returns a safer support message and recommends contacting trusted people or professional help.

### 5. Web Competition Demo

The `webapp/` directory provides a local control console for demonstration. It supports companion dialogue, prompt debugging, safety check visibility, memory display, training tasks, report browsing, and dataset/config inspection.

## System Architecture

```text
User Input
   |
   v
Emotion Detection ---- Safety Boundary
   |                         |
   |                         v
   |                  Safe Response
   v
Short-Term Memory Retrieval
   |
   v
Prompt Construction
   |
   v
Qwen2.5 + LoRA/DPO Adapter
   |
   v
Supportive Reply
   |
   v
Robot Interface Extension
```

## Directory Structure

```text
vision/                Robot vision: frame upload, emotion/object JSON, dashboard API
vision/frontend/       Vision dashboard + parent dashboard HTML
vision/scripts/        Environment checks and launch helpers
vision/tests/          Vision and dashboard tests
ros_ws/src/robot_voice/  ROS voice/decision/motion package
ros_ws/src/RoboEyes/     Robot expression display library
desktop_tools/          VM-side launch and emotion-sync scripts
docs/                  Technical report, runbooks, troubleshooting manuals
configs/               Training and data configuration files
scripts/               Data processing, training, evaluation, and demo scripts
src/lingxi/            Core Python modules (dialogue support module)
examples/              Example datasets and evaluation prompts
reports/               Experiment reports and design documents
data/raw/              Raw datasets
data/processed/        Processed instruction datasets
data/memory/           Local short-term memory examples and runtime state
models/                Local base models, not committed
outputs/               LoRA/DPO adapters and experiment outputs
webapp/                Local Web demo console
```

## Uploaded Adapters

The base model `Qwen2.5-1.5B-Instruct` is not uploaded to this repository. It should be downloaded locally to `models/Qwen2.5-1.5B-Instruct`.

The repository contains fine-tuned adapters produced by this project:

| Name | Path | Description |
| --- | --- | --- |
| SFT adapter | `outputs/lingxi-qwen25-1p5b-lora-30min` | Main LoRA-SFT adapter |
| DPO policy | `outputs/lingxi-qwen25-1p5b-dpo/policy` | Preference-aligned adapter after SFT |
| rank4_steps175 | `outputs/experiments/rank4_steps175` | LoRA rank ablation |
| rank8_steps90 | `outputs/experiments/rank8_steps90` | Short training-step ablation |
| rank8_steps175 | `outputs/experiments/rank8_steps175` | Standard rank-8 baseline |
| rank8_steps175_neftune | `outputs/experiments/rank8_steps175_neftune` | NEFTune enhancement experiment |
| rank8_steps350 | `outputs/experiments/rank8_steps350` | Longer training-step ablation |
| rank16_steps175 | `outputs/experiments/rank16_steps175` | Higher-rank LoRA ablation |
| rank16_steps175_rslora | `outputs/experiments/rank16_steps175_rslora` | Rank-stabilized LoRA experiment |

## Quick Start

Install dependencies:

```bash
cd .
python -m pip install -r requirements.txt
python scripts/check_env.py
```

Download the base model and public datasets:

```bash
python scripts/download_assets.py --dry-run
python scripts/download_assets.py
```

Prepare the instruction dataset:

```bash
python scripts/prepare_dataset.py --config configs/data_lingxi.yaml
```

Run a memory-aware dialogue test without loading the model:

```bash
python scripts/memory_chat.py \
  --dry-run \
  --user "我最近总觉得自己很失败。" \
  --scene "家庭陪伴"
```

Run inference with an adapter:

```bash
python scripts/memory_chat.py \
  --model models/Qwen2.5-1.5B-Instruct \
  --adapter outputs/lingxi-qwen25-1p5b-lora-30min \
  --user "我最近总觉得自己很失败。" \
  --scene "家庭陪伴"
```

## Web Demo

Start the local Web console:

```bash
python webapp/server.py --host 127.0.0.1 --port 7860
```

Open:

```text
http://127.0.0.1:7860
```

The Web console is designed for competition demonstration. It shows the complete process of receiving user input, detecting safety risks, retrieving emotional memory, building prompts, loading the model, generating tokens, and saving the conversation state.

## Training

Run the main QLoRA training:

```bash
accelerate launch --num_processes 2 --mixed_precision bf16 \
  scripts/train_lora.py --config configs/qwen25_1p5b_qlora_30min.yaml
```

Run a smoke test:

```bash
accelerate launch scripts/train_lora.py \
  --config configs/qwen25_1p5b_qlora_30min.yaml \
  --max-steps 2 \
  --max-train-samples 8 \
  --max-eval-samples 4
```

Run DPO preference alignment:

```bash
bash scripts/run_dpo_pipeline.sh
```

Run domain experiments and ablation studies:

```bash
bash scripts/run_domain_experiments.sh
```

## Evaluation And Reports

Compare base model and fine-tuned adapter:

```bash
python scripts/infer_compare.py \
  --model models/Qwen2.5-1.5B-Instruct \
  --adapter outputs/lingxi-qwen25-1p5b-lora-30min \
  --prompts examples/eval_prompts.jsonl \
  --out reports/before_after_compare.md
```

Important reports:

- `reports/before_after_compare.md`: base model vs. LoRA adapter response comparison.
- `reports/domain_qa_eval.md`: domain QA comparison across adapters.
- `reports/dpo_alignment_report.md`: DPO preference alignment summary.
- `reports/hyperparameter_analysis.md`: rank, step, and training strategy analysis.
- `reports/memory_module_design.md`: short-term emotional memory design.
- `reports/safety_boundary_design.md`: safety boundary design.

## Safety Notice

Lingxi is designed for daily companionship and emotional support. It is not a medical device and does not provide diagnosis, therapy, medication guidance, or emergency intervention. For self-harm risk, harm to others, severe distress, medication, diagnosis, or long-term sleep problems, the system should guide the user to contact trusted people, emergency services, or qualified professionals.

## Project Highlights

- Built a competition-oriented emotional support pipeline for companion robots.
- Combined emotion detection, memory-aware prompting, LoRA fine-tuning, DPO alignment, and safety rules.
- Provided a full local Web demo for end-to-end presentation.
- Included reproducible scripts for dataset construction, training, evaluation, and ablation experiments.
- Reserved extension points for robot speech, expression, and motion control.

## License

Original code and documentation in this repository are licensed under the MIT License. Third-party datasets, model weights, and external dependencies remain subject to their original licenses.
