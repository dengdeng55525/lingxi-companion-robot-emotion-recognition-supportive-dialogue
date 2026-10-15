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


## Robot Deployment (整机部署与使用)

### Hardware / Topology

```text
┌──────────────────────────────┐
│  Jetson (robot body) 192.168.210.91 │
│  - ASTRA camera + chassis (/cmd_vel) │
│  - RoboEyes expression driver        │
└───────────────┬──────────────┘
                │ ROS network / SSH
┌───────────────▼──────────────┐
│  VM (main controller)          │
│  - web_video_server :8080      │
│  - vision upload + JSON        │
│  - vision_result_api :8090     │
│  - parent_dashboard_api :8091  │
│  - static frontend :9000       │
│  - ROS voice/AI/motion         │
│  - emotion sync to Jetson      │
└───────────────┬──────────────┘
                │ SSH tunnel :6008
┌───────────────▼──────────────┐
│  AutoDL/SeetaCloud            │
│  - emotion + object inference │
│  - /api/frame, /api/latest-result │
└──────────────────────────────┘
```

### Key Environment Variables

| Variable | Purpose |
| --- | --- |
| `ROBOTVISION_STREAM_TOKEN` | AutoDL vision stream token |
| `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL` / `DEEPSEEK_API_KEY` | LLM backend (`~/.ros/lingxi_ai_env.sh`) |
| `ROS_MASTER_URI` | ROS master (Jetson: `http://192.168.210.91:11311`, VM local: `http://localhost:11311`) |

### Step 1 – Jetson (Terminal 1 & 2)

Start chassis bringup and camera:

```bash
ssh clbrobot@192.168.210.91
source /opt/ros/melodic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.210.91:11311
export ROS_IP=192.168.210.91
export ROS_HOSTNAME=192.168.210.91
roslaunch clbrobot bringup.launch          # Terminal 1
roslaunch clbrobot astra_laser.launch      # Terminal 2
```

Verify camera topic:

```bash
rostopic hz /camera/rgb/image_raw
```

### Step 2 – VM Video Stream (Terminal 3)

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.210.91:11311
export VM_IP=192.168.210.231
export ROS_IP=$VM_IP
export ROS_HOSTNAME=$VM_IP
rosrun web_video_server web_video_server _port:=8080
```

Browser check: `http://127.0.0.1:8080/stream?topic=/camera/rgb/image_raw`

### Step 3 – AutoDL SSH Tunnel (Terminal 4, must keep running)

```bash
ssh -N -L 6008:127.0.0.1:6008 -p 39265 root@connect.westc.seetacloud.com
```

Verify:

```bash
curl --noproxy '*' http://127.0.0.1:6008/api/latest-result -H "X-Stream-Token: 11111111"
```

### Step 4 – VM Vision Upload + Result (Terminal 5)

```bash
cd ~/yolov5_emotieff_vgaf_windows
export ROBOTVISION_STREAM_TOKEN=11111111

python3 send_webcam_http.py \
  --server-url http://127.0.0.1:6008 \
  --token "$ROBOTVISION_STREAM_TOKEN" \
  --camera-url "http://127.0.0.1:8080/stream?topic=/camera/rgb/image_raw" \
  --width 0 --height 0 \
  --upload-fps 5 --jpeg-quality 78 \
  --result-hz 0.5 \
  --json-output outputs/latest_emotion.json \
  --latest-emotion-output "$HOME/.ros/latest_emotion.json" \
  --reconnect-interval 2.0
```

Check JSON updates:

```bash
watch -n 0.5 cat ~/.ros/latest_emotion.json
```

### Step 5 – VM Vision API (Terminal 6)

```bash
cd ~/yolov5_emotieff_vgaf_windows
python3 -u vision_result_api.py \
  --json-path outputs/latest_emotion.json \
  --host 0.0.0.0 --port 8090
```

### Step 6 – VM Frontend Static Server (Terminal 7)

```bash
cd ~/yolov5_emotieff_vgaf_windows/frontend
python3 -m http.server 9000
```

Pages:

- Vision dashboard: `http://127.0.0.1:9000/vision_dashboard.html`
- Parent dashboard: `http://127.0.0.1:9000/parent_dashboard.html`

### Step 7 – VM Parent Dashboard API (Terminal 7A, optional but recommended)

```bash
cd ~/yolov5_emotieff_vgaf_windows
python3 parent_dashboard_api.py \
  --vision-url http://127.0.0.1:8090/vision/latest \
  --db outputs/parent_dashboard.sqlite3 \
  --dialogue-log "$HOME/.ros/ai_companion_dialogue.jsonl" \
  --learning-log "$HOME/.ros/ai_learning_events.jsonl" \
  --host 0.0.0.0 --port 8091 \
  --sample-interval 2.0
```

### Step 8 – VM ROScore + AI Companion Launch (Terminal 8)

First start roscore (if not already running):

```bash
cd ~/catkin_ws
source devel/setup.bash
unset ROS_IP ROS_HOSTNAME
export ROS_MASTER_URI=http://localhost:11311
roscore
```

Then start the AI companion dialogue stack:

```bash
cd ~/catkin_ws
source devel/setup.bash
source ~/.ros/lingxi_ai_env.sh
unset ROS_IP ROS_HOSTNAME
export ROS_MASTER_URI=http://localhost:11311

roslaunch robot_voice ai_companion_dialogue.launch \
  jetson_user:=clbrobot \
  jetson_host:=192.168.210.91 \
  model:=$DEEPSEEK_MODEL \
  base_url:=$DEEPSEEK_BASE_URL \
  enable_jetson_mic:=false \
  enable_reply_speaker:=true \
  enable_music_player:=true \
  api_enabled:=true \
  api_mode:=chat_completions \
  tts_backend:=local-edge-tts \
  jetson_audio_output_device:=pulse \
  music_dir:=$HOME/Music/robot_music \
  wake_word_required:=true \
  wake_session_enabled:=true \
  wake_words:=灵犀 \
  emotion_file:=$HOME/.ros/latest_emotion.json \
  emotion_file_poll_seconds:=0.5 \
  dangerous_object_alert_enabled:=true \
  dangerous_object_alert_interval:=20.0 \
  dangerous_object_min_confidence:=0.30
```

### Step 9 – VM Voice Input (Terminal 9)

```bash
cd ~/catkin_ws
source devel/setup.bash
unset ROS_IP ROS_HOSTNAME
export ROS_MASTER_URI=http://localhost:11311
roslaunch robot_voice iat_publish.launch
```

### Step 10 – Parent Dialogue Logger (Terminal 9A, recommended)

```bash
cd ~/catkin_ws
source devel/setup.bash
unset ROS_IP ROS_HOSTNAME
export ROS_MASTER_URI=http://localhost:11311
cd ~/yolov5_emotieff_vgaf_windows
python3 parent_dialogue_logger.py \
  --user-topic /iat_text \
  --assistant-topic /ai_reply \
  --log-file "$HOME/.ros/ai_companion_dialogue.jsonl"
```

### Step 11 – VM -> Jetson Emotion Sync (Terminal 10, RoboEyes)

```bash
python3 ~/桌面/sync_emotion_to_jetson.py
# or: bash ~/桌面/start_vm_emotion_sync.sh
```

### Step 12 – Usage

After the full stack is running:

1. Speak the wake word: `灵犀`
2. Issue a command, e.g. `灵犀，这是什么？` / `灵犀，前进 0.5 米` / `灵犀，左转 90 度`
3. Observe RoboEyes expression changes (happy/sad/angry/neutral)
4. Open parent dashboard: `http://127.0.0.1:9000/parent_dashboard.html`
5. Review emotion history, dialogue timeline, learning concepts, and dangerous-object alerts

### One-Click Alternative

For quick demos (less debug-friendly), the pre-made scripts in `desktop_tools/` start the whole stack:

```bash
bash ~/桌面/start_jetson_all.sh          # Jetson chassis + camera
bash ~/桌面/start_vm_vision_all.sh       # VM video + vision + dashboards
bash ~/桌面/start_vm_voice_all.sh        # VM voice input/output
bash ~/桌面/start_vm_emotion_sync.sh     # RoboEyes emotion sync
```

### Troubleshooting Quick Reference

See `docs/ROBOEYES_启动与排障手册.md` for the complete manual. Common checks:

| Symptom | Check |
| --- | --- |
| HTML has no video | `web_video_server` running? `/camera/rgb/image_raw` exists? |
| AutoDL result empty | SSH tunnel alive? `curl .../api/latest-result -H "X-Stream-Token: 11111111"` |
| AI API failed | `source ~/.ros/lingxi_ai_env.sh` valid? DeepSeek key works? |
| Robot does not move | `/cmd_vel` exists? `ssh clbrobot@192.168.210.91 'echo ssh_ok'`? |
| RoboEyes not changing | `sync_emotion_to_jetson.py` running? `latest_emotion_vm.json` updating on Jetson? |
| Parent dashboard empty | `parent_dashboard_api.py` on `:8091`? `/vision/latest` on `:8090` returning real data? |


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
