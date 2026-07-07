# Claude Code Robot Context

This workspace is a ROS1 robot companion project. The current working setup uses a
Ubuntu 20.04 VM as the AI/vision/voice-control host and a Jetson Nano robot
running Ubuntu 18.04 / ROS Melodic as the physical robot controller.

## Current Architecture

```text
PC microphone
  -> iFlyTek IAT node publishes /iat_text on VM ROS master
  -> robot_voice ai_companion_node reads /iat_text + latest vision JSON
  -> DeepSeek/OpenAI-compatible API returns JSON plan
  -> ai_companion_node publishes:
       /ai_reply       for speech reply
       /ai_motion_text for motion commands
  -> ssh_reply_speaker streams TTS MP3 to Jetson and plays via PulseAudio
  -> ssh_voice_control SSHes into Jetson and publishes /cmd_vel on Jetson ROS master

Jetson ASTRA camera
  -> Jetson ROS camera topics
  -> VM vision project consumes image stream / ROS image
  -> VM vision project writes ~/.ros/latest_emotion.json
  -> ai_companion_node polls that JSON and merges emotion into replies/plans
```

There are two ROS masters:

- VM local AI/voice master: `http://localhost:11311`
- Jetson robot/camera master: `http://192.168.79.91:11311`

Do not mix them. The VM AI nodes intentionally do **not** attach directly to the
Jetson ROS graph for motion; they use SSH into Jetson and publish `/cmd_vel`
locally there.

## Important Hosts

```text
VM user: lzc
VM catkin workspace: ~/catkin_ws
Jetson user: clbrobot
Jetson IP: 192.168.79.91
Jetson ROS distro: melodic
VM ROS distro: noetic
```

## Important Files In This Workspace

```text
src/robot_voice/scripts/ai_companion_node.py
  Main AI planner. Reads /iat_text and ~/.ros/latest_emotion.json.
  Calls the model API. Publishes /ai_reply and /ai_motion_text.
  Handles wake session logic:
    - inactive by default
    - "灵犀你好" starts conversation
    - after wake, later utterances do not need "灵犀"
    - "灵犀再见" ends conversation
  Handles startup face search:
    - if face_count is 0, publishes "右转90度"
    - repeats until face found or max turns reached.

src/robot_voice/scripts/ssh_voice_control.py
  Subscribes to /ai_motion_text on the VM ROS master.
  Parses Chinese motion text such as "前进1米", "左转90度".
  SSHes into Jetson and publishes geometry_msgs/Twist to Jetson /cmd_vel.

src/robot_voice/scripts/ssh_reply_speaker.py
  Subscribes to /ai_reply.
  Uses local edge-tts on VM to synthesize MP3.
  Streams MP3 to Jetson and plays it.
  Current stable output device is:
    jetson_audio_output_device:=pulse
  This uses `mpg123 -o pulse` on Jetson because PulseAudio owns the USB sound card.

src/robot_voice/scripts/music_player_node.py
  Subscribes to /iat_text.
  Handles local music commands, fuzzy matching, volume, stop playback.
  Streams music files from VM to Jetson.
  Also uses pulse output when `audio_device` is `pulse`.

src/robot_voice/scripts/ssh_arecord_vosk_iat.py
  Experimental Jetson microphone -> VM Vosk speech recognition.
  Currently not used because recognition was too slow/inaccurate.

src/robot_voice/launch/ai_companion_dialogue.launch
  Launches AI node, motion SSH node, reply speaker, optional music player,
  optional Jetson microphone node.

src/robot_voice/launch/iat_publish.launch
  Launches the iFlyTek speech recognizer that publishes /iat_text.
```

## Runtime Files

```text
~/.ros/latest_emotion.json
  Written by the vision project on the VM.
  Read by ai_companion_node.
  Expected compact format:
  {
    "timestamp": 1779270478.67,
    "face_count": 1,
    "dominant_emotion": "neutral",
    "confidence": 0.49,
    "scores": {
      "happy": 0.17,
      "sad": 0.23,
      "angry": 0.04,
      "neutral": 0.49,
      "excited": 0.03,
      "anxious": 0.01
    }
  }

~/.ros/ai_companion_latest_plan.json
  Written by ai_companion_node.
  Debug file showing the latest input, vision context, memory, status, and plan.

~/.ros/ai_companion_memory.json
  Persistent robot memory, for example robot name.

~/.ros/ai_dance_profile.json
  Persistent custom dance profile if the user teaches a new dance.

~/Music/robot_music
  VM music library used by music_player_node.
  Example layout:
    ~/Music/robot_music/周杰伦/青花瓷.mp3
```

## Model API

The code uses OpenAI-compatible environment variable names for compatibility, but
the runtime can point to DeepSeek:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.deepseek.com"
export OPENAI_MODEL="deepseek-v4-flash"
```

`ai_companion_node.py` supports:

```text
api_mode:=responses
api_mode:=chat_completions
api_mode:=auto
```

For DeepSeek use:

```text
api_mode:=chat_completions
```

If `api_mode:=auto` and `base_url` contains `deepseek.com`, the node uses
`/v1/chat/completions`.

## Full Startup Flow

Close old terminals first unless intentionally keeping something running.

### Terminal 0: VM ROS Master

```bash
cd ~/catkin_ws
source ~/.bashrc
source devel/setup.bash
unset ROS_IP
unset ROS_HOSTNAME
export ROS_MASTER_URI=http://localhost:11311

echo $OPENAI_BASE_URL
echo $OPENAI_MODEL

roscore
```

Expected:

```text
https://api.deepseek.com
deepseek-v4-flash
```

### Terminal 1: Jetson Base Bringup

```bash
ssh clbrobot@192.168.79.91
source /opt/ros/melodic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.79.91:11311
export ROS_IP=192.168.79.91
export ROS_HOSTNAME=192.168.79.91

roslaunch clbrobot bringup.launch
```

This starts the bottom controller and `/cmd_vel` subscriber on Jetson.

### Terminal 2: Jetson ASTRA Camera

Optional cleanup first:

```bash
ssh clbrobot@192.168.79.91 'bash -lc "
source /opt/ros/melodic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.79.91:11311
rosnode kill /camera/camera_nodelet_manager 2>/dev/null || true
pkill -f astra_laser.launch 2>/dev/null || true
pkill -f astra_camera 2>/dev/null || true
pkill -f nodelet 2>/dev/null || true
"'
```

Start camera:

```bash
ssh clbrobot@192.168.79.91
source /opt/ros/melodic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.79.91:11311
export ROS_IP=192.168.79.91
export ROS_HOSTNAME=192.168.79.91

roslaunch clbrobot astra_laser.launch
```

### Terminal 3: Verify Jetson Camera Topics

```bash
ssh clbrobot@192.168.79.91 'bash -lc "
source /opt/ros/melodic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.79.91:11311

echo ===== camera topics =====
rostopic list | grep camera

echo ===== image hz test =====
timeout 5s rostopic hz /camera/rgb/image_raw
"'
```

If `/camera/rgb/image_raw` is missing, the camera is not up.

### Terminal 4: VM Vision Stack And Dashboard

```bash
cd ~/yolov5_emotieff_vgaf_windows
chmod +x scripts/start_vm_ros_vision_stack.sh

ROBOT_IP=192.168.79.91 \
VM_IP=$(hostname -I | awk '{print $1}') \
PROJECT_DIR=$HOME/yolov5_emotieff_vgaf_windows \
OMP_NUM_THREADS=6 \
MKL_NUM_THREADS=6 \
SAMPLE_STEP=3 \
MAX_INFER_FPS=6 \
bash scripts/start_vm_ros_vision_stack.sh
```

Open dashboard:

```text
http://127.0.0.1:9000/vision_dashboard.html
```

The vision stack must write:

```bash
~/.ros/latest_emotion.json
```

### Terminal 5: Verify Emotion JSON

```bash
watch -n 0.5 cat ~/.ros/latest_emotion.json
```

The `timestamp` should keep changing. `face_count` can be 0 or 1 depending on
whether a face is visible.

### Terminal 6: AI Dialogue, Motion, Music, Startup Face Search

Start this only after `~/.ros/latest_emotion.json` is updating.

```bash
cd ~/catkin_ws
source ~/.bashrc
source devel/setup.bash
unset ROS_IP
unset ROS_HOSTNAME
export ROS_MASTER_URI=http://localhost:11311

roslaunch robot_voice ai_companion_dialogue.launch \
  jetson_user:=clbrobot \
  jetson_host:=192.168.79.91 \
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
  wake_word_aliases:=林夕,凌曦,灵溪,灵息,领席,零希 \
  vision_fresh_seconds:=10.0 \
  emotion_confidence_threshold:=0.4 \
  emotion_file:=$HOME/.ros/latest_emotion.json \
  emotion_file_poll_seconds:=0.5 \
  startup_face_search_enabled:=true \
  startup_face_search_min_delay:=2.0 \
  startup_face_search_turn_degrees:=90.0 \
  startup_face_search_after_turn_wait:=2.0 \
  startup_face_search_max_turns:=4 \
  startup_emotion_greeting_enabled:=true \
  startup_emotion_greeting_min_delay:=2.0 \
  startup_emotion_greeting_timeout:=20.0 \
  startup_emotion_greeting_min_confidence:=0.4 \
  startup_emotion_greeting_min_face_count:=1
```

Expected AI terminal messages:

```text
Vision emotion updated from file: ...
AI input: ...
AI plan: ...
Speaking reply on Jetson: ...
```

For startup face search, if `face_count` is 0:

```text
Startup face search started: no face detected.
Startup face search: 右转90度 (1).
```

### Terminal 7: iFlyTek Speech Recognition

```bash
cd ~/catkin_ws
source devel/setup.bash
unset ROS_IP
unset ROS_HOSTNAME
export ROS_MASTER_URI=http://localhost:11311

roslaunch robot_voice iat_publish.launch
```

Press `Space` to speak. This publishes `/iat_text`.

## Speech / Conversation Behavior

The current intended behavior:

```text
Before wake:
  "你好"       -> ignored
  "灵犀你好"   -> reply and enter active conversation

During active conversation:
  "晚上好"     -> reply
  "前进一米"   -> motion command
  "播放周杰伦的青花瓷" -> play music
  "停止播放"   -> stop music
  "灵犀再见"   -> exit conversation

After goodbye:
  any text without "灵犀" -> ignored
  "灵犀你好" -> wake again
```

## ROS Topics On VM Master

```text
/iat_text
  std_msgs/String. Speech recognition result.

/ai_reply
  std_msgs/String. Text reply to speak on Jetson.

/ai_motion_text
  std_msgs/String. Chinese motion command for ssh_voice_control.

/ai_plan
  std_msgs/String. JSON plan from ai_companion_node.

/vision_context
  Optional std_msgs/String JSON vision context topic.
  Current stable path is emotion JSON file polling, not this topic.
```

## ROS Topics On Jetson Master

```text
/cmd_vel
  geometry_msgs/Twist. Bottom motion command consumed by arduino_serial_node.

/camera/rgb/image_raw
  ASTRA RGB stream used by vision stack.

/camera/depth/*
  Depth topics from ASTRA launch.
```

## How To Plug In A Different Video Stream

The current AI node does not need raw video. It only needs compact emotion JSON.

To connect a different vision system, make it write:

```bash
/home/lzc/.ros/latest_emotion.json
```

atomically, for example:

```python
tmp = "/home/lzc/.ros/latest_emotion.json.tmp"
target = "/home/lzc/.ros/latest_emotion.json"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False)
    f.flush()
    os.fsync(f.fileno())
os.replace(tmp, target)
```

Minimum payload:

```json
{
  "timestamp": 1779270478.67,
  "face_count": 1,
  "dominant_emotion": "neutral",
  "confidence": 0.49,
  "scores": {
    "happy": 0.17,
    "sad": 0.23,
    "angry": 0.04,
    "neutral": 0.49,
    "excited": 0.03,
    "anxious": 0.01
  }
}
```

If no face:

```json
{
  "timestamp": 1779270478.67,
  "face_count": 0,
  "dominant_emotion": "unknown",
  "confidence": 0.0,
  "scores": {}
}
```

## How To Plug In A Different Speech Stream

Any speech recognizer can replace iFlyTek if it publishes:

```text
topic: /iat_text
type: std_msgs/String
ROS master: http://localhost:11311 on VM
```

Example manual test:

```bash
cd ~/catkin_ws
source devel/setup.bash
unset ROS_IP
unset ROS_HOSTNAME
export ROS_MASTER_URI=http://localhost:11311

rostopic pub -1 /iat_text std_msgs/String "data: '灵犀你好'"
```

## Debug Checklist

If AI terminal has `AI input` but no `Vision emotion updated`:

```bash
watch -n 0.5 cat ~/.ros/latest_emotion.json
```

If JSON is not changing, fix the vision stack.

If speech terminal has recognition text but AI does not react:

```bash
rostopic echo /iat_text
```

Make sure AI and speech are both using:

```bash
export ROS_MASTER_URI=http://localhost:11311
```

If the robot replies in logs but no sound comes out:

```bash
ssh clbrobot@192.168.79.91 'mpg123 -v -o pulse /tmp/robot_ai_reply.mp3'
```

The stable launch setting is:

```text
jetson_audio_output_device:=pulse
```

If motion text appears but robot does not move:

```bash
rostopic echo /ai_motion_text
```

Then verify Jetson base:

```bash
ssh clbrobot@192.168.79.91 'bash -lc "
source /opt/ros/melodic/setup.bash
source ~/catkin_ws/devel/setup.bash
rostopic info /cmd_vel
rosnode list
"'
```
