#!/usr/bin/env bash
set -euo pipefail

cd ~/catkin_ws
source ~/.bashrc
source devel/setup.bash
unset ROS_IP ROS_HOSTNAME
export ROS_MASTER_URI=http://localhost:11311

if ! pgrep -f "[r]oscore" >/dev/null 2>&1; then
  echo "[INFO] starting roscore ..."
  nohup roscore > /tmp/roscore_vm.log 2>&1 &
  sleep 3
fi

if ! pgrep -f "[r]oslaunch robot_voice ai_companion_dialogue.launch" >/dev/null 2>&1; then
  echo "[INFO] starting ai_companion_dialogue ..."
  nohup roslaunch robot_voice ai_companion_dialogue.launch     jetson_user:=clbrobot     jetson_host:=192.168.210.91     enable_jetson_mic:=false     enable_reply_speaker:=true     enable_music_player:=true     api_enabled:=true     api_mode:=chat_completions     tts_backend:=local-edge-tts     jetson_audio_output_device:=pulse     music_dir:=$HOME/Music/robot_music     wake_word_required:=true     wake_session_enabled:=true     wake_words:=灵犀     emotion_file:=$HOME/.ros/latest_emotion.json     emotion_file_poll_seconds:=0.5     > /tmp/ai_companion_dialogue.log 2>&1 &
  sleep 2
fi

echo "[INFO] voice OUTPUT stack running."
echo "[INFO] logs: /tmp/roscore_vm.log /tmp/ai_companion_dialogue.log"
tail -f /tmp/ai_companion_dialogue.log
