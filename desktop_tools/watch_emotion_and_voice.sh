#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
unset ROS_IP ROS_HOSTNAME
export ROS_MASTER_URI=http://localhost:11311

echo "[INFO] unified monitor started"
echo "[INFO] This terminal should continuously show: emotion + iat_text + ai_reply"
echo

while true; do
  clear
  echo "================ Emotion JSON (VM) ================"
  if [ -f "$HOME/.ros/latest_emotion.json" ]; then
    python3 - <<'PY2'
import json
from pathlib import Path
p=Path.home()/'.ros'/'latest_emotion.json'
try:
    d=json.loads(p.read_text())
    print('timestamp      :', d.get('timestamp'))
    print('face_count     :', d.get('face_count'))
    print('emotion        :', d.get('dominant_emotion'))
    print('confidence     :', d.get('confidence'))
except Exception as e:
    print('read error:', e)
PY2
  else
    echo "missing: $HOME/.ros/latest_emotion.json"
  fi

  echo
  echo "================ Latest /iat_text ================"
  timeout 1.5s rostopic echo -n 1 /iat_text 2>/dev/null || echo "(no new /iat_text in last window)"

  echo
  echo "================ Latest /ai_reply ================"
  timeout 1.5s rostopic echo -n 1 /ai_reply 2>/dev/null || echo "(no new /ai_reply in last window)"

  echo
  echo "================ Latest /ai_motion_text ================"
  timeout 1.5s rostopic echo -n 1 /ai_motion_text 2>/dev/null || echo "(no new /ai_motion_text in last window)"

  sleep 1

done
