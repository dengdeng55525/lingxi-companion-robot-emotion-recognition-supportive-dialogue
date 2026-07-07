#!/usr/bin/env bash
set -euo pipefail

cd ~/catkin_ws
source ~/.bashrc
source devel/setup.bash
unset ROS_IP ROS_HOSTNAME
export ROS_MASTER_URI=http://localhost:11311

if pgrep -f "[r]oslaunch robot_voice iat_publish.launch" >/dev/null 2>&1; then
  echo "[INFO] iat_publish already running"
else
  echo "[INFO] starting iat_publish ..."
  nohup roslaunch robot_voice iat_publish.launch > /tmp/iat_publish.log 2>&1 &
  sleep 2
fi

echo "[INFO] voice INPUT running. Press Space in iat terminal frontend if required."
echo "[INFO] log: /tmp/iat_publish.log"
tail -f /tmp/iat_publish.log
