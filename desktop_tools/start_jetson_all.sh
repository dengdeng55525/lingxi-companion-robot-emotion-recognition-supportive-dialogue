#!/usr/bin/env bash
set -eo pipefail

JETSON_USER="${JETSON_USER:-clbrobot}"
JETSON_HOST="${JETSON_HOST:-192.168.64.91}"

ssh "${JETSON_USER}@${JETSON_HOST}" 'bash -s' <<'REMOTE'
set -eo pipefail

source /opt/ros/melodic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.64.91:11311
export ROS_IP=192.168.64.91
export ROS_HOSTNAME=192.168.64.91

start_bg() {
  local name="$1"
  local cmd="$2"
  local log="$3"
  echo "[INFO] starting: $name"
  nohup bash -lc "$cmd" > "$log" 2>&1 &
  sleep 2
}

cleanup_stale_ros() {
  echo "[INFO] cleanup stale camera/launch processes"
  pkill -f "roslaunch clbrobot astra_laser.launch" >/dev/null 2>&1 || true
  pkill -f "nodelet manager __name:=camera_nodelet_manager" >/dev/null 2>&1 || true
  pkill -f "nodelet load astra_camera/AstraDriverNodelet" >/dev/null 2>&1 || true
  pkill -f "depthimage_to_laserscan" >/dev/null 2>&1 || true
  sleep 2
}

ensure_roscore() {
  if ! pgrep -f "[r]oscore" >/dev/null 2>&1; then
    start_bg "roscore" "source /opt/ros/melodic/setup.bash; roscore" "$HOME/roscore_jetson.log"
  else
    echo "[INFO] already running: roscore"
  fi
}

ensure_bringup() {
  if ! pgrep -f "[r]oslaunch clbrobot bringup.launch" >/dev/null 2>&1; then
    start_bg "bringup" "source /opt/ros/melodic/setup.bash; source ~/catkin_ws/devel/setup.bash; export ROS_MASTER_URI=http://192.168.64.91:11311; export ROS_IP=192.168.64.91; export ROS_HOSTNAME=192.168.64.91; roslaunch clbrobot bringup.launch" "$HOME/bringup.log"
  else
    echo "[INFO] already running: bringup"
  fi
}

cleanup_stale_ros
ensure_roscore
ensure_bringup

pkill -f roboeyes_json_driver.py >/dev/null 2>&1 || true
/home/clbrobot/RoboEyes/start_roboeyes.sh

sleep 2

echo "[INFO] check nodes/topics"
rosnode list | head -n 60 || true
rostopic list | grep -E "^/camera/rgb/image_raw$|^/camera/rgb/image_rect_color$|^/cmd_vel$|^/scan$" || true

echo "[INFO] Tail logs (roscore + bringup + camera). Ctrl-C to stop viewing only."
tail -n 80 -f "$HOME/roscore_jetson.log" "$HOME/bringup.log" "$HOME/astra_laser.log"
REMOTE
