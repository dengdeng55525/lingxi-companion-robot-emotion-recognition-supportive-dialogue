#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
CATKIN_SETUP="${CATKIN_SETUP:-}"

ROBOT_IP="${ROBOT_IP:-192.168.79.91}"
VM_IP="${VM_IP:-192.168.79.199}"
ROS_IMAGE_TOPIC="${ROS_IMAGE_TOPIC:-/camera/rgb/image_raw}"
ROS_IMAGE_FALLBACK_TOPIC="${ROS_IMAGE_FALLBACK_TOPIC:-/camera/rgb/image_rect_color}"
ENABLE_IMAGE_TOPIC_RELAY="${ENABLE_IMAGE_TOPIC_RELAY:-1}"

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8090}"
VIDEO_PORT="${VIDEO_PORT:-8080}"
WEB_PORT="${WEB_PORT:-9000}"
JSON_OUTPUT="${JSON_OUTPUT:-outputs/latest_emotion.json}"
LATEST_EMOTION_OUTPUT="${LATEST_EMOTION_OUTPUT:-/home/lzc/.ros/latest_emotion.json}"

FACE_WEIGHTS="${FACE_WEIGHTS:-yolov5-face/weights/yolov5s-face.pt}"
DEVICE="${DEVICE:-auto}"
EMOTIEFF_MODEL="${EMOTIEFF_MODEL:-enet_b0_8_best_vgaf}"
EMOTIEFF_ENGINE="${EMOTIEFF_ENGINE:-torch}"
CROP_MARGIN="${CROP_MARGIN:-0.30}"
SAMPLE_STEP="${SAMPLE_STEP:-3}"
MAX_INFER_FPS="${MAX_INFER_FPS:-8}"
FACE_CONF="${FACE_CONF:-0.4}"
IOU_THRES="${IOU_THRES:-0.5}"
IMG_SIZE="${IMG_SIZE:-640}"
TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-6}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-${TORCH_NUM_THREADS}}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-${TORCH_NUM_THREADS}}"
ENABLE_VIDEO_SERVER="${ENABLE_VIDEO_SERVER:-1}"
ENABLE_FRONTEND="${ENABLE_FRONTEND:-1}"
FRONTEND_DIR="${FRONTEND_DIR:-${PROJECT_DIR}/frontend}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_DIR}/logs/vm_ros_vision_${RUN_ID}"
PIDS=""

cleanup() {
  echo
  echo "[INFO] stopping VM ROS vision stack..."
  for pid in ${PIDS}; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
  sleep 1
  for pid in ${PIDS}; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill -9 "${pid}" >/dev/null 2>&1 || true
    fi
  done
  echo "[INFO] logs: ${LOG_DIR}"
}

trap cleanup EXIT INT TERM

start_bg() {
  name="$1"
  shift
  log_file="${LOG_DIR}/${name}.log"
  echo "[INFO] starting ${name}, log: ${log_file}"
  "$@" >"${log_file}" 2>&1 &
  pid="$!"
  PIDS="${PIDS} ${pid}"
  echo "[INFO] ${name} pid=${pid}"
}

cleanup_stale_services() {
  for pid in $(lsof -ti tcp:${VIDEO_PORT} tcp:${API_PORT} tcp:${WEB_PORT} 2>/dev/null || true); do
    kill "$pid" >/dev/null 2>&1 || true
  done
  sleep 2
}

if [ ! -f "${ROS_SETUP}" ]; then
  echo "[ERROR] ROS setup not found: ${ROS_SETUP}"
  exit 1
fi

if [ ! -d "${PROJECT_DIR}" ]; then
  echo "[ERROR] project dir not found: ${PROJECT_DIR}"
  exit 1
fi

mkdir -p "${LOG_DIR}"

source "${ROS_SETUP}"
if [ -n "${CATKIN_SETUP}" ] && [ -f "${CATKIN_SETUP}" ]; then
  source "${CATKIN_SETUP}"
fi

export ROS_MASTER_URI="http://${ROBOT_IP}:11311"
export ROS_IP="${VM_IP}"
export ROS_HOSTNAME="${VM_IP}"
export TORCH_NUM_THREADS
export OMP_NUM_THREADS
export MKL_NUM_THREADS

cleanup_stale_services

cd "${PROJECT_DIR}" || exit 1
mkdir -p "$(dirname "${JSON_OUTPUT}")"

echo "[INFO] project: ${PROJECT_DIR}"
echo "[INFO] ROS_MASTER_URI=${ROS_MASTER_URI}"
echo "[INFO] ROS_IP=${ROS_IP}"
echo "[INFO] topic: ${ROS_IMAGE_TOPIC}"
echo "[INFO] fallback topic: ${ROS_IMAGE_FALLBACK_TOPIC}"
echo "[INFO] device: ${DEVICE}"
echo "[INFO] torch threads: ${TORCH_NUM_THREADS}"
echo "[INFO] api: http://${VM_IP}:${API_PORT}/vision/latest"
echo "[INFO] video preview: http://${VM_IP}:${VIDEO_PORT}/stream?topic=${ROS_IMAGE_TOPIC}"
echo "[INFO] dashboard: http://${VM_IP}:${WEB_PORT}/vision_dashboard.html"
echo "[INFO] json: ${PROJECT_DIR}/${JSON_OUTPUT}"
echo "[INFO] latest emotion json: ${LATEST_EMOTION_OUTPUT}"

if [ "${ENABLE_VIDEO_SERVER}" = "1" ]; then
  if rospack find web_video_server >/dev/null 2>&1; then
    start_bg web_video_server rosrun web_video_server web_video_server _port:="${VIDEO_PORT}"
    sleep 2
  else
    echo "[WARN] web_video_server not found; video preview will be unavailable."
    echo "[WARN] install if needed: sudo apt install ros-noetic-web-video-server"
  fi
fi

if [ "${ENABLE_IMAGE_TOPIC_RELAY}" = "1" ]; then
  if [ "${ROS_IMAGE_TOPIC}" = "/camera/rgb/image_raw" ]; then
    if ! rostopic info "${ROS_IMAGE_TOPIC}" 2>/dev/null | grep -q "Publishers:" || rostopic info "${ROS_IMAGE_TOPIC}" 2>/dev/null | grep -q "Publishers: None"; then
      echo "[INFO] no publisher on ${ROS_IMAGE_TOPIC}, start relay from ${ROS_IMAGE_FALLBACK_TOPIC}"
      start_bg image_topic_relay rosrun topic_tools relay "${ROS_IMAGE_FALLBACK_TOPIC}" "${ROS_IMAGE_TOPIC}"
      sleep 1
    fi
  fi
fi

start_bg emotion_json env PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/yolov5-face:${PYTHONPATH:-}" python3 -u vm_ros_emotion_to_json.py \
  --ros-image-topic "${ROS_IMAGE_TOPIC}" \
  --face-weights "${FACE_WEIGHTS}" \
  --device "${DEVICE}" \
  --emotieff-model-name "${EMOTIEFF_MODEL}" \
  --emotieff-engine "${EMOTIEFF_ENGINE}" \
  --json-output "${JSON_OUTPUT}" \
  --latest-emotion-output "${LATEST_EMOTION_OUTPUT}" \
  --crop-margin "${CROP_MARGIN}" \
  --sample-step "${SAMPLE_STEP}" \
  --max-infer-fps "${MAX_INFER_FPS}" \
  --face-conf "${FACE_CONF}" \
  --iou-thres "${IOU_THRES}" \
  --img-size "${IMG_SIZE}"
sleep 3

start_bg vision_api python3 -u vision_result_api.py \
  --json-path "${JSON_OUTPUT}" \
  --host "${API_HOST}" \
  --port "${API_PORT}"

if [ "${ENABLE_FRONTEND}" = "1" ]; then
  if [ -f "${FRONTEND_DIR}/vision_dashboard.html" ]; then
    start_bg frontend_http bash -lc "cd '${FRONTEND_DIR}' && python3 -m http.server '${WEB_PORT}'"
  else
    echo "[WARN] frontend file not found: ${FRONTEND_DIR}/vision_dashboard.html"
  fi
fi

echo
echo "[INFO] all VM services started."
echo "[INFO] press Ctrl-C to stop everything."
echo "[INFO] live logs:"
echo "       tail -f ${LOG_DIR}/emotion_json.log"
echo "       tail -f ${LOG_DIR}/vision_api.log"
echo "       tail -f ${LOG_DIR}/web_video_server.log"
echo "       tail -f ${LOG_DIR}/frontend_http.log"
echo

while true; do
  for pid in ${PIDS}; do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo "[ERROR] one service stopped. Check logs in: ${LOG_DIR}"
      exit 1
    fi
  done
  sleep 2
done
