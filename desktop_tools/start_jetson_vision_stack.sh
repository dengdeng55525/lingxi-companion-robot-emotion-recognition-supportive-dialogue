#!/usr/bin/env bash

set -u

PROJECT_DIR="${PROJECT_DIR:-/home/clbrobot/catkin_ws/yolov5_emotieff_vgaf_jetson}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/melodic/setup.bash}"
CATKIN_SETUP="${CATKIN_SETUP:-/home/clbrobot/catkin_ws/devel/setup.bash}"
ROBOT_IP="${ROBOT_IP:-192.168.64.91}"

VIDEO_PORT="${VIDEO_PORT:-8080}"
API_PORT="${API_PORT:-8090}"
JSON_OUTPUT="${JSON_OUTPUT:-outputs/latest_emotion.json}"
SOURCE_URL="${SOURCE_URL:-http://127.0.0.1:${VIDEO_PORT}/stream?topic=/camera/rgb/image_raw}"

FACE_WEIGHTS="${FACE_WEIGHTS:-yolov5-face/weights/yolov5s-face_torch14.pt}"
EMOTIEFF_MODEL="${EMOTIEFF_MODEL:-enet_b0_8_best_vgaf}"
EMOTIEFF_ENGINE="${EMOTIEFF_ENGINE:-torch}"
CROP_MARGIN="${CROP_MARGIN:-0.60}"
SAMPLE_STEP="${SAMPLE_STEP:-3}"
FACE_CONF="${FACE_CONF:-0.4}"
IOU_THRES="${IOU_THRES:-0.5}"
IMG_SIZE="${IMG_SIZE:-640}"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-10.0}"
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/tegra:/usr/local/cuda-10.0/targets/aarch64-linux/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_DIR}/logs/vision_stack_${RUN_ID}"
mkdir -p "${LOG_DIR}"

PIDS=""

cleanup() {
  echo
  echo "[INFO] stopping vision stack..."
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

if [ ! -f "${ROS_SETUP}" ]; then
  echo "[ERROR] ROS setup not found: ${ROS_SETUP}"
  exit 1
fi

if [ ! -f "${CATKIN_SETUP}" ]; then
  echo "[ERROR] catkin setup not found: ${CATKIN_SETUP}"
  exit 1
fi

if [ ! -d "${PROJECT_DIR}" ]; then
  echo "[ERROR] project dir not found: ${PROJECT_DIR}"
  exit 1
fi

source "${ROS_SETUP}"
source "${CATKIN_SETUP}"

export ROS_MASTER_URI="http://${ROBOT_IP}:11311"
export ROS_IP="${ROBOT_IP}"
export ROS_HOSTNAME="${ROBOT_IP}"

cd "${PROJECT_DIR}" || exit 1
mkdir -p "$(dirname "${JSON_OUTPUT}")"

echo "[INFO] project: ${PROJECT_DIR}"
echo "[INFO] robot ip: ${ROBOT_IP}"
echo "[INFO] video url: ${SOURCE_URL}"
echo "[INFO] api: http://${ROBOT_IP}:${API_PORT}/vision/latest"
echo "[INFO] dashboard tunnel target ports: ${VIDEO_PORT}, ${API_PORT}"
echo "[INFO] camera mode: RGB-only stable path"

start_bg astra_rgb_only roslaunch astra_launch astra.launch \
  rgb_processing:=true \
  depth_processing:=false \
  depth_registered_processing:=false \
  ir_processing:=false \
  disparity_processing:=false \
  disparity_registered_processing:=false \
  hw_registered_processing:=false \
  sw_registered_processing:=false \
  num_worker_threads:=1
sleep 6

start_bg web_video_server rosrun web_video_server web_video_server _port:="${VIDEO_PORT}"
sleep 3

start_bg emotion_json env PYTHONPATH=./yolov5-face python3 jetson_emotion_to_json.py \
  --source "${SOURCE_URL}" \
  --face-weights "${FACE_WEIGHTS}" \
  --emotieff-model-name "${EMOTIEFF_MODEL}" \
  --emotieff-engine "${EMOTIEFF_ENGINE}" \
  --json-output "${JSON_OUTPUT}" \
  --crop-margin "${CROP_MARGIN}" \
  --sample-step "${SAMPLE_STEP}" \
  --face-conf "${FACE_CONF}" \
  --iou-thres "${IOU_THRES}" \
  --img-size "${IMG_SIZE}"
sleep 3

start_bg vision_api python3 vision_result_api.py \
  --json-path "${JSON_OUTPUT}" \
  --host 0.0.0.0 \
  --port "${API_PORT}"

echo
echo "[INFO] all services started."
echo "[INFO] press Ctrl-C to stop everything."
echo "[INFO] live logs:"
echo "       tail -f ${LOG_DIR}/emotion_json.log"
echo "       tail -f ${LOG_DIR}/vision_api.log"
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
