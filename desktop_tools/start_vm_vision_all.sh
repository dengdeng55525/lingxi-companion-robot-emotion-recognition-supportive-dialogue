#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/yolov5_emotieff_vgaf_windows}"
ROBOT_IP="${ROBOT_IP:-192.168.64.91}"
VM_IP="${VM_IP:-$(hostname -I | awk '{print $1}')}"

cd "${PROJECT_DIR}"

ROBOT_IP="${ROBOT_IP}" \
VM_IP="${VM_IP}" \
PROJECT_DIR="${PROJECT_DIR}" \
ROS_IMAGE_TOPIC="/camera/rgb/image_raw" \
ROS_IMAGE_FALLBACK_TOPIC="/camera/rgb/image_rect_color" \
ENABLE_VIDEO_SERVER=0 \
OMP_NUM_THREADS=6 \
MKL_NUM_THREADS=6 \
SAMPLE_STEP=3 \
MAX_INFER_FPS=5 \
FACE_CONF=0.4 \
bash scripts/start_vm_ros_vision_stack.sh
