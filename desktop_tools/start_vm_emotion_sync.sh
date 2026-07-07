#!/usr/bin/env bash
set -euo pipefail

SCRIPT="/home/lzc/桌面/sync_emotion_to_jetson.py"
PID_FILE="/home/lzc/桌面/sync_emotion_to_jetson.pid"
LOG_FILE="/home/lzc/桌面/sync_emotion_to_jetson.log"

if [ ! -f "${SCRIPT}" ]; then
  echo "[ERROR] script not found: ${SCRIPT}"
  exit 1
fi

if [ -f "${PID_FILE}" ] && kill -0 "$(cat ${PID_FILE})" >/dev/null 2>&1; then
  echo "[INFO] sync already running, pid=$(cat ${PID_FILE})"
  tail -f "${LOG_FILE}"
  exit 0
fi

nohup python3 "${SCRIPT}" > "${LOG_FILE}" 2>&1 &
echo $! > "${PID_FILE}"

echo "[INFO] sync started, pid=$(cat ${PID_FILE})"
echo "[INFO] log: ${LOG_FILE}"
tail -f "${LOG_FILE}"
