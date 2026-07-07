#!/usr/bin/env bash

set -u

ROBOT_USER="${ROBOT_USER:-clbrobot}"
ROBOT_HOST="${ROBOT_HOST:-192.168.64.91}"
LOCAL_API_PORT="${LOCAL_API_PORT:-8090}"
LOCAL_VIDEO_PORT="${LOCAL_VIDEO_PORT:-8080}"
REMOTE_API_PORT="${REMOTE_API_PORT:-8090}"
REMOTE_VIDEO_PORT="${REMOTE_VIDEO_PORT:-8080}"
WEB_PORT="${WEB_PORT:-9000}"

if [ -z "${FRONTEND_DIR:-}" ]; then
  if command -v xdg-user-dir >/dev/null 2>&1; then
    FRONTEND_DIR="$(xdg-user-dir DESKTOP)"
  else
    FRONTEND_DIR="${HOME}/Desktop"
  fi
fi

if [ ! -f "${FRONTEND_DIR}/vision_dashboard.html" ]; then
  FOUND_DASHBOARD="$(find "${HOME}" -maxdepth 3 -name vision_dashboard.html -print -quit 2>/dev/null || true)"
  if [ -n "${FOUND_DASHBOARD}" ]; then
    FRONTEND_DIR="$(dirname "${FOUND_DASHBOARD}")"
  fi
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${FRONTEND_DIR}/vision_dashboard_logs_${RUN_ID}"
mkdir -p "${LOG_DIR}"

HTTP_PID=""

cleanup() {
  echo
  echo "[INFO] stopping dashboard services..."
  if [ -n "${HTTP_PID}" ] && kill -0 "${HTTP_PID}" >/dev/null 2>&1; then
    kill "${HTTP_PID}" >/dev/null 2>&1 || true
  fi
  echo "[INFO] logs: ${LOG_DIR}"
}

trap cleanup EXIT INT TERM

if [ ! -d "${FRONTEND_DIR}" ]; then
  echo "[ERROR] frontend dir not found: ${FRONTEND_DIR}"
  exit 1
fi

if [ ! -f "${FRONTEND_DIR}/vision_dashboard.html" ]; then
  echo "[ERROR] frontend file not found: ${FRONTEND_DIR}/vision_dashboard.html"
  exit 1
fi

echo "[INFO] starting frontend web server..."
(
  cd "${FRONTEND_DIR}" || exit 1
  python3 -m http.server "${WEB_PORT}"
) >"${LOG_DIR}/frontend_http.log" 2>&1 &
HTTP_PID="$!"

sleep 1

echo "[INFO] frontend: http://127.0.0.1:${WEB_PORT}/vision_dashboard.html"
echo "[INFO] api tunnel: http://127.0.0.1:${LOCAL_API_PORT}/vision/latest"
echo "[INFO] video tunnel: http://127.0.0.1:${LOCAL_VIDEO_PORT}/stream?topic=/camera/rgb/image_raw"
echo
echo "[INFO] starting SSH tunnel. Enter the robot password if asked."
echo "[INFO] keep this terminal open. Press Ctrl-C to stop."
echo

ssh -N \
  -o ExitOnForwardFailure=yes \
  -L "${LOCAL_API_PORT}:127.0.0.1:${REMOTE_API_PORT}" \
  -L "${LOCAL_VIDEO_PORT}:127.0.0.1:${REMOTE_VIDEO_PORT}" \
  "${ROBOT_USER}@${ROBOT_HOST}"
