# Lingxi Companion Robot Emotion Recognition Supportive Dialogue

This repository is a sanitized source backup for the Lingxi companion robot stack.

## Layout

- `vision/`: VM-side vision bridge, AutoDL HTTP webcam uploader, dashboard API, and emotion JSON tools.
- `ros_ws/src/robot_voice/`: ROS voice input, AI companion dialogue, TTS, and motion-command integration.
- `ros_ws/src/RoboEyes/`: RoboEyes display library/source used by the robot expression stack.
- `desktop_tools/`: launch and sync helper scripts used from the VM desktop.
- `docs/`: runbooks and troubleshooting notes.

## Current Main Flow

The current production flow uses AutoDL/SeetaCloud for vision inference:

1. Jetson publishes camera frames through ROS.
2. VM `web_video_server` exposes the ROS image as HTTP/MJPEG.
3. `vision/send_webcam_http.py` uploads JPEG frames to AutoDL.
4. AutoDL returns emotion and object detection JSON.
5. The VM writes:
   - `vision/outputs/latest_emotion.json`
   - `$HOME/.ros/latest_emotion.json`
6. ROS AI dialogue reads the latest JSON and uses it for supportive replies, object questions, and RoboEyes emotion sync.

## Secrets

This GitHub backup intentionally does not contain real SSH passwords, API keys, or stream tokens.
Use environment variables or local private notes for values such as:

- `ROBOTVISION_STREAM_TOKEN`
- `OPENAI_API_KEY`
- SeetaCloud/AutoDL SSH credentials

