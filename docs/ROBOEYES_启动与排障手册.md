# RoboEyes 启动与排障手册（VM + Jetson）

> 目标：实现 **AutoDL 同时识别摄像头前的人脸情绪和物体**，情绪同步到 RoboEyes，物体/情绪交给 AI 推理模型，并且在 HTML 里看到视频流。

---

## 0. 先说明一个现象（避免误判）

RoboEyes 不是“嘴巴笑脸”，它主要是眼睑和眼形变化。
所以你看到的始终是“两只眼睛”是正常的，关键是眼型/眼睑是否变化。

- happy -> `HAPPY`（下眼睑更明显）
- sad -> `TIRED`（更“耷拉”）
- angry -> `ANGRY`（上眼睑压低）
- 无人脸/低置信度 -> `DEFAULT`

---

## 1. 启动顺序（分离启动版，推荐）

这次先不要用下面两个总控脚本：

```bash
/home/lzc/桌面/start_jetson_all.sh
/home/lzc/桌面/start_vm_vision_all.sh
```

原因是现在要把底盘、相机、视频流、情绪识别、Dashboard、语音控制全部拆开看日志。哪一段没起来，就只排查那一段。

当前推荐使用 **AutoDL/SeetaCloud 远端识别版**：

- VM 本地只负责：读取 `web_video_server` 的 HTTP/MJPEG 视频流、压缩 JPEG、上传到 AutoDL、拉回识别 JSON。
- AutoDL 负责：情绪识别 + 物体识别。
- VM 会把 AutoDL 返回结果写成原项目兼容的两份 JSON：
  - `~/yolov5_emotieff_vgaf_windows/outputs/latest_emotion.json`：完整视觉 JSON，包含 `faces`、`objects` 和 AutoDL 原始返回。
  - `~/.ros/latest_emotion.json`：AI/RoboEyes 读取的轻量 JSON，包含情绪字段和 `objects`。
- AI 推理/语音节点继续读取 `emotion_file=$HOME/.ros/latest_emotion.json`，不用改 ROS 工作空间。

> 注意：GitHub 备份版已脱敏；真实服务器密码和 Token 请只保存在本机私有手册或环境变量中。

AutoDL/SeetaCloud 信息：

```text
SSH：ssh -p 11189 <SEETACLOUD_SSH_USER_AND_HOST>
SSH 密码：<SEETACLOUD_SSH_PASSWORD>
本地映射地址：http://127.0.0.1:6008
视频流 Token：<ROBOTVISION_STREAM_TOKEN>
```

SSH 说明：

- 从 VM 登录小车已经配置过免密：`ssh clbrobot@192.168.64.91`
- 小车自己 SSH 到自己也已配置免密。
- 如果终端提示符已经是 `clbrobot@robot:~$`，说明你已经在小车里了，不需要再执行 `ssh clbrobot@192.168.64.91`，直接执行后面的 `source ...` 和 `roslaunch ...` 即可。

下面 `终端1` 到 `终端10` 是当前 AutoDL 方案的主启动流程，前面的每个终端都需要开。旧的 VM 本地 YOLO/情绪识别方案已经移到后面的“备用方案”章节，不要和 `终端5` 同时启动。

### 终端1（Jetson 底盘）

> 职责：只启动底盘。你已经确认手动 `bringup` 能带动底盘，所以这里以这个为准。

```bash
ssh clbrobot@192.168.64.91
source /opt/ros/melodic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.64.91:11311
export ROS_IP=192.168.64.91
export ROS_HOSTNAME=192.168.64.91
roslaunch clbrobot bringup.launch
```

这个终端不要关。另开一个 Jetson 终端检查：

```bash
ssh clbrobot@192.168.64.91
source /opt/ros/melodic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.64.91:11311
export ROS_IP=192.168.64.91
export ROS_HOSTNAME=192.168.64.91
rostopic info /cmd_vel
```

`/cmd_vel` 应该存在，并且底盘节点应该订阅它。

### 终端2（Jetson 相机）

> 职责：只启动 ASTRA 相机。

```bash
ssh clbrobot@192.168.64.91
source /opt/ros/melodic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.64.91:11311
export ROS_IP=192.168.64.91
export ROS_HOSTNAME=192.168.64.91
roslaunch clbrobot astra_laser.launch
```

这个终端也不要关。另开一个 Jetson 终端检查相机话题：

```bash
ssh clbrobot@192.168.64.91
source /opt/ros/melodic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.64.91:11311
export ROS_IP=192.168.64.91
export ROS_HOSTNAME=192.168.64.91
rostopic list | grep -E "/camera/rgb/image_raw|/camera/rgb/image_rect_color"
rostopic hz /camera/rgb/image_raw
```

如果你是在 VM 终端检查，用 Noetic：

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.64.91:11311
export VM_IP=$(hostname -I | awk '{print $1}')
export ROS_IP=$VM_IP
export ROS_HOSTNAME=$VM_IP
rostopic list | grep -E "/camera/rgb/image_raw|/camera/rgb/image_rect_color"
rostopic hz /camera/rgb/image_raw
```

如果 `/camera/rgb/image_raw` 没有，但 `/camera/rgb/image_rect_color` 有，后面的视觉命令就把 topic 改成 `/camera/rgb/image_rect_color`。

### 终端3（VM 视频流）

> 职责：只把 Jetson 的 ROS 图像话题转成 HTML 能看的 HTTP 视频流。

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.64.91:11311
export VM_IP=$(hostname -I | awk '{print $1}')
export ROS_IP=$VM_IP
export ROS_HOSTNAME=$VM_IP
rosrun web_video_server web_video_server _port:=8080
```

浏览器先直接打开这个，不经过 Dashboard：

```text
http://127.0.0.1:8080/stream?topic=/camera/rgb/image_raw
```

如果这里不出图，Dashboard 也一定收不到摄像头信息。先回到终端2查相机话题。

### 终端4（VM -> AutoDL SSH 隧道，必须开）

> 职责：把 VM 本地 `127.0.0.1:6008` 映射到 AutoDL 服务器的 `127.0.0.1:6008`。
> 
> 这个终端连接成功后通常没有输出，会一直停在那里。不要关闭。

```bash
ssh -N \
  -L 6008:127.0.0.1:6008 \
  -p 11189 \
  <SEETACLOUD_SSH_USER_AND_HOST>
```

出现下面提示时输入密码：

```text
<SEETACLOUD_SSH_PASSWORD>
```

第一次连接如果提示：

```text
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

输入：

```text
yes
```

另开一个 VM 终端测试隧道和 Token：

```bash
curl --noproxy '*' http://127.0.0.1:6008/api/latest-result \
  -H "X-Stream-Token: <ROBOTVISION_STREAM_TOKEN>"
```

正常情况会返回 JSON，例如 `status` 为 `waiting`、`ok` 等。只要不是下面两种就说明隧道和 Token 正常：

```text
Connection refused
Invalid stream token
```

### 终端5（VM 上传视频到 AutoDL，并拉回 JSON，必须开）

> 职责：
> 
> - 从终端3的 HTTP/MJPEG 视频流读取最新画面
> - 压缩 JPEG 并上传到 AutoDL
> - 拉取 AutoDL 识别结果
> - 写入 `outputs/latest_emotion.json`
> - 写入 AI/RoboEyes 共用的 `~/.ros/latest_emotion.json`

```bash
cd ~/yolov5_emotieff_vgaf_windows
export ROBOTVISION_STREAM_TOKEN=<ROBOTVISION_STREAM_TOKEN>

python3 send_webcam_http.py \
  --server-url http://127.0.0.1:6008 \
  --token "$ROBOTVISION_STREAM_TOKEN" \
  --camera-url "http://127.0.0.1:8080/stream?topic=/camera/rgb/image_raw" \
  --width 0 \
  --height 0 \
  --upload-fps 5 \
  --jpeg-quality 78 \
  --result-hz 0.5 \
  --json-output outputs/latest_emotion.json \
  --latest-emotion-output "$HOME/.ros/latest_emotion.json" \
  --reconnect-interval 2.0
```

正常日志应逐渐出现类似：

```text
[INFO] source=http://127.0.0.1:8080/stream?topic=/camera/rgb/image_raw
[INFO] target=http://127.0.0.1:6008/api/frame
[INFO] sent=... failed=0 status=...
```

检查交付给 AI 推理模型的 JSON：

```bash
watch -n 0.5 cat ~/.ros/latest_emotion.json
```

要看到：

```text
timestamp 持续变化
face_count 有人脸时为 1 或更多
dominant_emotion 随识别结果变化
confidence 为 0~1
objects/object_count 有识别到物体时会出现
```

### 终端6（VM 视觉 API）

> 职责：只提供 Dashboard 读取的 HTTP API，内容来自 AutoDL 回传后的完整视觉 JSON。

```bash
cd ~/yolov5_emotieff_vgaf_windows
python3 -u vision_result_api.py \
  --json-path outputs/latest_emotion.json \
  --host 0.0.0.0 \
  --port 8090
```

测试：

```bash
curl http://127.0.0.1:8090/health
curl http://127.0.0.1:8090/vision/latest
```

### 终端7（VM Dashboard）

> 职责：只打开 HTML 页面服务。

```bash
cd ~/yolov5_emotieff_vgaf_windows/frontend
python3 -m http.server 9000
```

启动后访问：

- Dashboard：`http://127.0.0.1:9000/vision_dashboard.html`
- 视频流直链：`http://127.0.0.1:8080/stream?topic=/camera/rgb/image_raw`
- 视觉 API：`http://127.0.0.1:8090/vision/latest`

### 终端8（VM 输出终端，必须开）

> 职责：
> 
> - 负责语音回复（TTS）
> - 负责 AI 回复/动作规划
> - 负责把视觉情绪融合到回复逻辑

```bash
cd ~/catkin_ws
source devel/setup.bash
unset ROS_IP ROS_HOSTNAME
export ROS_MASTER_URI=http://localhost:11311

roslaunch robot_voice ai_companion_dialogue.launch \
  jetson_user:=clbrobot \
  jetson_host:=192.168.64.91 \
  enable_jetson_mic:=false \
  enable_reply_speaker:=true \
  enable_music_player:=true \
  api_enabled:=true \
  api_mode:=chat_completions \
  tts_backend:=local-edge-tts \
  jetson_audio_output_device:=pulse \
  music_dir:=$HOME/Music/robot_music \
  wake_word_required:=true \
  wake_session_enabled:=true \
  wake_words:=灵犀 \
  emotion_file:=$HOME/.ros/latest_emotion.json \
  emotion_file_poll_seconds:=0.5
```

如果 VM 本地没有 roscore，先另开一个 VM 终端运行：

```bash
cd ~/catkin_ws
source devel/setup.bash
unset ROS_IP ROS_HOSTNAME
export ROS_MASTER_URI=http://localhost:11311
roscore
```

### 终端9（VM 输入终端，必须开）

> 职责：语音输入识别（你对机器人说话）

```bash
cd ~/catkin_ws
source devel/setup.bash
unset ROS_IP ROS_HOSTNAME
export ROS_MASTER_URI=http://localhost:11311
roslaunch robot_voice iat_publish.launch
```

### 终端10（RoboEyes 情绪同步）

> 让 RoboEyes 跟随 VM 识别结果，必须保持运行。

```bash
python3 /home/lzc/桌面/sync_emotion_to_jetson.py
```

如果这个脚本不在桌面，用原来的同步脚本启动也可以：

```bash
/home/lzc/桌面/start_vm_emotion_sync.sh
```

## 语音前进自动避障（默认关闭）

> 这条链路默认不启用雷达避障，前进/后退只走普通 `/cmd_vel` 控制。

如果你以后要重新开启，再看 `ssh_voice_control.py` 里的 `obstacle_avoid_enabled` 和 `scan_topic=/scan`。

---

## 2. 如果 HTML 打不开视频流（你这次的核心问题）

按下面 4 条检查：

### 2.1 检查 `web_video_server` 是否已启动（VM）

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.64.91:11311
export VM_IP=$(hostname -I | awk '{print $1}')
export ROS_IP=$VM_IP
export ROS_HOSTNAME=$VM_IP
rosnode list | grep web_video_server
```

应看到类似：`/web_video_server`

### 2.2 检查相机话题是否存在（VM 看 Jetson master）

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.64.91:11311
export VM_IP=$(hostname -I | awk '{print $1}')
export ROS_IP=$VM_IP
export ROS_HOSTNAME=$VM_IP
rostopic list | grep -E "/camera/rgb/image_raw|/camera/rgb/image_rect_color"
rostopic hz /camera/rgb/image_raw
```

### 2.3 直接打开视频流 URL（不经过 dashboard）

浏览器访问：

```text
http://127.0.0.1:8080/stream?topic=/camera/rgb/image_raw
```

如果这个都不出图，说明不是前端问题，而是 `web_video_server` 或相机话题问题。

### 2.4 看分离终端输出（最重要）

现在不用总控脚本，所以不看 `logs/vm_ros_vision_YYYYmmdd_HHMMSS`。直接看这几个正在运行的终端：

- 终端2：`astra_laser.launch` 有没有报相机设备找不到、USB 权限、nodelet 崩溃。
- 终端3：`web_video_server` 有没有出现 `Waiting For connections on 0.0.0.0:8080`，浏览器访问时有没有 `Handling Request`。
- 终端4：SSH 隧道是否还开着。这个终端正常会保持无输出状态。
- 终端5：`send_webcam_http.py` 是否持续打印 `sent=... failed=0 status=...`。

如果终端5出现：

```text
[WARN] video source disconnected; reconnecting in 2.0s
```

说明上传脚本打不开终端3提供的视频流。先确认浏览器能打开：

```text
http://127.0.0.1:8080/stream?topic=/camera/rgb/image_raw
```

如果终端5出现：

```text
Connection refused
```

说明本机 `127.0.0.1:6008` 没通，优先检查终端4的 SSH 隧道是否还在。

如果终端5或 curl 出现：

```text
Invalid stream token
```

说明 Token 不对，当前手册使用：

```text
<ROBOTVISION_STREAM_TOKEN>
```

如果后面的备用本地识别方案一直没有 `frame=...`，说明 VM 没收到 ROS 图像。回头检查：

```bash
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://192.168.64.91:11311
export VM_IP=$(hostname -I | awk '{print $1}')
export ROS_IP=$VM_IP
export ROS_HOSTNAME=$VM_IP
rostopic hz /camera/rgb/image_raw
```

如果这里也没有频率，问题在 Jetson 相机或 ROS 网络；如果这里有频率但 HTML 没图，问题在 `web_video_server`。

### 2.5 如果 AutoDL 识别结果没有交付给 AI

先看 AutoDL 接收服务是否可访问：

```bash
curl --noproxy '*' http://127.0.0.1:6008/api/latest-result \
  -H "X-Stream-Token: <ROBOTVISION_STREAM_TOKEN>"
```

如果返回：

```text
waiting for uploaded camera frames
```

说明 AutoDL 服务和 Token 正常，但还没有收到上传帧。检查终端5。

如果返回了识别 JSON，再看 VM 本地交付文件：

```bash
cat ~/yolov5_emotieff_vgaf_windows/outputs/latest_emotion.json
cat ~/.ros/latest_emotion.json
```

`~/.ros/latest_emotion.json` 是 AI 推理节点真正读取的文件，至少应包含：

```json
{
  "timestamp": 1783341495.3,
  "face_count": 1,
  "dominant_emotion": "happy",
  "confidence": 0.91,
  "scores": {
    "happy": 0.91,
    "sad": 0.02,
    "angry": 0.01,
    "neutral": 0.06
  },
  "object_count": 1,
  "objects": [
    {
      "class": "bench",
      "confidence": 0.57
    }
  ]
}
```

---

## 2.6 如果语音控制小车不动

先确认 Jetson 底盘终端1仍在运行，然后按下面三步从后往前测。

### A. 绕开 AI，直接测 SSH 控车节点

VM 终端执行：

```bash
cd ~/catkin_ws
source devel/setup.bash
unset ROS_IP ROS_HOSTNAME
export ROS_MASTER_URI=http://localhost:11311
rostopic pub -1 /ai_motion_text std_msgs/String "data: '前进0.5米'"
```

如果小车动，说明底盘和 SSH 控车没问题，问题在 AI 或语音识别。

如果不动，看终端8有没有：

```text
SSH voice control ready for clbrobot@192.168.64.91 on ai_motion_text
Voice command '前进0.5米' -> ...
SSH command failed ...
```

### B. 绕开语音识别，测 `/iat_text -> AI -> /ai_motion_text`

```bash
rostopic pub -1 /iat_text std_msgs/String "data: '灵犀，前进0.5米'"
```

终端8应该看到：

```text
AI input: ...
AI motion command: 前进0.5米
Voice command '前进0.5米' -> ...
```

如果这里能动，而你说话不能动，问题在终端9的语音识别。

### C. 看语音识别是否真的发布文字

```bash
rostopic echo /iat_text
```

对麦克风说话后，应该能看到识别文字。注意当前终端8启用了唤醒词：

```text
灵犀
```

所以实际语音建议说：

```text
灵犀，前进
灵犀，左转
灵犀，停止
```

---

## 3. 先做“强制表情测试”（不依赖你真人表情）

如果你怀疑没生效，先直接写入 happy/sad/angry：

```bash
python3 - <<'PY'
import json,time
p='/home/lzc/.ros/latest_emotion.json'
with open(p,'r',encoding='utf-8') as f:
    d=json.load(f)
for emo,conf in [('happy',0.92),('sad',0.91),('angry',0.93),('neutral',0.95)]:
    d['timestamp']=time.time()
    d['face_count']=1
    d['dominant_emotion']=emo
    d['confidence']=conf
    with open(p,'w',encoding='utf-8') as f:
        json.dump(d,f,ensure_ascii=False)
    print('emit',emo,conf)
    time.sleep(2)
PY
```

如果这一步都看不到变化，问题不在视觉模型，而在“同步/屏幕驱动”链路。

---

## 4. 三段链路检查（最关键）

## 4.1 VM 侧 JSON 是否在更新

```bash
watch -n 0.5 cat ~/.ros/latest_emotion.json
```

要看到：`timestamp`持续变化，`face_count`在有人脸时为 `1`。

## 4.2 Jetson 是否收到同步文件

```bash
ssh clbrobot@192.168.64.91 "watch -n 0.5 cat /home/clbrobot/RoboEyes/latest_emotion_vm.json"
```

要看到：内容跟 VM 侧同步，`dominant_emotion`会变化。

## 4.3 RoboEyes 是否读取并切换 mood

```bash
ssh clbrobot@192.168.64.91 "tail -f /home/clbrobot/RoboEyes/roboeyes_driver.log"
```

正常会出现类似：

```text
[MOOD] happy conf=0.92 -> 3
[MOOD] sad conf=0.91 -> 1
[MOOD] angry conf=0.93 -> 2
```

---

## 5. 常见原因：为什么你笑了它还不变

1. `confidence` 太低（当前阈值 0.40）
2. `face_count=0`（没检测到人脸，自动回到默认）
3. 同步脚本没运行（VM 改了，Jetson 没收到）
4. RoboEyes 进程没在跑或崩了

检查是否在跑：

```bash
ssh clbrobot@192.168.64.91 "pgrep -af roboeyes_json_driver.py"
```

---

## 6. 一键重启（RoboEyes）

```bash
ssh clbrobot@192.168.64.91 'bash -lc "/home/clbrobot/RoboEyes/start_roboeyes.sh"'
```

---

## 7. 备用方案：VM 本地视觉识别（不推荐）

当前主流程使用 AutoDL 同时做情绪识别和物体识别。下面这个旧方案只在 AutoDL 不可用时使用。

重要：

- 平时不要启动这一节。
- 不要和主流程的 `终端5（send_webcam_http.py）` 同时启动。
- 否则两个程序会抢着写同一个 JSON：
  - `~/yolov5_emotieff_vgaf_windows/outputs/latest_emotion.json`
  - `~/.ros/latest_emotion.json`
- 这个旧方案只做本地情绪识别，不做当前 AutoDL 版本里的物体识别。

```bash
cd ~/yolov5_emotieff_vgaf_windows
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.64.91:11311
export VM_IP=$(hostname -I | awk '{print $1}')
export ROS_IP=$VM_IP
export ROS_HOSTNAME=$VM_IP

PYTHONPATH="$PWD:$PWD/yolov5-face:$PYTHONPATH" python3 -u vm_ros_emotion_to_json.py \
  --ros-image-topic /camera/rgb/image_raw \
  --face-weights yolov5-face/weights/yolov5s-face.pt \
  --device auto \
  --emotieff-model-name enet_b0_8_best_vgaf \
  --emotieff-engine torch \
  --json-output outputs/latest_emotion.json \
  --latest-emotion-output ~/.ros/latest_emotion.json \
  --crop-margin 0.30 \
  --sample-step 3 \
  --max-infer-fps 6 \
  --face-conf 0.4 \
  --iou-thres 0.5 \
  --img-size 640
```

如果前面确认只有 `/camera/rgb/image_rect_color`，把这一行改掉：

```bash
--ros-image-topic /camera/rgb/image_rect_color
```

检查 JSON 是否持续更新：

```bash
watch -n 0.5 cat ~/.ros/latest_emotion.json
```

---

## 8. 已使用的关键文件

- AutoDL 视频上传/结果拉取/JSON交付：`/home/lzc/yolov5_emotieff_vgaf_windows/send_webcam_http.py`
- 完整视觉 JSON：`/home/lzc/yolov5_emotieff_vgaf_windows/outputs/latest_emotion.json`
- AI/RoboEyes 共用轻量视觉 JSON：`/home/lzc/.ros/latest_emotion.json`
- VM 同步脚本：`/home/lzc/桌面/sync_emotion_to_jetson.py`
- Jetson 驱动：`/home/clbrobot/RoboEyes/roboeyes_json_driver.py`
- Jetson 启动脚本：`/home/clbrobot/RoboEyes/start_roboeyes.sh`
- Jetson 自启动项：`/home/clbrobot/.config/autostart/roboeyes-json-driver.desktop`
- 语音控制：`/home/lzc/catkin_ws/src/robot_voice/scripts/ssh_voice_control.py`
- 输出总成 launch：`/home/lzc/catkin_ws/src/robot_voice/launch/ai_companion_dialogue.launch`

---

如果你按这份手册执行后仍不动，优先把这些输出贴出来：

1. `~/.ros/latest_emotion.json` 当前内容
2. `/home/clbrobot/RoboEyes/latest_emotion_vm.json` 当前内容
3. `/home/clbrobot/RoboEyes/roboeyes_driver.log` 最后 50 行
4. 终端2（Jetson 相机）最后 50 行
5. 终端3（VM 视频流）最后 50 行
6. 终端4（SSH 隧道）是否仍保持连接
7. 终端5（`send_webcam_http.py`）最后 50 行
8. 下面这个命令的输出：

```bash
curl --noproxy '*' http://127.0.0.1:6008/api/latest-result \
  -H "X-Stream-Token: <ROBOTVISION_STREAM_TOKEN>"
```

9. 终端8（VM 输出/SSH 控车）最后 50 行
