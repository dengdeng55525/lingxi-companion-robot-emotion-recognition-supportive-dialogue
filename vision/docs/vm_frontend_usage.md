# 虚拟机前端访问小车视觉参数

目标结构：

```text
小车 Jetson 负责相机、视觉模型、JSON API
Linux 虚拟机负责打开前端页面
虚拟机通过 SSH 端口转发访问小车的 8090 接口
```

## 1. 小车终端 1：启动 ASTRA 相机

```bash
source /opt/ros/melodic/setup.bash
source /home/clbrobot/catkin_ws/devel/setup.bash

ROS_MASTER_URI=http://192.168.79.91:11311 \
ROS_IP=192.168.79.91 \
ROS_HOSTNAME=192.168.79.91 \
roslaunch clbrobot astra_laser.launch
```

看到类似下面这行，说明相机起来了：

```text
Device "2bc5/0402@1/8" found.
```

## 2. 小车终端 2：启动 ROS 视频流

```bash
source /opt/ros/melodic/setup.bash
source /home/clbrobot/catkin_ws/devel/setup.bash

ROS_MASTER_URI=http://192.168.79.91:11311 \
ROS_IP=192.168.79.91 \
ROS_HOSTNAME=192.168.79.91 \
rosrun web_video_server web_video_server _port:=8080
```

看到下面这行即可：

```text
Waiting For connections on 0.0.0.0:8080
```

## 3. 小车终端 3：启动视觉 JSON 输出

```bash
cd /home/clbrobot/catkin_ws/yolov5_emotieff_vgaf_jetson

PYTHONPATH=./yolov5-face python3 jetson_emotion_to_json.py \
  --source 'http://127.0.0.1:8080/stream?topic=/camera/rgb/image_raw' \
  --face-weights yolov5-face/weights/yolov5s-face_torch14.pt \
  --emotieff-model-name enet_b0_8_best_vgaf \
  --emotieff-engine torch \
  --output-json outputs/latest_emotion.json \
  --crop-margin 0.30 \
  --sample-step 3
```

确认 JSON 正在更新：

```bash
cat /home/clbrobot/catkin_ws/yolov5_emotieff_vgaf_jetson/outputs/latest_emotion.json
```

## 4. 小车终端 4：启动视觉参数 API

```bash
cd /home/clbrobot/catkin_ws/yolov5_emotieff_vgaf_jetson

python3 vision_result_api.py \
  --json-path outputs/latest_emotion.json \
  --host 0.0.0.0 \
  --port 8090
```

小车本机测试：

```bash
curl http://127.0.0.1:8090/health
curl http://127.0.0.1:8090/vision/latest
curl http://127.0.0.1:8090/vision/decision-input
```

## 5. 虚拟机终端 1：端口转发

在 Linux 虚拟机执行：

```bash
ssh \
  -L 8090:127.0.0.1:8090 \
  -L 8080:127.0.0.1:8080 \
  clbrobot@192.168.79.91
```

这个终端保持不要关。转发成功后，虚拟机访问：

```text
http://127.0.0.1:8090/vision/latest
http://127.0.0.1:8080/stream?topic=/camera/rgb/image_raw
```

实际访问的是小车上的：

```text
http://127.0.0.1:8090/vision/latest
http://127.0.0.1:8080/stream?topic=/camera/rgb/image_raw
```

## 6. 虚拟机终端 2：启动前端页面

把前端文件放到虚拟机，例如：

```text
/home/lzc/桌面/vision_dashboard.html
```

然后运行：

```bash
cd /home/lzc/桌面
python3 -m http.server 9000
```

在虚拟机浏览器打开：

```text
http://127.0.0.1:9000/vision_dashboard.html
```

页面默认读取：

```text
http://127.0.0.1:8090/vision/latest
http://127.0.0.1:8080/stream?topic=/camera/rgb/image_raw
```

如果不使用 SSH 端口转发，也可以把页面里的 API 地址改成：

```text
http://192.168.79.91:8090
```

同时把页面里的视频流地址改成：

```text
http://192.168.79.91:8080/stream?topic=/camera/rgb/image_raw
```

## 前端文件位置

Windows 工程里的前端文件：

```text
D:\desktop\release_build\yolov5_emotieff_vgaf_windows\frontend\vision_dashboard.html
```

把这个文件复制到 Linux 虚拟机即可使用，不需要安装 npm。
