# 一键启动小车视觉系统和虚拟机前端

这套启动方式只需要两个终端：

- 小车终端：启动相机、视频流、表情检测、JSON API。
- 虚拟机终端：启动 SSH 端口转发和前端页面服务。

## 1. 上传启动脚本

把小车启动脚本传到小车：

```bash
scp /home/lzc/桌面/start_jetson_vision_stack.sh \
  clbrobot@192.168.210.91:/home/clbrobot/catkin_ws/yolov5_emotieff_vgaf_jetson/scripts/start_jetson_vision_stack.sh
```

在小车上加执行权限：

```bash
chmod +x /home/clbrobot/catkin_ws/yolov5_emotieff_vgaf_jetson/scripts/start_jetson_vision_stack.sh
```

虚拟机启动脚本放在：

```text
/home/lzc/桌面/start_vm_dashboard.sh
```

加执行权限：

```bash
chmod +x /home/lzc/桌面/start_vm_dashboard.sh
```

前端页面放在：

```text
/home/lzc/桌面/vision_dashboard.html
```

## 2. 小车终端：启动全部后端

```bash
/home/clbrobot/catkin_ws/yolov5_emotieff_vgaf_jetson/scripts/start_jetson_vision_stack.sh
```

这个脚本会启动：

- `roslaunch clbrobot astra_laser.launch`
- `web_video_server`，端口 `8080`
- `jetson_emotion_to_json.py`
- `vision_result_api.py`，端口 `8090`

如果要停止全部服务，在这个终端按：

```text
Ctrl-C
```

日志会保存在：

```text
/home/clbrobot/catkin_ws/yolov5_emotieff_vgaf_jetson/logs/
```

## 3. 虚拟机终端：启动前端和端口转发

```bash
/home/lzc/桌面/start_vm_dashboard.sh
```

这个脚本会启动：

- 前端页面服务：`http://127.0.0.1:9000/vision_dashboard.html`
- API 转发：`127.0.0.1:8090 -> 小车 127.0.0.1:8090`
- 视频转发：`127.0.0.1:8080 -> 小车 127.0.0.1:8080`

如果提示输入密码，输入小车用户 `clbrobot` 的密码。

打开虚拟机浏览器：

```text
http://127.0.0.1:9000/vision_dashboard.html
```

## 4. 常用自定义参数

小车端可以临时改参数，例如：

```bash
CROP_MARGIN=0.80 SAMPLE_STEP=2 \
/home/clbrobot/catkin_ws/yolov5_emotieff_vgaf_jetson/scripts/start_jetson_vision_stack.sh
```

虚拟机端如果端口被占用，可以改前端端口：

```bash
WEB_PORT=9010 /home/lzc/桌面/start_vm_dashboard.sh
```

然后打开：

```text
http://127.0.0.1:9010/vision_dashboard.html
```
