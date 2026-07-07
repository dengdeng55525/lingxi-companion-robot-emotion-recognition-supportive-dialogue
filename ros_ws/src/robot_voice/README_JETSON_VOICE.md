# Jetson Nano voice control deployment

This package turns recognized text on `iat_text` into velocity commands on
`/cmd_vel`. For Jetson Nano, use the Vosk launch path first because the bundled
iFlyTek `libmsc.so` files are x86/x64 only and cannot run on ARM64.

## Compatibility note

The original workspace was built for Ubuntu 20.04 + ROS 1 Noetic. Jetson Nano
official JetPack 4.x images are Ubuntu 18.04 based, where ROS Melodic is the
usual ROS 1 choice. If your Nano already runs Ubuntu 20.04, use Noetic. If it
runs the official Nano image, either port this workspace to Melodic or install a
community Ubuntu 20.04 image before using Noetic.

## Install on Ubuntu 20.04 + ROS Noetic

```bash
sudo apt update
sudo apt install -y ros-noetic-desktop-full python3-rosdep python3-catkin-tools \
  python3-pip portaudio19-dev libasound2-dev alsa-utils
sudo rosdep init
rosdep update
```

Copy this workspace to the Nano, then build only from source:

```bash
cd ~/catkin_ws
rm -rf build devel
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source devel/setup.bash
```

## Install voice dependencies

```bash
python3 -m pip install --user vosk sounddevice
mkdir -p ~/models
cd ~/models
wget https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip
unzip vosk-model-small-cn-0.22.zip
```

Check the microphone:

```bash
arecord -l
arecord -D plughw:1,0 -f S16_LE -r 16000 -c 1 /tmp/mic.wav
aplay /tmp/mic.wav
```

If your USB microphone is not the default input, pass part of its ALSA device
name to `device_name`.

## Launch

Start the base driver first. It must subscribe to `geometry_msgs/Twist` on
`/cmd_vel` or on a topic you remap with `cmd_vel_topic`.

```bash
roscore
```

In another terminal:

```bash
cd ~/catkin_ws
source devel/setup.bash
roslaunch robot_voice jetson_voice_control.launch
```

If your base driver listens on a different topic:

```bash
roslaunch robot_voice jetson_voice_control.launch cmd_vel_topic:=/your_base/cmd_vel
```

Try these commands in Mandarin:

- `前进`
- `后退`
- `左转`
- `右转`
- `停止`

## Verify each layer

Check speech recognition:

```bash
rostopic echo /iat_text
```

Bypass speech recognition and test motion mapping:

```bash
rostopic pub -1 /iat_text std_msgs/String "data: '前进'"
rostopic echo /cmd_vel
rostopic pub -1 /iat_text std_msgs/String "data: '停止'"
```

If `/cmd_vel` changes but the car does not move, the missing piece is the base
driver. It should translate `/cmd_vel` into motor controller commands for your
Nano car chassis.

## Optional iFlyTek ARM64 path

To keep using the original iFlyTek nodes, obtain an ARM64/aarch64 Linux
`libmsc.so` that matches your appid, put it here:

```text
src/robot_voice/libs/aarch64/libmsc.so
```

Then rebuild:

```bash
cd ~/catkin_ws
rm -rf build devel
catkin_make
```

Without that ARM64 library, `iat_publish` and `tts_subscribe` are skipped during
the build, while `voice_control` and the Vosk publisher still build normally.
