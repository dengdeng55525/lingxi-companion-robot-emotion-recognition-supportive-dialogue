#!/usr/bin/env python3
import argparse
import json
import os
import queue
import shlex
import subprocess
import time


COMMANDS = (
    ("停止", ("停止", "停车"), 0.0, 0.0),
    ("前进", ("前进", "向前"), 1.0, 0.0),
    ("后退", ("后退", "向后"), -1.0, 0.0),
    ("左转", ("左转", "向左"), 0.0, 1.0),
    ("右转", ("右转", "向右"), 0.0, -1.0),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Local Jetson voice command controller")
    parser.add_argument("--model-path", default=os.path.expanduser("~/models/vosk-model-small-cn-0.22"))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--block-size", type=int, default=8000)
    parser.add_argument("--device-name", default="")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--linear-speed", type=float, default=0.15)
    parser.add_argument("--angular-speed", type=float, default=0.45)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument(
        "--ros-setup",
        default="source /opt/ros/melodic/setup.bash && source ~/catkin_ws/devel/setup.bash",
    )
    return parser.parse_args()


def load_modules():
    try:
        import sounddevice as sd
        import vosk
    except ImportError as exc:
        raise SystemExit(
            "Missing Python dependency: %s. Install with: "
            "python3 -m pip install --user vosk sounddevice" % exc.name
        )
    return sd, vosk


def select_device(sd, device_name):
    if not device_name:
        return None

    for index, device in enumerate(sd.query_devices()):
        if device_name.lower() in device["name"].lower():
            return index

    raise SystemExit("Audio input device containing '%s' was not found" % device_name)


def twist_yaml(linear, angular):
    return (
        "{linear: {x: %.6f, y: 0.0, z: 0.0}, "
        "angular: {x: 0.0, y: 0.0, z: %.6f}}"
    ) % (linear, angular)


class CmdVelRunner:
    def __init__(self, args):
        self.args = args
        self.process = None

    def send(self, command_name, linear, angular):
        self.stop_current()
        if command_name == "停止":
            self.publish_once(0.0, 0.0)
            return

        msg = twist_yaml(linear, angular)
        stop_msg = twist_yaml(0.0, 0.0)
        shell_cmd = (
            "%s; "
            "timeout %.3fs rostopic pub -r %.3f %s geometry_msgs/Twist %s >/dev/null; "
            "rostopic pub -1 %s geometry_msgs/Twist %s >/dev/null"
        ) % (
            self.args.ros_setup,
            self.args.duration,
            self.args.rate,
            shlex.quote(self.args.cmd_vel_topic),
            shlex.quote(msg),
            shlex.quote(self.args.cmd_vel_topic),
            shlex.quote(stop_msg),
        )
        self.process = subprocess.Popen(["bash", "-lc", shell_cmd])

    def publish_once(self, linear, angular):
        msg = twist_yaml(linear, angular)
        shell_cmd = "%s; rostopic pub -1 %s geometry_msgs/Twist %s >/dev/null" % (
            self.args.ros_setup,
            shlex.quote(self.args.cmd_vel_topic),
            shlex.quote(msg),
        )
        subprocess.call(["bash", "-lc", shell_cmd])

    def stop_current(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None


def parse_command(text, args):
    normalized = text.replace(" ", "")
    for name, keywords, linear_scale, angular_scale in COMMANDS:
        if any(keyword in normalized for keyword in keywords):
            return (
                name,
                linear_scale * args.linear_speed,
                angular_scale * args.angular_speed,
            )
    return None


def main():
    args = parse_args()
    sd, vosk = load_modules()

    if not os.path.isdir(args.model_path):
        raise SystemExit("Vosk model directory does not exist: %s" % args.model_path)

    audio_queue = queue.Queue()
    device = select_device(sd, args.device_name)
    model = vosk.Model(args.model_path)
    recognizer = vosk.KaldiRecognizer(model, args.sample_rate)
    cmd_vel = CmdVelRunner(args)
    last_command = ""
    last_time = 0.0

    def audio_callback(indata, frames, time_info, status):
        if status:
            print("Audio status:", status)
        audio_queue.put(bytes(indata))

    print("Jetson local voice controller is listening.")
    print("Commands: 前进 / 后退 / 左转 / 右转 / 停止")

    with sd.RawInputStream(
        samplerate=args.sample_rate,
        blocksize=args.block_size,
        device=device,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        while True:
            data = audio_queue.get()
            if not recognizer.AcceptWaveform(data):
                continue

            result = json.loads(recognizer.Result())
            text = result.get("text", "").replace(" ", "")
            if not text:
                continue

            print("Recognized:", text)
            command = parse_command(text, args)
            if not command:
                continue

            name, linear, angular = command
            now = time.time()
            if name == last_command and now - last_time < 1.0:
                continue

            last_command = name
            last_time = now
            print("Command:", name, "linear=%.3f" % linear, "angular=%.3f" % angular)
            cmd_vel.send(name, linear, angular)


if __name__ == "__main__":
    main()
