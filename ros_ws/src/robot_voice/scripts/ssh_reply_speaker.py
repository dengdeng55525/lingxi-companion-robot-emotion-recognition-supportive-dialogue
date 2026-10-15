#!/usr/bin/env python3
import base64
import os
import queue
import shlex
import subprocess
import threading
import tempfile

import rospy
from std_msgs.msg import String


class SshReplySpeaker:
    def __init__(self):
        self.reply_topic = rospy.get_param("~reply_topic", "ai_reply")
        self.global_stop_topic = rospy.get_param("~global_stop_topic", "robot_stop_all")
        self.jetson_user = rospy.get_param("~jetson_user", "clbrobot")
        self.jetson_host = rospy.get_param("~jetson_host", "192.168.210.91")
        self.ssh_connect_timeout = int(rospy.get_param("~ssh_connect_timeout", 5))
        self.command_timeout = float(rospy.get_param("~command_timeout", 45.0))
        self.backend = rospy.get_param("~backend", "auto")
        self.audio_device = rospy.get_param("~audio_device", "")
        self.voice = rospy.get_param("~voice", "zh")
        self.speed = int(rospy.get_param("~speed", 155))
        self.amplitude = int(rospy.get_param("~amplitude", 130))
        self.max_chars = int(rospy.get_param("~max_chars", 120))

        self._target = "%s@%s" % (self.jetson_user, self.jetson_host)
        self._queue = queue.Queue()
        self._process_lock = threading.Lock()
        self._current_process = None
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        rospy.Subscriber(self.reply_topic, String, self._on_reply, queue_size=10)
        rospy.Subscriber(self.global_stop_topic, String, self._on_global_stop, queue_size=10)
        rospy.loginfo(
            "SSH reply speaker ready: topic=%s target=%s backend=%s",
            self.reply_topic,
            self._target,
            self.backend,
        )

    def _on_reply(self, msg):
        text = msg.data.strip()
        if not text:
            return
        if self.max_chars > 0 and len(text) > self.max_chars:
            text = text[: self.max_chars]
        self._queue.put(text)

    def _on_global_stop(self, _msg):
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        with self._process_lock:
            process = self._current_process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.8)
            except subprocess.TimeoutExpired:
                process.kill()

    def _worker_loop(self):
        while not rospy.is_shutdown():
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._speak(text)

    def _speak(self, text):
        rospy.loginfo("Speaking reply on Jetson: %s", text)
        if self.backend in ("local-edge-tts", "vm-edge-tts"):
            self._speak_with_local_edge_tts(text)
            return

        cmd = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=%d" % self.ssh_connect_timeout,
            self._target,
            "bash",
            "-s",
        ]
        process = None
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            with self._process_lock:
                self._current_process = process
            stdout, stderr = process.communicate(
                input=self._remote_script(text),
                timeout=self.command_timeout,
            )
        except subprocess.TimeoutExpired:
            if process is not None:
                process.kill()
                process.communicate()
            rospy.logerr("Timed out while speaking reply on Jetson.")
            return
        except OSError as exc:
            rospy.logerr("Failed to start SSH speaker: %s", exc)
            return
        finally:
            with self._process_lock:
                if self._current_process is process:
                    self._current_process = None

        if process.returncode != 0:
            rospy.logerr("Jetson speaker failed: %s", stderr.strip())
            return
        if stderr.strip():
            rospy.logwarn("Jetson speaker stderr: %s", stderr.strip())
        if stdout.strip():
            rospy.logdebug("Jetson speaker stdout: %s", stdout.strip())

    def _speak_with_local_edge_tts(self, text):
        voice = self.voice
        if not voice or voice == "zh":
            voice = "zh-CN-XiaoxiaoNeural"

        fd, mp3_path = tempfile.mkstemp(prefix="robot_ai_reply_", suffix=".mp3")
        os.close(fd)
        try:
            edge_cmd = [
                "python3",
                "-m",
                "edge_tts",
                "--voice",
                voice,
                "--text",
                text,
                "--write-media",
                mp3_path,
            ]
            result = subprocess.run(
                edge_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.command_timeout,
            )
            if result.returncode != 0:
                rospy.logerr("Local edge-tts failed: %s", result.stderr.strip())
                return

            with open(mp3_path, "rb") as handle:
                audio = handle.read()
            if not audio:
                rospy.logerr("Local edge-tts generated empty audio.")
                return
            self._play_mp3_on_jetson(audio)
        except subprocess.TimeoutExpired:
            rospy.logerr("Local edge-tts timed out.")
        except OSError as exc:
            rospy.logerr("Local edge-tts/playback failed: %s", exc)
        finally:
            try:
                os.remove(mp3_path)
            except OSError:
                pass

    def _play_mp3_on_jetson(self, audio):
        device_part = ""
        if self.audio_device == "pulse":
            device_part = "-o pulse"
        elif self.audio_device:
            device_part = "-o alsa -a %s" % shlex.quote(self.audio_device)

        remote_cmd = (
            "cat > /tmp/robot_ai_reply.mp3 && "
            "mpg123 -q %s /tmp/robot_ai_reply.mp3" % (device_part or "-o alsa")
        )
        cmd = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=%d" % self.ssh_connect_timeout,
            self._target,
            "bash",
            "-lc",
            remote_cmd,
        ]
        process = None
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            with self._process_lock:
                self._current_process = process
            stdout, stderr = process.communicate(input=audio, timeout=self.command_timeout)
        except subprocess.TimeoutExpired:
            if process is not None:
                process.kill()
                process.communicate()
            rospy.logerr("Timed out while playing local edge-tts audio on Jetson.")
            return
        finally:
            with self._process_lock:
                if self._current_process is process:
                    self._current_process = None

        if process.returncode != 0:
            detail = stderr.decode("utf-8", "replace").strip()
            if not detail:
                detail = "mpg123 exited with code %s" % process.returncode
            rospy.logerr(
                "Jetson MP3 playback failed: %s",
                detail,
            )
            return
        if stderr.strip():
            rospy.logwarn(
                "Jetson MP3 playback stderr: %s",
                stderr.decode("utf-8", "replace").strip(),
            )
        if stdout.strip():
            rospy.logdebug(
                "Jetson MP3 playback stdout: %s",
                stdout.decode("utf-8", "replace").strip(),
            )

    def _remote_script(self, text):
        text_b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
        device_b64 = base64.b64encode(self.audio_device.encode("utf-8")).decode("ascii")
        voice_b64 = base64.b64encode(self.voice.encode("utf-8")).decode("ascii")
        backend_b64 = base64.b64encode(self.backend.encode("utf-8")).decode("ascii")
        script = """\
set -u
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
TEXT_FILE=/tmp/robot_ai_reply.txt
MP3_FILE=/tmp/robot_ai_reply.mp3
WAV_FILE=/tmp/robot_ai_reply.wav
printf '__TEXT_B64__' | base64 -d > "$TEXT_FILE"
AUDIO_DEVICE="$(printf '__DEVICE_B64__' | base64 -d)"
VOICE="$(printf '__VOICE_B64__' | base64 -d)"
BACKEND="$(printf '__BACKEND_B64__' | base64 -d)"
TEXT="$(cat "$TEXT_FILE")"

play_wav() {
    if ! command -v aplay >/dev/null 2>&1; then
        return 1
    fi
    if [ -n "$AUDIO_DEVICE" ]; then
        aplay -q -D "$AUDIO_DEVICE" "$1"
    else
        aplay -q "$1"
    fi
}

play_stdin_wav() {
    if ! command -v aplay >/dev/null 2>&1; then
        cat >/dev/null
        return 1
    fi
    if [ -n "$AUDIO_DEVICE" ]; then
        aplay -q -D "$AUDIO_DEVICE"
    else
        aplay -q
    fi
}

try_edge_tts() {
    command -v edge-tts >/dev/null 2>&1 || return 1
    edge-tts --voice zh-CN-XiaoxiaoNeural --text "$TEXT" --write-media "$MP3_FILE" >/tmp/robot_edge_tts.log 2>&1 || return 1
    if command -v mpg123 >/dev/null 2>&1; then
        if [ "$AUDIO_DEVICE" = "pulse" ]; then
            mpg123 -q -o pulse "$MP3_FILE"
        elif [ -n "$AUDIO_DEVICE" ]; then
            mpg123 -q -o alsa -a "$AUDIO_DEVICE" "$MP3_FILE"
        else
            mpg123 -q -o alsa "$MP3_FILE"
        fi
        return $?
    fi
    if command -v ffplay >/dev/null 2>&1; then
        ffplay -nodisp -autoexit -loglevel quiet "$MP3_FILE"
        return $?
    fi
    if command -v ffmpeg >/dev/null 2>&1; then
        ffmpeg -y -loglevel quiet -i "$MP3_FILE" "$WAV_FILE" && play_wav "$WAV_FILE"
        return $?
    fi
    return 1
}

try_espeak_ng() {
    command -v espeak-ng >/dev/null 2>&1 || return 1
    espeak-ng -v "$VOICE" -s __SPEED__ -a __AMPLITUDE__ --stdout "$TEXT" | play_stdin_wav
}

try_espeak() {
    command -v espeak >/dev/null 2>&1 || return 1
    espeak -v "$VOICE" -s __SPEED__ -a __AMPLITUDE__ --stdout "$TEXT" | play_stdin_wav
}

try_spd_say() {
    command -v spd-say >/dev/null 2>&1 || return 1
    spd-say -l zh "$TEXT"
}

try_pico2wave() {
    command -v pico2wave >/dev/null 2>&1 || return 1
    pico2wave -l zh-CN -w "$WAV_FILE" "$TEXT" && play_wav "$WAV_FILE"
}

case "$BACKEND" in
    edge-tts) try_edge_tts && exit 0 ;;
    espeak-ng) try_espeak_ng && exit 0 ;;
    espeak) try_espeak && exit 0 ;;
    spd-say) try_spd_say && exit 0 ;;
    pico2wave) try_pico2wave && exit 0 ;;
    auto|"")
        try_edge_tts && exit 0
        try_espeak_ng && exit 0
        try_espeak && exit 0
        try_spd_say && exit 0
        try_pico2wave && exit 0
        ;;
    *)
        echo "Unknown TTS backend: $BACKEND" >&2
        exit 2
        ;;
esac

echo "No usable TTS backend found. Try: sudo apt-get install -y espeak-ng alsa-utils" >&2
exit 42
"""
        return (
            script.replace("__TEXT_B64__", text_b64)
            .replace("__DEVICE_B64__", device_b64)
            .replace("__VOICE_B64__", voice_b64)
            .replace("__BACKEND_B64__", backend_b64)
            .replace("__SPEED__", str(self.speed))
            .replace("__AMPLITUDE__", str(self.amplitude))
        )


if __name__ == "__main__":
    rospy.init_node("ssh_reply_speaker")
    SshReplySpeaker()
    rospy.spin()
