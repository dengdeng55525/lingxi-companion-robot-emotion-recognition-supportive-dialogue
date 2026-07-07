#!/usr/bin/env python3
import json
import shlex
import subprocess
import sys
import threading
import time

import rospy
from std_msgs.msg import String


def _load_vosk():
    try:
        import vosk
    except ImportError as exc:
        rospy.logfatal(
            "Missing Python dependency: %s. Install in the VM with "
            "`python3 -m pip install --user vosk`.",
            exc.name,
        )
        raise
    return vosk


class SshArecordVoskIat:
    def __init__(self):
        self.jetson_user = rospy.get_param("~jetson_user", "clbrobot")
        self.jetson_host = rospy.get_param("~jetson_host", "192.168.64.91")
        self.audio_device = rospy.get_param("~audio_device", "")
        self.model_path = rospy.get_param("~model_path", "")
        self.sample_rate = int(rospy.get_param("~sample_rate", 16000))
        self.channels = int(rospy.get_param("~channels", 1))
        self.chunk_size = int(rospy.get_param("~chunk_size", 4000))
        self.publish_topic = rospy.get_param("~publish_topic", "iat_text")
        self.publish_partials = bool(rospy.get_param("~publish_partials", False))
        self.wake_word_aliases = rospy.get_param("~wake_word_aliases", "林夕,凌曦,灵溪,灵息,领席,零希")
        self.ssh_connect_timeout = int(rospy.get_param("~ssh_connect_timeout", 5))
        self.reconnect_delay = float(rospy.get_param("~reconnect_delay", 2.0))

        if not self.model_path:
            rospy.logfatal("Parameter ~model_path is required for Vosk recognition.")
            raise RuntimeError("missing Vosk model_path")

        vosk = _load_vosk()
        self.model = vosk.Model(self.model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
        self.publisher = rospy.Publisher(self.publish_topic, String, queue_size=10)
        self._target = "%s@%s" % (self.jetson_user, self.jetson_host)
        self._process = None

        rospy.loginfo(
            "SSH arecord Vosk IAT ready: target=%s device=%s topic=%s model=%s",
            self._target,
            self.audio_device or "default",
            self.publish_topic,
            self.model_path,
        )

    def spin(self):
        while not rospy.is_shutdown():
            self._run_stream_once()
            if not rospy.is_shutdown():
                rospy.logwarn("Jetson audio stream ended; reconnecting in %.1fs.", self.reconnect_delay)
                time.sleep(self.reconnect_delay)

    def _run_stream_once(self):
        cmd = self._ssh_arecord_cmd()
        rospy.loginfo("Starting Jetson microphone stream.")
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        stderr_thread = threading.Thread(
            target=self._log_stderr,
            args=(self._process,),
            daemon=True,
        )
        stderr_thread.start()

        try:
            while not rospy.is_shutdown():
                data = self._process.stdout.read(self.chunk_size)
                if not data:
                    break
                self._accept_audio(data)
        finally:
            self._stop_process()

    def _ssh_arecord_cmd(self):
        remote = ["arecord", "-q"]
        if self.audio_device:
            remote.extend(["-D", self.audio_device])
        remote.extend(
            [
                "--format=S16_LE",
                "--rate=%d" % self.sample_rate,
                "--channels=%d" % self.channels,
                "--file-type=raw",
                "-",
            ]
        )

        remote_cmd = " ".join(shlex.quote(part) for part in remote)
        rospy.loginfo("Jetson arecord command: %s", remote_cmd)
        return [
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

    def _accept_audio(self, data):
        if self.recognizer.AcceptWaveform(data):
            result = json.loads(self.recognizer.Result())
            text = self._normalize_text(result.get("text", "").replace(" ", ""))
            if text:
                rospy.loginfo("Recognized from Jetson mic: %s", text)
                self.publisher.publish(String(text))
        elif self.publish_partials:
            result = json.loads(self.recognizer.PartialResult())
            partial = self._normalize_text(result.get("partial", "").replace(" ", ""))
            if partial:
                self.publisher.publish(String(partial))

    def _normalize_text(self, text):
        aliases = self.wake_word_aliases
        if isinstance(aliases, str):
            aliases = [item.strip() for item in aliases.split(",") if item.strip()]
        for alias in aliases:
            if alias:
                text = text.replace(alias, "灵犀")
        return text

    def _log_stderr(self, process):
        for raw in iter(process.stderr.readline, b""):
            if not raw:
                break
            line = raw.decode("utf-8", "replace").strip()
            if line:
                rospy.logwarn("Jetson arecord: %s", line)

    def _stop_process(self):
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    rospy.init_node("ssh_arecord_vosk_iat")
    try:
        node = SshArecordVoskIat()
        node.spin()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        rospy.logfatal("ssh_arecord_vosk_iat failed: %s", exc)
        sys.exit(1)
