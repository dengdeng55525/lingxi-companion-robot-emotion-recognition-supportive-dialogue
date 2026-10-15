#!/usr/bin/env python3
import difflib
import os
import queue
import re
import shlex
import subprocess
import threading

import rospy
from std_msgs.msg import String


SUPPORTED_EXTENSIONS = (".mp3", ".wav")


class MusicPlayerNode:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "iat_text")
        self.reply_topic = rospy.get_param("~reply_topic", "ai_reply")
        self.global_stop_topic = rospy.get_param("~global_stop_topic", "robot_stop_all")
        self.music_dir = os.path.expanduser(
            rospy.get_param("~music_dir", "~/Music/robot_music")
        )
        self.jetson_user = rospy.get_param("~jetson_user", "clbrobot")
        self.jetson_host = rospy.get_param("~jetson_host", "192.168.210.91")
        self.audio_device = rospy.get_param("~audio_device", "plughw:2,0")
        self.ssh_connect_timeout = int(rospy.get_param("~ssh_connect_timeout", 5))
        self.confirm_score = float(rospy.get_param("~confirm_score", 0.50))
        self.auto_score = float(rospy.get_param("~auto_score", 0.92))
        self.close_margin = float(rospy.get_param("~close_margin", 0.08))
        self.volume_step = int(rospy.get_param("~volume_step", 8))
        self.volume_percent = int(rospy.get_param("~initial_volume", 80))
        self.play_after_reply_delay = float(rospy.get_param("~play_after_reply_delay", 2.2))
        self.stop_reply_delay = float(rospy.get_param("~stop_reply_delay", 1.2))
        self.announce_before_play = bool(rospy.get_param("~announce_before_play", False))
        self.announce_after_stop = bool(rospy.get_param("~announce_after_stop", False))
        self.wake_word_required = bool(rospy.get_param("~wake_word_required", True))
        self.wake_session_enabled = bool(rospy.get_param("~wake_session_enabled", True))
        wake_words_param = rospy.get_param("~wake_words", "灵犀")
        if isinstance(wake_words_param, str):
            self.wake_words = [word.strip() for word in wake_words_param.split(",") if word.strip()]
        else:
            self.wake_words = [str(word).strip() for word in wake_words_param if str(word).strip()]
        self._conversation_active = not self.wake_word_required

        self._target = "%s@%s" % (self.jetson_user, self.jetson_host)
        self._lock = threading.Lock()
        self._current_process = None
        self._pending_match = None
        self._play_request_id = 0
        self._stopped_processes = set()
        self._queue = queue.Queue()

        self.reply_pub = rospy.Publisher(self.reply_topic, String, queue_size=10)
        rospy.Subscriber(self.input_topic, String, self._on_text, queue_size=10)
        rospy.Subscriber(self.global_stop_topic, String, self._on_global_stop, queue_size=10)

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        rospy.loginfo(
            "Music player ready: music_dir=%s target=%s device=%s",
            self.music_dir,
            self._target,
            self.audio_device,
        )

    def _on_text(self, msg):
        raw_text = msg.data.strip()
        if not raw_text:
            return

        text = self._prepare_session_text(raw_text)
        if text is None:
            return
        normalized = self._normalize(text)

        if self._pending_match is not None and self._is_yes(normalized):
            item = self._pending_match
            self._pending_match = None
            if self.announce_before_play:
                self.reply_pub.publish(String("好的，开始播放%s。" % item["display"]))
            self._enqueue_play(item, self.play_after_reply_delay)
            return

        if self._pending_match is not None and self._is_no(normalized):
            self._pending_match = None
            self.reply_pub.publish(String("好，那我先不播放。"))
            return

        if self._is_stop_music(normalized):
            self._pending_match = None
            self._stop_current()
            if self.announce_after_stop:
                self._publish_reply_later("已停止播放。", self.stop_reply_delay)
            return

        volume_delta = self._volume_delta(normalized)
        volume_handled = False
        if volume_delta is not None:
            self.volume_percent = max(0, min(100, self.volume_percent + volume_delta))
            self._set_remote_volume(self.volume_percent)
            if not self._is_playing():
                self.reply_pub.publish(String("音量已调到%d。" % self.volume_percent))
            volume_handled = True

        query = self._extract_music_query(text)
        if not query:
            if volume_handled:
                return
            return

        matches = self._find_matches(query)
        if not matches:
            self.reply_pub.publish(String("我没有在音乐文件夹里找到%s。" % query))
            return

        best = matches[0]
        second_score = matches[1]["score"] if len(matches) > 1 else 0.0
        if self._is_direct_match(query, best) and best["score"] >= self.confirm_score:
            if self.announce_before_play:
                self.reply_pub.publish(String("好的，开始播放%s。" % best["display"]))
            self._enqueue_play(best, self.play_after_reply_delay)
            return

        if best["score"] >= self.confirm_score and best["score"] - second_score >= self.close_margin:
            self._pending_match = best
            self.reply_pub.publish(
                String("我找到的是%s，你是想听这首吗？" % best["display"])
            )
            return

        names = "、".join(match["display"] for match in matches[:3])
        self.reply_pub.publish(String("我找到几个可能的歌曲：%s。请说完整一点。" % names))

    def _worker_loop(self):
        while not rospy.is_shutdown():
            try:
                action, item, request_id = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if action == "play":
                with self._lock:
                    if request_id != self._play_request_id:
                        continue
                self._play_item(item)

    def _enqueue_play(self, item, delay):
        with self._lock:
            self._play_request_id += 1
            request_id = self._play_request_id
        timer = threading.Timer(delay, lambda: self._queue.put(("play", item, request_id)))
        timer.daemon = True
        timer.start()

    def _publish_reply_later(self, text, delay):
        timer = threading.Timer(delay, lambda: self.reply_pub.publish(String(text)))
        timer.daemon = True
        timer.start()

    def _find_matches(self, query):
        query_key = self._normalize_query(query)
        if not query_key:
            return []

        items = self._scan_music()
        for item in items:
            item["score"] = self._score(query_key, item)
        items.sort(key=lambda item: item["score"], reverse=True)
        return [item for item in items if item["score"] >= 0.25]

    def _scan_music(self):
        if not os.path.isdir(self.music_dir):
            os.makedirs(self.music_dir, exist_ok=True)
            return []

        items = []
        for root, _, files in os.walk(self.music_dir):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue
                path = os.path.join(root, filename)
                stem = os.path.splitext(filename)[0]
                rel = os.path.relpath(path, self.music_dir)
                parent = os.path.basename(os.path.dirname(path))
                display = stem
                if parent and parent != "." and parent != os.path.basename(self.music_dir):
                    display = "%s - %s" % (parent, stem)
                key = "%s %s %s" % (stem, rel, parent)
                items.append(
                    {
                        "path": path,
                        "display": display,
                        "title_key": self._normalize_query(stem),
                        "search_key": self._normalize_query(key),
                        "ext": ext,
                    }
                )
        return items

    def _score(self, query_key, item):
        search_key = item["search_key"]
        title_key = item["title_key"]
        if not search_key:
            return 0.0
        if query_key == title_key or query_key == search_key:
            return 1.0
        score = difflib.SequenceMatcher(None, query_key, search_key).ratio()
        title_score = difflib.SequenceMatcher(None, query_key, title_key).ratio()
        score = max(score, title_score)
        if query_key in search_key:
            score = max(score, min(0.90, 0.55 + 0.35 * len(query_key) / max(len(search_key), 1)))
        if title_key and title_key in query_key:
            score = max(score, min(0.96, 0.70 + 0.26 * len(title_key) / max(len(query_key), 1)))
        return score

    def _is_direct_match(self, query, item):
        query_key = self._normalize_query(query)
        title_key = item["title_key"]
        search_key = item["search_key"]
        if query_key in (title_key, search_key):
            return True
        return bool(title_key and len(title_key) >= 3 and title_key in query_key)

    def _extract_music_query(self, text):
        normalized = self._normalize(text)
        if not any(word in normalized for word in ("播放", "放歌", "放首", "想听", "听歌", "听")):
            return ""
        if self._is_stop_music(normalized):
            return ""

        query = text
        replacements = (
            "请你", "帮我", "给我", "我想", "我要", "想要", "可以",
            "把", "然后", "再", "接着",
            "声音调小一点", "音量调小一点", "声音小一点", "音量小一点",
            "声音调大一点", "音量调大一点", "声音大一点", "音量大一点",
            "调小一点", "调大一点",
            "播放一下", "播放", "放一下", "放一首", "放首", "放歌",
            "想听一下", "想听", "听一下", "听听", "听歌", "听",
            "歌曲", "音乐", "一首", "一下", "吧", "好吗", "可以吗",
        )
        for word in replacements:
            query = query.replace(word, "")
        query = re.sub(r"[，,。！？?；;：:、（）()\[\]【】《》“”\"'\s]+", "", query)
        query = query.replace("的", "")
        return query.strip()

    def _normalize_query(self, text):
        text = self._normalize(text)
        for word in ("歌手", "歌曲", "音乐", "播放", "想听", "听", "的"):
            text = text.replace(word, "")
        return text

    def _normalize(self, text):
        text = text.lower()
        return re.sub(r"[\s，,。！？?；;：:、（）()\[\]【】《》“”\"'._\-]+", "", text)

    def _strip_wake_word(self, text):
        if not self.wake_word_required:
            return text.strip()

        normalized = self._normalize(text)
        for wake_word in self.wake_words:
            wake = self._normalize(wake_word)
            if not wake:
                continue
            if normalized == wake:
                return ""
            if normalized.startswith(wake):
                return normalized[len(wake):].strip()
        return None

    def _prepare_session_text(self, text):
        if not self.wake_word_required:
            return text.strip()

        stripped = self._strip_wake_word(text)
        if not self.wake_session_enabled:
            return stripped

        if self._conversation_active and self._is_sleep_command(text):
            self._conversation_active = False
            self._pending_match = None
            return None

        if self._conversation_active:
            if stripped is not None:
                return stripped
            return text.strip()

        if stripped is None:
            return None

        if self._is_sleep_text(stripped):
            return None

        self._conversation_active = True
        return stripped

    def _is_sleep_command(self, text):
        stripped = self._strip_wake_word(text)
        return stripped is not None and self._is_sleep_text(stripped)

    def _is_sleep_text(self, text):
        normalized = self._normalize(text)
        return normalized in (
            "再见",
            "拜拜",
            "退出",
            "结束对话",
            "停止对话",
            "先这样",
            "不用了",
            "休息吧",
        )

    def _on_global_stop(self, _msg):
        self._pending_match = None
        self._stop_current()

    def _is_yes(self, normalized):
        return normalized in ("是", "是的", "对", "对的", "没错", "嗯", "好", "好的", "可以", "确认", "播放")

    def _is_no(self, normalized):
        return normalized in ("不是", "不对", "不是的", "否", "不要", "别放", "不播放", "算了")

    def _is_stop_music(self, normalized):
        return any(
            word in normalized
            for word in (
                "停止播放", "停播放", "停音乐", "停止音乐", "别放了", "关音乐", "关闭音乐", "停歌", "关掉音乐",
                "停止", "停车", "停下来", "停下"
            )
        )

    def _volume_delta(self, normalized):
        if any(word in normalized for word in ("静音", "没声音")):
            return -self.volume_percent
        if any(word in normalized for word in ("最大音量", "声音最大")):
            return 100 - self.volume_percent
        if any(word in normalized for word in ("音量大一点", "声音大一点", "大声一点", "调大音量", "声音调大", "音量调大")):
            return self.volume_step
        if any(word in normalized for word in ("音量小一点", "声音小一点", "小声一点", "调小音量", "声音调小", "音量调小")):
            return -self.volume_step
        match = re.search(r"(?:音量|声音)(?:调到|设置到|设为)?(\d{1,3})", normalized)
        if match:
            target = max(0, min(100, int(match.group(1))))
            return target - self.volume_percent
        return None

    def _play_item(self, item):
        self._stop_current()
        if not os.path.exists(item["path"]):
            self.reply_pub.publish(String("这个音乐文件不见了。"))
            return

        rospy.loginfo("Playing music: %s", item["path"])
        remote_path = "/tmp/robot_music_current%s" % item["ext"]
        play_cmd = self._remote_play_command(remote_path, item["ext"])
        remote_cmd = "cat > %s && %s" % (shlex.quote(remote_path), play_cmd)
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

        try:
            with open(item["path"], "rb") as handle:
                audio = handle.read()
        except OSError as exc:
            rospy.logerr("Failed to read music file: %s", exc)
            self.reply_pub.publish(String("我读不到这个音乐文件。"))
            return

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        with self._lock:
            self._current_process = process

        try:
            stdout, stderr = process.communicate(input=audio)
        finally:
            with self._lock:
                was_stopped = id(process) in self._stopped_processes
                self._stopped_processes.discard(id(process))
                if self._current_process is process:
                    self._current_process = None

        if was_stopped:
            rospy.loginfo("Music playback stopped.")
            return

        if process.returncode not in (0, -15, 143):
            rospy.logerr(
                "Music playback failed: %s",
                stderr.decode("utf-8", "replace").strip(),
            )
        elif stderr.strip():
            rospy.logwarn(
                "Music playback stderr: %s",
                stderr.decode("utf-8", "replace").strip(),
            )
        if stdout.strip():
            rospy.logdebug(
                "Music playback stdout: %s",
                stdout.decode("utf-8", "replace").strip(),
            )

    def _remote_play_command(self, remote_path, ext):
        device = shlex.quote(self.audio_device) if self.audio_device else ""
        quoted_path = shlex.quote(remote_path)
        if ext == ".wav":
            if self.audio_device == "pulse":
                return "paplay %s" % quoted_path
            if device:
                return "aplay -q -D %s %s" % (device, quoted_path)
            return "aplay -q %s" % quoted_path

        if self.audio_device == "pulse":
            return "mpg123 -q -o pulse -f %d %s" % (self._mpg123_scale(), quoted_path)
        if device:
            return "mpg123 -q -o alsa -f %d -a %s %s" % (self._mpg123_scale(), device, quoted_path)
        return "mpg123 -q -o alsa -f %d %s" % (self._mpg123_scale(), quoted_path)

    def _mpg123_scale(self):
        return int(32768 * max(0, min(100, self.volume_percent)) / 100.0)

    def _set_remote_volume(self, percent):
        audio_device = shlex.quote(self.audio_device)
        cmd = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=%d" % self.ssh_connect_timeout,
            self._target,
            "bash",
            "-lc",
            """
AUDIO_DEVICE=%s
PERCENT=%d
CARD_ARG=""
case "$AUDIO_DEVICE" in
    plughw:*,*|hw:*,*)
        CARD="${AUDIO_DEVICE#*:}"
        CARD="${CARD%%,*}"
        CARD_ARG="-D hw:$CARD"
        ;;
esac
if command -v amixer >/dev/null 2>&1; then
    for CONTROL in Master PCM Speaker Headphone Playback; do
        amixer $CARD_ARG -q sset "$CONTROL" "$PERCENT%%" unmute >/dev/null 2>&1 && exit 0
    done
fi
exit 0
""" % (audio_device, percent),
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5.0)
        except (OSError, subprocess.TimeoutExpired) as exc:
            rospy.logwarn("Failed to set remote volume: %s", exc)

    def _is_playing(self):
        with self._lock:
            return self._current_process is not None and self._current_process.poll() is None

    def _stop_current(self):
        with self._lock:
            self._play_request_id += 1
            process = self._current_process
            if process is not None and process.poll() is None:
                rospy.loginfo("Stopping current music playback.")
                self._stopped_processes.add(id(process))

        self._stop_remote_players()

        if process is not None and process.poll() is None:
            process.terminate()
            timer = threading.Timer(0.8, self._kill_process_if_needed, args=(process,))
            timer.daemon = True
            timer.start()

    def _kill_process_if_needed(self, process):
        if process.poll() is None:
            process.kill()

    def _stop_remote_players(self):
        cmd = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=%d" % self.ssh_connect_timeout,
            self._target,
            "bash",
            "-lc",
            "pkill -INT -x mpg123 2>/dev/null || true; "
            "pkill -TERM -x aplay 2>/dev/null || true; "
            "sleep 0.15; "
            "pkill -KILL -x mpg123 2>/dev/null || true; "
            "pkill -KILL -x aplay 2>/dev/null || true",
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3.0)
        except (OSError, subprocess.TimeoutExpired) as exc:
            rospy.logwarn("Failed to stop remote music playback: %s", exc)


if __name__ == "__main__":
    rospy.init_node("music_player_node")
    MusicPlayerNode()
    rospy.spin()
