#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import threading
import time

import rospy
from std_msgs.msg import String


EMOTION_TO_MOOD = {
    "happy": "HAPPY",
    "excited": "HAPPY",
    "angry": "ANGRY",
    "sad": "TIRED",
    "anxious": "TIRED",
    "neutral": "DEFAULT",
    "unknown": "DEFAULT",
}


class RoboEyesBridge:
    def __init__(self):
        self.plan_topic = rospy.get_param("~plan_topic", "ai_plan")
        self.reply_topic = rospy.get_param("~reply_topic", "ai_reply")
        self.serial_port = rospy.get_param("~serial_port", "/dev/ttyUSB0")
        self.reconnect_interval = float(rospy.get_param("~reconnect_interval", 2.0))
        self.min_send_interval = float(rospy.get_param("~min_send_interval", 0.12))

        self._fd = None
        self._lock = threading.Lock()
        self._last_send_time = 0.0
        self._last_connect_try_time = 0.0
        self._last_mood = None
        self._last_sweat = None

        rospy.Subscriber(self.plan_topic, String, self._on_plan, queue_size=10)
        rospy.Subscriber(self.reply_topic, String, self._on_reply, queue_size=10)

        rospy.loginfo(
            "RoboEyes bridge ready: plan=%s reply=%s serial=%s",
            self.plan_topic,
            self.reply_topic,
            self.serial_port,
        )

    def _ensure_serial(self):
        with self._lock:
            if self._fd is not None:
                return True

            now = time.time()
            if now - self._last_connect_try_time < self.reconnect_interval:
                return False
            self._last_connect_try_time = now

            try:
                self._fd = os.open(self.serial_port, os.O_WRONLY | os.O_NOCTTY)
                rospy.loginfo("RoboEyes serial opened: %s", self.serial_port)
                return True
            except OSError as exc:
                rospy.logwarn_throttle(5.0, "RoboEyes serial open failed: %s", exc)
                self._fd = None
                return False

    def _close_serial(self):
        with self._lock:
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
                self._fd = None

    def _send_line(self, line):
        if not line:
            return
        if not self._ensure_serial():
            return

        now = time.time()
        delta = now - self._last_send_time
        if delta < self.min_send_interval:
            time.sleep(self.min_send_interval - delta)

        payload = (line.strip() + "\n").encode("utf-8")
        try:
            with self._lock:
                if self._fd is None:
                    return
                os.write(self._fd, payload)
            self._last_send_time = time.time()
        except OSError as exc:
            rospy.logwarn("RoboEyes serial write failed: %s", exc)
            self._close_serial()

    def _resolve_emotion(self, plan):
        vision_emotion = plan.get("vision_emotion")
        if isinstance(vision_emotion, dict):
            value = vision_emotion.get("emotion")
            if value:
                return str(value).strip().lower()
        value = plan.get("emotion")
        if value:
            return str(value).strip().lower()
        return "unknown"

    def _on_plan(self, msg):
        try:
            plan = json.loads(msg.data)
        except ValueError:
            return
        if not isinstance(plan, dict):
            return

        emotion = self._resolve_emotion(plan)
        mood = EMOTION_TO_MOOD.get(emotion, "DEFAULT")
        sweat = "ON" if emotion == "anxious" else "OFF"

        if mood != self._last_mood:
            self._send_line("MOOD %s" % mood)
            self._last_mood = mood

        if sweat != self._last_sweat:
            self._send_line("SWEAT %s" % sweat)
            self._last_sweat = sweat

        intent = str(plan.get("intent", "")).strip().lower()
        if "dance" in intent:
            self._send_line("ANIM LAUGH")
        elif "unknown" in intent or "failed" in intent:
            self._send_line("ANIM CONFUSED")

        actions = plan.get("actions", [])
        if isinstance(actions, list) and actions:
            self._send_line("ANIM BLINK")

    def _on_reply(self, msg):
        text = msg.data.strip()
        if not text:
            return
        lowered = text.lower()
        if any(token in text for token in ("你好", "您好")) or any(token in lowered for token in ("hi", "hello")):
            self._send_line("ANIM BLINK")


def main():
    rospy.init_node("roboeyes_bridge")
    RoboEyesBridge()
    rospy.spin()


if __name__ == "__main__":
    main()
