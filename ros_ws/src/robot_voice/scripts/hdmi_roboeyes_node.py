#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import random
import time

import rospy
import Tkinter as tk
from std_msgs.msg import String


EMOTION_TO_MOOD = {
    'happy': 'happy',
    'excited': 'happy',
    'neutral': 'neutral',
    'unknown': 'neutral',
    'angry': 'angry',
    'sad': 'tired',
    'anxious': 'tired',
}


class HdmiRoboEyesNode(object):
    def __init__(self):
        self.plan_topic = rospy.get_param('~plan_topic', 'ai_plan')
        self.reply_topic = rospy.get_param('~reply_topic', 'ai_reply')
        self.fullscreen = bool(rospy.get_param('~fullscreen', True))
        self.show_debug_text = bool(rospy.get_param('~show_debug_text', False))

        self.current_mood = 'neutral'
        self.last_emotion = 'unknown'
        self.last_update_at = time.time()

        self.root = tk.Tk()
        if self.fullscreen:
            self.root.attributes('-fullscreen', True)
        self.root.configure(bg='black')
        self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.w = self.root.winfo_screenwidth()
        self.h = self.root.winfo_screenheight()

        self.eye_w = int(self.w * 0.22)
        self.eye_h = int(self.h * 0.28)
        self.gap = int(self.w * 0.09)

        self.blink = 1.0
        self.blink_dir = 0
        self.next_blink = time.time() + random.uniform(1.2, 2.8)

        self.gx = 0.0
        self.gy = 0.0
        self.tx = 0.0
        self.ty = 0.0
        self.next_gaze = time.time() + random.uniform(0.7, 1.6)

        self.root.bind('<Escape>', self._on_exit)
        self.root.bind('q', self._on_exit)

        rospy.Subscriber(self.plan_topic, String, self._on_plan, queue_size=10)
        rospy.Subscriber(self.reply_topic, String, self._on_reply, queue_size=10)

        rospy.loginfo('HDMI RoboEyes node started: plan_topic=%s reply_topic=%s', self.plan_topic, self.reply_topic)

        self._tick()

    def _on_exit(self, _event):
        try:
            self.root.destroy()
        except Exception:
            pass

    def _normalize_emotion(self, emotion):
        value = str(emotion or '').strip().lower()
        mapping = {
            'happy': 'happy', 'happiness': 'happy', '开心': 'happy', '高兴': 'happy', '兴奋': 'excited',
            'excited': 'excited', 'surprise': 'excited', '惊讶': 'excited',
            'neutral': 'neutral', 'calm': 'neutral', '平静': 'neutral', '中性': 'neutral',
            'angry': 'angry', 'anger': 'angry', '生气': 'angry', '愤怒': 'angry',
            'sad': 'sad', 'sadness': 'sad', '难过': 'sad', '伤心': 'sad',
            'anxious': 'anxious', 'fear': 'anxious', '焦虑': 'anxious', '紧张': 'anxious',
            'unknown': 'unknown'
        }
        return mapping.get(value, value if value in EMOTION_TO_MOOD else 'unknown')

    def _apply_emotion(self, emotion):
        normalized = self._normalize_emotion(emotion)
        self.last_emotion = normalized
        self.current_mood = EMOTION_TO_MOOD.get(normalized, 'neutral')
        self.last_update_at = time.time()

    def _on_plan(self, msg):
        try:
            plan = json.loads(msg.data)
        except ValueError:
            return
        if not isinstance(plan, dict):
            return

        vision_emotion = plan.get('vision_emotion')
        if isinstance(vision_emotion, dict) and vision_emotion.get('emotion'):
            self._apply_emotion(vision_emotion.get('emotion'))
            return

        if plan.get('emotion'):
            self._apply_emotion(plan.get('emotion'))

    def _on_reply(self, msg):
        text = (msg.data or '').strip()
        if not text:
            return
        if any(k in text for k in ('哈哈', '高兴', '开心')):
            self._apply_emotion('happy')
        elif any(k in text for k in ('别急', '慢慢', '紧张')):
            self._apply_emotion('anxious')

    def _tick(self):
        if rospy.is_shutdown():
            try:
                self.root.destroy()
            except Exception:
                pass
            return

        now = time.time()

        if now >= self.next_gaze:
            self.tx = random.uniform(-1.0, 1.0)
            self.ty = random.uniform(-0.8, 0.8)
            self.next_gaze = now + random.uniform(0.7, 1.8)

        self.gx = self.gx * 0.85 + self.tx * 0.15
        self.gy = self.gy * 0.85 + self.ty * 0.15

        if self.blink_dir == 0 and now >= self.next_blink:
            self.blink_dir = -1
        if self.blink_dir == -1:
            self.blink -= 0.22
            if self.blink <= 0.08:
                self.blink = 0.08
                self.blink_dir = 1
        elif self.blink_dir == 1:
            self.blink += 0.22
            if self.blink >= 1.0:
                self.blink = 1.0
                self.blink_dir = 0
                self.next_blink = now + random.uniform(1.5, 3.5)

        self._draw()
        self.root.after(33, self._tick)

    def _draw_eye(self, x, y, w, h, mood):
        self.canvas.create_rectangle(x, y, x + w, y + h, fill='white', outline='white')

        if mood == 'angry':
            cut = int(h * 0.42)
            self.canvas.create_polygon(x, y, x + w, y, x + w, y + cut, fill='black', outline='black')
        elif mood == 'tired':
            cut = int(h * 0.26)
            self.canvas.create_polygon(x, y, x + w, y, x, y + cut, fill='black', outline='black')
            self.canvas.create_line(x, y + int(h * 0.65), x + w, y + int(h * 0.65), fill='black', width=3)
        elif mood == 'happy':
            cut = int(h * 0.34)
            self.canvas.create_rectangle(x - 2, y + h - cut, x + w + 2, y + h + 2, fill='black', outline='black')

    def _draw(self):
        self.canvas.delete('all')

        draw_h = max(int(self.eye_h * 0.28), int(self.eye_h * self.blink))
        cx = self.w // 2
        cy = self.h // 2
        ox = int(self.gx * (self.w * 0.03))
        oy = int(self.gy * (self.h * 0.03))

        lx = cx - self.gap // 2 - self.eye_w + ox
        rx = cx + self.gap // 2 + ox
        y = cy - draw_h // 2 + oy

        self._draw_eye(lx, y, self.eye_w, draw_h, self.current_mood)
        self._draw_eye(rx, y, self.eye_w, draw_h, self.current_mood)

        if self.show_debug_text:
            text = 'emotion: %s mood: %s' % (self.last_emotion, self.current_mood)
            self.canvas.create_text(20, 20, anchor='nw', fill='white', text=text, font=('Helvetica', 18))


def main():
    rospy.init_node('hdmi_roboeyes_node')
    node = HdmiRoboEyesNode()
    node.root.mainloop()


if __name__ == '__main__':
    main()
