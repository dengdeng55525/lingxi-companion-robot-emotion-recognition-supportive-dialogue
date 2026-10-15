#!/usr/bin/env python3
import hashlib
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque

import rospy
from std_msgs.msg import String


CHINESE_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

CHINESE_UNITS = {
    "十": 10,
    "百": 100,
    "千": 1000,
}

DANGEROUS_OBJECT_DISPLAY_NAMES = {
    "knife": "刀具",
    "scissors": "剪刀",
    "fork": "叉子",
    "baseball bat": "球棒",
    "hammer": "锤子",
    "cutter": "美工刀",
    "blade": "刀片",
    "saw": "锯子",
    "axe": "斧头",
    "gun": "枪形物品",
    "pistol": "枪形物品",
    "刀": "刀具",
    "刀具": "刀具",
    "菜刀": "菜刀",
    "剪刀": "剪刀",
    "锤子": "锤子",
}

DANGEROUS_OBJECT_KEYWORDS = (
    ("knife", "刀具"),
    ("scissors", "剪刀"),
    ("fork", "叉子"),
    ("baseball bat", "球棒"),
    ("cutter", "美工刀"),
    ("blade", "刀片"),
    ("hammer", "锤子"),
    ("saw", "锯子"),
    ("axe", "斧头"),
    ("刀", "刀具"),
    ("剪刀", "剪刀"),
    ("锤", "锤子"),
)


SYSTEM_PROMPT = """\
你是一个面向儿童的早教陪护机器人，也是小车的安全决策模块。
你会收到用户语音文字、可选视觉摘要、机器人状态。你的任务是理解用户意图和情绪，
并输出严格 JSON。不要输出 Markdown，不要输出解释文本。

可用动作白名单：
- forward: 前进，字段 meters，范围 0.0 到 5.0
- backward: 后退，字段 meters，范围 0.0 到 5.0
- turn: 原地转向，字段 degrees，正数表示左转/逆时针，负数表示右转/顺时针，范围 -180 到 180
- stop: 停止
- wait: 等待，字段 seconds，范围 0.0 到 3.0

规则：
1. 如果用户明确说“前进三米”“左转三十度”等，按用户指定输出动作。
2. 如果用户只说“前进”，输出 forward 0.5 米；只说“左转”，输出 turn 90 度。
3. 如果用户说“跳个舞”，输出短小安全舞蹈序列：左右转、小前进、小后退，单步不要超过 0.4 米或 90 度。
4. 如果用户情绪低落，可以 reply 安慰，但动作要轻微或不动。
5. 如果视觉摘要显示前方不安全或 obstacle_clear 为 false，不要前进，只能 stop 或小角度转向。
6. 不要输出任何不在白名单内的动作。
7. 如果用户只是聊天、问候、表达情绪或提出不需要移动的问题，reply 正常回答，actions 输出空数组 []。
8. 如果上下文 memory 中有用户或机器人的长期记忆，回答时优先使用这些记忆。
9. 如果用户询问眼前是什么、看到了什么物品，或表达“这是什么/我想知道这个”等求知欲，并且视觉摘要里有 objects 列表，根据 objects 中 class 和 confidence 简短回答；不要说只能识别人脸情绪。此类问题 intent 必须输出 "object_learning_query"。无法确定时说明不确定。
10. 早教/学习类问题中，如果用户的说法有明显事实错误，要先温和纠正，再回答。不要迎合错误前提。
    例：用户说“杜甫的静夜思”，reply 必须先说明“《静夜思》是李白写的，不是杜甫”，然后再背诵或继续回答。
11. reply 面向小孩说话，像耐心的早教老师：短句、温和、鼓励、清楚。不要用冷冰冰的工程口吻；纠错时先保护孩子积极性。
12. 多步动作要严格保持用户给出的先后顺序。单个动作参数不能超过白名单范围；如果用户给的距离或角度超过单步范围，优先拆成多个安全动作，不要丢掉剩余意图。
    例：前进10米可拆为 forward 5米 + forward 5米；左转300度可拆为 turn 180度 + turn 120度，然后再执行后续动作。

JSON 格式：
{
  "reply": "给用户的简短回复",
  "emotion": "happy|sad|angry|anxious|neutral|excited|unknown",
  "intent": "简短意图",
  "safety": {"can_move": true, "reason": "简短原因"},
  "actions": [
    {"type": "forward", "meters": 0.5},
    {"type": "turn", "degrees": 90},
    {"type": "stop"}
  ]
}
"""


class AiCompanionNode:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "iat_text")
        self.motion_topic = rospy.get_param("~motion_topic", "ai_motion_text")
        self.plan_topic = rospy.get_param("~plan_topic", "ai_plan")
        self.reply_topic = rospy.get_param("~reply_topic", "ai_reply")
        self.vision_topic = rospy.get_param("~vision_topic", "vision_context")
        self.global_stop_topic = rospy.get_param("~global_stop_topic", "robot_stop_all")

        self.model = rospy.get_param(
            "~model",
            os.environ.get("DEEPSEEK_MODEL", os.environ.get("OPENAI_MODEL", "deepseek-v4-flash")),
        )
        self.base_url = rospy.get_param(
            "~base_url",
            os.environ.get("DEEPSEEK_BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")),
        ).rstrip("/")
        default_api_key_env = "DEEPSEEK_API_KEY" if os.environ.get("DEEPSEEK_API_KEY") else "OPENAI_API_KEY"
        self.api_key_env = rospy.get_param("~api_key_env", default_api_key_env)
        self.request_timeout = float(rospy.get_param("~request_timeout", 30.0))
        self.api_enabled = bool(rospy.get_param("~api_enabled", True))
        self.api_mode = rospy.get_param("~api_mode", os.environ.get("AI_API_MODE", "auto")).strip().lower()
        self.wake_word_required = bool(rospy.get_param("~wake_word_required", True))
        self.wake_session_enabled = bool(rospy.get_param("~wake_session_enabled", True))
        wake_words_param = rospy.get_param("~wake_words", "灵犀")
        if isinstance(wake_words_param, str):
            self.wake_words = [word.strip() for word in wake_words_param.split(",") if word.strip()]
        else:
            self.wake_words = [str(word).strip() for word in wake_words_param if str(word).strip()]
        self._conversation_active = not self.wake_word_required

        self.linear_speed = float(rospy.get_param("~linear_speed", 0.15))
        self.angular_speed = float(rospy.get_param("~angular_speed", 0.45))
        self.default_motion_wait = float(rospy.get_param("~default_motion_wait", 3.0))
        self.action_gap = float(rospy.get_param("~action_gap", 0.5))
        self.max_meters = float(rospy.get_param("~max_meters", 5.0))
        self.max_degrees = float(rospy.get_param("~max_degrees", 180.0))
        self.vision_fresh_seconds = float(rospy.get_param("~vision_fresh_seconds", 3.0))
        self.emotion_confidence_threshold = float(
            rospy.get_param("~emotion_confidence_threshold", 0.65)
        )
        self.emotion_file = os.path.expanduser(
            rospy.get_param("~emotion_file", "~/.ros/latest_emotion.json")
        )
        self.emotion_file_poll_seconds = float(
            rospy.get_param("~emotion_file_poll_seconds", 0.5)
        )
        self.startup_emotion_greeting_enabled = bool(
            rospy.get_param("~startup_emotion_greeting_enabled", True)
        )
        self.startup_emotion_greeting_min_delay = float(
            rospy.get_param("~startup_emotion_greeting_min_delay", 2.0)
        )
        self.startup_emotion_greeting_timeout = float(
            rospy.get_param("~startup_emotion_greeting_timeout", 20.0)
        )
        self.startup_emotion_greeting_min_confidence = float(
            rospy.get_param(
                "~startup_emotion_greeting_min_confidence",
                self.emotion_confidence_threshold,
            )
        )
        self.startup_emotion_greeting_min_face_count = int(
            rospy.get_param("~startup_emotion_greeting_min_face_count", 1)
        )
        self.startup_face_search_enabled = bool(
            rospy.get_param("~startup_face_search_enabled", True)
        )
        self.startup_face_search_min_delay = float(
            rospy.get_param("~startup_face_search_min_delay", 2.0)
        )
        self.startup_face_search_turn_degrees = float(
            rospy.get_param("~startup_face_search_turn_degrees", 45.0)
        )
        self.startup_face_search_after_turn_wait = float(
            rospy.get_param("~startup_face_search_after_turn_wait", 2.0)
        )
        self.startup_face_search_max_turns = int(
            rospy.get_param("~startup_face_search_max_turns", 8)
        )
        self.proactive_face_search_enabled = bool(
            rospy.get_param("~proactive_face_search_enabled", True)
        )
        self.proactive_face_search_no_face_seconds = float(
            rospy.get_param("~proactive_face_search_no_face_seconds", 10.0)
        )
        self.proactive_face_search_turn_degrees = float(
            rospy.get_param("~proactive_face_search_turn_degrees", 45.0)
        )
        self.proactive_face_search_step_wait = float(
            rospy.get_param("~proactive_face_search_step_wait", 0.2)
        )
        self.plan_file = os.path.expanduser(
            rospy.get_param("~plan_file", "~/.ros/ai_companion_latest_plan.json")
        )
        self.learning_event_file = os.path.expanduser(
            rospy.get_param("~learning_event_file", "~/.ros/ai_learning_events.jsonl")
        )
        self.dangerous_object_alert_enabled = bool(
            rospy.get_param("~dangerous_object_alert_enabled", True)
        )
        self.dangerous_object_alert_interval = float(
            rospy.get_param("~dangerous_object_alert_interval", 20.0)
        )
        self.dangerous_object_min_confidence = float(
            rospy.get_param("~dangerous_object_min_confidence", 0.30)
        )
        self.dance_profile_file = os.path.expanduser(
            rospy.get_param("~dance_profile_file", "~/.ros/ai_dance_profile.json")
        )
        self.memory_file = os.path.expanduser(
            rospy.get_param("~memory_file", "~/.ros/ai_companion_memory.json")
        )
        self.reply_memory_limit = int(rospy.get_param("~reply_memory_limit", 3))
        self.dialogue_memory_limit = int(rospy.get_param("~dialogue_memory_limit", 20))

        self._latest_vision = {}
        self._latest_vision_time = 0.0
        self._dangerous_object_last_alert_at = {}
        self._startup_time = time.time()
        self._startup_greeting_done = False
        self._startup_face_search_started = False
        self._startup_face_search_done = False
        self._startup_face_search_turns = 0
        self._startup_first_face_greet_done = False
        self._last_face_seen_time = time.time()
        self._proactive_face_search_running = False
        self._manual_face_search_running = False
        self._user_has_spoken = False
        self._worker = None
        self._lock = threading.Lock()
        self._startup_greeting_lock = threading.Lock()
        self._startup_face_search_lock = threading.Lock()
        self._proactive_face_search_lock = threading.Lock()
        self._manual_face_search_lock = threading.Lock()
        self._dangerous_object_alert_lock = threading.Lock()
        self._reply_memory = deque(maxlen=max(1, self.reply_memory_limit))
        self._dialogue_history = deque(maxlen=max(10, self.dialogue_memory_limit))
        self._recent_user_texts = deque(maxlen=max(10, self.dialogue_memory_limit))
        self._stop_requested = threading.Event()

        self.motion_pub = rospy.Publisher(self.motion_topic, String, queue_size=10)
        self.plan_pub = rospy.Publisher(self.plan_topic, String, queue_size=10)
        self.reply_pub = rospy.Publisher(self.reply_topic, String, queue_size=10)
        self.global_stop_pub = rospy.Publisher(self.global_stop_topic, String, queue_size=10)

        rospy.Subscriber(self.input_topic, String, self._on_user_text, queue_size=10)
        rospy.Subscriber(self.vision_topic, String, self._on_vision, queue_size=1)

        self._emotion_file_thread = threading.Thread(
            target=self._emotion_file_loop,
            daemon=True,
        )
        self._emotion_file_thread.start()

        self._proactive_face_search_thread = threading.Thread(
            target=self._proactive_face_search_loop,
            daemon=True,
        )
        self._proactive_face_search_thread.start()

        rospy.loginfo(
            "AI companion ready: input=%s motion=%s model=%s base_url=%s api_key_env=%s",
            self.input_topic,
            self.motion_topic,
            self.model,
            self.base_url,
            self.api_key_env,
        )
        if self._is_deepseek_base_url():
            rospy.loginfo("DeepSeek API direct HTTPS mode enabled; environment proxy will be bypassed for model requests.")

    def _on_vision(self, msg):
        try:
            vision = json.loads(msg.data)
        except ValueError:
            vision = {"raw": msg.data}
        self._update_latest_vision(vision, source="topic")

    def _emotion_file_loop(self):
        last_mtime = 0.0
        while not rospy.is_shutdown():
            if self.emotion_file:
                try:
                    mtime = os.path.getmtime(self.emotion_file)
                    if mtime > last_mtime:
                        with open(self.emotion_file, "r", encoding="utf-8") as handle:
                            vision = json.load(handle)
                        last_mtime = mtime
                        self._update_latest_vision(vision, source="file")
                except FileNotFoundError:
                    pass
                except (OSError, ValueError) as exc:
                    rospy.logwarn_throttle(5.0, "Failed to read emotion file: %s", exc)
            time.sleep(max(0.1, self.emotion_file_poll_seconds))

    def _update_latest_vision(self, vision, source):
        if not isinstance(vision, dict):
            vision = {"raw": vision}
        self._latest_vision = vision
        self._latest_vision_time = time.time()
        face_count = self._vision_face_count()
        if face_count >= self.startup_emotion_greeting_min_face_count:
            self._last_face_seen_time = time.time()
        cue = self._extract_emotion_cue(self._latest_vision)
        if cue:
            rospy.loginfo(
                "Vision emotion updated from %s: %s confidence %.2f",
                source,
                cue["emotion"],
                cue["confidence"],
            )
        self._maybe_startup_face_search()
        self._maybe_alert_dangerous_objects()
        self._maybe_startup_emotion_greeting()

    def _on_user_text(self, msg):
        raw_text = msg.data.strip()
        if not raw_text:
            return

        text = self._prepare_session_text(raw_text)
        if text is None:
            return

        if not text:
            text = "你好"
        self._user_has_spoken = True
        self._append_user_dialogue(text)
        if self._is_stop_text(text):
            rospy.loginfo("Immediate stop requested.")
            self._stop_requested.set()
            with self._proactive_face_search_lock:
                self._proactive_face_search_running = False
            with self._manual_face_search_lock:
                self._manual_face_search_running = False
            self.motion_pub.publish(String("停止"))
            self.global_stop_pub.publish(String("STOP_ALL"))
            record = {
                "timestamp": time.time(),
                "source_text": text,
                "vision": self._latest_vision,
                "status": "done",
                "current_action_index": 1,
                "plan": {
                    "reply": "已停止。",
                    "emotion": "neutral",
                    "intent": "stop",
                    "safety": {"can_move": False, "reason": "user requested stop"},
                    "actions": [{"type": "stop"}],
                },
            }
            self.plan_pub.publish(String(json.dumps(record["plan"], ensure_ascii=False)))
            self._publish_reply("已停止。")
            self._write_plan_file(record)
            return

        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                rospy.logwarn("AI companion is busy; ignoring: %s", text)
                return
            self._worker = threading.Thread(target=self._handle_text, args=(text,), daemon=True)
            self._worker.start()

    def _prepare_session_text(self, text):
        if not self.wake_word_required:
            return text.strip()

        stripped = self._strip_wake_word(text)
        if not self.wake_session_enabled:
            if stripped is None:
                rospy.loginfo("Ignoring speech without wake word: %s", text)
            return stripped

        if self._conversation_active and self._is_sleep_command(text):
            self._conversation_active = False
            rospy.loginfo("Conversation session stopped by wake-word goodbye.")
            self._publish_reply("好的，我先安静了。需要我的时候再叫我灵犀。")
            return None

        if self._conversation_active:
            if stripped is not None:
                return stripped
            return text.strip()

        if stripped is None:
            rospy.loginfo("Ignoring speech before wake session starts: %s", text)
            return None

        if self._is_sleep_text(stripped):
            rospy.loginfo("Ignoring goodbye while conversation session is inactive: %s", text)
            return None

        self._conversation_active = True
        rospy.loginfo("Conversation session started by wake word.")
        return stripped

    def _is_sleep_command(self, text):
        stripped = self._strip_wake_word(text)
        return stripped is not None and self._is_sleep_text(stripped)

    def _is_sleep_text(self, text):
        normalized = self._normalize_text(text)
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

    def _is_stop_text(self, text):
        normalized = self._normalize_text(text)
        if any(word in normalized for word in ("停止播放", "停播放", "停止音乐", "停音乐", "停歌", "关音乐", "关闭音乐")):
            return False
        return "停止" in normalized or "停车" in normalized or "停下来" in normalized or "停下" in normalized

    def _strip_wake_word(self, text):
        if not self.wake_word_required:
            return text.strip()

        normalized = self._normalize_text(text)
        for wake_word in self.wake_words:
            wake = self._normalize_text(wake_word)
            if not wake:
                continue
            if normalized == wake:
                return ""
            if normalized.startswith(wake):
                remainder = normalized[len(wake):]
                return remainder.strip()
        return None

    def _handle_text(self, text):
        self._stop_requested.clear()
        rospy.loginfo("AI input: %s", text)
        local_plan = self._local_rule_plan(text)
        used_api = False
        if local_plan.get("local_priority"):
            plan = local_plan
        elif self.api_enabled:
            try:
                plan = self._call_model(text)
                used_api = True
            except Exception as exc:
                rospy.logerr("AI API failed: %s; falling back to local rule.", exc)
                plan = local_plan
        else:
            plan = local_plan

        plan = self._sanitize_plan(plan)
        plan = self._apply_educational_correction_to_plan(plan, text)
        plan = self._apply_vision_emotion_to_plan(plan)
        if (
            not plan.get("actions")
            and not plan.get("profile_saved")
            and not plan.get("memory_saved")
            and not plan.get("handled_elsewhere")
        ):
            if used_api and plan.get("reply"):
                rospy.loginfo("AI plan has reply only; treating it as conversation.")
            else:
                rospy.logwarn("AI plan has no executable actions; falling back to local rule.")
                plan = self._sanitize_plan(local_plan)
                plan = self._apply_educational_correction_to_plan(plan, text)
                plan = self._apply_vision_emotion_to_plan(plan)

        plan_json = json.dumps(plan, ensure_ascii=False)
        rospy.loginfo("AI plan: %s", plan_json)
        self.plan_pub.publish(String(plan_json))
        if plan.get("reply"):
            self._publish_reply(plan["reply"])

        record = {
            "timestamp": time.time(),
            "source_text": text,
            "vision": self._latest_vision,
            "memory": self._load_memory(),
            "status": "planned",
            "current_action_index": -1,
            "plan": plan,
        }
        self._write_plan_file(record)
        self._write_learning_event_if_needed(record)
        if plan.get("intent") == "manual_face_search":
            self._start_manual_face_search(record)
            return
        self._publish_actions(plan.get("actions", []), record)

    def _call_model(self, text):
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError("environment variable %s is not set" % self.api_key_env)

        context = {
            "user_text": text,
            "vision": self._latest_vision,
            "vision_age_seconds": round(max(0.0, time.time() - self._latest_vision_time), 2)
            if self._latest_vision_time else None,
            "vision_emotion_cue": self._current_emotion_cue(),
            "recent_replies": list(self._reply_memory),
            "recent_user_texts": list(self._recent_user_texts),
            "recent_dialogue": list(self._dialogue_history),
            "memory": self._load_memory(),
            "robot_state": {
                "motion_interface": "text_command_to_cmd_vel",
                "max_meters": self.max_meters,
                "max_degrees": self.max_degrees,
            },
        }
        payload, url = self._build_model_request(context)
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": "Bearer " + api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with self._open_api_request(req) as resp:
                response = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError("HTTP %s %s" % (exc.code, detail[:500]))

        text_out = self._extract_response_text(response)
        return self._parse_model_json(text_out)

    def _build_model_request(self, context):
        mode = self.api_mode
        if mode == "auto":
            mode = "chat_completions" if "deepseek.com" in self.base_url.lower() else "responses"

        user_content = json.dumps(context, ensure_ascii=False)
        if mode in ("chat", "chat_completions", "chat-completions"):
            return (
                {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "response_format": {"type": "json_object"},
                },
                self._chat_completions_url(),
            )

        return (
            {
                "model": self.model,
                "input": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            },
            self.base_url + "/v1/responses",
        )

    def _chat_completions_url(self):
        if self._is_deepseek_base_url():
            return self.base_url + "/chat/completions"
        return self.base_url + "/v1/chat/completions"

    def _is_deepseek_base_url(self):
        host = urllib.parse.urlparse(self.base_url).hostname or ""
        return host == "api.deepseek.com" or host.endswith(".deepseek.com")

    def _open_api_request(self, request):
        if self._is_deepseek_base_url():
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            return opener.open(request, timeout=self.request_timeout)
        return urllib.request.urlopen(request, timeout=self.request_timeout)

    def _extract_response_text(self, response):
        if isinstance(response.get("output_text"), str):
            return response["output_text"]

        chunks = []
        for item in response.get("output", []):
            for content in item.get("content", []):
                if isinstance(content.get("text"), str):
                    chunks.append(content["text"])
        if chunks:
            return "\n".join(chunks)

        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            if isinstance(message.get("content"), str):
                return message["content"]

        raise RuntimeError("cannot find text in API response")

    def _parse_model_json(self, text):
        cleaned = text.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
        if fenced:
            cleaned = fenced.group(1).strip()

        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first >= 0 and last > first:
            cleaned = cleaned[first:last + 1]

        return json.loads(cleaned)

    def _apply_educational_correction_to_plan(self, plan, user_text):
        normalized = self._normalize_text(user_text)
        if "静夜思" not in normalized or "杜甫" not in normalized:
            return plan

        reply = str(plan.get("reply", "") or "").strip()
        if "李白" in reply and ("不是杜甫" in reply or "杜甫" in reply):
            return plan

        correction = "《静夜思》是李白写的，不是杜甫。"
        if reply:
            plan["reply"] = self._limit_reply(correction + reply)
        else:
            plan["reply"] = correction + "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
        plan["intent"] = plan.get("intent") or "correct_and_recite_poem"
        plan["actions"] = []
        return plan

    def _apply_vision_emotion_to_plan(self, plan):
        cue = self._current_emotion_cue()
        if not cue or plan.get("handled_elsewhere"):
            return plan

        emotion = cue["emotion"]
        if emotion in ("neutral", "unknown"):
            return plan

        reply = plan.get("reply", "").strip()
        prefix = self._emotion_reply_prefix(emotion)
        if not prefix:
            return plan

        plan["vision_emotion"] = {
            "emotion": emotion,
            "confidence": cue["confidence"],
            "age_seconds": cue["age_seconds"],
        }

        if not reply:
            plan["reply"] = prefix
            return plan

        if prefix not in reply:
            plan["reply"] = self._limit_reply("%s%s" % (prefix, reply))
        return plan

    def _current_emotion_cue(self):
        if not self._latest_vision or not self._latest_vision_time:
            return None
        age = time.time() - self._latest_vision_time
        if age > self.vision_fresh_seconds:
            return None

        cue = self._extract_emotion_cue(self._latest_vision)
        if not cue:
            return None
        if cue["confidence"] < self.emotion_confidence_threshold:
            return None

        cue["age_seconds"] = round(age, 2)
        return cue

    def _extract_emotion_cue(self, vision):
        if not isinstance(vision, dict):
            return None

        raw_emotion = (
            vision.get("dominant_emotion")
            or vision.get("emotion")
            or vision.get("expression")
            or ""
        )
        emotion = self._normalize_emotion(raw_emotion)

        confidence = vision.get("confidence")
        scores = vision.get("scores")
        if confidence is None and isinstance(scores, dict) and emotion:
            confidence = scores.get(emotion) or scores.get(raw_emotion)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        if not emotion:
            return None
        return {"emotion": emotion, "confidence": confidence}

    def _normalize_emotion(self, emotion):
        text = str(emotion).strip().lower()
        mapping = {
            "happy": "happy",
            "happiness": "happy",
            "开心": "happy",
            "高兴": "happy",
            "sad": "sad",
            "sadness": "sad",
            "难过": "sad",
            "伤心": "sad",
            "angry": "angry",
            "anger": "angry",
            "生气": "angry",
            "愤怒": "angry",
            "anxious": "anxious",
            "fear": "anxious",
            "scared": "anxious",
            "焦虑": "anxious",
            "害怕": "anxious",
            "surprise": "excited",
            "surprised": "excited",
            "excited": "excited",
            "惊讶": "excited",
            "兴奋": "excited",
            "neutral": "neutral",
            "calm": "neutral",
            "平静": "neutral",
            "中性": "neutral",
            "unknown": "unknown",
        }
        return mapping.get(text, text if text in {"happy", "sad", "angry", "anxious", "neutral", "excited", "unknown"} else "")

    def _emotion_reply_prefix(self, emotion):
        prefixes = {
            "sad": "我看到你现在好像有点低落，我会轻一点陪你说。 ",
            "angry": "我感觉你现在有些烦躁，我先慢慢来。 ",
            "anxious": "我感觉你可能有点紧张，没关系，我在这儿。 ",
            "happy": "看起来你心情不错呀。 ",
            "excited": "你看起来挺有精神的。 ",
        }
        return prefixes.get(emotion, "")

    def _limit_reply(self, reply):
        return reply[:120]

    def _maybe_startup_emotion_greeting(self):
        if not self.startup_emotion_greeting_enabled:
            return

        with self._startup_greeting_lock:
            if self._startup_greeting_done or self._user_has_spoken:
                return
            if self.startup_face_search_enabled and not self._startup_face_search_done:
                return

            now = time.time()
            elapsed = now - self._startup_time
            if elapsed < self.startup_emotion_greeting_min_delay:
                return
            if elapsed > self.startup_emotion_greeting_timeout:
                self._startup_greeting_done = True
                rospy.loginfo("Startup emotion greeting window expired.")
                return

            cue = self._extract_emotion_cue(self._latest_vision)
            if not cue:
                return

            emotion = cue["emotion"]
            if emotion in ("neutral", "unknown"):
                return
            if cue["confidence"] < self.startup_emotion_greeting_min_confidence:
                return
            if self._vision_face_count() < self.startup_emotion_greeting_min_face_count:
                return

            reply = self._startup_emotion_reply(emotion)
            if not reply:
                return

            self._startup_greeting_done = True

        plan = {
            "reply": reply,
            "emotion": emotion,
            "intent": "startup_emotion_greeting",
            "safety": {"can_move": False, "reason": "startup emotion greeting only"},
            "actions": [],
            "vision_emotion": {
                "emotion": emotion,
                "confidence": cue["confidence"],
                "age_seconds": round(max(0.0, time.time() - self._latest_vision_time), 2),
            },
        }
        record = {
            "timestamp": time.time(),
            "source_text": "",
            "trigger": "startup_emotion_greeting",
            "vision": self._latest_vision,
            "memory": self._load_memory(),
            "status": "done",
            "current_action_index": 0,
            "plan": plan,
        }
        rospy.loginfo("Startup emotion greeting: %s", reply)
        self.plan_pub.publish(String(json.dumps(plan, ensure_ascii=False)))
        self._publish_reply(reply)
        self._write_plan_file(record)

    def _maybe_startup_face_search(self):
        if not self.startup_face_search_enabled:
            return

        with self._startup_face_search_lock:
            if (
                self._startup_face_search_done
                or self._startup_face_search_started
                or self._user_has_spoken
            ):
                return

            elapsed = time.time() - self._startup_time
            if elapsed < self.startup_face_search_min_delay:
                return

            if self._vision_face_count() >= self.startup_emotion_greeting_min_face_count:
                self._startup_face_search_done = True
                return

            self._startup_face_search_started = True
            thread = threading.Thread(target=self._startup_face_search_loop, daemon=True)
            thread.start()

    def _startup_face_search_loop(self):
        rospy.loginfo("Startup face search started: no face detected.")
        while not rospy.is_shutdown():
            with self._startup_face_search_lock:
                if self._startup_face_search_done or self._user_has_spoken:
                    self._startup_face_search_done = True
                    return
                if self._vision_face_count() >= self.startup_emotion_greeting_min_face_count:
                    self._startup_face_search_done = True
                    self.motion_pub.publish(String("停止"))
                    rospy.loginfo("Startup face search finished: face detected, stop immediately.")
                    self._greet_on_first_face_detected()
                    return
                if (
                    self.startup_face_search_max_turns > 0
                    and self._startup_face_search_turns >= self.startup_face_search_max_turns
                ):
                    self._startup_face_search_done = True
                    rospy.loginfo(
                        "Startup face search stopped after %d turn(s); no face detected.",
                        self._startup_face_search_turns,
                    )
                    return

                self._startup_face_search_turns += 1
                turn_index = self._startup_face_search_turns

            degrees = abs(self.startup_face_search_turn_degrees)
            command_text = "右转%s度" % self._format_number(degrees)
            rospy.loginfo(
                "Startup face search: %s (%d).",
                command_text,
                turn_index,
            )
            self.motion_pub.publish(String(command_text))
            self._write_startup_face_search_plan(command_text, turn_index)

            wait_until = time.time() + self._angular_wait(degrees) + self.startup_face_search_after_turn_wait
            while not rospy.is_shutdown() and time.time() < wait_until:
                if self._user_has_spoken:
                    with self._startup_face_search_lock:
                        self._startup_face_search_done = True
                    return
                if self._vision_face_count() >= self.startup_emotion_greeting_min_face_count:
                    with self._startup_face_search_lock:
                        self._startup_face_search_done = True
                    self.motion_pub.publish(String("停止"))
                    rospy.loginfo("Startup face search finished: face detected, stop immediately.")
                    self._greet_on_first_face_detected()
                    return
                time.sleep(0.2)

    def _proactive_face_search_loop(self):
        while not rospy.is_shutdown():
            if not self.proactive_face_search_enabled:
                time.sleep(0.2)
                continue
            # Keep compatibility with existing thread model, but disable automatic runtime search behavior.
            time.sleep(0.5)

    def _start_manual_face_search(self, record):
        with self._manual_face_search_lock:
            if self._manual_face_search_running:
                self._publish_reply("我正在找你，稍等一下。")
                return
            self._manual_face_search_running = True

        thread = threading.Thread(
            target=self._manual_face_search_loop,
            args=(record,),
            daemon=True,
        )
        thread.start()

    def _manual_face_search_loop(self, record):
        max_turns = max(1, int(round(360.0 / max(1.0, abs(self.startup_face_search_turn_degrees)))))
        degrees = abs(self.startup_face_search_turn_degrees)

        try:
            for turn_index in range(1, max_turns + 1):
                if self._stop_requested.is_set() or rospy.is_shutdown():
                    self.motion_pub.publish(String("停止"))
                    return

                if self._vision_face_count() >= self.startup_emotion_greeting_min_face_count:
                    self.motion_pub.publish(String("停止"))
                    self._publish_reply(self._manual_found_face_reply())
                    return

                command_text = "右转%s度" % self._format_number(degrees)
                self.motion_pub.publish(String(command_text))

                step_end = time.time() + self._angular_wait(degrees) + self.startup_face_search_after_turn_wait
                while not rospy.is_shutdown() and time.time() < step_end:
                    if self._stop_requested.is_set():
                        self.motion_pub.publish(String("停止"))
                        return
                    if self._vision_face_count() >= self.startup_emotion_greeting_min_face_count:
                        self.motion_pub.publish(String("停止"))
                        self._publish_reply(self._manual_found_face_reply())
                        return
                    time.sleep(0.1)

            self.motion_pub.publish(String("停止"))
            self._publish_reply("我暂时还没看到你。")
        finally:
            with self._manual_face_search_lock:
                self._manual_face_search_running = False
            done_record = dict(record)
            done_record["status"] = "done"
            self._write_plan_file(done_record)

    def _manual_found_face_reply(self):
        cue = self._extract_emotion_cue(self._latest_vision)
        emotion = cue.get("emotion") if cue else "neutral"
        emotion_text = {
            "happy": "不错",
            "excited": "很棒",
            "sad": "有点低落",
            "angry": "有点烦躁",
            "anxious": "有点紧张",
            "neutral": "平静",
            "unknown": "平静",
        }.get(emotion, "平静")
        return "看到你了，看起来你心情%s。" % emotion_text

    def _write_startup_face_search_plan(self, command_text, turn_index):
        record = {
            "timestamp": time.time(),
            "source_text": "",
            "trigger": "startup_face_search",
            "vision": self._latest_vision,
            "memory": self._load_memory(),
            "status": "executing",
            "current_action_index": turn_index - 1,
            "current_motion_text": command_text,
            "plan": {
                "reply": "",
                "emotion": "unknown",
                "intent": "startup_face_search",
                "safety": {"can_move": True, "reason": "startup face search"},
                "actions": [{"type": "turn", "degrees": -abs(self.startup_face_search_turn_degrees)}],
            },
        }
        self._write_plan_file(record)

    def _vision_face_count(self):
        try:
            return int(self._latest_vision.get("face_count", 0))
        except (TypeError, ValueError):
            return 0

    def _startup_emotion_reply(self, emotion):
        name = self._load_memory().get("robot_name", "灵犀")
        replies = {
            "sad": "我看到你现在好像有点低落，我是%s。要不要我陪你说会儿话？" % name,
            "angry": "我感觉你现在有些烦躁，我是%s。我先安静陪着你，我们慢慢来。" % name,
            "anxious": "你看起来可能有点紧张，我是%s。别急，我在这儿陪你。" % name,
            "happy": "看起来你心情不错呀，我是%s，准备好陪你聊天啦。" % name,
            "excited": "你看起来挺有精神的，我是%s，今天也一起加油。" % name,
        }
        return replies.get(emotion, "")

    def _startup_first_face_greet_reply(self, emotion):
        name = self._load_memory().get("robot_name", "灵犀")
        replies = {
            "sad": "你好呀，我是%s。我看到你有点低落，我在这儿陪你。" % name,
            "angry": "你好呀，我是%s。我感觉你有点着急，我们慢慢来。" % name,
            "anxious": "你好呀，我是%s。别紧张，我会一直陪着你。" % name,
            "happy": "你好呀，我是%s。看到你这么开心我也很高兴。" % name,
            "excited": "你好呀，我是%s。你看起来很有精神，我们开始吧。" % name,
            "neutral": "你好呀，我是%s，很高兴见到你。" % name,
            "unknown": "你好呀，我是%s，很高兴见到你。" % name,
        }
        return replies.get(emotion, "你好呀，我是%s，很高兴见到你。" % name)

    def _greet_on_first_face_detected(self):
        with self._startup_face_search_lock:
            if self._startup_first_face_greet_done:
                return
            self._startup_first_face_greet_done = True

        cue = self._extract_emotion_cue(self._latest_vision)
        emotion = "neutral"
        confidence = 0.0
        if cue:
            emotion = cue.get("emotion") or "neutral"
            confidence = cue.get("confidence") or 0.0

        reply = self._startup_first_face_greet_reply(emotion)
        plan = {
            "reply": reply,
            "emotion": emotion,
            "intent": "startup_first_face_greeting",
            "safety": {"can_move": False, "reason": "face detected, greet first"},
            "actions": [],
            "vision_emotion": {
                "emotion": emotion,
                "confidence": confidence,
                "age_seconds": round(max(0.0, time.time() - self._latest_vision_time), 2)
                if self._latest_vision_time else None,
            },
        }
        record = {
            "timestamp": time.time(),
            "source_text": "",
            "trigger": "startup_first_face_greeting",
            "vision": self._latest_vision,
            "memory": self._load_memory(),
            "status": "done",
            "current_action_index": 0,
            "plan": plan,
        }
        rospy.loginfo("Startup first-face greeting: %s", reply)
        self.plan_pub.publish(String(json.dumps(plan, ensure_ascii=False)))
        self._publish_reply(reply)
        self._write_plan_file(record)

    def _sanitize_plan(self, plan):
        if not isinstance(plan, dict):
            plan = {}

        sanitized = {
            "reply": str(plan.get("reply", ""))[:120],
            "emotion": str(plan.get("emotion", "unknown"))[:30],
            "intent": str(plan.get("intent", ""))[:80],
            "safety": plan.get("safety") if isinstance(plan.get("safety"), dict) else {},
            "actions": [],
        }
        if plan.get("profile_saved"):
            sanitized["profile_saved"] = True
        if plan.get("memory_saved"):
            sanitized["memory_saved"] = True
        if plan.get("handled_elsewhere"):
            sanitized["handled_elsewhere"] = True
        if plan.get("local_priority"):
            sanitized["local_priority"] = True
        if isinstance(plan.get("vision_emotion"), dict):
            sanitized["vision_emotion"] = plan["vision_emotion"]

        for action in plan.get("actions", []):
            if not isinstance(action, dict):
                continue
            action_type = action.get("type")
            if action_type == "forward":
                meters = self._clamp_float(action.get("meters", 0.5), 0.0, self.max_meters)
                if meters > 0:
                    sanitized["actions"].append({"type": "forward", "meters": meters})
            elif action_type == "backward":
                meters = self._clamp_float(action.get("meters", 0.5), 0.0, self.max_meters)
                if meters > 0:
                    sanitized["actions"].append({"type": "backward", "meters": meters})
            elif action_type == "turn":
                degrees = self._clamp_float(action.get("degrees", 0.0), -self.max_degrees, self.max_degrees)
                if abs(degrees) > 0.1:
                    sanitized["actions"].append({"type": "turn", "degrees": degrees})
            elif action_type == "wait":
                seconds = self._clamp_float(action.get("seconds", 0.5), 0.0, 3.0)
                if seconds > 0:
                    sanitized["actions"].append({"type": "wait", "seconds": seconds})
            elif action_type == "stop":
                sanitized["actions"].append({"type": "stop"})

        return sanitized

    def _local_rule_plan(self, text):
        normalized = self._normalize_text(text)

        if self._is_recall_poem_question(normalized):
            poem = self._find_recent_poem_request()
            if poem:
                return {
                    "reply": "你刚才让我背《%s》。" % poem,
                    "emotion": "neutral",
                    "intent": "recall_recent_poem_request",
                    "safety": {"can_move": False, "reason": "conversation recall only"},
                    "actions": [],
                    "local_priority": True,
                }
            return {
                "reply": "我这会儿没想起来你让我背哪首诗，你再说一遍，我马上记住。",
                "emotion": "neutral",
                "intent": "recall_recent_poem_request_miss",
                "safety": {"can_move": False, "reason": "conversation recall only"},
                "actions": [],
                "local_priority": True,
            }

        robot_name = self._extract_robot_name_memory(text)
        if robot_name:
            memory = self._load_memory()
            memory["robot_name"] = robot_name
            self._save_memory(memory)
            return {
                "reply": "好呀，我记住了，我叫%s。" % robot_name,
                "emotion": "happy",
                "intent": "remember_robot_name",
                "safety": {"can_move": False, "reason": "memory update only"},
                "actions": [],
                "memory_saved": True,
                "local_priority": True,
            }

        if self._is_robot_name_question(normalized):
            name = self._load_memory().get("robot_name", "小车")
            return {
                "reply": "我叫%s。" % name,
                "emotion": "neutral",
                "intent": "answer_robot_name",
                "safety": {"can_move": False, "reason": "conversation only"},
                "actions": [],
                "local_priority": True,
            }

        if self._is_date_question(normalized):
            now = time.localtime()
            date_text = "%d月%d号" % (now.tm_mon, now.tm_mday)
            return {
                "reply": "今天是%s。" % date_text,
                "emotion": "neutral",
                "intent": "tell_date",
                "safety": {"can_move": False, "reason": "date query"},
                "actions": [],
                "local_priority": True,
            }

        if self._is_time_question(normalized):
            now_text = time.strftime("%H点%M分", time.localtime())
            return {
                "reply": "现在时间是%s。" % now_text,
                "emotion": "neutral",
                "intent": "tell_time",
                "safety": {"can_move": False, "reason": "time query"},
                "actions": [],
                "local_priority": True,
            }

        if self._is_find_me_request(normalized):
            return {
                "reply": "好，我来找你。",
                "emotion": "neutral",
                "intent": "manual_face_search",
                "safety": {"can_move": True, "reason": "user explicitly asked robot to find face"},
                "actions": [],
                "local_priority": True,
            }

        if self._is_greeting_or_chat(normalized):
            name = self._load_memory().get("robot_name", "灵犀")
            return {
                "reply": "你好呀，我是%s，我在这儿陪你。" % name,
                "emotion": "neutral",
                "intent": "greeting",
                "safety": {"can_move": False, "reason": "conversation only"},
                "actions": [],
            }

        fact = self._extract_memory_fact(text)
        if fact:
            memory = self._load_memory()
            facts = memory.setdefault("facts", [])
            facts.append({"text": fact, "timestamp": time.time()})
            memory["facts"] = facts[-30:]
            self._save_memory(memory)
            return {
                "reply": "好，我记住了。",
                "emotion": "happy",
                "intent": "remember_fact",
                "safety": {"can_move": False, "reason": "memory update only"},
                "actions": [],
                "memory_saved": True,
                "local_priority": True,
            }

        if self._is_music_related_text(normalized):
            return {
                "reply": "",
                "emotion": "neutral",
                "intent": "handled_by_music_player",
                "safety": {"can_move": False, "reason": "music player handles this input"},
                "actions": [],
                "handled_elsewhere": True,
                "local_priority": True,
            }

        if "跳舞" in normalized or "舞" in normalized:
            if self._is_dance_profile_update(normalized):
                actions = self._parse_motion_sequence(normalized)
                if actions:
                    self._save_dance_profile(actions, text)
                    return {
                        "reply": "记住了，以后跳舞就按这套动作来。",
                        "emotion": "happy",
                        "intent": "set_dance_profile",
                        "safety": {"can_move": False, "reason": "dance profile saved only"},
                        "actions": [],
                        "profile_saved": True,
                        "local_priority": True,
                    }
                return {
                    "reply": "我听到你想改跳舞动作，但还没解析出具体动作。",
                    "emotion": "neutral",
                    "intent": "set_dance_profile_failed",
                    "safety": {"can_move": False, "reason": "no dance actions parsed"},
                    "actions": [],
                    "local_priority": True,
                }

            custom_actions = self._parse_motion_sequence(normalized)
            if custom_actions:
                plan = {
                    "reply": "好呀，我按你刚才说的动作跳。",
                    "emotion": "happy",
                    "intent": "custom_dance",
                    "safety": {"can_move": True, "reason": "local custom dance parsed"},
                    "actions": custom_actions,
                }
                if self._is_simple_local_motion_request(normalized, custom_actions):
                    plan["local_priority"] = True
                return plan

            saved_actions = self._load_dance_profile()
            if saved_actions:
                return {
                    "reply": "好呀，我按你保存的动作跳。",
                    "emotion": "happy",
                    "intent": "dance_profile",
                    "safety": {"can_move": True, "reason": "saved dance profile"},
                    "actions": saved_actions,
                    "local_priority": True,
                }

            return {
                "reply": "好呀，我跳一个简单的小舞。",
                "emotion": "happy",
                "intent": "dance",
                "safety": {"can_move": True, "reason": "small dance motion"},
                "actions": [
                    {"type": "turn", "degrees": 45},
                    {"type": "turn", "degrees": -45},
                    {"type": "forward", "meters": 0.3},
                    {"type": "backward", "meters": 0.3},
                    {"type": "stop"},
                ],
            }

        actions = self._parse_motion_sequence(normalized)
        if actions:
            plan = {
                "reply": "好呀，我慢慢来。",
                "emotion": "neutral",
                "intent": "multi_step_motion" if len(actions) > 1 else "direct_motion",
                "safety": {"can_move": True, "reason": "local rule parsed motion command"},
                "actions": actions,
            }
            if self._is_simple_local_motion_request(normalized, actions):
                plan["local_priority"] = True
            return plan

        return {
            "reply": "这个我还没听明白。你可以慢慢说，比如：前进一点点，或者左转一下。",
            "emotion": "unknown",
            "intent": "unknown",
            "safety": {"can_move": False, "reason": "no safe executable action parsed"},
            "actions": [],
        }

    def _parse_motion_sequence(self, normalized):
        command_re = re.compile(
            r"(?P<forward>向前|往前|前进)|"
            r"(?P<backward>向后|往后|后退)|"
            r"(?P<left>向左转|左转|向左)|"
            r"(?P<right>向右转|右转|向右)|"
            r"(?P<stop>停止|停车)"
        )
        matches = list(command_re.finditer(normalized))
        actions = []

        for index, match in enumerate(matches):
            segment_end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
            segment = normalized[match.end():segment_end]

            if match.lastgroup == "forward":
                meters = self._extract_quantity(segment, ("米", "m", "M"))
                actions.append({"type": "forward", "meters": meters if meters is not None else 0.5})
            elif match.lastgroup == "backward":
                meters = self._extract_quantity(segment, ("米", "m", "M"))
                actions.append({"type": "backward", "meters": meters if meters is not None else 0.5})
            elif match.lastgroup == "left":
                degrees = self._extract_turn_degrees(segment)
                self._append_turn_actions(actions, degrees)
            elif match.lastgroup == "right":
                degrees = self._extract_turn_degrees(segment)
                self._append_turn_actions(actions, -degrees)
            elif match.lastgroup == "stop":
                actions.append({"type": "stop"})

        return actions

    def _is_simple_local_motion_request(self, normalized, actions):
        if len(actions) != 1:
            return False
        if "舞" in normalized or "跳舞" in normalized:
            return False
        if any(word in normalized for word in ("先", "再", "然后", "接着", "最后", "顺便", "并且", "同时")):
            return False

        action = actions[0]
        action_type = action.get("type")
        if action_type in ("forward", "backward"):
            return float(action.get("meters", 0.0) or 0.0) <= self.max_meters
        if action_type == "turn":
            return abs(float(action.get("degrees", 0.0) or 0.0)) <= self.max_degrees
        return action_type == "stop"

    def _is_find_me_request(self, normalized):
        keywords = (
            "你找一下我",
            "找一下我",
            "你找找我",
            "找找我",
            "你看看我在哪",
            "看看我在哪",
            "看我在哪",
            "你看下我在哪",
            "你看看我在哪里",
            "找我在哪",
            "找我在哪里",
        )
        if any(word in normalized for word in keywords):
            return True
        return ("找" in normalized and "我" in normalized and ("在哪" in normalized or "哪里" in normalized))

    def _is_recall_poem_question(self, normalized):
        phrases = (
            "刚才让你背什么诗",
            "刚才让你背什么",
            "我让你背什么诗",
            "我让你背什么",
            "之前让你背什么诗",
            "之前让你背什么",
        )
        return any(phrase in normalized for phrase in phrases)

    def _find_recent_poem_request(self):
        for text in reversed(self._recent_user_texts):
            normalized = self._normalize_text(text)
            if "什么" in normalized:
                continue
            if not any(word in normalized for word in ("背", "背诵", "朗诵", "念")):
                continue
            match = re.search(r"(?:背诵|朗诵|背|念)([^，,。！？?；;]+)", text)
            if not match:
                continue
            poem = self._clean_memory_value(match.group(1))
            if poem and len(poem) <= 20:
                return poem.strip("《》")
        return ""

    def _is_dance_profile_update(self, normalized):
        update_words = (
            "以后",
            "今后",
            "往后",
            "之后",
            "下次",
            "下回",
            "改成",
            "换成",
            "设置成",
            "设成",
            "定义成",
            "记住",
        )
        spoken_trigger = (
            ("我说跳舞" in normalized or "说跳舞" in normalized)
            and ("时候" in normalized or "时" in normalized)
            and ("你就" in normalized or "就" in normalized)
        )
        return any(word in normalized for word in update_words) or spoken_trigger

    def _normalize_text(self, text):
        return re.sub(r"[\s，,。！？?；;：:、（）()\[\]【】《》“”\"']+", "", text)

    def _extract_robot_name_memory(self, text):
        if not any(word in text for word in ("记住", "名字", "叫做", "叫")):
            return ""

        patterns = (
            r"(?:你的名字|你名字|名字)(?:叫做|叫|是)([^，,。！？?；;\s]+)",
            r"(?:以后你|今后你|往后你|从现在起你|请你|你)(?:就)?(?:叫做|叫)([^，,。！？?；;\s]+)",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            name = self._clean_memory_value(match.group(1))
            if name and "什么" not in name and len(name) <= 12:
                return name
        return ""

    def _is_robot_name_question(self, normalized):
        return (
            ("你叫什么" in normalized)
            or ("你的名字" in normalized and any(word in normalized for word in ("什么", "叫啥", "叫什")))
            or ("你名字" in normalized and any(word in normalized for word in ("什么", "叫啥", "叫什")))
        )

    def _is_greeting_or_chat(self, normalized):
        greetings = (
            "你好",
            "您好",
            "早上好",
            "中午好",
            "下午好",
            "晚上好",
            "哈喽",
            "hello",
            "嗨",
            "在吗",
            "你在吗",
        )
        if normalized in greetings:
            return True
        return any(word in normalized for word in ("陪我聊", "聊聊天", "说说话"))

    def _is_music_related_text(self, normalized):
        if normalized in ("是", "是的", "对", "对的", "没错", "好", "好的", "可以", "不是", "不对", "不要", "算了"):
            return True
        return any(
            word in normalized
            for word in (
                "播放",
                "放歌",
                "放首",
                "想听",
                "听歌",
                "停止播放",
                "停播放",
                "停音乐",
                "停止音乐",
                "关音乐",
                "关掉音乐",
                "停歌",
                "音量",
                "声音大",
                "声音小",
                "大声",
                "小声",
                "静音",
            )
        )

    def _is_date_question(self, normalized):
        return any(
            phrase in normalized
            for phrase in (
                "今天几月几号",
                "今天几号",
                "今天日期",
                "今天多少号",
                "今天是几号",
                "几月几号",
                "几号",
                "日期",
            )
        )

    def _is_time_question(self, normalized):
        return any(
            phrase in normalized
            for phrase in (
                "几点",
                "现在几点",
                "当前时间",
                "现在时间",
                "什么时间",
                "几点了",
                "时间多少",
            )
        )

    def _extract_memory_fact(self, text):
        if not any(word in text for word in ("记住", "记一下", "你要记得", "帮我记住")):
            return ""

        match = re.search(r"(?:请你|你要|帮我|给我)?记(?:住|一下|得)(.*)", text)
        if not match:
            return ""
        fact = self._clean_memory_value(match.group(1))
        if not fact:
            return ""
        if "名字" in fact and ("叫" in fact or "是" in fact):
            return ""
        return fact[:120]

    def _clean_memory_value(self, value):
        value = re.sub(r"^[：:，,。！？?；;\s]+", "", value)
        value = re.split(r"[，,。！？?；;\n]", value, 1)[0]
        value = value.strip()
        value = re.sub(r"^(就是|叫做|叫|是)", "", value).strip()
        return value

    def _load_memory(self):
        if not self.memory_file or not os.path.exists(self.memory_file):
            return {}
        try:
            with open(self.memory_file, "r", encoding="utf-8") as handle:
                memory = json.load(handle)
        except (OSError, ValueError) as exc:
            rospy.logwarn("Failed to load companion memory: %s", exc)
            return {}
        if not isinstance(memory, dict):
            return {}
        return memory

    def _save_memory(self, memory):
        if not self.memory_file:
            return
        memory["updated_at"] = time.time()
        directory = os.path.dirname(self.memory_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp_path = self.memory_file + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(memory, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, self.memory_file)
        rospy.loginfo("Saved companion memory to %s", self.memory_file)

    def _save_dance_profile(self, actions, source_text):
        record = {
            "updated_at": time.time(),
            "source_text": source_text,
            "actions": actions,
        }
        directory = os.path.dirname(self.dance_profile_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp_path = self.dance_profile_file + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, self.dance_profile_file)
        rospy.loginfo("Saved dance profile to %s", self.dance_profile_file)

    def _load_dance_profile(self):
        if not self.dance_profile_file or not os.path.exists(self.dance_profile_file):
            return []
        try:
            with open(self.dance_profile_file, "r", encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, ValueError) as exc:
            rospy.logwarn("Failed to load dance profile: %s", exc)
            return []
        actions = record.get("actions", [])
        if not isinstance(actions, list):
            return []
        return actions

    def _extract_turn_degrees(self, segment):
        degrees = self._extract_quantity(segment, ("度",))
        if degrees is not None:
            return degrees

        turns = self._extract_quantity(segment, ("圈", "周"))
        if turns is not None:
            return turns * 360.0

        return 90.0

    def _append_turn_actions(self, actions, degrees):
        if abs(degrees) <= 0.1:
            return
        limit = max(1.0, min(abs(self.max_degrees), 180.0))
        sign = 1.0 if degrees > 0 else -1.0
        remaining = abs(degrees)
        while remaining > 0.1:
            chunk = min(limit, remaining)
            actions.append({"type": "turn", "degrees": sign * chunk})
            remaining -= chunk

    def _extract_quantity(self, text, unit_suffixes):
        if "半" in text and any(unit in text for unit in unit_suffixes):
            return 0.5

        suffix_pattern = "|".join(re.escape(unit) for unit in unit_suffixes)
        arabic = re.search(r"(\d+(?:\.\d+)?)(?:个)?(?:%s)" % suffix_pattern, text)
        if arabic:
            return float(arabic.group(1))

        chinese = re.search(r"([零一二两三四五六七八九十百千点]+)(?:个)?(?:%s)" % suffix_pattern, text)
        if chinese:
            return self._chinese_number_to_float(chinese.group(1))

        return None

    def _chinese_number_to_float(self, text):
        if "点" not in text:
            return float(self._chinese_integer_to_int(text))

        integer_part, decimal_part = text.split("点", 1)
        integer_value = self._chinese_integer_to_int(integer_part) if integer_part else 0
        decimal_digits = []
        for char in decimal_part:
            if char not in CHINESE_DIGITS:
                break
            decimal_digits.append(str(CHINESE_DIGITS[char]))
        if not decimal_digits:
            return float(integer_value)
        return float("%d.%s" % (integer_value, "".join(decimal_digits)))

    def _chinese_integer_to_int(self, text):
        if not text:
            return 0

        total = 0
        current = 0
        for char in text:
            if char in CHINESE_DIGITS:
                current = CHINESE_DIGITS[char]
            elif char in CHINESE_UNITS:
                unit = CHINESE_UNITS[char]
                if current == 0:
                    current = 1
                total += current * unit
                current = 0
        return total + current

    def _publish_actions(self, actions, record):
        runnable_actions = list(actions)
        if (
            len(runnable_actions) > 1
            and runnable_actions[-1].get("type") == "stop"
        ):
            rospy.loginfo("Skipping trailing stop; each motion command already sends zero velocity.")
            runnable_actions = runnable_actions[:-1]

        for index, action in enumerate(runnable_actions):
            if self._stop_requested.is_set():
                rospy.loginfo("Action execution interrupted by stop request before action %d.", index)
                break

            command_text, wait_seconds = self._action_to_text(action)
            if not command_text:
                continue

            record["status"] = "executing"
            record["current_action_index"] = index
            record["current_action"] = action
            record["current_motion_text"] = command_text
            self._write_plan_file(record)

            rospy.loginfo("AI motion command: %s", command_text)
            self.motion_pub.publish(String(command_text))
            if wait_seconds > 0:
                end_time = time.time() + wait_seconds + self.action_gap
                while time.time() < end_time:
                    if self._stop_requested.is_set():
                        rospy.loginfo("Action execution interrupted by stop request during wait.")
                        break
                    time.sleep(0.05)
                if self._stop_requested.is_set():
                    break

        record["status"] = "done"
        record["current_action_index"] = len(runnable_actions)
        record.pop("current_action", None)
        record.pop("current_motion_text", None)
        self._write_plan_file(record)

    def _action_to_text(self, action):
        action_type = action.get("type")
        if action_type == "raw_text":
            return action.get("text", ""), self.default_motion_wait
        if action_type == "stop":
            return "停止", 0.2
        if action_type == "wait":
            return "", float(action.get("seconds", 0.0))
        if action_type == "forward":
            meters = float(action.get("meters", 0.5))
            return "前进%s米" % self._format_number(meters), self._linear_wait(meters)
        if action_type == "backward":
            meters = float(action.get("meters", 0.5))
            return "后退%s米" % self._format_number(meters), self._linear_wait(meters)
        if action_type == "turn":
            degrees = float(action.get("degrees", 0.0))
            if degrees >= 0:
                return "左转%s度" % self._format_number(abs(degrees)), self._angular_wait(degrees)
            return "右转%s度" % self._format_number(abs(degrees)), self._angular_wait(degrees)
        return "", 0.0

    def _linear_wait(self, meters):
        return abs(meters) / max(abs(self.linear_speed), 0.001)

    def _angular_wait(self, degrees):
        return math.radians(abs(degrees)) / max(abs(self.angular_speed), 0.001)

    def _clamp_float(self, value, low, high):
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = low
        return max(low, min(high, number))

    def _format_number(self, value):
        value = float(value)
        if abs(value - round(value)) < 1e-6:
            return str(int(round(value)))
        return ("%.2f" % value).rstrip("0").rstrip(".")

    def _append_user_dialogue(self, text):
        value = str(text or "").strip()
        if not value:
            return
        self._recent_user_texts.append(value)
        self._dialogue_history.append({"role": "user", "text": value, "timestamp": time.time()})

    def _publish_reply(self, reply):
        text = str(reply or "").strip()
        if not text:
            return
        self.reply_pub.publish(String(text))
        self._reply_memory.append(text)
        self._dialogue_history.append({"role": "assistant", "text": text, "timestamp": time.time()})

    def _write_learning_event_if_needed(self, record):
        plan = record.get("plan") if isinstance(record, dict) else {}
        if not isinstance(plan, dict) or plan.get("intent") != "object_learning_query":
            return

        objects = self._extract_learning_objects(record.get("vision"))
        if not objects:
            return

        timestamp = float(record.get("timestamp") or time.time())
        source_text = str(record.get("source_text") or "").strip()
        reply = str(plan.get("reply") or "").strip()
        event_key = "%s|%s|%s" % (
            "%.6f" % timestamp,
            source_text,
            ",".join(item["class_name"] for item in objects),
        )
        event = {
            "event_id": hashlib.sha1(event_key.encode("utf-8")).hexdigest(),
            "timestamp": timestamp,
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(timestamp)),
            "source": "ai_companion_node",
            "intent": "object_learning_query",
            "user_text": source_text,
            "reply": reply,
            "objects": objects,
        }
        try:
            directory = os.path.dirname(self.learning_event_file)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.learning_event_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            rospy.loginfo("Recorded learning event with %d object(s).", len(objects))
        except Exception as exc:
            rospy.logwarn("Failed to write learning event: %s", exc)

    def _extract_learning_objects(self, vision):
        if not isinstance(vision, dict):
            return []
        raw_objects = vision.get("objects") or []
        if isinstance(raw_objects, dict):
            raw_objects = raw_objects.get("items") or []
        if not isinstance(raw_objects, list):
            return []

        objects = []
        seen = set()
        for item in raw_objects:
            if not isinstance(item, dict):
                continue
            class_name = str(item.get("class") or item.get("class_name") or item.get("name") or item.get("label") or "").strip()
            if not class_name or class_name in seen:
                continue
            seen.add(class_name)
            raw_confidence = item.get("confidence")
            if raw_confidence is None:
                raw_confidence = item.get("score")
            try:
                confidence = float(raw_confidence) if raw_confidence is not None else 1.0
            except (TypeError, ValueError):
                confidence = 0.0
            objects.append({"class_name": class_name, "confidence": confidence})
        return objects

    def _normalize_object_name(self, value):
        return str(value or "").strip()

    def _normalize_object_key(self, value):
        return self._normalize_object_name(value).lower().replace("_", " ")

    def _dangerous_object_display_name(self, class_name):
        key = self._normalize_object_key(class_name)
        if not key:
            return ""
        if key in DANGEROUS_OBJECT_DISPLAY_NAMES:
            return DANGEROUS_OBJECT_DISPLAY_NAMES[key]
        for keyword, display_name in DANGEROUS_OBJECT_KEYWORDS:
            if keyword in key:
                return display_name
        return ""

    def _extract_dangerous_objects(self, vision):
        if not isinstance(vision, dict):
            return []
        raw_objects = vision.get("objects") or []
        if isinstance(raw_objects, dict):
            raw_objects = raw_objects.get("items") or []
        if not isinstance(raw_objects, list):
            return []

        dangerous = []
        seen = set()
        for item in raw_objects:
            if not isinstance(item, dict):
                continue
            class_name = self._normalize_object_name(
                item.get("class") or item.get("class_name") or item.get("name") or item.get("label")
            )
            display_name = self._dangerous_object_display_name(class_name)
            if not class_name or not display_name:
                continue
            try:
                confidence = float(item.get("confidence") or item.get("score") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < self.dangerous_object_min_confidence:
                continue
            key = self._normalize_object_key(class_name)
            if key in seen:
                continue
            seen.add(key)
            dangerous.append(
                {
                    "key": key,
                    "class_name": class_name,
                    "display_name": display_name,
                    "confidence": confidence,
                }
            )
        return dangerous

    def _maybe_alert_dangerous_objects(self):
        if not self.dangerous_object_alert_enabled:
            return
        dangerous = self._extract_dangerous_objects(self._latest_vision)
        if not dangerous:
            return

        now = time.time()
        interval = max(1.0, self.dangerous_object_alert_interval)
        due = []
        with self._dangerous_object_alert_lock:
            for obj in dangerous:
                last_alert = self._dangerous_object_last_alert_at.get(obj["key"], 0.0)
                if now - last_alert >= interval:
                    due.append(obj)
                    self._dangerous_object_last_alert_at[obj["key"]] = now
        if not due:
            return

        names = "、".join(obj["display_name"] for obj in due[:3])
        if len(due) == 1:
            reply = "我看到%s，这个物品很危险，请小朋友马上放下，离它远一点，请大人来帮忙。" % names
        else:
            reply = "我看到%s，这些物品很危险，请小朋友马上放下，离它远一点，请大人来帮忙。" % names
        rospy.logwarn(
            "Dangerous object detected: %s",
            ",".join("%s:%.2f" % (obj["class_name"], obj["confidence"]) for obj in due),
        )
        self._publish_reply(reply)

    def _write_plan_file(self, record):
        if not self.plan_file:
            return

        directory = os.path.dirname(self.plan_file)
        if directory:
            os.makedirs(directory, exist_ok=True)

        tmp_path = self.plan_file + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, self.plan_file)


if __name__ == "__main__":
    rospy.init_node("ai_companion_node")
    AiCompanionNode()
    rospy.spin()
