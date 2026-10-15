#!/usr/bin/env python3
import math
import re
import subprocess
import threading
import time

from collections import deque, namedtuple

import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


Command = namedtuple("Command", "linear angular duration name is_stop")

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


class SshVoiceControl:
    def __init__(self):
        self.jetson_user = rospy.get_param("~jetson_user", "clbrobot")
        self.jetson_host = rospy.get_param("~jetson_host", "192.168.210.91")
        self.remote_ros_setup = rospy.get_param(
            "~remote_ros_setup",
            "source /opt/ros/melodic/setup.bash && source ~/catkin_ws/devel/setup.bash",
        )
        self.remote_ros_master_uri = rospy.get_param("~remote_ros_master_uri", "")
        self.input_topic = rospy.get_param("~input_topic", "iat_text")
        self.linear_speed = float(rospy.get_param("~linear_speed", 0.15))
        self.angular_speed = float(rospy.get_param("~angular_speed", 0.45))
        self.command_duration = float(rospy.get_param("~command_duration", 3.0))
        self.stop_duration = float(rospy.get_param("~stop_duration", 0.3))
        self.default_turn_degrees = float(rospy.get_param("~default_turn_degrees", 90.0))
        self.max_command_duration = float(rospy.get_param("~max_command_duration", 30.0))
        self.ssh_connect_timeout = int(rospy.get_param("~ssh_connect_timeout", 5))
        self.obstacle_avoid_enabled = bool(rospy.get_param("~obstacle_avoid_enabled", False))
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.obstacle_distance = float(rospy.get_param("~obstacle_distance", 0.55))
        self.clear_distance = float(rospy.get_param("~clear_distance", 0.70))
        self.front_sector_degrees = float(rospy.get_param("~front_sector_degrees", 25.0))
        self.side_sector_start_degrees = float(rospy.get_param("~side_sector_start_degrees", 35.0))
        self.side_sector_end_degrees = float(rospy.get_param("~side_sector_end_degrees", 85.0))
        self.avoid_turn_speed = float(rospy.get_param("~avoid_turn_speed", 0.45))
        self.avoid_turn_step_seconds = float(rospy.get_param("~avoid_turn_step_seconds", 0.45))
        self.avoid_pause_seconds = float(rospy.get_param("~avoid_pause_seconds", 0.15))
        self.scan_timeout = float(rospy.get_param("~scan_timeout", 1.0))

        self._lock = threading.Lock()
        self._process_lock = threading.Lock()
        self._queue_condition = threading.Condition(self._lock)
        self._queue = deque()
        self._current_process = None
        self._target = "%s@%s" % (self.jetson_user, self.jetson_host)
        self._latest_scan = None
        self._latest_scan_time = 0.0
        self._scan_lock = threading.Lock()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        rospy.Subscriber(self.input_topic, String, self._on_text, queue_size=10)
        rospy.Subscriber(self.scan_topic, LaserScan, self._on_scan, queue_size=1)
        rospy.loginfo("SSH voice control ready for %s on %s", self._target, self.input_topic)

    def _on_scan(self, msg):
        with self._scan_lock:
            self._latest_scan = msg
            self._latest_scan_time = time.time()

    def _on_text(self, msg):
        command = self._parse_command(msg.data)
        if command is None:
            rospy.logwarn("Unrecognized voice command: %s", msg.data)
            return

        if command.is_stop:
            self._cancel_current_process()

        rospy.loginfo(
            "Voice command '%s' -> linear %.3f angular %.3f duration %.2fs",
            command.name,
            command.linear,
            command.angular,
            command.duration,
        )

        with self._queue_condition:
            if command.is_stop:
                self._queue.clear()
            self._queue.append(command)
            queue_size = len(self._queue)
            self._queue_condition.notify()

        if queue_size > 1:
            rospy.loginfo("Queued '%s'; %d command(s) waiting.", command.name, queue_size - 1)

    def _worker_loop(self):
        while not rospy.is_shutdown():
            with self._queue_condition:
                while not self._queue and not rospy.is_shutdown():
                    self._queue_condition.wait(0.2)
                if rospy.is_shutdown():
                    return
                command = self._queue.popleft()

            if self._should_use_obstacle_avoidance(command):
                self._send_cmd_vel_with_avoidance(command)
            else:
                self._send_cmd_vel(
                    command.linear,
                    command.angular,
                    command.duration,
                    command.name,
                )

    def _parse_command(self, text):
        normalized = text.replace(" ", "")

        if "停止" in normalized or "停车" in normalized or "停下来" in normalized or "停下" in normalized:
            return Command(0.0, 0.0, self.stop_duration, "停止", True)

        if "前进" in normalized or "向前" in normalized:
            distance = self._extract_quantity(normalized, ("米", "m", "M"))
            if distance is None:
                duration = self.command_duration
                name = "前进"
            else:
                duration = distance / max(abs(self.linear_speed), 0.001)
                name = "前进%s米" % self._format_number(distance)
            return Command(
                self.linear_speed,
                0.0,
                self._bounded_duration(duration, name),
                name,
                False,
            )

        if "后退" in normalized or "向后" in normalized:
            distance = self._extract_quantity(normalized, ("米", "m", "M"))
            if distance is None:
                duration = self.command_duration
                name = "后退"
            else:
                duration = distance / max(abs(self.linear_speed), 0.001)
                name = "后退%s米" % self._format_number(distance)
            return Command(
                -self.linear_speed,
                0.0,
                self._bounded_duration(duration, name),
                name,
                False,
            )

        if "左转" in normalized or "向左" in normalized:
            degrees = self._extract_quantity(normalized, ("度",))
            if degrees is None:
                degrees = self.default_turn_degrees
                name = "左转%s度" % self._format_number(degrees)
            else:
                name = "左转%s度" % self._format_number(degrees)
            duration = math.radians(degrees) / max(abs(self.angular_speed), 0.001)
            return Command(
                0.0,
                abs(self.angular_speed),
                self._bounded_duration(duration, name),
                name,
                False,
            )

        if "右转" in normalized or "向右" in normalized:
            degrees = self._extract_quantity(normalized, ("度",))
            if degrees is None:
                degrees = self.default_turn_degrees
                name = "右转%s度" % self._format_number(degrees)
            else:
                name = "右转%s度" % self._format_number(degrees)
            duration = math.radians(degrees) / max(abs(self.angular_speed), 0.001)
            return Command(
                0.0,
                -abs(self.angular_speed),
                self._bounded_duration(duration, name),
                name,
                False,
            )

        return None

    def _bounded_duration(self, duration, name):
        if duration > self.max_command_duration:
            rospy.logwarn(
                "Command '%s' duration %.2fs exceeds max %.2fs; clamping.",
                name,
                duration,
                self.max_command_duration,
            )
            return self.max_command_duration
        return max(duration, 0.0)

    def _extract_quantity(self, text, unit_suffixes):
        suffix_pattern = "|".join(re.escape(unit) for unit in unit_suffixes)
        arabic = re.search(r"(\d+(?:\.\d+)?)(?:%s)" % suffix_pattern, text)
        if arabic:
            return float(arabic.group(1))

        chinese = re.search(r"([零一二两三四五六七八九十百千点]+)(?:%s)" % suffix_pattern, text)
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

    def _format_number(self, value):
        if abs(value - round(value)) < 1e-6:
            return str(int(round(value)))
        return ("%.2f" % value).rstrip("0").rstrip(".")

    def _cancel_current_process(self):
        with self._process_lock:
            process = self._current_process
            if process is None or process.poll() is not None:
                return
            rospy.loginfo("Canceling active SSH velocity command before stop.")
            process.terminate()

    def _should_use_obstacle_avoidance(self, command):
        return (
            self.obstacle_avoid_enabled
            and not command.is_stop
            and command.linear > 0.0
            and abs(command.angular) < 1e-6
        )

    def _get_scan_snapshot(self):
        with self._scan_lock:
            return self._latest_scan, self._latest_scan_time

    def _range_at_angle(self, scan, angle_rad):
        if scan is None or not scan.ranges:
            return float("inf")
        if angle_rad < scan.angle_min or angle_rad > scan.angle_max:
            return float("inf")
        index = int((angle_rad - scan.angle_min) / scan.angle_increment)
        if index < 0 or index >= len(scan.ranges):
            return float("inf")
        value = scan.ranges[index]
        if math.isinf(value) or math.isnan(value):
            return float("inf")
        return value

    def _sector_min_distance(self, scan, start_deg, end_deg):
        start_rad = math.radians(start_deg)
        end_rad = math.radians(end_deg)
        step = max(scan.angle_increment, math.radians(1.0))
        angle = start_rad
        min_dist = float("inf")
        while angle <= end_rad + 1e-6:
            dist = self._range_at_angle(scan, angle)
            if dist < min_dist:
                min_dist = dist
            angle += step
        return min_dist

    def _is_front_blocked(self):
        scan, scan_time = self._get_scan_snapshot()
        if scan is None or (time.time() - scan_time) > self.scan_timeout:
            return False, None, float("inf"), float("inf")

        front_min = self._sector_min_distance(
            scan,
            -abs(self.front_sector_degrees),
            abs(self.front_sector_degrees),
        )
        left_min = self._sector_min_distance(
            scan,
            abs(self.side_sector_start_degrees),
            abs(self.side_sector_end_degrees),
        )
        right_min = self._sector_min_distance(
            scan,
            -abs(self.side_sector_end_degrees),
            -abs(self.side_sector_start_degrees),
        )
        blocked = front_min < self.obstacle_distance
        return blocked, scan, left_min, right_min

    def _send_cmd_vel_with_avoidance(self, command):
        total_end = time.time() + command.duration
        while not rospy.is_shutdown() and time.time() < total_end:
            blocked, _, left_min, right_min = self._is_front_blocked()
            if blocked:
                turn_left = left_min >= right_min
                turn_name = "左绕避障" if turn_left else "右绕避障"
                turn_angular = abs(self.avoid_turn_speed) if turn_left else -abs(self.avoid_turn_speed)
                rospy.loginfo(
                    "Obstacle ahead during '%s': left=%.2f right=%.2f, %s",
                    command.name,
                    left_min,
                    right_min,
                    turn_name,
                )
                self._send_cmd_vel(0.0, turn_angular, self.avoid_turn_step_seconds, turn_name)
                self._send_cmd_vel(0.0, 0.0, self.avoid_pause_seconds, "避障停顿")
                continue

            remaining = max(0.0, total_end - time.time())
            step = min(0.6, remaining)
            if step <= 0.0:
                break
            self._send_cmd_vel(command.linear, 0.0, step, command.name)

        self._send_cmd_vel(0.0, 0.0, self.stop_duration, "停止")

    def _send_cmd_vel(self, linear, angular, duration, name):
        timeout = max(duration + self.ssh_connect_timeout + 5.0, 10.0)
        script = self._remote_script(linear, angular, duration)
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

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        with self._process_lock:
            self._current_process = process

        try:
            stdout, stderr = process.communicate(input=script, timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            rospy.logerr("SSH command timed out while sending '%s'", name)
            return
        finally:
            with self._process_lock:
                if self._current_process is process:
                    self._current_process = None

        if process.returncode != 0:
            if process.returncode in (-15, 143):
                rospy.loginfo("SSH command canceled for '%s'", name)
            else:
                rospy.logerr("SSH command failed for '%s': %s", name, stderr.strip())
            return

        if stderr.strip():
            rospy.logwarn("SSH command stderr for '%s': %s", name, stderr.strip())
        if stdout.strip():
            rospy.logdebug("SSH command stdout for '%s': %s", name, stdout.strip())

    def _remote_script(self, linear, angular, duration):
        master_line = ""
        if self.remote_ros_master_uri:
            master_line = "export ROS_MASTER_URI=%s\n" % self.remote_ros_master_uri

        return """\
set -e
%s
%spython - <<'PY'
import rospy
import time
from geometry_msgs.msg import Twist

linear = %.6f
angular = %.6f
duration = %.3f

rospy.init_node("ssh_voice_cmd_vel", anonymous=True, disable_signals=True)
pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
time.sleep(0.8)

msg = Twist()
msg.linear.x = linear
msg.angular.z = angular

start = time.time()
while not rospy.is_shutdown() and time.time() - start < duration:
    pub.publish(msg)
    time.sleep(0.1)

msg.linear.x = 0.0
msg.angular.z = 0.0
for _ in range(5):
    pub.publish(msg)
    time.sleep(0.1)
PY
""" % (self.remote_ros_setup, master_line, linear, angular, duration)


if __name__ == "__main__":
    rospy.init_node("ssh_voice_control")
    SshVoiceControl()
    rospy.spin()
