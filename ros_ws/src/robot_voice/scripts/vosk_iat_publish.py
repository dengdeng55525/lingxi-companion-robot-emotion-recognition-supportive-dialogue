#!/usr/bin/env python3
import json
import os
import queue
import sys

import rospy
from std_msgs.msg import String


def _load_modules():
    try:
        import sounddevice as sd
        import vosk
    except ImportError as exc:
        rospy.logfatal(
            "Missing Python dependency: %s. Install with "
            "`python3 -m pip install vosk sounddevice`.", exc.name
        )
        raise
    return sd, vosk


def _select_device(sd, device_name):
    if not device_name:
        return None

    devices = sd.query_devices()
    for index, device in enumerate(devices):
        if device_name.lower() in device["name"].lower():
            return index

    raise RuntimeError("Audio input device containing '%s' was not found" % device_name)


def main():
    rospy.init_node("vosk_iat_publish")

    sd, vosk = _load_modules()

    model_path = rospy.get_param("~model_path", "")
    sample_rate = int(rospy.get_param("~sample_rate", 16000))
    block_size = int(rospy.get_param("~block_size", 8000))
    device_name = rospy.get_param("~device_name", "")
    publish_partials = bool(rospy.get_param("~publish_partials", False))

    if not model_path:
        rospy.logfatal("Parameter ~model_path is required.")
        return 1
    if not os.path.isdir(model_path):
        rospy.logfatal("Vosk model directory does not exist: %s", model_path)
        return 1

    audio_queue = queue.Queue()
    publisher = rospy.Publisher("iat_text", String, queue_size=10)

    model = vosk.Model(model_path)
    recognizer = vosk.KaldiRecognizer(model, sample_rate)
    device = _select_device(sd, device_name)

    def audio_callback(indata, frames, time_info, status):
        if status:
            rospy.logwarn_throttle(5.0, "Audio input status: %s", status)
        audio_queue.put(bytes(indata))

    rospy.loginfo("Vosk IAT publisher is listening at %s Hz.", sample_rate)
    with sd.RawInputStream(
        samplerate=sample_rate,
        blocksize=block_size,
        device=device,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        while not rospy.is_shutdown():
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").replace(" ", "")
                if text:
                    rospy.loginfo("Recognized: %s", text)
                    publisher.publish(String(text))
            elif publish_partials:
                result = json.loads(recognizer.PartialResult())
                partial = result.get("partial", "").replace(" ", "")
                if partial:
                    publisher.publish(String(partial))

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
