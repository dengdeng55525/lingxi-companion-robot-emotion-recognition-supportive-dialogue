from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

import parent_dialogue_logger as dialogue_logger


class ParentDialogueLoggerTests(unittest.TestCase):
    def test_make_dialogue_record_normalizes_role_and_text(self):
        record = dialogue_logger.make_dialogue_record(
            "assistant",
            "  你好呀  ",
            timestamp=1783349000.0,
            source_topic="/ai_reply",
        )

        self.assertIsNotNone(record)
        self.assertEqual(record["role"], "robot")
        self.assertEqual(record["text"], "你好呀")
        self.assertEqual(record["source_topic"], "/ai_reply")
        self.assertIn("iso_time", record)

    def test_jsonl_logger_appends_records(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "dialogue.jsonl"
            logger = dialogue_logger.JsonlDialogueLogger(log_path, clock=lambda: 1783349000.0)

            first = logger.append("child", "灵犀你好", source_topic="/iat_text")
            second = logger.append("robot", "你好呀", source_topic="/ai_reply")
            lines = log_path.read_text(encoding="utf-8").splitlines()

        self.assertTrue(first)
        self.assertTrue(second)
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["role"], "child")
        self.assertEqual(json.loads(lines[1])["role"], "robot")

    def test_empty_text_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "dialogue.jsonl"
            logger = dialogue_logger.JsonlDialogueLogger(log_path)

            written = logger.append("child", "   ")

        self.assertFalse(written)
        self.assertFalse(log_path.exists())

    def test_text_can_be_truncated(self):
        record = dialogue_logger.make_dialogue_record(
            "child",
            "abcdef",
            timestamp=1783349000.0,
            max_text_length=3,
        )

        self.assertEqual(record["text"], "abc")


if __name__ == "__main__":
    unittest.main()
