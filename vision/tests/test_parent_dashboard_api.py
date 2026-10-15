from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

import parent_dashboard_api as parent_api


def make_payload(frame_index=1, emotion="happy", confidence=0.82, objects=None):
    return {
        "status": "ok",
        "valid": True,
        "timestamp": 1783348000.0 + frame_index,
        "frame_index": frame_index,
        "face_count": 1,
        "main_person": {
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": {
                emotion: confidence,
                "neutral": max(0.0, 1.0 - confidence),
            },
        },
        "objects": objects or [],
    }


class ParentDashboardApiTests(unittest.TestCase):
    def test_record_vision_payload_stores_sample_but_not_learned_object(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "parent.sqlite3"
            payload = make_payload(
                objects=[
                    {"class": "book", "confidence": 0.72},
                    {"class": "toy car", "confidence": 0.66},
                ]
            )

            record = parent_api.record_vision_payload(db_path, payload, now=1783349000.0)
            dashboard = parent_api.build_dashboard_payload(
                db_path,
                Path(tmp_dir) / "dialogue.jsonl",
                Path(tmp_dir) / "learning.jsonl",
                date_key=parent_api.local_date_key(1783349000.0),
                mock_on_empty=False,
            )

        self.assertEqual(record["dominant_emotion"], "happy")
        self.assertEqual(record["object_count"], 2)
        self.assertFalse(dashboard["mock"])
        self.assertEqual(dashboard["summary"]["sample_count"], 1)
        self.assertEqual(dashboard["summary"]["current_emotion"], "happy")
        self.assertEqual(dashboard["emotions"][0]["object_count"], 2)
        self.assertEqual(dashboard["summary"]["learned_object_count"], 0)
        self.assertEqual(dashboard["objects"], [])

    def test_dangerous_object_classifier_ignores_safe_objects(self):
        dangerous = parent_api.extract_dangerous_objects(
            [
                {"class": "book", "confidence": 0.91},
                {"class": "knife", "confidence": 0.87},
                {"class_name": "hammer", "confidence": 0.66},
                {"class": "knife", "confidence": 0.80},
            ]
        )

        self.assertEqual([item["display_name"] for item in dangerous], ["刀具", "锤子"])

    def test_dangerous_objects_are_recorded_from_vision_samples(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "parent.sqlite3"
            dialogue_log = Path(tmp_dir) / "dialogue.jsonl"
            learning_log = Path(tmp_dir) / "learning.jsonl"
            now = 1783349000.0
            parent_api.record_vision_payload(
                db_path,
                make_payload(objects=[{"class": "knife", "confidence": 0.87}, {"class": "book", "confidence": 0.72}]),
                now=now,
            )
            parent_api.record_vision_payload(
                db_path,
                make_payload(frame_index=2, objects=[{"class_name": "hammer", "confidence": 0.66}]),
                now=now + 10,
            )

            dashboard = parent_api.build_dashboard_payload(
                db_path,
                dialogue_log,
                learning_log,
                date_key=parent_api.local_date_key(now),
                mock_on_empty=False,
            )

        self.assertEqual(dashboard["summary"]["dangerous_object_count"], 2)
        self.assertEqual(len(dashboard["dangerous_objects"]), 2)
        display_names = {item["display_name"] for item in dashboard["dangerous_objects"]}
        self.assertEqual(display_names, {"刀具", "锤子"})
        self.assertEqual(dashboard["summary"]["learned_object_count"], 0)

    def test_learning_events_create_deduplicated_learned_objects(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "parent.sqlite3"
            learning_log = Path(tmp_dir) / "learning.jsonl"
            now = 1783349000.0
            learning_log.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "event_id": "event-1",
                                "timestamp": now,
                                "intent": "object_learning_query",
                                "user_text": "这是什么？",
                                "reply": "这是一本书。",
                                "objects": [{"class_name": "book", "confidence": 0.51}],
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "event_id": "event-2",
                                "timestamp": now + 10,
                                "intent": "object_learning_query",
                                "user_text": "眼前是什么？",
                                "reply": "还是一本书。",
                                "objects": [{"class_name": "book", "confidence": 0.91}],
                            },
                            ensure_ascii=False,
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            inserted_first = parent_api.ingest_learning_events(db_path, learning_log, parent_api.local_date_key(now))
            inserted_second = parent_api.ingest_learning_events(db_path, learning_log, parent_api.local_date_key(now))
            objects = parent_api.load_learned_objects(db_path, parent_api.local_date_key(now))

        self.assertEqual(inserted_first, 2)
        self.assertEqual(inserted_second, 0)
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["class_name"], "book")
        self.assertEqual(objects[0]["count"], 2)
        self.assertAlmostEqual(objects[0]["max_confidence"], 0.91)

    def test_sample_once_uses_injected_fetcher(self):
        calls = []

        def fake_fetcher(url, timeout):
            calls.append((url, timeout))
            return make_payload(objects=[{"class": "ball", "confidence": 0.8}])

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "parent.sqlite3"
            record = parent_api.sample_once(
                db_path,
                "http://example.test/vision/latest",
                fetcher=fake_fetcher,
                fetch_timeout=0.25,
                now=1783349000.0,
            )

        self.assertEqual(calls, [("http://example.test/vision/latest", 0.25)])
        self.assertEqual(record["objects"][0]["class_name"], "ball")

    def test_loopback_url_is_detected_for_proxy_bypass(self):
        self.assertTrue(parent_api.is_loopback_url("http://127.0.0.1:8090/vision/latest"))
        self.assertTrue(parent_api.is_loopback_url("http://localhost:8090/vision/latest"))
        self.assertFalse(parent_api.is_loopback_url("http://example.test/vision/latest"))

    def test_dialogue_log_is_loaded_and_normalized(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "dialogue.jsonl"
            ts = 1783349000.0
            log_path.write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": ts, "role": "user", "text": "灵犀你好"}, ensure_ascii=False),
                        json.dumps({"timestamp": ts + 1, "role": "assistant", "text": "你好呀"}, ensure_ascii=False),
                        "not-json",
                        json.dumps({"timestamp": ts + 2, "role": "assistant", "text": ""}, ensure_ascii=False),
                    ]
                ),
                encoding="utf-8",
            )

            entries = parent_api.load_dialogue_entries(log_path, parent_api.local_date_key(ts))

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["role"], "child")
        self.assertEqual(entries[1]["role"], "robot")

    def test_empty_dashboard_can_return_mock_payload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = parent_api.build_dashboard_payload(
                Path(tmp_dir) / "missing.sqlite3",
                Path(tmp_dir) / "missing.jsonl",
                Path(tmp_dir) / "missing_learning.jsonl",
                date_key="2026-07-07",
                mock_on_empty=True,
                now=1783349000.0,
            )

        self.assertTrue(payload["mock"])
        self.assertGreater(len(payload["emotions"]), 0)
        self.assertGreater(len(payload["dialogue"]), 0)
        self.assertGreater(len(payload["objects"]), 0)

    def test_trend_summary_is_descriptive_not_diagnostic(self):
        samples = [
            {"emotion": "sad"},
            {"emotion": "sad"},
            {"emotion": "fear"},
            {"emotion": "angry"},
            {"emotion": "sad"},
        ]

        text = parent_api.build_trend_summary(samples)

        self.assertIn("建议家长", text)
        self.assertNotIn("心理健康评分", text)
        self.assertNotIn("诊断", text)
        self.assertNotIn("医学结论", text)

    def test_trend_summary_prioritizes_attention_emotions_over_neutral(self):
        samples = [
            {"emotion": "neutral"},
            {"emotion": "neutral"},
            {"emotion": "nature"},
            {"emotion": "happy"},
            {"emotion": "angry"},
            {"emotion": "neutral"},
        ]

        text = parent_api.build_trend_summary(samples)

        self.assertIn("生气", text)
        self.assertIn("优先", text)
        self.assertNotIn("整体比较平稳", text)

    def test_trend_summary_mentions_positive_before_neutral(self):
        samples = [
            {"emotion": "neutral"},
            {"emotion": "nature"},
            {"emotion": "happy"},
            {"emotion": "neutral"},
        ]

        text = parent_api.build_trend_summary(samples)

        self.assertIn("开心", text)
        self.assertNotIn("整体比较平稳", text)


if __name__ == "__main__":
    unittest.main()
