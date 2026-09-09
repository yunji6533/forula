import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage1_console import json_object


class JsonOutputTest(unittest.TestCase):
    def test_extracts_json_object_from_extra_text(self):
        self.assertEqual(json_object('결과: {"record_id":"r1","answers":[]} 완료'), {"record_id": "r1", "answers": []})

    def test_rejects_malformed_json(self):
        with self.assertRaises(ValueError):
            json_object('{"question_id":"8" "value":[3]}')


if __name__ == "__main__":
    unittest.main()
