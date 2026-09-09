import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from parser import ParseError, parse_record


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ai_match_from_expected(record: dict, expected: dict) -> dict:
    answers = []
    for response in record["responses"]:
        qid = str(response["question_id"])
        key = f'{response["phase"]}.{qid if qid.startswith("Q") else "Q" + qid}.{response["type"]}'
        answers.append({"question_id": qid, "value": expected["values"][key]})
    return {"record_id": record["record_id"], "answers": answers}


class ParserTest(unittest.TestCase):
    def test_matches_reviewed_ontology_inputs(self):
        datasets = [
            (ROOT / "data" / "survey-responses.jsonl", ROOT / "data" / "ontology-input.jsonl"),
            (ROOT / "pdf_test" / "survey-responses.jsonl", ROOT / "pdf_test" / "ontology-input.jsonl"),
        ]
        for source_path, expected_path in datasets:
            source, expected = rows(source_path), rows(expected_path)
            actual = [parse_record(record, ai_match_from_expected(record, answer)) for record, answer in zip(source, expected)]
            self.assertEqual(actual, expected)

    def test_rejects_unknown_option_id(self):
        record = rows(ROOT / "data" / "survey-responses.jsonl")[0]
        expected = rows(ROOT / "data" / "ontology-input.jsonl")[0]
        match = ai_match_from_expected(record, expected)
        match["answers"][0]["value"] = [999]
        with self.assertRaises(ParseError):
            parse_record(record, match)

    def test_rejects_missing_ai_answer(self):
        record = rows(ROOT / "data" / "survey-responses.jsonl")[0]
        expected = rows(ROOT / "data" / "ontology-input.jsonl")[0]
        match = ai_match_from_expected(record, expected)
        match["answers"].pop()
        with self.assertRaises(ParseError):
            parse_record(record, match)

    def test_rejects_out_of_range_scale(self):
        record = rows(ROOT / "data" / "survey-responses.jsonl")[0]
        expected = rows(ROOT / "data" / "ontology-input.jsonl")[0]
        match = ai_match_from_expected(record, expected)
        match["answers"][1]["value"] = 11
        with self.assertRaises(ParseError):
            parse_record(record, match)

    def test_source_requires_transcript(self):
        record = deepcopy(rows(ROOT / "data" / "survey-responses.jsonl")[0])
        del record["responses"][0]["transcript"]
        expected = rows(ROOT / "data" / "ontology-input.jsonl")[0]
        with self.assertRaises(ParseError):
            parse_record(record, ai_match_from_expected(record, expected))


if __name__ == "__main__":
    unittest.main()
