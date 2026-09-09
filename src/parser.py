"""Validate AI option matches and build Ontology key-value input."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PARSER_VERSION = "semantic-hybrid-1.0.0"
ONTOLOGY_SCHEMA_VERSION = "0.3.0"
CHOICE_TYPES = {"SINGLE_CHOICE", "CONFIRMATION"}


class ParseError(ValueError):
    pass


def _question_key(response: dict[str, Any]) -> str:
    qid = str(response["question_id"])
    return f'{response["phase"]}.{qid if qid.startswith("Q") else "Q" + qid}.{response["type"]}'


def validate_record(record: dict[str, Any]) -> None:
    required = {
        "record_id", "session_id", "survey_id", "survey_version",
        "phobia_type", "stage", "responses", "created_at",
    }
    if missing := required - record.keys():
        raise ParseError(f"missing required fields: {sorted(missing)}")
    if not isinstance(record["responses"], list) or not record["responses"]:
        raise ParseError("responses must be a non-empty list")

    seen: set[str] = set()
    for response in record["responses"]:
        base = {"question_id", "phase", "type", "question", "transcript"}
        if missing := base - response.keys():
            raise ParseError(f"response missing fields: {sorted(missing)}")
        qid = str(response["question_id"])
        if qid in seen:
            raise ParseError(f"duplicate question_id: {qid}")
        seen.add(qid)
        if response["phase"] not in {"PRE", "POST"}:
            raise ParseError(f"invalid phase: {qid}")
        if not isinstance(response["transcript"], str) or not response["transcript"].strip():
            raise ParseError(f"empty transcript: {qid}")

        response_type = response["type"]
        if response_type == "SCALE":
            scale = response.get("scale")
            if not isinstance(scale, dict) or not {"min", "max"} <= scale.keys():
                raise ParseError(f"missing scale range: {qid}")
            if scale["min"] > scale["max"]:
                raise ParseError(f"invalid scale range: {qid}")
        elif response_type == "MULTI_SELECT" or response_type in CHOICE_TYPES:
            options = response.get("options")
            if not isinstance(options, list) or not options:
                raise ParseError(f"missing options: {qid}")
            option_ids = [option.get("option_id") for option in options]
            if any(isinstance(value, bool) or not isinstance(value, int) for value in option_ids):
                raise ParseError(f"invalid option_id: {qid}")
            if len(option_ids) != len(set(option_ids)):
                raise ParseError(f"duplicate option_id: {qid}")
            if any(not isinstance(option.get("label"), str) or not option["label"].strip() for option in options):
                raise ParseError(f"invalid option label: {qid}")
        else:
            raise ParseError(f"unsupported response type: {response_type}")


def validate_ai_match(record: dict[str, Any], ai_match: dict[str, Any]) -> dict[str, Any]:
    if ai_match.get("record_id") != record["record_id"]:
        raise ParseError("AI result record_id does not match source record")
    answers = ai_match.get("answers")
    if not isinstance(answers, list):
        raise ParseError("AI result answers must be a list")

    answer_by_id: dict[str, Any] = {}
    for answer in answers:
        if not isinstance(answer, dict) or set(answer) != {"question_id", "value"}:
            raise ParseError("each AI answer must contain only question_id and value")
        qid = str(answer["question_id"])
        if qid in answer_by_id:
            raise ParseError(f"duplicate AI answer: {qid}")
        answer_by_id[qid] = answer["value"]

    expected_ids = {str(response["question_id"]) for response in record["responses"]}
    if set(answer_by_id) != expected_ids:
        missing = sorted(expected_ids - set(answer_by_id))
        extra = sorted(set(answer_by_id) - expected_ids)
        raise ParseError(f"AI answer coverage mismatch; missing={missing}, extra={extra}")

    for response in record["responses"]:
        qid = str(response["question_id"])
        value = answer_by_id[qid]
        if response["type"] == "SCALE":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ParseError(f"invalid SCALE value: {qid}")
            if not response["scale"]["min"] <= value <= response["scale"]["max"]:
                raise ParseError(f"SCALE value out of range: {qid}")
            continue

        valid_ids = {option["option_id"] for option in response["options"]}
        if response["type"] == "MULTI_SELECT":
            if not isinstance(value, list) or not value:
                raise ParseError(f"invalid MULTI_SELECT value: {qid}")
            if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
                raise ParseError(f"invalid MULTI_SELECT option_id: {qid}")
            if len(value) != len(set(value)) or not set(value) <= valid_ids:
                raise ParseError(f"unknown or duplicate option_id: {qid}")
        elif isinstance(value, bool) or not isinstance(value, int) or value not in valid_ids:
            raise ParseError(f"unknown option_id: {qid}")
    return answer_by_id


def parse_record(record: dict[str, Any], ai_match: dict[str, Any]) -> dict[str, Any]:
    validate_record(record)
    answers = validate_ai_match(record, ai_match)
    values = {
        _question_key(response): answers[str(response["question_id"])]
        for response in record["responses"]
    }
    if "stay_time_seconds" in record:
        values["STAGE.STAY_TIME_SECONDS"] = record["stay_time_seconds"]
    return {
        "ontology_schema_version": ONTOLOGY_SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "context": {
            "source_record_id": record["record_id"],
            "session_id": record["session_id"],
            "survey_id": record["survey_id"],
            "survey_version": record["survey_version"],
            "phobia_type": record["phobia_type"],
            "stage": record["stage"],
        },
        "values": values,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def parse_jsonl(source: Path, matches: Path, target: Path) -> None:
    records = _read_jsonl(source)
    match_by_id = {item.get("record_id"): item for item in _read_jsonl(matches)}
    with target.open("w", encoding="utf-8") as stream:
        for record in records:
            match = match_by_id.get(record.get("record_id"))
            if match is None:
                raise ParseError(f'missing AI match for {record.get("record_id")}')
            result = parse_record(record, match)
            stream.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("input", type=Path, help="speech survey JSONL")
    cli.add_argument("matches", type=Path, help="AI match JSONL")
    cli.add_argument("output", type=Path, help="Ontology input JSONL")
    args = cli.parse_args()
    parse_jsonl(args.input, args.matches, args.output)
