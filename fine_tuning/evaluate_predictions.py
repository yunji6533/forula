"""Compare model output JSONL with the semantic holdout answers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("predictions", type=Path, help="one model JSON object per holdout row")
    cli.add_argument("--holdout", type=Path, default=Path(__file__).with_name("semantic-matcher-holdout.jsonl"))
    args = cli.parse_args()

    expected = [json.loads(line)["expected"] for line in args.holdout.read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_predictions = [line for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    valid = exact = correct_answers = total_answers = 0

    for index, answer in enumerate(expected):
        total_answers += len(answer["answers"])
        if index >= len(raw_predictions):
            continue
        try:
            prediction = json.loads(raw_predictions[index])
        except json.JSONDecodeError:
            continue
        valid += 1
        if prediction == answer:
            exact += 1
        expected_by_id = {str(item["question_id"]): item["value"] for item in answer["answers"]}
        predicted_answers = prediction.get("answers", []) if isinstance(prediction, dict) else []
        if isinstance(predicted_answers, list):
            for item in predicted_answers:
                if isinstance(item, dict) and expected_by_id.get(str(item.get("question_id"))) == item.get("value"):
                    correct_answers += 1

    total = len(expected)
    print(json.dumps({
        "total_records": total,
        "json_validity_rate": round(valid / total * 100, 2),
        "exact_match_rate": round(exact / total * 100, 2),
        "question_value_accuracy": round(correct_answers / total_answers * 100, 2),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
