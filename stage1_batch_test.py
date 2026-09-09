"""Run repeatable Stage 1 semantic matching cases and write an accuracy report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage1_console import DEFAULT_ADAPTER, PROMPT_VERSION, ROOT, build_record, load_model, run_model

DEFAULT_CASES = ROOT / "pdf_test" / "stage1-test-cases.jsonl"
DEFAULT_REPORT = ROOT / "pdf_test" / "stage1-batch-report.json"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalized(value):
    return sorted(value) if isinstance(value, list) else value


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    cli.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    cli.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = cli.parse_args()

    cases = rows(args.cases)
    print(f"모델을 한 번만 불러와 {len(cases)}개 Stage 테스트를 실행합니다...")
    runtime = load_model(args.adapter)
    details = []
    correct_questions = exact_cases = 0
    error_cases = 0
    total_seconds = 0.0
    for index, case in enumerate(cases, 1):
        record = build_record(case["answers"])
        record["record_id"] = f"stage1-batch-{case['case_id']}"
        try:
            match, seconds = run_model(record, args.adapter, runtime)
            actual = {str(item["question_id"]): item["value"] for item in match["answers"]}
            error = None
        except Exception as exc:
            seconds, actual, error = 0.0, {}, str(exc)
            error_cases += 1
        checks = {qid: normalized(actual.get(qid)) == normalized(expected) for qid, expected in case["expected"].items()}
        correct_questions += sum(checks.values())
        exact = all(checks.values())
        exact_cases += exact
        total_seconds += seconds
        details.append({"case_id": case["case_id"], "answers": case["answers"], "expected": case["expected"], "actual": actual, "correct": checks, "exact": exact, "error": error, "seconds": round(seconds, 2)})
        print(f"[{index}/{len(cases)}] {case['case_id']}: {'PASS' if exact else 'FAIL'}", flush=True)

    total_questions = sum(len(case["expected"]) for case in cases)
    report = {
        "summary": {
            "prompt_version": PROMPT_VERSION,
            "total_cases": len(cases),
            "exact_cases": exact_cases,
            "case_exact_match_rate": round(exact_cases / len(cases) * 100, 2),
            "total_questions": total_questions,
            "correct_questions": correct_questions,
            "question_accuracy": round(correct_questions / total_questions * 100, 2),
            "error_cases": error_cases,
            "total_inference_seconds": round(total_seconds, 2),
        },
        "details": details,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"보고서 저장: {args.report}")


if __name__ == "__main__":
    main()
