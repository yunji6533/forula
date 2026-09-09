"""Interactive Stage 1 speech-text matcher and Ontology pair check."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "fine_tuning"))
from parser import parse_record
from build_semantic_matcher_dataset import SYSTEM

BASE_MODEL = "google/gemma-4-E4B-it"
DEFAULT_ADAPTER = Path.home() / "Downloads" / "job_d9d5ac53-adapter" / "adapter"
DEFAULT_OUTPUT = ROOT / "pdf_test" / "stage1-live-result.json"
OPTIONS = [
    {"option_id": 1, "label": "숨이 답답해짐"},
    {"option_id": 2, "label": "가슴이 두근거림"},
    {"option_id": 3, "label": "손에 땀이 남"},
    {"option_id": 4, "label": "목소리가 떨림"},
    {"option_id": 5, "label": "빨리 나가고 싶어짐"},
    {"option_id": 6, "label": "식은땀이 남"},
    {"option_id": 7, "label": "호흡이 빨라질 것"},
    {"option_id": 8, "label": "별 다른 일은 일어나지 않음"},
    {"option_id": 9, "label": "잘 모르겠다"},
    {"option_id": 10, "label": "기타"},
]
DEMO_ANSWERS = [
    "숨이 막히는 건 아니고 심장이 두근거리고 손바닥에 땀이 날 것 같아요",
    "처음에는 3점 같았는데 지금은 6점 정도예요",
    "두 점이요",
]


def ask_answers(demo: bool) -> list[str]:
    prompts = [
        "주변을 둘러보세요. 어떤 일이 일어날 것 같은가요?",
        "현재 불안이나 불편감은 1~10 중 어느 정도인가요?",
        "체험 후 불안이나 불편감은 1~10 중 어느 정도인가요?",
    ]
    print("\n[Stage 1: 넓은 강당]")
    print("복수 선택 보기:")
    for option in OPTIONS:
        print(f"  {option['option_id']}. {option['label']}")
    answers = []
    for prompt, preset in zip(prompts, DEMO_ANSWERS):
        print(f"\n{prompt}")
        answer = preset if demo else input("> ").strip()
        if demo:
            print(f"> {answer}")
        if not answer:
            raise ValueError("답변은 비어 있을 수 없습니다.")
        answers.append(answer)
    return answers


def build_record(answers: list[str]) -> dict:
    now = datetime.now().astimezone()
    return {
        "record_id": f"stage1-live-{now:%Y%m%d-%H%M%S}",
        "session_id": f"console-{now:%Y%m%d-%H%M%S}",
        "survey_id": "claustrophobia-vr",
        "survey_version": "pdf-test-1",
        "phobia_type": "claustrophobia",
        "stage": 1,
        "scenario": "넓은 강당",
        "responses": [
            {"question_id": "8", "phase": "PRE", "type": "MULTI_SELECT", "question": "주변을 둘러보세요. 어떤 일이 일어날 것 같은가요?", "options": OPTIONS, "transcript": answers[0]},
            {"question_id": "9", "phase": "PRE", "type": "SCALE", "question": "현재 불안이나 불편감은 1~10 중 어느 정도인가요?", "scale": {"min": 1, "max": 10}, "transcript": answers[1]},
            {"question_id": "14", "phase": "POST", "type": "SCALE", "question": "체험 후 불안이나 불편감은 1~10 중 어느 정도인가요?", "scale": {"min": 1, "max": 10}, "transcript": answers[2]},
        ],
        "created_at": now.isoformat(timespec="seconds"),
    }


def run_model(record: dict, adapter: Path) -> tuple[dict, float]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"어댑터를 찾을 수 없습니다: {adapter}")
    model_input = {key: record[key] for key in ("record_id", "survey_id", "survey_version", "phobia_type", "stage", "responses")}
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(model_input, ensure_ascii=False, separators=(",", ":"))},
    ]
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    if torch.cuda.is_available():
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=quantization, device_map="auto", local_files_only=True,
        )
        print(f"GPU 4-bit 실행: {torch.cuda.get_device_name(0)}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=torch.bfloat16, device_map={"": "cpu"}, local_files_only=True,
        )
        print("CUDA를 사용할 수 없어 CPU로 실행합니다.")
    model = PeftModel.from_pretrained(model, adapter)
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=192, do_sample=False)
    text = tokenizer.decode(output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"모델이 JSON을 반환하지 않았습니다: {text}")
    return json.loads(text[start:end + 1]), time.perf_counter() - started


def selected_labels(record: dict, match: dict) -> list[dict]:
    answer_by_id = {str(item["question_id"]): item["value"] for item in match["answers"]}
    result = []
    for response in record["responses"]:
        value = answer_by_id[str(response["question_id"])]
        if "options" in response:
            labels = {item["option_id"]: item["label"] for item in response["options"]}
            selected = [labels[item] for item in value] if isinstance(value, list) else labels[value]
        else:
            selected = value
        result.append({"question_id": response["question_id"], "transcript": response["transcript"], "selected": selected})
    return result


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    cli.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    cli.add_argument("--demo", action="store_true", help="다양한 예시 답변으로 자동 실행")
    args = cli.parse_args()
    record = build_record(ask_answers(args.demo))
    print("\n모델을 불러와 답변을 매칭하고 있습니다...")
    match, seconds = run_model(record, args.adapter)
    ontology = parse_record(record, match)
    result = {"source": record, "ai_match": match, "resolved_answers": selected_labels(record, match), "ontology_input": ontology, "inference_seconds": round(seconds, 2)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[AI 보기 매칭 결과]")
    for item in result["resolved_answers"]:
        print(f"Q{item['question_id']} | {item['transcript']} -> {item['selected']}")
    print("\n[Ontology key-value]")
    print(json.dumps(ontology["values"], ensure_ascii=False, indent=2))
    print(f"\n저장 완료: {args.output}")


if __name__ == "__main__":
    main()
