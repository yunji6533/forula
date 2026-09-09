"""Generate PDF-based SFT and holdout data for speech option matching."""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN_OUTPUT = ROOT / "semantic-matcher-sft.jsonl"
HOLDOUT_OUTPUT = ROOT / "semantic-matcher-holdout.jsonl"

SYSTEM = """너는 VR 설문 음성 응답 매칭기다. 각 transcript의 의미를 현재 입력에 제공된 보기 또는 척도에만 매칭한다.
MULTI_SELECT는 명시되거나 분명히 함축된 보기를 모두 선택하고 부정된 증상은 제외한다. 목록에 없는 증상은 '기타', 판단할 수 없다는 답은 '잘 모르겠다'를 선택한다.
SCALE은 사용자가 최종적으로 말한 의도값을 허용 범위 안에서 추출한다. SINGLE_CHOICE와 CONFIRMATION은 보기 하나만 선택한다.
진단이나 공포증 점수를 계산하지 않는다. 설명이나 마크다운 없이 {\"record_id\":\"...\",\"answers\":[{\"question_id\":\"...\",\"value\":값}]} JSON 객체 하나만 출력한다.
MULTI_SELECT value는 option_id 정수 배열, 나머지 선택형은 option_id 정수, SCALE은 숫자다."""

SYMPTOMS = [
    (1, "숨이 답답해짐", ["숨이 답답해요", "숨이 막힐 것 같아요", "가슴이 꽉 막힌 느낌이에요", "숨쉬기 힘들 것 같아요"], ["공기가 잘 안 들어오는 기분이에요", "숨통이 조여 오는 느낌이에요"]),
    (2, "가슴이 두근거림", ["가슴이 두근거려요", "심장이 쿵쾅거릴 것 같아요", "가슴이 벌렁거려요", "맥박이 세게 뛰어요"], ["심장 소리가 크게 느껴져요", "가슴이 빠르게 뛸 것 같아요"]),
    (3, "손에 땀이 남", ["손에 땀이 날 것 같아요", "손바닥이 축축해져요", "손에서 땀이 나요", "손이 땀으로 젖을 것 같아요"], ["손바닥이 미끄러울 만큼 땀이 나요", "손에 땀이 배는 느낌이에요"]),
    (4, "목소리가 떨림", ["목소리가 떨릴 것 같아요", "말할 때 음성이 떨려요", "말이 떨려서 잘 안 나와요", "목소리가 흔들려요"], ["말소리가 덜덜 떨릴 것 같아요", "입을 열면 목소리가 흔들릴 것 같아요"]),
    (5, "빨리 나가고 싶어짐", ["빨리 나가고 싶어요", "여기서 당장 벗어나고 싶어요", "문 열고 나가고 싶어요", "그만하고 싶어요"], ["더 못 있겠고 탈출하고 싶어요", "여기서 얼른 빠져나가고 싶어요"]),
    (6, "식은땀이 남", ["식은땀이 날 것 같아요", "온몸에 식은땀이 나요", "등에서 차가운 땀이 나요", "몸에 식은땀이 배어요"], ["갑자기 차가운 땀이 흐를 것 같아요", "전신에 식은땀이 날 것 같아요"]),
    (7, "호흡이 빨라질 것", ["호흡이 빨라질 것 같아요", "숨을 가쁘게 쉴 것 같아요", "헐떡거리게 될 것 같아요", "숨이 점점 빨라져요"], ["과호흡처럼 숨이 가빠질 것 같아요", "호흡 속도가 확 빨라질 것 같아요"]),
    (8, "별 다른 일은 일어나지 않음", ["아무 일도 없을 것 같아요", "별일 없을 것 같아요", "그냥 괜찮을 것 같아요", "특별한 증상은 없어요"], ["평소와 다르지 않을 것 같아요", "문제없이 지나갈 것 같아요"]),
    (9, "잘 모르겠다", ["잘 모르겠어요", "어떻게 될지 모르겠는데요", "예상하기 어려워요", "모르겠습니다"], ["무슨 일이 생길지 감이 안 와요", "딱히 예상이 안 돼요"]),
    (10, "기타", ["머리가 아플 것 같아요", "속이 안 좋아요", "어지러울 것 같아요", "그냥 느낌이 싫어요"], ["토할 것 같은 기분이에요", "다리에 힘이 풀릴 것 같아요"]),
]

CONFIRMATION = [
    (1, "그렇다", ["네 실제로 그랬어요", "맞아요 그런 일이 생겼어요", "예 일어났습니다"], ["네 예상한 그대로였어요", "맞습니다 실제로 발생했어요"]),
    (2, "아니다", ["아니요 일어나지 않았어요", "그런 일은 없었어요", "예상과 달리 괜찮았어요"], ["실제로는 생기지 않았어요", "그렇지 않았습니다"]),
]

OUTCOMES = [
    (1, "예상했던 일이 일어나지 않음", ["생각했던 일은 안 일어났어요", "아무 일도 발생하지 않았어요", "예상과 달리 괜찮았어요"], ["걱정한 상황은 실제로 없었어요", "결국 별일 없이 끝났어요"]),
    (2, "피해를 입히는 중대한 일이 벌어짐", ["실제로 위험한 일이 생겼어요", "피해를 입을 만한 일이 벌어졌어요", "큰 문제가 발생했어요"], ["안전에 문제가 생길 정도의 일이 있었어요", "실제로 심각한 상황이 벌어졌어요"]),
]

STAGES = {
    1: [("MULTI_SELECT", "8", "PRE"), ("SCALE", "9", "PRE"), ("SCALE", "14", "POST")],
    2: [("MULTI_SELECT", "18", "PRE"), ("SCALE", "19", "PRE"), ("CONFIRMATION", "23", "POST"), ("SCALE", "24", "POST")],
    3: [("MULTI_SELECT", "28", "PRE"), ("SCALE", "29", "PRE"), ("CONFIRMATION", "34", "POST"), ("SCALE", "35", "POST")],
    4: [("MULTI_SELECT", "39", "PRE"), ("SCALE", "40", "PRE"), ("CONFIRMATION", "44", "POST"), ("SCALE", "45", "POST")],
    5: [("MULTI_SELECT", "49", "PRE"), ("SCALE", "50", "PRE"), ("CONFIRMATION", "54", "POST"), ("SCALE", "55", "POST")],
    6: [("MULTI_SELECT", "59", "PRE"), ("SCALE", "60", "PRE"), ("CONFIRMATION", "65", "POST"), ("SCALE", "66", "POST")],
    7: [("MULTI_SELECT", "70", "PRE"), ("SCALE", "71", "PRE"), ("CONFIRMATION", "76", "POST"), ("SCALE", "77", "POST")],
    8: [("MULTI_SELECT", "81", "PRE"), ("SCALE", "82", "PRE"), ("SINGLE_CHOICE", "88", "POST"), ("SCALE", "89", "POST")],
    9: [("MULTI_SELECT", "93", "PRE"), ("SCALE", "94", "PRE"), ("SINGLE_CHOICE", "101", "POST"), ("SCALE", "102", "POST")],
}
KOREAN_NUMBERS = {1: "한", 2: "두", 3: "세", 4: "네", 5: "다섯", 6: "여섯", 7: "일곱", 8: "여덟", 9: "아홉", 10: "열"}


def _options(bank: list[tuple], rng: random.Random) -> list[dict]:
    options = [{"option_id": item[0], "label": item[1]} for item in bank]
    rng.shuffle(options)
    return options


def _utterance(item: tuple, split: str, rng: random.Random) -> str:
    phrases = item[2] if split == "train" else item[3]
    prefix = rng.choice(["", "음... ", "아마 ", "저는 "])
    return prefix + rng.choice(phrases)


def _multi(qid: str, phase: str, split: str, rng: random.Random) -> tuple[dict, list[int]]:
    special_ids = {8, 9, 10}
    if rng.random() < 0.25:
        selected = [rng.choice([item for item in SYMPTOMS if item[0] in special_ids])]
    else:
        selected = rng.sample([item for item in SYMPTOMS if item[0] not in special_ids], rng.randint(1, 3))
        if rng.random() < 0.15:
            selected.append(next(item for item in SYMPTOMS if item[0] == 10))
    phrases = [_utterance(item, split, rng) for item in selected]
    transcript = rng.choice([" 그리고 ", ", ", " 또 "]).join(phrases)
    if selected[0][0] not in special_ids and rng.random() < 0.2:
        excluded = rng.choice([item for item in SYMPTOMS[:7] if item not in selected])
        transcript = f"{_utterance(excluded, split, rng)}는 아니고 {transcript}"
    response = {
        "question_id": qid, "phase": phase, "type": "MULTI_SELECT",
        "question": "어떤 일이 생길 것 같으세요? 해당되는 것을 모두 말씀해 주세요.",
        "options": _options(SYMPTOMS, rng), "transcript": transcript,
    }
    return response, sorted({item[0] for item in selected})


def _scale(qid: str, phase: str, split: str, rng: random.Random) -> tuple[dict, int]:
    value = rng.randint(1, 10)
    if split == "train":
        phrases = [f"{value}점이요", f"한 {value} 정도예요", f"지금은 {value}인 것 같아요", f"{KOREAN_NUMBERS[value]} 점 정도입니다"]
    else:
        phrases = [f"십 점 만점에 {value} 정도요", f"불편감은 {KOREAN_NUMBERS[value]}쯤 돼요", f"처음엔 {max(1, value - 2)} 같았는데 지금은 {value}점이에요"]
    return {
        "question_id": qid, "phase": phase, "type": "SCALE",
        "question": "지금 느끼는 불안이나 불편감은 어느 정도인가요?",
        "scale": {"min": 1, "max": 10}, "transcript": rng.choice(phrases),
    }, value


def _single(kind: str, qid: str, phase: str, split: str, rng: random.Random) -> tuple[dict, int]:
    bank = CONFIRMATION if kind == "CONFIRMATION" else OUTCOMES
    selected = rng.choice(bank)
    question = "예상했던 일이 실제로 일어났나요?" if kind == "CONFIRMATION" else "실제로 일어난 일은 무엇인가요?"
    return {
        "question_id": qid, "phase": phase, "type": kind, "question": question,
        "options": _options(bank, rng), "transcript": _utterance(selected, split, rng),
    }, selected[0]


def _response(kind: str, qid: str, phase: str, split: str, rng: random.Random) -> tuple[dict, object]:
    if kind == "MULTI_SELECT":
        return _multi(qid, phase, split, rng)
    if kind == "SCALE":
        return _scale(qid, phase, split, rng)
    return _single(kind, qid, phase, split, rng)


def _record(index: int, split: str, rng: random.Random) -> tuple[dict, dict]:
    stage = rng.choice(list(STAGES))
    available = STAGES[stage]
    count = rng.choices(range(1, len(available) + 1), weights=[3, 4, 2, 1][:len(available)])[0]
    selected = rng.sample(available, count)
    responses, answers = [], []
    for kind, qid, phase in selected:
        response, value = _response(kind, qid, phase, split, rng)
        responses.append(response)
        answers.append({"question_id": response["question_id"], "value": value})
    record = {
        "record_id": f"semantic-{split}-{index:04d}",
        "survey_id": "claustrophobia-vr", "survey_version": "pdf-test-1",
        "phobia_type": "claustrophobia", "stage": stage,
        "responses": responses,
    }
    return record, {"record_id": record["record_id"], "answers": answers}


def _validate(record: dict, expected: dict) -> None:
    assert [item["question_id"] for item in record["responses"]] == [item["question_id"] for item in expected["answers"]]
    for response, answer in zip(record["responses"], expected["answers"]):
        value = answer["value"]
        if response["type"] == "SCALE":
            assert response["scale"]["min"] <= value <= response["scale"]["max"]
        else:
            valid = {item["option_id"] for item in response["options"]}
            assert set(value) <= valid if isinstance(value, list) else value in valid


def main() -> None:
    specs = [(TRAIN_OUTPUT, "train", 1200, random.Random(20260909)), (HOLDOUT_OUTPUT, "holdout", 150, random.Random(20261009))]
    for path, split, count, rng in specs:
        with path.open("w", encoding="utf-8") as stream:
            for index in range(1, count + 1):
                record, expected = _record(index, split, rng)
                _validate(record, expected)
                if split == "train":
                    item = {"messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": json.dumps(record, ensure_ascii=False, separators=(",", ":"))},
                        {"role": "assistant", "content": json.dumps(expected, ensure_ascii=False, separators=(",", ":"))},
                    ]}
                else:
                    item = {"input": record, "expected": expected}
                stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"train=1200: {TRAIN_OUTPUT}")
    print(f"holdout=150: {HOLDOUT_OUTPUT}")


if __name__ == "__main__":
    main()
