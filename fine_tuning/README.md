# 음성 의미 매칭 파인튜닝

`build_semantic_matcher_dataset.py`는 폐쇄공포증 PDF의 실제 보기와 1~10 척도를 기준으로 다음 파일을 생성합니다.

- `semantic-matcher-sft.jsonl`: messages 형식 학습 데이터 1,600건
- `semantic-matcher-holdout.jsonl`: 학습에 넣지 않는 검증 데이터 150건

학습 목표는 자유로운 STT 전사문을 현재 제공된 `option_id` 또는 척도값에 매칭하는 것입니다. 모델은 진단이나 Ontology 점수를 계산하지 않고 `record_id`, `question_id`, `value`만 출력합니다.

기존 1·2차 모델은 이미 선택된 ID를 JSON으로 재구성하는 학습이었으므로 새 의미 매칭 정확도의 근거로 사용하지 않습니다.

새 모델이 생성한 150줄의 예측 JSONL은 다음 명령으로 검증합니다.

```powershell
python fine_tuning/evaluate_predictions.py predictions.jsonl
```

JSON 유효성, 레코드 완전 일치율, 질문별 값 정확도를 출력합니다.
