- `survey-responses.jsonl`: VR 프로그램이 한 세션의 한 Stage 응답을 저장하는 원본 포맷입니다.
- `ontology-input.jsonl`: Parser가 원본 Stage 응답을 `context + key-value`로 변환한 Ontology 입력 포맷입니다.

`context`에는 세션·설문·버전·공포증·Stage를 한 번만 저장합니다. `values`에는 `PRE.Q59.MULTI_SELECT`, `POST.Q66.SCALE`, `STAGE.STAY_TIME_SECONDS`처럼 규칙에서 바로 조회할 키와 값을 저장합니다.
