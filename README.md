# VR 음성 설문 Ontology Input Parser

VR 사용자의 음성을 STT로 변환한 뒤, AI가 현재 질문의 보기 중 의미가 가장 가까운 항목을 선택하고 Python Parser가 결과를 검증해 Ontology용 키·값으로 만드는 프로젝트입니다. 공포증 정도의 계산과 판단은 Parser가 아니라 Ontology 규칙·수식이 담당합니다.

## 처리 흐름

```text
VR 음성 → STT transcript → AI 보기 매칭 → Python 검증/변환 → Ontology key-value
```

AI는 전체 Ontology JSON을 만들지 않고 아래 최소 결과만 반환합니다.

```json
{"record_id":"rec-001-s06","answers":[{"question_id":"59","value":[1,5]},{"question_id":"60","value":8}]}
```

Python Parser는 질문 누락·중복, 존재하지 않는 `option_id`, 척도 범위 오류를 차단한 뒤 최종 데이터를 만듭니다.

## 저장 JSONL

- `data/survey-responses.jsonl`: Stage별 질문, 현재 보기/척도, STT 전사문을 저장하는 원본 형식
- `data/ontology-input.jsonl`: 검증된 응답을 `PHASE.Q질문ID.TYPE → 값` 형태로 저장한 Ontology 입력 형식

원본 JSONL 한 줄은 한 세션의 한 Stage입니다. 콘텐츠 전·후는 각 응답의 `phase` 값 `PRE`/`POST`로 구분합니다.

```json
{
  "question_id": "59",
  "phase": "PRE",
  "type": "MULTI_SELECT",
  "question": "버튼을 누르고 문이 닫히면 어떻게 될 것 같으세요?",
  "options": [
    {"option_id": 1, "label": "숨이 답답해짐"},
    {"option_id": 5, "label": "빨리 나가고 싶어짐"}
  ],
  "transcript": "숨이 막힐 것 같고 빨리 나가고 싶어요"
}
```

최종 Ontology 입력 예시는 다음과 같습니다.

```json
{
  "context": {"session_id":"session-001","phobia_type":"claustrophobia","stage":6},
  "values": {
    "PRE.Q59.MULTI_SELECT": [1,5],
    "PRE.Q60.SCALE": 8,
    "STAGE.STAY_TIME_SECONDS": 40
  }
}
```

질문과 보기 문구가 바뀌어도 입력에 현재 문구와 ID를 함께 제공하므로 같은 Parser 구조를 유지할 수 있습니다. 다른 공포증은 해당 설문의 실제 질문·보기 정의와 학습 예제를 추가해 확장합니다.

## 주요 파일

- `src/parser.py`: AI 매칭 결과를 검증하고 Ontology 키·값으로 변환
- `schemas/survey-stage.schema.json`: STT 원본 Stage JSONL 규격
- `schemas/ontology-input.schema.json`: Ontology 입력 JSONL 규격
- `fine_tuning/build_semantic_matcher_dataset.py`: 폐쇄공포증 PDF 보기 기반 학습/검증 데이터 생성
- `pdf_test/`: PDF의 9개 Stage 구조를 반영한 테스트 fixture

## 실행

```powershell
python fine_tuning/build_semantic_matcher_dataset.py
python -m unittest discover -s tests -v
```

실제 모델 사용 시 모델의 최소 매칭 결과 JSONL을 준비한 뒤 다음처럼 변환합니다.

```powershell
python src/parser.py data/survey-responses.jsonl model-matches.jsonl data/ontology-input.jsonl
```

현재 학습 데이터는 폐쇄공포증 PDF의 실제 보기만 사용합니다. 의료기기 공포증은 최종 질문·보기가 공유된 후 별도 예제를 추가해야 합니다.

## Stage 1 직접 입력 테스트

PyCharm에서 `stage1_console.py`를 실행하면 Stage 1 질문에 텍스트로 답하고 AI가 선택한 보기와 최종 Ontology 키·값을 확인할 수 있습니다. 현재 로컬 환경에서는 모델 크기 때문에 CPU로 실행되어 한 회 약 2분이 걸립니다.

기본 어댑터 위치는 `Downloads/job_d9d5ac53-adapter/adapter`입니다. PyCharm 프로젝트 인터프리터는 Formula의 `.venv`를 사용합니다.

```powershell
python stage1_console.py
python stage1_console.py --demo
```

결과는 `pdf_test/stage1-live-result.json`에 저장됩니다.
