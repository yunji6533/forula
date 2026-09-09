# PDF 설문 포맷 적용 테스트

제공된 PDF의 Stage 1~9 문항 구성을 현재 JSONL 포맷에 대입한 예시입니다. PDF에는 사용자 응답이 없으므로 `answer`와 `stay_time_seconds`는 구조 확인용 가상값입니다.

- `survey-responses.jsonl`: Stage별 원본 응답 예시이며 한 줄이 한 Stage입니다.
- `ontology-input.jsonl`: 위 원본을 `src/parser.py`로 변환한 `context + key-value` Ontology 입력입니다.

## 적용 규칙

- 콘텐츠 전 설문: `phase: "PRE"`
- 콘텐츠 후 설문: `phase: "POST"`
- 1~10 척도: `answer: {"value": 8}`
- 단일 선택/확인: `answer: {"option_id": 1}`
- 복수 선택: `answer: {"option_ids": [1, 5]}`
- PDF의 화면 번호는 이 테스트에서 임시 `question_id`와 `screen_id`로 사용했습니다.

## PDF 복수 선택 번호

1. 숨이 답답해짐
2. 가슴이 두근거림
3. 손에 땀이 남
4. 목소리가 떨림
5. 빨리 나가고 싶어짐
6. 식은땀이 남
7. 호흡이 빨리 질 것
8. 별다른 일은 일어나지 않음
9. 잘 모르겠다
10. 기타

확인 문항은 테스트 포맷에서 `1 = 그렇다`, `2 = 아니다`로 두었습니다. Stage 8~9의 결과 문항은 `1 = 피해 없음`, `2 = 피해를 입히는 중대한 일`로 두었습니다. 이 번호 규칙은 최종 프로그램의 선택지 ID가 확정되면 맞춰 변경합니다.
