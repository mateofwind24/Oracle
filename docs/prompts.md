# Prompt Editing Guide

프롬프트 템플릿은 기본적으로 `configs/prompts.json`만 수정하면 됩니다.

## 템플릿

현재 런타임에서 쓰는 LLM 프롬프트는 네 개입니다.

- `personal_face_analysis`: 개인 리포트에서 LLM 관상 모드를 쓸 때 이미지와 함께 들어가는 프롬프트
- `personal_final`: 개인 리포트에서 사주/만세력 정보, 관상정보, 추천 정보를 함께 넣는 최종 프롬프트
- `compatibility_face_analysis`: 궁합 리포트에서 LLM 관상 모드를 쓸 때 각 사람 이미지와 함께 들어가는 프롬프트
- `compatibility_final`: 궁합 리포트에서 두 사람의 사주/만세력 정보와 관상정보를 함께 넣는 최종 프롬프트

랜드마크 규칙 기반 관상 모드(`ORACLE_FACE_ANALYSIS_MODE=2`)는 LLM 관상 프롬프트를 사용하지 않습니다. 캡처 단계에서 만든 rule-based 관상 텍스트가 최종 프롬프트에 바로 들어갑니다.

각 템플릿은 JSON 문자열 배열입니다. 배열의 각 항목은 한 줄로 합쳐집니다.

## 자리표시자

템플릿 안의 `${name}`, `${saju_text}`, `${face_analysis}` 같은 값은 실행 시 코드가 채웁니다. 자리표시자는 삭제하거나 이름을 바꾸면 실행 중 오류가 납니다.

자주 쓰는 자리표시자:

- `${name}`: 이름
- `${gender}`: 입력 성별
- `${birth_datetime}`: 생년월일시
- `${birth_time_text}`: 태어난 시간 입력 여부
- `${timezone}`: 시간대
- `${quality_text}`: 캡처 품질 요약
- `${saju_text}`: 사주/만세력 결과 블록
- `${face_analysis}`: 관상 분석 결과
- `${recommendation_text}`: 추천 얼굴 정보
- `${mode}`: 궁합 모드
- `${person_label}`: 궁합 관상 분석에서 현재 분석 대상
- `${left_*}`, `${right_*}`: 궁합 최종 프롬프트의 두 사람 정보

## 확인 명령

개인 관상 분석 LLM 프롬프트:

```bash
./run.sh prompt personal-face-analysis --name tester --birth-date 1995-03-15 --birth-time 14:30 --gender male
```

궁합 관상 분석 LLM 프롬프트:

```bash
./run.sh prompt compatibility-face-analysis --name tester --birth-date 1995-03-15 --birth-time 14:30 --gender male --mode 연인 --person-label "첫 번째 사람"
```

사주/만세력 입력 블록:

```bash
./run.sh prompt saju-reading --name tester --birth-date 1995-03-15 --birth-time 14:30 --gender male
```

개인 최종 프롬프트:

```bash
./run.sh prompt personal-final \
  --name tester \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male \
  --target-gender female \
  --face-analysis "관상 분석 결과 예시" \
  --recommendation-text "추천 후보 예시"
```

궁합 최종 프롬프트:

```bash
./run.sh prompt compatibility-final \
  --name tester-a \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male \
  --right-name tester-b \
  --right-birth-date 1997-05-20 \
  --right-birth-time 09:00 \
  --right-gender female \
  --mode 연인 \
  --face-analysis "두 사람 관상 분석 결과 예시"
```

LLM 결과까지 확인하려면 `prompt` 대신 `prompt-run`을 사용합니다. `saju-reading`은 LLM 단계가 없으므로 `prompt-run` 대상이 아닙니다.

```bash
./run.sh prompt-run personal-face-analysis \
  --name tester \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male \
  --image runs/session-001/capture.jpg
```

```bash
./run.sh prompt-run personal-final \
  --name tester \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male \
  --target-gender female \
  --face-analysis "관상 분석 결과 예시" \
  --recommendation-text "추천 후보 예시"
```

`personal-final` 출력 JSON은 Flask 결과 화면과 `runs/.../personal_report.html`에 같은 섹션 구조로 렌더링됩니다.

## 별도 템플릿 파일 사용

기본 파일 대신 다른 JSON 파일을 쓰려면 `ORACLE_PROMPTS_PATH`를 지정합니다.

```bash
ORACLE_PROMPTS_PATH=configs/prompts.local.json ./run.sh prompt personal-face-analysis \
  --name tester \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male
```
