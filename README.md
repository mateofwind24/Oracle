# Oracle Report

Raspberry Pi에서 얼굴 캡처, 관상 보조 분석, 사주/만세력 조회, 개인/궁합 리포트를 실행하는 로컬 앱입니다. 기본 실행은 `./run.sh` 하나로 합니다.

## Modes

- `debug`: 출력 로그와 산출물을 `runs/debug/<timestamp>/` 아래에 저장하고 화면에도 출력합니다.
- `release`: 산출물을 임시 디렉터리에만 만들고 실행 뒤 삭제합니다.
- `release`에서는 `--output`, `--output-dir`를 사용할 수 없습니다.

## Build

```bash
./run.sh build
```

이미 `models/*.gguf`가 있으면 모델 다운로드를 건너뜁니다.

## Full Run

웹 UI 실행:

```bash
./run.sh
```

브라우저 접속:

```text
http://<raspberry-pi-ip>:8501
```

태어난 시간을 모르면 웹 UI의 태어난 시간에서 `모름`을 선택합니다. 이 경우 내부 만세력 조회는 기존 MVP 정책대로 `12:00` 오시 대표값을 사용하지만, 프로필에는 `birth_time_known=False`로 저장하고 리포트에는 시간 미상으로 표시합니다.

## Prompt Debugging

개인 리포트 관상 분석 LLM 프롬프트 확인:

```bash
./run.sh prompt personal-face-analysis \
  --name "홍길동" \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male
```

궁합 리포트 관상 분석 LLM 프롬프트 확인:

```bash
./run.sh prompt compatibility-face-analysis \
  --name "홍길동" \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male \
  --mode 연인 \
  --person-label "첫 번째 사람"
```

사주/만세력 조회 결과가 최종 리포트에 어떤 텍스트로 들어가는지 확인:

```bash
./run.sh prompt saju-reading \
  --name "홍길동" \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male
```

사주/만세력 조회 결과를 `configs/prompts.json`의 `saju_reading` 프롬프트에 넣고 실제 LLM 결과만 확인:

```bash
./run.sh llm saju-reading \
  --name "홍길동" \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male
```

개인 최종 리포트 프롬프트 확인:

```bash
./run.sh prompt personal-final \
  --name "홍길동" \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male \
  --target-gender female \
  --face-analysis "관상 분석 결과 예시" \
  --recommendation-text "추천 후보 예시"
```

궁합 최종 리포트 프롬프트 확인:

```bash
./run.sh prompt compatibility-final \
  --name "홍길동" \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male \
  --right-name "김영희" \
  --right-birth-date 1997-05-20 \
  --right-birth-time 09:00 \
  --right-gender female \
  --mode 연인 \
  --face-analysis "두 사람 관상 분석 결과 예시"
```

개인 관상 분석 프롬프트를 실제 LLM에 보내고 결과 확인:

```bash
./run.sh prompt-run personal-face-analysis \
  --name "홍길동" \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male \
  --image runs/session-001/capture.jpg
```

궁합 관상 분석 프롬프트를 실제 LLM에 보내고 결과 확인:

```bash
./run.sh prompt-run compatibility-face-analysis \
  --name "홍길동" \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male \
  --mode 연인 \
  --person-label "첫 번째 사람" \
  --image runs/session-001/capture.jpg
```

개인 최종 리포트 프롬프트를 실제 LLM에 보내고 결과 확인:

```bash
./run.sh prompt-run personal-final \
  --name "홍길동" \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male \
  --target-gender female \
  --face-analysis "관상 분석 결과 예시" \
  --recommendation-text "추천 후보 예시"
```

`llm`은 `prompt-run`과 같은 실행 경로를 쓰지만, 결과 확인용 이름입니다. 예를 들어 아래 두 명령은 모두 프롬프트를 로컬 LLM에 보내고 LLM 응답만 출력합니다.

```bash
./run.sh llm personal-final \
  --name "홍길동" \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male \
  --target-gender female \
  --face-analysis "관상 분석 결과 예시" \
  --recommendation-text "추천 후보 예시"
```

`personal-final` 출력 JSON은 Flask 결과 화면과 `runs/.../personal_report.html`에 같은 섹션 구조로 렌더링됩니다.

정리하면 `personal-face-analysis`와 `compatibility-face-analysis`는 LLM 관상 모드에서만 쓰는 입력입니다. 랜드마크 모드는 프롬프트 없이 rule-based 관상 텍스트를 사용합니다. `prompt saju-reading`은 LLM에 넣기 전 사주/만세력 텍스트 블록만 확인하고, `llm saju-reading`은 같은 조회 결과를 `saju_reading` 템플릿에 넣어 LLM 응답만 출력합니다. `personal-final`과 `compatibility-final`은 사주/만세력 정보와 관상정보를 함께 넣은 최종 리포트 프롬프트입니다.

프롬프트 템플릿 수정:

```text
configs/prompts.json
```

주요 템플릿 키:

- `saju_reading`: 이름, 생년월일, 시간, 성별로 만세력 DB를 조회한 뒤 LLM에 보내는 사주 해설 프롬프트
- `personal_face_analysis`: 캡처 이미지 기반 개인 관상 메모 프롬프트
- `compatibility_face_analysis`: 캡처 이미지 기반 궁합 관상 메모 프롬프트
- `personal_final`: 사주/만세력, 관상 메모, 얼굴 추천 정보를 합친 개인 최종 JSON 리포트 프롬프트
- `compatibility_final`: 두 사람의 사주/만세력과 관상 메모를 합친 궁합 최종 JSON 리포트 프롬프트

LLM 결과만 확인할 때 자주 쓰는 설정:

```env
ORACLE_PROMPTS_PATH=configs/prompts.json
ORACLE_LLM_BASE_URL=http://127.0.0.1:8080/v1
ORACLE_REPORT_LLM_MODEL=local-model
ORACLE_REPORT_LLM_MAX_OUTPUT_TOKENS=1800
ORACLE_MANSE_DB_PATH=data/manse.sqlite
```

## Capture Debugging

얼굴 캡처만 디버그 실행:

```bash
./run.sh debug capture
```

얼굴 캡처만 릴리즈 실행:

```bash
./run.sh release capture
```

랜드마크 규칙 기반 관상 모드로 캡처 디버깅:

```bash
./run.sh debug capture --face-analysis-mode 2
```

## Useful Settings

`run.sh` 상단 또는 `.env`에서 자주 바꾸는 값:

```env
ORACLE_APP_PORT=8501
ORACLE_LLAMA_MODEL_PATH=models/gemma-3-1b-it-Q4_0.gguf
ORACLE_CAMERA_INDEX=0
ORACLE_SHOW_PREVIEW=0
ORACLE_FACE_ANALYSIS_MODE=1
ORACLE_PROMPTS_PATH=configs/prompts.json
```

관상 모드:

- `ORACLE_FACE_ANALYSIS_MODE=1`: 캡처 이미지 기반 LLM 관상 분석
- `ORACLE_FACE_ANALYSIS_MODE=2`: MediaPipe 랜드마크 규칙 기반 분석

태어난 시간 선택:

- `모름`: 내부 조회는 `12:00` 오시 대표값, 리포트 표시는 시간 미상
- `자시`부터 `해시`: 선택한 2시간 단위 시지의 대표시간으로 만세력 DB 조회

## Verification

```bash
python -m pytest
```

## Project Structure

```text
.
├── build.sh
├── run.sh
├── configs/
│   └── prompts.json
├── data/
│   └── manse.sqlite
├── docs/
├── models/
│   └── README.md
├── scripts/
├── src/
│   └── oracle_report/
│       ├── cli.py
│       ├── config.py
│       ├── report.py
│       ├── web.py
│       ├── workflow.py
│       ├── saju/
│       └── vision/
└── tests/
```
