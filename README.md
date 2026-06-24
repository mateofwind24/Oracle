# Oracle Report

Raspberry Pi에서 얼굴 캡처, 관상 보조 분석, 사주/만세력 조회, 개인/궁합 리포트를 실행하는 로컬 앱입니다. 실행은 기본적으로 `./run.sh` 하나로 합니다.

## Modes

- `debug`: 출력 로그와 산출물을 `runs/debug/<timestamp>/` 아래에 저장하고 화면에도 출력합니다.
- `release`: 산출물을 임시 디렉터리에만 만들고 실행 후 삭제합니다. 화면 출력만 남습니다.
- `release`에서는 `--output`, `--output-dir`을 사용할 수 없습니다.

## Build

전체 빌드:

```bash
./run.sh build
```

이미 `models/*.gguf`가 있으면 모델 다운로드는 건너뜁니다.

## Full Run

웹 UI 전체 실행:

```bash
./run.sh
```

브라우저 접속:

```text
http://<raspberry-pi-ip>:8501
```

CLI 전체 워크플로우 디버그 실행, 결과 저장:

```bash
./run.sh debug run \
  --name "홍길동" \
  --birth-date 1995-03-15 \
  --birth-time 14:30
```

CLI 전체 워크플로우 릴리즈 실행, 화면 출력만:

```bash
./run.sh release run \
  --name "홍길동" \
  --birth-date 1995-03-15 \
  --birth-time 14:30
```

## Prompt Debugging

프롬프트만 확인하고 저장:

```bash
./run.sh debug prompt face-analysis \
  --name "홍길동" \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male
```

프롬프트만 확인하고 화면 출력만:

```bash
./run.sh release prompt face-analysis \
  --name "홍길동" \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male
```

프롬프트를 실제 LLM에 보내고 결과 저장:

```bash
./run.sh debug prompt-run face-analysis \
  --name "홍길동" \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male \
  --image runs/session-001/capture.jpg
```

프롬프트 템플릿 수정:

```text
configs/prompts.json
```

## Capture Debugging

얼굴 캡처만 디버그 실행, 이미지와 로그 저장:

```bash
./run.sh debug capture
```

얼굴 캡처만 릴리즈 실행, 임시 캡처 후 삭제:

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
ORACLE_LLAMA_MODEL_PATH=models/model.gguf
ORACLE_CAMERA_INDEX=0
ORACLE_SHOW_PREVIEW=0
ORACLE_FACE_ANALYSIS_MODE=1
ORACLE_PROMPTS_PATH=configs/prompts.json
```

관상 모드:

- `ORACLE_FACE_ANALYSIS_MODE=1`: 캡처 이미지 기반 LLM 관상 분석
- `ORACLE_FACE_ANALYSIS_MODE=2`: MediaPipe 랜드마크 규칙 기반 분석

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
│   ├── prompts.json
│   └── raspberry_pi.env
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
