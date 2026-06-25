# Oracle Report 실행/재현 문서

이 문서는 Oracle Report를 새 환경에서 설치, 실행, 테스트, 프롬프트 검증까지 재현하기 위한 절차를 정리합니다.

## 1. 환경 준비

권장 환경:

- Python 3.10 이상
- Linux/Raspberry Pi 또는 Windows + Bash 실행 환경
- 로컬 llama.cpp `llama-server`
- USB/CSI 카메라

기본 설치:

```bash
./run.sh build
```

`build.sh`는 Python 환경, Python 패키지, llama.cpp, 기본 GGUF 모델, 만세력 DB를 준비합니다. 이미 `models/*.gguf`가 있으면 모델 다운로드를 건너뜁니다.

고정 버전 목록만 직접 설치해야 하는 환경에서는 다음 파일을 참고합니다.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 2. 웹 UI 실행

```bash
./run.sh
```

브라우저에서 접속:

```text
http://<raspberry-pi-ip>:8501
```

태어난 시간을 모르면 웹 UI의 태어난 시간에서 `모름`을 선택합니다. 내부 만세력 조회는 `12:00` 오시 대표값을 사용하지만, 프로필에는 `birth_time_known=False`로 저장하고 리포트에는 시간 미상으로 표시합니다.

## 3. 실행 모드

- `debug`: 출력 로그와 산출물을 `runs/debug/<timestamp>/` 아래에 저장하고 화면에도 출력합니다.
- `release`: 산출물을 임시 디렉터리에만 만들고 실행 뒤 삭제합니다.
- `release`에서는 `--output`, `--output-dir`를 사용할 수 없습니다.

예시:

```bash
./run.sh debug capture
./run.sh release capture
```

## 4. 프롬프트 확인

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

## 5. LLM 결과만 확인

`llm`은 프롬프트를 로컬 LLM에 보내고 LLM 응답만 출력합니다.

사주/만세력 조회 결과를 `configs/prompts.json`의 `saju_reading` 프롬프트에 넣고 LLM 결과만 확인:

```bash
./run.sh llm saju-reading \
  --name "홍길동" \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male
```

개인 최종 리포트 LLM 결과만 확인:

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

이미 캡처한 이미지로 관상 프롬프트를 LLM에 보낼 때:

```bash
./run.sh prompt-run personal-face-analysis \
  --name "홍길동" \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male \
  --image runs/session-001/capture.jpg
```

## 6. 설정 파일

자주 수정하는 값은 `run.sh` 상단 또는 `.env`에서 조정합니다.

```env
ORACLE_APP_PORT=8501
ORACLE_LLAMA_MODEL_PATH=models/gemma-3-1b-it-Q4_0.gguf
ORACLE_CAMERA_INDEX=0
ORACLE_SHOW_PREVIEW=0
ORACLE_FACE_ANALYSIS_MODE=1
ORACLE_PROMPTS_PATH=configs/prompts.json
ORACLE_MANSE_DB_PATH=data/manse.sqlite
ORACLE_FACE_DB_PATH=data/face_recommendations.sqlite
```

프롬프트 템플릿은 다음 파일을 수정합니다.

```text
configs/prompts.json
```

주요 템플릿 키:

- `saju_reading`: 만세력 DB 조회 결과를 LLM에 보내는 사주 해설 프롬프트
- `personal_face_analysis`: 캡처 이미지 기반 개인 관상 메모 프롬프트
- `compatibility_face_analysis`: 캡처 이미지 기반 궁합 관상 메모 프롬프트
- `personal_final`: 사주/만세력, 관상 메모, 얼굴 추천 정보를 합친 개인 최종 JSON 리포트 프롬프트
- `compatibility_final`: 두 사람의 사주/만세력과 관상 메모를 합친 궁합 최종 JSON 리포트 프롬프트

관상 모드:

- `ORACLE_FACE_ANALYSIS_MODE=1`: 캡처 이미지 기반 LLM 관상 분석
- `ORACLE_FACE_ANALYSIS_MODE=2`: MediaPipe 랜드마크 규칙 기반 분석

## 7. 데이터 파일

- `data/manse.sqlite`: 사전 생성된 만세력 DB
- `data/physiognomy_rules.sqlite`: 랜드마크 규칙 기반 관상 보조 DB
- `data/face_recommendations.sqlite`: 개인 리포트 추천 후보용 로컬 샘플 DB. 없으면 실행 중 자동 생성됩니다.

만세력 DB를 다시 생성:

```bash
oracle-build-manse-db --db data/manse.sqlite --start-year 1900 --end-year 2100
```

## 8. 테스트 및 결과 저장

테스트 실행:

```bash
python -m pytest
```

테스트 결과를 `test-results/`에 저장:

```bash
mkdir -p test-results
python -m pytest 2>&1 | tee test-results/pytest-latest.txt
```

현재 저장된 최신 결과:

```text
test-results/pytest-latest.txt
```

## 9. 재현 체크리스트

1. `requirements.txt` 또는 `./run.sh build`로 라이브러리 설치
2. `data/manse.sqlite` 존재 확인
3. `models/*.gguf` 또는 `ORACLE_LLAMA_MODEL_PATH` 확인
4. `configs/prompts.json` 프롬프트 확인
5. `python -m pytest` 실행
6. `./run.sh`로 웹 UI 실행
