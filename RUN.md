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
./build.sh
```

`build.sh`는 Python 환경, Python 패키지, llama.cpp, 기본 GGUF 모델을 준비합니다. 이미 `models/*.gguf`가 있으면 모델 다운로드를 건너뜁니다. `run.sh`는 빌드를 수행하지 않고 실행만 담당합니다.

노트북과 GPU가 없는 환경에서 재현되도록 llama.cpp 빌드와 실행은 기본적으로 CPU 모드입니다. CUDA 빌드가 필요할 때만 `./build.sh --cuda` 또는 `./build.sh --auto-gpu`를 사용하고, 실행 시 GPU offload가 필요할 때만 `./run.sh -ngl 99`처럼 명시합니다.

기본 Python 설치 위치는 프로젝트 루트의 `.venv`입니다. 다만 현재 실행(활성화)되어 있는 가상환경(anaconda, uv, venv 등)이 있다면 자동으로 해당 환경을 사용하며, 다른 환경을 강제로 쓰려면 `--python-env uv`, `--python-env conda`, `--python-env active-conda`, `--python-env active-venv`, `--python-env auto` 중 하나를 명시할 수 있습니다.

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

`./run.sh`가 직접 시작한 llama.cpp 서버는 `./run.sh` 종료 시 함께 종료됩니다. 웹 UI 실행 중 `Ctrl+C`로 끄면 Flask UI와 이번 실행에서 시작된 llama-server가 같이 내려갑니다. 이미 실행 중이던 외부 llama-server를 재사용한 경우에는 해당 프로세스를 종료하지 않습니다.

태어난 시간을 모르면 웹 UI의 태어난 시간에서 `모름`을 선택합니다. 내부 사주 계산은 `12:30` 오시 대표값을 사용하지만, 프로필에는 `birth_time_known=False`로 저장하고 리포트에는 시간 미상으로 표시합니다.

실제 카메라 입력으로 랜드마크 metric과 룰 판정만 실시간 확인할 때:

```bash
./run.sh camera_compare
```

`compare_camera`도 같은 모드로 동작합니다. 이 모드는 얼굴이 인식되어도 캡처 완료로 멈추지 않고, 오른쪽 패널의 측정값과 판정 결과를 계속 갱신합니다.

UI의 핑크 가이드와 OpenCV 가이드가 맞는지 확인할 때는 노란 OpenCV 박스를 임시로 함께 표시할 수 있습니다.

```bash
ORACLE_WEB_CAPTURE_SHOW_OPENCV_GUIDE=1 ./run.sh camera_compare
```

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

실행 중 어떤 운영 프롬프트 파일이 사용되는지는 터미널 stderr의 `[PROMPT]` 로그에서 확인할 수 있습니다.

```text
[PROMPT] name=saju_reading source=configs/prompts_personal.json id_slot=1 prefix_chars=... body_chars=...
[PROMPT] name=saju_reading_couple source=configs/prompts_compatibility.json id_slot=3 prefix_chars=... body_chars=...
```

개인 리포트 관상 분석 LLM 프롬프트 확인:


궁합 리포트 관상 분석 LLM 프롬프트 확인:


개인 사주/만세력 조회 결과가 리포트에 어떤 텍스트로 들어가는지 확인:

```bash
./run.sh prompt saju-reading \
  --name "홍길동" \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male
```

궁합 사주/만세력 조회 결과가 리포트에 어떤 텍스트로 들어가는지 확인:

```bash
./run.sh prompt saju-reading-couple \
  --name "홍길동" \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male \
  --right-name "김영희" \
  --right-birth-date 1997-05-20 \
  --right-birth-time 09:00 \
  --right-gender female \
  --mode 연인
```

개인 최종 리포트 legacy/debug 프롬프트 확인:

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

## 5. LLM 결과만 확인

`llm`은 프롬프트를 로컬 LLM에 보내고 LLM 응답만 출력합니다.

개인 사주/만세력 조회 결과를 `configs/prompts_personal.json`의 `saju_reading` 프롬프트에 넣고 LLM 결과만 확인:

```bash
./run.sh llm saju-reading \
  --name "홍길동" \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male
```

궁합 사주/만세력 조회 결과를 `configs/prompts_compatibility.json`의 `saju_reading_couple` 프롬프트에 넣고 LLM 결과만 확인:

```bash
./run.sh llm saju-reading-couple \
  --name "홍길동" \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --gender male \
  --right-name "김영희" \
  --right-birth-date 1997-05-20 \
  --right-birth-time 09:00 \
  --right-gender female \
  --mode 연인
```

개인 최종 리포트 legacy/debug 프롬프트를 `configs/prompts_debug.json`에서 읽어 LLM 결과만 확인:

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


## 6. 프롬프트 토큰 확인

`token`은 현재 운영 프롬프트 manifest가 include하는 각 프롬프트 prefix/body 템플릿 토큰 크기를 출력합니다. 기본 실행은 로컬 llama.cpp `/tokenize`를 사용하므로 실제 모델 기준에 가깝습니다.

```bash
./run.sh token
```

llama.cpp 서버 없이 대략적인 크기만 확인할 때:

```bash
./run.sh token --offline
```

## 7. 설정 파일

자주 수정하는 값은 `.env`에서 조정합니다.

```env
ORACLE_APP_PORT=8501
ORACLE_LLAMA_MODEL_PATH=models/gemma-4-E2B-it-UD-Q2_K_XL.gguf
ORACLE_LLAMA_MODEL_URL=https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-UD-Q2_K_XL.gguf
ORACLE_LLAMA_MODEL_SHA256=dd279a54c0c0dc9724ed11d7f73ad7fb4489a45f58fefe9447da2429a727de0c
ORACLE_LLM_PROMPT_CACHE=0
ORACLE_CAMERA_INDEX=0
ORACLE_SHOW_PREVIEW=0
ORACLE_PROMPTS_PATH=configs/prompts.json
ORACLE_DEBUG_PROMPTS_PATH=configs/prompts_debug.json
ORACLE_FACE_DB_PATH=data/face_recommendations.sqlite
```

운영 프롬프트 템플릿은 개인/궁합 파일을 분리해서 수정합니다. `configs/prompts.json`은 두 파일을 include하는 manifest이므로 직접 문구를 넣지 않습니다.

```text
configs/prompts_personal.json        # 개인 리포트 사주 프롬프트
configs/prompts_compatibility.json  # 궁합 리포트 사주 프롬프트
configs/prompts.json                # include manifest
```

legacy/debug 프롬프트는 다음 파일을 수정합니다.

```text
configs/prompts_debug.json
```

주요 템플릿 키:

- `saju_reading`: `configs/prompts_personal.json`에 있는 개인 사주 해설 프롬프트
- `saju_reading_couple`: `configs/prompts_compatibility.json`에 있는 궁합 사주 해설 프롬프트
- `personal_final`: `configs/prompts_debug.json`에만 있는 legacy/debug 프롬프트
- `compatibility_final`: `configs/prompts_debug.json`에만 있는 legacy/debug 프롬프트

운영 프롬프트는 각 분리 파일의 항목 안에서 `id_slot`, `prefix`, `body`로 명시적으로 관리합니다. 기본 manifest인 `configs/prompts.json`을 통해 두 파일을 병합해서 읽으므로 런타임과 분산추론에서 사용하는 프롬프트 이름과 `id_slot`은 기존과 같습니다. 일반 `./run.sh` 실행은 이전 방식처럼 전체 프롬프트를 하나의 user message로 보냅니다. 고정 slot prompt cache를 테스트하려면 `./run.sh kvfix ...`로 실행합니다. 이 모드에서는 `prefix`를 system message로, `body`를 user message로 보내며 프롬프트별 고정 `id_slot`과 `cache_prompt=true`를 함께 보내고, 별도 `--ctx-size`를 주지 않으면 llama.cpp context 기본값을 `20480`으로 올립니다.


카메라 사용이 어렵거나 랜드마크 룰베이스를 재현 테스트해야 할 때는 mock capture를 켭니다.

- `ORACLE_MOCK_CAPTURE_ENABLED=1`: 카메라 촬영 대신 mock 이미지를 생성하고, 개인 리포트용 preset 랜드마크 값을 자동 적용합니다.
- 궁합 리포트에서는 같은 옵션 하나로 첫 번째/두 번째 사람의 preset 랜드마크 값을 각각 자동 적용합니다.
- `ORACLE_MOCK_LANDMARK_METRICS_JSON`: 개인 리포트나 공통 mock capture에 적용할 랜드마크 metric JSON override
- `ORACLE_MOCK_PAIR_LEFT_LANDMARK_METRICS_JSON`: 궁합 첫 번째 사람에게 적용할 랜드마크 metric JSON override
- `ORACLE_MOCK_PAIR_RIGHT_LANDMARK_METRICS_JSON`: 궁합 두 번째 사람에게 적용할 랜드마크 metric JSON override

개인 리포트 mock 실행:

```bash
export ORACLE_MOCK_CAPTURE_ENABLED=1
./run.sh
```

궁합 리포트 mock 실행도 같은 설정을 사용합니다.

```bash
export ORACLE_MOCK_CAPTURE_ENABLED=1
./run.sh
```

값을 직접 바꿔 테스트하고 싶을 때만 JSON override를 추가합니다.

```bash
export ORACLE_MOCK_CAPTURE_ENABLED=1
export ORACLE_MOCK_PAIR_LEFT_LANDMARK_METRICS_JSON='{"eye_width_ratio":0.19,"eye_spacing_ratio":0.28}'
export ORACLE_MOCK_PAIR_RIGHT_LANDMARK_METRICS_JSON='{"mouth_width_ratio":0.43,"jaw_width_ratio":0.72}'
./run.sh
```

## 8. 데이터 파일

- `data/physiognomy_rules.sqlite`: 랜드마크 규칙 기반 관상 보조 DB
- `data/face_recommendations.sqlite`: 개인 리포트 추천 후보용 로컬 샘플 DB. 없으면 실행 중 자동 생성됩니다.

## 9. 테스트 및 결과 저장

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

## 10. 재현 체크리스트

1. `requirements.txt` 또는 `./build.sh`로 라이브러리 설치
2. `.env`의 `ORACLE_LLAMA_MODEL_PATH`와 `models/*.gguf` 확인
3. `configs/prompts_personal.json`, `configs/prompts_compatibility.json` 프롬프트 확인
4. `python -m pytest` 실행
5. `./run.sh`로 웹 UI 실행
