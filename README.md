# Oracle Report

라즈베리파이에서 Flask 기반 경량 UI를 띄우고, 개인정보 입력 뒤 카메라로 얼굴을 2초 이상 안정적으로 캡처해 관상, 사주, 궁합 리포트를 만드는 온디바이스 프로젝트입니다. 개인정보 기반 사주 조회는 카메라 캡처와 병렬로 실행되고, 캡처 이미지로 관상 LLM을 먼저 호출한 뒤 관상정보와 사주팔자 정보를 최종 리포트 LLM에 넣어 Markdown 결과를 생성합니다. 두 사람 궁합은 두 사람의 개인정보를 먼저 입력받고, 첫 번째 사람 촬영 후 3초 뒤 두 번째 사람을 순차 촬영합니다.

## 구성

- 얼굴 캡처 하네스: OpenCV Haar cascade 기반 저해상도 얼굴 탐지, 2초 지속 추적, 품질 경고 후 캡처
- 품질 게이트: 눈 감김 의심, 눈썹 가림 의심, 얼굴 크기/다중 얼굴 경고
- 사주 룰엔진: 양력 기준 간지 산출, 오행 분포, 일간 중심 룰 해석
- 경량 UI: Flask 기반 로컬 웹 앱으로 개인 리포트와 두 사람 궁합 메뉴 제공
- 리포트 생성: localhost의 llama.cpp 로컬 LLM을 관상 분석 LLM과 최종 리포트 LLM 역할로 분리
- 내부 DB: 사전 생성된 SQLite 만세력 DB와 얼굴 추천 DB를 로컬 파일로 사용
- 테스트 하네스: 실제 카메라/LLM 없이 상태 머신과 만세력 DB 조회를 검증

현재 전체 플로우는 `docs/oracle_flow.puml`의 PlantUML 다이어그램과 `docs/oracle_service_blueprint.md`에 정리되어 있습니다.

## 설치

라즈베리파이에서는 기본적으로 아래 위치에 클론하는 것을 기준으로 둡니다.

```bash
mkdir -p /home/willtek/work
cd /home/willtek/work
git clone <repo-url> oracle
cd /home/willtek/work/oracle
```

```bash
./build.sh
```

`build.sh`는 Debian/Raspberry Pi 계열에서 필요한 apt 패키지를 확인하고, 이미 설치된 항목은 건너뜁니다. Python venv, 프로젝트 패키지, 테스트, llama.cpp 서버 빌드도 같은 방식으로 준비합니다.

반복 실행 시에는 이미 준비된 항목을 최대한 건너뜁니다.

- apt 패키지가 설치되어 있으면 apt install 생략
- Python venv가 있으면 재생성 생략
- `cv2`, `flask`, `requests`, `pytest`, `oracle_report`와 `oracle-report` 실행 파일이 준비되어 있으면 pip install 생략
- `llama-server`가 PATH에 있거나 `.deps/llama.cpp/build/bin/llama-server`가 있으면 llama.cpp clone/build 생략
- `.env`가 있으면 덮어쓰지 않음
- `data`, `models`, `runs` 디렉터리가 없으면 생성
- `data/manse.sqlite` 만세력 DB가 설정 범위 기준으로 준비되어 있으면 재생성 생략

테스트까지 생략해야 하는 빠른 재빌드는 `ORACLE_SKIP_TESTS=1 ./build.sh`를 사용할 수 있습니다.

레포에는 기본 만세력 DB인 `data/manse.sqlite`가 포함됩니다. 라즈베리파이에서 `/home/willtek/work/oracle`로 클론한 뒤 별도 DB 복사 없이 바로 조회할 수 있고, `./build.sh`는 파일의 범위와 행 수가 맞으면 생성을 건너뜁니다.

`ORACLE_MANSE_START_YEAR`와 `ORACLE_MANSE_END_YEAR`는 만세력 DB 생성 범위입니다. 기본값은 1900-2100년입니다. 스키마는 각 날짜의 12개 시지를 행으로 저장하고, 성별에 따라 달라지는 대운 방향은 남성/여성 컬럼에 함께 저장합니다. 런타임에서는 DB에 없는 생년월일/시간/성별을 사주 엔진으로 즉석 계산하지 않고 오류로 처리합니다.

용어는 아래처럼 구분합니다.

- 만세력 DB: 생년월일, 태어난 시간, 성별 기준으로 년주/월주/일주/시주, 오행, 대운 방향을 미리 저장한 원천 데이터
- 사주정보: 만세력 DB에서 조회한 값을 리포트 생성용으로 가공한 해석 입력

## 실행

기본 앱 실행은 아래 명령을 사용합니다.

```bash
./run.sh
```

`run.sh` 맨 위의 설정 블록에서 Flask host/port, 로컬 LLM 서버 주소, 모델 경로, 카메라 해상도, DB 경로를 수정할 수 있습니다. 기본 UI 주소는 `http://0.0.0.0:8501`입니다.

기본 모델 위치는 `/home/willtek/work/oracle/models/model.gguf`입니다. 레포에는 Gemma 4 E2B 계열의 경량 GGUF인 `unsloth/gemma-4-E2B-it-GGUF`의 `gemma-4-E2B-it-UD-IQ2_M.gguf`를 Git LFS part 파일로 포함합니다. 다른 경로를 쓰려면 `run.sh` 상단의 `RUN_ORACLE_LLAMA_MODEL_PATH`를 수정합니다.

모델 파일은 약 2.29GB라 GitHub LFS 단일 파일 한도를 넘습니다. 그래서 `models/model.gguf.part01`, `models/model.gguf.part02`를 LFS로 저장하고, `./build.sh` 또는 `./run.sh`가 `models/model.gguf`로 재조립합니다.

llama.cpp 서버를 직접 실행하려면 아래처럼 실행합니다.

```bash
llama-server -m /home/willtek/work/oracle/models/model.gguf --host 127.0.0.1 --port 8080
```

또는 `.env`에 `ORACLE_LLAMA_MODEL_PATH`를 설정한 뒤 `./run.sh`를 실행하면 서버가 없을 때 자동으로 시작합니다.

멀티모달 관상까지 사진을 직접 넣으려면 llama.cpp에서 지원하는 비전 모델과 projector 설정이 필요합니다. 기본 포함 모델은 가장 작은 텍스트 우선 GGUF이므로 `ORACLE_LLM_SEND_IMAGE=0`, `ORACLE_FACE_LLM_SEND_IMAGE=0`이 기본값입니다. 이 모드에서는 캡처 품질 정보와 사주 룰 해석만 LLM에 전달합니다.

## Face Detection 경량화

현재 얼굴 탐지는 quantized DNN이 아니라 OpenCV Haar cascade입니다. 신경망 모델이 아니므로 int8 quantization을 적용하는 구조는 아닙니다. 대신 라즈베리파이에서 가볍게 돌도록 기본값은 640x480 프레임을 0.5배, 즉 320x240으로 줄인 뒤 얼굴만 찾고 좌표를 원본으로 복원합니다.

```env
ORACLE_FRAME_WIDTH=640
ORACLE_FRAME_HEIGHT=480
ORACLE_CAMERA_FPS=15
ORACLE_FACE_DETECTION_SCALE=0.5
ORACLE_FACE_DETECTION_INTERVAL=2
ORACLE_HAAR_CASCADE_DIR=/usr/share/opencv4/haarcascades
```

더 가볍게 하려면 `ORACLE_FRAME_WIDTH=320`, `ORACLE_FRAME_HEIGHT=240`, `ORACLE_FACE_DETECTION_SCALE=1.0`으로 낮춰도 됩니다. 눈/눈썹 품질검사는 매 프레임이 아니라 얼굴이 2초 이상 유지된 뒤 캡처 직전에만 실행합니다. apt OpenCV에서 `cv2.data`가 없으면 `opencv-data` 패키지의 `/usr/share/opencv4/haarcascades`를 사용하거나 `ORACLE_HAAR_CASCADE_DIR`로 직접 지정합니다.

캡처만 실행:

```bash
./run.sh capture --output-dir runs/session-001
```

이미지와 생년월일시로 리포트 생성:

```bash
./run.sh report \
  --name "홍길동" \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --image runs/session-001/capture.jpg \
  --output reports/hong.md
```

캡처부터 리포트까지 한 번에 실행:

```bash
./run.sh run \
  --name "홍길동" \
  --birth-date 1995-03-15 \
  --birth-time 14:30 \
  --output reports/hong.md
```

인자를 생략하고 `./run.sh`만 실행하면 이름, 생년월일, 태어난 시간을 물어본 뒤 전체 흐름을 실행합니다.

기본 LLM 설정은 `ORACLE_LLM_BASE_URL=http://127.0.0.1:8080/v1`입니다. 클라우드 LLM과 API 키 방식은 지원하지 않으며, `ORACLE_LLM_BASE_URL`은 `localhost`, `127.0.0.1`, `::1`만 허용합니다.

## 검증

```bash
pytest
```

## 참고

`../tmi`의 LLM 경계, 이미지 전송량 제한, `.env` 기반 설정 패턴을 반영했습니다. `../cvlib/.agent`는 존재하지 않아 `../cvlib/AGENTS.md`와 `.codex` 가이드를 참고했습니다.
