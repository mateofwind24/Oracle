# Oracle Report

Raspberry Pi에서 얼굴 캡처, 관상 보조 분석, 사주/만세력 조회, 개인/궁합 리포트를 실행하는 로컬 앱입니다.

이 프로젝트는 카메라 입력, 만세력 SQLite DB, 로컬 llama.cpp 호환 LLM, Flask UI를 묶어 전시/부트캠프용 리포트 생성 흐름을 제공합니다. 개인 정보와 캡처 이미지는 기본적으로 로컬 장치 안에서 처리합니다.

## 주요 기능

- 개인 리포트: 얼굴 캡처, 사주/만세력 조회, 관상 메모, 얼굴 추천 후보를 종합한 HTML 리포트
- 두 사람 궁합 리포트: 두 명의 사주/만세력과 순차 캡처 기반 관상 메모를 종합한 HTML 리포트
- 로컬 LLM 실행: llama.cpp OpenAI-compatible API를 사용해 프롬프트 실행
- 프롬프트 확인/실행 모드: 프롬프트만 출력하거나 LLM 결과만 출력
- Raspberry Pi 실행 보조: `build.sh`, `run.sh`, `systemd/` 서비스 파일 제공

## 프로젝트 구조

```text
.
├── src/                 # 소스 코드
│   └── oracle_report/
├── tests/               # 테스트 코드
├── test-results/        # 테스트 실행 결과 로그/리포트
├── configs/             # 실행 설정과 프롬프트 템플릿
├── data/                # 만세력/관상 규칙 SQLite DB
├── docs/                # 설계 문서와 다이어그램
├── models/              # 로컬 GGUF 모델 배치 위치
├── scripts/             # 설치, DB 생성, 서비스 실행 보조 스크립트
├── systemd/             # Raspberry Pi 서비스 파일
├── README.md            # 프로젝트 개요
├── RUN.md               # 실행/재현 문서
├── requirements.txt     # 재현용 고정 버전 라이브러리 목록
├── pyproject.toml       # Python 패키지 설정
├── build.sh             # 빌드/설치 자동화
└── run.sh               # 실행 진입점
```

Git 히스토리는 제출 시 `.git/` 디렉터리로 포함되며, 텍스트 제출이 필요하면 아래 명령으로 생성할 수 있습니다.

```bash
git log --oneline --decorate --graph --all > git_log.txt
```

## 빠른 시작

자세한 실행, 재현, 설정 방법은 [RUN.md](RUN.md)를 확인하세요.

```bash
./run.sh build
./run.sh
```

브라우저 접속:

```text
http://<raspberry-pi-ip>:8501
```

## 테스트

```bash
python -m pytest
```

최근 테스트 결과는 [test-results/pytest-latest.txt](test-results/pytest-latest.txt)에 저장합니다.

## 주요 설정 파일

- [configs/prompts.json](configs/prompts.json): LLM 프롬프트 템플릿
- [.env.example](.env.example): 실행 환경 변수 예시
- [configs/raspberry_pi.env](configs/raspberry_pi.env): Raspberry Pi 기본 설정 예시
- [requirements.txt](requirements.txt): 고정 버전 라이브러리 목록
