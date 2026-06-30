# Oracle Report 요구사항 명세서

문서 버전: v2.1<br>
수정일: 2026-06-30<br>
기준: 현재 `release` 브랜치 구현

## 1. 적용 범위

Oracle Report는 로컬 장치에서 얼굴 캡처, 랜드마크 룰 기반 관상 보조 분석, 런타임 사주 계산, 로컬 LLM 해설을 결합해 개인 리포트와 두 사람 궁합 리포트를 생성하는 웹 애플리케이션이다.

이 문서는 다음 범위만 다룬다.

- Flask 웹 UI와 비동기 workflow job
- `build.sh`, `run.sh`, `kvfix`, debug/release 실행 모드
- OpenCV/MediaPipe 기반 캡처와 랜드마크 품질 분석
- Python 사주/만세력 계산 엔진
- llama.cpp OpenAI-compatible 로컬 LLM 호출
- 개인/궁합 리포트 HTML, fragment, fallback, 테스트 계약

## 2. 제품 요구사항

| ID | 요구사항 | 우선순위 | 인수 기준 |
|---|---|---:|---|
| PRD-001 | 얼굴 이미지와 입력 정보는 기본적으로 로컬 장치 안에서 처리한다. | P0 | LLM base URL은 로컬 주소만 허용한다. |
| PRD-002 | 개인 리포트와 두 사람 궁합 리포트를 모두 생성한다. | P0 | 두 흐름이 full HTML과 fragment HTML을 반환한다. |
| PRD-003 | 관상은 랜드마크 룰 기반 보조 정보로만 사용한다. | P0 | 얼굴 이미지를 LLM/VLM payload로 보내지 않는다. |
| PRD-004 | 사주는 LLM이 계산하지 않는다. | P0 | Python 엔진이 년월일시주와 오행 분포를 만든다. |
| PRD-005 | LLM은 계산 결과를 해설하고 JSON 구조를 채우는 역할만 한다. | P0 | 프롬프트에 계산된 사주/관상 payload가 들어간다. |
| PRD-006 | 결과는 엔터테인먼트/참고용임을 고지한다. | P0 | 의료, 법률, 재정, 채용, 평가 판단 금지 문구가 표시된다. |
| PRD-007 | 실패 시 사용자가 깨진 JSON이나 원시 오류를 보지 않게 한다. | P0 | fallback 리포트 또는 보정 블록을 렌더링한다. |

## 3. 실행 및 설정 요구사항

| ID | 요구사항 | 우선순위 | 인수 기준 |
|---|---|---:|---|
| RUN-001 | 설치와 빌드는 `./build.sh`가 담당한다. | P0 | `./run.sh build`는 실행 경로가 아니다. |
| RUN-002 | `./run.sh`는 기본적으로 Flask UI를 시작한다. | P0 | 기본 포트 `8501`에서 접속 가능하다. |
| RUN-003 | 활성 가상환경이 있으면 우선 사용하고, 없으면 `.venv`를 사용한다. | P0 | conda/uv/venv 활성 상태를 감지한다. |
| RUN-004 | `.env`를 로드해 `ORACLE_*` 런타임 설정을 export한다. | P0 | `.env`가 없으면 복구 안내와 함께 실패한다. |
| RUN-005 | `debug` 모드는 실행 로그와 산출물을 보관한다. | P0 | `runs/debug/<timestamp>/` 아래에 저장된다. |
| RUN-006 | `release` 모드는 임시 산출물을 실행 후 삭제한다. | P0 | `--output`, `--output-dir` 사용을 거부한다. |
| RUN-007 | `capture`, `prompt`, `prompt-run`, `llm`, `token` CLI 명령을 지원한다. | P0 | 각 명령이 `oracle_report.cli`로 전달된다. |
| RUN-008 | 필요 시 로컬 `llama-server`를 자동 시작한다. | P0 | `ORACLE_START_LLAMA_SERVER=1`이면 모델 경로와 context로 서버를 띄운다. |
| RUN-009 | 기존 로컬 LLM 서버가 있으면 재사용한다. | P0 | `/models` 확인이 성공하면 새 서버를 시작하지 않는다. |
| RUN-010 | `kvfix`는 prompt cache와 고정 slot 실행을 켠다. | P0 | 기본 context 20480, 기본 parallel 5를 적용한다. |
| RUN-011 | legacy 분산 wrapper 옵션을 안전하게 받는다. | P0 | `--distributed-role`, `--distributed-split`, `--slave-addrs`가 CLI command로 오인되지 않는다. |
| RUN-012 | 자주 바뀌는 값은 환경 변수와 config 파일로 관리한다. | P1 | `.env.example`, `configs/raspberry_pi.env`, `configs/prompts.json`에서 확인 가능하다. |

## 4. 입력 요구사항

| ID | 요구사항 | 우선순위 | 인수 기준 |
|---|---|---:|---|
| IN-001 | 개인 리포트는 이름, 생년월일, 태어난 시간, 성별을 입력받는다. | P0 | 필수값 누락 시 workflow가 진행되지 않는다. |
| IN-002 | 개인 리포트는 추천 대상 성별을 받을 수 있다. | P1 | 추천 후보 조회에 해당 값이 반영된다. |
| IN-003 | 개인 리포트는 얼굴 캡처 생략을 지원한다. | P0 | `skip_face=True`이면 사주 리포트만 생성된다. |
| IN-004 | 궁합 리포트는 두 사람의 이름, 생년월일, 태어난 시간, 성별을 받는다. | P0 | 한 사람이라도 필수값이 빠지면 오류가 난다. |
| IN-005 | 궁합 리포트는 관계 모드를 받는다. | P0 | `연인`, `친구`, `직장동료`만 허용한다. |
| IN-006 | 태어난 시간 미상을 지원한다. | P0 | 내부 계산은 `12:30`, 화면 표시는 `시간 미상`으로 분리한다. |
| IN-007 | 시진명 입력을 대표 시간으로 변환한다. | P1 | 예: `오시`, `午時`, `오시(午時)`는 `12:30`으로 해석한다. |
| IN-008 | 성별 입력은 한국어와 영어 별칭을 정규화한다. | P0 | `남/남자/male/m`, `여/여자/female/f`가 처리된다. |

## 5. 얼굴 캡처 요구사항

| ID | 요구사항 | 우선순위 | 인수 기준 |
|---|---|---:|---|
| CAP-001 | USB 또는 CSI 카메라를 사용할 수 있어야 한다. | P0 | 카메라 인덱스와 해상도를 환경 변수로 지정한다. |
| CAP-002 | Linux 카메라 권한 문제를 자동 보정하려고 시도한다. | P1 | `/dev/video*` 접근 권한이 없으면 ACL/chmod 수정을 시도한다. |
| CAP-003 | 한 명의 얼굴만 안정적으로 보일 때 캡처한다. | P0 | 얼굴 없음, 다중 얼굴, 흔들림은 캡처하지 않는다. |
| CAP-004 | 얼굴 품질을 확인한다. | P0 | 얼굴 크기, 눈, 눈썹, 정면성, 가림 정도를 본다. |
| CAP-005 | 조건 충족 후 일정 시간 유지되면 자동 캡처한다. | P0 | 기본 안정 시간은 `ORACLE_MIN_FACE_SECONDS`다. |
| CAP-006 | 웹 미리보기는 현재 상태 메시지와 오버레이를 표시한다. | P1 | `/video-feed`가 MJPEG 프레임을 반환한다. |
| CAP-007 | 궁합 리포트는 두 사람을 순차 캡처한다. | P0 | `person_1/capture.jpg`, `person_2/capture.jpg`를 생성한다. |
| CAP-008 | 카메라 없는 테스트를 위해 mock capture를 제공한다. | P0 | `ORACLE_MOCK_CAPTURE_ENABLED=1`이면 mock 이미지와 metric을 생성한다. |

## 6. 관상 분석 요구사항

| ID | 요구사항 | 우선순위 | 인수 기준 |
|---|---|---:|---|
| FACE-001 | 관상 분석은 MediaPipe 랜드마크와 룰 DB 기반으로만 동작한다. | P0 | `landmark_matches_json`에서 관상 JSON payload를 만든다. |
| FACE-002 | 얼굴 이미지는 LLM 요청 payload에 포함하지 않는다. | P0 | `LlamaCppChatClient._build_payload()`가 `image_path`를 폐기한다. |
| FACE-003 | 얼굴 전용 LLM/VLM 분석 경로는 사용하지 않는다. | P0 | workflow의 face 단계는 `_build_*_rule_based_face_analysis()`를 호출한다. |
| FACE-004 | 관상 payload는 관찰 가능한 얼굴 요소만 근거로 한다. | P0 | 비율, 눈/눈썹, 코, 입, 하관, 삼정 균형을 사용한다. |
| FACE-005 | 민감 속성 추정과 평가성 표현을 금지한다. | P0 | 신원, 나이, 민족, 건강, 직업, 경제력, 외모 점수를 출력하지 않는다. |
| FACE-006 | 개인 관상은 `face_blocks` 중심 JSON으로 병합된다. | P0 | 사주 payload와 합쳐 개인 리포트를 구성한다. |
| FACE-007 | 궁합 관상은 `pair_blocks` 중심 JSON으로 병합된다. | P0 | 두 사람의 랜드마크 매칭 결과를 관계 모드와 함께 사용한다. |
| FACE-008 | 룰 DB가 없거나 스키마가 깨지면 복구 안내를 제공한다. | P1 | `oracle-build-physiognomy-db` 생성 안내가 가능해야 한다. |
| FACE-009 | legacy face LLM/env 값은 현재 런타임의 필수 설정이 아니다. | P1 | `ORACLE_FACE_LLM_*`, `ORACLE_FACE_ANALYSIS_MODE`는 호환용 잔여 값으로 취급한다. |

## 7. 사주/만세력 요구사항

| ID | 요구사항 | 우선순위 | 인수 기준 |
|---|---|---:|---|
| SAJU-001 | 사주 명식은 Python 런타임 엔진이 계산한다. | P0 | `ManseRepository.lookup()`이 `build_saju_reading()`을 사용한다. |
| SAJU-002 | 년주, 월주, 일주, 시주를 반환한다. | P0 | 리포트와 프롬프트에 네 기둥이 포함된다. |
| SAJU-003 | 오행 분포와 대운 방향을 계산한다. | P0 | `element_counts`, `daeun_direction`이 생성된다. |
| SAJU-004 | 성별 정규화 실패는 명확한 입력 오류로 처리한다. | P0 | 허용되지 않은 값은 `ValueError`를 낸다. |
| SAJU-005 | 시간 미상은 실제 출생시간처럼 표시하지 않는다. | P0 | 화면에는 `시간 미상 (오시(午時) 보조 기준)`이 표시된다. |
| SAJU-006 | legacy `manse_db_path`는 현재 계산 필수값이 아니다. | P1 | 누락된 DB 경로가 있어도 런타임 계산은 계속된다. |
| SAJU-007 | LLM 프롬프트는 계산된 사주 텍스트를 근거로 한다. | P0 | `saju_reading`, `saju_reading_couple` body에 계산 결과가 들어간다. |
| SAJU-008 | LLM이 없는 사주 사실을 지어내지 않도록 제한한다. | P0 | 프롬프트는 제공된 사주 데이터 중심 해설을 요구한다. |

## 8. LLM 요구사항

| ID | 요구사항 | 우선순위 | 인수 기준 |
|---|---|---:|---|
| LLM-001 | llama.cpp OpenAI-compatible API를 사용한다. | P0 | `/v1/chat/completions`로 요청한다. |
| LLM-002 | LLM base URL은 로컬 주소만 허용한다. | P0 | `localhost`, `127.0.0.1`, `::1` 외 주소는 오류다. |
| LLM-003 | 최종 리포트 LLM 설정은 `ORACLE_REPORT_LLM_*`로 관리한다. | P0 | model, base URL, timeout, token, temperature가 반영된다. |
| LLM-004 | 출력 토큰 한도를 환경 변수로 조정한다. | P0 | `ORACLE_REPORT_LLM_MAX_OUTPUT_TOKENS`, fallback `ORACLE_MAX_OUTPUT_TOKENS`, 기본 4096. |
| LLM-005 | reasoning 모드는 명시적으로 켤 때만 사용한다. | P1 | `--reasoning` 또는 `ORACLE_REASONING=1`이 필요하다. |
| LLM-006 | prompt cache는 `kvfix` 또는 환경 변수로 제어한다. | P1 | `cache_prompt=true`, `id_slot`이 prompt template 기준으로 적용된다. |
| LLM-007 | LLM 응답이 비면 실패로 처리한다. | P0 | 사용자 화면에는 fallback 결과가 표시된다. |
| LLM-008 | `finish_reason=length`는 불완전 응답으로 처리한다. | P0 | 잘린 JSON을 그대로 렌더링하지 않는다. |
| LLM-009 | LLM 요청 로그는 진단에 필요한 요약만 남긴다. | P1 | prompt/response 전문을 운영 로그에 불필요하게 남기지 않는다. |
| LLM-010 | token 명령은 프롬프트 prefix/body 크기를 확인한다. | P1 | `/tokenize` 또는 offline 추정값을 출력한다. |

## 9. 분산 실행 요구사항

| ID | 요구사항 | 우선순위 | 인수 기준 |
|---|---|---:|---|
| DIST-001 | 분산 split은 명시적으로 켤 때만 활성화된다. | P1 | `ORACLE_DISTRIBUTED_SPLIT=1`과 role `master/hybrid`가 필요하다. |
| DIST-002 | 사주 해설을 카테고리 단위 작업으로 나눌 수 있다. | P1 | metadata 작업과 block 작업을 별도 생성한다. |
| DIST-003 | slave 상태를 조회해 idle/score 기반으로 작업을 배정한다. | P1 | `/api/distributed/status`가 `status`, `tps`, `compute_score`를 반환한다. |
| DIST-004 | local worker는 HTTP 대신 직접 LLM client를 사용할 수 있다. | P1 | local 주소는 직접 실행 경로로 처리된다. |
| DIST-005 | 실패한 작업은 제한된 횟수만 재시도한다. | P1 | worker별 연속 실패와 task retry를 관리한다. |
| DIST-006 | 분산 결과는 원래 카테고리 순서로 재조립한다. | P0 | 누락 카테고리는 기본 오류 block으로 채운다. |

## 10. 리포트 요구사항

| ID | 요구사항 | 우선순위 | 인수 기준 |
|---|---|---:|---|
| REP-001 | 개인 리포트는 핵심 요약, 관상, 사주, 통합 요약을 포함한다. | P0 | `essence`, `face_blocks`, `saju_blocks`, synthesis 영역이 렌더링된다. |
| REP-002 | 얼굴 생략 개인 리포트는 사주 중심으로 렌더링한다. | P0 | 관상 섹션 없이도 HTML이 완성된다. |
| REP-003 | 궁합 리포트는 두 사람 요약, 관상 궁합, 사주 궁합, 실천 제안을 포함한다. | P0 | `pair_blocks`, `saju_blocks`, `action_*`가 표시된다. |
| REP-004 | full document와 fragment HTML을 모두 생성한다. | P0 | 다운로드용 HTML과 화면 삽입 HTML이 분리된다. |
| REP-005 | 개인 최종 JSON의 필수 필드를 검증한다. | P0 | `essence`, `saju_blocks` 부족 시 fallback/default를 사용한다. |
| REP-006 | 궁합 최종 JSON의 필수 필드를 검증한다. | P0 | 관계 요약과 block 부족 시 default를 사용한다. |
| REP-007 | placeholder 문구는 화면에 노출하지 않는다. | P0 | `제목`, `요약`, `본문`, `1~2문장`류 문구를 보정한다. |
| REP-008 | 제목, 요약, 본문 반복을 방어한다. | P0 | 중복 텍스트는 fallback 또는 기본 block으로 대체한다. |
| REP-009 | 사용자 입력과 LLM 출력은 HTML escape 처리한다. | P0 | script 입력이 실행되지 않는다. |
| REP-010 | timing log를 남겨 병목을 확인한다. | P1 | `timings.log`에 단계별 소요 시간이 기록된다. |

## 11. 데이터 요구사항

| ID | 요구사항 | 우선순위 | 인수 기준 |
|---|---|---:|---|
| DATA-001 | 사용자 입력은 운영 모드에서 장기 저장하지 않는다. | P0 | 세션 종료 후 입력값 보관을 전제로 하지 않는다. |
| DATA-002 | 캡처 이미지는 실행 산출물로만 저장한다. | P0 | release 모드는 임시 디렉터리를 삭제한다. |
| DATA-003 | `data/physiognomy_rules.sqlite`는 랜드마크 룰 DB다. | P0 | 관상 룰 매칭에만 사용하고 개인정보를 담지 않는다. |
| DATA-004 | `data/face_recommendations.sqlite`는 추천 후보 샘플 DB다. | P1 | 실제 사용자 얼굴 매칭 DB가 아니다. |
| DATA-005 | `runs/`는 리포트, 캡처, 로그 산출물 저장소다. | P1 | debug/일반 실행 산출물이 여기에 쌓인다. |
| DATA-006 | `test-results/`는 테스트 재현 자료 저장소다. | P1 | pytest 로그와 결과 파일을 남길 수 있다. |
| DATA-007 | mock capture metric override는 테스트용 설정이다. | P1 | 운영 얼굴 데이터 대체 저장소로 쓰지 않는다. |

## 12. 보안 및 개인정보 요구사항

| ID | 요구사항 | 우선순위 | 인수 기준 |
|---|---|---:|---|
| SEC-001 | 촬영 전 개인정보 처리 안내와 동의를 제공한다. | P0 | 사용자는 촬영/처리 범위를 확인할 수 있다. |
| SEC-002 | 로컬 처리 원칙을 사용자에게 알린다. | P0 | UI와 문서에서 온디바이스 처리 흐름을 설명한다. |
| SEC-003 | 외부 LLM endpoint를 차단한다. | P0 | config validation에서 비로컬 base URL을 거부한다. |
| SEC-004 | 운영 로그에 민감정보를 과도하게 남기지 않는다. | P1 | 이름, 생년월일, 얼굴 경로, 프롬프트 전문 로깅을 최소화한다. |
| SEC-005 | 결과는 중요 의사결정에 쓰지 않도록 고지한다. | P0 | 리포트 disclaimer에 금지 목적이 포함된다. |
| SEC-006 | 다운로드/공유 기능은 결과 준비 후에만 노출한다. | P1 | job complete 전에는 다운로드가 404 또는 hidden이다. |

## 13. 비기능 요구사항

| ID | 요구사항 | 목표 | 우선순위 |
|---|---|---:|---:|
| NFR-001 | 웹 첫 화면 로딩 | 3초 이내 | P1 |
| NFR-002 | 카메라 미리보기 시작 | 5초 이내 | P1 |
| NFR-003 | 얼굴 조건 충족 후 캡처 | 기본 2초 안정 후 | P0 |
| NFR-004 | 개인 리포트 생성 | 120초 이내 목표 | P1 |
| NFR-005 | 궁합 리포트 생성 | 180초 이내 목표 | P1 |
| NFR-006 | LLM 실패 대응 | fallback 표시 | P0 |
| NFR-007 | 카메라 실패 안내 | 행동 지침 표시 | P0 |
| NFR-008 | 동시 촬영 제어 | job 1개만 허용 | P0 |
| NFR-009 | 프롬프트 유지보수 | config 수정으로 가능 | P1 |
| NFR-010 | 장치 없는 재현성 | mock capture 지원 | P0 |

## 14. 테스트 요구사항

| ID | 요구사항 | 우선순위 | 인수 기준 |
|---|---|---:|---|
| TEST-001 | 사주 런타임 계산을 테스트한다. | P0 | 기준 샘플의 년월일시주가 일치한다. |
| TEST-002 | 출생시간 미상 처리를 테스트한다. | P0 | 내부 대표값과 표시 문구가 분리된다. |
| TEST-003 | 성별 정규화를 테스트한다. | P0 | 한국어/영어 별칭이 처리된다. |
| TEST-004 | 로컬 LLM URL 제한을 테스트한다. | P0 | 외부 URL 설정은 오류가 난다. |
| TEST-005 | LLM JSON 파싱과 fence 제거를 테스트한다. | P0 | 정상/깨진/fenced JSON이 처리된다. |
| TEST-006 | LLM 실패 fallback을 테스트한다. | P0 | 예외 발생 시 기본 리포트가 생성된다. |
| TEST-007 | 잘린 최종 JSON을 테스트한다. | P0 | 부족한 필드는 fallback/default로 채운다. |
| TEST-008 | placeholder와 반복 block 방어를 테스트한다. | P0 | 스키마 문구가 화면에 노출되지 않는다. |
| TEST-009 | 랜드마크 룰 기반 관상을 테스트한다. | P0 | metric이 rule match와 payload로 변환된다. |
| TEST-010 | mock capture를 테스트한다. | P0 | mock 이미지와 preset metric이 생성된다. |
| TEST-011 | 웹 route smoke test를 제공한다. | P0 | `/`, `/personal`, `/compatibility`, `/health`가 응답한다. |
| TEST-012 | 실행 스크립트 wrapper 옵션을 테스트한다. | P0 | legacy distributed 옵션이 안전하게 소비된다. |
| TEST-013 | 분산 split 조립을 테스트한다. | P1 | 카테고리별 결과가 순서대로 합쳐진다. |
| TEST-014 | XSS escape를 테스트한다. | P0 | 사용자 입력과 LLM 출력이 실행되지 않는다. |

## 15. 완료 기준

| ID | 완료 조건 | 우선순위 |
|---|---|---:|
| DONE-001 | `./build.sh` 후 `./run.sh`로 웹 UI가 실행된다. | P0 |
| DONE-002 | 개인 리포트와 궁합 리포트가 각각 생성된다. | P0 |
| DONE-003 | 관상은 랜드마크 룰 기반 payload만 사용한다. | P0 |
| DONE-004 | 얼굴 이미지는 LLM/VLM payload로 전송되지 않는다. | P0 |
| DONE-005 | 사주 정보는 Python 런타임 계산 결과를 기준으로 한다. | P0 |
| DONE-006 | 출생시간 미상과 성별 별칭이 정상 처리된다. | P0 |
| DONE-007 | LLM 실패, 잘린 JSON, placeholder, 반복 block fallback이 동작한다. | P0 |
| DONE-008 | 리포트에 엔터테인먼트 목적과 중요 의사결정 금지 고지가 표시된다. | P0 |
| DONE-009 | release 모드에서 임시 산출물이 실행 후 삭제된다. | P0 |
| DONE-010 | `python -m pytest`가 통과한다. | P0 |
| DONE-011 | 카메라, 로컬 LLM, Raspberry Pi systemd는 실제 장치에서 별도 검증한다. | P1 |
