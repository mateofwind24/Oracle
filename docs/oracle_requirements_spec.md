# Oracle Report 요구사항 명세서

문서 버전: v3.0<br>
수정일: 2026-07-01<br>
기준: 현재 `release` 브랜치 구현<br>
작성 원칙: 기능/비기능을 사전 분류하지 않고, 중립 요구사항 ID(`R-01`, `R-02`, ...)로 단일 목록화한다.

| 요구사항 ID | 요구사항 내용 | 검증 기준 (완료 조건) | 시연 타임스탬프 |
|---|---|---|---|
| R-01 | 사용자는 웹 브라우저에서 Oracle Report 홈 화면에 접속할 수 있다. | `./run.sh` 실행 후 `http://<host>:8501/` 요청이 HTTP 200을 반환하고 홈 화면에 개인 리포트와 궁합 리포트 진입 카드가 표시된다. | N/A (시연 영상 미제공) |
| R-02 | 사용자는 개인 리포트 입력 화면으로 이동할 수 있다. | `/personal` 요청이 HTTP 200을 반환하고 이름, 생년월일, 태어난 시간, 성별 입력 폼이 표시된다. | N/A (시연 영상 미제공) |
| R-03 | 사용자는 두 사람 궁합 입력 화면으로 이동할 수 있다. | `/compatibility` 요청이 HTTP 200을 반환하고 두 사람의 이름, 생년월일, 태어난 시간, 성별, 관계 모드 입력 폼이 표시된다. | N/A (시연 영상 미제공) |
| R-04 | 개인 리포트 생성 요청은 비동기 작업으로 등록된다. | `/api/personal` POST 요청이 `job_id`와 `result_url`을 포함한 JSON을 반환한다. | N/A (시연 영상 미제공) |
| R-05 | 궁합 리포트 생성 요청은 비동기 작업으로 등록된다. | `/api/compatibility` POST 요청이 `job_id`와 `result_url`을 포함한 JSON을 반환한다. | N/A (시연 영상 미제공) |
| R-06 | 작업 상태는 API로 조회할 수 있다. | `/api/jobs/<job_id>` 요청이 `status`, `html`, `error`, `phase`, `message` 필드를 포함한 JSON을 반환한다. | N/A (시연 영상 미제공) |
| R-07 | 완료된 리포트는 HTML 파일로 다운로드할 수 있다. | 작업 상태가 `complete`일 때 `/api/jobs/<job_id>/download`가 `text/html; charset=utf-8` 콘텐츠와 attachment filename 헤더를 반환한다. | N/A (시연 영상 미제공) |
| R-08 | 완료되지 않은 리포트 다운로드는 차단된다. | 작업이 없거나 `complete`가 아닐 때 `/api/jobs/<job_id>/download`가 HTTP 404를 반환한다. | N/A (시연 영상 미제공) |
| R-09 | 웹 UI는 실시간 카메라 미리보기 스트림을 제공한다. | `/video-feed` 요청이 `multipart/x-mixed-replace` MJPEG 응답을 반환한다. | N/A (시연 영상 미제공) |
| R-10 | 얼굴 캡처 중 현재 상태와 안내 문구를 제공한다. | `/api/capture-debug` 응답에 `state`, `message`, `ready`, `metrics_text`, `observations_text`, `rules_text`, `warnings`가 포함된다. | N/A (시연 영상 미제공) |
| R-11 | 사용자는 랜드마크 비교 카메라 화면을 열 수 있다. | `./run.sh compare_camera` 또는 `./run.sh camera_compare` 실행 시 `/compare-camera` 화면이 열리고 실시간 측정 시작 버튼이 표시된다. | N/A (시연 영상 미제공) |
| R-12 | 랜드마크 비교 카메라를 API로 시작할 수 있다. | `/api/compare-camera/start` POST 요청이 JSON 응답을 반환하고 비교 카메라 작업이 시작된다. | N/A (시연 영상 미제공) |
| R-13 | 얼굴 캡처는 한 명의 얼굴만 안정적으로 보일 때 수행된다. | 얼굴 없음, 다중 얼굴, 가이드 위치/거리 불일치, 품질 경고 상태에서는 캡처 이미지가 저장되지 않는다. | N/A (시연 영상 미제공) |
| R-14 | 얼굴이 조건을 만족한 뒤 일정 시간 유지되면 자동 캡처된다. | `ORACLE_MIN_FACE_SECONDS` 이상 안정 상태가 유지되면 `capture.jpg`가 생성된다. | N/A (시연 영상 미제공) |
| R-15 | 궁합 리포트는 두 사람을 순차 캡처한다. | 궁합 workflow 실행 후 `person_1/capture.jpg`와 `person_2/capture.jpg`가 생성된다. | N/A (시연 영상 미제공) |
| R-16 | 카메라가 없는 환경에서도 mock capture로 재현할 수 있다. | `ORACLE_MOCK_CAPTURE_ENABLED=1` 또는 `./run.sh --mock-capture` 사용 시 실제 카메라 없이 mock 이미지와 랜드마크 metric이 생성된다. | N/A (시연 영상 미제공) |
| R-17 | 궁합 mock capture는 첫 번째/두 번째 사람의 metric preset을 분리한다. | `ORACLE_MOCK_PAIR_LEFT_LANDMARK_METRICS_JSON`, `ORACLE_MOCK_PAIR_RIGHT_LANDMARK_METRICS_JSON` 설정값이 각각 `person_1`, `person_2` 캡처 품질 분석에 반영된다. | N/A (시연 영상 미제공) |
| R-18 | Linux 카메라 권한 문제를 자동 보정하려고 시도한다. | `/dev/video*` 장치가 있으나 현재 사용자에게 읽기/쓰기 권한이 없으면 ACL 또는 chmod 보정 시도를 로그로 남긴다. | N/A (시연 영상 미제공) |
| R-19 | 태어난 시간 미상을 입력할 수 있다. | 태어난 시간이 `모름`, `미상`, `unknown`, `none`, 빈 값이면 내부 계산은 `12:30`을 사용하고 화면에는 시간 미상 문구를 표시한다. | N/A (시연 영상 미제공) |
| R-20 | 시진명 입력은 대표 시간으로 변환된다. | `오시`, `午時`, `오시(午時)` 입력이 `12:30`으로 변환되어 사주 계산에 사용된다. | N/A (시연 영상 미제공) |
| R-21 | 성별 입력은 한국어와 영어 별칭을 처리한다. | `남`, `남자`, `남성`, `male`, `m`, `여`, `여자`, `여성`, `female`, `f`가 정상 정규화된다. | N/A (시연 영상 미제공) |
| R-22 | 허용되지 않은 성별 입력은 오류로 처리된다. | 성별이 남성/여성 계열 값이 아니면 workflow가 `ValueError`를 발생시키고 오류 패널이 표시된다. | N/A (시연 영상 미제공) |
| R-23 | 궁합 관계 모드는 허용된 값만 사용할 수 있다. | 관계 모드가 `연인`, `친구`, `직장동료` 중 하나가 아니면 workflow가 오류를 반환한다. | N/A (시연 영상 미제공) |
| R-24 | 사주 명식은 Python 런타임 엔진에서 계산된다. | `ManseRepository.lookup()` 호출 결과에 년주, 월주, 일주, 시주가 포함된다. | N/A (시연 영상 미제공) |
| R-25 | 사주 계산은 오행 분포를 반환한다. | `ManseLookupResult.reading.element_counts`가 목, 화, 토, 금, 수 5개 오행 count를 포함한다. | N/A (시연 영상 미제공) |
| R-26 | 사주 계산은 대운 방향을 반환한다. | `ManseLookupResult.daeun_direction`이 `순행` 또는 `역행`으로 설정된다. | N/A (시연 영상 미제공) |
| R-27 | 절기 경계가 사주 계산에 반영된다. | 테스트 기준일에서 입춘과 절기 경계에 따라 년주/월주가 기대값과 일치한다. | N/A (시연 영상 미제공) |
| R-28 | 관상 분석은 랜드마크 룰 기반 payload로 생성된다. | 캡처 품질 정보의 `landmark_matches_json`으로부터 개인 `face_subtitle`, `face_blocks` JSON이 생성된다. | N/A (시연 영상 미제공) |
| R-29 | 궁합 관상 분석은 두 사람의 룰 매칭 결과를 함께 사용한다. | 두 캡처의 `landmark_matches_json`으로부터 `pair_subtitle`, `pair_blocks` JSON이 생성된다. | N/A (시연 영상 미제공) |
| R-30 | 얼굴 이미지는 LLM 요청 payload에 포함되지 않는다. | `LlamaCppChatClient._build_payload()`가 `image_path` 인자를 무시하고 text-only messages를 생성한다. | N/A (시연 영상 미제공) |
| R-31 | LLM base URL은 로컬 주소만 허용된다. | `ORACLE_LLM_BASE_URL` 또는 `ORACLE_REPORT_LLM_BASE_URL`이 `localhost`, `127.0.0.1`, `::1` 외 host이면 설정 로드가 실패한다. | N/A (시연 영상 미제공) |
| R-32 | 로컬 llama.cpp 서버를 자동 시작할 수 있다. | 기존 `/models` 응답이 없고 `ORACLE_START_LLAMA_SERVER=1`이면 `llama-server`가 모델 경로와 지정된 port로 실행된다. | N/A (시연 영상 미제공) |
| R-33 | 이미 실행 중인 로컬 LLM 서버는 재사용된다. | `/models` 확인이 성공하면 `run.sh`가 새 `llama-server`를 시작하지 않는다. | N/A (시연 영상 미제공) |
| R-34 | llama.cpp는 CPU 실행을 기본값으로 사용한다. | `-ngl/--ngl` 옵션을 주지 않으면 GPU offload 인자를 추가하지 않고 CPU 기본 실행 로그를 남긴다. | N/A (시연 영상 미제공) |
| R-35 | llama.cpp thread 수를 실행 옵션으로 조정할 수 있다. | `./run.sh --threads N` 실행 시 `llama-server` 인자에 `-t N`이 포함된다. | N/A (시연 영상 미제공) |
| R-36 | llama.cpp parallel slot 수를 실행 옵션으로 조정할 수 있다. | `./run.sh --parallel N` 실행 시 `llama-server` 인자에 `--parallel N`이 포함된다. | N/A (시연 영상 미제공) |
| R-37 | reasoning mode는 명시적으로 켤 수 있다. | `./run.sh --reasoning` 또는 `ORACLE_REASONING=1` 사용 시 `llama-server` 인자에 reasoning on 설정이 포함된다. | N/A (시연 영상 미제공) |
| R-38 | 운영 프롬프트는 manifest include 구조로 관리된다. | `configs/prompts.json`이 `prompts_personal.json`, `prompts_compatibility.json`을 include하고 중복 template 이름은 오류 처리된다. | N/A (시연 영상 미제공) |
| R-39 | 개인 사주 프롬프트는 별도 파일에서 로드된다. | `saju_reading` 렌더링 시 source 로그가 `configs/prompts_personal.json`을 가리킨다. | N/A (시연 영상 미제공) |
| R-40 | 궁합 사주 프롬프트는 별도 파일에서 로드된다. | `saju_reading_couple` 렌더링 시 source 로그가 `configs/prompts_compatibility.json`을 가리킨다. | N/A (시연 영상 미제공) |
| R-41 | 프롬프트 렌더링 로그를 제공한다. | 프롬프트 렌더링 시 stderr에 `[PROMPT] name=... source=... id_slot=... prefix_chars=... body_chars=...` 형식 로그가 출력된다. | N/A (시연 영상 미제공) |
| R-42 | 프롬프트 token 크기를 확인할 수 있다. | `./run.sh token`이 각 프롬프트의 `id_slot`, prefix token, body token, full token 정보를 표로 출력한다. | N/A (시연 영상 미제공) |
| R-43 | llama.cpp 서버 없이 token 크기를 추정할 수 있다. | `./run.sh token --offline`이 `/tokenize` 호출 없이 estimated token count를 출력한다. | N/A (시연 영상 미제공) |
| R-44 | LLM 응답이 비어 있으면 실패로 처리된다. | 빈 응답을 받은 경우 `empty LLM response` 오류가 발생하고 사용자 화면에는 fallback 리포트가 표시된다. | N/A (시연 영상 미제공) |
| R-45 | LLM 응답이 길이 제한으로 중단되면 실패로 처리된다. | `finish_reason=length` 응답은 incomplete response 오류로 처리된다. | N/A (시연 영상 미제공) |
| R-46 | LLM JSON 응답의 일반적인 형식 오류를 보정한다. | fenced JSON, 일부 trailing comma, 일부 누락 comma, block 배열 닫힘 오류가 보정 가능한 경우 payload로 파싱된다. | N/A (시연 영상 미제공) |
| R-47 | 사주 LLM 결과의 어려운 일간 표현은 생활 언어로 완화된다. | 생성 결과에 `갑목`, `을목`, `병화` 등 일간 용어가 있으면 자연물/생활 언어 표현으로 후처리된다. | N/A (시연 영상 미제공) |
| R-48 | 개인 리포트는 full HTML과 fragment HTML을 모두 생성한다. | 개인 workflow 결과에 `report_html`, `report_fragment_html`, `output_path`가 포함되고 `personal_report.html`이 저장된다. | N/A (시연 영상 미제공) |
| R-49 | 궁합 리포트는 full HTML과 fragment HTML을 모두 생성한다. | 궁합 workflow 결과에 `report_html`, `report_fragment_html`, `output_path`가 포함되고 `compatibility_report.html`이 저장된다. | N/A (시연 영상 미제공) |
| R-50 | 개인 리포트는 사주 블록 6개를 표시한다. | 개인 결과 HTML에 종합 형국, 타고난 성향과 심리 패턴, 재물운과 적성, 연애운과 인간관계, 올해의 운세, 총평 및 인생의 조언 블록이 표시된다. | N/A (시연 영상 미제공) |
| R-51 | 궁합 리포트는 사주 블록 6개를 표시한다. | 궁합 결과 HTML에 관계 구조, 상호 보완, 갈등 관리, 현재 관계 흐름, 실천 제안, 총평 및 조언 블록이 표시된다. | N/A (시연 영상 미제공) |
| R-52 | 개인 얼굴 생략 리포트는 관상 섹션 없이 완성된다. | `skip_face=True` 개인 workflow 결과 HTML에 관상 섹션이 숨겨지고 사주 중심 리포트가 표시된다. | N/A (시연 영상 미제공) |
| R-53 | 궁합 리포트는 궁합 점수를 표시한다. | 궁합 결과 payload에 `compatibility_score`, `compatibility_saju_score`, `compatibility_face_score`, `compatibility_mode_bonus`가 포함되고 HTML score card에 표시된다. | N/A (시연 영상 미제공) |
| R-54 | 궁합 점수는 입력 조합에 따라 달라진다. | 서로 다른 생년월일시/랜드마크 metric 조합 4개 이상에서 최소 3개 이상의 서로 다른 총점이 산출된다. | N/A (시연 영상 미제공) |
| R-55 | 궁합 점수 요약은 관계 모드에 따라 달라진다. | `친구` 모드 요약에는 친구 문맥이, `직장동료` 모드 요약에는 업무/협업 문맥이 포함된다. | N/A (시연 영상 미제공) |
| R-56 | 리포트에는 엔터테인먼트 목적과 중요 의사결정 금지 고지가 표시된다. | 결과 HTML에 의료, 법률, 재정, 채용, 평가 목적 사용 금지 또는 참고용 고지가 포함된다. | N/A (시연 영상 미제공) |
| R-57 | 사용자 입력과 LLM 출력은 HTML escape 처리된다. | `<script>`가 포함된 입력이나 생성 텍스트가 결과 HTML에서 실행 가능한 스크립트로 렌더링되지 않는다. | N/A (시연 영상 미제공) |
| R-58 | placeholder 문구는 결과 화면에 노출되지 않는다. | `제목`, `요약`, `본문`, `1~2문장` 같은 schema placeholder가 결과 HTML에 그대로 표시되지 않는다. | N/A (시연 영상 미제공) |
| R-59 | 반복된 summary/body 문장은 보정된다. | body가 summary와 같은 문장으로 시작하면 해당 prefix가 제거된 텍스트가 렌더링된다. | N/A (시연 영상 미제공) |
| R-60 | workflow 단계별 소요 시간을 기록한다. | 개인/궁합 workflow 완료 후 세션 디렉터리에 `timings.log`가 생성되고 단계명과 elapsed seconds가 기록된다. | N/A (시연 영상 미제공) |
| R-61 | debug 모드는 실행 산출물을 보관한다. | `./run.sh debug <cmd>` 실행 시 `runs/debug/<timestamp>/command.txt`, `output.log`, `artifacts/`가 생성된다. | N/A (시연 영상 미제공) |
| R-62 | release 모드는 임시 산출물을 삭제한다. | `./run.sh release <cmd>` 실행 종료 후 임시 output directory가 삭제된다. | N/A (시연 영상 미제공) |
| R-63 | release 모드는 명시적 output path 인자를 거부한다. | `./run.sh release ... --output` 또는 `--output-dir` 사용 시 오류 메시지와 함께 실행이 중단된다. | N/A (시연 영상 미제공) |
| R-64 | 분산 worker는 생성 API를 제공한다. | `/api/distributed/generate` POST 요청이 prompt 정보를 받아 LLM 생성 결과 또는 오류 JSON을 반환한다. | N/A (시연 영상 미제공) |
| R-65 | 분산 worker는 상태 API를 제공한다. | `/api/distributed/status` 요청이 `status`, `tps`, `compute_score`, `model`을 포함한 JSON을 반환한다. | N/A (시연 영상 미제공) |
| R-66 | 분산 사주 해설은 metadata와 카테고리 작업으로 나눌 수 있다. | `render_distributed_prompt_template()`가 metadata 전용 prompt와 특정 category 전용 prompt를 각각 생성한다. | N/A (시연 영상 미제공) |
| R-67 | 분산 결과는 원래 카테고리 순서로 재조립된다. | 분산 작업 결과가 누락되어도 기본 오류 block을 채워 정해진 카테고리 순서의 `saju_blocks`를 만든다. | N/A (시연 영상 미제공) |
| R-68 | 분산 실행은 명시적으로 켤 때만 활성화된다. | `ORACLE_DISTRIBUTED_SPLIT=1`이고 role이 `master` 또는 `hybrid`일 때만 분산 생성 경로를 사용한다. | N/A (시연 영상 미제공) |
| R-69 | speculative work stealing은 명시적으로 켤 수 있다. | `--speculative` 또는 `ORACLE_DISTRIBUTED_SPECULATIVE=1` 설정 시 분산 작업의 speculative 경로가 활성화된다. | N/A (시연 영상 미제공) |
| R-70 | compute score는 실측 또는 수동 override로 계산된다. | `ORACLE_COMPUTE_SCORE`가 있으면 해당 값을 사용하고, 없으면 LLM TPS benchmark와 모델 크기 휴리스틱으로 계산한다. | N/A (시연 영상 미제공) |
| R-71 | 애플리케이션 health check를 제공한다. | `/health` 요청이 HTTP 200과 `ok` 텍스트를 반환한다. | N/A (시연 영상 미제공) |
| R-72 | 테스트 실행 결과를 파일로 보관할 수 있다. | `python -m pytest 2>&1 | tee test-results/pytest-latest.txt` 실행 시 테스트 결과 로그가 `test-results/pytest-latest.txt`에 저장된다. | N/A (시연 영상 미제공) |
