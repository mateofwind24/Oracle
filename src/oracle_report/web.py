from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import threading
import uuid

from flask import Flask, Response, jsonify, redirect, request
from markupsafe import escape

from oracle_report.config import (
    load_app_config,
    load_capture_config,
    load_report_llm_config,
)
from oracle_report.workflow import (
    COMPATIBILITY_MODES,
    CompatibilityWorkflowInput,
    PersonalWorkflowResult,
    PersonalWorkflowInput,
    run_compatibility_workflow,
    run_personal_workflow,
)
from oracle_report.saju.repository import (
    MANSE_TIME_BRANCH_LABELS,
    time_branch_range_display_from_index,
)
from oracle_report.vision.runtime import run_capture


@dataclass
class _WorkflowJob:
    status: str
    html: str = ""
    error: str = ""
    phase: str = ""
    message: str = ""
    download_html: str = ""
    download_filename: str = "oracle_report.html"


class _PreviewStream:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame: bytes | None = None
        self._version = 0

    def publish(self, cv2, frame) -> None:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 75],
        )
        if ok:
            with self._condition:
                self._frame = encoded.tobytes()
                self._version = self._version + 1
                self._condition.notify_all()

    def frames(self):
        last_version = -1
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._frame is not None
                    and self._version != last_version,
                    timeout=1.0,
                )
                frame = self._frame
                last_version = self._version
            if frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )


_PREVIEW_STREAM = _PreviewStream()
_CAPTURE_LOCK = threading.Lock()
_JOBS_LOCK = threading.Lock()
_JOBS: dict[str, _WorkflowJob] = {}


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        body = f"""
        <div class="oracle-home-shell">
          <section id="oracle-home" class="home-view" aria-label="홈">
            <header class="home-nav">
              <div class="brand-lockup">
                <div class="logo">ORACLE<span class="stamp serif">運</span></div>
                <div class="tag">관상 &amp; 사주 · 운명 해설</div>
              </div>
            </header>

            <section class="home-hero">
              <div class="speech">안녕하세요!</div>
              <h1 aria-label="오늘도 운명을 함께 찾아볼까요?">오늘도 <span>운명</span>을<br>함께 찾아볼까요?</h1>
              <p aria-label="오라와 함께 나의 운명과 인연을 쉽고 재미있게 알아보세요!"><strong>오라</strong>와 함께 나의 운명과 인연을<br>쉽고 재미있게 알아보세요!</p>
              <img class="oracle-character" src="/static/assets/oracle-character.png" alt="돋보기로 운명을 살피는 오라 캐릭터">
            </section>

            <div class="cards">
              <a class="mode solo" href="/personal">
                <div class="mode-copy">
                  <span class="ic">☘</span>
                  <h2>나의 운세 보기</h2>
                  <p>사주 · 관상 · 운세를<br>종합적으로 분석해드려요!</p>
                  <span class="go">시작하기 <span class="arr">→</span></span>
                </div>
                <img class="mode-art mode-art-solo" src="/static/assets/oracle-solo-card.png" alt="" aria-hidden="true">
              </a>

              <a class="mode pair" href="/compatibility">
                <div class="mode-copy">
                  <span class="ic">♥</span>
                  <h2>우리 궁합 보기</h2>
                  <p>두 사람의 인연과 궁합을<br>AI가 정밀하게 분석해드려요!</p>
                  <span class="go">시작하기 <span class="arr">→</span></span>
                </div>
                <img class="mode-art mode-art-pair" src="/static/assets/oracle-pair-card.png" alt="" aria-hidden="true">
              </a>
            </div>

            <div class="feature-row">
              <div class="feature">
                <span class="feature-ic camera"><img src="/static/assets/camera.png" width="35" height="35" alt="" aria-hidden="true"></span>
                <strong>얼굴 분석</strong>
                <p>사진 한 장으로<br>관상을 분석해요</p>
              </div>
              <div class="feature">
                <span class="feature-ic calendar"><img src="/static/assets/calendar.png" width="35" height="35" alt="" aria-hidden="true"></span>
                <strong>사주 분석</strong>
                <p>생년월일시를 기반으로<br>사주를 분석해요</p>
              </div>
              <div class="feature">
                <span class="feature-ic ai"><img src="/static/assets/ai.png" width="35" height="35" alt="" aria-hidden="true"></span>
                <strong>100% 온디바이스</strong>
                <p>모든 분석이 기기 내에서<br>이루어져 안전해요</p>
              </div>
              <div class="feature">
                <span class="feature-ic lock"><img src="/static/assets/privacy.png" width="35" height="35" alt="" aria-hidden="true"></span>
                <strong>프라이버시 보호</strong>
                <p>당신의 데이터는 외부로<br>전송되지 않아요</p>
              </div>
            </div>
          </section>

          <section id="oracle-more" class="more-section" aria-label="더보기">
            <div class="more-head">
              <div class="logo">ORACLE<span class="stamp serif">運</span></div>
              <h2>오라를 더 자세히 소개할게요!</h2>
              <p>분석 방식부터 진행 과정, 자주 묻는 질문까지 한 번에 확인해보세요.</p>
            </div>

            <div class="more-grid">
              <article class="more-card">
                <div class="more-copy">
                  <span class="more-kicker">ORACLE은 이렇게 달라요</span>
                  <h3>전통 지혜와 AI 기술의 완벽한 만남</h3>
                  <ul>
                    <li><span>◎</span><strong>정확하고 신뢰할 수 있는 AI 분석</strong><em>전통 명리학과 AI 기술을 결합해 분석해요.</em></li>
                    <li><span>▤</span><strong>전통 명리학 기반의 깊이 있는 해석</strong><em>오랜 지식을 AI가 체계적으로 풀어줘요.</em></li>
                    <li><span>♡</span><strong>쉽고 귀여운 UI</strong><em>복잡한 내용을 누구나 즐겁게 이해할 수 있어요.</em></li>
                    <li><span>▣</span><strong>완벽한 프라이버시 보호</strong><em>모든 데이터는 기기 안에서 안전하게 처리돼요.</em></li>
                  </ul>
                  <div class="more-note">
                    <img src="/static/assets/oracle-pair-card.png" alt="" aria-hidden="true">
                    <p><strong>당신의 운명은 특별합니다.</strong><br>오라와 함께 더 나은 미래를 만들어가요!</p>
                  </div>
                </div>
              </article>

              <article class="more-card">
                <div class="more-copy">
                  <span class="more-kicker">분석 진행 과정</span>
                  <h3>오라와 함께 운명을 분석하는 과정</h3>
                  <ol>
                    <li><span>1</span><strong>얼굴 인식</strong><em>사진을 촬영하거나 업로드해요.</em></li>
                    <li><span>2</span><strong>정보 입력</strong><em>생년월일시 등 필요한 정보를 입력해요.</em></li>
                    <li><span>3</span><strong>AI 분석</strong><em>AI가 사주와 관상을 종합 분석해요.</em></li>
                    <li><span>4</span><strong>결과 확인</strong><em>나만의 운세 리포트를 확인해요.</em></li>
                    <li><span>5</span><strong>안전한 저장</strong><em>결과는 기기 내에서만 안전하게 저장돼요.</em></li>
                  </ol>
                  <div class="more-note">
                    <img src="/static/assets/oracle-solo-card.png" alt="" aria-hidden="true">
                    <p><strong>모든 과정이 기기 내에서 안전하게!</strong><br>당신의 프라이버시를 최우선으로 생각해요.</p>
                  </div>
                </div>
              </article>

              <article class="more-card">
                <div class="more-copy">
                  <span class="more-kicker">오라가 도와줄게요</span>
                  <h3>궁금할 때 바로 확인하는 도움말</h3>
                  <div class="faq-list">
                    <div><span>▣</span><strong>얼굴 사진은 어떻게 찍어야 하나요?</strong><em>정면을 바라보고 밝은 곳에서 찍어주세요.</em></div>
                    <div><span>▦</span><strong>사주 정보는 어떻게 입력하나요?</strong><em>정확한 생년월일시를 입력해주시면 돼요.</em></div>
                    <div><span>▤</span><strong>분석 결과는 어디서 확인하나요?</strong><em>운세 리포트에서 언제든 확인할 수 있어요.</em></div>
                    <div><span>▣</span><strong>개인 정보는 안전한가요?</strong><em>100% 온디바이스 처리로 안전해요.</em></div>
                  </div>
                  <div class="more-note">
                    <img src="/static/assets/oracle-pair-card.png" alt="" aria-hidden="true">
                    <p><strong>오라는 언제나 여러분의 운명 친구예요!</strong><br>신뢰할 수 있는 AI 분석을 약속드려요.</p>
                  </div>
                </div>
              </article>
            </div>
          </section>

          {_bottom_nav("home", home_tabs=True)}
        </div>
        """
        result = _render_page("Oracle", body, page_class="home-page", show_heading=False)
        return result

    @app.route("/personal", methods=["GET", "POST"])
    def personal():
        body = _personal_form()
        if request.method == "POST":
            try:
                workflow_input = _personal_workflow_input_from_form()
                job_id = _start_personal_workflow_job(workflow_input)
                result = redirect(
                    _personal_result_url(job_id, workflow_input.skip_face),
                    code=303,
                )
                return result
            except Exception as exc:
                body = _error_panel(exc) + _personal_form()
        result = _render_page(
            "개인 리포트",
            body,
            page_class="input-page personal-page",
            show_heading=False,
        )
        return result

    @app.post("/api/personal")
    def personal_api():
        workflow_input = _personal_workflow_input_from_form()
        job_id = _start_personal_workflow_job(workflow_input)
        result = jsonify(
            {
                "job_id": job_id,
                "result_url": _personal_result_url(job_id, workflow_input.skip_face),
            },
        )
        return result

    @app.get("/personal/result/<job_id>")
    def personal_result_page(job_id: str):
        skip_face = _query_bool("skip_face", False)
        body = _personal_result_page(job_id, skip_face)
        result = _render_page(
            "개인 리포트 결과",
            body,
            page_class="result-page personal-result-page",
            show_heading=False,
        )
        return result

    @app.route("/compatibility", methods=["GET", "POST"])
    def compatibility():
        body = _compatibility_form()
        if request.method == "POST":
            try:
                workflow_input = _compatibility_workflow_input_from_form()
                job_id = _start_compatibility_workflow_job(workflow_input)
                result = redirect(
                    _compatibility_result_url(job_id),
                    code=303,
                )
                return result
            except Exception as exc:
                body = _error_panel(exc) + _compatibility_form()
        result = _render_page(
            "두 사람 궁합",
            body,
            page_class="input-page compatibility-page",
            show_heading=False,
        )
        return result

    @app.post("/api/compatibility")
    def compatibility_api():
        workflow_input = _compatibility_workflow_input_from_form()
        job_id = _start_compatibility_workflow_job(workflow_input)
        result = jsonify(
            {
                "job_id": job_id,
                "result_url": _compatibility_result_url(job_id),
            },
        )
        return result

    @app.get("/compatibility/result/<job_id>")
    def compatibility_result_page(job_id: str):
        body = _compatibility_result_page(job_id)
        result = _render_page(
            "두 사람 궁합 결과",
            body,
            page_class="result-page",
            show_heading=False,
        )
        return result

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str):
        job = _get_job(job_id)
        status_code = 200
        payload = {
            "status": "missing",
            "html": "",
            "error": "job not found",
            "phase": "",
            "message": "",
        }
        if job is not None:
            payload = {
                "status": job.status,
                "html": job.html,
                "error": job.error,
                "phase": job.phase,
                "message": job.message,
            }
        else:
            status_code = 404
        result = jsonify(payload), status_code
        return result

    @app.get("/api/jobs/<job_id>/download")
    def job_download(job_id: str):
        job = _get_job(job_id)
        if job is None or job.status != "complete" or job.download_html == "":
            result = ("report not ready", 404)
        else:
            result = Response(
                job.download_html,
                content_type="text/html; charset=utf-8",
                headers={
                    "Content-Disposition": (
                        f'attachment; filename="{job.download_filename}"'
                    ),
                },
            )
        return result

    @app.get("/video-feed")
    def video_feed():
        result = Response(
            _PREVIEW_STREAM.frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
        return result

    @app.get("/health")
    def health():
        result = "ok"
        return result

    @app.get("/favicon.ico")
    def favicon():
        result = ("", 204)
        return result

    @app.post("/api/distributed/generate")
    def distributed_generate():
        payload = request.json or {}
        prompt_name = payload.get("prompt_name")
        target_category = payload.get("target_category")
        is_metadata = payload.get("is_metadata", False)
        values = payload.get("values", {})
        image_base64 = payload.get("image_base64")

        from oracle_report.prompt_templates import render_distributed_prompt_template
        rendered = render_distributed_prompt_template(
            name=prompt_name,
            values=values,
            target_category=target_category,
            is_metadata=is_metadata,
        )

        temp_img_path = None
        if image_base64:
            import base64
            img_data = base64.b64decode(image_base64)
            temp_dir = Path("runs/temp")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_img_path = temp_dir / f"distributed_temp_{uuid.uuid4().hex}.jpg"
            temp_img_path.write_bytes(img_data)

        from oracle_report.llm import LlamaCppChatClient
        from oracle_report.config import load_report_llm_config

        llm_config = load_report_llm_config()
        client = LlamaCppChatClient(llm_config)

        try:
            output = client.generate(rendered, image_path=temp_img_path)
            result = jsonify({"status": "success", "output": output})
        except Exception as exc:
            result = jsonify({"status": "error", "error": str(exc)}), 500
        finally:
            if temp_img_path and temp_img_path.exists():
                try:
                    temp_img_path.unlink()
                except Exception:
                    pass
        return result

    @app.get("/api/distributed/status")
    def distributed_status():
        from oracle_report.llm import is_local_llm_running, LlamaCppChatClient
        from oracle_report.config import load_llm_config
        is_busy = _CAPTURE_LOCK.locked() or is_local_llm_running()
        
        llm_config = load_llm_config()
        client = LlamaCppChatClient(llm_config)
        
        tps = 1.0
        score = 2.0
        model_name = llm_config.model
        if not is_busy:
            try:
                tps = client.get_or_measure_tps()
                score = client.get_compute_score()
            except Exception:
                pass
                
        result = jsonify({
            "status": "busy" if is_busy else "idle",
            "tps": tps,
            "compute_score": score,
            "model": model_name
        })
        return result

    result = app
    return result


def serve() -> None:
    config = load_app_config()

    if config.distributed_role in ("master", "hybrid") and config.distributed_warmup:
        import threading
        def run_warmup_background():
            import time
            time.sleep(5.0)
            print("[Distributed] Starting LLM warmup for distributed nodes...", flush=True)
            try:
                from oracle_report.workflow import _generate_distributed
                dummy_values = {
                    "name": "더미",
                    "gender": "남성",
                    "timezone": "KST",
                    "saju_text": "사주 테스트 텍스트",
                }
                dummy_categories = [
                    "종합 형국",
                    "타고난 성향과 심리 패턴",
                    "재물운과 적성",
                    "연애운과 인간관계",
                    "올해의 운세",
                    "총평 및 인생의 조언",
                ]
                _generate_distributed(
                    prompt_name="saju_reading",
                    values=dummy_values,
                    categories=dummy_categories,
                    image_path=None,
                    app_config=config,
                )
                print("[Distributed] LLM warmup complete.", flush=True)
            except Exception as e:
                print(f"[Distributed][Error] LLM warmup failed: {e}", flush=True)

        t = threading.Thread(target=run_warmup_background, daemon=True)
        t.start()

    app = create_app()
    app.run(host=config.host, port=config.port, debug=config.debug, threaded=True)


def _form_value(name: str) -> str:
    result = request.form.get(name, "").strip()
    return result


def _form_bool(name: str, default: bool) -> bool:
    raw_value = _form_value(name)
    result = default
    if raw_value != "":
        result = raw_value.lower() in ("1", "true", "yes", "y", "on")
    return result


def _query_bool(name: str, default: bool) -> bool:
    raw_value = request.args.get(name, "").strip()
    result = default
    if raw_value != "":
        result = raw_value.lower() in ("1", "true", "yes", "y", "on")
    return result


def _personal_workflow_input_from_form() -> PersonalWorkflowInput:
    result = PersonalWorkflowInput(
        name=_form_value("name"),
        birth_date=_form_value("birth_date"),
        birth_time=_form_value("birth_time"),
        gender=_form_value("gender"),
        target_gender=_form_value("target_gender"),
        skip_face=_form_bool("skip_face", False),
    )
    return result


def _compatibility_workflow_input_from_form() -> CompatibilityWorkflowInput:
    result = CompatibilityWorkflowInput(
        left_name=_form_value("left_name"),
        left_birth_date=_form_value("left_birth_date"),
        left_birth_time=_form_value("left_birth_time"),
        left_gender=_form_value("left_gender"),
        right_name=_form_value("right_name"),
        right_birth_date=_form_value("right_birth_date"),
        right_birth_time=_form_value("right_birth_time"),
        right_gender=_form_value("right_gender"),
        mode=_form_value("mode"),
    )
    return result


def _start_personal_workflow_job(workflow_input: PersonalWorkflowInput) -> str:
    job_id = uuid.uuid4().hex
    initial_phase = "generating" if workflow_input.skip_face else "capturing"
    initial_message = (
        "사주 리포트 생성 중"
        if workflow_input.skip_face
        else "얼굴을 카메라 중앙에 맞춰 주세요"
    )

    def capture_runner(config, output_dir: Path | None = None):
        capture_artifact = _preview_capture_runner(config, output_dir)
        _set_job(
            job_id,
            _WorkflowJob(
                status="running",
                phase="generating",
                message="얼굴 인식이 완료되어 리포트를 생성하고 있습니다",
            ),
        )
        return capture_artifact

    def run_job() -> _WorkflowJob:
        workflow_result = run_personal_workflow(
            workflow_input=workflow_input,
            capture_config=load_capture_config(),
            report_llm_config=load_report_llm_config(),
            manse_db_path=_manse_db_path(),
            recommendation_db_path=_face_db_path(),
            capture_runner=capture_runner,
        )
        result = _WorkflowJob(
            status="complete",
            html=_personal_result(workflow_result),
            download_html=workflow_result.report_html,
            download_filename=workflow_result.output_path.name,
        )
        return result

    result = _start_workflow_job(
        run_job,
        job_id=job_id,
        initial_job=_WorkflowJob(
            status="running",
            phase=initial_phase,
            message=initial_message,
        ),
    )
    return result


def _start_compatibility_workflow_job(
    workflow_input: CompatibilityWorkflowInput,
) -> str:
    job_id = uuid.uuid4().hex
    capture_count = 0

    def capture_runner(config, output_dir: Path | None = None):
        nonlocal capture_count
        capture_artifact = _preview_capture_runner(config, output_dir)
        capture_count = capture_count + 1
        if capture_count == 1:
            _set_job(
                job_id,
                _WorkflowJob(
                    status="running",
                    phase="capturing",
                    message="첫 번째 촬영이 완료되었습니다. 두 번째 사람 촬영을 준비해 주세요",
                ),
            )
        else:
            _set_job(
                job_id,
                _WorkflowJob(
                    status="running",
                    phase="generating",
                    message="두 사람 얼굴 인식이 완료되어 리포트를 생성하고 있습니다",
                ),
            )
        return capture_artifact

    def run_job() -> _WorkflowJob:
        workflow_result = run_compatibility_workflow(
            workflow_input=workflow_input,
            capture_config=load_capture_config(),
            report_llm_config=load_report_llm_config(),
            manse_db_path=_manse_db_path(),
            capture_runner=capture_runner,
        )
        result = _WorkflowJob(
            status="complete",
            html=_compatibility_result(
                workflow_result.markdown,
                workflow_result,
            ),
            download_html=workflow_result.report_html,
            download_filename=workflow_result.output_path.name,
        )
        return result

    result = _start_workflow_job(
        run_job,
        job_id=job_id,
        initial_job=_WorkflowJob(
            status="running",
            phase="capturing",
            message="첫 번째 사람의 얼굴을 카메라 중앙에 맞춰 주세요",
        ),
    )
    return result


def _personal_result_url(job_id: str, skip_face: bool) -> str:
    skip_value = "1" if skip_face else "0"
    result = f"/personal/result/{job_id}?skip_face={skip_value}"
    return result


def _compatibility_result_url(job_id: str) -> str:
    result = f"/compatibility/result/{job_id}"
    return result


def _preview_capture_runner(config, output_dir: Path | None = None):
    result = run_capture(
        config,
        output_dir=output_dir,
        frame_callback=_PREVIEW_STREAM.publish,
    )
    return result


def _start_workflow_job(
    run_job,
    *,
    job_id: str | None = None,
    initial_job: _WorkflowJob | None = None,
) -> str:
    active_job_id = job_id or uuid.uuid4().hex
    _set_job(active_job_id, initial_job or _WorkflowJob(status="running"))
    thread = threading.Thread(
        target=_run_workflow_job,
        args=(active_job_id, run_job),
        daemon=True,
    )
    thread.start()
    result = active_job_id
    return result


def _run_workflow_job(job_id: str, run_job) -> None:
    acquired = _CAPTURE_LOCK.acquire(blocking=False)
    if not acquired:
        _set_job(
            job_id,
            _WorkflowJob(
                status="error",
                html=_error_panel(RuntimeError("다른 촬영 작업이 진행 중입니다.")),
                error="다른 촬영 작업이 진행 중입니다.",
            ),
        )
    else:
        try:
            job_result = run_job()
            if isinstance(job_result, _WorkflowJob):
                completed_job = job_result
            else:
                completed_job = _WorkflowJob(status="complete", html=job_result)
            _set_job(job_id, completed_job)
        except Exception as exc:
            _set_job(
                job_id,
                _WorkflowJob(
                    status="error",
                    html=_error_panel(exc),
                    error=str(exc),
                ),
            )
        finally:
            _CAPTURE_LOCK.release()


def _set_job(job_id: str, job: _WorkflowJob) -> None:
    with _JOBS_LOCK:
        _JOBS[job_id] = job


def _get_job(job_id: str) -> _WorkflowJob | None:
    with _JOBS_LOCK:
        result = _JOBS.get(job_id)
    return result


def _manse_db_path() -> Path:
    result = Path(".")
    return result


def _face_db_path() -> Path:
    result = Path(
        os.getenv("ORACLE_FACE_DB_PATH", "data/face_recommendations.sqlite"),
    )
    return result


def _bottom_nav(active: str, *, home_tabs: bool = False) -> str:
    home_href = "#oracle-home" if home_tabs else "/#oracle-home"
    more_href = "#oracle-more" if home_tabs else "/#oracle-more"
    home_tab = ' data-home-tab="home"' if home_tabs else ""
    more_tab = ' data-home-tab="more"' if home_tabs else ""
    home_active = " foot-item-active" if active == "home" else ""
    personal_active = " foot-item-active" if active == "personal" else ""
    compatibility_active = " foot-item-active" if active == "compatibility" else ""
    more_active = " foot-item-active" if active == "more" else ""
    result = f"""
    <footer class="home-foot" aria-label="하단 메뉴">
      <a class="foot-item foot-item-home{home_active}" href="{home_href}"{home_tab}>
        <span class="foot-icon">⌂</span>
        <span>홈</span>
      </a>
      <a class="foot-item{personal_active}" href="/personal">
        <span class="foot-icon">▤</span>
        <span>운세 리포트</span>
      </a>
      <a class="foot-item{compatibility_active}" href="/compatibility">
        <span class="foot-icon">♡</span>
        <span>궁합 리포트</span>
      </a>
      <a class="foot-item foot-item-more{more_active}" href="{more_href}"{more_tab}>
        <span class="foot-icon">•••</span>
        <span>더보기</span>
      </a>
    </footer>
    """
    return result


def _personal_form() -> str:
    mode_options = _face_analysis_mode_options()
    gender_options = _gender_options(required=True)
    target_gender_options = _gender_options(required=False)
    birth_time_options = _birth_time_options()
    result = f"""
    <div class="oracle-input-shell personal-oracle-shell">
      <div class="input-topbar" aria-label="페이지 이동">
        <a class="back-link" href="/" aria-label="홈으로 돌아가기">‹</a>
        <span class="heart-mark" aria-hidden="true">♡</span>
      </div>

      <div class="brand personal-brand">
        <div class="logo"><span aria-hidden="true">✧</span>ORACLE<span aria-hidden="true">✧</span></div>
        <div class="tag">관상 &amp; 사주 리포트</div>
        <div class="ornament"></div>
      </div>

      <div class="input-card personal-card">
        <div class="cloud cloud-left" aria-hidden="true"></div>
        <div class="cloud cloud-right" aria-hidden="true"></div>

        <div class="card-head personal-card-head">
          <div class="title-block">
            <h1><span aria-hidden="true">♡</span>개인 리포트<span aria-hidden="true">♡</span></h1>
            <p>당신의 얼굴과 사주가 그리는 한 장의 이야기</p>
          </div>
          <div class="guide-ora">
            <div class="speech-bubble">정확한 리포트를<br>위해 정보를 입력해줘!</div>
            <img src="/static/assets/oracle-solo-card.png" alt="" aria-hidden="true">
          </div>
        </div>

        <form method="post" class="workflow-form input-form personal-form" data-workflow-api="/api/personal">
          <input type="hidden" name="skip_face" value="0">
          <div class="personal-field-list">
            <div class="field personal-field lead">
              <span class="field-icon" aria-hidden="true">☻</span>
              <label>이름</label>
              <input name="name" placeholder="이름을 입력해주세요" required>
            </div>
            <div class="field personal-field">
              <span class="field-icon" aria-hidden="true">▣</span>
              <label>생년월일</label>
              <input name="birth_date" type="date" min="1800-01-01" max="2300-12-31" required>
            </div>
            <div class="field personal-field">
              <span class="field-icon" aria-hidden="true">◷</span>
              <label>태어난 시간<span class="hint">모르면 '모름'을 선택하세요</span></label>
              <select name="birth_time">{birth_time_options}</select>
            </div>
            <div class="field personal-field">
              <span class="field-icon" aria-hidden="true">♡</span>
              <label>성별</label>
              <select name="gender" required>{gender_options}</select>
            </div>
          </div>
          <div class="actions">
            <button type="submit" class="btn btn-primary" onclick="this.form.skip_face.value='0';">
              <img src="/static/assets/oracle-solo-card.png" alt="" aria-hidden="true">
              개인 리포트 촬영 시작
              <span aria-hidden="true">✦</span>
            </button>
            <button type="submit" class="btn btn-ghost" onclick="this.form.skip_face.value='1';">
              <img src="/static/assets/oracle-solo-card.png" alt="" aria-hidden="true">
              관상 없이 사주만 보기
            </button>
          </div>
        </form>

        <p class="footnote personal-footnote">
          <span aria-hidden="true">♡</span>
          입력한 정보와 촬영 이미지는 기기 안에서만 처리돼요.<br>Oracle은 재미를 위한 콘텐츠예요.
          <img src="/static/assets/oracle-solo-card.png" alt="" aria-hidden="true">
        </p>
      </div>
    </div>
    {_bottom_nav("personal")}
    """
    return result


def _compatibility_form() -> str:
    mode_options = "".join(
        f'<option value="{escape(mode)}">{escape(mode)}</option>'
        for mode in COMPATIBILITY_MODES
    )
    gender_options = _gender_options(required=True)
    birth_time_options = _birth_time_options()
    face_mode_options = _face_analysis_mode_options()
    result = f"""

    <div class="oracle-input-shell compatibility-oracle-shell">
      <div class="input-topbar compatibility-topbar" aria-label="페이지 이동">
        <a class="back-link" href="/" aria-label="홈으로 돌아가기">‹</a>
        <span class="compatibility-heart-mark" aria-hidden="true">♡</span>
      </div>

      <div class="brand compatibility-brand">
        <div class="logo"><span aria-hidden="true">✦</span>ORACLE<span aria-hidden="true">✦</span></div>
        <div class="tag">우리 궁합 리포트</div>
        <div class="ornament"></div>
      </div>

      <div class="input-card compatibility-card">
        <div class="compat-cloud compat-cloud-left" aria-hidden="true"></div>
        <span class="compat-spark compat-spark-left" aria-hidden="true">✦</span>
        <span class="compat-spark compat-spark-right" aria-hidden="true">✦</span>
        <img class="compat-hero-ora" src="/static/assets/oracle-pair-card.png" alt="" aria-hidden="true">

        <div class="card-head compatibility-card-head">
          <div class="title-block">
            <h1><span aria-hidden="true">♥</span>우리 궁합 리포트<span aria-hidden="true">♥</span></h1>
            <p>두 사람의 인연과 궁합을 분석해드려요!</p>
          </div>
          <div class="compat-speech">두 사람의 정보를<br>입력해보세요!</div>
        </div>

        <form method="post" class="workflow-form compatibility-form" data-workflow-api="/api/compatibility">
          <div class="compat-people">
            <fieldset class="compat-person">
              <legend><span aria-hidden="true">♥</span>첫 번째 사람</legend>
              <label>이름<input name="left_name" placeholder="이름을 입력해주세요" required></label>
              <label>생년월일<input name="left_birth_date" type="date" required></label>
              <label>태어난 시간<span class="hint">모르면 모름을 선택</span><select name="left_birth_time">{birth_time_options}</select></label>
              <label>성별<select name="left_gender" required>{gender_options}</select></label>
            </fieldset>

            <div class="compat-heart-bridge" aria-hidden="true">
              <span>♥</span>
            </div>

            <fieldset class="compat-person">
              <legend><span aria-hidden="true">♥</span>두 번째 사람</legend>
              <label>이름<input name="right_name" placeholder="이름을 입력해주세요" required></label>
              <label>생년월일<input name="right_birth_date" type="date" required></label>
              <label>태어난 시간<span class="hint">모르면 모름을 선택</span><select name="right_birth_time">{birth_time_options}</select></label>
              <label>성별<select name="right_gender" required>{gender_options}</select></label>
            </fieldset>
          </div>

          <div class="compat-options">
            <label>궁합 모드<select name="mode">{mode_options}</select></label>
            <label>관상 분석 모드<select name="face_analysis_mode">{face_mode_options}</select></label>
          </div>

          <p class="compat-note"><span aria-hidden="true">💡</span>두 사람 정보를 먼저 입력한 뒤 첫 번째 사람을 촬영하고, 3초 후 두 번째 사람을 촬영합니다.</p>

          <div class="compat-actions">
            <button type="submit" class="btn btn-primary compat-submit">
              <img src="/static/assets/oracle-solo-card.png" alt="" aria-hidden="true">
              우리 궁합 촬영 시작
            </button>
          </div>
        </form>

        <img class="compat-bottom-ora" src="/static/assets/oracle-solo-card.png" alt="" aria-hidden="true">
      </div>
    </div>
    {_bottom_nav("compatibility")}
    """
    return result


def _gender_options(required: bool) -> str:
    first_option = '<option value="">상관없음</option>'
    if required:
        first_option = '<option value="" selected disabled>선택</option>'
    result = f"""
      {first_option}
      <option value="남성">남성</option>
      <option value="여성">여성</option>
    """
    return result


def _birth_time_options() -> str:
    branch_options = tuple(
        (
            label,
            time_branch_range_display_from_index(index),
        )
        for index, label in enumerate(MANSE_TIME_BRANCH_LABELS)
    )
    options = (("", "모름"),) + branch_options
    result = "\n".join(
        f'<option value="{escape(value)}">{escape(label)}</option>'
        for value, label in options
    )
    return result


def _face_analysis_mode_options() -> str:
    selected_mode = os.getenv("ORACLE_FACE_ANALYSIS_MODE", "2")
    mode_one_selected = " selected" if selected_mode == "1" else ""
    mode_two_selected = " selected" if selected_mode == "2" else ""
    result = f"""
      <option value="1"{mode_one_selected}>1 - 이미지 LLM 분석</option>
      <option value="2"{mode_two_selected}>2 - 랜드마크 룰 기반 분석</option>
    """
    return result


def _personal_result_page(job_id: str, skip_face: bool) -> str:
    result = f"""
    <div class="oracle-result-shell personal-result-shell">
      <div class="input-topbar result-topbar" aria-label="페이지 이동">
        <a class="back-link" href="/personal" aria-label="입력 화면으로 돌아가기">‹</a>
        <span class="heart-mark" aria-hidden="true">♡</span>
      </div>

      <div class="brand personal-result-brand">
        <div class="logo">ORACLE</div>
        <div class="tag">개인 리포트 결과</div>
        <div class="ornament"></div>
      </div>

      <div class="result-sky" aria-hidden="true">
        <span class="result-cloud result-cloud-left"></span>
        <span class="result-spark result-spark-left">✧</span>
        <span class="result-spark result-spark-right">✧</span>
        <img class="result-hero-ora" src="/static/assets/oracle-character.png" alt="">
      </div>

      <div class="result-actions">
        <a class="result-action" href="/personal">입력 다시 하기</a>
        <a class="result-action" href="/">처음으로</a>
        <a id="download-report-link" class="result-action result-action-primary download-link" href="/api/jobs/{escape(job_id)}/download" hidden>리포트 다운로드</a>
      </div>
      {_capture_preview_panel(job_id=job_id, skip_face=skip_face, cute=True)}
    </div>
    """
    return result


def _compatibility_result_page(job_id: str) -> str:
    result = f"""
    <div class="oracle-result-shell">
      <div class="brand">
        <div class="logo">ORACLE</div>
        <div class="tag">두 사람 궁합 결과</div>
        <div class="ornament"></div>
      </div>
      <div class="result-actions">
        <a class="result-action" href="/compatibility">입력 다시 하기</a>
        <a class="result-action" href="/">처음으로</a>
        <a id="download-report-link" class="result-action result-action-primary download-link" href="/api/jobs/{escape(job_id)}/download" hidden>리포트 다운로드</a>
      </div>
      {_capture_preview_panel(job_id=job_id, skip_face=False)}
    </div>
    """
    return result


def _capture_preview_panel(
    *,
    job_id: str = "",
    skip_face: bool = False,
    cute: bool = False,
) -> str:
    job_attr = f' data-workflow-result-job="{escape(job_id)}"' if job_id != "" else ""
    skip_attr = ' data-skip-face="1"' if skip_face else ' data-skip-face="0"'
    loading_hidden = "" if job_id != "" else " hidden"
    preview_hidden = " hidden" if skip_face or job_id == "" else ""
    loading_title = (
        "사주 리포트 생성 중입니다"
        if skip_face
        else "촬영 및 리포트 생성 중입니다"
    )
    loading_message = (
        "입력한 생년월일과 태어난 시간으로 사주 리포트를 만들고 있습니다. 잠시만 기다려 주세요."
        if skip_face
        else "얼굴 촬영과 사주 분석을 진행한 뒤 리포트를 만들고 있습니다. 잠시만 기다려 주세요."
    )
    status_text = (
        "리포트 생성 중"
        if skip_face
        else "정면 얼굴을 카메라 중앙에 맞춰 주세요."
    )
    loading_heart = (
        '<span class="loading-heart" aria-hidden="true">♡</span>' if cute else ""
    )
    if cute:
        preview_content = f"""
      <h2><span class="capture-title-icon" aria-hidden="true">▣</span>실시간 촬영 상태</h2>
      <div class="capture-stage">
        <div class="capture-direction">
          <span class="capture-heart" aria-hidden="true">♡</span>
          <img src="/static/assets/oracle-solo-card.png" alt="" aria-hidden="true">
          <strong>정면 얼굴을 <span>카메라 중앙</span>에 맞춰 주세요.</strong>
          <span class="capture-heart" aria-hidden="true">♡</span>
        </div>
        <div class="capture-visual">
          <img id="capture-preview-image" alt="실시간 촬영 상태">
          <span class="camera-corner camera-corner-tl" aria-hidden="true"></span>
          <span class="camera-corner camera-corner-tr" aria-hidden="true"></span>
          <span class="camera-corner camera-corner-bl" aria-hidden="true"></span>
          <span class="camera-corner camera-corner-br" aria-hidden="true"></span>
          <span class="face-guide" aria-hidden="true"></span>
          <div class="capture-side-note capture-side-note-left" aria-hidden="true">
            <p>카메라를<br>눈높이에 맞추고<br>정면을 바라봐 주세요!</p>
            <span>✧</span>
            <img src="/static/assets/oracle-pair-card.png" alt="">
          </div>
          <div class="capture-side-note capture-side-note-right" aria-hidden="true">
            <p>밝은 곳에서<br>촬영하면 더<br>정확해요!</p>
            <span>✧</span>
            <img src="/static/assets/oracle-solo-card.png" alt="">
          </div>
          <div class="capture-privacy-veil" hidden>
            <div class="veil-card">
              <span class="veil-mark" aria-hidden="true">✦</span>
              <strong>얼굴 인식 완료</strong>
              <p>이제 리포트를 예쁘게 빚는 중이에요.</p>
            </div>
          </div>
        </div>
        <p id="workflow-status" class="hint capture-tip"><span aria-hidden="true">✧</span>{status_text}<span aria-hidden="true">☆</span></p>
      </div>
        """
    else:
        preview_content = f"""
      <h2>실시간 촬영 상태</h2>
      <img id="capture-preview-image" alt="실시간 촬영 상태">
      <div class="capture-privacy-veil" hidden>
        <div class="veil-card">
          <span class="veil-mark" aria-hidden="true">✦</span>
          <strong>얼굴 인식 완료</strong>
          <p>이제 리포트를 예쁘게 빚는 중이에요.</p>
        </div>
      </div>
      <p id="workflow-status" class="hint">{status_text}</p>
        """
    cute_class = " capture-preview-cute" if cute else ""
    result = f"""
    <section id="workflow-loading" class="panel workflow-loading" role="status" aria-live="polite" aria-busy="true"{job_attr}{skip_attr}{loading_hidden}>
      <span class="loading-spinner" aria-hidden="true"></span>
      <div>
        <strong id="workflow-loading-title">{loading_title}</strong>
        <p id="workflow-loading-message" class="hint">{loading_message}</p>
      </div>
      {loading_heart}
    </section>
    <section class="panel capture-preview{cute_class}"{preview_hidden}>
      {preview_content}
    </section>
    <section id="workflow-result"></section>
    """
    return result


def _personal_result(workflow_result: PersonalWorkflowResult) -> str:
    result = workflow_result.report_fragment_html
    return result


def _compatibility_result(markdown: str, workflow_result) -> str:
    del markdown
    result = workflow_result.report_fragment_html
    return result


def _error_panel(exc: Exception) -> str:
    result = f"""
    <section class="error">
      <strong>처리 중 오류가 발생했습니다.</strong>
      <p>{escape(str(exc))}</p>
    </section>
    """
    return result


def _render_page(
    title: str,
    body: str,
    *,
    page_class: str = "",
    show_heading: bool = True,
) -> str:
    main_class = f' class="{page_class}"' if page_class != "" else ""
    heading = "<h1>Oracle</h1>" if show_heading else ""
    result = f"""
    <!doctype html>
    <html lang="ko">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{escape(title)}</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Gowun+Batang:wght@400;700&family=Gowun+Dodum&family=Song+Myung&display=swap" rel="stylesheet">
        <style>
          :root {{
            color-scheme: light;
            --paper: #fff8ef;
            --paper-2: #ffffff;
            --ink: #4a2f26;
            --ink-soft: #7a6257;
            --line: #f1d8cf;
            --line-soft: #f7e6df;
            --mok: #42b883;
            --mok-deep: #16845a;
            --hwa: #ff6f82;
            --gold: #d8a24b;
            font-family: "Gowun Dodum", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: var(--paper);
            color: var(--ink);
          }}
          * {{
            box-sizing: border-box;
          }}
          body {{
            margin: 0;
            min-height: 100vh;
            background: var(--paper);
            background-image:
              linear-gradient(180deg, #fffaf4 0%, #fff3e8 56%, #ffe9ea 100%),
              repeating-linear-gradient(90deg, rgba(216, 162, 75, 0.08) 0 1px, transparent 1px 120px);
            -webkit-font-smoothing: antialiased;
          }}
          main {{
            width: min(960px, calc(100vw - 32px));
            margin: 0 auto;
            padding: 32px 0;
          }}
          main.input-page {{
            width: min(860px, calc(100vw - 48px));
            padding: 24px 0 112px;
          }}
          main.home-page {{
            width: min(1180px, calc(100vw - 48px));
            padding: 22px 0 112px;
          }}
          h1 {{
            margin: 0 0 24px;
            font-size: 32px;
          }}
          h2 {{
            margin-top: 0;
          }}
          .menu {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
          }}
          .menu-card, .panel, .error {{
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--paper-2);
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05);
          }}
          .menu-card {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            padding: 20px;
            color: inherit;
            text-decoration: none;
          }}
          .menu-card strong {{
            font-size: 22px;
          }}
          .panel {{
            padding: 20px;
          }}
          .serif {{
            font-family: "Gowun Batang", serif;
          }}
          .oracle-home-shell {{
            width: 100%;
            overflow: hidden;
          }}
          .oracle-home-shell.more-active .home-view {{
            display: none;
          }}
          .oracle-home-shell.more-active .more-section {{
            display: block;
          }}
          .oracle-home-shell.more-active .foot-item-active {{
            color: var(--ink-soft);
          }}
          .oracle-home-shell.more-active .foot-item-more {{
            color: var(--hwa);
          }}
          .home-nav {{
            display: flex;
            align-items: flex-start;
            justify-content: flex-start;
            gap: 18px;
            margin-bottom: 6px;
          }}
          .brand-lockup .logo {{
            font-family: "Song Myung", serif;
            font-size: 36px;
            letter-spacing: 0;
            color: var(--ink);
            position: relative;
            display: inline-block;
          }}
          .brand-lockup .logo .stamp {{
            position: absolute;
            top: -2px;
            right: -34px;
            width: 30px;
            height: 30px;
            border: 2px solid var(--hwa);
            border-radius: 8px;
            color: var(--hwa);
            font-size: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            transform: rotate(9deg);
            opacity: 0.9;
          }}
          .brand-lockup .tag {{
            font-size: 15px;
            color: var(--ink);
            margin-top: 6px;
          }}
          .home-hero {{
            position: relative;
            text-align: center;
            padding-top: 14px;
          }}
          .home-hero .speech {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 44px;
            padding: 8px 24px;
            border: 2px solid #ffc6cf;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.64);
            color: var(--hwa);
            font-family: "Gowun Batang", serif;
            font-size: 18px;
            font-weight: 700;
            position: relative;
          }}
          .home-hero .speech::after {{
            content: "";
            position: absolute;
            bottom: -8px;
            left: 50%;
            width: 14px;
            height: 14px;
            border-right: 2px solid #ffc6cf;
            border-bottom: 2px solid #ffc6cf;
            background: #fff8f3;
            transform: translateX(-50%) rotate(45deg);
          }}
          .home-hero h1 {{
            margin: 18px 0 10px;
            font-family: "Gowun Batang", serif;
            font-size: 50px;
            line-height: 1.12;
            color: var(--ink);
            letter-spacing: 0;
          }}
          .home-hero h1 span, .home-hero p strong {{
            color: var(--hwa);
          }}
          .home-hero p {{
            margin: 0 auto 14px;
            color: var(--ink);
            font-size: 18px;
            line-height: 1.55;
          }}
          .oracle-character {{
            position: relative;
            z-index: 2;
            display: block;
            width: min(620px, 78vw);
            height: auto;
            object-fit: contain;
            object-position: center center;
            margin: 4px auto -62px;
            border: 0;
            border-radius: 0;
            mix-blend-mode: normal;
          }}
          .cards {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 18px;
            width: 100%;
            max-width: 960px;
            margin: -16px auto 20px;
            position: relative;
            z-index: 4;
          }}
          .mode {{
            position: relative;
            display: flex;
            min-height: 210px;
            align-items: center;
            justify-content: space-between;
            gap: 14px;
            background: rgba(255, 255, 255, 0.82);
            border: 2px solid var(--line);
            border-radius: 8px;
            padding: 28px 30px;
            color: inherit;
            cursor: pointer;
            overflow: hidden;
            text-decoration: none;
            transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s;
            box-shadow: 0 6px 20px -14px rgba(46, 37, 32, 0.4);
          }}
          .mode::after {{
            content: "";
            position: absolute;
            inset: 0;
            opacity: 0.5;
            pointer-events: none;
          }}
          .mode.solo {{
            border-color: #a9dfc5;
          }}
          .mode.pair {{
            border-color: #ffc5cb;
          }}
          .mode.solo::after {{
            background: linear-gradient(120deg, rgba(66, 184, 131, 0.12), transparent 60%);
          }}
          .mode.pair::after {{
            background: linear-gradient(120deg, rgba(255, 111, 130, 0.13), transparent 60%);
          }}
          .mode:hover {{
            transform: translateY(-6px);
            box-shadow: 0 18px 40px -18px rgba(46, 37, 32, 0.5);
          }}
          .mode.solo:hover {{
            border-color: var(--mok);
          }}
          .mode.pair:hover {{
            border-color: var(--hwa);
          }}
          .mode-copy {{
            position: relative;
            z-index: 2;
          }}
          .mode .ic {{
            width: 38px;
            height: 38px;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 12px;
            font-size: 27px;
            line-height: 1;
          }}
          .mode.solo .ic {{
            color: var(--mok);
          }}
          .mode.pair .ic {{
            color: var(--hwa);
          }}
          .mode h2 {{
            font-family: "Gowun Batang", serif;
            font-size: 31px;
            font-weight: 700;
            margin: 0 0 14px;
          }}
          .mode.solo h2 {{
            color: var(--mok-deep);
          }}
          .mode.pair h2 {{
            color: var(--hwa);
          }}
          .mode p {{
            font-size: 15px;
            color: var(--ink);
            line-height: 1.7;
            min-height: 46px;
            margin: 0;
          }}
          .mode .go {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            justify-content: center;
            min-width: 160px;
            min-height: 46px;
            margin-top: 18px;
            padding: 10px 20px;
            border-radius: 999px;
            font-family: "Gowun Batang", serif;
            font-size: 15px;
            font-weight: 700;
            color: #ffffff;
          }}
          .mode.solo .go {{
            background: linear-gradient(90deg, #42b883, #66cfa5);
          }}
          .mode.pair .go {{
            background: linear-gradient(90deg, #ff6f82, #ff8fa0);
          }}
          .mode .go .arr {{
            transition: transform 0.2s;
          }}
          .mode:hover .go .arr {{
            transform: translateX(4px);
          }}
          .mode .mode-art {{
            position: relative;
            z-index: 2;
            width: 170px;
            height: 184px;
            object-fit: contain;
            border-radius: 0;
            mix-blend-mode: normal;
            flex: 0 0 auto;
            align-self: flex-end;
            margin: 0 -14px -18px 0;
          }}
          .mode .mode-art-pair {{
            width: 164px;
            height: 188px;
            margin-right: -12px;
          }}
          .feature-row {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0;
            max-width: 960px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 14px 40px -28px rgba(74, 47, 38, 0.4);
            overflow: hidden;
          }}
          .feature {{
            min-height: 132px;
            padding: 22px 18px 20px;
            text-align: center;
            border-right: 1px dashed #e8cfc6;
          }}
          .feature:last-child {{
            border-right: 0;
          }}
          .feature-ic {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 54px;
            height: 54px;
            margin-bottom: 10px;
            border-radius: 999px;
            font-weight: 700;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.7);
          }}
          .feature-ic.camera {{
            background: #ece3ff;
            color: #7650bf;
          }}
          .feature-ic.calendar {{
            background: #e4f2ff;
            color: #4381b7;
          }}
          .feature-ic.ai {{
            background: #dff8f1;
            color: #158672;
          }}
          .feature-ic.lock {{
            background: #fff0c9;
            color: #c68011;
          }}
          .feature strong {{
            display: block;
            font-family: "Gowun Batang", serif;
            font-size: 17px;
            color: var(--ink);
          }}
          .feature p {{
            margin: 9px 0 0;
            color: var(--ink);
            font-size: 12px;
            line-height: 1.6;
          }}
          .more-section {{
            display: none;
            max-width: 1040px;
            margin: 0 auto;
            scroll-margin-top: 34px;
          }}
          .more-head {{
            margin-bottom: 22px;
            text-align: center;
          }}
          .more-head .logo {{
            position: relative;
            display: inline-block;
            font-family: "Song Myung", serif;
            font-size: 32px;
            letter-spacing: 0;
            color: var(--ink);
          }}
          .more-head .logo .stamp {{
            position: absolute;
            top: -2px;
            right: -30px;
            width: 25px;
            height: 25px;
            border: 2px solid var(--hwa);
            border-radius: 7px;
            color: var(--hwa);
            font-size: 13px;
            display: flex;
            align-items: center;
            justify-content: center;
            transform: rotate(9deg);
          }}
          .more-head h2 {{
            margin: 14px 0 6px;
            font-family: "Gowun Batang", serif;
            font-size: 34px;
            color: var(--ink);
          }}
          .more-head p {{
            margin: 0;
            color: var(--ink-soft);
            font-size: 15px;
          }}
          .more-grid {{
            display: grid;
            gap: 18px;
          }}
          .more-card {{
            padding: 24px;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.82);
            box-shadow: 0 18px 46px -30px rgba(74, 47, 38, 0.34);
            overflow: hidden;
          }}
          .more-copy {{
            min-width: 0;
          }}
          .more-kicker {{
            display: inline-flex;
            align-items: center;
            min-height: 32px;
            padding: 6px 14px;
            border-radius: 999px;
            background: rgba(255, 111, 130, 0.1);
            color: var(--hwa);
            font-family: "Gowun Batang", serif;
            font-size: 14px;
            font-weight: 700;
          }}
          .more-copy h3 {{
            margin: 12px 0 16px;
            font-family: "Gowun Batang", serif;
            font-size: 27px;
            line-height: 1.25;
            color: var(--ink);
          }}
          .more-copy ul, .more-copy ol {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
            list-style: none;
            margin: 0;
            padding: 0;
          }}
          .more-copy li, .faq-list div {{
            display: grid;
            grid-template-columns: 46px 1fr;
            column-gap: 12px;
            align-items: center;
            min-height: 76px;
            padding: 12px 14px;
            border: 1px solid var(--line-soft);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.82);
          }}
          .more-copy li > span, .faq-list div > span {{
            grid-row: span 2;
            width: 46px;
            height: 46px;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #fff2e8;
            color: var(--gold);
            font-family: "Gowun Batang", serif;
            font-size: 20px;
            font-weight: 700;
          }}
          .more-copy li strong, .faq-list strong {{
            display: block;
            color: var(--ink);
            font-family: "Gowun Batang", serif;
            font-size: 18px;
            line-height: 1.35;
          }}
          .more-copy li em, .faq-list em {{
            display: block;
            margin-top: 4px;
            color: var(--ink-soft);
            font-size: 13px;
            font-style: normal;
            line-height: 1.55;
          }}
          .faq-list {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
          }}
          .more-note {{
            display: flex;
            align-items: center;
            gap: 14px;
            margin-top: 14px;
            padding: 14px 18px;
            border-radius: 8px;
            background: linear-gradient(90deg, rgba(255, 111, 130, 0.12), rgba(255, 255, 255, 0.72));
          }}
          .more-note img {{
            width: 72px;
            height: 72px;
            object-fit: contain;
            flex: 0 0 auto;
          }}
          .more-note p {{
            margin: 0;
            color: var(--ink-soft);
            font-size: 14px;
            line-height: 1.55;
          }}
          .more-note strong {{
            color: var(--hwa);
            font-family: "Gowun Batang", serif;
            font-size: 18px;
          }}
          .home-foot {{
            position: fixed;
            left: 50%;
            bottom: max(14px, env(safe-area-inset-bottom));
            z-index: 30;
            transform: translateX(-50%);
            display: flex;
            justify-content: space-around;
            align-items: center;
            width: min(960px, calc(100vw - 48px));
            min-height: 68px;
            margin: 0;
            padding: 6px 0;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.94);
            box-shadow: 0 16px 42px -24px rgba(74, 47, 38, 0.42);
            backdrop-filter: blur(14px);
            text-align: center;
            color: var(--ink);
          }}
          .foot-item {{
            flex: 1;
            min-height: 54px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 4px;
            border-right: 1px solid var(--line-soft);
            color: var(--ink-soft);
            font-family: "Gowun Batang", serif;
            font-size: 12px;
            font-weight: 700;
            text-decoration: none;
            letter-spacing: 0;
          }}
          .foot-item:last-child {{
            border-right: 0;
          }}
          .foot-item-active {{
            color: var(--hwa);
          }}
          .foot-icon {{
            display: block;
            min-height: 20px;
            font-family: "Gowun Dodum", sans-serif;
            font-size: 20px;
            line-height: 1;
          }}
          .oracle-input-shell {{
            width: 100%;
          }}
          .brand {{
            text-align: center;
            margin-bottom: 18px;
          }}
          .brand .logo {{
            font-family: "Song Myung", serif;
            font-size: 31px;
            letter-spacing: 0;
            color: var(--ink);
          }}
          .brand .tag {{
            font-size: 12px;
            letter-spacing: 0;
            color: var(--gold);
            text-transform: uppercase;
            margin-top: 6px;
          }}
          .brand .ornament {{
            margin: 12px auto 0;
            width: 50px;
            height: 1px;
            background: var(--gold);
            position: relative;
          }}
          .brand .ornament::before {{
            content: "※";
            position: absolute;
            top: -12px;
            left: 50%;
            transform: translateX(-50%);
            color: var(--gold);
            font-size: 13px;
            background: var(--paper);
            padding: 0 8px;
          }}
          .input-card {{
            background: var(--paper-2);
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 30px 34px 26px;
            box-shadow: 0 14px 40px -22px rgba(46, 37, 32, 0.4);
            position: relative;
          }}
          .input-card::before {{
            content: "";
            position: absolute;
            inset: 7px;
            border: 1px solid var(--line-soft);
            border-radius: 6px;
            pointer-events: none;
          }}
          .card-head {{
            position: relative;
            text-align: center;
            margin-bottom: 22px;
          }}
          .card-head h1 {{
            font-family: "Gowun Batang", serif;
            font-size: 23px;
            font-weight: 700;
            margin: 0;
          }}
          .card-head p {{
            font-size: 13px;
            color: var(--ink-soft);
            margin: 6px 0 0;
          }}
          .input-form {{
            position: relative;
          }}
          .field {{
            margin-bottom: 0;
          }}
          .field.lead {{
            margin-bottom: 16px;
          }}
          .field-stack {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 16px;
          }}
          .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 16px;
          }}
          fieldset {{
            border: 1px solid #dedbd2;
            border-radius: 8px;
            padding: 16px;
          }}
          label {{
            display: block;
            font-family: "Gowun Batang", serif;
            font-size: 14px;
            color: var(--ink);
            margin-bottom: 8px;
            font-weight: 600;
          }}
          label .hint {{
            display: block;
            font-family: "Gowun Dodum", sans-serif;
            font-size: 11.5px;
            font-weight: 400;
            color: var(--ink-soft);
            margin-top: 2px;
            letter-spacing: 0;
          }}
          input, select, button {{
            min-height: 42px;
            border: 1px solid var(--line);
            border-radius: 7px;
            padding: 8px 10px;
            font: inherit;
          }}
          input, select {{
            width: 100%;
            color: var(--ink);
            background: #ffffff;
            padding: 12px 15px;
            transition: border-color 0.18s, box-shadow 0.18s;
          }}
          input::placeholder {{
            color: #b6ac9c;
          }}
          input:focus, select:focus {{
            outline: none;
            border-color: var(--mok);
            box-shadow: 0 0 0 3px rgba(58, 125, 92, 0.13);
          }}
          select {{
            appearance: none;
            background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'><path d='M1 1l5 5 5-5' fill='none' stroke='%236B6256' stroke-width='1.6' stroke-linecap='round'/></svg>");
            background-repeat: no-repeat;
            background-position: right 15px center;
            padding-right: 38px;
            cursor: pointer;
          }}
          input[type="date"] {{
            cursor: text;
          }}
          input[type="date"]::-webkit-calendar-picker-indicator {{
            cursor: pointer;
            opacity: 0.55;
          }}
          .field.lead input {{
            background: #f1f5f0;
            border-color: #cadbcf;
          }}
          button {{
            font-weight: 700;
            cursor: pointer;
          }}
          button:disabled {{
            cursor: wait;
            opacity: 0.72;
          }}
          .actions {{
            display: flex;
            gap: 10px;
            margin-top: 22px;
          }}
          .btn {{
            flex: 1;
            font-family: "Gowun Batang", serif;
            font-size: 15px;
            padding: 14px 16px;
            border-radius: 8px;
            border: 1px solid transparent;
            transition: transform 0.15s, box-shadow 0.2s, background 0.2s;
            letter-spacing: 0;
          }}
          .btn-primary {{
            background: var(--mok);
            color: #ffffff;
            box-shadow: 0 8px 20px -10px rgba(58, 125, 92, 0.7);
          }}
          .btn-primary:hover {{
            background: var(--mok-deep);
            transform: translateY(-2px);
          }}
          .btn-ghost {{
            background: transparent;
            color: var(--ink);
            border-color: var(--line);
          }}
          .btn-ghost:hover {{
            background: #f0eadc;
            transform: translateY(-2px);
          }}
          button:disabled:hover {{
            transform: none;
          }}
          .footnote {{
            position: relative;
            text-align: center;
            font-size: 11px;
            color: var(--ink-soft);
            margin: 16px 0 0;
            line-height: 1.7;
          }}
          main.personal-page {{
            width: min(940px, calc(100vw - 58px));
            padding: 24px 0 104px;
          }}
          .personal-oracle-shell {{
            position: relative;
            width: 100%;
          }}
          .input-topbar {{
            position: absolute;
            top: 4px;
            left: 0;
            right: 0;
            z-index: 4;
            display: flex;
            justify-content: space-between;
            align-items: center;
            pointer-events: none;
          }}
          .back-link, .heart-mark {{
            width: 44px;
            height: 44px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border: 1px solid #ffd7cc;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.68);
            color: var(--ink);
            box-shadow: 0 12px 28px -22px rgba(74, 47, 38, 0.45);
          }}
          .back-link {{
            pointer-events: auto;
            color: #5a332a;
            font-size: 38px;
            line-height: 0.65;
            text-decoration: none;
            padding-bottom: 6px;
          }}
          .heart-mark {{
            color: #ffb6c2;
            font-size: 31px;
            transform: rotate(-18deg);
          }}
          .personal-brand {{
            margin-bottom: 30px;
          }}
          .personal-brand .logo {{
            display: inline-flex;
            align-items: center;
            gap: 18px;
            font-size: 32px;
            line-height: 1;
          }}
          .personal-brand .logo span {{
            color: #f8ad55;
            font-family: "Gowun Dodum", sans-serif;
            font-size: 22px;
            font-weight: 700;
          }}
          .personal-brand .tag {{
            margin-top: 11px;
            color: #a66a42;
            font-family: "Gowun Batang", serif;
            font-size: 14px;
            text-transform: none;
          }}
          .personal-brand .ornament {{
            margin-top: 12px;
            width: 58px;
          }}
          .personal-brand .ornament::before {{
            top: -13px;
            background: #fff8ef;
            font-size: 13px;
          }}
          .personal-card {{
            min-height: 0;
            padding: 42px 44px 28px;
            border: 2px solid #ffd8ce;
            border-radius: 26px;
            background:
              radial-gradient(circle at 84% 17%, rgba(255, 239, 242, 0.9), transparent 24%),
              radial-gradient(circle at 18% 48%, rgba(255, 246, 235, 0.9), transparent 32%),
              rgba(255, 255, 255, 0.87);
            box-shadow:
              0 22px 52px -36px rgba(74, 47, 38, 0.42),
              inset 0 0 0 8px rgba(255, 249, 245, 0.92);
          }}
          .personal-card::before {{
            inset: 9px;
            border: 1px solid #ffdcd4;
            border-radius: 19px;
          }}
          .cloud {{
            position: absolute;
            top: -38px;
            width: 76px;
            height: 33px;
            border: 2px solid #ffc77e;
            border-top: 0;
            border-radius: 0 0 25px 25px;
            opacity: 0.86;
          }}
          .cloud::before, .cloud::after {{
            content: "";
            position: absolute;
            bottom: 12px;
            border: 2px solid #ffc77e;
            border-bottom: 0;
            background: #fff8ef;
          }}
          .cloud::before {{
            left: 13px;
            width: 29px;
            height: 29px;
            border-radius: 999px 999px 0 0;
          }}
          .cloud::after {{
            right: 9px;
            width: 38px;
            height: 38px;
            border-radius: 999px 999px 0 0;
          }}
          .cloud-left {{
            left: 92px;
          }}
          .cloud-right {{
            right: 142px;
            top: -35px;
            transform: scale(0.74);
          }}
          .personal-card-head {{
            display: flex;
            align-items: flex-start;
            justify-content: center;
            gap: 32px;
            min-height: 126px;
            margin-bottom: 14px;
            text-align: center;
          }}
          .personal-card-head .title-block {{
            padding-top: 12px;
            min-width: 310px;
          }}
          .personal-card-head h1 {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            color: #3f211b;
            font-family: "Gowun Batang", serif;
            font-size: 32px;
            line-height: 1.15;
          }}
          .personal-card-head h1 span {{
            color: #ff8fab;
            font-family: "Gowun Dodum", sans-serif;
            font-size: 18px;
          }}
          .personal-card-head p {{
            margin-top: 13px;
            color: #92746a;
            font-family: "Gowun Batang", serif;
            font-size: 15px;
          }}
          .guide-ora {{
            display: flex;
            align-items: flex-start;
            gap: 8px;
            margin-top: -8px;
          }}
          .guide-ora img {{
            width: 142px;
            height: 138px;
            object-fit: contain;
          }}
          .speech-bubble {{
            position: relative;
            min-width: 145px;
            margin-top: 16px;
            padding: 16px 20px;
            border: 2px solid #f6b8c5;
            border-radius: 50%;
            background: #fff3f6;
            color: #5b3b34;
            font-family: "Gowun Batang", serif;
            font-size: 13px;
            font-weight: 700;
            line-height: 1.45;
          }}
          .speech-bubble::after {{
            content: "";
            position: absolute;
            right: -6px;
            bottom: 19px;
            width: 17px;
            height: 17px;
            border-right: 2px solid #f6b8c5;
            border-bottom: 2px solid #f6b8c5;
            background: #fff3f6;
            transform: rotate(-22deg);
          }}
          .personal-form {{
            z-index: 1;
          }}
          .personal-field-list {{
            overflow: hidden;
            border: 1px solid #ffe0d8;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.72);
          }}
          .personal-field {{
            display: grid;
            grid-template-columns: 58px 128px minmax(0, 1fr);
            align-items: center;
            gap: 14px;
            min-height: 82px;
            margin: 0;
            padding: 14px 22px;
            border-bottom: 1px solid #ffe0d8;
          }}
          .personal-field:last-child {{
            border-bottom: 0;
          }}
          .personal-field .field-icon {{
            width: 46px;
            height: 46px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border: 1px solid #ffd9ce;
            border-radius: 999px;
            background: #fff9f5;
            color: #ff7c91;
            font-family: "Gowun Dodum", sans-serif;
            font-size: 25px;
            box-shadow: 0 10px 24px -20px rgba(255, 111, 130, 0.55);
          }}
          .personal-field:nth-child(5) .field-icon,
          .personal-field:nth-child(6) .field-icon {{
            color: #f8a94e;
          }}
          .personal-field label {{
            margin: 0;
            color: #3f211b;
            font-family: "Gowun Batang", serif;
            font-size: 18px;
            line-height: 1.35;
            font-weight: 700;
          }}
          .personal-field label .hint {{
            color: #ff7890;
            font-family: "Gowun Dodum", sans-serif;
            font-size: 11px;
            line-height: 1.3;
          }}
          .personal-field input,
          .personal-field select {{
            min-height: 52px;
            border: 1px solid #ffd7ce;
            border-radius: 12px;
            background-color: rgba(255, 255, 255, 0.92);
            color: #5b3b34;
            font-size: 16px;
            padding: 12px 18px;
          }}
          .personal-field input::placeholder {{
            color: #bca8a2;
          }}
          .personal-field input:focus,
          .personal-field select:focus {{
            border-color: #ff9fad;
            box-shadow: 0 0 0 4px rgba(255, 143, 171, 0.14);
          }}
          .personal-field select {{
            background-position: right 20px center;
            padding-right: 46px;
          }}
          .personal-field.lead input {{
            background: rgba(255, 255, 255, 0.92);
            border-color: #ffd7ce;
          }}
          .personal-form .actions {{
            gap: 18px;
            margin-top: 24px;
          }}
          .personal-form .btn {{
            min-height: 66px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            border-radius: 18px;
            font-size: 18px;
            box-shadow: 0 18px 34px -24px rgba(255, 111, 130, 0.7);
          }}
          .personal-form .btn img {{
            width: 44px;
            height: 38px;
            object-fit: contain;
          }}
          .personal-form .btn span {{
            color: #ffd55a;
          }}
          .personal-form .btn-primary {{
            border: 2px solid #ff86a4;
            background: linear-gradient(135deg, #ff7898 0%, #ff5f93 100%);
            color: #ffffff;
          }}
          .personal-form .btn-primary:hover {{
            background: linear-gradient(135deg, #ff6b91 0%, #f65186 100%);
          }}
          .personal-form .btn-ghost {{
            border: 1px solid #ffd8ce;
            background: linear-gradient(135deg, rgba(255, 250, 246, 0.96), rgba(255, 255, 255, 0.8));
            color: #4a2f26;
          }}
          .personal-form .btn-ghost:hover {{
            background: #fff7f0;
          }}
          .personal-footnote {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 16px;
            min-height: 68px;
            margin-top: 24px;
            padding: 13px 78px;
            border: 1px solid #ffe0d8;
            border-radius: 16px;
            background: linear-gradient(90deg, rgba(255, 242, 247, 0.94), rgba(255, 255, 255, 0.78));
            color: #6e4940;
            font-family: "Gowun Batang", serif;
            font-size: 14px;
            line-height: 1.55;
          }}
          .personal-footnote > span {{
            position: absolute;
            left: 22px;
            top: 17px;
            color: #ffb1c0;
            font-family: "Gowun Dodum", sans-serif;
            font-size: 24px;
          }}
          .personal-footnote img {{
            position: absolute;
            right: 24px;
            bottom: 2px;
            width: 54px;
            height: 58px;
            object-fit: contain;
          }}
          main.compatibility-page {{
            width: min(1120px, calc(100vw - 58px));
            padding: 24px 0 104px;
          }}
          .compatibility-oracle-shell {{
            position: relative;
            width: 100%;
          }}
          .compatibility-topbar {{
            top: 4px;
          }}
          .compatibility-heart-mark {{
            width: 44px;
            height: 44px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border: 1px solid #ffd7cc;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.68);
            color: #ffb6c2;
            font-size: 31px;
            box-shadow: 0 12px 28px -22px rgba(74, 47, 38, 0.45);
            transform: rotate(-16deg);
          }}
          .compatibility-brand {{
            margin-bottom: 42px;
          }}
          .compatibility-brand .logo {{
            display: inline-flex;
            align-items: center;
            gap: 22px;
            font-size: 36px;
            line-height: 1;
          }}
          .compatibility-brand .logo span {{
            color: #ffd36f;
            font-family: "Gowun Dodum", sans-serif;
            font-size: 24px;
          }}
          .compatibility-brand .tag {{
            margin-top: 14px;
            color: #8f5d3e;
            font-family: "Gowun Batang", serif;
            font-size: 15px;
            text-transform: none;
          }}
          .compatibility-brand .ornament {{
            margin-top: 13px;
          }}
          .compatibility-brand .ornament::before {{
            content: "×";
            top: -13px;
            background: #fff8ef;
          }}
          .compatibility-card {{
            position: relative;
            min-height: 0;
            padding: 40px 36px 30px;
            border: 2px solid #ffd8ce;
            border-radius: 24px;
            background:
              radial-gradient(circle at 82% 10%, rgba(255, 239, 242, 0.9), transparent 22%),
              radial-gradient(circle at 18% 58%, rgba(255, 246, 235, 0.88), transparent 32%),
              rgba(255, 255, 255, 0.86);
            box-shadow:
              0 22px 52px -36px rgba(74, 47, 38, 0.42),
              inset 0 0 0 8px rgba(255, 249, 245, 0.92);
          }}
          .compatibility-card::before {{
            inset: 9px;
            border-color: #ffdcd4;
            border-radius: 17px;
          }}
          .compat-cloud {{
            position: absolute;
            top: -58px;
            width: 124px;
            height: 48px;
            border: 2px solid #ffd7b0;
            border-top: 0;
            border-radius: 0 0 38px 38px;
            opacity: 0.9;
          }}
          .compat-cloud::before,
          .compat-cloud::after {{
            content: "";
            position: absolute;
            bottom: 17px;
            border: 2px solid #ffd7b0;
            border-bottom: 0;
            background: #fff8ef;
          }}
          .compat-cloud::before {{
            left: 24px;
            width: 44px;
            height: 44px;
            border-radius: 999px 999px 0 0;
          }}
          .compat-cloud::after {{
            right: 18px;
            width: 58px;
            height: 58px;
            border-radius: 999px 999px 0 0;
          }}
          .compat-cloud-left {{
            left: 120px;
          }}
          .compat-spark {{
            position: absolute;
            color: #ffd36f;
            font-family: "Gowun Dodum", sans-serif;
            font-size: 22px;
          }}
          .compat-spark-left {{
            left: 52px;
            top: -44px;
          }}
          .compat-spark-right {{
            right: 44px;
            top: 42px;
          }}
          .compat-hero-ora {{
            position: absolute;
            right: 90px;
            top: -118px;
            width: 190px;
            height: 170px;
            object-fit: contain;
          }}
          .compatibility-card-head {{
            display: flex;
            align-items: flex-start;
            justify-content: center;
            gap: 58px;
            min-height: 112px;
            margin-bottom: 22px;
            text-align: center;
          }}
          .compatibility-card-head .title-block {{
            min-width: 360px;
          }}
          .compatibility-card-head h1 {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 16px;
            margin: 0;
            color: #3f211b;
            font-family: "Gowun Batang", serif;
            font-size: 34px;
            line-height: 1.15;
          }}
          .compatibility-card-head h1 span {{
            color: #ff8fab;
            font-family: "Gowun Dodum", sans-serif;
            font-size: 22px;
          }}
          .compatibility-card-head p {{
            margin: 16px 0 0;
            color: #9a6d75;
            font-family: "Gowun Batang", serif;
            font-size: 17px;
          }}
          .compat-speech {{
            position: relative;
            min-width: 172px;
            margin-top: -2px;
            padding: 22px 24px;
            border: 2px solid #f6b8c5;
            border-radius: 50%;
            background: #fff3f6;
            color: #7a5158;
            font-family: "Gowun Batang", serif;
            font-size: 15px;
            font-weight: 700;
            line-height: 1.55;
          }}
          .compat-speech::after {{
            content: "";
            position: absolute;
            left: 34px;
            bottom: -8px;
            width: 20px;
            height: 20px;
            border-left: 2px solid #f6b8c5;
            border-bottom: 2px solid #f6b8c5;
            background: #fff3f6;
            transform: rotate(-26deg);
          }}
          .compatibility-form {{
            position: relative;
            z-index: 1;
          }}
          .compat-people {{
            position: relative;
            display: grid;
            grid-template-columns: minmax(0, 1fr) 120px minmax(0, 1fr);
            gap: 22px;
            align-items: center;
          }}
          .compat-person {{
            min-height: 406px;
            margin: 0;
            padding: 24px 22px 22px;
            border: 1px solid #ffd8ce;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.74);
          }}
          .compat-person legend {{
            padding: 0 8px;
            color: #ff6f82;
            font-family: "Gowun Batang", serif;
            font-size: 19px;
            font-weight: 700;
          }}
          .compat-person legend span {{
            margin-right: 8px;
            color: #ffb4c3;
          }}
          .compat-person label,
          .compat-options label {{
            margin: 16px 0 0;
            color: #3f211b;
            font-family: "Gowun Batang", serif;
            font-size: 15px;
            font-weight: 700;
          }}
          .compat-person label:first-of-type {{
            margin-top: 10px;
          }}
          .compat-person label .hint {{
            color: #a88a83;
            font-family: "Gowun Dodum", sans-serif;
            font-size: 11px;
          }}
          .compat-person input,
          .compat-person select,
          .compat-options select {{
            min-height: 52px;
            margin-top: 8px;
            border: 1px solid #ffd7ce;
            border-radius: 11px;
            background-color: rgba(255, 255, 255, 0.92);
            color: #5b3b34;
            font-size: 16px;
            padding: 12px 18px;
          }}
          .compat-person input::placeholder {{
            color: #bca8a2;
          }}
          .compat-person input:focus,
          .compat-person select:focus,
          .compat-options select:focus {{
            border-color: #ff9fad;
            box-shadow: 0 0 0 4px rgba(255, 143, 171, 0.14);
          }}
          .compat-person select,
          .compat-options select {{
            background-position: right 20px center;
            padding-right: 46px;
          }}
          .compat-heart-bridge {{
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 406px;
            color: #ff9fb1;
          }}
          .compat-heart-bridge::before,
          .compat-heart-bridge::after {{
            content: "···";
            position: absolute;
            top: 50%;
            color: #ffc2ce;
            font-size: 28px;
            letter-spacing: 5px;
            transform: translateY(-50%);
          }}
          .compat-heart-bridge::before {{
            left: 2px;
          }}
          .compat-heart-bridge::after {{
            right: -3px;
          }}
          .compat-heart-bridge span {{
            width: 72px;
            height: 72px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin: 0 10px;
            border: 2px solid #ffbfca;
            border-radius: 24px;
            background: radial-gradient(circle at 35% 30%, #ffe4eb, #ffb8c7);
            color: #ffffff;
            font-size: 38px;
            box-shadow: 0 14px 30px -22px rgba(255, 111, 130, 0.74);
          }}
          .compat-options {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 22px;
            margin-top: 18px;
            padding: 20px 22px;
            border: 1px solid #ffd8ce;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.7);
          }}
          .compat-options label {{
            margin: 0;
          }}
          .compat-note {{
            min-height: 58px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            margin: 10px 0 0;
            padding: 12px 22px;
            border: 1px solid #ffd8ce;
            border-radius: 12px;
            background: linear-gradient(90deg, rgba(255, 242, 247, 0.94), rgba(255, 255, 255, 0.78));
            color: #7e5a53;
            font-family: "Gowun Batang", serif;
            font-size: 14px;
            line-height: 1.5;
            text-align: center;
          }}
          .compat-note span {{
            font-family: "Gowun Dodum", sans-serif;
            font-size: 20px;
          }}
          .compat-actions {{
            display: flex;
            justify-content: center;
            margin-top: 20px;
          }}
          .compat-submit {{
            flex: 0 1 420px;
            min-height: 68px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 16px;
            border: 2px solid #ff86a4;
            border-radius: 14px;
            background: linear-gradient(135deg, #ff7898 0%, #ff5f93 100%);
            color: #ffffff;
            font-size: 20px;
            box-shadow: 0 16px 30px -22px rgba(255, 111, 130, 0.78);
          }}
          .compat-submit:hover {{
            background: linear-gradient(135deg, #ff6b91 0%, #f65186 100%);
          }}
          .compat-submit img {{
            width: 58px;
            height: 46px;
            object-fit: contain;
          }}
          .compat-bottom-ora {{
            position: absolute;
            right: 34px;
            bottom: 20px;
            width: 128px;
            height: 142px;
            object-fit: contain;
            pointer-events: none;
          }}
          pre {{
            overflow: auto;
            white-space: pre-wrap;
            line-height: 1.55;
            background: #f7f7f7;
            border-radius: 6px;
            padding: 16px;
          }}
          .hint {{
            color: var(--ink-soft);
            font-size: 13px;
            font-weight: 500;
          }}
          .error {{
            margin-bottom: 16px;
            padding: 16px;
            border-color: #d1242f;
            background: #fff5f5;
          }}
          .capture-preview {{
            margin-top: 16px;
            position: relative;
          }}
          .oracle-result-shell {{
            position: relative;
            width: 100%;
          }}
          main.result-page {{
            width: min(960px, calc(100vw - 40px));
            padding: 24px 0 60px;
          }}
          main.personal-result-page {{
            width: min(1120px, calc(100vw - 56px));
            padding: 24px 0 112px;
          }}
          .personal-result-shell {{
            min-height: 820px;
          }}
          .result-topbar {{
            top: 7px;
          }}
          .personal-result-brand {{
            margin-bottom: 18px;
          }}
          .personal-result-brand .logo {{
            font-size: 39px;
            line-height: 1;
          }}
          .personal-result-brand .tag {{
            margin-top: 14px;
            color: #c8844a;
            font-family: "Gowun Batang", serif;
            font-size: 15px;
            text-transform: none;
          }}
          .personal-result-brand .ornament {{
            margin-top: 12px;
          }}
          .personal-result-brand .ornament::before {{
            background: #fff8ef;
          }}
          .result-sky {{
            position: absolute;
            top: 50px;
            left: 0;
            right: 0;
            height: 150px;
            pointer-events: none;
          }}
          .result-cloud {{
            position: absolute;
            width: 72px;
            height: 31px;
            border: 2px solid #ffd7ce;
            border-top: 0;
            border-radius: 0 0 24px 24px;
            opacity: 0.86;
          }}
          .result-cloud::before,
          .result-cloud::after {{
            content: "";
            position: absolute;
            bottom: 11px;
            border: 2px solid #ffd7ce;
            border-bottom: 0;
            background: #fff8ef;
          }}
          .result-cloud::before {{
            left: 12px;
            width: 28px;
            height: 28px;
            border-radius: 999px 999px 0 0;
          }}
          .result-cloud::after {{
            right: 8px;
            width: 37px;
            height: 37px;
            border-radius: 999px 999px 0 0;
          }}
          .result-cloud-left {{
            left: 126px;
            top: 26px;
          }}
          .result-spark {{
            position: absolute;
            color: #ffc36d;
            font-family: "Gowun Dodum", sans-serif;
            font-size: 30px;
          }}
          .result-spark-left {{
            left: 96px;
            top: 84px;
          }}
          .result-spark-right {{
            right: 218px;
            top: 58px;
          }}
          .result-hero-ora {{
            position: absolute;
            top: 18px;
            right: 78px;
            width: 170px;
            height: 130px;
            object-fit: contain;
            object-position: center top;
          }}
          .result-actions {{
            display: flex;
            justify-content: center;
            gap: 10px;
            margin: 2px 0 22px;
            flex-wrap: wrap;
          }}
          .personal-result-shell .result-actions {{
            position: relative;
            z-index: 2;
            gap: 14px;
            margin: 8px 0 26px;
          }}
          .result-action {{
            min-height: 46px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 11px 18px;
            border: 1px solid var(--line);
            border-radius: 999px;
            background: var(--paper-2);
            color: var(--ink);
            font-family: "Gowun Batang", serif;
            font-size: 15px;
            font-weight: 700;
            text-decoration: none;
            box-shadow: 0 8px 20px -16px rgba(46, 37, 32, 0.5);
            transition: transform 0.15s, box-shadow 0.2s, border-color 0.2s, background 0.2s;
          }}
          .result-action:hover {{
            transform: translateY(-2px);
            border-color: var(--mok);
            box-shadow: 0 14px 28px -18px rgba(46, 37, 32, 0.6);
          }}
          .result-action-primary {{
            background: var(--mok);
            border-color: var(--mok);
            color: #ffffff;
          }}
          .result-action-primary:hover {{
            background: var(--mok-deep);
            border-color: var(--mok-deep);
          }}
          .personal-result-shell .result-action {{
            min-width: 134px;
            min-height: 48px;
            padding: 11px 22px;
            border-color: #ffd8ce;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.82);
            box-shadow: 0 10px 26px -20px rgba(74, 47, 38, 0.42);
          }}
          .personal-result-shell .result-action-primary {{
            min-width: 174px;
            border-color: #42b883;
            background: linear-gradient(135deg, #42b883, #3bbb86);
            color: #ffffff;
          }}
          .workflow-loading {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-top: 16px;
            border-color: #b7c7aa;
            background: #fbfdf8;
          }}
          .personal-result-shell .workflow-loading {{
            position: relative;
            z-index: 2;
            min-height: 88px;
            margin: 0 auto 22px;
            padding: 20px 66px 20px 24px;
            border: 1px solid #bdd8c3;
            border-radius: 10px;
            background:
              radial-gradient(circle at 4% 50%, rgba(66, 184, 131, 0.14), transparent 18%),
              rgba(255, 255, 255, 0.8);
            box-shadow: 0 16px 34px -30px rgba(74, 47, 38, 0.38);
          }}
          .workflow-loading[hidden] {{
            display: none;
          }}
          .workflow-loading strong {{
            display: block;
            margin-bottom: 4px;
          }}
          .workflow-loading p {{
            margin: 0;
          }}
          .personal-result-shell .workflow-loading strong {{
            color: #3f211b;
            font-family: "Gowun Batang", serif;
            font-size: 18px;
          }}
          .personal-result-shell .workflow-loading p {{
            color: #8b6f64;
            font-size: 14px;
          }}
          .loading-spinner {{
            width: 24px;
            height: 24px;
            flex: 0 0 24px;
            border: 3px solid #d9dfd2;
            border-top-color: #2f7d57;
            border-radius: 999px;
            animation: oracle-spin 0.8s linear infinite;
          }}
          .personal-result-shell .loading-spinner {{
            position: relative;
            width: 34px;
            height: 34px;
            flex-basis: 34px;
            border: 0;
            animation: oracle-spin 1.8s linear infinite;
          }}
          .personal-result-shell .loading-spinner::before {{
            content: "☘";
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #42b883;
            font-size: 30px;
          }}
          .loading-heart {{
            display: none;
          }}
          .personal-result-shell .loading-heart {{
            position: absolute;
            right: 28px;
            top: 50%;
            display: block;
            color: #ff8fab;
            font-size: 32px;
            transform: translateY(-50%);
          }}
          @keyframes oracle-spin {{
            to {{
              transform: rotate(360deg);
            }}
          }}
          #capture-preview-image {{
            display: block;
            width: 100%;
            max-height: 70vh;
            object-fit: contain;
            border-radius: 6px;
            background: #111111;
          }}
          .capture-preview.capture-complete #capture-preview-image {{
            filter: blur(12px) saturate(0.75);
          }}
          .personal-result-shell .capture-preview {{
            position: relative;
            z-index: 2;
            overflow: visible;
            padding: 24px 24px 22px;
            border: 1px solid #ffd8ce;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.86);
            box-shadow: 0 18px 44px -34px rgba(74, 47, 38, 0.42);
          }}
          .personal-result-shell .capture-preview h2 {{
            display: flex;
            align-items: center;
            gap: 14px;
            margin: 0 0 22px;
            color: #3f211b;
            font-family: "Gowun Batang", serif;
            font-size: 26px;
          }}
          .personal-result-shell .capture-title-icon {{
            width: 38px;
            height: 38px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border: 1px solid #ffd8ce;
            border-radius: 999px;
            background: #fff9f5;
            color: #ff7c91;
            font-family: "Gowun Dodum", sans-serif;
            font-size: 20px;
          }}
          .personal-result-shell .capture-stage {{
            position: relative;
            overflow: visible;
            border: 1px solid #ffcfd8;
            border-radius: 8px;
            background: #fff9f9;
          }}
          .personal-result-shell .capture-direction {{
            min-height: 86px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 18px;
            padding: 16px 24px;
            border-bottom: 1px solid #ffd8ce;
            background: linear-gradient(90deg, #fff8fb, #fff1f5, #fff8fb);
            text-align: center;
          }}
          .personal-result-shell .capture-direction img {{
            width: 74px;
            height: 58px;
            object-fit: contain;
          }}
          .personal-result-shell .capture-direction strong {{
            color: #4a2f26;
            font-family: "Gowun Batang", serif;
            font-size: 27px;
            line-height: 1.25;
          }}
          .personal-result-shell .capture-direction strong span {{
            color: #ff6f82;
          }}
          .personal-result-shell .capture-heart {{
            color: #ff9caf;
            font-size: 36px;
          }}
          .personal-result-shell .capture-visual {{
            position: relative;
            min-height: 500px;
            overflow: visible;
            background: #111111;
          }}
          .personal-result-shell #capture-preview-image {{
            height: 500px;
            max-height: none;
            object-fit: cover;
            border-radius: 0;
          }}
          .personal-result-shell .camera-corner {{
            position: absolute;
            z-index: 3;
            width: 42px;
            height: 42px;
            color: rgba(255, 255, 255, 0.9);
            border-color: rgba(255, 255, 255, 0.9);
          }}
          .personal-result-shell .camera-corner-tl {{
            left: 26px;
            top: 26px;
            border-left: 3px solid;
            border-top: 3px solid;
            border-radius: 14px 0 0 0;
          }}
          .personal-result-shell .camera-corner-tr {{
            right: 26px;
            top: 26px;
            border-right: 3px solid;
            border-top: 3px solid;
            border-radius: 0 14px 0 0;
          }}
          .personal-result-shell .camera-corner-bl {{
            left: 26px;
            bottom: 26px;
            border-left: 3px solid;
            border-bottom: 3px solid;
            border-radius: 0 0 0 14px;
          }}
          .personal-result-shell .camera-corner-br {{
            right: 26px;
            bottom: 26px;
            border-right: 3px solid;
            border-bottom: 3px solid;
            border-radius: 0 0 14px 0;
          }}
          .personal-result-shell .face-guide {{
            position: absolute;
            z-index: 4;
            left: 50%;
            top: 50%;
            width: 290px;
            height: 330px;
            border: 3px dashed #ffb0c0;
            border-radius: 10px;
            box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.54);
            transform: translate(-50%, -46%);
          }}
          .personal-result-shell .face-guide::before,
          .personal-result-shell .face-guide::after {{
            content: "";
            position: absolute;
            width: 24px;
            height: 24px;
            border: 7px solid #ff9caf;
          }}
          .personal-result-shell .face-guide::before {{
            left: -10px;
            top: -10px;
            border-right: 0;
            border-bottom: 0;
            border-radius: 10px 0 0 0;
          }}
          .personal-result-shell .face-guide::after {{
            right: -10px;
            bottom: -10px;
            border-left: 0;
            border-top: 0;
            border-radius: 0 0 10px 0;
          }}
          .personal-result-shell .capture-side-note {{
            position: absolute;
            z-index: 5;
            width: 174px;
            color: #5b3b34;
            text-align: center;
            pointer-events: none;
          }}
          .personal-result-shell .capture-side-note p {{
            position: relative;
            margin: 0;
            padding: 22px 20px;
            border: 1px solid #ffcfc4;
            border-radius: 44% 48% 42% 46%;
            background: #fff2ed;
            font-family: "Gowun Batang", serif;
            font-size: 15px;
            font-weight: 700;
            line-height: 1.55;
          }}
          .personal-result-shell .capture-side-note p::after {{
            content: "";
            position: absolute;
            bottom: -12px;
            width: 22px;
            height: 22px;
            border-right: 1px solid #ffcfc4;
            border-bottom: 1px solid #ffcfc4;
            background: #fff2ed;
            transform: rotate(45deg);
          }}
          .personal-result-shell .capture-side-note span {{
            position: absolute;
            right: 18px;
            top: 20px;
            color: #ffc36d;
            font-size: 22px;
          }}
          .personal-result-shell .capture-side-note img {{
            display: block;
            width: 110px;
            height: 112px;
            margin: 8px auto 0;
            object-fit: contain;
          }}
          .personal-result-shell .capture-side-note-left {{
            left: -186px;
            bottom: 26px;
          }}
          .personal-result-shell .capture-side-note-left p::after {{
            right: 42px;
          }}
          .personal-result-shell .capture-side-note-right {{
            right: -182px;
            bottom: 18px;
          }}
          .personal-result-shell .capture-side-note-right p::after {{
            left: 42px;
          }}
          .personal-result-shell .capture-tip {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 22px;
            min-height: 54px;
            margin: 18px 0 0;
            border: 1px solid #ffd8ce;
            border-radius: 8px;
            background: linear-gradient(90deg, #fff5f7, #fffafa);
            color: #d36472;
            font-family: "Gowun Batang", serif;
            font-size: 16px;
            font-weight: 700;
          }}
          .personal-result-shell .capture-tip span {{
            color: #ffc36d;
            font-family: "Gowun Dodum", sans-serif;
            font-size: 22px;
          }}
          .capture-privacy-veil {{
            position: absolute;
            inset: 20px 20px 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 10px;
            background:
              radial-gradient(circle at 30% 25%, rgba(255, 255, 255, 0.8), transparent 36%),
              rgba(251, 248, 241, 0.82);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(218, 208, 190, 0.7);
            text-align: center;
          }}
          .capture-privacy-veil[hidden] {{
            display: none;
          }}
          .veil-card {{
            width: min(320px, 82%);
            padding: 22px 20px;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid var(--line-soft);
            box-shadow: 0 16px 36px -26px rgba(46, 37, 32, 0.55);
          }}
          .veil-mark {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 34px;
            height: 34px;
            margin-bottom: 8px;
            border-radius: 999px;
            background: rgba(58, 125, 92, 0.12);
            color: var(--mok-deep);
            font-size: 18px;
          }}
          .veil-card strong {{
            display: block;
            font-family: "Gowun Batang", serif;
            font-size: 18px;
            color: var(--ink);
          }}
          .veil-card p {{
            margin: 6px 0 0;
            color: var(--ink-soft);
            font-size: 13px;
          }}
          @media (max-width: 480px) {{
            main.home-page {{
              width: min(100vw - 24px, 1540px);
              padding-top: 20px;
            }}
            .brand-lockup .logo {{
              font-size: 34px;
            }}
            .brand-lockup .logo .stamp {{
              right: -33px;
              width: 30px;
              height: 30px;
              font-size: 15px;
            }}
            .brand-lockup .tag {{
              font-size: 14px;
            }}
            .home-hero {{
              padding-top: 18px;
            }}
            .home-hero .speech {{
              min-height: 46px;
              padding: 9px 22px;
              font-size: 19px;
            }}
            .home-hero h1 {{
              font-size: 42px;
              margin-top: 22px;
            }}
            .home-hero p {{
              font-size: 17px;
            }}
            .oracle-character {{
              width: min(560px, 122vw);
              margin: 8px -11vw -52px;
            }}
            .mode {{
              min-height: 230px;
              padding: 26px 24px;
            }}
            .mode h2 {{
              font-size: 27px;
            }}
            .mode p {{
              font-size: 15px;
              min-height: 0;
            }}
            .mode .go {{
              min-width: 132px;
              min-height: 48px;
              font-size: 16px;
            }}
            .mode .mode-art {{
              width: 150px;
              height: 170px;
              margin: 0 -12px -12px 0;
            }}
            .mode .mode-art-pair {{
              width: 142px;
              height: 172px;
            }}
            .home-foot {{
              width: calc(100vw - 24px);
              min-height: 74px;
              bottom: max(10px, env(safe-area-inset-bottom));
            }}
            .foot-item {{
              min-height: 56px;
              font-size: 12px;
            }}
            .foot-icon {{
              font-size: 21px;
            }}
            .feature-row {{
              grid-template-columns: 1fr;
            }}
            .feature {{
              min-height: 142px;
              border-right: 0;
              border-bottom: 1px dashed #e8cfc6;
            }}
            .feature:last-child {{
              border-bottom: 0;
            }}
            main.input-page {{
              width: min(100vw - 32px, 860px);
              padding: 34px 0;
            }}
            .input-card {{
              padding: 30px 22px 26px;
            }}
            .actions {{
              flex-direction: column;
            }}
            .brand .logo {{
              font-size: 28px;
            }}
          }}
          @media (max-width: 680px) {{
            .cards {{
              grid-template-columns: 1fr;
              margin-top: -18px;
            }}
            .mode p {{
              min-height: 0;
            }}
          }}
          @media (min-width: 481px) and (max-width: 900px) {{
            .home-hero h1 {{
              font-size: 58px;
            }}
            .home-hero p {{
              font-size: 21px;
            }}
            .feature-row {{
              grid-template-columns: repeat(2, 1fr);
            }}
            .feature:nth-child(2) {{
              border-right: 0;
            }}
            .feature:nth-child(-n+2) {{
              border-bottom: 1px dashed #e8cfc6;
            }}
          }}
        </style>
      </head>
      <body>
        <main{main_class}>
          {heading}
          {body}
        </main>
        <script>

          const homeShell = document.querySelector(".oracle-home-shell");
          if (homeShell) {{
            const homeTabLinks = homeShell.querySelectorAll("[data-home-tab]");
            const showHomeTab = (tabName, updateUrl = true) => {{
              const isMore = tabName === "more";
              homeShell.classList.toggle("more-active", isMore);
              homeTabLinks.forEach((link) => {{
                link.classList.toggle("foot-item-active", link.dataset.homeTab === tabName);
              }});
              if (updateUrl) {{
                const targetHash = isMore ? "#oracle-more" : "#oracle-home";
                history.replaceState(null, "", targetHash);
              }}
              window.scrollTo({{ top: 0, behavior: "smooth" }});
            }};
            homeTabLinks.forEach((link) => {{
              link.addEventListener("click", (event) => {{
                const tabName = link.dataset.homeTab;
                if (tabName === "home" || tabName === "more") {{
                  event.preventDefault();
                  showHomeTab(tabName);
                }}
              }});
            }});
            showHomeTab(window.location.hash === "#oracle-more" ? "more" : "home", false);
          }}

          const forms = document.querySelectorAll(".workflow-form");
          forms.forEach((form) => {{
            form.addEventListener("submit", async (event) => {{
              event.preventDefault();
              const skipFaceInput = form.querySelector('[name="skip_face"]');
              const skipFace = skipFaceInput && skipFaceInput.value === "1";
              const ui = workflowUi();
              if (ui) {{
                prepareWorkflowUi(ui, skipFace);
              }}
              const buttons = form.querySelectorAll("button");
              buttons.forEach(btn => btn.disabled = true);
              try {{
                const startResponse = await fetch(form.dataset.workflowApi, {{
                  method: "POST",
                  body: new FormData(form),
                }});
                const startPayload = await startResponse.json();
                if (startPayload.result_url) {{
                  window.location.href = startPayload.result_url;
                  return;
                }}
                if (!ui) {{
                  throw new Error("결과를 표시할 영역을 찾을 수 없습니다.");
                }}
                await pollWorkflow(startPayload.job_id, ui.status, ui.result, ui.loading);
              }} catch (error) {{
                if (ui) {{
                  ui.result.innerHTML = '<section class="error"><strong>처리 중 오류가 발생했습니다.</strong><p>' + String(error) + '</p></section>';
                  ui.status.textContent = "오류";
                }} else {{
                  window.alert(String(error));
                }}
              }} finally {{
                if (ui && !ui.loading.dataset.workflowResultJob) {{
                  ui.loading.hidden = true;
                  ui.loading.setAttribute("aria-busy", "false");
                }}
                buttons.forEach(btn => btn.disabled = false);
              }}
            }});
          }});

          const resultUi = workflowUi();
          const resultJobId = resultUi && resultUi.loading.dataset.workflowResultJob;
          if (resultJobId) {{
            const skipFace = resultUi.loading.dataset.skipFace === "1";
            prepareWorkflowUi(resultUi, skipFace);
            pollWorkflow(resultJobId, resultUi.status, resultUi.result, resultUi.loading);
          }}

          function workflowUi() {{
            const loading = document.getElementById("workflow-loading");
            const preview = document.querySelector(".capture-preview");
            const previewImage = document.getElementById("capture-preview-image");
            const status = document.getElementById("workflow-status");
            const result = document.getElementById("workflow-result");
            let ui = null;
            if (loading && preview && previewImage && status && result) {{
              ui = {{ loading, preview, previewImage, status, result }};
            }}
            return ui;
          }}

          function prepareWorkflowUi(ui, skipFace) {{
            const loadingTitle = document.getElementById("workflow-loading-title");
            const loadingMessage = document.getElementById("workflow-loading-message");
            ui.result.innerHTML = "";
            ui.loading.hidden = false;
            ui.loading.setAttribute("aria-busy", "true");
            if (skipFace) {{
              ui.preview.hidden = true;
              loadingTitle.textContent = "사주 리포트 생성 중입니다";
              loadingMessage.textContent = "입력한 생년월일과 태어난 시간으로 사주 리포트를 만들고 있습니다. 잠시만 기다려 주세요.";
              ui.status.textContent = "리포트 생성 중";
            }} else {{
              ui.preview.hidden = false;
              loadingTitle.textContent = "촬영 및 리포트 생성 중입니다";
              loadingMessage.textContent = "얼굴 촬영과 사주 분석을 진행한 뒤 리포트를 만들고 있습니다. 잠시만 기다려 주세요.";
              ui.status.textContent = "정면 얼굴을 카메라 중앙에 맞춰 주세요.";
              ui.previewImage.src = "/video-feed?ts=" + Date.now();
            }}
          }}

          async function pollWorkflow(jobId, status, result, loading) {{
            const downloadLink = document.getElementById("download-report-link");
            const preview = document.querySelector(".capture-preview");
            let done = false;
            while (!done) {{
              await new Promise((resolve) => setTimeout(resolve, 5000));
              const response = await fetch("/api/jobs/" + encodeURIComponent(jobId));
              const payload = await response.json();
              if (payload.phase === "generating") {{
                activatePrivacyVeil(preview);
                status.textContent = payload.message || "얼굴 인식 완료, 리포트를 생성하고 있습니다";
              }}
              if (payload.status === "complete") {{
                result.innerHTML = payload.html;
                status.textContent = "완료";
                loading.hidden = true;
                loading.setAttribute("aria-busy", "false");
                if (preview) {{
                  preview.hidden = true;
                }}
                if (downloadLink) {{
                  downloadLink.hidden = false;
                }}
                done = true;
              }} else if (payload.status === "error") {{
                result.innerHTML = payload.html;
                status.textContent = "오류";
                loading.hidden = true;
                loading.setAttribute("aria-busy", "false");
                if (preview) {{
                  activatePrivacyVeil(preview);
                }}
                done = true;
              }} else {{
                if (payload.phase !== "generating") {{
                  status.textContent = payload.message || "촬영 및 리포트 생성 중";
                }}
              }}
            }}
          }}

          function activatePrivacyVeil(preview) {{
            if (!preview) {{
              return;
            }}
            const veil = preview.querySelector(".capture-privacy-veil");
            preview.classList.add("capture-complete");
            if (veil) {{
              veil.hidden = false;
            }}
          }}
        </script>
      </body>
    </html>
    """
    return result
