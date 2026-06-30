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
    load_face_llm_config,
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
        body = """
        <div class="oracle-home-shell">
          <div class="top">
            <div class="logo">ORACLE<span class="stamp serif">運</span></div>
            <div class="tag">관상 &amp; 사주 · 운명 해설</div>
          </div>

          <div class="hero">
            <div class="halo"></div>
            <div class="ring"></div>
            <div class="illust">
              <img src="/static/assets/saju.jpg" alt="관상 일러스트" onerror="this.style.display='none';this.nextElementSibling.style.display='block'">
              <div class="ph" style="display:none"><span class="em">🔮</span>assets/saju.jpg<br>관상 일러스트 삽입</div>
            </div>
          </div>

          <div class="greet serif">당신의 얼굴과 사주를 펼쳐볼까요?</div>

          <div class="cards">
            <a class="mode solo" href="/personal">
              <span class="tagline">혼자서</span>
              <div class="ic">🧑‍🦰</div>
              <h2>개인 리포트</h2>
              <p>관상 · 사주 · 보완 분석에<br>궁합 좋은 얼굴 추천까지</p>
              <span class="go">시작하기 <span class="arr">→</span></span>
            </a>

            <a class="mode pair" href="/compatibility">
              <span class="tagline">둘이서</span>
              <div class="ic">💞</div>
              <h2>두 사람 궁합</h2>
              <p>첫 번째 사람 촬영 후, 3초 뒤<br>두 번째 사람을 순차 촬영</p>
              <span class="go">시작하기 <span class="arr">→</span></span>
            </a>
          </div>

          <div class="home-foot">
            <div class="chips">
              <span class="chip">📷 카메라 관상</span>
              <span class="chip">🗓️ 만세력 사주</span>
              <span class="chip">🔒 100% 온-디바이스</span>
            </div>
            얼굴 이미지와 개인정보는 기기 안에서만 처리되고 외부로 전송되지 않아요.<br>Oracle은 재미를 위한 콘텐츠예요.
          </div>
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
            page_class="input-page",
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
            page_class="result-page",
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
        result = _render_page("두 사람 궁합", body)
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
        from oracle_report.config import load_face_llm_config, load_report_llm_config

        is_face = "face" in prompt_name
        llm_config = load_face_llm_config() if is_face else load_report_llm_config()
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

    @app.get("/favicon.ico")
    def favicon():
        result = ("", 204)
        return result

    result = app
    return result


def serve() -> None:
    config = load_app_config()

    # Run distributed warmup in the background if enabled
    if config.distributed_role in ("master", "hybrid") and config.distributed_warmup:
        import threading
        def run_warmup_background():
            import time
            time.sleep(5.0)  # Wait for slave servers to start fully
            print("[Distributed] Starting LLM warmup for distributed nodes...", flush=True)
            try:
                from oracle_report.workflow import _generate_distributed
                dummy_values = {
                    "name": "더미",
                    "gender": "남성",
                    "birth_datetime": "1990-01-01 12:00",
                    "birth_time_text": "오시(午時)",
                    "quality_text": "정면 얼굴이 안정적으로 감지되었습니다.",
                }
                dummy_categories = [
                    "눈과 눈썹",
                    "얼굴 비율과 중심감",
                    "표정과 소통 분위기",
                    "첫인상 리듬",
                    "관상 기반 생활 팁",
                ]
                _generate_distributed(
                    prompt_name="personal_face_analysis",
                    values=dummy_values,
                    categories=dummy_categories,
                    image_path=None,
                    app_config=config,
                )
                print("[Distributed] LLM warmup complete. Prefix KV caches are now initialized.", flush=True)
            except Exception as e:
                print(f"[Distributed][Warn] Warmup failed: {e}", flush=True)

        threading.Thread(target=run_warmup_background, daemon=True).start()

    app = create_app()
    app.run(host=config.host, port=config.port, debug=config.debug, threaded=True)


def _form_value(name: str) -> str:
    result = request.form.get(name, "").strip()
    return result


def _form_int(name: str, default: int) -> int:
    raw_value = _form_value(name)
    result = default
    if raw_value != "":
        result = int(raw_value)
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
        face_analysis_mode=_form_int("face_analysis_mode", 1),
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
        face_analysis_mode=_form_int("face_analysis_mode", 1),
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
            face_llm_config=load_face_llm_config(),
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
            face_llm_config=load_face_llm_config(),
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


def _personal_form() -> str:
    mode_options = _face_analysis_mode_options()
    gender_options = _gender_options(required=True)
    target_gender_options = _gender_options(required=False)
    birth_time_options = _birth_time_options()
    result = f"""
    <div class="oracle-input-shell">
      <div class="brand">
        <div class="logo">ORACLE</div>
        <div class="tag">관상 &amp; 사주 리포트</div>
        <div class="ornament"></div>
      </div>

      <div class="input-card">
        <div class="card-head">
          <h1>개인 리포트</h1>
          <p>당신의 얼굴과 사주가 그리는 한 장의 이야기</p>
        </div>

        <form method="post" class="workflow-form input-form" data-workflow-api="/api/personal">
          <input type="hidden" name="skip_face" value="0">
          <div class="field lead">
            <label>이름</label>
            <input name="name" placeholder="이름을 입력하세요" required>
          </div>
          <div class="field-stack">
            <div class="field">
              <label>생년월일</label>
              <input name="birth_date" type="date" required>
            </div>
            <div class="field">
              <label>태어난 시간<span class="hint">모르면 '모름'을 선택하세요</span></label>
              <select name="birth_time">{birth_time_options}</select>
            </div>
            <div class="field">
              <label>성별</label>
              <select name="gender" required>{gender_options}</select>
            </div>
            <div class="field">
              <label>추천받고 싶은 얼굴 성별</label>
              <select name="target_gender">{target_gender_options}</select>
            </div>
            <div class="field">
              <label>관상 분석 모드</label>
              <select name="face_analysis_mode">{mode_options}</select>
            </div>
          </div>
          <div class="actions">
            <button type="submit" class="btn btn-primary" onclick="this.form.skip_face.value='0';">개인 리포트 촬영 시작</button>
            <button type="submit" class="btn btn-ghost" onclick="this.form.skip_face.value='1';">관상 없이 사주만 보기</button>
          </div>
        </form>

        <p class="footnote">입력한 정보와 촬영 이미지는 기기 안에서만 처리돼요.<br>Oracle은 재미를 위한 콘텐츠예요.</p>
      </div>
    </div>
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
    <form method="post" class="panel workflow-form" data-workflow-api="/api/compatibility">
      <h2>두 사람 궁합</h2>
      <div class="grid">
        <fieldset>
          <legend>첫 번째 사람</legend>
          <label>이름<input name="left_name" required></label>
          <label>생년월일<input name="left_birth_date" type="date" required></label>
          <label>태어난 시간<span class="hint">모르면 모름 선택</span><select name="left_birth_time">{birth_time_options}</select></label>
          <label>성별<select name="left_gender" required>{gender_options}</select></label>
        </fieldset>
        <fieldset>
          <legend>두 번째 사람</legend>
          <label>이름<input name="right_name" required></label>
          <label>생년월일<input name="right_birth_date" type="date" required></label>
          <label>태어난 시간<span class="hint">모르면 모름 선택</span><select name="right_birth_time">{birth_time_options}</select></label>
          <label>성별<select name="right_gender" required>{gender_options}</select></label>
        </fieldset>
      </div>
      <label>궁합 모드<select name="mode">{mode_options}</select></label>
      <label>관상 분석 모드<select name="face_analysis_mode">{face_mode_options}</select></label>
      <p class="hint">두 사람 정보를 먼저 입력한 뒤 첫 번째 사람을 촬영하고, 3초 후 두 번째 사람을 촬영합니다.</p>
      <button type="submit">두 사람 궁합 촬영 시작</button>
    </form>
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
    selected_mode = os.getenv("ORACLE_FACE_ANALYSIS_MODE", "1")
    mode_one_selected = " selected" if selected_mode == "1" else ""
    mode_two_selected = " selected" if selected_mode == "2" else ""
    result = f"""
      <option value="1"{mode_one_selected}>1 - 이미지 LLM 분석</option>
      <option value="2"{mode_two_selected}>2 - 랜드마크 룰 기반 분석</option>
    """
    return result


def _personal_result_page(job_id: str, skip_face: bool) -> str:
    result = f"""
    <div class="oracle-result-shell">
      <div class="brand">
        <div class="logo">ORACLE</div>
        <div class="tag">개인 리포트 결과</div>
        <div class="ornament"></div>
      </div>
      <div class="result-actions">
        <a class="result-action" href="/personal">입력 다시 하기</a>
        <a class="result-action" href="/">처음으로</a>
        <a id="download-report-link" class="result-action result-action-primary download-link" href="/api/jobs/{escape(job_id)}/download" hidden>리포트 다운로드</a>
      </div>
      {_capture_preview_panel(job_id=job_id, skip_face=skip_face)}
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
    result = f"""
    <section id="workflow-loading" class="panel workflow-loading" role="status" aria-live="polite" aria-busy="true"{job_attr}{skip_attr}{loading_hidden}>
      <span class="loading-spinner" aria-hidden="true"></span>
      <div>
        <strong id="workflow-loading-title">{loading_title}</strong>
        <p id="workflow-loading-message" class="hint">{loading_message}</p>
      </div>
    </section>
    <section class="panel capture-preview"{preview_hidden}>
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
            --paper: #f6f1e7;
            --paper-2: #fbf8f1;
            --ink: #2a2520;
            --ink-soft: #6b6256;
            --line: #dad0be;
            --line-soft: #e7dece;
            --mok: #3a7d5c;
            --mok-deep: #2f6549;
            --hwa: #c25239;
            --gold: #a8823c;
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
              radial-gradient(circle at 18% 8%, rgba(58, 125, 92, 0.05), transparent 42%),
              radial-gradient(circle at 86% 26%, rgba(194, 82, 57, 0.045), transparent 42%);
            -webkit-font-smoothing: antialiased;
          }}
          main {{
            width: min(960px, calc(100vw - 32px));
            margin: 0 auto;
            padding: 32px 0;
          }}
          main.input-page {{
            width: min(860px, calc(100vw - 48px));
            padding: 24px 0;
          }}
          main.home-page {{
            width: min(860px, calc(100vw - 40px));
            padding: 40px 0 60px;
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
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 100%;
          }}
          .top {{
            text-align: center;
            margin-bottom: 30px;
          }}
          .top .logo {{
            font-family: "Song Myung", serif;
            font-size: 46px;
            letter-spacing: 0.2em;
            color: var(--ink);
            position: relative;
            display: inline-block;
          }}
          .top .logo .stamp {{
            position: absolute;
            top: -6px;
            right: -46px;
            width: 40px;
            height: 40px;
            border: 2px solid var(--hwa);
            border-radius: 8px;
            color: var(--hwa);
            font-size: 19px;
            display: flex;
            align-items: center;
            justify-content: center;
            transform: rotate(9deg);
            opacity: 0.9;
          }}
          .top .tag {{
            font-size: 13px;
            letter-spacing: 0.4em;
            color: var(--gold);
            text-transform: uppercase;
            margin-top: 14px;
          }}
          .hero {{
            position: relative;
            margin: 6px 0 36px;
            width: 230px;
            height: 230px;
            display: flex;
            align-items: center;
            justify-content: center;
          }}
          .hero .halo {{
            position: absolute;
            inset: 0;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(204, 154, 59, 0.18), rgba(194, 82, 57, 0.06) 60%, transparent 72%);
          }}
          .hero .ring {{
            position: absolute;
            inset: 14px;
            border: 1.5px dashed var(--gold);
            border-radius: 50%;
            opacity: 0.5;
            animation: oracle-spin 40s linear infinite;
          }}
          .hero .illust {{
            position: relative;
            width: 200px;
            height: 200px;
            border-radius: 50%;
            overflow: hidden;
            background: #ffffff;
            border: 3px solid #ffffff;
            box-shadow: 0 14px 36px -14px rgba(46, 37, 32, 0.5);
            display: flex;
            align-items: flex-end;
            justify-content: center;
          }}
          .hero .illust img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            object-position: center 12%;
          }}
          .illust .ph {{
            font-size: 12px;
            color: #b6a98c;
            text-align: center;
            padding: 14px;
            line-height: 1.6;
          }}
          .illust .ph .em {{
            font-size: 40px;
            display: block;
            margin-bottom: 6px;
          }}
          .greet {{
            font-family: "Gowun Batang", serif;
            font-size: 18px;
            color: var(--ink-soft);
            margin-bottom: 24px;
          }}
          .cards {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            width: 100%;
            max-width: 780px;
          }}
          .mode {{
            position: relative;
            display: block;
            background: var(--paper-2);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 30px 26px 26px;
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
            inset: 6px;
            border: 1px solid var(--line-soft);
            border-radius: 9px;
            pointer-events: none;
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
          .mode .ic {{
            width: 54px;
            height: 54px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 16px;
            font-size: 27px;
          }}
          .mode.solo .ic {{
            background: rgba(58, 125, 92, 0.12);
          }}
          .mode.pair .ic {{
            background: rgba(194, 82, 57, 0.12);
          }}
          .mode h2 {{
            font-family: "Gowun Batang", serif;
            font-size: 22px;
            font-weight: 700;
            margin: 0 0 8px;
          }}
          .mode p {{
            font-size: 14px;
            color: var(--ink-soft);
            line-height: 1.6;
            min-height: 42px;
            margin: 0;
          }}
          .mode .go {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            margin-top: 18px;
            font-family: "Gowun Batang", serif;
            font-size: 14px;
            font-weight: 700;
          }}
          .mode.solo .go {{
            color: var(--mok-deep);
          }}
          .mode.pair .go {{
            color: var(--hwa);
          }}
          .mode .go .arr {{
            transition: transform 0.2s;
          }}
          .mode:hover .go .arr {{
            transform: translateX(4px);
          }}
          .mode .tagline {{
            position: absolute;
            top: 18px;
            right: 18px;
            font-size: 11px;
            padding: 4px 10px;
            border-radius: 20px;
            font-family: "Gowun Dodum", sans-serif;
          }}
          .mode.solo .tagline {{
            background: rgba(58, 125, 92, 0.1);
            color: var(--mok-deep);
          }}
          .mode.pair .tagline {{
            background: rgba(194, 82, 57, 0.1);
            color: var(--hwa);
          }}
          .home-foot {{
            margin-top: 34px;
            text-align: center;
            font-size: 12px;
            color: var(--ink-soft);
            line-height: 1.8;
            max-width: 520px;
          }}
          .home-foot .chips {{
            display: flex;
            gap: 8px;
            justify-content: center;
            flex-wrap: wrap;
            margin-bottom: 14px;
          }}
          .home-foot .chip {{
            background: var(--paper-2);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 6px 14px;
            font-size: 12px;
            color: var(--ink);
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
            letter-spacing: 0.22em;
            color: var(--ink);
          }}
          .brand .tag {{
            font-size: 12px;
            letter-spacing: 0.4em;
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
            letter-spacing: 0.02em;
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
            letter-spacing: 0.02em;
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
            width: 100%;
          }}
          main.result-page {{
            width: min(960px, calc(100vw - 40px));
            padding: 24px 0 60px;
          }}
          .result-actions {{
            display: flex;
            justify-content: center;
            gap: 10px;
            margin: 2px 0 22px;
            flex-wrap: wrap;
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
          .workflow-loading {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-top: 16px;
            border-color: #b7c7aa;
            background: #fbfdf8;
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
          .loading-spinner {{
            width: 24px;
            height: 24px;
            flex: 0 0 24px;
            border: 3px solid #d9dfd2;
            border-top-color: #2f7d57;
            border-radius: 999px;
            animation: oracle-spin 0.8s linear infinite;
          }}
          @keyframes oracle-spin {{
            to {{
              transform: rotate(360deg);
            }}
          }}
          .capture-preview img {{
            display: block;
            width: 100%;
            max-height: 70vh;
            object-fit: contain;
            border-radius: 6px;
            background: #111111;
          }}
          .capture-preview.capture-complete img {{
            filter: blur(12px) saturate(0.75);
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
            }}
            .top .logo {{
              font-size: 38px;
            }}
            .top .logo .stamp {{
              right: -40px;
            }}
            .mode p {{
              min-height: 0;
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
          document.querySelectorAll('input[type="date"]').forEach((input) => {{
            input.addEventListener("input", (e) => {{
              const val = e.target.value;
              if (val) {{
                const parts = val.split("-");
                if (parts[0] && parts[0].length > 4) {{
                  parts[0] = parts[0].substring(0, 4);
                  e.target.value = parts.join("-");
                }}
              }}
            }});

            input.addEventListener("keyup", (e) => {{
              const val = e.target.value;
              if (val) {{
                const parts = val.split("-");
                if (parts[0] && parts[0].length === 4 && !e.target.dataset.focusedNext) {{
                  e.target.dataset.focusedNext = "true";
                  setTimeout(() => {{
                    const tabEvent = new KeyboardEvent("keydown", {{
                      key: "Tab",
                      keyCode: 9,
                      code: "Tab",
                      bubbles: true,
                      cancelable: true
                    }});
                    e.target.dispatchEvent(tabEvent);
                  }}, 50);
                }} else if (parts[0] && parts[0].length < 4) {{
                  delete e.target.dataset.focusedNext;
                }}
              }}
            }});
          }});

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
