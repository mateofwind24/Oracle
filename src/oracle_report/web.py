from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import threading
import uuid

from flask import Flask, Response, jsonify, request
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
from oracle_report.vision.runtime import run_capture


@dataclass
class _WorkflowJob:
    status: str
    html: str = ""
    error: str = ""


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
        <section class="menu">
          <a class="menu-card" href="/personal">
            <strong>개인 리포트</strong>
            <span>관상 + 사주 + 궁합 좋은 얼굴 추천</span>
          </a>
          <a class="menu-card" href="/compatibility">
            <strong>두 사람 궁합</strong>
            <span>첫 번째 사람 촬영 후 3초 뒤 두 번째 사람을 순차 촬영</span>
          </a>
        </section>
        """
        result = _render_page("Oracle", body)
        return result

    @app.route("/personal", methods=["GET", "POST"])
    def personal():
        body = _personal_form()
        if request.method == "POST":
            try:
                workflow_input = PersonalWorkflowInput(
                    name=_form_value("name"),
                    birth_date=_form_value("birth_date"),
                    birth_time=_form_value("birth_time"),
                    gender=_form_value("gender"),
                    target_gender=_form_value("target_gender"),
                    face_analysis_mode=_form_int("face_analysis_mode", 1),
                    skip_face=_form_bool("skip_face", False),
                )
                workflow_result = run_personal_workflow(
                    workflow_input=workflow_input,
                    capture_config=load_capture_config(),
                    face_llm_config=load_face_llm_config(),
                    report_llm_config=load_report_llm_config(),
                    manse_db_path=_manse_db_path(),
                    recommendation_db_path=_face_db_path(),
                )
                body = _personal_result(workflow_result)
            except Exception as exc:
                body = _error_panel(exc) + _personal_form()
        result = _render_page("개인 리포트", body)
        return result

    @app.post("/api/personal")
    def personal_api():
        workflow_input = PersonalWorkflowInput(
            name=_form_value("name"),
            birth_date=_form_value("birth_date"),
            birth_time=_form_value("birth_time"),
            gender=_form_value("gender"),
            target_gender=_form_value("target_gender"),
            face_analysis_mode=_form_int("face_analysis_mode", 1),
            skip_face=_form_bool("skip_face", False),
        )

        def run_job() -> str:
            workflow_result = run_personal_workflow(
                workflow_input=workflow_input,
                capture_config=load_capture_config(),
                face_llm_config=load_face_llm_config(),
                report_llm_config=load_report_llm_config(),
                manse_db_path=_manse_db_path(),
                recommendation_db_path=_face_db_path(),
                capture_runner=_preview_capture_runner,
            )
            result = _personal_result(workflow_result)
            return result

        job_id = _start_workflow_job(run_job)
        result = jsonify({"job_id": job_id})
        return result

    @app.route("/compatibility", methods=["GET", "POST"])
    def compatibility():
        body = _compatibility_form()
        if request.method == "POST":
            try:
                workflow_input = CompatibilityWorkflowInput(
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
                workflow_result = run_compatibility_workflow(
                    workflow_input=workflow_input,
                    capture_config=load_capture_config(),
                    face_llm_config=load_face_llm_config(),
                    report_llm_config=load_report_llm_config(),
                    manse_db_path=_manse_db_path(),
                )
                body = _compatibility_result(
                    workflow_result.markdown,
                    workflow_result,
                )
            except Exception as exc:
                body = _error_panel(exc) + _compatibility_form()
        result = _render_page("두 사람 궁합", body)
        return result

    @app.post("/api/compatibility")
    def compatibility_api():
        workflow_input = CompatibilityWorkflowInput(
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

        def run_job() -> str:
            workflow_result = run_compatibility_workflow(
                workflow_input=workflow_input,
                capture_config=load_capture_config(),
                face_llm_config=load_face_llm_config(),
                report_llm_config=load_report_llm_config(),
                manse_db_path=_manse_db_path(),
                capture_runner=_preview_capture_runner,
            )
            result = _compatibility_result(
                workflow_result.markdown,
                workflow_result,
            )
            return result

        job_id = _start_workflow_job(run_job)
        result = jsonify({"job_id": job_id})
        return result

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str):
        job = _get_job(job_id)
        status_code = 200
        payload = {"status": "missing", "html": "", "error": "job not found"}
        if job is not None:
            payload = {"status": job.status, "html": job.html, "error": job.error}
        else:
            status_code = 404
        result = jsonify(payload), status_code
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

    result = app
    return result


def serve() -> None:
    config = load_app_config()
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


def _preview_capture_runner(config, output_dir: Path | None = None):
    result = run_capture(
        config,
        output_dir=output_dir,
        frame_callback=_PREVIEW_STREAM.publish,
    )
    return result


def _start_workflow_job(run_job) -> str:
    job_id = uuid.uuid4().hex
    _set_job(job_id, _WorkflowJob(status="running"))
    thread = threading.Thread(
        target=_run_workflow_job,
        args=(job_id, run_job),
        daemon=True,
    )
    thread.start()
    result = job_id
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
            html = run_job()
            _set_job(job_id, _WorkflowJob(status="complete", html=html))
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
    result = Path(os.getenv("ORACLE_MANSE_DB_PATH", "data/manse.sqlite"))
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
    <form method="post" class="panel workflow-form" data-workflow-api="/api/personal">
      <h2>개인 리포트</h2>
      <input type="hidden" name="skip_face" value="0">
      <label>이름<input name="name" required></label>
      <label>생년월일<input name="birth_date" type="date" required></label>
      <label>태어난 시간<span class="hint">모르면 모름 선택</span><select name="birth_time">{birth_time_options}</select></label>
      <label>성별<select name="gender" required>{gender_options}</select></label>
      <label>추천받고 싶은 얼굴 성별<select name="target_gender">{target_gender_options}</select></label>
      <label>관상 분석 모드<select name="face_analysis_mode">{mode_options}</select></label>
      <div style="display: flex; gap: 8px; flex-wrap: wrap;">
        <button type="submit" onclick="this.form.skip_face.value='0';">개인 리포트 촬영 시작</button>
        <button type="submit" onclick="this.form.skip_face.value='1';" style="background: #4b5563; border-color: #4b5563;">관상 없이 사주만 보기</button>
      </div>
    </form>
    {_capture_preview_panel()}
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
    {_capture_preview_panel()}
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
    options = (
        ("", "모름"),
        ("00:00", "자시 (23:00-00:59)"),
        ("02:00", "축시 (01:00-02:59)"),
        ("04:00", "인시 (03:00-04:59)"),
        ("06:00", "묘시 (05:00-06:59)"),
        ("08:00", "진시 (07:00-08:59)"),
        ("10:00", "사시 (09:00-10:59)"),
        ("12:00", "오시 (11:00-12:59)"),
        ("14:00", "미시 (13:00-14:59)"),
        ("16:00", "신시 (15:00-16:59)"),
        ("18:00", "유시 (17:00-18:59)"),
        ("20:00", "술시 (19:00-20:59)"),
        ("22:00", "해시 (21:00-22:59)"),
    )
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


def _capture_preview_panel() -> str:
    result = """
    <section class="panel capture-preview" hidden>
      <h2>실시간 촬영 상태</h2>
      <img id="capture-preview-image" alt="실시간 촬영 상태">
      <p id="workflow-status" class="hint">촬영 준비 중</p>
    </section>
    <section id="workflow-result"></section>
    """
    return result


def _personal_result(workflow_result: PersonalWorkflowResult) -> str:
    result = workflow_result.report_fragment_html
    return result


def _compatibility_result(markdown: str, workflow_result) -> str:
    result = f"""
    <section class="panel">
      <h2>두 사람 궁합 결과</h2>
      <p>첫 번째 사람 캡처 이미지: <code>{escape(str(workflow_result.left_capture_path))}</code></p>
      <p>두 번째 사람 캡처 이미지: <code>{escape(str(workflow_result.right_capture_path))}</code></p>
      <p>리포트 파일: <code>{escape(str(workflow_result.output_path))}</code></p>
      <p>수행 시간 로그: <code>{escape(str(workflow_result.timing_log_path))}</code></p>
      <p>첫 번째 사람 만세력 DB: {escape(workflow_result.left_manse_status)}</p>
      <p>두 번째 사람 만세력 DB: {escape(workflow_result.right_manse_status)}</p>
      <pre>{escape(markdown)}</pre>
    </section>
    <p><a href="/">처음으로</a></p>
    """
    return result


def _error_panel(exc: Exception) -> str:
    result = f"""
    <section class="error">
      <strong>처리 중 오류가 발생했습니다.</strong>
      <p>{escape(str(exc))}</p>
    </section>
    """
    return result


def _render_page(title: str, body: str) -> str:
    result = f"""
    <!doctype html>
    <html lang="ko">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{escape(title)}</title>
        <style>
          :root {{
            color-scheme: light;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #f5f5f2;
            color: #202124;
          }}
          body {{
            margin: 0;
          }}
          main {{
            width: min(960px, calc(100vw - 32px));
            margin: 0 auto;
            padding: 32px 0;
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
            border: 1px solid #d7d5cd;
            border-radius: 8px;
            background: #ffffff;
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
            display: grid;
            gap: 6px;
            margin-bottom: 14px;
            font-weight: 600;
          }}
          input, select, button {{
            min-height: 42px;
            border: 1px solid #c7c4bb;
            border-radius: 6px;
            padding: 8px 10px;
            font: inherit;
          }}
          button {{
            border-color: #1f6feb;
            background: #1f6feb;
            color: #ffffff;
            font-weight: 700;
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
            color: #6b6f76;
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
          }}
          .capture-preview img {{
            display: block;
            width: 100%;
            max-height: 70vh;
            object-fit: contain;
            border-radius: 6px;
            background: #111111;
          }}
        </style>
      </head>
      <body>
        <main>
          <h1>Oracle</h1>
          {body}
        </main>
        <script>
          const forms = document.querySelectorAll(".workflow-form");
          forms.forEach((form) => {{
            form.addEventListener("submit", async (event) => {{
              event.preventDefault();
              const skipFaceInput = form.querySelector('[name="skip_face"]');
              const skipFace = skipFaceInput && skipFaceInput.value === "1";
              const preview = document.querySelector(".capture-preview");
              const previewImage = document.getElementById("capture-preview-image");
              const status = document.getElementById("workflow-status");
              const result = document.getElementById("workflow-result");
              result.innerHTML = "";
              if (skipFace) {{
                preview.hidden = true;
                status.textContent = "리포트 생성 중";
              }} else {{
                preview.hidden = false;
                status.textContent = "촬영 중";
                previewImage.src = "/video-feed?ts=" + Date.now();
              }}
              const buttons = form.querySelectorAll("button");
              buttons.forEach(btn => btn.disabled = true);
              try {{
                const startResponse = await fetch(form.dataset.workflowApi, {{
                  method: "POST",
                  body: new FormData(form),
                }});
                const startPayload = await startResponse.json();
                await pollWorkflow(startPayload.job_id, status, result);
              }} catch (error) {{
                result.innerHTML = '<section class="error"><strong>처리 중 오류가 발생했습니다.</strong><p>' + String(error) + '</p></section>';
                status.textContent = "오류";
              }} finally {{
                buttons.forEach(btn => btn.disabled = false);
              }}
            }});
          }});

          async function pollWorkflow(jobId, status, result) {{
            let done = false;
            while (!done) {{
              await new Promise((resolve) => setTimeout(resolve, 5000));
              const response = await fetch("/api/jobs/" + encodeURIComponent(jobId));
              const payload = await response.json();
              if (payload.status === "complete") {{
                result.innerHTML = payload.html;
                status.textContent = "완료";
                done = true;
              }} else if (payload.status === "error") {{
                result.innerHTML = payload.html;
                status.textContent = "오류";
                done = true;
              }} else {{
                status.textContent = "촬영 및 리포트 생성 중";
              }}
            }}
          }}
        </script>
      </body>
    </html>
    """
    return result
