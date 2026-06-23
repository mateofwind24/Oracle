from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, request
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
    PersonalWorkflowInput,
    run_compatibility_workflow,
    run_personal_workflow,
)


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
                )
                workflow_result = run_personal_workflow(
                    workflow_input=workflow_input,
                    capture_config=load_capture_config(),
                    face_llm_config=load_face_llm_config(),
                    report_llm_config=load_report_llm_config(),
                    manse_db_path=_manse_db_path(),
                    recommendation_db_path=_face_db_path(),
                )
                body = _personal_result(workflow_result.markdown, workflow_result)
            except Exception as exc:
                body = _error_panel(exc) + _personal_form()
        result = _render_page("개인 리포트", body)
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

    @app.get("/health")
    def health():
        result = "ok"
        return result

    result = app
    return result


def serve() -> None:
    config = load_app_config()
    app = create_app()
    app.run(host=config.host, port=config.port, debug=config.debug, threaded=False)


def _form_value(name: str) -> str:
    result = request.form.get(name, "").strip()
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
    result = """
    <form method="post" class="panel">
      <h2>개인 리포트</h2>
      <label>이름<input name="name" required></label>
      <label>생년월일<input name="birth_date" type="date" required></label>
      <label>태어난 시간<span class="hint">선택</span><input name="birth_time" type="time"></label>
      <label>성별<input name="gender" placeholder="예: 남성 또는 여성" required></label>
      <label>추천받고 싶은 이성 얼굴<input name="target_gender" placeholder="예: 남성 또는 여성"></label>
      <button type="submit">개인 리포트 촬영 시작</button>
    </form>
    """
    return result


def _compatibility_form() -> str:
    mode_options = "".join(
        f'<option value="{escape(mode)}">{escape(mode)}</option>'
        for mode in COMPATIBILITY_MODES
    )
    result = f"""
    <form method="post" class="panel">
      <h2>두 사람 궁합</h2>
      <div class="grid">
        <fieldset>
          <legend>첫 번째 사람</legend>
          <label>이름<input name="left_name" required></label>
          <label>생년월일<input name="left_birth_date" type="date" required></label>
          <label>태어난 시간<span class="hint">선택</span><input name="left_birth_time" type="time"></label>
          <label>성별<input name="left_gender" placeholder="예: 남성 또는 여성" required></label>
        </fieldset>
        <fieldset>
          <legend>두 번째 사람</legend>
          <label>이름<input name="right_name" required></label>
          <label>생년월일<input name="right_birth_date" type="date" required></label>
          <label>태어난 시간<span class="hint">선택</span><input name="right_birth_time" type="time"></label>
          <label>성별<input name="right_gender" placeholder="예: 남성 또는 여성" required></label>
        </fieldset>
      </div>
      <label>궁합 모드<select name="mode">{mode_options}</select></label>
      <p class="hint">두 사람 정보를 먼저 입력한 뒤 첫 번째 사람을 촬영하고, 3초 후 두 번째 사람을 촬영합니다.</p>
      <button type="submit">두 사람 궁합 촬영 시작</button>
    </form>
    """
    return result


def _personal_result(markdown: str, workflow_result) -> str:
    recommendation_items = "".join(
        f"<li>{escape(item.display_name)} - {escape(item.reason)}</li>"
        for item in workflow_result.recommendations
    )
    result = f"""
    <section class="panel">
      <h2>개인 리포트 결과</h2>
      <p>캡처 이미지: <code>{escape(str(workflow_result.capture_path))}</code></p>
      <p>리포트 파일: <code>{escape(str(workflow_result.output_path))}</code></p>
      <p>만세력 DB: {escape(workflow_result.manse_status)}</p>
      <h3>추천 후보</h3>
      <ul>{recommendation_items}</ul>
      <pre>{escape(markdown)}</pre>
    </section>
    <p><a href="/">처음으로</a></p>
    """
    return result


def _compatibility_result(markdown: str, workflow_result) -> str:
    result = f"""
    <section class="panel">
      <h2>두 사람 궁합 결과</h2>
      <p>첫 번째 사람 캡처 이미지: <code>{escape(str(workflow_result.left_capture_path))}</code></p>
      <p>두 번째 사람 캡처 이미지: <code>{escape(str(workflow_result.right_capture_path))}</code></p>
      <p>리포트 파일: <code>{escape(str(workflow_result.output_path))}</code></p>
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
        </style>
      </head>
      <body>
        <main>
          <h1>Oracle</h1>
          {body}
        </main>
      </body>
    </html>
    """
    return result
