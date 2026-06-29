from __future__ import annotations

import pytest


def test_favicon_returns_no_content() -> None:
    pytest.importorskip("flask")
    from oracle_report.web import create_app

    app = create_app()

    response = app.test_client().get("/favicon.ico")

    assert response.status_code == 204


def test_home_page_uses_oracle_home_layout_and_hover_effects() -> None:
    pytest.importorskip("flask")
    from oracle_report.web import create_app

    app = create_app()

    response = app.test_client().get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'class="oracle-home-shell"' in html
    assert 'ORACLE<span class="stamp serif">運</span>' in html
    assert 'class="hero"' in html
    assert 'class="mode solo"' in html
    assert 'href="/personal"' in html
    assert 'class="mode pair"' in html
    assert 'href="/compatibility"' in html
    assert ".mode:hover" in html
    assert "transform: translateY(-6px);" in html
    assert ".mode:hover .go .arr" in html
    assert "transform: translateX(4px);" in html


def test_home_page_serves_saju_illustration() -> None:
    pytest.importorskip("flask")
    from oracle_report.web import create_app

    app = create_app()
    client = app.test_client()

    response = client.get("/")
    html = response.get_data(as_text=True)
    image_response = client.get("/static/assets/saju.jpg")

    assert response.status_code == 200
    assert 'src="/static/assets/saju.jpg"' in html
    assert image_response.status_code == 200
    assert image_response.content_type == "image/jpeg"


def test_personal_input_page_links_to_separate_result_page() -> None:
    pytest.importorskip("flask")
    from oracle_report.web import create_app

    app = create_app()

    response = app.test_client().get("/personal")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'data-workflow-api="/api/personal"' in html
    assert 'id="workflow-loading"' not in html
    assert 'id="workflow-result"' not in html
    assert "startPayload.result_url" in html
    assert "window.location.href" in html


def test_personal_result_page_includes_workflow_loading_state() -> None:
    pytest.importorskip("flask")
    from oracle_report.web import create_app

    app = create_app()

    response = app.test_client().get("/personal/result/test-job?skip_face=1")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'data-workflow-result-job="test-job"' in html
    assert 'data-skip-face="1"' in html
    assert 'id="workflow-loading"' in html
    assert 'role="status"' in html
    assert 'class="capture-privacy-veil"' in html
    assert "사주 리포트 생성 중입니다" in html
    assert "pollWorkflow(resultJobId, resultUi.status, resultUi.result, resultUi.loading)" in html
    assert 'id="download-report-link"' in html
    assert 'class="result-action result-action-primary download-link"' in html
    assert 'href="/api/jobs/test-job/download"' in html
    assert "리포트 다운로드" in html
    assert "min-height: 46px;" in html
    assert "border-radius: 999px;" in html
    assert "activatePrivacyVeil" in html
    assert "capture-complete" in html


def test_running_job_status_includes_phase() -> None:
    pytest.importorskip("flask")
    from oracle_report.web import _WorkflowJob, _set_job, create_app

    app = create_app()
    _set_job(
        "phase-job",
        _WorkflowJob(
            status="running",
            phase="generating",
            message="리포트 생성 중",
        ),
    )

    response = app.test_client().get("/api/jobs/phase-job")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["status"] == "running"
    assert payload["phase"] == "generating"
    assert payload["message"] == "리포트 생성 중"


def test_completed_job_downloads_report_html() -> None:
    pytest.importorskip("flask")
    from oracle_report.web import _WorkflowJob, _set_job, create_app

    app = create_app()
    _set_job(
        "downloadable-job",
        _WorkflowJob(
            status="complete",
            html="<section>fragment</section>",
            download_html="<!doctype html><html><body>full report</body></html>",
            download_filename="personal_report.html",
        ),
    )

    response = app.test_client().get("/api/jobs/downloadable-job/download")

    assert response.status_code == 200
    assert response.content_type == "text/html; charset=utf-8"
    assert response.headers["Content-Disposition"] == (
        'attachment; filename="personal_report.html"'
    )
    assert "full report" in response.get_data(as_text=True)


def test_running_job_download_returns_not_found() -> None:
    pytest.importorskip("flask")
    from oracle_report.web import _WorkflowJob, _set_job, create_app

    app = create_app()
    _set_job("running-job", _WorkflowJob(status="running"))

    response = app.test_client().get("/api/jobs/running-job/download")

    assert response.status_code == 404


def test_personal_page_uses_oracle_input_card_layout() -> None:
    pytest.importorskip("flask")
    from oracle_report.web import create_app

    app = create_app()

    response = app.test_client().get("/personal")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'class="oracle-input-shell"' in html
    assert 'class="brand"' in html
    assert 'class="input-card"' in html
    assert "당신의 얼굴과 사주가 그리는 한 장의 이야기" in html
    assert "입력한 정보와 촬영 이미지는 기기 안에서만 처리돼요." in html
    assert 'data-workflow-api="/api/personal"' in html


def test_personal_page_prevents_input_overflow_and_uses_wide_single_column_layout() -> None:
    pytest.importorskip("flask")
    from oracle_report.web import create_app

    app = create_app()

    response = app.test_client().get("/personal")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "* {" in html
    assert "box-sizing: border-box;" in html
    assert "width: min(860px, calc(100vw - 48px));" in html
    assert 'class="field-stack"' in html
    assert "grid-template-columns: 1fr;" in html


def test_compatibility_input_page_links_to_separate_result_page() -> None:
    pytest.importorskip("flask")
    from oracle_report.web import create_app

    app = create_app()

    response = app.test_client().get("/compatibility")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'data-workflow-api="/api/compatibility"' in html
    assert 'id="workflow-loading"' not in html
    assert 'id="workflow-result"' not in html
    assert "startPayload.result_url" in html
    assert "window.location.href" in html


def test_compatibility_result_page_includes_workflow_loading_state() -> None:
    pytest.importorskip("flask")
    from oracle_report.web import create_app

    app = create_app()

    response = app.test_client().get("/compatibility/result/pair-job")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "두 사람 궁합 결과" in html
    assert 'data-workflow-result-job="pair-job"' in html
    assert 'id="workflow-loading"' in html
    assert 'href="/compatibility"' in html
    assert 'href="/api/jobs/pair-job/download"' in html
    assert "리포트 다운로드" in html


def test_compatibility_api_returns_result_url(monkeypatch) -> None:
    pytest.importorskip("flask")
    import oracle_report.web as web

    monkeypatch.setattr(web, "_start_compatibility_workflow_job", lambda _: "pair-job")
    app = web.create_app()

    response = app.test_client().post(
        "/api/compatibility",
        data={
            "left_name": "A",
            "left_birth_date": "1997-04-12",
            "left_birth_time": "16:00",
            "left_gender": "여성",
            "right_name": "B",
            "right_birth_date": "1996-03-10",
            "right_birth_time": "10:00",
            "right_gender": "남성",
            "mode": "연인",
            "face_analysis_mode": "2",
        },
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["job_id"] == "pair-job"
    assert payload["result_url"] == "/compatibility/result/pair-job"
