from __future__ import annotations

from io import BytesIO
import json
import smtplib
import zipfile

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
    assert "로그인" not in html
    assert "회원가입" not in html
    assert 'aria-label="메뉴"' not in html
    assert "오늘도 운명을 함께 찾아볼까요?" in html
    assert "오라와 함께 나의 운명과 인연을" in html
    assert 'class="hero-orbit"' not in html
    assert 'class="cloud cloud-left"' not in html
    assert 'class="mode solo"' in html
    assert "나의 운세 보기" in html
    assert 'href="/personal"' in html
    assert 'class="mode pair"' in html
    assert "우리 궁합 보기" in html
    assert 'href="/compatibility"' in html
    assert ".mode:hover" in html
    assert "transform: translateY(-6px);" in html
    assert ".mode:hover .go .arr" in html
    assert "transform: translateX(4px);" in html


def test_home_page_serves_oracle_character_illustration() -> None:
    pytest.importorskip("flask")
    from oracle_report.web import create_app

    app = create_app()
    client = app.test_client()

    response = client.get("/")
    html = response.get_data(as_text=True)
    image_response = client.get("/static/assets/oracle-character.png")

    assert response.status_code == 200
    assert 'src="/static/assets/oracle-character.png"' in html
    assert image_response.status_code == 200
    assert image_response.content_type == "image/png"
    assert image_response.data.startswith(b"\x89PNG\r\n\x1a\n")


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

    response = app.test_client().get("/compatibility/result/test-job")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'data-workflow-result-job="test-job"' in html
    assert 'data-skip-face="0"' in html
    assert 'id="workflow-loading"' in html
    assert 'class="capture-privacy-veil"' in html
    assert "촬영 및 리포트 생성 중입니다" in html
    assert 'href="/compatibility"' in html
    assert 'href="/api/jobs/test-job/download"' in html


def test_compare_camera_page_uses_live_metric_dashboard() -> None:
    pytest.importorskip("flask")
    from oracle_report.web import create_app

    app = create_app()

    response = app.test_client().get("/compare-camera")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "실시간 랜드마크 측정" in html
    assert "실시간 metric" in html
    assert "판정 결과" in html
    assert ".compare-camera-shell .capture-debug-panel" in html
    assert "max-height: none;" in html
    assert "column-count: 2;" in html
    assert "height: 320px;" in html
    assert "aspect-ratio: 0.879;" in html
    assert "height: 49.5%;" in html
    assert "transform: translate(-50%, -50%);" in html
    assert "object-fit: cover;" not in html
    assert "main.compare-camera-page" in html
    assert "oracle-solo-card.png" not in html
    assert "oracle-pair-card.png" not in html
    assert "pollWorkflow(payload.job_id" not in html


def test_compare_camera_start_returns_live_status(monkeypatch) -> None:
    pytest.importorskip("flask")
    from oracle_report.web import create_app

    monkeypatch.setattr(
        "oracle_report.web._start_compare_camera_stream",
        lambda: {"status": "running", "message": "ok"},
    )
    app = create_app()

    response = app.test_client().post("/api/compare-camera/start")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload == {"status": "running", "message": "ok"}
    assert "job_id" not in payload


def test_web_capture_opencv_guide_toggle(monkeypatch) -> None:
    from oracle_report.web import _web_capture_show_opencv_guide

    monkeypatch.delenv("ORACLE_WEB_CAPTURE_SHOW_OPENCV_GUIDE", raising=False)

    assert _web_capture_show_opencv_guide() is False

    monkeypatch.setenv("ORACLE_WEB_CAPTURE_SHOW_OPENCV_GUIDE", "1")

    assert _web_capture_show_opencv_guide() is True


def test_capture_debug_payload_shows_only_judgement_item_and_tag() -> None:
    from oracle_report.models import CaptureDecision, FaceQuality
    from oracle_report.web import _capture_debug_payload

    quality = FaceQuality(
        ready=True,
        landmark_metrics_text="third_balance_error=0.123",
        landmark_context_text=(
            "- 항목: 삼정균형 | 판정: 삼정 편차 강조형 | 근거: 샘플 | 관찰: 샘플 | 측정값: 0.123"
        ),
        landmark_matches_json=json.dumps(
            [
                {
                    "title": "삼정균형",
                    "tag": "삼정 편차 강조형",
                    "basis": "샘플",
                    "observation": "샘플",
                    "value": 0.123,
                },
                {
                    "title": "눈 비율",
                    "tag": "눈매 강조형",
                    "basis": "샘플",
                    "observation": "샘플",
                    "value": 0.456,
                },
            ],
            ensure_ascii=False,
        ),
    )
    decision = CaptureDecision(
        state="tracking",
        elapsed_seconds=0.0,
        face=None,
        quality=quality,
        should_capture=False,
        message="측정 중",
    )

    payload = _capture_debug_payload(decision, live=True)

    assert payload["observations_text"] == (
        "항목: 삼정균형 | 판정: 삼정 편차 강조형\n"
        "항목: 눈 비율 | 판정: 눈매 강조형"
    )
    assert "근거" not in payload["observations_text"]
    assert "관찰" not in payload["observations_text"]
    assert "측정값" not in payload["observations_text"]


def test_compatibility_api_returns_result_url(monkeypatch) -> None:
    pytest.importorskip("flask")
    from oracle_report.web import create_app

    monkeypatch.setattr(
        "oracle_report.web._start_compatibility_workflow_job",
        lambda workflow_input: "compat-job",
    )
    app = create_app()

    response = app.test_client().post(
        "/api/compatibility",
        data={
            "left_name": "왼쪽",
            "left_birth_date": "1997-04-12",
            "left_birth_time": "",
            "left_gender": "여성",
            "right_name": "오른쪽",
            "right_birth_date": "1999-08-18",
            "right_birth_time": "",
            "right_gender": "남성",
            "mode": "직장동료",
            "face_analysis_mode": "2",
        },
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload == {
        "job_id": "compat-job",
        "result_url": "/compatibility/result/compat-job",
    }


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


def test_completed_job_sends_report_email(monkeypatch) -> None:
    pytest.importorskip("flask")
    from oracle_report.web import _WorkflowJob, _set_job, create_app

    sent_args: dict[str, tuple[str, str, str]] = {}

    def fake_send_report_email(recipient: str, filename: str, html: str) -> None:
        sent_args["email"] = (recipient, filename, html)

    monkeypatch.setattr(
        "oracle_report.web._send_report_email",
        fake_send_report_email,
    )
    app = create_app()
    _set_job(
        "email-job",
        _WorkflowJob(
            status="complete",
            html="<section>fragment</section>",
            download_html="<!doctype html><html><body>full report</body></html>",
            download_filename="personal_report.html",
        ),
    )

    response = app.test_client().post(
        "/api/jobs/email-job/email",
        json={"email": "reader@example.com"},
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["status"] == "sent"
    assert sent_args["email"] == (
        "reader@example.com",
        "personal_report.html",
        "<!doctype html><html><body>full report</body></html>",
    )


def test_completed_job_email_reports_smtp_authentication_error(monkeypatch) -> None:
    pytest.importorskip("flask")
    from oracle_report.web import _WorkflowJob, _set_job, create_app

    def fake_send_report_email(recipient: str, filename: str, html: str) -> None:
        del recipient, filename, html
        raise smtplib.SMTPAuthenticationError(535, b"Authentication failed")

    monkeypatch.setattr(
        "oracle_report.web._send_report_email",
        fake_send_report_email,
    )
    app = create_app()
    _set_job(
        "email-auth-error-job",
        _WorkflowJob(
            status="complete",
            html="<section>fragment</section>",
            download_html="<!doctype html><html><body>full report</body></html>",
            download_filename="personal_report.html",
        ),
    )

    response = app.test_client().post(
        "/api/jobs/email-auth-error-job/email",
        json={"email": "reader@example.com"},
    )
    payload = response.get_json()

    assert response.status_code == 500
    assert payload["status"] == "error"
    assert "SMTP 인증" in payload["message"]
    assert "ORACLE_SMTP_PASSWORD" in payload["message"]


def test_completed_job_email_reports_smtp_data_error(monkeypatch) -> None:
    pytest.importorskip("flask")
    from oracle_report.web import _WorkflowJob, _set_job, create_app

    def fake_send_report_email(recipient: str, filename: str, html: str) -> None:
        del recipient, filename, html
        raise smtplib.SMTPDataError(552, b"Message size exceeds fixed limit")

    monkeypatch.setattr(
        "oracle_report.web._send_report_email",
        fake_send_report_email,
    )
    app = create_app()
    _set_job(
        "email-data-error-job",
        _WorkflowJob(
            status="complete",
            html="<section>fragment</section>",
            download_html="<!doctype html><html><body>full report</body></html>",
            download_filename="personal_report.html",
        ),
    )

    response = app.test_client().post(
        "/api/jobs/email-data-error-job/email",
        json={"email": "reader@example.com"},
    )
    payload = response.get_json()

    assert response.status_code == 500
    assert payload["status"] == "error"
    assert "첨부 용량" in payload["message"]


def test_large_report_email_attachment_is_zipped(monkeypatch) -> None:
    pytest.importorskip("flask")
    from email.message import EmailMessage

    from oracle_report import web

    message = EmailMessage()
    monkeypatch.setattr(web, "MAIL_HTML_ATTACHMENT_MAX_BYTES", 16)

    web._attach_report_file(message, "personal_report.html", "<html>large report</html>")

    attachments = list(message.iter_attachments())

    assert len(attachments) == 1
    assert attachments[0].get_filename() == "personal_report.zip"
    assert attachments[0].get_content_type() == "application/zip"
    with zipfile.ZipFile(BytesIO(attachments[0].get_payload(decode=True))) as archive:
        assert archive.namelist() == ["personal_report.html"]
        assert archive.read("personal_report.html") == b"<html>large report</html>"


def test_email_report_attachment_removes_images() -> None:
    pytest.importorskip("flask")
    from email.message import EmailMessage

    from oracle_report import web

    message = EmailMessage()
    html = (
        '<html><body><h1>Report</h1>'
        '<img src="data:image/png;base64,AAAA" alt="">'
        '<p>Keep this text</p>'
        '<IMG src="/static/assets/oracle-character.png" alt="">'
        "</body></html>"
    )

    web._attach_report_file(message, "personal_report.html", html)

    attachments = list(message.iter_attachments())
    payload = attachments[0].get_payload(decode=True).decode("utf-8")

    assert len(attachments) == 1
    assert attachments[0].get_filename() == "personal_report.html"
    assert attachments[0].get_content_type() == "text/html"
    assert "<img" not in payload.lower()
    assert "data:image" not in payload
    assert "<h1>Report</h1>" in payload
    assert "<p>Keep this text</p>" in payload


def test_running_job_email_returns_not_found() -> None:
    pytest.importorskip("flask")
    from oracle_report.web import _WorkflowJob, _set_job, create_app

    app = create_app()
    _set_job("running-email-job", _WorkflowJob(status="running"))

    response = app.test_client().post(
        "/api/jobs/running-email-job/email",
        json={"email": "reader@example.com"},
    )
    payload = response.get_json()

    assert response.status_code == 404
    assert payload["status"] == "error"


def test_personal_page_uses_oracle_input_card_layout() -> None:
    pytest.importorskip("flask")
    from oracle_report.web import create_app

    app = create_app()

    response = app.test_client().get("/personal")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "oracle-input-shell" in html
    assert "personal-oracle-shell" in html
    assert "brand" in html
    assert "personal-brand" in html
    assert "input-card" in html
    assert "personal-card" in html
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
    assert "personal-field-list" in html
    assert "personal-field" in html
    assert "grid-template-columns: 1fr;" in html
