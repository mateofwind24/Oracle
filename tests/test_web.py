from __future__ import annotations

import pytest


def test_favicon_returns_no_content() -> None:
    pytest.importorskip("flask")
    from oracle_report.web import create_app

    app = create_app()

    response = app.test_client().get("/favicon.ico")

    assert response.status_code == 204


def test_personal_page_includes_visible_workflow_loading_state() -> None:
    pytest.importorskip("flask")
    from oracle_report.web import create_app

    app = create_app()

    response = app.test_client().get("/personal")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'id="workflow-loading"' in html
    assert 'role="status"' in html
    assert "사주 리포트 생성 중입니다" in html
    assert "loading.hidden = false" in html


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


def test_personal_page_prevents_input_overflow_and_uses_wide_compact_layout() -> None:
    pytest.importorskip("flask")
    from oracle_report.web import create_app

    app = create_app()

    response = app.test_client().get("/personal")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "* {" in html
    assert "box-sizing: border-box;" in html
    assert "width: min(860px, calc(100vw - 48px));" in html
    assert 'class="field-grid"' in html
    assert "grid-template-columns: repeat(2, minmax(0, 1fr));" in html
