from __future__ import annotations

import pytest

from oracle_report.prompt_templates import render_distributed_prompt_template
from oracle_report.workflow import DistributedTaskScheduler


def test_distributed_prompt_split_metadata() -> None:
    values = {
        "name": "홍길동",
        "gender": "남성",
        "birth_datetime": "1999-10-20T10:25:00",
        "birth_time_text": "오시(午時)",
        "timezone": "KST",
        "saju_text": "사주 텍스트",
    }

    rendered = render_distributed_prompt_template(
        name="saju_reading",
        values=values,
        is_metadata=True,
    )

    assert "SAJU_SUBTITLE" in rendered.body
    assert '"saju_blocks":' not in rendered.prefix
    assert "saju_reading_split" in rendered.name


def test_distributed_prompt_split_category() -> None:
    values = {
        "name": "홍길동",
        "gender": "남성",
        "birth_datetime": "1999-10-20T10:25:00",
        "birth_time_text": "오시(午時)",
        "timezone": "KST",
        "saju_text": "사주 텍스트",
    }

    rendered = render_distributed_prompt_template(
        name="saju_reading",
        values=values,
        target_category="종합 형국",
    )

    assert "종합 형국" in rendered.body
    assert "CATEGORY" in rendered.body
    assert "BODY" in rendered.body
    assert '"saju_blocks":' not in rendered.prefix
    assert '"saju_subtitle":' not in rendered.prefix
    assert "CATEGORY:" in rendered.body


def test_distributed_prompt_saju_split() -> None:
    values = {
        "name": "홍길동",
        "gender": "남성",
        "birth_datetime": "1999-10-20T10:25:00",
        "birth_time_text": "오시(午時)",
        "timezone": "KST",
        "saju_text": "사주 테스트 텍스트",
    }

    rendered = render_distributed_prompt_template(
        name="saju_reading",
        values=values,
        is_metadata=True,
    )
    assert "ESSENCE" in rendered.body
    assert "SAJU_SUBTITLE" in rendered.body


def test_distributed_task_scheduler_round_robin() -> None:
    slaves = ["http://192.168.0.10:8501", "http://192.168.0.11:8501"]
    scheduler = DistributedTaskScheduler(slaves)

    assert scheduler.select_slave("task1") == "http://192.168.0.10:8501"
    assert scheduler.select_slave("task2") == "http://192.168.0.11:8501"
    assert scheduler.select_slave("task3") == "http://192.168.0.10:8501"
