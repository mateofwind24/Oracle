from __future__ import annotations

import json
from pathlib import Path

from oracle_report.cli import main
from oracle_report.saju.repository import build_manse_database


class FakeLlamaClient:
    prompt: str = ""
    image_path: Path | None = None

    def __init__(self, config) -> None:
        del config

    def generate(self, prompt: str, image_path: Path | None = None) -> str:
        FakeLlamaClient.prompt = prompt
        FakeLlamaClient.image_path = image_path
        result = "LLM RESULT ONLY"
        return result


def test_llm_command_runs_saju_reading_prompt_from_config(
    capsys,
    monkeypatch,
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(
        json.dumps(
            {
                "saju_reading": (
                    "CUSTOM ${name}\n"
                    "${birth_datetime}\n"
                    "${birth_time_text}\n"
                    "${saju_text}"
                ),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manse_db_path = _build_test_manse_db(tmp_path)
    monkeypatch.setenv("ORACLE_PROMPTS_PATH", str(prompt_path))
    monkeypatch.setattr("oracle_report.cli.LlamaCppChatClient", FakeLlamaClient)

    result = main(
        [
            "llm",
            "saju-reading",
            "--name",
            "tester",
            "--birth-date",
            "1995-03-15",
            "--birth-time",
            "모름",
            "--gender",
            "male",
            "--manse-db",
            str(manse_db_path),
        ],
    )

    output = capsys.readouterr().out

    assert result == 0
    assert output == "LLM RESULT ONLY\n"
    assert "CUSTOM tester" in FakeLlamaClient.prompt
    assert "1995-03-15 12:00:00" in FakeLlamaClient.prompt
    assert "미입력" in FakeLlamaClient.prompt
    assert "[만세력/사주명식]" in FakeLlamaClient.prompt
    assert FakeLlamaClient.image_path is None


def _build_test_manse_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "manse.sqlite"
    build_manse_database(db_path, start_year=1995, end_year=1995)
    result = db_path
    return result
