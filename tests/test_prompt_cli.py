from __future__ import annotations

from pathlib import Path

from oracle_report.cli import main
from oracle_report.saju.repository import build_manse_database


def test_prompt_command_prints_face_analysis_prompt(capsys) -> None:
    result = main(
        [
            "prompt",
            "personal-face-analysis",
            "--name",
            "tester",
            "--birth-date",
            "1995-03-15",
            "--birth-time",
            "14:30",
            "--gender",
            "male",
        ],
    )

    output = capsys.readouterr().out

    assert result == 0
    assert "1995-03-15 14:30:00" in output


def test_prompt_command_prints_compatibility_face_analysis_prompt(capsys) -> None:
    result = main(
        [
            "prompt",
            "compatibility-face-analysis",
            "--name",
            "tester",
            "--birth-date",
            "1995-03-15",
            "--birth-time",
            "14:30",
            "--gender",
            "male",
            "--mode",
            "친구",
            "--person-label",
            "첫 번째 사람",
        ],
    )

    output = capsys.readouterr().out

    assert result == 0
    assert "궁합 모드: 친구" in output
    assert "현재 분석 대상: 첫 번째 사람" in output


def test_prompt_command_prints_saju_reading(capsys, tmp_path: Path) -> None:
    manse_db_path = _build_test_manse_db(tmp_path)

    result = main(
        [
            "prompt",
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
    assert "1995-03-15" in output


def test_prompt_command_prints_personal_final_prompt(
    capsys,
    tmp_path: Path,
) -> None:
    manse_db_path = _build_test_manse_db(tmp_path)

    result = main(
        [
            "prompt",
            "personal-final",
            "--name",
            "tester",
            "--birth-date",
            "1995-03-15",
            "--birth-time",
            "모름",
            "--gender",
            "male",
            "--target-gender",
            "female",
            "--manse-db",
            str(manse_db_path),
            "--face-db",
            str(tmp_path / "faces.sqlite"),
            "--face-analysis",
            "face analysis fixture",
        ],
    )

    output = capsys.readouterr().out

    assert result == 0
    assert "1995-03-15" in output
    assert "미입력" in output
    assert "정오 기준" in output
    assert "face analysis fixture" in output


def test_prompt_command_prints_compatibility_final_prompt(
    capsys,
    tmp_path: Path,
) -> None:
    manse_db_path = _build_test_manse_db(tmp_path)

    result = main(
        [
            "prompt",
            "compatibility-final",
            "--name",
            "left",
            "--birth-date",
            "1995-03-15",
            "--birth-time",
            "14:30",
            "--gender",
            "male",
            "--right-name",
            "right",
            "--right-birth-date",
            "1995-03-16",
            "--right-birth-time",
            "모름",
            "--right-gender",
            "female",
            "--mode",
            "연인",
            "--manse-db",
            str(manse_db_path),
            "--face-analysis",
            "pair face analysis fixture",
        ],
    )

    output = capsys.readouterr().out

    assert result == 0
    assert "left" in output
    assert "right" in output
    assert "미입력" in output
    assert "정오 기준" in output
    assert "pair face analysis fixture" in output
    assert "\"pair_blocks\"" in output
    assert "\"action_title\"" in output


def _build_test_manse_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "manse.sqlite"
    build_manse_database(db_path, start_year=1995, end_year=1995)
    result = db_path
    return result
