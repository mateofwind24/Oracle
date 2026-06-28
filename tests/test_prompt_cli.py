from __future__ import annotations

from pathlib import Path

from oracle_report.cli import main





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
    assert "시간 미상" in output
    assert "오시(午時) 보조 기준" in output
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
    assert "시간 미상" in output
    assert "오시(午時) 보조 기준" in output
    assert "pair face analysis fixture" in output
    assert "\"pair_blocks\"" in output
    assert "\"action_title\"" in output


def test_token_command_prints_prompt_prefix_sizes(capsys) -> None:
    result = main(["token", "--offline"])

    output = capsys.readouterr().out
    lines = output.splitlines()
    header = next(line for line in lines if line.startswith("name "))
    personal_row = next(line for line in lines if line.startswith("personal_face_analysis "))
    saju_row = next(line for line in lines if line.startswith("saju_reading "))

    assert result == 0
    assert "source=estimated" in output
    assert "name" in header
    assert "id_slot" in header
    assert "prefix_tokens" in header
    assert header.index("id_slot") == personal_row.index("0")
    assert header.index("id_slot") == saju_row.index("1")
    assert "face_analysis_copule" in output
    assert "saju_reading_couple" in output


def _build_test_manse_db(tmp_path: Path) -> Path:
    result = tmp_path / "unused-manse.sqlite"
    return result
