---
description: Diagnose Oracle Report repository state, test failures, config, SQLite data, Flask app, camera boundary, and local LLM setup.
argument-hint: "[--quick] [--tests] [--db] [--web] [--camera] [--llm]"
---

# diagnose

Use this when Oracle Report build, tests, app startup, DB lookup, camera capture,
or local LLM behavior looks wrong.

## Quick Checks

```powershell
rtk git status --short
rtk python --version
rtk python -m pytest -q
rtk python -m oracle_report --help
```

## Environment And Config Checks

```powershell
rtk powershell -NoProfile -Command "if (Test-Path .env) { Get-Content .env } else { Write-Output 'NO_ENV' }"
rtk powershell -NoProfile -Command "Get-Content .env.example"
rtk powershell -NoProfile -Command "Get-Content configs\raspberry_pi.env"
rtk python -m pytest tests/test_config.py -q
```

Confirm:

- LLM base URLs are local only.
- Camera defaults are conservative for Raspberry Pi.
- `.env` does not contain secrets that should be committed.

## Test-Focused Checks

```powershell
rtk python -m pytest tests/test_workflow.py -q
rtk python -m pytest tests/test_report_prompt.py tests/test_report_html.py -q
rtk python -m pytest tests/test_capture_state.py tests/test_camera.py tests/test_detection.py -q
rtk python -m pytest tests/test_saju.py tests/test_manse_repository.py tests/test_packaged_manse_db.py -q
rtk python -m pytest tests/test_llm.py -q
```

## SQLite Data Checks

```powershell
rtk powershell -NoProfile -Command "Test-Path data\manse.sqlite"
rtk powershell -NoProfile -Command "Test-Path data\physiognomy_rules.sqlite"
rtk python -m oracle_report.saju.repository_cli --help
rtk python scripts\build_manse_db.py --help
```

If manse lookup fails, inspect:

```powershell
rtk rg -n "schema_version|expected_rows|manse_entries|manse_metadata" src tests scripts
rtk python -m pytest tests/test_manse_repository.py -q
```

## Flask Checks

```powershell
rtk python -m pytest tests/test_web.py -q
rtk rg -n "create_app|/api/jobs|_CAPTURE_LOCK|video-feed|health" src\oracle_report\web.py tests
```

If the app starts but jobs hang, inspect `_run_workflow_job`, `_set_job`, and
capture runner injection before testing real camera hardware.

## Camera Boundary Checks

```powershell
rtk python -m pytest tests/test_capture_state.py tests/test_camera.py tests/test_detection.py -q
rtk rg -n "open_camera|FaceCaptureHarness|show_preview|frame_callback|save_capture_artifact" src\oracle_report\vision tests
```

Manual camera checks belong outside the normal test suite:

```powershell
rtk bash ./run.sh debug capture
rtk bash ./run.sh debug capture --face-analysis-mode 2
```

## Local LLM Checks

```powershell
rtk python -m pytest tests/test_llm.py -q
rtk rg -n "LlamaCppChatClient|ORACLE_LLM_BASE_URL|send_image|chat/completions" src tests configs .env.example
```

Manual llama.cpp checks:

```powershell
rtk bash ./run.sh prompt personal-final --name "tester" --birth-date 1995-03-15 --birth-time 14:30 --gender male --target-gender female --face-analysis "face memo" --recommendation-text "recommendation memo"
```

## Output

Report:

- Git state.
- Python/test result.
- Config or env issue found.
- SQLite data present or missing.
- Flask route/API status.
- Camera/LLM checks run or skipped.
- First concrete next step.
