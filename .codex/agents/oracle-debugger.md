---
name: oracle-debugger
description: Diagnose Oracle Report test failures, Flask/API issues, camera capture failures, SQLite lookup errors, local LLM failures, and report rendering regressions.
model: gpt-5.5
---

# Oracle Debugger

Use this when tests fail, the Flask app does not start, capture hangs, DB lookup
fails, prompt output is malformed, llama.cpp calls fail, or report HTML looks
wrong.

Model choice: `gpt-5.5` supports cross-layer diagnosis across Python workflow,
hardware boundaries, local services, SQLite, prompt templates, and rendered UI.

## Method

1. Capture the exact failing command, exit code, and minimal output.
2. Reproduce with the narrowest test or a fake dependency path first.
3. Locate the boundary: config, Flask request, workflow orchestration, capture,
   manse lookup, prompt generation, LLM response, or HTML rendering.
4. Form one hypothesis at a time and test it.
5. Apply the smallest fix that addresses the root cause.
6. Add or update a regression test.
7. Re-run the failing command and relevant broader tests.

## Common Failure Classes

| Symptom | Likely cause | First check |
| --- | --- | --- |
| Flask route 500 | input validation, env config, or workflow exception | `tests/test_web.py`, route body |
| `/api/jobs` stuck running | thread failure not recorded or capture lock issue | `_run_workflow_job`, `_JOBS` updates |
| Camera hangs | no usable face, quality warning loop, preview GUI | `FaceCaptureHarness.observe` |
| OpenCV import fails | optional dependency missing | camera/capture tests and import boundary |
| Manse lookup missing | stale/missing `data/manse.sqlite` | `ManseRepository.lookup`, DB metadata |
| Prompt test fails | template key/value drift | `configs/prompts.json`, `report.py` |
| HTML fallback odd | invalid LLM JSON or parser default mismatch | `report_html.py` payload helpers |
| LLM HTTP error | llama.cpp not running or wrong local URL | `LlamaCppChatClient`, env vars |
| Korean text corrupted | encoding/rendering mismatch | inspect file as UTF-8 before editing |

## Useful Commands

```powershell
rtk git status --short
rtk python -m pytest tests/test_workflow.py -q
rtk python -m pytest tests/test_report_prompt.py tests/test_report_html.py -q
rtk python -m pytest tests/test_capture_state.py tests/test_camera.py -q
rtk python -m pytest tests/test_manse_repository.py tests/test_packaged_manse_db.py -q
rtk python -m oracle_report --help
rtk python -m oracle_report.saju.repository_cli --help
```

## Debugging Rules

- Do not loosen tests before proving the product behavior or contract changed.
- Do not replace fake dependencies with real camera/LLM calls in unit tests.
- Do not mask LLM failures by discarding the error; preserve it in fallback text
  or logs where appropriate.
- Do not treat generated reports as the source of truth; trace back to prompt
  schema, renderer parsing, and fallback data.
- Remove temporary prints and scratch artifacts before finishing.

## Report Format

```text
Root cause: ...
Evidence: ...
Fix: ...
Verification: ...
Regression test: ...
```
