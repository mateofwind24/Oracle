---
name: oracle-tdd
description: >
  Red-Green-Refactor workflow for Oracle Report behavior changes across Python
  workflows, Flask UI, prompt/report rendering, saju data, and capture logic.
  Use for bug fixes, validation changes, public behavior changes, and refactors
  where behavior must stay stable.
---

# Oracle TDD

## Trigger

Use this skill when changing behavior, fixing a bug, adding validation, touching
workflow contracts, or editing public prompt/report behavior.

## Cycle

1. Locate the narrowest test target.
2. Write a failing regression test first.
3. Run the narrow test and confirm it fails for the expected reason.
4. Implement the smallest fix.
5. Run the narrow test and confirm it passes.
6. Refactor only if needed.
7. Run broader tests that cover the module.

## Test Placement

- Workflow orchestration: `tests/test_workflow.py`.
- Flask routes/API state: `tests/test_web.py`.
- Prompt context or prompt CLI: `tests/test_report_prompt.py`,
  `tests/test_prompt_cli.py`.
- Report HTML and generated JSON fallback: `tests/test_report_html.py`.
- Config/env parsing: `tests/test_config.py`.
- LLM client: `tests/test_llm.py`.
- Capture state and vision behavior: `tests/test_capture_state.py`,
  `tests/test_camera.py`, `tests/test_detection.py`, `tests/test_landmark_rules.py`.
- Saju/manse DB: `tests/test_saju.py`, `tests/test_manse_repository.py`,
  `tests/test_packaged_manse_db.py`.

## Patterns

### Workflow Bug

- Arrange: create `tmp_path` DBs, fake capture artifacts, fake LLM clients.
- Act: run `run_personal_workflow` or `run_compatibility_workflow`.
- Assert: output path, fragment HTML, timing log, capture path, and key rendered
  content.

### Prompt Schema Bug

- Arrange: build profile and prompt input.
- Act: call `build_*_prompt` or CLI prompt command.
- Assert: required keys exist, unsafe claims are absent, and override behavior is
  preserved.

### HTML Fallback Bug

- Arrange: pass invalid JSON or partial JSON to report renderer.
- Act: render full and/or fragment HTML.
- Assert: user-visible fallback sections render and raw unsafe text is escaped.

### Capture State Bug

- Arrange: fake detector, fake analyzer, fake clock, small NumPy frame.
- Act: call `FaceCaptureHarness.observe` over a sequence.
- Assert: state, elapsed time, warnings, and `should_capture` transitions.

### Config Boundary Bug

- Arrange: set environment with `monkeypatch`.
- Act: call the relevant `load_*_config`.
- Assert: parsed values, validation errors, and localhost-only LLM checks.

## Commands

```powershell
rtk python -m pytest tests/test_workflow.py -q
rtk python -m pytest tests/test_report_prompt.py tests/test_report_html.py -q
rtk python -m pytest tests/test_capture_state.py tests/test_camera.py tests/test_detection.py -q
rtk python -m pytest
```

## Final Gate

State exactly which tests ran. If a camera, local LLM server, Raspberry Pi, or
systemd check was not run, report it as a verification gap.
