---
description: Efficient navigation, investigation, and verification rules for Oracle Report.
---

# Oracle Search And Testing

## Search Priority

1. Prefer `rg` when available.
2. If `rg` is unavailable, use `git grep` for tracked code.
3. Use `git ls-files` for source discovery.
4. Read whole files only after locating relevant symbols.
5. Parallelize independent reads and searches when possible.

All examples assume the repository root as the working directory.

```powershell
rtk rg -n "run_personal_workflow|run_compatibility_workflow" src tests
rtk rg -n "ORACLE_|LlmConfig|CaptureConfig" src configs tests
rtk rg -n "personal_final|compatibility_final|face_blocks|pair_blocks" src configs tests
rtk git ls-files src tests configs docs scripts systemd
```

## Module Map

- `src/oracle_report/models.py`: shared dataclasses for captures, profiles, and
  report artifacts.
- `src/oracle_report/config.py`: environment parsing, capture settings, local LLM
  URL validation, Flask app config.
- `src/oracle_report/workflow.py`: end-to-end personal and compatibility
  orchestration.
- `src/oracle_report/web.py`: Flask UI, API jobs, preview stream, capture lock.
- `src/oracle_report/llm.py`: OpenAI-compatible llama.cpp chat client and image
  payload encoding.
- `src/oracle_report/report.py`: prompt value assembly.
- `src/oracle_report/prompt_templates.py`: prompt template loading and rendering.
- `configs/prompts.json`: product prompt text and JSON output schema.
- `src/oracle_report/report_html.py`: generated JSON parsing, fallback payloads,
  and final HTML rendering.
- `src/oracle_report/recommender.py`: face recommendation DB and formatting.
- `src/oracle_report/saju/`: calendar, saju engine, manse SQLite repository, CLI.
- `src/oracle_report/vision/`: camera open, face detection, quality analysis,
  capture state, runtime capture loop.
- `scripts/`: Raspberry Pi install, llama server, capture run, DB build helpers.
- `systemd/`: deployable service units.

## Investigation Patterns

### Trace an end-to-end personal report

1. `web.py` form/API input.
2. `PersonalWorkflowInput` and `_build_birth_profile`.
3. `ManseRepository.lookup`.
4. `run_capture` or injected fake capture runner.
5. face analysis prompt or landmark rule mode.
6. recommendation lookup.
7. final prompt and `report_html.py` rendering.
8. `tests/test_workflow.py`, `tests/test_web.py`, prompt and HTML tests.

### Trace a prompt/report schema change

1. `configs/prompts.json`.
2. `report.py` template values.
3. `report_html.py` payload parsing and fallback defaults.
4. `tests/test_report_prompt.py`.
5. `tests/test_report_html.py`.
6. CLI prompt tests if the change is exposed through `run.sh prompt`.

### Trace a capture issue

1. `config.py` capture defaults and env variables.
2. `vision/camera.py` camera open and processor construction.
3. `vision/capture.py` `FaceCaptureHarness.observe`.
4. `vision/quality.py` and `vision/detection.py`.
5. `vision/runtime.py` loop, preview, and artifact saving.
6. `tests/test_capture_state.py`, `tests/test_camera.py`, `tests/test_detection.py`.

### Trace a manse/saju issue

1. `saju/calendar.py` stem/branch conventions.
2. `saju/engine.py` reading and element counts.
3. `saju/repository.py` SQLite schema, generation, and lookup.
4. `scripts/build_manse_db.py` and `data/manse.sqlite` expectations.
5. `tests/test_saju.py`, `tests/test_manse_repository.py`,
   `tests/test_packaged_manse_db.py`.

## Test Selection

- Workflow orchestration: `rtk python -m pytest tests/test_workflow.py -q`.
- Flask route/API behavior: `rtk python -m pytest tests/test_web.py -q`.
- Prompt templates or CLI prompt output:
  `rtk python -m pytest tests/test_report_prompt.py tests/test_prompt_cli.py -q`.
- HTML report rendering:
  `rtk python -m pytest tests/test_report_html.py -q`.
- Capture state/camera/detection:
  `rtk python -m pytest tests/test_capture_state.py tests/test_camera.py tests/test_detection.py -q`.
- Config/env parsing: `rtk python -m pytest tests/test_config.py -q`.
- Saju/manse data: `rtk python -m pytest tests/test_saju.py tests/test_manse_repository.py tests/test_packaged_manse_db.py -q`.
- Full local gate: `rtk python -m pytest`.

## Verification Report

When finishing a task, state:

- Files changed.
- Behavior changed.
- Tests run and exact result.
- Tests not run and why.
- Any hardware, local LLM, or Raspberry Pi verification gap.
