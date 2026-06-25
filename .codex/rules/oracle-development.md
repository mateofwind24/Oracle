---
description: Always-on Oracle Report development rules for Python workflows, Flask UI, local LLM, camera capture, and SQLite data.
---

# Oracle Development Rules

## Core Constraints

- Follow the repository's existing explicit Python style: type hints on public
  and non-trivial internal functions, dataclasses for data contracts, and one
  final return statement for every non-void function.
- Keep changes scoped to the workflow, UI, prompt, data, or device path requested.
- Preserve public CLI, environment variable, dataclass, and Flask route behavior
  unless the task explicitly requires a breaking change.
- Prefer dependency injection for camera, LLM, clock, and database boundaries so
  tests can run without real hardware or services.
- Do not add dependencies unless the existing standard library or current
  project dependencies cannot reasonably solve the problem.
- Keep generated outputs, capture images, reports, local model files, and bootcamp
  archive content out of code changes unless the task explicitly concerns them.

## Local-First Privacy

- Keep `LlmConfig.base_url` restricted to `localhost`, `127.0.0.1`, or `::1`.
- Do not send camera frames, birth data, generated reports, or prompt payloads to
  remote services without an explicit user request and a clear code path.
- Avoid logging names, birth dates, raw prompt bodies, base64 image data, or
  capture file paths in routine logs.
- Timing logs may record phase names and durations, not personal input values.
- Treat face analysis as entertainment/reference content, not identity, health,
  age, gender, wealth, employment, or legal inference.

## Encoding And User-Facing Text

- Preserve UTF-8 Korean strings in prompts, UI labels, tests, and report HTML.
- If terminal output shows mojibake, inspect with an editor or encoding-aware
  tool before modifying text.
- Do not "fix" Korean text opportunistically while making unrelated changes.
- Keep prompt template keys stable: `personal_face_analysis`,
  `compatibility_face_analysis`, `personal_final`, and `compatibility_final`.
- JSON-shaped LLM output must remain parseable and should degrade through the
  existing fallback rendering path.

## Workflow Boundaries

- `workflow.py` orchestrates capture, manse lookup, face analysis, recommendation,
  final report generation, HTML rendering, saving, and timing.
- `vision/` owns camera access, detection, quality analysis, capture state, and
  image artifact writing.
- `saju/` owns calendar/manse generation, lookup, gender normalization, time
  branch mapping, and saju reading text.
- `report.py`, `prompt_templates.py`, and `configs/prompts.json` own prompt
  assembly.
- `report_html.py` owns structured report parsing and HTML presentation.
- `web.py` owns Flask routes, async job state, preview streaming, and request
  form/API handling.

## Error Handling

- Validate user input at Flask, CLI, config, and repository boundaries.
- Keep ordinary flow explicit. Do not use exceptions for expected branching.
- Catch broad exceptions only at user-facing boundaries or fallback boundaries,
  and preserve useful diagnostics in the error message.
- For LLM failures, prefer existing fallback report behavior over crashing after
  capture and manse lookup have already completed.
- For missing or stale SQLite data, fail clearly and point to the DB build path;
  do not silently calculate replacement manse rows during lookup.

## Tests

- Add focused tests whenever behavior changes.
- Use fake LLM clients, fake capture runners, fake clocks, and temp SQLite DBs.
- Avoid tests that require a real camera, OpenCV GUI, systemd, downloaded GGUF
  models, or a live llama.cpp server unless explicitly marked/manual.
- For workflow changes, cover personal and compatibility paths when both are
  affected.
- For prompt or HTML changes, assert required schema keys and visible rendered
  text, not only that the function returns a string.

## Common Commands

```powershell
rtk python -m pytest
rtk python -m pytest tests/test_workflow.py -q
rtk python -m pytest tests/test_report_prompt.py tests/test_report_html.py -q
rtk python -m pytest tests/test_capture_state.py tests/test_camera.py -q
rtk bash ./run.sh prompt personal-final --name "tester" --birth-date 1995-03-15 --birth-time 14:30 --gender male --target-gender female --face-analysis "face memo" --recommendation-text "recommendation memo"
```
