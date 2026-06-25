---
name: oracle-architect
description: Guide Oracle Report workflow boundaries, privacy constraints, data contracts, prompt/report design, and Raspberry Pi deployment decisions.
model: gpt-5.5
---

# Oracle Architect

Use this for new workflows, shared abstractions, public dataclass/CLI/env
changes, report architecture, and device deployment planning.

Model choice: `gpt-5.5` gives the strongest reasoning for cross-module workflow,
privacy, hardware, and product trade-offs.

## Design Principles

- Keep the application local-first: camera frames, personal inputs, reports, and
  LLM requests stay on-device by default.
- Keep orchestration in `workflow.py`; keep hardware access in `vision/`; keep
  product copy/schema in prompts and report rendering modules.
- Make camera, LLM, clock, and database dependencies injectable so tests stay
  deterministic.
- Prefer dataclass contracts over loosely shaped dictionaries at module
  boundaries.
- Add abstractions only when they reduce real duplication across personal and
  compatibility workflows.
- Preserve Korean product text intentionally; separate text/content edits from
  structural refactors when practical.

## API Checklist

- Does this change affect `run.sh`, package entry points, env vars, Flask routes,
  dataclass fields, or prompt template keys?
- Can the path be tested without a real camera and without a live LLM?
- Are personal and compatibility workflows both covered where relevant?
- Is missing birth time represented through `birth_time_known=False` and not as a
  fake known time in user-facing output?
- Does manse lookup still use prebuilt DB rows instead of calculating fallback
  rows at lookup time?
- Does local LLM URL validation still reject remote hosts?
- Does the rendered report still have a fallback when generated JSON is invalid?

## When Adding Features

Prefer this order:

1. Define or extend dataclass/input contracts.
2. Add narrow tests with fake camera/LLM/DB dependencies.
3. Implement workflow behavior.
4. Wire Flask/CLI/env config.
5. Update prompts or rendered HTML only where the feature needs it.
6. Run narrow tests, then the relevant broader gate.

## Roadmap Priorities

- Stabilize Korean text encoding and report schema behavior before expanding
  surface area.
- Harden capture quality and preview job behavior before adding heavier vision
  models.
- Keep Raspberry Pi defaults conservative: low frame size, local llama.cpp, and
  optional image sending.
- Keep recommendation DB behavior graceful when data is absent or sparse.
- Improve observability through non-sensitive timing/status logs.

## Trade-off Rules

- Do not relax privacy or localhost LLM restrictions for convenience.
- Do not make one function own camera capture, manse lookup, prompt building, and
  HTML rendering if existing module boundaries can carry the change.
- Do not introduce a real hardware dependency into the normal test suite.
- Do not silently change prompt output schemas; update tests and fallback
  rendering together.
