---
description: Oracle Report guidance for local face-capture, saju data, LLM prompts, report rendering, and Raspberry Pi deployment.
---

# Oracle Report Engineering Guidance

Focus areas:

- Python 3.10 package for Raspberry Pi face capture, local LLM report
  generation, saju/manse SQLite lookup, face recommendation, and Flask UI.
- Local-first privacy: camera frames, birth data, generated reports, and LLM
  calls stay on-device unless the task explicitly changes that policy.
- Deterministic tests should use fake cameras, fake LLM clients, and temporary
  SQLite databases instead of real hardware or live llama.cpp servers.
- User-facing Korean copy, prompt templates, and rendered HTML are product
  behavior. Preserve UTF-8 content and validate JSON-shaped LLM output paths.

## Layout

- `rules/`: always-on guidance for development, search, and verification.
- `agents/`: role prompts that can be used as checklists or sub-agent specs.
- `commands/`: repeatable audit and diagnostic workflows.
- `skills/`: Codex-style `SKILL.md` workflows for TDD, device/capture
  diagnostics, and report/prompt quality.

## Agent Model Routing

- `oracle-architect`: `gpt-5.5` for workflow, module boundary, data/privacy,
  and deployment design.
- `oracle-code-reviewer`: `codex-auto-review` for diff-focused review.
- `oracle-debugger`: `gpt-5.5` for cross-layer failures across camera, Flask,
  SQLite, prompts, and local LLM.
- `oracle-test-specialist`: `gpt-5.4` for focused tests with hardware/LLM
  isolation.

## Primary Workflows

- Use `rules/oracle-development.md` before editing application code.
- Use `rules/oracle-search-and-testing.md` when locating code or choosing tests.
- Use `commands/diagnose.md` when tests, app startup, DB lookup, camera capture,
  or LLM requests fail.
- Use `commands/audit-codebase.md` for broad project health audits.
- Use `skills/oracle-tdd/SKILL.md` for bug fixes and behavior changes.
- Use `skills/oracle-device-capture/SKILL.md` for camera, Raspberry Pi, and
  capture quality changes.
- Use `skills/oracle-report-quality/SKILL.md` for prompt templates, JSON report
  payloads, Korean copy, and rendered HTML changes.

All shell examples assume this repository is the working directory and use
`rtk` as the command proxy.
