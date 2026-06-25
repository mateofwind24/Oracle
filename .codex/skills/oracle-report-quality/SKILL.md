---
name: oracle-report-quality
description: >
  Prompt template, generated JSON schema, Korean copy, report HTML, disclaimer,
  and local LLM output-quality workflow for Oracle Report. Use before changing
  configs/prompts.json, report.py, prompt_templates.py, report_html.py, or
  user-visible report copy.
---

# Oracle Report Quality

## Focus

Oracle reports are user-facing entertainment/reference content generated from
structured saju data, face observation memos, and local LLM responses. Quality
changes must preserve safety constraints, JSON schema compatibility, Korean text,
and renderer fallback behavior.

## Safety Rules

- Do not infer identity, age, health, wealth, job, legal status, or other
  sensitive traits from face images.
- Keep report language as non-deterministic entertainment/reference guidance.
- Keep disclaimers visible through generated or fallback payloads.
- Do not expose prompt rules, template instructions, or internal schema notes in
  rendered user text.
- Keep prompt output as a single JSON object for final report prompts.

## Schema Contracts

Personal final output expects:

- `essence`
- `element_note`
- `face_subtitle`
- `face_blocks`
- `saju_subtitle`
- `saju_blocks`
- `synthesis_title`
- `synthesis_body`
- `convergence`
- `synthesis_summary`
- `tags`
- `recommendation_title`
- `recommendation_lead`
- `disclaimer`

Compatibility final output expects:

- `essence`
- `pair_subtitle`
- `pair_blocks`
- `saju_subtitle`
- `saju_blocks`
- `synthesis_title`
- `synthesis_body`
- `convergence`
- `action_title`
- `action_body`
- `tags`
- `disclaimer`

## Workflow

1. Identify whether the change is prompt context, template text, parser fallback,
   HTML layout, or generated content safety.
2. Read `configs/prompts.json`, `report.py`, `prompt_templates.py`, and the
   relevant part of `report_html.py`.
3. Add/update prompt or HTML tests before editing behavior.
4. Keep schema keys stable unless the renderer and tests change together.
5. Validate invalid/partial JSON fallback behavior.
6. Run prompt, report HTML, workflow, and CLI prompt tests as relevant.

## Searches

```powershell
rtk rg -n "personal_final|compatibility_final|personal_face_analysis|compatibility_face_analysis" configs src tests
rtk rg -n "face_blocks|saju_blocks|pair_blocks|convergence|disclaimer|action_title|recommendation" src configs tests
rtk rg -n "identity|age|health|wealth|job|legal|remote|prompt rules|schema" configs src tests
```

## Test Commands

```powershell
rtk python -m pytest tests/test_report_prompt.py -q
rtk python -m pytest tests/test_report_html.py -q
rtk python -m pytest tests/test_prompt_cli.py -q
rtk python -m pytest tests/test_workflow.py -q
```

Manual prompt checks when useful:

```powershell
rtk bash ./run.sh prompt personal-face-analysis --name "tester" --birth-date 1995-03-15 --birth-time 14:30 --gender male
rtk bash ./run.sh prompt personal-final --name "tester" --birth-date 1995-03-15 --birth-time 14:30 --gender male --target-gender female --face-analysis "face memo" --recommendation-text "recommendation memo"
rtk bash ./run.sh prompt compatibility-final --name "left" --birth-date 1995-03-15 --birth-time 14:30 --gender male --right-name "right" --right-birth-date 1997-05-20 --right-birth-time 09:00 --right-gender female --mode "?곗씤" --face-analysis "pair memo"
```

## Acceptance Criteria

- Final prompt output schema remains parseable by `report_html.py`.
- Invalid JSON and partial JSON still render useful fallback report sections.
- Tests cover required keys and at least one rendered visible section.
- Safety constraints remain explicit in prompts and visible disclaimers.
- Verification states whether a live local LLM run was skipped or completed.
