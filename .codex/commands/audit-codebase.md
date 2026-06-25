---
description: Broad Oracle Report health audit workflow with prioritized findings and TODO output.
argument-hint: "[--category privacy|workflow|prompts|reports|vision|saju|web|tests|deploy] [--fix] [--json]"
---

# audit-codebase

Run this workflow to audit Oracle Report and produce prioritized findings or a
fix plan.

## Categories

- `privacy`: local-only LLM, personal data logging, capture/report artifacts.
- `workflow`: personal/compatibility orchestration, fallback paths, timing logs.
- `prompts`: prompt templates, JSON schema keys, unsafe claims, overrides.
- `reports`: HTML rendering, parser fallbacks, escaping, responsive layout.
- `vision`: camera open, detection, capture state, quality checks, preview.
- `saju`: manse DB generation/lookup, time branch, gender normalization, rules.
- `web`: Flask routes, async jobs, capture lock, API errors.
- `tests`: coverage, fake dependency isolation, hardware-free normal gate.
- `deploy`: Raspberry Pi config, scripts, systemd units, local model paths.

## Workflow

1. Confirm repository state.

```powershell
rtk git status --short
rtk git ls-files src tests configs docs scripts systemd pyproject.toml README.md
```

2. Scan privacy and local-only constraints.

```powershell
rtk rg -n "http://|https://|requests\.post|base64|image_path|birth|prompt|print|logging|ORACLE_.*LLM_BASE_URL" src tests configs scripts systemd .env.example
rtk rg -n "allowed_hosts|localhost|127\.0\.0\.1|::1" src\oracle_report\config.py tests
```

3. Scan workflow and concurrency risks.

```powershell
rtk rg -n "ThreadPoolExecutor|_CAPTURE_LOCK|_JOBS|except Exception|timing|run_personal_workflow|run_compatibility_workflow" src tests
```

4. Scan prompt and report schema drift.

```powershell
rtk rg -n "personal_face_analysis|compatibility_face_analysis|personal_final|compatibility_final|face_blocks|saju_blocks|pair_blocks|action_title|convergence" src configs tests
```

5. Scan camera and device behavior.

```powershell
rtk rg -n "cv2|imshow|waitKey|VideoCapture|FaceCaptureHarness|FaceQuality|face_analysis_mode|mediapipe" src tests configs scripts
```

6. Scan SQLite and saju data behavior.

```powershell
rtk rg -n "sqlite|manse_entries|manse_metadata|build_manse_database|normalize_gender|time_branch" src tests scripts data
```

7. Run available tests.

```powershell
rtk python -m pytest
```

## Scoring

Use 0-10 per category:

- 0-4: critical, unsafe or unusable for real users.
- 5-7: usable but missing hardening, tests, or deployment clarity.
- 8-10: production-ready for the audited scope.

Weight privacy, workflow, prompts/reports, and vision twice as much as docs.

## Output

```text
Oracle audit YYYY-MM-DD

Category scores:
- privacy: N/10
- workflow: N/10
- prompts: N/10
- reports: N/10
- vision: N/10
- saju: N/10
- web: N/10
- tests: N/10
- deploy: N/10

P0:
- path:line: issue. Fix.

P1:
- ...

Verification:
- command: result
```

If `--fix` is requested, propose the smallest safe implementation sequence.
