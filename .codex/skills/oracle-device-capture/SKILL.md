---
name: oracle-device-capture
description: >
  Raspberry Pi, OpenCV camera, face detection, capture quality, preview stream,
  and systemd deployment workflow for Oracle Report. Use for device-facing
  changes, capture hangs, camera config, performance on Pi, and preview behavior.
---

# Oracle Device Capture

## Focus

Oracle Report runs on Raspberry Pi-like hardware with a local camera and
optional local LLM service. Device changes must be conservative, testable without
hardware where possible, and explicit about manual verification gaps.

## Targets

- Keep capture state deterministic: exactly one usable face, quality ready, stable
  for `min_face_seconds`.
- Keep normal tests hardware-free by using fake detector/analyzer/camera paths.
- Keep Raspberry Pi defaults bounded: reduced detection scale, detection
  interval, modest frame size, optional preview.
- Do not open GUI windows in web/API or automated test paths.
- Avoid expensive per-frame work outside configured detection/quality cadence.

## Workflow

1. Identify whether the change is capture state, camera open, quality analysis,
   preview streaming, Flask job coordination, or deployment config.
2. Add or run the narrow fake-dependency test first.
3. Inspect config defaults and environment variable names.
4. Implement the smallest change in the owning module.
5. Run capture/vision tests.
6. If the change affects real hardware, document the exact manual command and
   whether it was run.

## Module Boundaries

- `vision/capture.py`: state machine, stable face timing, save artifact.
- `vision/camera.py`: OpenCV import/open and processor construction.
- `vision/detection.py`: face detector behavior.
- `vision/quality.py`: quality warnings and readiness.
- `vision/runtime.py`: read loop, overlay, preview window, callback, cleanup.
- `web.py`: preview stream, async job state, capture lock.
- `configs/raspberry_pi.env`, `scripts/`, `systemd/`: deploy/runtime defaults.

## Searches

```powershell
rtk rg -n "FaceCaptureHarness|should_capture|warning|captured|min_face_seconds|face_min_size" src\oracle_report\vision tests
rtk rg -n "VideoCapture|imshow|waitKey|destroyAllWindows|frame_callback|preview" src tests configs scripts
rtk rg -n "ORACLE_CAMERA|ORACLE_FRAME|ORACLE_FACE|ORACLE_SHOW_PREVIEW" src configs .env.example scripts
```

## Anti-Patterns

- Adding real camera or GUI dependencies to normal tests.
- Starting multiple capture jobs concurrently.
- Doing LLM/image analysis on every frame.
- Swallowing camera read/write failures without a clear error.
- Changing capture thresholds without updating tests or deployment docs.
- Writing captures outside configured output/session directories.

## Verification

Run relevant automated tests:

```powershell
rtk python -m pytest tests/test_capture_state.py tests/test_camera.py tests/test_detection.py tests/test_landmark_rules.py -q
rtk python -m pytest tests/test_web.py tests/test_workflow.py -q
```

Manual checks when hardware is available:

```powershell
rtk bash ./run.sh debug capture
rtk bash ./run.sh debug capture --face-analysis-mode 2
rtk bash ./run.sh
```

Always report whether manual camera/Raspberry Pi checks were run or skipped.
