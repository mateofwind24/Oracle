from __future__ import annotations

from dataclasses import dataclass

from oracle_report.models import FaceBox


_CENTER_REGION_RATIO = 1.0 / 3.0
_HEAD_WIDTH_MIN_DIMENSION_RATIO = 0.34
_HEAD_HEIGHT_TO_WIDTH_RATIO = 1.25
_HEAD_TOP_FRAME_RATIO = 0.16
_HEAD_CENTER_TOLERANCE_RATIO = 0.26
_HEAD_SIZE_TOLERANCE_RATIO = 0.32
_SHOULDER_WIDTH_TO_HEAD_RATIO = 2.35
_SHOULDER_BOTTOM_FRAME_RATIO = 0.86
_CENTER_WARNING = "얼굴 중심을 화면 중앙 1/3 영역 안에 맞춰 주세요."
_GUIDE_POSITION_WARNING = "머리를 화면의 머리 가이드 박스에 맞춰 주세요."
_TOO_FAR_WARNING = "카메라에 조금 더 가까이 와서 머리 크기를 가이드에 맞춰 주세요."
_TOO_CLOSE_WARNING = "카메라에서 조금 뒤로 물러나 머리 크기를 가이드에 맞춰 주세요."


@dataclass(frozen=True)
class CaptureGuide:
    center_region: FaceBox
    head_box: FaceBox
    shoulder_points: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class FaceFraming:
    ready: bool
    warning: str = ""


def build_capture_guide(frame_width: int, frame_height: int) -> CaptureGuide:
    min_dimension = min(frame_width, frame_height)
    center_width = int(round(frame_width * _CENTER_REGION_RATIO))
    center_height = int(round(frame_height * _CENTER_REGION_RATIO))
    head_width = int(round(min_dimension * _HEAD_WIDTH_MIN_DIMENSION_RATIO))
    head_height = int(round(head_width * _HEAD_HEIGHT_TO_WIDTH_RATIO))
    head_x = int(round((frame_width - head_width) * 0.5))
    head_y = int(round(frame_height * _HEAD_TOP_FRAME_RATIO))
    head_box = FaceBox(head_x, head_y, head_width, head_height)
    result = CaptureGuide(
        center_region=FaceBox(
            int(round((frame_width - center_width) * 0.5)),
            int(round((frame_height - center_height) * 0.5)),
            center_width,
            center_height,
        ),
        head_box=head_box,
        shoulder_points=_build_shoulder_points(frame_width, frame_height, head_box),
    )
    return result


def evaluate_face_framing(
    face: FaceBox,
    frame_width: int,
    frame_height: int,
) -> FaceFraming:
    guide = build_capture_guide(frame_width, frame_height)
    warning = ""
    if not _face_center_inside(face, guide.center_region):
        warning = _CENTER_WARNING
    elif not _face_center_matches_guide(face, guide.head_box):
        warning = _GUIDE_POSITION_WARNING
    elif _face_too_small(face, guide.head_box):
        warning = _TOO_FAR_WARNING
    elif _face_too_large(face, guide.head_box):
        warning = _TOO_CLOSE_WARNING
    result = FaceFraming(ready=warning == "", warning=warning)
    return result


def _build_shoulder_points(
    frame_width: int,
    frame_height: int,
    head_box: FaceBox,
) -> tuple[tuple[int, int], ...]:
    head_center_x = head_box.x + (head_box.width * 0.5)
    neck_y = head_box.y + head_box.height
    shoulder_half_width = head_box.width * _SHOULDER_WIDTH_TO_HEAD_RATIO * 0.5
    shoulder_y = frame_height * _SHOULDER_BOTTOM_FRAME_RATIO
    result = (
        (int(round(head_center_x - (head_box.width * 0.36))), int(round(neck_y))),
        (int(round(head_center_x + (head_box.width * 0.36))), int(round(neck_y))),
        (int(round(max(0.0, head_center_x - shoulder_half_width))), int(round(shoulder_y))),
        (
            int(round(min(float(frame_width - 1), head_center_x + shoulder_half_width))),
            int(round(shoulder_y)),
        ),
    )
    return result


def _face_center_inside(face: FaceBox, region: FaceBox) -> bool:
    center_x = face.x + (face.width * 0.5)
    center_y = face.y + (face.height * 0.5)
    result = (
        region.x <= center_x <= region.x + region.width
        and region.y <= center_y <= region.y + region.height
    )
    return result


def _face_center_matches_guide(face: FaceBox, guide_box: FaceBox) -> bool:
    face_center_x = face.x + (face.width * 0.5)
    face_center_y = face.y + (face.height * 0.5)
    guide_center_x = guide_box.x + (guide_box.width * 0.5)
    guide_center_y = guide_box.y + (guide_box.height * 0.5)
    result = (
        abs(face_center_x - guide_center_x)
        <= guide_box.width * _HEAD_CENTER_TOLERANCE_RATIO
        and abs(face_center_y - guide_center_y)
        <= guide_box.height * _HEAD_CENTER_TOLERANCE_RATIO
    )
    return result


def _face_too_small(face: FaceBox, guide_box: FaceBox) -> bool:
    min_width = guide_box.width * (1.0 - _HEAD_SIZE_TOLERANCE_RATIO)
    min_height = guide_box.height * (1.0 - _HEAD_SIZE_TOLERANCE_RATIO)
    result = face.width < min_width or face.height < min_height
    return result


def _face_too_large(face: FaceBox, guide_box: FaceBox) -> bool:
    max_width = guide_box.width * (1.0 + _HEAD_SIZE_TOLERANCE_RATIO)
    max_height = guide_box.height * (1.0 + _HEAD_SIZE_TOLERANCE_RATIO)
    result = face.width > max_width or face.height > max_height
    return result
