from __future__ import annotations

import numpy as np

from expresso_calib.board import DEFAULT_BOARD
from expresso_calib.calibration import CalibrationAccumulator
from expresso_calib.detection import DetectionResult


def fake_detection(
    *,
    frame_index: int,
    center_x: float,
    center_y: float,
    area_fraction: float = 0.10,
    sharpness: float = 120.0,
) -> DetectionResult:
    width = 960
    height = 540
    side = (area_fraction * width * height) ** 0.5
    cx = center_x * width
    cy = center_y * height
    xs = np.linspace(cx - side / 2, cx + side / 2, 4)
    ys = np.linspace(cy - side / 2, cy + side / 2, 4)
    points = np.array([[[x, y]] for y in ys for x in xs], dtype=np.float32)
    ids = np.arange(len(points), dtype=np.int32).reshape(-1, 1)
    return DetectionResult(
        frame_index=frame_index,
        timestamp_sec=frame_index / 10.0,
        width=width,
        height=height,
        marker_count=12,
        charuco_count=len(points),
        sharpness=sharpness,
        corners=points,
        ids=ids,
        center_x=center_x,
        center_y=center_y,
        bbox_width=side / width,
        bbox_height=side / height,
        area_fraction=area_fraction,
        angle_deg=0.0,
    )


def test_accumulator_rejects_duplicate_pose(tmp_path) -> None:
    accumulator = CalibrationAccumulator(DEFAULT_BOARD, tmp_path)
    image = np.zeros((540, 960, 3), dtype=np.uint8)

    accepted, reason = accumulator.observe(
        fake_detection(frame_index=1, center_x=0.5, center_y=0.5), image
    )
    assert accepted is True
    assert reason == "accepted"

    accepted, reason = accumulator.observe(
        fake_detection(frame_index=2, center_x=0.5, center_y=0.5), image
    )
    assert accepted is False
    assert reason == "duplicate pose"
    assert len(accumulator.candidates) == 1


def test_accumulator_accepts_novel_pose(tmp_path) -> None:
    accumulator = CalibrationAccumulator(DEFAULT_BOARD, tmp_path)
    image = np.zeros((540, 960, 3), dtype=np.uint8)

    accumulator.observe(fake_detection(frame_index=1, center_x=0.5, center_y=0.5), image)
    accepted, reason = accumulator.observe(
        fake_detection(frame_index=2, center_x=0.2, center_y=0.3), image
    )

    assert accepted is True
    assert reason == "accepted"
    assert len(accumulator.candidates) == 2


def test_accumulator_rejects_weak_frames(tmp_path) -> None:
    accumulator = CalibrationAccumulator(DEFAULT_BOARD, tmp_path)
    image = np.zeros((540, 960, 3), dtype=np.uint8)

    accepted, reason = accumulator.observe(
        fake_detection(
            frame_index=1,
            center_x=0.5,
            center_y=0.5,
            area_fraction=0.004,
        ),
        image,
    )

    assert accepted is False
    assert reason == "board is too small"
