from __future__ import annotations

import asyncio
import json

import cv2
import numpy as np
import pytest

from expresso_calib.board import DEFAULT_BOARD
from expresso_calib.calibration import (
    CalibrationAccumulator,
    CalibrationResult,
    SolveInsufficientData,
    SolveNumericalFailure,
    SolveOk,
)
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


def test_accumulator_rejects_duplicate_still_frame(tmp_path) -> None:
    accumulator = CalibrationAccumulator(DEFAULT_BOARD, tmp_path)
    image = np.zeros((540, 960, 3), dtype=np.uint8)

    accepted, reason = accumulator.observe(
        fake_detection(frame_index=1, center_x=0.50, center_y=0.50), image
    )
    assert accepted is True
    assert reason == "accepted"

    accepted, reason = accumulator.observe(
        fake_detection(frame_index=2, center_x=0.56, center_y=0.50), image
    )

    assert accepted is False
    assert reason == "duplicate frame"
    assert accumulator.duplicate_image_rejections == 1
    assert len(accumulator.candidates) == 1


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


def test_solve_does_not_export_artifacts_by_default(tmp_path) -> None:
    accumulator = CalibrationAccumulator(
        DEFAULT_BOARD,
        tmp_path,
        min_solve_frames=2,
        solve_every_new_frames=2,
        create_run_dir=False,
    )
    image = np.zeros((540, 960, 3), dtype=np.uint8)

    def fake_calibrate(selected, **_kwargs):
        return CalibrationResult(
            rms_reprojection_error_px=0.72,
            camera_matrix=np.eye(3, dtype=float),
            distortion_coefficients=np.zeros(5, dtype=float),
            per_view_errors_px=[0.5 for _ in selected],
            selected_count=len(selected),
            flags=0,
        )

    accumulator._calibrate = fake_calibrate
    accumulator.observe(fake_detection(frame_index=1, center_x=0.2, center_y=0.2), image)
    accumulator.observe(fake_detection(frame_index=2, center_x=0.7, center_y=0.2), image)
    accumulator.observe(fake_detection(frame_index=3, center_x=0.2, center_y=0.7), image)
    accumulator.observe(fake_detection(frame_index=4, center_x=0.7, center_y=0.7), image)
    accumulator.solve_if_due()

    assert accumulator.last_calibration is not None
    assert list(tmp_path.rglob("calibration.json")) == []
    assert list(tmp_path.rglob("report.md")) == []
    assert list(tmp_path.rglob("detections.csv")) == []


def test_snapshot_solve_preserves_samples_collected_while_solving(tmp_path) -> None:
    accumulator = CalibrationAccumulator(
        DEFAULT_BOARD,
        tmp_path,
        min_solve_frames=4,
        solve_every_new_frames=4,
        create_run_dir=False,
    )
    image = np.zeros((540, 960, 3), dtype=np.uint8)

    def fake_calibrate(selected, **_kwargs):
        return CalibrationResult(
            rms_reprojection_error_px=0.72,
            camera_matrix=np.eye(3, dtype=float),
            distortion_coefficients=np.zeros(5, dtype=float),
            per_view_errors_px=[0.5 for _ in selected],
            selected_count=len(selected),
            flags=0,
        )

    accumulator._calibrate = fake_calibrate
    for index, center in enumerate([(0.2, 0.2), (0.7, 0.2), (0.2, 0.7), (0.7, 0.7)], start=1):
        accepted, _ = accumulator.observe(
            fake_detection(frame_index=index, center_x=center[0], center_y=center[1]),
            image,
        )
        assert accepted is True

    consumed = accumulator.accepted_since_solve
    outcome = accumulator.solve_snapshot(list(accumulator.candidates))
    assert isinstance(outcome, SolveOk)
    result = outcome.solve
    assert not any(item.selected for item in accumulator.candidates)

    for index, center in enumerate([(0.5, 0.25), (0.25, 0.5)], start=5):
        accepted, _ = accumulator.observe(
            fake_detection(frame_index=index, center_x=center[0], center_y=center[1]),
            image,
        )
        assert accepted is True

    accumulator.commit_solve_result(result, consumed_new_frames=consumed)

    assert len(accumulator.candidates) == 6
    assert accumulator.accepted_since_solve == 2
    assert accumulator.last_calibration is not None
    assert accumulator.solve_history[-1]["candidateFrames"] == 4
    assert accumulator.last_quality["usableFrames"] == 4


def test_snapshot_solve_rejects_high_error_selected_outlier(tmp_path) -> None:
    accumulator = CalibrationAccumulator(
        DEFAULT_BOARD,
        tmp_path,
        min_solve_frames=5,
        max_calib_frames=5,
        min_outlier_refine_frames=4,
        create_run_dir=False,
    )
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    for index, center in enumerate(
        [(0.2, 0.2), (0.7, 0.2), (0.2, 0.7), (0.7, 0.7), (0.45, 0.45)], start=1
    ):
        accepted, _ = accumulator.observe(
            fake_detection(frame_index=index, center_x=center[0], center_y=center[1]),
            image,
        )
        assert accepted is True

    calibrations = iter(
        [
            CalibrationResult(
                rms_reprojection_error_px=2.10,
                camera_matrix=np.eye(3, dtype=float),
                distortion_coefficients=np.zeros(5, dtype=float),
                per_view_errors_px=[0.45, 0.50, 4.20, 0.55, 0.48],
                selected_count=5,
                flags=0,
            ),
            CalibrationResult(
                rms_reprojection_error_px=0.52,
                camera_matrix=np.eye(3, dtype=float),
                distortion_coefficients=np.zeros(5, dtype=float),
                per_view_errors_px=[0.42, 0.48, 0.46, 0.44],
                selected_count=4,
                flags=0,
            ),
        ]
    )

    def fake_calibrate(_selected, **_kwargs):
        return next(calibrations)

    accumulator._calibrate = fake_calibrate

    outcome = accumulator.solve_snapshot(list(accumulator.candidates))
    assert isinstance(outcome, SolveOk)
    result = outcome.solve

    assert result.calibration.rms_reprojection_error_px == 0.52
    assert len(result.selected) == 4
    assert result.quality["initialRmsReprojectionErrorPx"] == 2.10
    assert result.quality["outlierRejection"]["rejectedFrames"] == 1
    assert result.quality["verdict"] == "REDO"
    assert len(result.rejected) == 1
    assert result.rejected[0].detection.frame_index == 3


def test_refine_outliers_marks_rejected_frames_and_rescores_them(tmp_path, monkeypatch) -> None:
    accumulator = CalibrationAccumulator(
        DEFAULT_BOARD,
        tmp_path,
        min_solve_frames=5,
        max_calib_frames=5,
        min_outlier_refine_frames=4,
        create_run_dir=False,
    )
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    for index, center in enumerate(
        [(0.2, 0.2), (0.7, 0.2), (0.2, 0.7), (0.7, 0.7), (0.45, 0.45)], start=1
    ):
        accepted, _ = accumulator.observe(
            fake_detection(frame_index=index, center_x=center[0], center_y=center[1]),
            image,
        )
        assert accepted is True

    calibrations = iter(
        [
            CalibrationResult(
                rms_reprojection_error_px=2.10,
                camera_matrix=np.eye(3, dtype=float),
                distortion_coefficients=np.zeros(5, dtype=float),
                per_view_errors_px=[0.45, 0.50, 4.20, 0.55, 0.48],
                selected_count=5,
                flags=0,
            ),
            CalibrationResult(
                rms_reprojection_error_px=0.52,
                camera_matrix=np.eye(3, dtype=float),
                distortion_coefficients=np.zeros(5, dtype=float),
                per_view_errors_px=[0.42, 0.48, 0.46, 0.44],
                selected_count=4,
                flags=0,
            ),
        ]
    )

    def fake_calibrate(_selected, **_kwargs):
        return next(calibrations)

    accumulator._calibrate = fake_calibrate

    rescore_calls: list[int] = []

    def fake_rescore(candidate, camera_matrix, dist_coeffs):
        rescore_calls.append(candidate.detection.frame_index)
        return 3.75

    monkeypatch.setattr(accumulator, "_project_per_view_error", fake_rescore)

    outcome = accumulator.solve_snapshot(list(accumulator.candidates))
    assert isinstance(outcome, SolveOk)
    accumulator.commit_solve_result(outcome.solve)

    rejected = [item for item in accumulator.candidates if item.rejected]
    kept = [item for item in accumulator.candidates if item.selected]
    assert len(rejected) == 1
    assert len(kept) == 4
    assert rejected[0].detection.frame_index == 3
    assert rejected[0].per_view_error_px == pytest.approx(3.75)
    assert rejected[0].selected is False
    assert rescore_calls == [3]
    for item in kept:
        assert item.rejected is False


def test_candidate_frame_default_rejected_is_false(tmp_path) -> None:
    accumulator = CalibrationAccumulator(DEFAULT_BOARD, tmp_path, create_run_dir=False)
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    accumulator.observe(fake_detection(frame_index=1, center_x=0.5, center_y=0.5), image)
    assert accumulator.candidates[-1].rejected is False


def test_snapshot_solve_prefers_strong_frames_when_available(tmp_path) -> None:
    accumulator = CalibrationAccumulator(
        DEFAULT_BOARD,
        tmp_path,
        min_solve_frames=4,
        max_calib_frames=8,
        create_run_dir=False,
    )
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    for index, center in enumerate(
        [(0.2, 0.2), (0.7, 0.2), (0.2, 0.7), (0.7, 0.7), (0.45, 0.45), (0.55, 0.55)],
        start=1,
    ):
        detection = fake_detection(
            frame_index=index,
            center_x=center[0],
            center_y=center[1],
            sharpness=35.0,
        )
        if index <= 2:
            detection.charuco_count = 12
        else:
            detection.charuco_count = 24
        accepted, _ = accumulator.observe(detection, image)
        assert accepted is True

    def fake_calibrate(selected, **_kwargs):
        assert all(
            item.detection.charuco_count >= accumulator.min_solve_corners for item in selected
        )
        return CalibrationResult(
            rms_reprojection_error_px=0.62,
            camera_matrix=np.eye(3, dtype=float),
            distortion_coefficients=np.zeros(5, dtype=float),
            per_view_errors_px=[0.5 for _ in selected],
            selected_count=len(selected),
            flags=0,
        )

    accumulator._calibrate = fake_calibrate

    outcome = accumulator.solve_snapshot(list(accumulator.candidates))
    assert isinstance(outcome, SolveOk)
    result = outcome.solve

    assert result.quality["acceptedFrames"] == 6
    assert result.quality["solvePoolFrames"] == 4
    assert result.quality["weakSolveFrames"] == 2
    assert result.quality["usingStrongSolvePool"] is True
    assert len(result.selected) == 4


def test_candidate_screenshot_is_written(tmp_path) -> None:
    accumulator = CalibrationAccumulator(DEFAULT_BOARD, tmp_path, create_run_dir=False)
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    accepted, _ = accumulator.observe(
        fake_detection(frame_index=1, center_x=0.5, center_y=0.5), image
    )

    assert accepted is True
    path = accumulator.write_candidate_screenshot(
        accumulator.candidates[-1], tmp_path / "screenshots"
    )
    assert path.exists()
    assert path.suffix == ".jpg"


def test_solve_snapshot_returns_insufficient_data_when_too_few_candidates(tmp_path) -> None:
    accumulator = CalibrationAccumulator(
        DEFAULT_BOARD, tmp_path, min_solve_frames=5, create_run_dir=False
    )
    outcome = accumulator.solve_snapshot([])
    assert isinstance(outcome, SolveInsufficientData)
    assert outcome.reason == "too few candidates"


def test_solve_snapshot_returns_ok_on_successful_solve(tmp_path) -> None:
    accumulator = CalibrationAccumulator(
        DEFAULT_BOARD,
        tmp_path,
        min_solve_frames=4,
        solve_every_new_frames=4,
        create_run_dir=False,
    )
    image = np.zeros((540, 960, 3), dtype=np.uint8)

    def fake_calibrate(selected, **_kwargs):
        return CalibrationResult(
            rms_reprojection_error_px=0.72,
            camera_matrix=np.eye(3, dtype=float),
            distortion_coefficients=np.zeros(5, dtype=float),
            per_view_errors_px=[0.5 for _ in selected],
            selected_count=len(selected),
            flags=0,
        )

    accumulator._calibrate = fake_calibrate
    for index, center in enumerate([(0.2, 0.2), (0.7, 0.2), (0.2, 0.7), (0.7, 0.7)], start=1):
        accumulator.observe(
            fake_detection(frame_index=index, center_x=center[0], center_y=center[1]),
            image,
        )

    outcome = accumulator.solve_snapshot(list(accumulator.candidates))
    assert isinstance(outcome, SolveOk)
    assert outcome.solve.calibration.rms_reprojection_error_px == 0.72


def test_solve_snapshot_returns_numerical_failure_when_cv2_raises(tmp_path) -> None:
    accumulator = CalibrationAccumulator(
        DEFAULT_BOARD,
        tmp_path,
        min_solve_frames=4,
        solve_every_new_frames=4,
        create_run_dir=False,
    )
    image = np.zeros((540, 960, 3), dtype=np.uint8)

    def exploding_calibrate(_selected, **_kwargs):
        raise cv2.error("rank-deficient input")

    accumulator._calibrate = exploding_calibrate
    for index, center in enumerate([(0.2, 0.2), (0.7, 0.2), (0.2, 0.7), (0.7, 0.7)], start=1):
        accumulator.observe(
            fake_detection(frame_index=index, center_x=center[0], center_y=center[1]),
            image,
        )

    outcome = accumulator.solve_snapshot(list(accumulator.candidates))
    assert isinstance(outcome, SolveNumericalFailure)
    assert "rank-deficient input" in outcome.reason


def test_export_writes_expected_files(tmp_path) -> None:
    accumulator = CalibrationAccumulator(
        DEFAULT_BOARD,
        tmp_path,
        min_solve_frames=4,
        solve_every_new_frames=4,
    )
    image = np.zeros((540, 960, 3), dtype=np.uint8)

    def fake_calibrate(selected, **_kwargs):
        return CalibrationResult(
            rms_reprojection_error_px=0.72,
            camera_matrix=np.eye(3, dtype=float),
            distortion_coefficients=np.zeros(5, dtype=float),
            per_view_errors_px=[0.5 for _ in selected],
            selected_count=len(selected),
            flags=0,
        )

    accumulator._calibrate = fake_calibrate
    for index, center in enumerate([(0.2, 0.2), (0.7, 0.2), (0.2, 0.7), (0.7, 0.7)], start=1):
        accumulator.observe(
            fake_detection(frame_index=index, center_x=center[0], center_y=center[1]),
            image,
        )
    accumulator.solve_if_due()
    run_dir = accumulator.export()

    assert (run_dir / "calibration.json").exists()
    assert (run_dir / "detections.csv").exists()
    assert (run_dir / "report.md").exists()

    payload = json.loads((run_dir / "calibration.json").read_text())
    assert payload["calibration"]["rms_reprojection_error_px"] == pytest.approx(0.72)
    assert "rejected" in payload["selected_frames"][0]

    csv_text = (run_dir / "detections.csv").read_text()
    assert "rejected" in csv_text.splitlines()[0]


@pytest.mark.asyncio
async def test_concurrent_observe_and_commit_does_not_corrupt_candidates(tmp_path) -> None:
    accumulator = CalibrationAccumulator(
        DEFAULT_BOARD, tmp_path, min_solve_frames=4, create_run_dir=False
    )
    image = np.zeros((540, 960, 3), dtype=np.uint8)

    def fake_calibrate(selected, **_kwargs):
        return CalibrationResult(
            rms_reprojection_error_px=0.7,
            camera_matrix=np.eye(3, dtype=float),
            distortion_coefficients=np.zeros(5, dtype=float),
            per_view_errors_px=[0.5 for _ in selected],
            selected_count=len(selected),
            flags=0,
        )

    accumulator._calibrate = fake_calibrate

    for i, (cx, cy) in enumerate([(0.2, 0.2), (0.7, 0.2), (0.2, 0.7), (0.7, 0.7)], start=1):
        accumulator.observe(fake_detection(frame_index=i, center_x=cx, center_y=cy), image)

    async def keep_observing() -> None:
        for i in range(5, 15):
            async with accumulator.lock:
                accumulator.observe(
                    fake_detection(frame_index=i, center_x=0.3 + i * 0.02, center_y=0.3),
                    image,
                )
                await asyncio.sleep(0)

    async def keep_solving() -> None:
        for _ in range(5):
            async with accumulator.lock:
                snapshot = list(accumulator.candidates)
            outcome = await asyncio.to_thread(accumulator.solve_snapshot, snapshot)
            async with accumulator.lock:
                if isinstance(outcome, SolveOk):
                    accumulator.commit_solve_result(outcome.solve)
            await asyncio.sleep(0.001)

    await asyncio.gather(keep_observing(), keep_solving())

    selected = sum(1 for item in accumulator.candidates if item.selected)
    assert selected <= len(accumulator.candidates)
    for item in accumulator.candidates:
        assert item.per_view_error_px is None or isinstance(item.per_view_error_px, float)
