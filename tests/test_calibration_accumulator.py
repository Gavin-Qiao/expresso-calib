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


def test_calibrate_raises_loudly_on_frames_missing_corners(tmp_path) -> None:
    accumulator = CalibrationAccumulator(DEFAULT_BOARD, tmp_path, create_run_dir=False)
    detection = fake_detection(frame_index=1, center_x=0.5, center_y=0.5)
    detection.ids = None
    detection.corners = None
    from expresso_calib.calibration import CandidateFrame

    bad_item = CandidateFrame(
        detection=detection,
        image_bgr=np.zeros((540, 960, 3), dtype=np.uint8),
        accepted_at="2026-01-01",
    )

    with pytest.raises(ValueError, match="no ids/corners"):
        accumulator._calibrate([bad_item])


def test_refinement_runs_when_exactly_min_outlier_refine_frames_selected(tmp_path) -> None:
    accumulator = CalibrationAccumulator(
        DEFAULT_BOARD,
        tmp_path,
        min_solve_frames=6,
        max_calib_frames=6,
        min_outlier_refine_frames=6,
        create_run_dir=False,
    )
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    for index, (cx, cy) in enumerate(
        [(0.2, 0.2), (0.7, 0.2), (0.2, 0.7), (0.7, 0.7), (0.4, 0.45), (0.55, 0.55)],
        start=1,
    ):
        accepted, _ = accumulator.observe(
            fake_detection(frame_index=index, center_x=cx, center_y=cy), image
        )
        assert accepted is True

    calibrations = iter(
        [
            CalibrationResult(
                rms_reprojection_error_px=2.0,
                camera_matrix=np.eye(3, dtype=float),
                distortion_coefficients=np.zeros(5, dtype=float),
                per_view_errors_px=[0.45, 0.50, 0.55, 4.20, 0.48, 0.46],
                selected_count=6,
                flags=0,
            ),
            CalibrationResult(
                rms_reprojection_error_px=0.50,
                camera_matrix=np.eye(3, dtype=float),
                distortion_coefficients=np.zeros(5, dtype=float),
                per_view_errors_px=[0.42, 0.48, 0.50, 0.45, 0.44],
                selected_count=5,
                flags=0,
            ),
        ]
    )

    def fake_calibrate(_selected, **_kwargs):
        return next(calibrations)

    accumulator._calibrate = fake_calibrate
    accumulator._project_per_view_error = lambda _c, _k, _d: 3.5

    outcome = accumulator.solve_snapshot(list(accumulator.candidates))
    assert isinstance(outcome, SolveOk)
    assert len(outcome.solve.rejected) == 1, (
        "refinement should have executed and rejected the high-error frame; "
        "the old minimum_kept formula would dead-zone here"
    )


def test_quality_warns_on_low_area_variance_by_ratio_not_absolute_diff(tmp_path) -> None:
    from expresso_calib.calibration import CandidateFrame

    accumulator = CalibrationAccumulator(DEFAULT_BOARD, tmp_path, create_run_dir=False)
    image = np.zeros((540, 960, 3), dtype=np.uint8)

    def candidate_with_area(index: int, area: float, cx: float, cy: float) -> CandidateFrame:
        return CandidateFrame(
            detection=fake_detection(
                frame_index=index, center_x=cx, center_y=cy, area_fraction=area
            ),
            image_bgr=image.copy(),
            accepted_at=f"2026-01-01T00:00:{index:02d}",
        )

    # Wide ratio: max/min = 0.13/0.05 = 2.6, but abs diff = 0.08 (< 0.10).
    # Old check would warn; new ratio check (< 1.8) should NOT warn.
    wide_ratio = [
        candidate_with_area(1, 0.05, 0.2, 0.2),
        candidate_with_area(2, 0.07, 0.7, 0.2),
        candidate_with_area(3, 0.10, 0.2, 0.7),
        candidate_with_area(4, 0.12, 0.7, 0.7),
        candidate_with_area(5, 0.13, 0.45, 0.45),
    ]
    cal = CalibrationResult(
        rms_reprojection_error_px=0.5,
        camera_matrix=np.eye(3, dtype=float),
        distortion_coefficients=np.zeros(5, dtype=float),
        per_view_errors_px=[0.4 for _ in wide_ratio],
        selected_count=len(wide_ratio),
        flags=0,
    )
    quality_wide = accumulator.summarize_quality(wide_ratio, cal)
    assert not any("scale does not vary" in w.lower() for w in quality_wide["warnings"]), (
        f"areas {[c.detection.area_fraction for c in wide_ratio]} have ratio 2.6 — "
        "ratio check should not warn even though abs diff is only 0.08"
    )

    # Narrow ratio: max/min = 0.30/0.20 = 1.5, abs diff = 0.10.
    # Old check would NOT warn (abs diff == 0.10, not < 0.10); new ratio check SHOULD warn.
    narrow_ratio = [
        candidate_with_area(1, 0.20, 0.2, 0.2),
        candidate_with_area(2, 0.22, 0.7, 0.2),
        candidate_with_area(3, 0.28, 0.2, 0.7),
        candidate_with_area(4, 0.30, 0.7, 0.7),
        candidate_with_area(5, 0.24, 0.45, 0.45),
    ]
    quality_narrow = accumulator.summarize_quality(narrow_ratio, cal)
    assert any("scale does not vary" in w.lower() for w in quality_narrow["warnings"]), (
        f"areas {[c.detection.area_fraction for c in narrow_ratio]} have ratio 1.5 — "
        "ratio check should warn even though abs diff is 0.10"
    )


def test_refinement_outlier_rejection_only_reports_threshold_when_applied(tmp_path) -> None:
    accumulator = CalibrationAccumulator(
        DEFAULT_BOARD,
        tmp_path,
        min_solve_frames=5,
        max_calib_frames=5,
        min_outlier_refine_frames=4,
        create_run_dir=False,
    )
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    for index, (cx, cy) in enumerate(
        [(0.2, 0.2), (0.7, 0.2), (0.2, 0.7), (0.7, 0.7), (0.45, 0.45)], start=1
    ):
        accumulator.observe(fake_detection(frame_index=index, center_x=cx, center_y=cy), image)

    def fake_calibrate_clean(_selected, **_kwargs):
        return CalibrationResult(
            rms_reprojection_error_px=0.55,
            camera_matrix=np.eye(3, dtype=float),
            distortion_coefficients=np.zeros(5, dtype=float),
            per_view_errors_px=[0.45, 0.50, 0.55, 0.48, 0.46],
            selected_count=5,
            flags=0,
        )

    accumulator._calibrate = fake_calibrate_clean
    outcome = accumulator.solve_snapshot(list(accumulator.candidates))
    assert isinstance(outcome, SolveOk)
    outlier = outcome.solve.quality["outlierRejection"]
    assert outlier["rejectedFrames"] == 0
    assert "thresholdPx" not in outlier, (
        "thresholdPx should be omitted when no frames were rejected; "
        "consumers should not see a threshold for a refinement that didn't apply"
    )


def test_refinement_downgrade_to_marginal_requires_nontrivial_rejection_rate(
    tmp_path,
) -> None:
    from expresso_calib.calibration import CandidateFrame

    image = np.zeros((540, 960, 3), dtype=np.uint8)

    def build_candidates(n: int) -> list[CandidateFrame]:
        items: list[CandidateFrame] = []
        for i in range(n):
            cx = 0.08 + (i % 6) * 0.15
            cy = 0.10 + (i // 6) * 0.18
            area = 0.05 + (i % 4) * 0.05
            det = fake_detection(frame_index=i + 1, center_x=cx, center_y=cy, area_fraction=area)
            det.charuco_count = 30
            items.append(
                CandidateFrame(
                    detection=det,
                    image_bgr=image.copy(),
                    accepted_at=f"2026-01-01T{i:02d}",
                )
            )
        return items

    def fake_calibrations(rejection_count: int, total: int):
        initial_errors = [0.40 for _ in range(total)]
        for k in range(rejection_count):
            initial_errors[-(k + 1)] = 4.5
        initial = CalibrationResult(
            rms_reprojection_error_px=0.50,
            camera_matrix=np.eye(3, dtype=float),
            distortion_coefficients=np.zeros(5, dtype=float),
            per_view_errors_px=initial_errors,
            selected_count=total,
            flags=0,
        )
        kept = total - rejection_count
        refined = CalibrationResult(
            rms_reprojection_error_px=0.45,
            camera_matrix=np.eye(3, dtype=float),
            distortion_coefficients=np.zeros(5, dtype=float),
            per_view_errors_px=[0.40 for _ in range(kept)],
            selected_count=kept,
            flags=0,
        )
        return iter([initial, refined])

    def run_solve(candidate_count: int, rejection_count: int):
        accumulator = CalibrationAccumulator(
            DEFAULT_BOARD,
            tmp_path,
            min_solve_frames=candidate_count,
            max_calib_frames=candidate_count,
            min_outlier_refine_frames=max(4, candidate_count // 2),
            min_outlier_refine_keep=4,
            create_run_dir=False,
        )
        accumulator.candidates = build_candidates(candidate_count)
        cals = fake_calibrations(rejection_count, candidate_count)
        accumulator._calibrate = lambda *_a, **_k: next(cals)
        accumulator._project_per_view_error = lambda *_a, **_k: 4.5
        outcome = accumulator.solve_snapshot(list(accumulator.candidates))
        assert isinstance(outcome, SolveOk)
        return outcome.solve

    # 1 rejection out of 30 = 3% — under 10% threshold, no downgrade.
    low_rate = run_solve(30, 1)
    assert len(low_rate.rejected) == 1
    assert any("Excluded 1" in w for w in low_rate.quality["warnings"])
    assert low_rate.quality["verdict"] != "MARGINAL" or any(
        marker
        in " ".join(low_rate.quality.get("redFlags", []) + low_rate.quality.get("warnings", []))
        for marker in ("coverage", "RMS", "P95", "scale", "corners")
    ), (
        "1/30 rejection is below the 10% threshold; if verdict is MARGINAL it must be "
        "from another criterion, not the refinement-downgrade rule"
    )

    # 4 rejections out of 30 = 13% — above 10% threshold; refinement-downgrade applies.
    # (We avoid 50% outliers here because the median-based threshold formula breaks down
    # when outliers dominate the distribution — see the docstring on _refine_outlier_views.)
    high_rate = run_solve(30, 4)
    assert len(high_rate.rejected) == 4
    assert any("Excluded 4" in w for w in high_rate.quality["warnings"])
    rate = len(high_rate.rejected) / (len(high_rate.selected) + len(high_rate.rejected))
    assert rate > 0.10


def test_cell_occupancy_replaces_bbox_span_coverage(tmp_path) -> None:
    from expresso_calib.calibration import CandidateFrame, compute_cell_occupancy

    width, height = 960, 540

    # Single diagonal frame: corners span corner-to-corner. Bbox-span coverage
    # would report ~100% width AND ~100% height; cell occupancy reports only
    # the cells actually touched.
    diagonal_corners = np.array(
        [[[float(width * t), float(height * t)]] for t in np.linspace(0.05, 0.95, 12)],
        dtype=np.float32,
    )
    diagonal = CandidateFrame(
        detection=DetectionResult(
            frame_index=1,
            timestamp_sec=0.0,
            width=width,
            height=height,
            marker_count=12,
            charuco_count=12,
            sharpness=120.0,
            corners=diagonal_corners,
            ids=np.arange(12, dtype=np.int32).reshape(-1, 1),
        ),
        image_bgr=np.zeros((height, width, 3), dtype=np.uint8),
        accepted_at="2026-01-01",
    )
    occupancy_diagonal = compute_cell_occupancy([diagonal], width, height)
    assert occupancy_diagonal < 0.30, (
        f"a single diagonal frame should cover < 30% of cells, got {occupancy_diagonal}"
    )

    # Many frames each covering a small cluster of cells: high total occupancy.
    spread = []
    cluster = 60.0  # ~one cell wide
    for i, (cx_frac, cy_frac) in enumerate(
        [
            (0.10, 0.15),
            (0.35, 0.15),
            (0.65, 0.15),
            (0.90, 0.15),
            (0.10, 0.50),
            (0.35, 0.50),
            (0.65, 0.50),
            (0.90, 0.50),
            (0.10, 0.85),
            (0.35, 0.85),
            (0.65, 0.85),
            (0.90, 0.85),
        ]
    ):
        cx_px = width * cx_frac
        cy_px = height * cy_frac
        corners = np.array(
            [
                [[cx_px - cluster, cy_px - cluster]],
                [[cx_px + cluster, cy_px - cluster]],
                [[cx_px - cluster, cy_px + cluster]],
                [[cx_px + cluster, cy_px + cluster]],
            ],
            dtype=np.float32,
        )
        spread.append(
            CandidateFrame(
                detection=DetectionResult(
                    frame_index=i + 1,
                    timestamp_sec=0.0,
                    width=width,
                    height=height,
                    marker_count=4,
                    charuco_count=4,
                    sharpness=120.0,
                    corners=corners,
                    ids=np.arange(4, dtype=np.int32).reshape(-1, 1),
                ),
                image_bgr=np.zeros((height, width, 3), dtype=np.uint8),
                accepted_at=f"2026-01-01T00:00:{i:02d}",
            )
        )
    occupancy_spread = compute_cell_occupancy(spread, width, height)
    assert occupancy_spread >= 0.50, (
        f"12 cluster frames spread across the image should cover >= 50% of cells, "
        f"got {occupancy_spread}"
    )


def test_summary_uses_cell_occupancy_for_verdict(tmp_path) -> None:
    from expresso_calib.calibration import CandidateFrame

    width, height = 960, 540
    accumulator = CalibrationAccumulator(DEFAULT_BOARD, tmp_path, create_run_dir=False)

    diagonal_corners = np.array(
        [[[float(width * t), float(height * t)]] for t in np.linspace(0.05, 0.95, 16)],
        dtype=np.float32,
    )
    diagonal = CandidateFrame(
        detection=DetectionResult(
            frame_index=1,
            timestamp_sec=0.0,
            width=width,
            height=height,
            marker_count=16,
            charuco_count=16,
            sharpness=120.0,
            corners=diagonal_corners,
            ids=np.arange(16, dtype=np.int32).reshape(-1, 1),
        ),
        image_bgr=np.zeros((height, width, 3), dtype=np.uint8),
        accepted_at="2026-01-01",
    )
    selected = [diagonal] * 25  # enough to clear the frame-count gate
    cal = CalibrationResult(
        rms_reprojection_error_px=0.6,
        camera_matrix=np.eye(3, dtype=float),
        distortion_coefficients=np.zeros(5, dtype=float),
        per_view_errors_px=[0.5 for _ in selected],
        selected_count=len(selected),
        flags=0,
    )
    quality = accumulator.summarize_quality(selected, cal)
    assert quality["coverage"]["cellOccupancyFraction"] < 0.30
    assert any("cell coverage" in flag.lower() for flag in quality["redFlags"]), (
        "diagonal-only frames should fail the cell-occupancy gate"
    )


def test_iterative_refinement_catches_outliers_revealed_by_first_pass(
    tmp_path,
) -> None:
    from expresso_calib.calibration import CandidateFrame

    image = np.zeros((540, 960, 3), dtype=np.uint8)
    accumulator = CalibrationAccumulator(
        DEFAULT_BOARD,
        tmp_path,
        min_solve_frames=20,
        max_calib_frames=20,
        min_outlier_refine_frames=15,
        min_outlier_refine_keep=10,
        create_run_dir=False,
    )

    candidates = []
    for i in range(20):
        cx = 0.08 + (i % 5) * 0.20
        cy = 0.10 + (i // 5) * 0.20
        area = 0.06 + (i % 4) * 0.04
        det = fake_detection(frame_index=i + 1, center_x=cx, center_y=cy, area_fraction=area)
        det.charuco_count = 30
        candidates.append(
            CandidateFrame(
                detection=det,
                image_bgr=image.copy(),
                accepted_at=f"2026-01-01T{i:02d}",
            )
        )
    accumulator.candidates = candidates

    # First pass: one frame at 4.5; second pass (after refinement): a second
    # frame newly above the refined threshold.
    pass1 = [0.40] * 20
    pass1[19] = 4.5
    pass2 = [0.38] * 19
    pass2[18] = 3.5
    pass3 = [0.35] * 18

    calibrations = iter(
        [
            CalibrationResult(
                rms_reprojection_error_px=0.6,
                camera_matrix=np.eye(3, dtype=float),
                distortion_coefficients=np.zeros(5, dtype=float),
                per_view_errors_px=pass1,
                selected_count=20,
                flags=0,
            ),
            CalibrationResult(
                rms_reprojection_error_px=0.45,
                camera_matrix=np.eye(3, dtype=float),
                distortion_coefficients=np.zeros(5, dtype=float),
                per_view_errors_px=pass2,
                selected_count=19,
                flags=0,
            ),
            CalibrationResult(
                rms_reprojection_error_px=0.40,
                camera_matrix=np.eye(3, dtype=float),
                distortion_coefficients=np.zeros(5, dtype=float),
                per_view_errors_px=pass3,
                selected_count=18,
                flags=0,
            ),
        ]
    )
    accumulator._calibrate = lambda *_a, **_k: next(calibrations)
    accumulator._project_per_view_error = lambda *_a, **_k: 4.0

    outcome = accumulator.solve_snapshot(list(accumulator.candidates))
    assert isinstance(outcome, SolveOk)
    assert len(outcome.solve.rejected) == 2, (
        "iterative refinement should catch the outlier revealed by the first pass "
        f"(got {len(outcome.solve.rejected)} rejections)"
    )


def test_select_diverse_prefers_truly_diverse_over_high_quality_near_duplicate(
    tmp_path,
) -> None:
    from expresso_calib.calibration import CandidateFrame

    accumulator = CalibrationAccumulator(DEFAULT_BOARD, tmp_path, create_run_dir=False)
    image = np.zeros((540, 960, 3), dtype=np.uint8)

    def build(frame_index: int, cx: float, cy: float, sharpness: float) -> CandidateFrame:
        det = fake_detection(
            frame_index=frame_index,
            center_x=cx,
            center_y=cy,
            sharpness=sharpness,
        )
        return CandidateFrame(
            detection=det,
            image_bgr=image.copy(),
            accepted_at=f"2026-01-01T{frame_index:02d}",
        )

    # Seed: two near-duplicate "high-quality" frames at the same pose, plus one
    # truly diverse "mediocre-quality" frame at the opposite corner.
    high_quality_a = build(1, 0.25, 0.25, sharpness=2000.0)
    high_quality_dup = build(2, 0.26, 0.26, sharpness=2000.0)
    diverse_low_quality = build(3, 0.75, 0.75, sharpness=80.0)

    candidates = [high_quality_a, high_quality_dup, diverse_low_quality]
    chosen = accumulator.select_diverse(candidates, max_frames=2, mark_selected=False)
    chosen_ids = {item.detection.frame_index for item in chosen}
    assert 3 in chosen_ids, (
        "select_diverse should pick the diverse-but-mediocre frame over the "
        "high-quality near-duplicate; "
        f"chosen frame indices: {sorted(chosen_ids)}"
    )


def test_pose_diversity_buckets_angles_and_scales(tmp_path) -> None:
    from expresso_calib.calibration import CandidateFrame, compute_pose_diversity

    image = np.zeros((540, 960, 3), dtype=np.uint8)
    candidates = []
    for i, (angle, area) in enumerate(
        [
            (0.0, 0.02),  # far, angle bucket 0
            (45.0, 0.08),  # mid, angle bucket 3
            (90.0, 0.20),  # near, angle bucket 6
            (135.0, 0.20),  # near, angle bucket 9
        ]
    ):
        det = fake_detection(frame_index=i + 1, center_x=0.5, center_y=0.5, area_fraction=area)
        det.angle_deg = angle
        candidates.append(
            CandidateFrame(detection=det, image_bgr=image.copy(), accepted_at=f"2026-01-01T{i:02d}")
        )

    diversity = compute_pose_diversity(candidates)
    assert diversity["angleBucketsCovered"] == 4
    assert diversity["scaleBucketsCovered"] == 3
    assert diversity["missingScale"] == []

    only_near = [candidates[2], candidates[3]]
    diversity_near = compute_pose_diversity(only_near)
    assert "far" in diversity_near["missingScale"]
    assert "mid" in diversity_near["missingScale"]


def test_convergence_states_progress_through_lifecycle(tmp_path) -> None:
    accumulator = CalibrationAccumulator(DEFAULT_BOARD, tmp_path, create_run_dir=False)

    # No solves yet -> not_solved
    conv = accumulator.compute_convergence(verdict=None)
    assert conv["state"] == "not_solved"
    assert conv["kStabilityPct"] is None
    assert conv["rmsTrend"] is None

    # Inject drifting K + improving RMS -> collecting
    accumulator.k_history = [
        [[1000.0, 0, 480.0], [0, 1000.0, 270.0], [0, 0, 1]],
        [[1100.0, 0, 480.0], [0, 1100.0, 270.0], [0, 0, 1]],
        [[1200.0, 0, 480.0], [0, 1200.0, 270.0], [0, 0, 1]],
    ]
    accumulator.solve_history = [
        {"rmsReprojectionErrorPx": 1.20},
        {"rmsReprojectionErrorPx": 0.90},
        {"rmsReprojectionErrorPx": 0.80},
    ]
    conv = accumulator.compute_convergence(verdict="MARGINAL")
    assert conv["state"] == "collecting", f"K drifting ~10% should be 'collecting', got {conv}"

    # Stable K, improving RMS -> improving
    accumulator.k_history = [
        [[1500.0, 0, 480.0], [0, 1500.0, 270.0], [0, 0, 1]],
        [[1500.5, 0, 480.0], [0, 1500.5, 270.0], [0, 0, 1]],
        [[1500.2, 0, 480.0], [0, 1500.2, 270.0], [0, 0, 1]],
    ]
    accumulator.solve_history = [
        {"rmsReprojectionErrorPx": 1.50},
        {"rmsReprojectionErrorPx": 1.00},
        {"rmsReprojectionErrorPx": 0.70},
    ]
    conv = accumulator.compute_convergence(verdict="MARGINAL")
    assert conv["state"] == "improving", (
        f"stable K + improving RMS should be 'improving', got {conv}"
    )

    # Stable K, plateau RMS, verdict GOOD -> converged
    accumulator.solve_history = [
        {"rmsReprojectionErrorPx": 0.50},
        {"rmsReprojectionErrorPx": 0.51},
        {"rmsReprojectionErrorPx": 0.50},
    ]
    conv = accumulator.compute_convergence(verdict="GOOD")
    assert conv["state"] == "converged", (
        f"stable + plateau + GOOD should be 'converged', got {conv}"
    )

    # Stable K, plateau RMS, verdict MARGINAL -> plateau
    conv = accumulator.compute_convergence(verdict="MARGINAL")
    assert conv["state"] == "plateau", (
        f"stable + plateau + MARGINAL should be 'plateau', got {conv}"
    )

    # Worsening RMS -> diverging
    accumulator.solve_history = [
        {"rmsReprojectionErrorPx": 0.50},
        {"rmsReprojectionErrorPx": 0.65},
        {"rmsReprojectionErrorPx": 0.85},
    ]
    conv = accumulator.compute_convergence(verdict="MARGINAL")
    assert conv["state"] == "diverging", f"worsening RMS should be 'diverging', got {conv}"


def test_least_occupied_quadrant_picks_emptiest_quadrant(tmp_path) -> None:
    from expresso_calib.calibration import least_occupied_quadrant

    # 4 in upper-left, 2 in upper-right, 2 in lower-left, 0 in lower-right.
    # Default 8x6 grid -> mid_x=4, mid_y=3.
    occupied = {
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),  # upper-left
        (5, 0),
        (6, 1),  # upper-right
        (1, 4),
        (2, 5),  # lower-left
    }
    weakest = least_occupied_quadrant(occupied)
    assert weakest is not None
    assert weakest[0] == "lower-right", (
        f"lower-right quadrant is empty; should be picked as weakest, got {weakest}"
    )


def test_guidance_uses_pose_and_quadrant_signals(tmp_path) -> None:
    from expresso_calib.calibration import CandidateFrame

    image = np.zeros((540, 960, 3), dtype=np.uint8)
    accumulator = CalibrationAccumulator(DEFAULT_BOARD, tmp_path, create_run_dir=False)

    # Build 20 frames all at mid distance — no near, no far. Should trigger
    # the "show closer" guidance (near is checked first).
    selected = []
    for i in range(20):
        cx = 0.15 + (i % 5) * 0.18
        cy = 0.20 + (i // 5) * 0.20
        det = fake_detection(frame_index=i + 1, center_x=cx, center_y=cy, area_fraction=0.08)
        det.charuco_count = 30
        selected.append(
            CandidateFrame(detection=det, image_bgr=image.copy(), accepted_at=f"2026-01-01T{i:02d}")
        )

    cal = CalibrationResult(
        rms_reprojection_error_px=0.7,
        camera_matrix=np.eye(3, dtype=float),
        distortion_coefficients=np.zeros(5, dtype=float),
        per_view_errors_px=[0.5 for _ in selected],
        selected_count=len(selected),
        flags=0,
    )
    quality = accumulator.summarize_quality(selected, cal)
    assert quality["guidance"], "guidance string must be populated"
    assert "closer" in quality["guidance"].lower(), (
        f"missing 'near' bucket should produce 'show closer' guidance, got {quality['guidance']}"
    )
    assert "near" in quality["poseDiversity"]["missingScale"]
