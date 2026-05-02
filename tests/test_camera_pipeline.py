from __future__ import annotations

import asyncio

import numpy as np

from expresso_calib.calibration import CalibrationResult, CalibrationSolveResult
from expresso_calib.detection import DetectionResult
from expresso_calib.server import ManagedCamera, SolveJob, _put_latest


class DummyManager:
    def __init__(self, tmp_path) -> None:
        self.session_dir = tmp_path
        self.target_metadata = {}
        self.started_at = 0.0

    async def broadcast(self) -> None:
        return None


def fake_detection(frame_index: int, center_x: float, center_y: float) -> DetectionResult:
    width = 960
    height = 540
    side = (0.10 * width * height) ** 0.5
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
        sharpness=120.0,
        corners=points,
        ids=ids,
        center_x=center_x,
        center_y=center_y,
        bbox_width=side / width,
        bbox_height=side / height,
        area_fraction=0.10,
        angle_deg=0.0,
    )


def test_put_latest_replaces_stale_queue_item() -> None:
    queue = asyncio.Queue(maxsize=1)

    assert _put_latest(queue, "old") == 0
    assert _put_latest(queue, "new") == 1

    assert queue.qsize() == 1
    assert queue.get_nowait() == "new"
    queue.task_done()


def test_camera_ignores_stale_solve_result(tmp_path) -> None:
    camera = ManagedCamera(
        DummyManager(tmp_path),
        "cam-test",
        "Camera",
        "http://example.test/stream.mjpg",
    )
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    for index, center in enumerate(
        [(0.2, 0.2), (0.7, 0.2), (0.2, 0.7), (0.7, 0.7)], start=1
    ):
        accepted, _ = camera.accumulator.observe(
            fake_detection(index, center[0], center[1]), image
        )
        assert accepted is True

    calibration = CalibrationResult(
        rms_reprojection_error_px=0.72,
        camera_matrix=np.eye(3, dtype=float),
        distortion_coefficients=np.zeros(5, dtype=float),
        per_view_errors_px=[0.5 for _ in camera.accumulator.candidates],
        selected_count=len(camera.accumulator.candidates),
        flags=0,
    )
    result = CalibrationSolveResult(
        calibration=calibration,
        quality={"usableFrames": len(camera.accumulator.candidates)},
        selected=list(camera.accumulator.candidates),
        candidate_count=len(camera.accumulator.candidates),
    )
    job = SolveJob(
        generation=camera.generation,
        candidates=list(camera.accumulator.candidates),
        consumed_new_frames=camera.accumulator.accepted_since_solve,
    )

    camera.generation += 1

    assert camera._commit_solve_result(job, result) is False
    assert camera.accumulator.last_calibration is None


def test_camera_can_queue_followup_solve_while_solver_finishes(tmp_path) -> None:
    camera = ManagedCamera(
        DummyManager(tmp_path),
        "cam-test",
        "Camera",
        "http://example.test/stream.mjpg",
    )
    camera.accumulator.min_solve_frames = 2
    camera.accumulator.solve_every_new_frames = 1
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    for index, center in enumerate([(0.2, 0.2), (0.7, 0.7)], start=1):
        accepted, _ = camera.accumulator.observe(
            fake_detection(index, center[0], center[1]), image
        )
        assert accepted is True

    camera.solver_running = True

    assert camera._enqueue_solve_if_due(camera.generation) is False
    assert camera._enqueue_solve_if_due(
        camera.generation, allow_while_running=True
    ) is True
    assert camera.solve_queue.qsize() == 1
