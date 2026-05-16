from __future__ import annotations

import asyncio

import numpy as np
import pytest

from expresso_calib.calibration import CalibrationResult, CalibrationSolveResult
from expresso_calib.detection import DetectionResult
from expresso_calib.server import ManagedCamera, MetricsHub, SolveJob, _put_latest


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
        rejected=[],
        rejected_per_view_errors_px=[],
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


@pytest.mark.asyncio
async def test_metrics_hub_slow_client_does_not_stall_fast_client() -> None:
    hub = MetricsHub()

    class FakeClient:
        def __init__(self, name: str, delay: float) -> None:
            self.name = name
            self.delay = delay
            self.received: list[str] = []

        async def accept(self) -> None:
            pass

        async def send_text(self, payload: str) -> None:
            await asyncio.sleep(self.delay)
            self.received.append(payload)

    slow = FakeClient("slow", delay=0.5)
    fast = FakeClient("fast", delay=0.0)
    await hub.connect(slow)  # type: ignore[arg-type]
    await hub.connect(fast)  # type: ignore[arg-type]

    start = asyncio.get_event_loop().time()
    await asyncio.wait_for(hub.broadcast({"n": 1}), timeout=0.1)
    elapsed = asyncio.get_event_loop().time() - start

    assert elapsed < 0.1, f"broadcast blocked for {elapsed:.3f}s on slow client"
    await asyncio.sleep(0.05)
    assert fast.received == ['{"n": 1}']


@pytest.mark.asyncio
async def test_remove_camera_stops_before_popping(tmp_path) -> None:
    from expresso_calib.server import MultiCameraCalibrationState

    live = MultiCameraCalibrationState()

    class TrackingCamera:
        def __init__(self) -> None:
            self.stopped = False
            self.id = "cam-1"
            self.label = "test"
            self.accumulator = None

        async def stop(self) -> None:
            assert "cam-1" in live.cameras, (
                "camera was popped before stop() — broadcasts may still hit a half-torn-down camera"
            )
            self.stopped = True

    live.cameras["cam-1"] = TrackingCamera()  # type: ignore[assignment]

    removed = await live.remove_camera("cam-1")
    assert removed is True
    assert live.cameras == {}


@pytest.mark.asyncio
async def test_lifespan_stops_all_cameras_on_shutdown() -> None:
    from expresso_calib.server import create_app

    app = create_app()
    stopped = asyncio.Event()

    class TrackingCamera:
        def __init__(self) -> None:
            self.id = "cam-1"
            self.label = "test"

        async def stop(self) -> None:
            stopped.set()

        def public_snapshot(self, _now: float) -> dict:
            return {"id": self.id, "label": self.label}

    async with app.router.lifespan_context(app):
        app.state.live.cameras["cam-1"] = TrackingCamera()  # type: ignore[assignment]

    assert stopped.is_set()


def test_oversized_post_body_is_rejected() -> None:
    from fastapi.testclient import TestClient
    from expresso_calib.server import create_app, MAX_REQUEST_BODY_BYTES

    app = create_app()
    with TestClient(app) as client:
        oversized = b"x" * (MAX_REQUEST_BODY_BYTES + 1)
        response = client.post(
            "/api/cameras",
            content=oversized,
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 413


def test_mjpeg_oversized_content_length_is_dropped() -> None:
    from expresso_calib.server import MjpegCapture, MJPEG_MAX_FRAME_BYTES

    capture = MjpegCapture.__new__(MjpegCapture)
    capture.url = "fake"
    capture.buffer = bytearray()
    capture.opened = True
    capture.boundary = b"--frame"

    class FakeResponse:
        def __init__(self) -> None:
            self.lines = iter([
                b"--frame\r\n",
                f"content-length: {MJPEG_MAX_FRAME_BYTES + 1}\r\n".encode(),
                b"\r\n",
            ])

        def readline(self) -> bytes:
            return next(self.lines, b"")

        def read(self, n: int) -> bytes:
            raise AssertionError(
                f"read({n}) called for oversized declared content-length"
            )

    capture.response = FakeResponse()
    ok, _ = capture._read_multipart_frame()
    assert ok is False
    assert capture.opened is False


def test_public_snapshot_emits_rms_thresholds(tmp_path) -> None:
    from expresso_calib.server import MultiCameraCalibrationState, RMS_GOOD_MAX_PX

    live = MultiCameraCalibrationState()
    camera = live.add_camera("test", "http://example.invalid/stream.mjpg")
    snapshot = camera.public_snapshot(now=0.0)
    assert snapshot["rmsThresholds"]["goodMaxPx"] == RMS_GOOD_MAX_PX
    assert snapshot["rmsThresholds"]["marginalMaxPx"] > snapshot["rmsThresholds"]["goodMaxPx"]
    assert snapshot["rmsThresholds"]["poorP95MaxPx"] > snapshot["rmsThresholds"]["marginalMaxPx"]
