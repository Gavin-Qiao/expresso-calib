from __future__ import annotations

import asyncio
import json
import socket
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import qrcode
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from .board import DEFAULT_BOARD, target_pdf_bytes, target_png_bytes
from .calibration import CalibrationAccumulator
from .calibration_worker import (
    DETECTION_FPS,
    CalibrationWorker,
    DetectionJob,
    ScreenshotJob,
    SolveJob,
    _put_latest,
)
from .camera_pipeline import (
    FRAME_STALE_SEC,
    MJPEG_MAX_FRAME_BYTES,
    PREVIEW_STREAM_FPS,
    CameraPipeline,
    MjpegCapture,
)
from .multi_camera import FocusTracker, clean_label, slugify_label

__all__ = [
    "DETECTION_FPS",
    "DetectionJob",
    "FRAME_STALE_SEC",
    "MAX_REQUEST_BODY_BYTES",
    "MJPEG_MAX_FRAME_BYTES",
    "ManagedCamera",
    "MetricsHub",
    "MjpegCapture",
    "MultiCameraCalibrationState",
    "PREVIEW_STREAM_FPS",
    "RMS_GOOD_MAX_PX",
    "RMS_MARGINAL_MAX_PX",
    "RMS_POOR_P95_MAX_PX",
    "ScreenshotJob",
    "SolveJob",
    "_put_latest",
    "create_app",
    "main",
]

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = Path(__file__).resolve().parent / "web"
RUNS_DIR = ROOT / "runs"
PORT = 3987
MAX_REQUEST_BODY_BYTES = 64 * 1024
DEFAULT_CAMERA_URL = "http://127.0.0.1:3988/stream.mjpg"


class MetricsHub:
    def __init__(self) -> None:
        self.clients: dict[Any, asyncio.Queue[str]] = {}
        self.latest: dict[str, Any] | None = None
        self._tasks: dict[Any, asyncio.Task[None]] = {}

    async def connect(self, websocket: Any) -> None:
        await websocket.accept()
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=4)
        self.clients[websocket] = queue
        self._tasks[websocket] = asyncio.create_task(self._pump(websocket, queue))
        if self.latest is not None:
            queue.put_nowait(json.dumps(self.latest))

    def disconnect(self, websocket: Any) -> None:
        self.clients.pop(websocket, None)
        task = self._tasks.pop(websocket, None)
        if task is not None:
            task.cancel()

    async def broadcast(self, payload: dict[str, Any]) -> None:
        self.latest = payload
        encoded = json.dumps(payload)
        for queue in list(self.clients.values()):
            while True:
                try:
                    queue.put_nowait(encoded)
                    break
                except asyncio.QueueFull:
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

    async def _pump(self, websocket: Any, queue: asyncio.Queue[str]) -> None:
        try:
            while True:
                encoded = await queue.get()
                try:
                    await websocket.send_text(encoded)
                except Exception:
                    self.disconnect(websocket)
                    return
        except asyncio.CancelledError:
            return


def _task_running(task: asyncio.Task[Any] | None) -> bool:
    return task is not None and not task.done()


class ManagedCamera:
    """Thin composer that wires a CameraPipeline into a CalibrationWorker.

    The pipeline owns capture + preview + the MJPEG client and produces
    decoded BGR frames. The worker owns the calibration accumulator and
    runs detection + solve + screenshot loops driven by those frames.
    ManagedCamera holds the manager reference for broadcasting and
    shapes the public snapshot.
    """

    def __init__(
        self,
        manager: MultiCameraCalibrationState,
        camera_id: str,
        label: str,
        url: str,
    ) -> None:
        self.manager = manager
        self.id = camera_id
        self.label = clean_label(label, fallback=camera_id)
        self.url = url
        accumulator = self._new_accumulator(manager.session_dir, manager.target_metadata)
        screenshot_dir = manager.session_dir / "screenshots" / slugify_label(self.label, self.id)
        self.pipeline = CameraPipeline(url)
        self.worker = CalibrationWorker(
            accumulator=accumulator,
            screenshot_dir=screenshot_dir,
            broadcast=manager.broadcast,
            source_id=self.id,
            wall_clock_start=manager.started_at,
        )
        self.pipeline.set_broadcast(manager.broadcast)
        self.pipeline.on_frame(self.worker.handle_frame)

    def _new_accumulator(
        self, session_dir: Path, target_metadata: dict[str, Any]
    ) -> CalibrationAccumulator:
        accumulator = CalibrationAccumulator(
            DEFAULT_BOARD,
            session_dir / "ephemeral" / self.id,
            auto_export=False,
            create_run_dir=False,
        )
        accumulator.source_id = self.id
        accumulator.target_metadata = target_metadata
        return accumulator

    # --- proxy properties used by tests, routes, and snapshot shaping ---

    @property
    def accumulator(self) -> CalibrationAccumulator:
        return self.worker.accumulator

    @property
    def generation(self) -> int:
        return self.worker.generation

    @generation.setter
    def generation(self, value: int) -> None:
        self.worker.generation = value

    @property
    def last_error(self) -> str | None:
        return self.worker.last_error or self.pipeline.last_error

    @property
    def last_screenshot_path(self) -> str | None:
        return self.worker.last_screenshot_path

    @property
    def latest_detection(self) -> Any:
        return self.worker.latest_detection

    @property
    def latest_detection_wall_time(self) -> float | None:
        return self.worker.latest_detection_wall_time

    @property
    def latest_jpeg(self) -> bytes | None:
        return self.pipeline.latest_jpeg

    @property
    def latest_jpeg_seq(self) -> int:
        return self.pipeline.latest_jpeg_seq

    @property
    def last_frame_at(self) -> float | None:
        return self.pipeline.last_frame_at

    @property
    def frames_seen(self) -> int:
        return self.pipeline.frames_seen

    @property
    def running(self) -> bool:
        return self.pipeline.running or self.worker.running

    @property
    def solver_running(self) -> bool:
        return self.worker.solver_running

    @solver_running.setter
    def solver_running(self, value: bool) -> None:
        self.worker.solver_running = value

    @property
    def detection_running(self) -> bool:
        return self.worker.detection_running

    @property
    def screenshot_running(self) -> bool:
        return self.worker.screenshot_running

    @property
    def detection_queue(self) -> asyncio.Queue[DetectionJob]:
        return self.worker.detection_queue

    @property
    def solve_queue(self) -> asyncio.Queue[SolveJob]:
        return self.worker.solve_queue

    @property
    def screenshot_queue(self) -> asyncio.Queue[ScreenshotJob]:
        return self.worker.screenshot_queue

    @property
    def dropped_detection_frames(self) -> int:
        return self.worker.dropped_detection_frames

    @property
    def dropped_screenshot_jobs(self) -> int:
        return self.worker.dropped_screenshot_jobs

    # --- proxy methods used by tests ---

    def _enqueue_solve_if_due(self, generation: int, *, allow_while_running: bool = False) -> bool:
        return self.worker._enqueue_solve_if_due(
            generation, allow_while_running=allow_while_running
        )

    def _commit_solve_result(self, job: SolveJob, result: Any) -> bool:
        return self.worker._commit_solve_result(job, result)

    # --- lifecycle ---

    async def start(self) -> None:
        if self.running:
            return
        await self.worker.start()
        await self.pipeline.start()
        await self.manager.broadcast()

    async def stop(self) -> None:
        await self.pipeline.stop()
        await self.worker.stop()
        await self.manager.broadcast()

    def reset_calibration(self, session_dir: Path, target_metadata: dict[str, Any]) -> None:
        accumulator = self._new_accumulator(session_dir, target_metadata)
        screenshot_dir = session_dir / "screenshots" / slugify_label(self.label, self.id)
        self.worker.reset(accumulator=accumulator, screenshot_dir=screenshot_dir)

    # --- view helpers ---

    def fps(self) -> float:
        return self.pipeline.fps()

    def has_fresh_preview(self, now: float) -> bool:
        return self.pipeline.has_fresh_preview(now)

    def connected(self, now: float) -> bool:
        return self.pipeline.connected(now)

    def detecting_charuco(self, now: float) -> bool:
        detection = self.latest_detection
        wall_time = self.latest_detection_wall_time
        if detection is None or wall_time is None:
            return False
        if self.last_frame_at is None or now - self.last_frame_at > FRAME_STALE_SEC:
            return False
        if now - wall_time > 1.25:
            return False
        return detection.charuco_count >= self.accumulator.min_corners

    def rms_value(self) -> float | None:
        calibration = self.accumulator.last_calibration
        if calibration is None:
            return None
        return float(calibration.rms_reprojection_error_px)

    def public_snapshot(self, now: float) -> dict[str, Any]:
        rms = self.rms_value()
        solve_pool_stats = self.accumulator.solve_pool_stats()
        selected_frames = (
            self.accumulator.last_calibration.selected_count
            if self.accumulator.last_calibration
            else 0
        )
        detection = self.latest_detection.to_public_dict() if self.latest_detection else None
        return {
            "id": self.id,
            "generation": self.generation,
            "label": self.label,
            "url": self.url,
            "running": self.running,
            "connected": self.connected(now),
            "framesSeen": self.frames_seen,
            "fps": self.fps(),
            "lastFrameAt": self.last_frame_at,
            "lastError": self.last_error,
            "hasLatestFrame": self.has_fresh_preview(now),
            "detectingCharuco": self.detecting_charuco(now),
            "detection": detection,
            "minCandidateCorners": self.accumulator.min_corners,
            "minSolveFrames": self.accumulator.min_solve_frames,
            "minSolveCorners": self.accumulator.min_solve_corners,
            "maxCalibrationFrames": self.accumulator.max_calib_frames,
            "lastAcceptReason": self.accumulator.last_accept_reason,
            "candidateFrames": len(self.accumulator.candidates),
            "solvePoolFrames": solve_pool_stats["solvePoolFrames"],
            "weakSolveFrames": solve_pool_stats["weakSolveFrames"],
            "usingStrongSolvePool": solve_pool_stats["usingStrongSolvePool"],
            "selectedFrames": selected_frames,
            "calculationFrames": selected_frames,
            "acceptedSinceSolve": self.accumulator.accepted_since_solve,
            "duplicatePoseFrames": self.accumulator.duplicate_pose_rejections,
            "duplicateImageFrames": self.accumulator.duplicate_image_rejections,
            "rejectedFrames": sum(1 for item in self.accumulator.candidates if item.rejected),
            "quality": self.accumulator.last_quality,
            "solveHistory": self.accumulator.solve_history[-6:],
            "solveDue": self.accumulator.should_solve(),
            "rms": rms,
            "rmsDisplay": "--" if rms is None else f"{rms:.2f}",
            "errorGrade": rms_grade(rms),
            "errorColor": rms_color(rms),
            "rmsThresholds": {
                "goodMaxPx": RMS_GOOD_MAX_PX,
                "marginalMaxPx": RMS_MARGINAL_MAX_PX,
                "poorP95MaxPx": RMS_POOR_P95_MAX_PX,
            },
            "lastScreenshotPath": self.last_screenshot_path,
            "pipeline": {
                "captureRunning": _task_running(self.pipeline.capture_task),
                "previewRunning": _task_running(self.pipeline.preview_task),
                "detectionRunning": self.detection_running,
                "solverRunning": self.solver_running,
                "screenshotRunning": self.screenshot_running,
                "detectionQueueDepth": self.detection_queue.qsize(),
                "solveQueueDepth": self.solve_queue.qsize(),
                "screenshotQueueDepth": self.screenshot_queue.qsize(),
                "droppedDetectionFrames": self.dropped_detection_frames,
                "droppedScreenshotJobs": self.dropped_screenshot_jobs,
                "previewFpsLimit": PREVIEW_STREAM_FPS,
                "detectionFpsLimit": DETECTION_FPS,
            },
        }


class MultiCameraCalibrationState:
    def __init__(self) -> None:
        self.hub = MetricsHub()
        self.started_at = time.monotonic()
        self.session_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.cameras: dict[str, ManagedCamera] = {}
        self.next_camera_id = 1
        self.focus = FocusTracker()
        self.target_metadata: dict[str, Any] = {}
        self.add_camera("MacBook", DEFAULT_CAMERA_URL)

    def add_camera(self, label: str, url: str) -> ManagedCamera:
        clean_url = _clean_camera_url(url)
        if _is_self_preview_stream(clean_url):
            raise ValueError("Use the upstream camera stream URL, not this app's preview proxy.")
        camera_id = f"cam-{self.next_camera_id}"
        self.next_camera_id += 1
        camera = ManagedCamera(self, camera_id, label, clean_url)
        self.cameras[camera_id] = camera
        self.focus.clear_if_removed(set(self.cameras))
        return camera

    async def remove_camera(self, camera_id: str) -> bool:
        camera = self.cameras.get(camera_id)
        if camera is None:
            return False
        await camera.stop()
        self.cameras.pop(camera_id, None)
        self.focus.clear_if_removed(set(self.cameras))
        await self.broadcast()
        return True

    async def start_all(self) -> None:
        for camera in list(self.cameras.values()):
            await camera.start()
        await self.broadcast()

    async def stop_all(self) -> None:
        for camera in list(self.cameras.values()):
            await camera.stop()
        self.focus = FocusTracker()
        await self.broadcast()

    async def update_target_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        clean = {
            "model": str(metadata.get("model") or "unknown/manual")[:120],
            "screen_preset_id": str(metadata.get("screen_preset_id") or "")[:80],
            "screen_diagonal_in": _optional_float(metadata.get("screen_diagonal_in")),
            "screen_width_mm": _optional_float(metadata.get("screen_width_mm")),
            "screen_height_mm": _optional_float(metadata.get("screen_height_mm")),
            "screen_width_px": _optional_int(metadata.get("screen_width_px")),
            "screen_height_px": _optional_int(metadata.get("screen_height_px")),
            "device_pixel_ratio": _optional_float(metadata.get("device_pixel_ratio")),
            "user_agent": str(metadata.get("user_agent") or "")[:500],
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.target_metadata = clean
        for camera in self.cameras.values():
            camera.accumulator.target_metadata = clean
        await self.broadcast()
        return clean

    async def reset(self) -> dict[str, Any]:
        self.started_at = time.monotonic()
        self.session_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        for camera in self.cameras.values():
            camera.reset_calibration(self.session_dir, self.target_metadata)
        self.focus = FocusTracker()
        await self.broadcast()
        return self.metrics()

    async def broadcast(self) -> None:
        await self.hub.broadcast(self.metrics())

    def metrics(self) -> dict[str, Any]:
        now = time.time()
        camera_payload = [camera.public_snapshot(now) for camera in self.cameras.values()]
        focused_camera_id = self.focus.update(camera_payload, now)
        return {
            "serverTime": now,
            "sessionDir": str(self.session_dir),
            "focusedCameraId": focused_camera_id,
            "cameras": camera_payload,
            "targetMetadata": self.target_metadata,
        }


RMS_GOOD_MAX_PX = 0.80
RMS_MARGINAL_MAX_PX = 1.20
RMS_POOR_P95_MAX_PX = 1.80


def rms_grade(rms: float | None) -> str:
    if rms is None:
        return "pending"
    if rms <= RMS_GOOD_MAX_PX:
        return "good"
    if rms <= RMS_MARGINAL_MAX_PX:
        return "marginal"
    return "poor"


def rms_color(rms: float | None) -> str:
    if rms is None:
        return "rgba(255,255,255,0.36)"
    stops = [
        (RMS_GOOD_MAX_PX, (22, 163, 74)),
        (RMS_MARGINAL_MAX_PX, (234, 179, 8)),
        (RMS_POOR_P95_MAX_PX, (220, 38, 38)),
    ]
    if rms <= stops[0][0]:
        rgb = stops[0][1]
    elif rms >= stops[-1][0]:
        rgb = stops[-1][1]
    elif rms <= stops[1][0]:
        rgb = _interpolate_rgb(stops[0], stops[1], rms)
    else:
        rgb = _interpolate_rgb(stops[1], stops[2], rms)
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"


def _interpolate_rgb(
    left: tuple[float, tuple[int, int, int]],
    right: tuple[float, tuple[int, int, int]],
    value: float,
) -> tuple[int, int, int]:
    span = max(1e-9, right[0] - left[0])
    t = (value - left[0]) / span
    return tuple(
        int(round(left[1][index] + (right[1][index] - left[1][index]) * t)) for index in range(3)
    )


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.live = MultiCameraCalibrationState()
        try:
            yield
        finally:
            await app.state.live.stop_all()

    app = FastAPI(title="Expresso Calib", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def cap_body_size(request: Request, call_next):
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > MAX_REQUEST_BODY_BYTES:
                    return JSONResponse({"detail": "Request body too large."}, status_code=413)
            except ValueError:
                return JSONResponse({"detail": "Invalid Content-Length."}, status_code=400)
        return await call_next(request)

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/operator")

    @app.get("/operator", include_in_schema=False)
    async def operator_page() -> FileResponse:
        return FileResponse(WEB_DIR / "operator.html")

    @app.get("/target", include_in_schema=False)
    async def target_page() -> FileResponse:
        return FileResponse(WEB_DIR / "target.html")

    @app.get("/manifest.webmanifest", include_in_schema=False)
    async def manifest() -> JSONResponse:
        return JSONResponse(
            {
                "name": "Expresso Calib Target",
                "short_name": "Expresso Target",
                "start_url": "/target",
                "scope": "/",
                "display": "fullscreen",
                "orientation": "landscape",
                "background_color": "#ffffff",
                "theme_color": "#ffffff",
            },
            media_type="application/manifest+json",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/session")
    async def session(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "targetUrl": target_url_for_request(request),
                "operatorUrl": str(request.url_for("operator_page")),
                "defaultCameraUrl": DEFAULT_CAMERA_URL,
                "board": DEFAULT_BOARD.manifest(),
                "lanIp": local_lan_ip(),
            }
        )

    @app.get("/api/qr.png")
    async def qr_png(request: Request) -> Response:
        image = qrcode.make(target_url_for_request(request))
        return Response(_pil_png_bytes(image), media_type="image/png")

    @app.get("/api/target.png")
    async def target_png(w: int = 1800, h: int = 1200) -> Response:
        return Response(target_png_bytes(DEFAULT_BOARD, w, h), media_type="image/png")

    @app.get("/api/target.pdf")
    async def target_pdf(
        width_mm: float = 280.6,
        height_mm: float = 194.7,
        landscape: bool = True,
    ) -> Response:
        w = float(width_mm)
        h = float(height_mm)
        if landscape and h > w:
            w, h = h, w
        return Response(
            target_pdf_bytes(DEFAULT_BOARD, w, h),
            media_type="application/pdf",
            headers={
                "Cache-Control": "no-store",
                "Content-Disposition": 'inline; filename="expresso-charuco-target.pdf"',
            },
        )

    @app.get("/api/status")
    async def status(request: Request) -> JSONResponse:
        return JSONResponse(request.app.state.live.metrics())

    @app.post("/api/reset")
    async def reset(request: Request) -> JSONResponse:
        return JSONResponse(await request.app.state.live.reset())

    @app.post("/api/target-metadata")
    async def target_metadata(request: Request) -> JSONResponse:
        body = await request.json()
        return JSONResponse(await request.app.state.live.update_target_metadata(body))

    @app.get("/api/cameras")
    async def list_cameras(request: Request) -> JSONResponse:
        return JSONResponse(request.app.state.live.metrics())

    @app.post("/api/cameras")
    async def add_camera(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            camera = request.app.state.live.add_camera(
                str(body.get("label") or ""), str(body.get("url") or "")
            )
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        await request.app.state.live.broadcast()
        return JSONResponse({"ok": True, "camera": camera.public_snapshot(time.time())})

    @app.delete("/api/cameras/{camera_id}")
    async def remove_camera(camera_id: str, request: Request) -> JSONResponse:
        removed = await request.app.state.live.remove_camera(camera_id)
        return JSONResponse({"ok": removed}, status_code=200 if removed else 404)

    @app.post("/api/cameras/start-all")
    async def start_all(request: Request) -> JSONResponse:
        await request.app.state.live.start_all()
        return JSONResponse({"ok": True, "metrics": request.app.state.live.metrics()})

    @app.post("/api/cameras/stop-all")
    async def stop_all(request: Request) -> JSONResponse:
        await request.app.state.live.stop_all()
        return JSONResponse({"ok": True, "metrics": request.app.state.live.metrics()})

    @app.get("/api/cameras/{camera_id}/latest.jpg")
    async def camera_latest(camera_id: str, request: Request) -> Response:
        camera = request.app.state.live.cameras.get(camera_id)
        if camera is None or not camera.has_fresh_preview(time.time()):
            return Response(status_code=204)
        return Response(
            camera.latest_jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/cameras/{camera_id}/stream.mjpg")
    async def camera_stream(camera_id: str, request: Request) -> StreamingResponse:
        async def frames() -> Any:
            last_seq = -1
            boundary = b"--frame\r\n"
            stream_generation: int | None = None
            while True:
                if await request.is_disconnected():
                    break
                camera = request.app.state.live.cameras.get(camera_id)
                if camera is None:
                    break
                if stream_generation is None:
                    stream_generation = camera.generation
                if camera.generation != stream_generation or not camera.running:
                    break
                latest = camera.latest_jpeg
                if (
                    latest is not None
                    and camera.has_fresh_preview(time.time())
                    and camera.latest_jpeg_seq != last_seq
                ):
                    last_seq = camera.latest_jpeg_seq
                    yield (
                        boundary
                        + b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(latest)}\r\n\r\n".encode("ascii")
                        + latest
                        + b"\r\n"
                    )
                await asyncio.sleep(1.0 / PREVIEW_STREAM_FPS)

        return StreamingResponse(
            frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache",
            },
        )

    @app.websocket("/ws/metrics")
    async def ws_metrics(websocket: WebSocket) -> None:
        hub = websocket.app.state.live.hub
        await hub.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            hub.disconnect(websocket)

    return app


def target_url_for_request(request: Request) -> str:
    host = request.headers.get("host") or f"127.0.0.1:{PORT}"
    hostname, _, port = host.partition(":")
    if hostname in {"127.0.0.1", "localhost", "::1"}:
        host = f"{local_lan_ip()}:{port or PORT}"
    return f"http://{host}/target"


def _clean_camera_url(url: str) -> str:
    clean = url.strip()
    parsed = urlparse(clean)
    if parsed.scheme not in {"http", "https", "rtsp"} or not parsed.netloc:
        raise ValueError("Camera URL must be an absolute http, https, or rtsp URL.")
    return clean


def _is_self_preview_stream(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    self_hosts = {"127.0.0.1", "localhost", "::1", local_lan_ip()}
    return host in self_hosts and port == PORT and parsed.path.startswith("/api/cameras/")


def local_lan_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        except OSError:
            return socket.gethostbyname(socket.gethostname())


def _pil_png_bytes(image: Any) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def main() -> None:
    uvicorn.run(
        "expresso_calib.server:create_app",
        factory=True,
        host="0.0.0.0",
        port=PORT,
    )
