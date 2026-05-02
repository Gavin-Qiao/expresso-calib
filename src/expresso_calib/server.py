from __future__ import annotations

import asyncio
import json
import socket
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import qrcode
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .board import DEFAULT_BOARD, target_pdf_bytes, target_png_bytes
from .calibration import CalibrationAccumulator
from .detection import CharucoDetector, Frame
from .multi_camera import FocusTracker, clean_label, slugify_label

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = Path(__file__).resolve().parent / "web"
RUNS_DIR = ROOT / "runs"
PORT = 3987
PREVIEW_STREAM_FPS = 30
DETECTION_FPS = 6
DEFAULT_CAMERA_URL = "http://127.0.0.1:3988/stream.mjpg"


class MetricsHub:
    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.latest: dict[str, Any] | None = None

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.clients.add(websocket)
        if self.latest is not None:
            await websocket.send_text(json.dumps(self.latest))

    def disconnect(self, websocket: WebSocket) -> None:
        self.clients.discard(websocket)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        self.latest = payload
        dead: list[WebSocket] = []
        encoded = json.dumps(payload)
        for client in list(self.clients):
            try:
                await client.send_text(encoded)
            except Exception:
                dead.append(client)
        for client in dead:
            self.disconnect(client)


class ManagedCamera:
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
        self.detector = CharucoDetector(DEFAULT_BOARD)
        self.accumulator = CalibrationAccumulator(
            DEFAULT_BOARD,
            manager.session_dir / "ephemeral" / camera_id,
            auto_export=False,
            create_run_dir=False,
        )
        self.accumulator.source_id = camera_id
        self.accumulator.target_metadata = manager.target_metadata

        self.running = False
        self.task: asyncio.Task[None] | None = None
        self.last_error: str | None = None
        self.frames_seen = 0
        self.frame_index = 0
        self.started_at: float | None = None
        self.last_frame_at: float | None = None
        self.frame_times: deque[float] = deque()
        self.latest_jpeg: bytes | None = None
        self.latest_jpeg_seq = 0
        self.latest_detection: Any = None
        self.latest_detection_wall_time: float | None = None
        self.detection_running = False
        self.last_detection_started_at = 0.0
        self.last_screenshot_path: str | None = None
        self.last_broadcast_at = 0.0

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.last_error = None
        self.started_at = time.time()
        self.task = asyncio.create_task(self._run(), name=f"camera-source-{self.id}")

    async def stop(self) -> None:
        self.running = False
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

    async def _run(self) -> None:
        capture = await asyncio.to_thread(cv2.VideoCapture, self.url)
        if capture is None or not capture.isOpened():
            self.last_error = f"Could not open camera URL: {self.url}"
            self.running = False
            await self.manager.broadcast()
            return

        try:
            while self.running:
                ok, frame = await asyncio.to_thread(capture.read)
                if not ok or frame is None:
                    self.last_error = "Camera URL did not return a frame."
                    await self._maybe_broadcast()
                    await asyncio.sleep(0.15)
                    continue

                self.last_error = None
                self.frames_seen += 1
                now = time.time()
                self.last_frame_at = now
                self.frame_times.append(now)
                while self.frame_times and now - self.frame_times[0] > 2.0:
                    self.frame_times.popleft()

                encode_ok, encoded = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78]
                )
                if encode_ok:
                    self.latest_jpeg = encoded.tobytes()
                    self.latest_jpeg_seq += 1

                if self._should_detect(now):
                    self.detection_running = True
                    self.last_detection_started_at = now
                    self.frame_index += 1
                    timestamp = time.monotonic() - self.manager.started_at
                    asyncio.create_task(
                        self._process_detection(frame.copy(), self.frame_index, timestamp),
                        name=f"charuco-detection-{self.id}",
                    )

                await self._maybe_broadcast()
                await asyncio.sleep(0)
        finally:
            await asyncio.to_thread(capture.release)

    def _should_detect(self, now: float) -> bool:
        return (
            not self.detection_running
            and now - self.last_detection_started_at >= 1.0 / DETECTION_FPS
        )

    async def _process_detection(
        self, image_bgr: Any, frame_index: int, timestamp_sec: float
    ) -> None:
        try:
            frame = Frame(
                index=frame_index,
                timestamp_sec=timestamp_sec,
                image_bgr=image_bgr,
                source_id=self.id,
            )
            detection = await asyncio.to_thread(self.detector.detect, frame)
            accepted, _ = self.accumulator.observe(detection, image_bgr)
            self.latest_detection = detection
            self.latest_detection_wall_time = time.time()
            if accepted:
                self._save_latest_candidate_screenshot()
                self.accumulator.solve_if_due()
        except Exception as exc:
            self.last_error = f"Detection failed: {exc}"
        finally:
            self.detection_running = False
            await self.manager.broadcast()

    def _save_latest_candidate_screenshot(self) -> None:
        if not self.accumulator.candidates:
            return
        item = self.accumulator.candidates[-1]
        camera_dir = self.manager.session_dir / "screenshots" / slugify_label(
            self.label, self.id
        )
        self.last_screenshot_path = str(
            self.accumulator.write_candidate_screenshot(item, camera_dir)
        )

    async def _maybe_broadcast(self) -> None:
        now = time.time()
        if now - self.last_broadcast_at >= 0.20:
            self.last_broadcast_at = now
            await self.manager.broadcast()

    def fps(self) -> float:
        if len(self.frame_times) < 2:
            return 0.0
        elapsed = self.frame_times[-1] - self.frame_times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.frame_times) - 1) / elapsed

    def detecting_charuco(self, now: float) -> bool:
        detection = self.latest_detection
        if detection is None or self.latest_detection_wall_time is None:
            return False
        if now - self.latest_detection_wall_time > 1.25:
            return False
        return detection.charuco_count >= self.accumulator.min_corners

    def rms_value(self) -> float | None:
        calibration = self.accumulator.last_calibration
        if calibration is None:
            return None
        return float(calibration.rms_reprojection_error_px)

    def public_snapshot(self, now: float) -> dict[str, Any]:
        rms = self.rms_value()
        detection = (
            self.latest_detection.to_public_dict() if self.latest_detection else None
        )
        return {
            "id": self.id,
            "label": self.label,
            "url": self.url,
            "running": self.running,
            "connected": self.running and self.last_frame_at is not None,
            "framesSeen": self.frames_seen,
            "fps": self.fps(),
            "lastFrameAt": self.last_frame_at,
            "lastError": self.last_error,
            "hasLatestFrame": self.latest_jpeg is not None,
            "detectingCharuco": self.detecting_charuco(now),
            "detection": detection,
            "candidateFrames": len(self.accumulator.candidates),
            "selectedFrames": self.accumulator.last_calibration.selected_count
            if self.accumulator.last_calibration
            else 0,
            "rms": rms,
            "rmsDisplay": "--" if rms is None else f"{rms:.2f}",
            "errorGrade": rms_grade(rms),
            "errorColor": rms_color(rms),
            "lastScreenshotPath": self.last_screenshot_path,
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
            raise ValueError(
                "Use the upstream camera stream URL, not this app's preview proxy."
            )
        camera_id = f"cam-{self.next_camera_id}"
        self.next_camera_id += 1
        camera = ManagedCamera(self, camera_id, label, clean_url)
        self.cameras[camera_id] = camera
        self.focus.clear_if_removed(set(self.cameras))
        return camera

    async def remove_camera(self, camera_id: str) -> bool:
        camera = self.cameras.pop(camera_id, None)
        if camera is None:
            return False
        await camera.stop()
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
            camera.accumulator = CalibrationAccumulator(
                DEFAULT_BOARD,
                self.session_dir / "ephemeral" / camera.id,
                auto_export=False,
                create_run_dir=False,
            )
            camera.accumulator.source_id = camera.id
            camera.accumulator.target_metadata = self.target_metadata
            camera.latest_detection = None
            camera.latest_detection_wall_time = None
            camera.last_screenshot_path = None
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


def rms_grade(rms: float | None) -> str:
    if rms is None:
        return "pending"
    if rms <= 0.80:
        return "good"
    if rms <= 1.20:
        return "marginal"
    return "poor"


def rms_color(rms: float | None) -> str:
    if rms is None:
        return "rgba(255,255,255,0.36)"
    stops = [
        (0.60, (39, 196, 111)),
        (1.00, (240, 180, 60)),
        (1.60, (255, 80, 69)),
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
        int(round(left[1][index] + (right[1][index] - left[1][index]) * t))
        for index in range(3)
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Expresso Calib", version="0.1.0")
    app.state.live = MultiCameraCalibrationState()
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
        if camera is None or camera.latest_jpeg is None:
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
            while True:
                if await request.is_disconnected():
                    break
                camera = request.app.state.live.cameras.get(camera_id)
                if camera is None:
                    break
                latest = camera.latest_jpeg
                if latest is not None and camera.latest_jpeg_seq != last_seq:
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

    @app.get("/api/latest-frame.jpg", include_in_schema=False)
    async def legacy_latest_frame(request: Request) -> Response:
        camera = _preview_camera(request.app.state.live)
        if camera is None or camera.latest_jpeg is None:
            return Response(status_code=204)
        return Response(
            camera.latest_jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/latest-stream.mjpg", include_in_schema=False)
    async def legacy_latest_stream(request: Request) -> StreamingResponse:
        async def frames() -> Any:
            last_camera_id = ""
            last_seq = -1
            boundary = b"--frame\r\n"
            while True:
                if await request.is_disconnected():
                    break
                camera = _preview_camera(request.app.state.live)
                if camera is None:
                    await asyncio.sleep(1.0 / PREVIEW_STREAM_FPS)
                    continue
                latest = camera.latest_jpeg
                if latest is not None and (
                    camera.id != last_camera_id or camera.latest_jpeg_seq != last_seq
                ):
                    last_camera_id = camera.id
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
    return (
        host in self_hosts
        and port == PORT
        and (
            parsed.path.startswith("/api/cameras/")
            or parsed.path == "/api/latest-stream.mjpg"
        )
    )


def _preview_camera(live: MultiCameraCalibrationState) -> ManagedCamera | None:
    if live.focus.focused_camera_id:
        camera = live.cameras.get(live.focus.focused_camera_id)
        if camera is not None:
            return camera
    return next(iter(live.cameras.values()), None)


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


app = create_app()
