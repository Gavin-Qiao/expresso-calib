from __future__ import annotations

import asyncio
import json
import socket
import time
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
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
from .detection import CharucoDetector, Frame, decode_jpeg_frame

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = Path(__file__).resolve().parent / "web"
RUNS_DIR = ROOT / "runs"
PORT = 3987
PREVIEW_STREAM_FPS = 30
STREAM_DETECTION_FPS = 6
DEFAULT_CAMERA_URL = "http://127.0.0.1:3988/stream.mjpg"
STREAMING_SOURCE_IDS = {"native_camera_bridge", "url_camera"}


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


class LiveCalibrationState:
    def __init__(self) -> None:
        self.board_config = DEFAULT_BOARD
        self.detector = CharucoDetector(self.board_config)
        self.accumulator = CalibrationAccumulator(self.board_config, RUNS_DIR)
        self.hub = MetricsHub()
        self.lock = asyncio.Lock()
        self.frame_index = 0
        self.started_at = time.monotonic()
        self.backend_camera: BackendCameraSource | None = None
        self.url_camera: UrlCameraSource | None = None
        self.latest_jpeg: bytes | None = None
        self.latest_jpeg_seq = 0
        self.last_stream_detection_at = 0.0
        self.stream_detection_running = False
        self.native_bridge_enabled = False
        self.native_bridge_frames_seen = 0
        self.native_bridge_last_frame_at: float | None = None
        self.native_bridge_frame_times: deque[float] = deque()

    async def reset(self) -> dict[str, Any]:
        async with self.lock:
            self.frame_index = 0
            self.started_at = time.monotonic()
            self.accumulator.reset()
            payload = self.metrics()
        await self.hub.broadcast(payload)
        return payload

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
        async with self.lock:
            self.accumulator.target_metadata = clean
            payload = self.metrics()
        await self.hub.broadcast(payload)
        return clean

    async def process_frame(
        self, payload: bytes, source_id: str = "browser_upload"
    ) -> dict[str, Any]:
        scheduled_detection: tuple[bytes, int, float, str] | None = None
        async with self.lock:
            self.latest_jpeg = payload
            self.latest_jpeg_seq += 1
            if source_id in STREAMING_SOURCE_IDS:
                now_wall = time.time()
                if source_id == "native_camera_bridge":
                    self.native_bridge_frames_seen += 1
                    self.native_bridge_last_frame_at = now_wall
                    self.native_bridge_frame_times.append(now_wall)
                    while (
                        self.native_bridge_frame_times
                        and now_wall - self.native_bridge_frame_times[0] > 2.0
                    ):
                        self.native_bridge_frame_times.popleft()
                should_detect = (
                    not self.stream_detection_running
                    and now_wall - self.last_stream_detection_at >= 1.0 / STREAM_DETECTION_FPS
                )
                if should_detect:
                    self.stream_detection_running = True
                    self.last_stream_detection_at = now_wall
                    self.frame_index += 1
                    timestamp = time.monotonic() - self.started_at
                    scheduled_detection = (
                        payload,
                        self.frame_index,
                        timestamp,
                        source_id,
                    )
                metrics = self.metrics()
                metrics["sourceId"] = source_id
                metrics["frameAccepted"] = False
                metrics["frameAcceptReason"] = (
                    "detecting frame" if scheduled_detection else "preview frame"
                )
                metrics["frameSkipped"] = (
                    "detection_scheduled" if scheduled_detection else "preview_only"
                )
            else:
                self.frame_index += 1
                timestamp = time.monotonic() - self.started_at
                frame = decode_jpeg_frame(payload, self.frame_index, timestamp)
                metrics = self._process_frame_locked(frame, source_id=source_id)

        if scheduled_detection is not None:
            asyncio.create_task(
                self._process_scheduled_detection(*scheduled_detection),
                name="stream-charuco-detection",
            )
        await self.hub.broadcast(metrics)
        return metrics

    async def _process_scheduled_detection(
        self, payload: bytes, frame_index: int, timestamp: float, source_id: str
    ) -> None:
        try:
            frame = await asyncio.to_thread(
                decode_jpeg_frame, payload, frame_index, timestamp
            )
            detection = await asyncio.to_thread(self.detector.detect, frame)
            async with self.lock:
                self.accumulator.source_id = source_id
                accepted, reason = self.accumulator.observe(detection, frame.image_bgr)
                if accepted:
                    self.accumulator.solve_if_due()
                metrics = self.metrics()
                metrics["frameAccepted"] = accepted
                metrics["frameAcceptReason"] = reason
                metrics["sourceId"] = source_id
                self.stream_detection_running = False
        except Exception as exc:
            async with self.lock:
                self.stream_detection_running = False
                metrics = {
                    **self.metrics(),
                    "error": f"Stream frame processing failed: {exc}",
                    "sourceId": source_id,
                }
        await self.hub.broadcast(metrics)

    async def set_native_bridge_enabled(self, enabled: bool) -> dict[str, Any]:
        async with self.lock:
            self.native_bridge_enabled = enabled
            if not enabled:
                self.latest_jpeg = None
                self.latest_jpeg_seq += 1
            payload = self.metrics()
        await self.hub.broadcast(payload)
        return payload

    async def start_url_camera(self, url: str) -> dict[str, Any]:
        try:
            clean_url = _clean_camera_url(url)
        except ValueError as exc:
            return {"ok": False, "error": str(exc), "metrics": self.metrics()}
        if _is_self_latest_stream(clean_url):
            return {
                "ok": False,
                "error": "Use the upstream camera stream URL, not this app's /api/latest-stream.mjpg preview proxy.",
                "metrics": self.metrics(),
            }

        existing = self.url_camera
        if existing is not None:
            await existing.stop()

        async with self.lock:
            self.native_bridge_enabled = False
            self.latest_jpeg = None
            self.latest_jpeg_seq += 1
            self.url_camera = UrlCameraSource(self, clean_url)
            camera = self.url_camera

        await camera.start()
        payload = self.metrics()
        await self.hub.broadcast(payload)
        return {"ok": True, "metrics": payload}

    async def stop_url_camera(self) -> dict[str, Any]:
        camera = self.url_camera
        if camera is not None:
            await camera.stop()
        async with self.lock:
            self.url_camera = None
            self.latest_jpeg = None
            self.latest_jpeg_seq += 1
            payload = self.metrics()
        await self.hub.broadcast(payload)
        return {"ok": True, "metrics": payload}

    async def process_backend_image(self, image_bgr: Any) -> dict[str, Any]:
        async with self.lock:
            self.frame_index += 1
            timestamp = time.monotonic() - self.started_at
            frame = Frame(
                index=self.frame_index,
                timestamp_sec=timestamp,
                image_bgr=image_bgr,
                source_id="backend_camera",
            )
            metrics = self._process_frame_locked(frame, source_id="backend_camera")
        await self.hub.broadcast(metrics)
        return metrics

    def _process_frame_locked(self, frame: Frame, source_id: str) -> dict[str, Any]:
        self.accumulator.source_id = source_id
        detection = self.detector.detect(frame)
        accepted, reason = self.accumulator.observe(detection, frame.image_bgr)
        if accepted:
            self.accumulator.solve_if_due()
        metrics = self.metrics()
        metrics["frameAccepted"] = accepted
        metrics["frameAcceptReason"] = reason
        metrics["sourceId"] = source_id
        return metrics

    def metrics(self) -> dict[str, Any]:
        payload = self.accumulator.snapshot()
        payload["serverTime"] = time.time()
        payload["hasLatestFrame"] = self.latest_jpeg is not None
        payload["nativeBridge"] = {
            "enabled": self.native_bridge_enabled,
            "framesSeen": self.native_bridge_frames_seen,
            "lastFrameAt": self.native_bridge_last_frame_at,
            "fps": self._native_bridge_fps(),
        }
        payload["backendCamera"] = (
            self.backend_camera.status() if self.backend_camera is not None else None
        )
        payload["urlCamera"] = (
            self.url_camera.status() if self.url_camera is not None else None
        )
        return payload

    async def export(self, *, force_solve: bool = True) -> dict[str, Any]:
        async with self.lock:
            if force_solve:
                self.accumulator.solve_if_due(force=True)
            run_dir = self.accumulator.export()
            payload = self.metrics()
        await self.hub.broadcast(payload)
        return {"ok": True, "runDir": str(run_dir), "metrics": payload}

    def _native_bridge_fps(self) -> float:
        if len(self.native_bridge_frame_times) < 2:
            return 0.0
        elapsed = self.native_bridge_frame_times[-1] - self.native_bridge_frame_times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.native_bridge_frame_times) - 1) / elapsed


class UrlCameraSource:
    def __init__(self, live: LiveCalibrationState, url: str) -> None:
        self.live = live
        self.url = url
        self.source_id = "url_camera"
        self.running = False
        self.task: asyncio.Task[None] | None = None
        self.last_error: str | None = None
        self.frames_seen = 0
        self.started_at: float | None = None
        self.last_frame_at: float | None = None
        self.frame_times: deque[float] = deque()

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.last_error = None
        self.frames_seen = 0
        self.started_at = time.time()
        self.last_frame_at = None
        self.frame_times.clear()
        self.task = asyncio.create_task(self._run(), name="url-camera-source")

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
            await self.live.hub.broadcast(self.live.metrics())
            return

        try:
            while self.running:
                ok, frame = await asyncio.to_thread(capture.read)
                if not ok or frame is None:
                    self.last_error = "Camera URL did not return a frame."
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
                    await self.live.process_frame(
                        encoded.tobytes(), source_id=self.source_id
                    )
                await asyncio.sleep(0)
        finally:
            await asyncio.to_thread(capture.release)

    def fps(self) -> float:
        if len(self.frame_times) < 2:
            return 0.0
        elapsed = self.frame_times[-1] - self.frame_times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.frame_times) - 1) / elapsed

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "url": self.url,
            "framesSeen": self.frames_seen,
            "lastFrameAt": self.last_frame_at,
            "fps": self.fps(),
            "lastError": self.last_error,
            "startedAt": self.started_at,
        }


class BackendCameraSource:
    def __init__(self, live: LiveCalibrationState, index: int = 0) -> None:
        self.live = live
        self.index = index
        self.running = False
        self.task: asyncio.Task[None] | None = None
        self.latest_jpeg: bytes | None = None
        self.last_error: str | None = None
        self.frames_seen = 0
        self.started_at: float | None = None

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.last_error = None
        self.frames_seen = 0
        self.started_at = time.time()
        self.task = asyncio.create_task(self._run(), name="backend-camera-source")

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
        capture = await asyncio.to_thread(self._open_capture)
        if capture is None or not capture.isOpened():
            self.last_error = (
                f"Could not open OpenCV camera index {self.index}. macOS often denies "
                "camera access to terminal-launched Python, and OpenCV indices can "
                "point at Continuity Camera/iPhone devices. Prefer the native bridge, "
                "or try another OpenCV index from the operator page."
            )
            self.running = False
            await self.live.hub.broadcast(self.live.metrics())
            return

        try:
            while self.running:
                ok, frame = await asyncio.to_thread(capture.read)
                if not ok or frame is None:
                    self.last_error = "OpenCV camera did not return a frame."
                    await asyncio.sleep(0.15)
                    continue
                self.frames_seen += 1
                encode_ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
                if encode_ok:
                    self.latest_jpeg = encoded.tobytes()
                try:
                    await self.live.process_backend_image(frame)
                except Exception as exc:
                    self.last_error = f"OpenCV camera processing failed: {exc}"
                await asyncio.sleep(0.08)
        finally:
            await asyncio.to_thread(capture.release)

    def _open_capture(self) -> Any:
        capture = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            capture.set(cv2.CAP_PROP_FPS, 15)
            return capture
        capture.release()
        capture = cv2.VideoCapture(self.index)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            capture.set(cv2.CAP_PROP_FPS, 15)
        return capture

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "index": self.index,
            "framesSeen": self.frames_seen,
            "lastError": self.last_error,
            "hasPreview": self.latest_jpeg is not None,
            "startedAt": self.started_at,
        }


def create_app() -> FastAPI:
    app = FastAPI(title="Expresso Calib", version="0.1.0")
    app.state.live = LiveCalibrationState()
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
        target_url = target_url_for_request(request)
        return JSONResponse(
            {
                "targetUrl": target_url,
                "operatorUrl": str(request.url_for("operator_page")),
                "defaultCameraUrl": DEFAULT_CAMERA_URL,
                "jpegIngestUrl": str(request.url_for("ingest_jpeg_frame")),
                "board": DEFAULT_BOARD.manifest(),
                "lanIp": local_lan_ip(),
            }
        )

    @app.get("/api/qr.png")
    async def qr_png(request: Request) -> Response:
        image = qrcode.make(target_url_for_request(request))
        buffer = _pil_png_bytes(image)
        return Response(buffer, media_type="image/png")

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
        pdf = target_pdf_bytes(DEFAULT_BOARD, w, h)
        return Response(
            pdf,
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

    @app.post("/api/export")
    async def export(request: Request) -> JSONResponse:
        return JSONResponse(await request.app.state.live.export())

    @app.post("/api/target-metadata")
    async def target_metadata(request: Request) -> JSONResponse:
        body = await request.json()
        return JSONResponse(await request.app.state.live.update_target_metadata(body))

    @app.post("/api/frames/jpeg")
    async def ingest_jpeg_frame(request: Request) -> JSONResponse:
        payload = await request.body()
        if not payload:
            return JSONResponse({"ok": False, "error": "Missing JPEG payload."}, status_code=400)
        live = request.app.state.live
        if not live.native_bridge_enabled:
            return JSONResponse(
                {
                    "ok": False,
                    "error": "Native camera bridge is paused.",
                    "metrics": live.metrics(),
                },
                status_code=409,
                headers={"X-Expresso-Bridge-Command": "pause"},
            )
        metrics = await live.process_frame(payload, source_id="native_camera_bridge")
        return JSONResponse({"ok": True, "metrics": metrics})

    @app.get("/api/latest-frame.jpg")
    async def latest_frame(request: Request) -> Response:
        latest = request.app.state.live.latest_jpeg
        if latest is None:
            return Response(status_code=204)
        return Response(
            latest,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/latest-stream.mjpg")
    async def latest_stream(request: Request) -> StreamingResponse:
        async def frames() -> Any:
            last_seq = -1
            boundary = b"--frame\r\n"
            while True:
                if await request.is_disconnected():
                    break
                live = request.app.state.live
                latest = live.latest_jpeg
                seq = live.latest_jpeg_seq
                if latest is not None and seq != last_seq:
                    last_seq = seq
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

    @app.get("/api/native-bridge/control")
    async def native_bridge_control(request: Request) -> JSONResponse:
        live = request.app.state.live
        return JSONResponse(
            {
                "enabled": live.native_bridge_enabled,
                "framesSeen": live.native_bridge_frames_seen,
                "lastFrameAt": live.native_bridge_last_frame_at,
            },
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/native-bridge/start")
    async def native_bridge_start(request: Request) -> JSONResponse:
        metrics = await request.app.state.live.set_native_bridge_enabled(True)
        return JSONResponse({"ok": True, "metrics": metrics})

    @app.post("/api/native-bridge/stop")
    async def native_bridge_stop(request: Request) -> JSONResponse:
        metrics = await request.app.state.live.set_native_bridge_enabled(False)
        return JSONResponse({"ok": True, "metrics": metrics})

    @app.post("/api/camera-source/start")
    async def camera_source_start(request: Request) -> JSONResponse:
        body: dict[str, Any] = {}
        try:
            body = await request.json()
        except Exception:
            body = {}
        result = await request.app.state.live.start_url_camera(
            str(body.get("url") or DEFAULT_CAMERA_URL)
        )
        return JSONResponse(result, status_code=200 if result.get("ok") else 400)

    @app.post("/api/camera-source/stop")
    async def camera_source_stop(request: Request) -> JSONResponse:
        return JSONResponse(await request.app.state.live.stop_url_camera())

    @app.post("/api/backend-camera/start")
    async def backend_camera_start(request: Request) -> JSONResponse:
        body: dict[str, Any] = {}
        try:
            body = await request.json()
        except Exception:
            body = {}
        index = _optional_int(body.get("index")) if isinstance(body, dict) else None
        live = request.app.state.live
        if live.backend_camera is None or not live.backend_camera.running:
            live.backend_camera = BackendCameraSource(live, index=index or 0)
            await live.backend_camera.start()
        return JSONResponse({"ok": True, "backendCamera": live.backend_camera.status()})

    @app.post("/api/backend-camera/stop")
    async def backend_camera_stop(request: Request) -> JSONResponse:
        live = request.app.state.live
        if live.backend_camera is not None:
            await live.backend_camera.stop()
        return JSONResponse(
            {
                "ok": True,
                "backendCamera": live.backend_camera.status()
                if live.backend_camera is not None
                else None,
            }
        )

    @app.get("/api/backend-camera/latest.jpg")
    async def backend_camera_latest(request: Request) -> Response:
        live = request.app.state.live
        camera = live.backend_camera
        if camera is None or camera.latest_jpeg is None:
            return Response(status_code=204)
        return Response(
            camera.latest_jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
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

    @app.websocket("/ws/frames")
    async def ws_frames(websocket: WebSocket) -> None:
        await websocket.accept()
        live = websocket.app.state.live
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                payload = message.get("bytes")
                if payload:
                    try:
                        await live.process_frame(payload)
                    except Exception as exc:
                        await live.hub.broadcast(
                            {
                                **live.metrics(),
                                "error": f"Frame processing failed: {exc}",
                            }
                        )
        finally:
            await websocket.close()

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


def _is_self_latest_stream(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    self_hosts = {"127.0.0.1", "localhost", "::1", local_lan_ip()}
    return (
        host in self_hosts
        and port == PORT
        and parsed.path == "/api/latest-stream.mjpg"
    )


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
    uvicorn.run("expresso_calib.server:create_app", factory=True, host="0.0.0.0", port=PORT)


app = create_app()
