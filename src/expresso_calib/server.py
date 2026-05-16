from __future__ import annotations

import asyncio
import json
import socket
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

import cv2
import numpy as np
import qrcode
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .board import DEFAULT_BOARD, target_pdf_bytes, target_png_bytes
from .calibration import (
    CalibrationAccumulator,
    CalibrationSolveResult,
    CandidateFrame,
    SolveInsufficientData,
    SolveNumericalFailure,
    SolveOk,
)
from .detection import CharucoDetector, Frame
from .multi_camera import FocusTracker, clean_label, slugify_label

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = Path(__file__).resolve().parent / "web"
RUNS_DIR = ROOT / "runs"
PORT = 3987
PREVIEW_STREAM_FPS = 18
DETECTION_FPS = 6
CAPTURE_OPEN_TIMEOUT_SEC = 8.0
CAPTURE_READ_TIMEOUT_SEC = 2.0
CAPTURE_RECONNECT_DELAY_SEC = 3.0
FRAME_STALE_SEC = 3.0
MJPEG_READ_CHUNK_BYTES = 4096
MJPEG_MAX_BUFFER_BYTES = 8 * 1024 * 1024
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


@dataclass(frozen=True)
class DetectionJob:
    generation: int
    frame_index: int
    timestamp_sec: float
    image_bgr: Any


@dataclass(frozen=True)
class ScreenshotJob:
    generation: int
    candidate: CandidateFrame


@dataclass(frozen=True)
class SolveJob:
    generation: int
    candidates: list[CandidateFrame]
    consumed_new_frames: int


def _put_latest(queue: asyncio.Queue[Any], item: Any) -> int:
    dropped = 0
    while True:
        try:
            queue.put_nowait(item)
            return dropped
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
                queue.task_done()
                dropped += 1
            except asyncio.QueueEmpty:
                continue


def _clear_queue(queue: asyncio.Queue[Any]) -> None:
    while True:
        try:
            queue.get_nowait()
            queue.task_done()
        except asyncio.QueueEmpty:
            return


def _task_running(task: asyncio.Task[Any] | None) -> bool:
    return task is not None and not task.done()


def _safe_int(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


class MjpegCapture:
    def __init__(self, url: str) -> None:
        self.url = url
        request = UrlRequest(
            url,
            headers={"User-Agent": "ExpressoCalib/0.1", "Connection": "close"},
        )
        self.response = urlopen(request, timeout=CAPTURE_OPEN_TIMEOUT_SEC)
        self.buffer = bytearray()
        self.opened = True
        self.boundary = self._parse_boundary(
            self.response.headers.get("Content-Type", "")
        )

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, Any]:
        if self.boundary is not None:
            ok, image = self._read_multipart_frame()
            if ok:
                return ok, image
            if not self.opened:
                return False, None

        return self._scan_jpeg_frame()

    def _parse_boundary(self, content_type: str) -> bytes | None:
        for part in content_type.split(";"):
            name, separator, value = part.strip().partition("=")
            if separator and name.lower() == "boundary":
                clean = value.strip().strip('"')
                if clean:
                    if not clean.startswith("--"):
                        clean = "--" + clean
                    return clean.encode("ascii", "ignore")
        return None

    def _read_multipart_frame(self) -> tuple[bool, Any]:
        while self.opened:
            line = self.response.readline()
            if not line:
                self.opened = False
                return False, None
            if self.boundary not in line.strip():
                continue

            headers: dict[str, str] = {}
            while True:
                line = self.response.readline()
                if not line:
                    self.opened = False
                    return False, None
                if line in {b"\r\n", b"\n"}:
                    break
                key, separator, value = line.partition(b":")
                if separator:
                    headers[key.decode("latin1").strip().lower()] = (
                        value.decode("latin1").strip()
                    )

            content_length = _safe_int(headers.get("content-length"))
            if content_length is None or content_length <= 0:
                self.boundary = None
                return False, None

            jpeg = self.response.read(content_length)
            if len(jpeg) != content_length:
                self.opened = False
                return False, None
            return self._decode_jpeg(jpeg)

        return False, None

    def _scan_jpeg_frame(self) -> tuple[bool, Any]:
        while self.opened:
            start = self.buffer.find(b"\xff\xd8")
            end = self.buffer.find(b"\xff\xd9", start + 2) if start >= 0 else -1
            if start >= 0 and end >= 0:
                jpeg = bytes(self.buffer[start : end + 2])
                del self.buffer[: end + 2]
                return self._decode_jpeg(jpeg)

            read = getattr(self.response, "read1", self.response.read)
            chunk = read(MJPEG_READ_CHUNK_BYTES)
            if not chunk:
                self.opened = False
                return False, None
            self.buffer.extend(chunk)
            if len(self.buffer) > MJPEG_MAX_BUFFER_BYTES:
                del self.buffer[: len(self.buffer) - MJPEG_READ_CHUNK_BYTES]
        return False, None

    def _decode_jpeg(self, jpeg: bytes) -> tuple[bool, Any]:
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        return (image is not None), image

    def release(self) -> None:
        self.opened = False
        try:
            self.response.close()
        except Exception:
            pass

    def set(self, *_args: Any) -> bool:
        return False


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
        self.accumulator = self._new_accumulator(manager.session_dir, manager.target_metadata)

        self.running = False
        self.task: asyncio.Task[None] | None = None
        self.capture_task: asyncio.Task[None] | None = None
        self.preview_task: asyncio.Task[None] | None = None
        self.detection_task: asyncio.Task[None] | None = None
        self.solver_task: asyncio.Task[None] | None = None
        self.screenshot_task: asyncio.Task[None] | None = None
        self.last_error: str | None = None
        self.frames_seen = 0
        self.frame_index = 0
        self.started_at: float | None = None
        self.last_frame_at: float | None = None
        self.frame_times: deque[float] = deque()
        self.latest_frame: Any | None = None
        self.latest_frame_seq = 0
        self.latest_jpeg: bytes | None = None
        self.latest_jpeg_seq = 0
        self.latest_detection: Any = None
        self.latest_detection_wall_time: float | None = None
        self.detection_running = False
        self.solver_running = False
        self.screenshot_running = False
        self.last_detection_started_at = 0.0
        self.last_detection_sample_at = 0.0
        self.last_screenshot_path: str | None = None
        self.last_preview_encoded_at = 0.0
        self.last_broadcast_at = 0.0
        self.generation = 0
        self.dropped_detection_frames = 0
        self.dropped_screenshot_jobs = 0
        self.detection_queue: asyncio.Queue[DetectionJob] = asyncio.Queue(maxsize=1)
        self.solve_queue: asyncio.Queue[SolveJob] = asyncio.Queue(maxsize=1)
        self.screenshot_queue: asyncio.Queue[ScreenshotJob] = asyncio.Queue(maxsize=64)

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

    async def start(self) -> None:
        if self.running:
            return
        self.generation += 1
        self._clear_pipeline_queues()
        self.running = True
        self.last_error = None
        self.started_at = time.time()
        self.last_detection_sample_at = 0.0
        self.last_detection_started_at = 0.0
        self._clear_preview_state()
        self.capture_task = asyncio.create_task(
            self._capture_loop(), name=f"camera-capture-{self.id}"
        )
        self.preview_task = asyncio.create_task(
            self._preview_loop(), name=f"camera-preview-{self.id}"
        )
        self.detection_task = asyncio.create_task(
            self._detection_loop(), name=f"camera-detection-{self.id}"
        )
        self.solver_task = asyncio.create_task(
            self._solver_loop(), name=f"camera-solver-{self.id}"
        )
        self.screenshot_task = asyncio.create_task(
            self._screenshot_loop(), name=f"camera-screenshot-{self.id}"
        )
        self.task = self.capture_task

    async def stop(self) -> None:
        self.generation += 1
        self.running = False
        tasks = [
            task
            for task in (
                self.capture_task,
                self.preview_task,
                self.detection_task,
                self.solver_task,
                self.screenshot_task,
            )
            if task is not None
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.task = None
        self.capture_task = None
        self.preview_task = None
        self.detection_task = None
        self.solver_task = None
        self.screenshot_task = None
        self.detection_running = False
        self.solver_running = False
        self.screenshot_running = False
        self._clear_pipeline_queues()
        self._clear_preview_state()

    async def _capture_loop(self) -> None:
        while self.running:
            try:
                capture = await asyncio.wait_for(
                    asyncio.to_thread(self._open_capture),
                    timeout=CAPTURE_OPEN_TIMEOUT_SEC,
                )
            except asyncio.TimeoutError:
                self._mark_capture_unavailable("Camera open timed out; reconnecting.")
                await self.manager.broadcast()
                await asyncio.sleep(CAPTURE_RECONNECT_DELAY_SEC)
                continue
            except Exception as exc:
                self._mark_capture_unavailable(
                    f"Camera open failed: {exc}; reconnecting."
                )
                await self.manager.broadcast()
                await asyncio.sleep(CAPTURE_RECONNECT_DELAY_SEC)
                continue
            if capture is None or not capture.isOpened():
                self._mark_capture_unavailable(f"Could not open camera URL: {self.url}")
                await self._maybe_broadcast()
                await asyncio.sleep(CAPTURE_RECONNECT_DELAY_SEC)
                continue

            try:
                while self.running:
                    try:
                        ok, frame = await asyncio.wait_for(
                            asyncio.to_thread(capture.read),
                            timeout=CAPTURE_READ_TIMEOUT_SEC,
                        )
                    except asyncio.TimeoutError:
                        self._mark_capture_unavailable(
                            "Camera read timed out; reconnecting."
                        )
                        await self.manager.broadcast()
                        break
                    except Exception as exc:
                        self._mark_capture_unavailable(
                            f"Camera read failed: {exc}; reconnecting."
                        )
                        await self.manager.broadcast()
                        break

                    if not ok or frame is None:
                        self._mark_capture_unavailable(
                            "Camera URL did not return a frame; reconnecting."
                        )
                        await self._maybe_broadcast()
                        break

                    self.last_error = None
                    self.frames_seen += 1
                    self.frame_index += 1
                    now = time.time()
                    self.last_frame_at = now
                    self.frame_times.append(now)
                    while self.frame_times and now - self.frame_times[0] > 2.0:
                        self.frame_times.popleft()

                    self.latest_frame = frame
                    self.latest_frame_seq += 1
                    if self._should_sample_detection(now):
                        self.last_detection_sample_at = now
                        timestamp = time.monotonic() - self.manager.started_at
                        dropped = _put_latest(
                            self.detection_queue,
                            DetectionJob(
                                generation=self.generation,
                                frame_index=self.frame_index,
                                timestamp_sec=timestamp,
                                image_bgr=frame,
                            ),
                        )
                        self.dropped_detection_frames += dropped

                    await self._maybe_broadcast()
                    await asyncio.sleep(0)
            finally:
                await asyncio.to_thread(capture.release)

            if self.running:
                await asyncio.sleep(CAPTURE_RECONNECT_DELAY_SEC)

    def _open_capture(self) -> Any:
        parsed = urlparse(self.url)
        if parsed.scheme in {"http", "https"}:
            return MjpegCapture(self.url)

        params: list[int] = []
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000])
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000])

        capture = None
        if params and hasattr(cv2, "CAP_FFMPEG"):
            capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, params)
            if capture is not None and not capture.isOpened():
                capture.release()
                capture = None

        if capture is None:
            capture = cv2.VideoCapture(self.url)
        if capture is not None and capture.isOpened():
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    async def _preview_loop(self) -> None:
        last_encoded_source_seq = -1
        while self.running:
            frame = self.latest_frame
            source_seq = self.latest_frame_seq
            if frame is not None and source_seq != last_encoded_source_seq:
                try:
                    encode_ok, encoded = await asyncio.to_thread(
                        cv2.imencode,
                        ".jpg",
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 78],
                    )
                    if encode_ok:
                        self.latest_jpeg = encoded.tobytes()
                        self.latest_jpeg_seq += 1
                        self.last_preview_encoded_at = time.time()
                        last_encoded_source_seq = source_seq
                        await self._maybe_broadcast()
                except Exception as exc:
                    self.last_error = f"Preview encode failed: {exc}"
                    await self.manager.broadcast()
            await asyncio.sleep(1.0 / PREVIEW_STREAM_FPS)

    def _should_encode_preview(self, now: float) -> bool:
        return (
            self.latest_jpeg is None
            or now - self.last_preview_encoded_at >= 1.0 / PREVIEW_STREAM_FPS
        )

    def _should_sample_detection(self, now: float) -> bool:
        return (
            self.last_detection_sample_at <= 0
            or now - self.last_detection_sample_at >= 1.0 / DETECTION_FPS
        )

    async def _detection_loop(self) -> None:
        while self.running:
            try:
                job = await asyncio.wait_for(self.detection_queue.get(), timeout=0.20)
            except asyncio.TimeoutError:
                continue

            try:
                if job.generation != self.generation:
                    continue

                self.detection_running = True
                self.last_detection_started_at = time.time()
                frame = Frame(
                    index=job.frame_index,
                    timestamp_sec=job.timestamp_sec,
                    image_bgr=job.image_bgr,
                    source_id=self.id,
                )
                detection = await asyncio.to_thread(self.detector.detect, frame)
                if job.generation != self.generation:
                    continue

                accepted, _ = self.accumulator.observe(detection, job.image_bgr)
                self.latest_detection = detection
                self.latest_detection_wall_time = time.time()
                if accepted and self.accumulator.candidates:
                    candidate = self.accumulator.candidates[-1]
                    self.dropped_screenshot_jobs += _put_latest(
                        self.screenshot_queue,
                        ScreenshotJob(generation=job.generation, candidate=candidate),
                    )
                    self._enqueue_solve_if_due(job.generation)
            except Exception as exc:
                self.last_error = f"Detection failed: {exc}"
            finally:
                self.detection_running = False
                self.detection_queue.task_done()
                await self.manager.broadcast()

    async def _solver_loop(self) -> None:
        while self.running:
            try:
                job = await asyncio.wait_for(self.solve_queue.get(), timeout=0.20)
            except asyncio.TimeoutError:
                continue

            try:
                if job.generation != self.generation:
                    continue
                self.solver_running = True
                outcome = await asyncio.to_thread(
                    self.accumulator.solve_snapshot, job.candidates
                )
                if job.generation != self.generation:
                    continue
                match outcome:
                    case SolveOk(solve=result):
                        self._commit_solve_result(job, result)
                        self.last_error = None
                        if self.accumulator.should_solve():
                            self._enqueue_solve_if_due(
                                job.generation, allow_while_running=True
                            )
                    case SolveInsufficientData():
                        pass
                    case SolveNumericalFailure(reason=reason):
                        self.last_error = f"Calibration solve failed: {reason}"
            finally:
                self.solver_running = False
                self.solve_queue.task_done()
                await self.manager.broadcast()

    async def _screenshot_loop(self) -> None:
        while self.running:
            try:
                job = await asyncio.wait_for(self.screenshot_queue.get(), timeout=0.20)
            except asyncio.TimeoutError:
                continue

            try:
                if job.generation != self.generation:
                    continue
                self.screenshot_running = True
                path = await asyncio.to_thread(
                    self._write_candidate_screenshot, job.candidate
                )
                if job.generation == self.generation:
                    self.last_screenshot_path = str(path)
            except Exception as exc:
                self.last_error = f"Screenshot write failed: {exc}"
            finally:
                self.screenshot_running = False
                self.screenshot_queue.task_done()
                await self.manager.broadcast()

    def _enqueue_solve_if_due(
        self, generation: int, *, allow_while_running: bool = False
    ) -> bool:
        if generation != self.generation:
            return False
        if (
            self.solver_running
            and not allow_while_running
        ) or self.solve_queue.qsize() > 0:
            return False
        if not self.accumulator.should_solve():
            return False
        try:
            self.solve_queue.put_nowait(
                SolveJob(
                    generation=generation,
                    candidates=list(self.accumulator.candidates),
                    consumed_new_frames=self.accumulator.accepted_since_solve,
                )
            )
        except asyncio.QueueFull:
            return False
        return True

    def _commit_solve_result(
        self, job: SolveJob, result: CalibrationSolveResult
    ) -> bool:
        if job.generation != self.generation:
            return False
        self.accumulator.commit_solve_result(
            result, consumed_new_frames=job.consumed_new_frames
        )
        return True

    def _write_candidate_screenshot(self, item: CandidateFrame) -> Path:
        camera_dir = self.manager.session_dir / "screenshots" / slugify_label(
            self.label, self.id
        )
        return self.accumulator.write_candidate_screenshot(item, camera_dir)

    def _clear_pipeline_queues(self) -> None:
        _clear_queue(self.detection_queue)
        _clear_queue(self.solve_queue)
        _clear_queue(self.screenshot_queue)

    def _clear_preview_state(self) -> None:
        self.latest_frame = None
        self.latest_frame_seq += 1
        self.latest_jpeg = None
        self.latest_jpeg_seq += 1
        self.last_frame_at = None
        self.last_preview_encoded_at = 0.0
        self.frame_times.clear()

    def _mark_capture_unavailable(self, reason: str) -> None:
        self.last_error = reason
        self._clear_preview_state()

    def reset_calibration(
        self, session_dir: Path, target_metadata: dict[str, Any]
    ) -> None:
        self.generation += 1
        self._clear_pipeline_queues()
        self.accumulator = self._new_accumulator(session_dir, target_metadata)
        self.latest_detection = None
        self.latest_detection_wall_time = None
        self.last_screenshot_path = None
        self.detection_running = False
        self.solver_running = False
        self.screenshot_running = False

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
        if self.last_frame_at is None or now - self.last_frame_at > FRAME_STALE_SEC:
            return False
        if now - self.latest_detection_wall_time > 1.25:
            return False
        return detection.charuco_count >= self.accumulator.min_corners

    def has_fresh_preview(self, now: float) -> bool:
        return (
            self.latest_jpeg is not None
            and self.last_frame_at is not None
            and now - self.last_frame_at <= FRAME_STALE_SEC
        )

    def connected(self, now: float) -> bool:
        return (
            self.running
            and self.last_frame_at is not None
            and now - self.last_frame_at <= FRAME_STALE_SEC
        )

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
        detection = (
            self.latest_detection.to_public_dict() if self.latest_detection else None
        )
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
            "lastScreenshotPath": self.last_screenshot_path,
            "pipeline": {
                "captureRunning": _task_running(self.capture_task),
                "previewRunning": _task_running(self.preview_task),
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
        (0.80, (22, 163, 74)),
        (1.20, (234, 179, 8)),
        (1.80, (220, 38, 38)),
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

    @app.get("/api/latest-frame.jpg", include_in_schema=False)
    async def legacy_latest_frame(request: Request) -> Response:
        camera = _preview_camera(request.app.state.live)
        if camera is None or not camera.has_fresh_preview(time.time()):
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
            last_generation = -1
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
                    camera.has_fresh_preview(time.time())
                    and (
                        camera.id != last_camera_id
                        or camera.generation != last_generation
                        or camera.latest_jpeg_seq != last_seq
                    )
                ):
                    last_camera_id = camera.id
                    last_generation = camera.generation
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
