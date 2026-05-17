from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

import cv2
import numpy as np

from .filters import FilterPipeline

CAPTURE_OPEN_TIMEOUT_SEC = 8.0
CAPTURE_READ_TIMEOUT_SEC = 2.0
CAPTURE_RECONNECT_DELAY_SEC = 3.0
FRAME_STALE_SEC = 3.0
MJPEG_READ_CHUNK_BYTES = 4096
MJPEG_MAX_BUFFER_BYTES = 8 * 1024 * 1024
MJPEG_MAX_FRAME_BYTES = 8 * 1024 * 1024
PREVIEW_STREAM_FPS = 18


FrameListener = Callable[[Any, float, int], Awaitable[None]]


def _safe_int(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


class MjpegCapture:
    def __init__(self, url: str) -> None:
        self.url = url
        self.response = None
        self.buffer = bytearray()
        self.opened = False
        self.boundary = None
        request = UrlRequest(
            url,
            headers={"User-Agent": "ExpressoCalib/0.1", "Connection": "close"},
        )
        response = urlopen(request, timeout=CAPTURE_OPEN_TIMEOUT_SEC)
        try:
            self.boundary = self._parse_boundary(response.headers.get("Content-Type", ""))
        except Exception:
            response.close()
            raise
        self.response = response
        self.opened = True

    def isOpened(self) -> bool:
        return self.opened

    def release(self) -> None:
        self.opened = False
        response = self.response
        self.response = None
        if response is not None:
            with contextlib.suppress(Exception):
                response.close()

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
                    headers[key.decode("latin1").strip().lower()] = value.decode("latin1").strip()

            content_length = _safe_int(headers.get("content-length"))
            if content_length is None or content_length <= 0:
                self.boundary = None
                return False, None
            if content_length > MJPEG_MAX_FRAME_BYTES:
                self.opened = False
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

    def set(self, *_args: Any) -> bool:
        return False


class CameraPipeline:
    """Owns the capture + preview loops and the latest BGR/JPEG state.

    Consumers register frame listeners via ``on_frame`` to receive decoded
    BGR frames (e.g. the CalibrationWorker subscribes to drive detection).
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self.running = False
        self.latest_frame: Any | None = None
        self.latest_frame_seq = 0
        self.latest_jpeg: bytes | None = None
        self.latest_jpeg_seq = 0
        self.last_frame_at: float | None = None
        self.last_preview_encoded_at = 0.0
        self.started_at: float | None = None
        self.frames_seen = 0
        self.frame_index = 0
        self.last_error: str | None = None
        self.frame_times: deque[float] = deque()
        self.capture_task: asyncio.Task[None] | None = None
        self.preview_task: asyncio.Task[None] | None = None
        self._frame_listeners: list[FrameListener] = []
        self._broadcast: Callable[[], Awaitable[None]] | None = None
        self._last_broadcast_at = 0.0
        self.filters = FilterPipeline()

    def on_frame(self, listener: FrameListener) -> None:
        self._frame_listeners.append(listener)

    def set_broadcast(self, broadcast: Callable[[], Awaitable[None]]) -> None:
        """Wire in the shared metrics broadcaster used by reconnect notices."""
        self._broadcast = broadcast

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.last_error = None
        self.started_at = time.time()
        self._clear_preview_state()
        self.capture_task = asyncio.create_task(self._capture_loop())
        self.preview_task = asyncio.create_task(self._preview_loop())

    async def stop(self) -> None:
        self.running = False
        tasks = [t for t in (self.capture_task, self.preview_task) if t is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.capture_task = None
        self.preview_task = None
        self._clear_preview_state()

    def fps(self) -> float:
        if len(self.frame_times) < 2:
            return 0.0
        elapsed = self.frame_times[-1] - self.frame_times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.frame_times) - 1) / elapsed

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

    async def _capture_loop(self) -> None:
        while self.running:
            try:
                capture = await asyncio.wait_for(
                    asyncio.to_thread(self._open_capture),
                    timeout=CAPTURE_OPEN_TIMEOUT_SEC,
                )
            except TimeoutError:
                self._mark_capture_unavailable("Camera open timed out; reconnecting.")
                await self._broadcast_now()
                await asyncio.sleep(CAPTURE_RECONNECT_DELAY_SEC)
                continue
            except Exception as exc:
                self._mark_capture_unavailable(f"Camera open failed: {exc}; reconnecting.")
                await self._broadcast_now()
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
                    except TimeoutError:
                        self._mark_capture_unavailable("Camera read timed out; reconnecting.")
                        await self._broadcast_now()
                        break
                    except Exception as exc:
                        self._mark_capture_unavailable(f"Camera read failed: {exc}; reconnecting.")
                        await self._broadcast_now()
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

                    if self.filters.settings.is_default():
                        filtered = frame
                    else:
                        try:
                            filtered = await asyncio.to_thread(self.filters.apply, frame)
                        except Exception as exc:
                            self.last_error = f"Filter apply failed: {exc}"
                            filtered = frame
                    self.latest_frame = filtered
                    self.latest_frame_seq += 1

                    await self._notify_frame(filtered, now, self.frame_index)

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

        if parsed.scheme == "device":
            # Local webcam via OpenCV's platform-native backend. Index parsed
            # out of device://N. No FFmpeg params (they only apply to network
            # streams).
            index_str = (parsed.netloc or parsed.path.lstrip("/") or "0").strip()
            try:
                index = int(index_str)
            except ValueError:
                index = 0
            capture = cv2.VideoCapture(index)
            if capture is not None and capture.isOpened() and hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return capture

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
        if capture is not None and capture.isOpened() and hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
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
                    await self._broadcast_now()
            await asyncio.sleep(1.0 / PREVIEW_STREAM_FPS)

    async def _notify_frame(self, bgr: Any, ts_sec: float, frame_idx: int) -> None:
        for listener in list(self._frame_listeners):
            try:
                await listener(bgr, ts_sec, frame_idx)
            except Exception as exc:
                self.last_error = f"Frame listener failed: {exc}"

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

    async def _broadcast_now(self) -> None:
        if self._broadcast is not None:
            await self._broadcast()

    async def _maybe_broadcast(self) -> None:
        if self._broadcast is None:
            return
        now = time.time()
        if now - self._last_broadcast_at >= 0.20:
            self._last_broadcast_at = now
            await self._broadcast()
