from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .calibration import (
    CalibrationAccumulator,
    CalibrationSolveResult,
    CandidateFrame,
    SolveInsufficientData,
    SolveNumericalFailure,
    SolveOk,
)
from .detection import CharucoDetector, DetectionResult, Frame

DETECTION_FPS = 6


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


BroadcastCallback = Callable[[], Awaitable[None]]


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


class CalibrationWorker:
    """Owns the CalibrationAccumulator and runs detection, solve, and
    screenshot loops.

    Frames flow in via ``handle_frame`` (typically wired up as a listener
    on a ``CameraPipeline``). The worker rate-limits detection sampling,
    enqueues solve / screenshot jobs as candidates are accepted, and
    calls ``broadcast`` after each unit of progress so the metrics hub
    can fan out updates.
    """

    def __init__(
        self,
        *,
        accumulator: CalibrationAccumulator,
        screenshot_dir: Path,
        broadcast: BroadcastCallback,
        source_id: str,
        wall_clock_start: float,
    ) -> None:
        self.accumulator = accumulator
        self.screenshot_dir = screenshot_dir
        self.broadcast = broadcast
        self.source_id = source_id
        self.wall_clock_start = wall_clock_start
        self.detector = CharucoDetector(accumulator.board_config)
        self.generation = 0
        self.last_error: str | None = None
        self.last_screenshot_path: str | None = None
        self.latest_detection: DetectionResult | None = None
        self.latest_detection_wall_time: float | None = None
        self.detection_queue: asyncio.Queue[DetectionJob] = asyncio.Queue(maxsize=1)
        self.solve_queue: asyncio.Queue[SolveJob] = asyncio.Queue(maxsize=1)
        self.screenshot_queue: asyncio.Queue[ScreenshotJob] = asyncio.Queue(maxsize=64)
        self.detection_running = False
        self.solver_running = False
        self.screenshot_running = False
        self.last_detection_started_at = 0.0
        self.last_detection_sample_at = 0.0
        self.dropped_detection_frames = 0
        self.dropped_screenshot_jobs = 0
        self.detection_task: asyncio.Task[None] | None = None
        self.solver_task: asyncio.Task[None] | None = None
        self.screenshot_task: asyncio.Task[None] | None = None
        self.running = False

    async def start(self) -> None:
        if self.running:
            return
        self.generation += 1
        self._clear_pipeline_queues()
        self.running = True
        self.last_error = None
        self.last_detection_sample_at = 0.0
        self.last_detection_started_at = 0.0
        self.detection_task = asyncio.create_task(self._detection_loop())
        self.solver_task = asyncio.create_task(self._solver_loop())
        self.screenshot_task = asyncio.create_task(self._screenshot_loop())

    async def stop(self) -> None:
        self.generation += 1
        self.running = False
        tasks = [
            t
            for t in (self.detection_task, self.solver_task, self.screenshot_task)
            if t is not None
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.detection_task = None
        self.solver_task = None
        self.screenshot_task = None
        self.detection_running = False
        self.solver_running = False
        self.screenshot_running = False
        self._clear_pipeline_queues()

    async def handle_frame(self, bgr: Any, ts_sec: float, frame_idx: int) -> None:
        """Entry point used by the camera pipeline. Rate-limits detection
        sampling and enqueues a DetectionJob when due."""
        if not self.running:
            return
        if not self._should_sample_detection(ts_sec):
            return
        self.last_detection_sample_at = ts_sec
        timestamp = time.monotonic() - self.wall_clock_start
        dropped = _put_latest(
            self.detection_queue,
            DetectionJob(
                generation=self.generation,
                frame_index=frame_idx,
                timestamp_sec=timestamp,
                image_bgr=bgr,
            ),
        )
        self.dropped_detection_frames += dropped

    def _should_sample_detection(self, now: float) -> bool:
        return (
            self.last_detection_sample_at <= 0
            or now - self.last_detection_sample_at >= 1.0 / DETECTION_FPS
        )

    async def _detection_loop(self) -> None:
        while self.running:
            try:
                job = await asyncio.wait_for(self.detection_queue.get(), timeout=0.20)
            except TimeoutError:
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
                    source_id=self.source_id,
                )
                detection = await asyncio.to_thread(self.detector.detect, frame)
                if job.generation != self.generation:
                    continue

                async with self.accumulator.lock:
                    accepted, _ = self.accumulator.observe(detection, job.image_bgr)
                    candidate = (
                        self.accumulator.candidates[-1]
                        if accepted and self.accumulator.candidates
                        else None
                    )
                self.latest_detection = detection
                self.latest_detection_wall_time = time.time()
                if candidate is not None:
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
                await self.broadcast()

    async def _solver_loop(self) -> None:
        while self.running:
            try:
                job = await asyncio.wait_for(self.solve_queue.get(), timeout=0.20)
            except TimeoutError:
                continue

            try:
                if job.generation != self.generation:
                    continue
                self.solver_running = True
                async with self.accumulator.lock:
                    snapshot = list(self.accumulator.candidates)
                outcome = await asyncio.to_thread(self.accumulator.solve_snapshot, snapshot)
                if job.generation != self.generation:
                    continue
                match outcome:
                    case SolveOk(solve=result):
                        async with self.accumulator.lock:
                            self._commit_solve_result(job, result)
                            should_resolve = self.accumulator.should_solve()
                        self.last_error = None
                        if should_resolve:
                            self._enqueue_solve_if_due(job.generation, allow_while_running=True)
                    case SolveInsufficientData():
                        pass
                    case SolveNumericalFailure(reason=reason):
                        self.last_error = f"Calibration solve failed: {reason}"
            finally:
                self.solver_running = False
                self.solve_queue.task_done()
                await self.broadcast()

    async def _screenshot_loop(self) -> None:
        while self.running:
            try:
                job = await asyncio.wait_for(self.screenshot_queue.get(), timeout=0.20)
            except TimeoutError:
                continue

            try:
                if job.generation != self.generation:
                    continue
                self.screenshot_running = True
                path = await asyncio.to_thread(self._write_candidate_screenshot, job.candidate)
                if job.generation == self.generation:
                    self.last_screenshot_path = str(path)
            except Exception as exc:
                self.last_error = f"Screenshot write failed: {exc}"
            finally:
                self.screenshot_running = False
                self.screenshot_queue.task_done()
                await self.broadcast()

    def _enqueue_solve_if_due(self, generation: int, *, allow_while_running: bool = False) -> bool:
        if generation != self.generation:
            return False
        if (self.solver_running and not allow_while_running) or self.solve_queue.qsize() > 0:
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

    def _commit_solve_result(self, job: SolveJob, result: CalibrationSolveResult) -> bool:
        if job.generation != self.generation:
            return False
        self.accumulator.commit_solve_result(result, consumed_new_frames=job.consumed_new_frames)
        return True

    def _write_candidate_screenshot(self, item: CandidateFrame) -> Path:
        return self.accumulator.write_candidate_screenshot(item, self.screenshot_dir)

    def _clear_pipeline_queues(self) -> None:
        _clear_queue(self.detection_queue)
        _clear_queue(self.solve_queue)
        _clear_queue(self.screenshot_queue)

    def reset(
        self,
        *,
        accumulator: CalibrationAccumulator,
        screenshot_dir: Path,
    ) -> None:
        self.generation += 1
        self._clear_pipeline_queues()
        self.accumulator = accumulator
        self.screenshot_dir = screenshot_dir
        self.detector = CharucoDetector(accumulator.board_config)
        self.last_error = None
        self.last_screenshot_path = None
        self.latest_detection = None
        self.latest_detection_wall_time = None
        self.detection_running = False
        self.solver_running = False
        self.screenshot_running = False
