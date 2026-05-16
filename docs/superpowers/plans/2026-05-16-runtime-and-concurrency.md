# Runtime & Concurrency Implementation Plan (Plan B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the FastAPI runtime correct under load — close the accumulator data race, serialize broadcast so a slow WS client can't stall every producer, clean up cameras on shutdown, fix the `MjpegCapture` socket leak and `remove_camera` ordering, cap untrusted body sizes, delete dead code, and split the 533-line `ManagedCamera` god-class into a `CameraPipeline` + `CalibrationWorker` pair.

**Architecture:** Two new modules — `camera_pipeline.py` (capture + preview + `MjpegCapture`) and `calibration_worker.py` (detection + solve + screenshot loops + accumulator ownership). `ManagedCamera` shrinks to a thin composition + public-snapshot shim. `server.py` keeps `MultiCameraCalibrationState`, `MetricsHub`, `create_app`, routes. Concurrency primitive: a per-accumulator `asyncio.Lock` guards `observe` + `commit_solve_result`. Broadcast switches to fan-out via per-client bounded queues so slow clients drop frames instead of blocking producers. FastAPI lifespan stops all cameras on `SIGTERM`.

**Tech Stack:** asyncio, FastAPI lifespan, OpenCV, pytest (`pytest.mark.asyncio` via `pytest-asyncio` — add to dev deps if missing).

**Out of scope (Plan C):** server-authoritative threshold emission to JS, frontend bug pile, README local-only note, CI, dep pinning.

---

## Task 1: Accumulator data race + broadcast fan-out

The detection loop calls `accumulator.observe(...)` on the event loop ([server.py:524](src/expresso_calib/server.py:524)) while `_solver_loop` runs `accumulator.solve_snapshot` in a worker thread ([server.py:552](src/expresso_calib/server.py:552)) that reads `candidates`. `commit_solve_result` then mutates `selected`/`rejected`/`per_view_error_px` flags on the live `candidates` list. Today there's no lock — a concurrent `.append` (or the `candidates = self.select_diverse(...)` reassignment at [calibration.py:239](src/expresso_calib/calibration.py:239) when over `max_candidates`) during the solver's pass causes torn reads and id-based mapping mismatches.

Separately, `MetricsHub.broadcast` ([server.py:65-75](src/expresso_calib/server.py:65)) awaits each client's `send_text` serially. A slow client stalls every producer because every pipeline stage calls `manager.broadcast()` from its own coroutine.

**Files:**
- Modify: `src/expresso_calib/calibration.py` (add `asyncio.Lock`, expose `observe_locked` / `commit_solve_result_locked` async-friendly methods OR put the lock on the caller side — the design below puts it on `ManagedCamera`)
- Modify: `src/expresso_calib/server.py` (`MetricsHub`, `ManagedCamera` solver/detection loops)
- Modify: `tests/test_camera_pipeline.py` (new concurrency tests)
- Modify: `pyproject.toml` ([dependency-groups].dev) — add `pytest-asyncio>=0.23`

- [ ] **Step 1: Add pytest-asyncio to dev deps**

```toml
[dependency-groups]
dev = [
  "pytest>=8",
  "pytest-asyncio>=0.23",
]
```

Run `uv sync` to install.

- [ ] **Step 2: Write a failing test for the broadcast slow-client isolation**

In `tests/test_camera_pipeline.py` add:

```python
import asyncio
import pytest

from expresso_calib.server import MetricsHub


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
```

- [ ] **Step 3: Run the test to verify it fails**

```
uv run pytest tests/test_camera_pipeline.py::test_metrics_hub_slow_client_does_not_stall_fast_client -v
```

Expected: FAIL with `asyncio.TimeoutError` because the serial `await send_text` stalls 0.5s on the slow client.

- [ ] **Step 4: Switch `MetricsHub` to per-client bounded queues with drop-oldest**

Replace `MetricsHub` in `src/expresso_calib/server.py:51-75` with:

```python
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
```

- [ ] **Step 5: Run the test — it should pass now**

```
uv run pytest tests/test_camera_pipeline.py::test_metrics_hub_slow_client_does_not_stall_fast_client -v
```

Expected: PASS.

- [ ] **Step 6: Write failing test for the accumulator data race**

Add to `tests/test_calibration_accumulator.py`:

```python
import asyncio
import pytest


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

    # Seed enough candidates to solve
    for i, (cx, cy) in enumerate(
        [(0.2, 0.2), (0.7, 0.2), (0.2, 0.7), (0.7, 0.7)], start=1
    ):
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
```

- [ ] **Step 7: Run the test — should fail with `AttributeError`**

```
uv run pytest tests/test_calibration_accumulator.py::test_concurrent_observe_and_commit_does_not_corrupt_candidates -v
```

Expected: FAIL with `AttributeError: 'CalibrationAccumulator' object has no attribute 'lock'`.

- [ ] **Step 8: Add `asyncio.Lock` to `CalibrationAccumulator`**

At the top of `calibration.py`, add `import asyncio`. In `__init__` after `self.candidates: list[CandidateFrame] = []`, add:

```python
self.lock = asyncio.Lock()
```

- [ ] **Step 9: Use the lock in the server's detection and solver loops**

In `src/expresso_calib/server.py:_detection_loop`, find the `self.accumulator.observe(detection, job.image_bgr)` call and wrap it:

```python
async with self.accumulator.lock:
    accepted, _ = self.accumulator.observe(detection, job.image_bgr)
    if accepted and self.accumulator.candidates:
        candidate = self.accumulator.candidates[-1]
```

In `src/expresso_calib/server.py:_solver_loop`, wrap the snapshot-copy and the commit:

```python
async with self.accumulator.lock:
    snapshot = list(self.accumulator.candidates)
outcome = await asyncio.to_thread(self.accumulator.solve_snapshot, snapshot)
if job.generation != self.generation:
    continue
match outcome:
    case SolveOk(solve=result):
        async with self.accumulator.lock:
            self._commit_solve_result(job, result)
        self.last_error = None
        if self.accumulator.should_solve():
            self._enqueue_solve_if_due(job.generation, allow_while_running=True)
    case SolveInsufficientData():
        pass
    case SolveNumericalFailure(reason=reason):
        self.last_error = f"Calibration solve failed: {reason}"
```

Drop the existing inline `snapshot = job.candidates` argument — the job's snapshot was taken without a lock at enqueue time; we now snapshot inside the lock here. (Optionally remove the `candidates` field from `SolveJob` since it's unused after this change — do that under Task 5's class split, not here.)

- [ ] **Step 10: Run all tests**

```
uv run pytest -v
```

Expected: 28 PASS (26 prior + 2 new).

- [ ] **Step 11: Commit**

```bash
git add src/expresso_calib/calibration.py src/expresso_calib/server.py tests/test_calibration_accumulator.py tests/test_camera_pipeline.py pyproject.toml uv.lock
git commit -m "$(cat <<'EOF'
fix: lock accumulator and fan out WS broadcast

Detection loop and solver thread were mutating CalibrationAccumulator
.candidates without synchronization. Add an asyncio.Lock on the
accumulator and acquire it around observe + commit_solve_result in
ManagedCamera. The snapshot for solve_snapshot is taken under the
lock as well so the worker thread reads a consistent list.

MetricsHub.broadcast was awaiting send_text serially per client, so a
slow client stalled every pipeline producer. Switch to per-client
bounded queues (maxsize=4) with drop-oldest on overflow plus a pump
task; broadcast() now only blocks on queue.put_nowait. New test
verifies broadcast completes within 100ms when one client sleeps 500ms.
EOF
)"
```

---

## Task 2: Lifecycle correctness — lifespan, `remove_camera` ordering, `MjpegCapture` cleanup

Three independent lifecycle bugs:
- No `app.on_event("shutdown")` / lifespan. `uvicorn`'s SIGTERM cancels the loop but the 5 tasks per camera plus the `asyncio.to_thread(capture.read)` workers keep the process alive until each read times out.
- `remove_camera` ([server.py:809-816](src/expresso_calib/server.py:809)) pops from `self.cameras` BEFORE awaiting `camera.stop()`. The popped camera's tasks still run, still broadcast, still hit the accumulator, until `stop()` finishes.
- `MjpegCapture.__init__` ([server.py:135-146](src/expresso_calib/server.py:135)) stores `self.response` from `urlopen` then immediately calls `_parse_boundary`. If `_parse_boundary` raises (it doesn't today, but if anything else above the `release()` call ever does), the HTTP connection leaks. Also, when `_capture_loop` is cancelled while blocked inside `asyncio.to_thread(capture.read)`, the thread keeps the socket open until the read returns.

**Files:**
- Modify: `src/expresso_calib/server.py` (lifespan, `remove_camera`, `MjpegCapture.__init__` + close)
- Test: `tests/test_camera_pipeline.py` (lifecycle tests)

- [ ] **Step 1: Write failing test for `remove_camera` stop-before-pop**

Add to `tests/test_camera_pipeline.py`:

```python
@pytest.mark.asyncio
async def test_remove_camera_stops_before_popping(tmp_path) -> None:
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
    assert live.cameras.get("cam-1") is None
```

- [ ] **Step 2: Run the test — should fail**

Expected: AssertionError because `remove_camera` pops first.

- [ ] **Step 3: Fix `remove_camera` ordering**

Replace `src/expresso_calib/server.py:809-816` with:

```python
async def remove_camera(self, camera_id: str) -> bool:
    camera = self.cameras.get(camera_id)
    if camera is None:
        return False
    await camera.stop()
    self.cameras.pop(camera_id, None)
    self.focus.clear_if_removed(set(self.cameras))
    await self.broadcast()
    return True
```

- [ ] **Step 4: Run test — should pass**

- [ ] **Step 5: Add FastAPI lifespan that stops all cameras on shutdown**

In `src/expresso_calib/server.py:create_app`, replace:

```python
def create_app() -> FastAPI:
    app = FastAPI(title="Expresso Calib", version="0.1.0")
    app.state.live = MultiCameraCalibrationState()
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
```

with:

```python
from contextlib import asynccontextmanager


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.live = MultiCameraCalibrationState()
        try:
            yield
        finally:
            await app.state.live.stop_all()

    app = FastAPI(title="Expresso Calib", version="0.1.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
```

Move the `from contextlib import asynccontextmanager` to the top-of-file import block.

- [ ] **Step 6: Add `close()` to `MjpegCapture` and call it from `release()` (or as `release` itself)**

Replace `MjpegCapture` `__init__` ([server.py:135-146](src/expresso_calib/server.py:135)) with try/except cleanup, and add an explicit `release` method:

```python
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
        self.boundary = self._parse_boundary(
            response.headers.get("Content-Type", "")
        )
    except Exception:
        response.close()
        raise
    self.response = response
    self.opened = True


def release(self) -> None:
    self.opened = False
    response = self.response
    self.response = None
    if response is not None:
        try:
            response.close()
        except Exception:
            pass
```

Find any existing `capture.release()` site in `_capture_loop` and leave it; ensure `release()` is also called from the camera's `stop()` path under the cancellation handler. Specifically, in `_capture_loop`, change:

```python
finally:
    capture.release()
```

to ensure `capture` is bound in scope and `release()` is idempotent (which the implementation above is).

- [ ] **Step 7: Add a test that lifespan tears down a camera task on shutdown**

Add to `tests/test_camera_pipeline.py`:

```python
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

    async with app.router.lifespan_context(app):
        app.state.live.cameras["cam-1"] = TrackingCamera()  # type: ignore[assignment]

    assert stopped.is_set()
```

- [ ] **Step 8: Run all tests**

```
uv run pytest -v
```

Expected: 30 PASS.

- [ ] **Step 9: Commit**

```bash
git add src/expresso_calib/server.py tests/test_camera_pipeline.py
git commit -m "$(cat <<'EOF'
fix: tidy camera lifecycle (lifespan, remove ordering, capture cleanup)

Three independent lifecycle bugs:
- create_app had no lifespan, so SIGTERM left the per-camera asyncio
  tasks running until their to_thread(capture.read) calls returned.
  Add a lifespan that calls live.stop_all() on shutdown.
- remove_camera popped the camera from the registry before awaiting
  stop(), so in-flight broadcasts and accumulator writes could still
  hit a camera that had already disappeared from the public state.
  Stop first, then pop.
- MjpegCapture.__init__ stored self.response from urlopen and then
  parsed boundary headers; if parsing raised, the HTTP socket leaked.
  Guard with try/close and add an explicit release() that nulls out
  the response so double-release is safe.
EOF
)"
```

---

## Task 3: Security caps — body size + MJPEG content-length

`POST /api/cameras`, `/api/reset`, `/api/target-metadata` call `await request.json()` with no `Content-Length` guard ([server.py:987,996](src/expresso_calib/server.py:987)). Trivial memory DoS for any operator with API access. Separately, MJPEG `Content-Length` is attacker-controlled and `response.read(content_length)` allocates upfront ([server.py:200-203](src/expresso_calib/server.py:200)) — a malicious or malformed upstream can pin gigabytes.

**Files:**
- Modify: `src/expresso_calib/server.py` (middleware for body cap, content-length cap in `_read_multipart_frame`)
- Test: `tests/test_camera_pipeline.py`

- [ ] **Step 1: Add constants for the caps**

Near `MJPEG_MAX_BUFFER_BYTES` ([server.py:47](src/expresso_calib/server.py:47)) add:

```python
MAX_REQUEST_BODY_BYTES = 64 * 1024
MJPEG_MAX_FRAME_BYTES = 8 * 1024 * 1024
```

- [ ] **Step 2: Add middleware to reject oversized request bodies**

In `create_app`, just after the `app = FastAPI(...)` line:

```python
@app.middleware("http")
async def cap_body_size(request: Request, call_next):
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > MAX_REQUEST_BODY_BYTES:
                return JSONResponse(
                    {"detail": "Request body too large."}, status_code=413
                )
        except ValueError:
            return JSONResponse(
                {"detail": "Invalid Content-Length."}, status_code=400
            )
    return await call_next(request)
```

- [ ] **Step 3: Cap the MJPEG content-length read**

In `MjpegCapture._read_multipart_frame` ([server.py:195-203](src/expresso_calib/server.py:195)), replace:

```python
content_length = _safe_int(headers.get("content-length"))
if content_length is None or content_length <= 0:
    self.boundary = None
    return False, None

jpeg = self.response.read(content_length)
```

with:

```python
content_length = _safe_int(headers.get("content-length"))
if content_length is None or content_length <= 0:
    self.boundary = None
    return False, None
if content_length > MJPEG_MAX_FRAME_BYTES:
    self.opened = False
    return False, None

jpeg = self.response.read(content_length)
```

- [ ] **Step 4: Write tests**

Add to `tests/test_camera_pipeline.py`:

```python
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
```

- [ ] **Step 5: Run tests + full suite**

```
uv run pytest -v
```

Expected: 32 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/expresso_calib/server.py tests/test_camera_pipeline.py
git commit -m "$(cat <<'EOF'
fix: cap untrusted body sizes on POST and MJPEG frame reads

Two unrelated DoS holes:
- POST /api/cameras, /api/reset, /api/target-metadata called
  request.json() with no size guard; any operator with API access
  could pin memory with a multi-GB body. Add a middleware that
  returns 413 when Content-Length > 64 KiB.
- MJPEG part Content-Length is attacker-controlled and
  response.read(content_length) allocates upfront; a malicious or
  malformed upstream could pin gigabytes via a single declared
  frame. Cap to 8 MiB and drop the connection.
EOF
)"
```

---

## Task 4: Delete dead code

- `sources.py` defines `CameraSource` protocol that nothing implements.
- `EphemeralCameraRegistry` in `multi_camera.py:18-38` is unused (the registry is the dict in `MultiCameraCalibrationState.cameras`).
- `/api/latest-frame.jpg` and `/api/latest-stream.mjpg` ([server.py:1073-1126](src/expresso_calib/server.py:1073)) plus `_preview_camera` helper at [server.py:1185-1190](src/expresso_calib/server.py:1185) are labeled `legacy_` / `include_in_schema=False`. The per-camera endpoints `/api/cameras/{camera_id}/latest.jpg` and `.../stream.mjpg` replace them. No frontend code references the legacy ones — grep `latest-frame\|latest-stream` in `src/expresso_calib/web/` to confirm.

**Files:**
- Delete: `src/expresso_calib/sources.py`
- Modify: `src/expresso_calib/multi_camera.py` (delete `EphemeralCameraRegistry`)
- Modify: `src/expresso_calib/server.py` (delete legacy endpoints + `_preview_camera`)
- Modify: `src/expresso_calib/__init__.py` (if it re-exports anything from sources)

- [ ] **Step 1: Confirm nothing references the dead code**

```
grep -rn "from .sources\|from expresso_calib.sources\|import sources" src/ tests/
grep -rn "EphemeralCameraRegistry" src/ tests/
grep -rn "latest-frame.jpg\|latest-stream.mjpg\|_preview_camera" src/ tests/
```

Expected: only the definitions themselves (sources.py, multi_camera.py, server.py). If frontend `web/operator.js` or any test references them, STOP and report — this task assumes they're truly dead.

- [ ] **Step 2: Delete `sources.py`**

```
rm src/expresso_calib/sources.py
```

- [ ] **Step 3: Delete `EphemeralCameraRegistry` from `multi_camera.py`**

Open `src/expresso_calib/multi_camera.py`, find the class definition and any related helpers used only by it, delete them. Leave `FocusTracker`, `clean_label`, `slugify_label`, and any other actively used exports alone.

- [ ] **Step 4: Delete legacy endpoints and `_preview_camera` from `server.py`**

Find and delete:
- The `legacy_latest_frame` route (currently around [server.py:1086](src/expresso_calib/server.py:1086))
- The `legacy_latest_stream` route (around [server.py:1097](src/expresso_calib/server.py:1097))
- The `_preview_camera` helper function (around [server.py:1185](src/expresso_calib/server.py:1185))
- Any `_is_self_preview_stream` branch that references `/api/latest-stream.mjpg` ([server.py:1180](src/expresso_calib/server.py:1180)) — narrow the check to only `/api/cameras/` paths.

- [ ] **Step 5: Run the full test suite**

```
uv run pytest -v
```

Expected: 32 PASS (no test depended on the deleted code).

- [ ] **Step 6: Compile-check + line-count win**

```
uv run python -m compileall src tests
wc -l src/expresso_calib/server.py src/expresso_calib/multi_camera.py
```

Expected: server.py drops ~55 lines; multi_camera.py drops ~20 lines.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
chore: delete dead code (sources.py, EphemeralCameraRegistry, legacy preview endpoints)

- sources.py defined a CameraSource protocol no module implemented.
- EphemeralCameraRegistry in multi_camera.py was unused; the real
  registry is the dict in MultiCameraCalibrationState.cameras.
- /api/latest-frame.jpg, /api/latest-stream.mjpg, _preview_camera
  predated the per-camera endpoints and are no longer wired into
  the operator UI. _is_self_preview_stream loses its branch for
  the deleted route.
EOF
)"
```

---

## Task 5: Split `ManagedCamera` into `CameraPipeline` + `CalibrationWorker`

`ManagedCamera` in `server.py` is ~533 lines: capture, MJPEG decoding, preview encoding, detection scheduling, solve scheduling, screenshot writing, broadcast throttling, AND public payload shaping. The natural seam is:

- **`CameraPipeline`** (`src/expresso_calib/camera_pipeline.py` — new): owns `MjpegCapture`, `_capture_loop`, `_preview_loop`, the latest-frame state (`latest_jpeg`, `latest_frame_at`, `latest_bgr`), the preview FPS state. Exposes a `frame_stream()` async iterator that yields decoded BGR frames to consumers.
- **`CalibrationWorker`** (`src/expresso_calib/calibration_worker.py` — new): owns the `CalibrationAccumulator`, `_detection_loop`, `_solver_loop`, `_screenshot_loop`, the related queues, the `generation` counter, `last_error`, `last_screenshot_path`.
- **`ManagedCamera`** (stays in `server.py`): composes a `CameraPipeline` + a `CalibrationWorker`, wires the frame stream from pipeline → worker, owns `public_snapshot`. Net: ~150 lines.

**Files:**
- Create: `src/expresso_calib/camera_pipeline.py` (~250 lines)
- Create: `src/expresso_calib/calibration_worker.py` (~300 lines)
- Modify: `src/expresso_calib/server.py` (shrink `ManagedCamera` to a composer; move `MjpegCapture`, `DetectionJob`, `ScreenshotJob`, `SolveJob` to the new modules)
- Modify: `tests/test_camera_pipeline.py` (imports + maybe targeted new tests)

**TDD note:** This is a refactor; the existing 32 tests are the safety net. Land the move task in one commit; if any existing test breaks the refactor is wrong.

- [ ] **Step 1: Sketch the public interfaces in the new files**

Create `src/expresso_calib/camera_pipeline.py`:

```python
from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

import cv2
import numpy as np
from urllib.request import Request as UrlRequest, urlopen

CAPTURE_OPEN_TIMEOUT_SEC = 8.0
CAPTURE_READ_TIMEOUT_SEC = 2.0
CAPTURE_RECONNECT_DELAY_SEC = 3.0
FRAME_STALE_SEC = 3.0
MJPEG_READ_CHUNK_BYTES = 4096
MJPEG_MAX_BUFFER_BYTES = 8 * 1024 * 1024
MJPEG_MAX_FRAME_BYTES = 8 * 1024 * 1024
PREVIEW_STREAM_FPS = 18


class MjpegCapture:
    # ... move the class from server.py verbatim, including the
    # release() + size-cap fixes from Tasks 2 and 3.


class CameraPipeline:
    def __init__(self, url: str) -> None:
        self.url = url
        self.latest_jpeg: bytes | None = None
        self.last_frame_at: float | None = None
        self.latest_bgr: Any = None
        self.frames_seen = 0
        self.running = False
        self.capture_task: asyncio.Task[None] | None = None
        self.preview_task: asyncio.Task[None] | None = None
        self._frame_event = asyncio.Event()
        self._fps_window: deque[float] = deque(maxlen=30)

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def frames(self): ...  # async generator yielding (bgr, timestamp_sec, frame_index)
    def fps(self) -> float | None: ...
```

Create `src/expresso_calib/calibration_worker.py`:

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .calibration import (
    CalibrationAccumulator,
    CandidateFrame,
    SolveInsufficientData,
    SolveNumericalFailure,
    SolveOk,
)
from .detection import CharucoDetector, DetectionResult


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
    consumed_new_frames: int


class CalibrationWorker:
    def __init__(self, accumulator: CalibrationAccumulator, screenshot_dir: Path) -> None:
        self.accumulator = accumulator
        self.screenshot_dir = screenshot_dir
        self.generation = 0
        self.detection_queue: asyncio.Queue[DetectionJob] = asyncio.Queue(maxsize=1)
        self.solve_queue: asyncio.Queue[SolveJob] = asyncio.Queue(maxsize=2)
        self.screenshot_queue: asyncio.Queue[ScreenshotJob] = asyncio.Queue(maxsize=8)
        self.last_error: str | None = None
        self.last_screenshot_path: str | None = None
        self.detection_running = False
        self.solver_running = False
        self.screenshot_running = False

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def submit_frame(self, job: DetectionJob) -> None: ...
```

- [ ] **Step 2: Move code from `ManagedCamera` into the new classes**

This is the bulk of the work. Walk through the existing `ManagedCamera` methods and reassign each to its new home:

| Existing method | New home |
|---|---|
| `_capture_loop`, `_preview_loop`, `fps`, `has_fresh_preview`, `connected` | `CameraPipeline` |
| `_detection_loop`, `_solver_loop`, `_screenshot_loop`, `_commit_solve_result`, `_enqueue_solve_if_due`, `_write_candidate_screenshot` | `CalibrationWorker` |
| `__init__`, `start`, `stop`, `reset_calibration`, `detecting_charuco`, `rms_value`, `public_snapshot` | `ManagedCamera` (composes the two) |

`ManagedCamera.start()` becomes:

```python
async def start(self) -> None:
    await self.pipeline.start()
    await self.worker.start()
    self._consume_task = asyncio.create_task(self._consume_frames())
    self.running = True

async def _consume_frames(self) -> None:
    async for bgr, ts, idx in self.pipeline.frames():
        await self.worker.submit_frame(
            DetectionJob(
                generation=self.worker.generation,
                frame_index=idx,
                timestamp_sec=ts,
                image_bgr=bgr,
            )
        )
```

- [ ] **Step 3: Update imports in `server.py`**

Replace the top-of-file MJPEG/job class definitions with:

```python
from .camera_pipeline import (
    CameraPipeline,
    MjpegCapture,
    CAPTURE_OPEN_TIMEOUT_SEC,
    CAPTURE_READ_TIMEOUT_SEC,
    CAPTURE_RECONNECT_DELAY_SEC,
    FRAME_STALE_SEC,
    MJPEG_MAX_BUFFER_BYTES,
    MJPEG_MAX_FRAME_BYTES,
    PREVIEW_STREAM_FPS,
)
from .calibration_worker import (
    CalibrationWorker,
    DetectionJob,
    ScreenshotJob,
    SolveJob,
)
```

Delete the now-moved definitions from `server.py`.

- [ ] **Step 4: Run the full test suite**

```
uv run pytest -v
```

Expected: 32 PASS. Any existing test failure here means the refactor moved behavior. Investigate before adjusting tests.

- [ ] **Step 5: Compile-check + line-count win**

```
uv run python -m compileall src tests
wc -l src/expresso_calib/server.py src/expresso_calib/camera_pipeline.py src/expresso_calib/calibration_worker.py
```

Expected:
- `server.py`: ~1200 → ~700
- `camera_pipeline.py`: ~250
- `calibration_worker.py`: ~300

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: split ManagedCamera into CameraPipeline and CalibrationWorker

Move capture + preview + MjpegCapture into camera_pipeline.py and
detection + solve + screenshot loops + the calibration accumulator
ownership into calibration_worker.py. ManagedCamera in server.py
becomes a thin composer that wires the pipeline's frame stream into
the worker. Net effect: server.py shrinks ~500 lines and the
calibration runtime is independently testable from the streaming
runtime.
EOF
)"
```

---

## Verification checklist (after all five tasks)

- [ ] `uv run pytest -v` — all green (~32 tests)
- [ ] `uv run python -m compileall src tests` — no errors
- [ ] `git log --oneline c2dbc97..HEAD` — five plan-B commits in addition to plan-A
- [ ] `wc -l src/expresso_calib/server.py src/expresso_calib/camera_pipeline.py src/expresso_calib/calibration_worker.py` — server.py < 800 lines
- [ ] `grep -rn "EphemeralCameraRegistry\|legacy_latest" src/ tests/` — no hits
- [ ] `test src/expresso_calib/sources.py` && echo "STILL THERE" — should be absent
