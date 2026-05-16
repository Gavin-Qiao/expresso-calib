# Calibration Core Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make calibration semantics honest — `solve_snapshot` returns a typed outcome so the server can guide the operator on numerical failures, rejected outlier frames are explicitly flagged and re-scored against the refined model so reports stop mixing two model vintages, and report-writing lives outside `CalibrationAccumulator`.

**Architecture:** No new types directory; the typed `SolveOutcome` union lives next to `CalibrationSolveResult` in `calibration.py`. A new `src/expresso_calib/reports.py` absorbs the ~170 lines of CSV/JSON/Markdown/debug-image writing currently inside the accumulator; `CalibrationAccumulator.export()` becomes a one-line delegating wrapper so no caller breaks. Rescoring uses `cv2.solvePnP` per rejected frame against the refined intrinsics, so reports show what the *shipped* model actually says about each frame.

**Tech Stack:** Python 3.11+ (`match` statements, `Literal`/`Union` types), OpenCV (`cv2.calibrateCamera`, `cv2.solvePnP`, `cv2.projectPoints`), pytest.

**Out of scope (deferred to Plan B):** the `ManagedCamera` split, concurrency fixes around the accumulator data race, broadcast serialization, lifespan handlers. This plan ships honest solver semantics; Plan B ships a correct runtime around them.

---

### Task 1: Typed solver result (`SolveOutcome`)

`solve_snapshot` today returns `CalibrationSolveResult | None` and lets `cv2` exceptions propagate, which the server catches generically as a string `last_error` ([server.py:562-563](src/expresso_calib/server.py:562)). Replace with a tagged union so the server can distinguish "not enough data yet" from "OpenCV refused to solve this."

**Files:**
- Modify: `src/expresso_calib/calibration.py` (add types, change return shape of `_calibrate` and `solve_snapshot`, update `solve_if_due`)
- Modify: `src/expresso_calib/server.py:541-567` (handle typed outcome in `_solver_loop`)
- Test: `tests/test_calibration_accumulator.py` (add three new tests; update existing tests that assert on `solve_snapshot` return)

- [ ] **Step 1: Write failing tests for the three outcome variants**

Add to `tests/test_calibration_accumulator.py`:

```python
import cv2
import pytest

from expresso_calib.calibration import (
    SolveOk,
    SolveInsufficientData,
    SolveNumericalFailure,
)


def test_solve_snapshot_returns_insufficient_data_when_too_few_candidates(tmp_path) -> None:
    accumulator = CalibrationAccumulator(
        DEFAULT_BOARD, tmp_path, min_solve_frames=5, create_run_dir=False
    )
    outcome = accumulator.solve_snapshot([])
    assert isinstance(outcome, SolveInsufficientData)
    assert outcome.reason == "too few candidates"
    assert outcome.candidate_count == 0
    assert outcome.needed == 5


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
    for index, center in enumerate(
        [(0.2, 0.2), (0.7, 0.2), (0.2, 0.7), (0.7, 0.7)], start=1
    ):
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
    for index, center in enumerate(
        [(0.2, 0.2), (0.7, 0.2), (0.2, 0.7), (0.7, 0.7)], start=1
    ):
        accumulator.observe(
            fake_detection(frame_index=index, center_x=center[0], center_y=center[1]),
            image,
        )

    outcome = accumulator.solve_snapshot(list(accumulator.candidates))
    assert isinstance(outcome, SolveNumericalFailure)
    assert "rank-deficient input" in outcome.reason
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_calibration_accumulator.py::test_solve_snapshot_returns_insufficient_data_when_too_few_candidates tests/test_calibration_accumulator.py::test_solve_snapshot_returns_ok_on_successful_solve tests/test_calibration_accumulator.py::test_solve_snapshot_returns_numerical_failure_when_cv2_raises -v`

Expected: 3 failures with `ImportError: cannot import name 'SolveOk' from 'expresso_calib.calibration'`.

- [ ] **Step 3: Add the typed `SolveOutcome` to calibration.py**

Add after `CalibrationSolveResult` (around [calibration.py:46](src/expresso_calib/calibration.py:46)):

```python
from typing import Literal, Union


@dataclass(frozen=True)
class SolveOk:
    solve: CalibrationSolveResult
    kind: Literal["ok"] = "ok"


@dataclass(frozen=True)
class SolveInsufficientData:
    reason: str
    candidate_count: int
    needed: int
    kind: Literal["insufficient_data"] = "insufficient_data"


@dataclass(frozen=True)
class SolveNumericalFailure:
    reason: str
    kind: Literal["numerical_failure"] = "numerical_failure"


SolveOutcome = Union[SolveOk, SolveInsufficientData, SolveNumericalFailure]
```

- [ ] **Step 4: Change `solve_snapshot` to return `SolveOutcome`**

Replace [calibration.py:237-278](src/expresso_calib/calibration.py:237) with:

```python
def solve_snapshot(self, candidates: list[CandidateFrame]) -> SolveOutcome:
    if len(candidates) < self.min_solve_frames:
        return SolveInsufficientData(
            reason="too few candidates",
            candidate_count=len(candidates),
            needed=self.min_solve_frames,
        )

    solve_pool, solve_pool_stats = self._solve_pool(candidates)
    selected = self.select_diverse(
        solve_pool, self.max_calib_frames, mark_selected=False
    )
    if len(selected) < 4:
        return SolveInsufficientData(
            reason="solve pool too small",
            candidate_count=len(candidates),
            needed=max(4, self.min_solve_frames),
        )

    try:
        initial_calibration = self._calibrate(selected, assign_per_view_errors=False)
        selected, calibration, rejected_outliers, outlier_threshold = (
            self._refine_outlier_views(selected, initial_calibration)
        )
    except cv2.error as exc:
        return SolveNumericalFailure(reason=str(exc))
    except (ValueError, RuntimeError) as exc:
        return SolveNumericalFailure(reason=str(exc))

    quality = self.summarize_quality(
        selected, calibration, usable_frames=len(solve_pool)
    )
    quality.update(solve_pool_stats)
    quality["initialRmsReprojectionErrorPx"] = (
        initial_calibration.rms_reprojection_error_px
    )
    quality["outlierRejection"] = {
        "rejectedFrames": rejected_outliers,
        "thresholdPx": outlier_threshold,
        "initialSelectedFrames": len(selected) + rejected_outliers,
    }
    if rejected_outliers:
        quality["warnings"].append(
            f"Excluded {rejected_outliers} high-error selected frame"
            f"{'' if rejected_outliers == 1 else 's'} before final solve."
        )
        if quality.get("verdict") == "GOOD":
            quality["verdict"] = "MARGINAL"
    return SolveOk(
        solve=CalibrationSolveResult(
            calibration=calibration,
            quality=quality,
            selected=selected,
            candidate_count=len(candidates),
        )
    )
```

- [ ] **Step 5: Update `solve_if_due` for the new return type**

Replace [calibration.py:226-235](src/expresso_calib/calibration.py:226) with:

```python
def solve_if_due(self, *, force: bool = False) -> CalibrationResult | None:
    if len(self.candidates) < self.min_solve_frames:
        return self.last_calibration
    if not force and not self.should_solve():
        return self.last_calibration

    outcome = self.solve_snapshot(list(self.candidates))
    if isinstance(outcome, SolveOk):
        return self.commit_solve_result(outcome.solve)
    return self.last_calibration
```

- [ ] **Step 6: Run new tests to verify they pass**

Run: `uv run pytest tests/test_calibration_accumulator.py -v`

Expected: all tests PASS. Two existing tests assert on `solve_snapshot` return shape (`test_snapshot_solve_preserves_samples_collected_while_solving`, `test_snapshot_solve_rejects_high_error_selected_outlier`, `test_snapshot_solve_prefers_strong_frames_when_available`) — update them in the next step.

- [ ] **Step 7: Migrate existing snapshot tests to unwrap `SolveOk`**

In `tests/test_calibration_accumulator.py`, replace `result = accumulator.solve_snapshot(...)` followed by `assert result is not None` with:

```python
outcome = accumulator.solve_snapshot(list(accumulator.candidates))
assert isinstance(outcome, SolveOk)
result = outcome.solve
```

There are three call sites: `test_snapshot_solve_preserves_samples_collected_while_solving` (one call), `test_snapshot_solve_rejects_high_error_selected_outlier` (one call), `test_snapshot_solve_prefers_strong_frames_when_available` (one call). Update the call site in each test and pass `result` to `commit_solve_result` as before.

Run: `uv run pytest tests/test_calibration_accumulator.py -v`

Expected: all tests PASS.

- [ ] **Step 8: Adapt the server's solver loop to the typed outcome**

Replace [server.py:548-567](src/expresso_calib/server.py:548) with:

```python
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
        case SolveOk(solve=solve):
            self._commit_solve_result(job, solve)
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
```

Add the imports near the top of `server.py` (find the existing `from .calibration import ...` line):

```python
from .calibration import (
    CalibrationAccumulator,
    CalibrationSolveResult,
    CandidateFrame,
    SolveInsufficientData,
    SolveNumericalFailure,
    SolveOk,
)
```

Drop the now-unused outer `except Exception` — the solver no longer raises for known failure modes; an unexpected exception should propagate and fail the test or log loudly rather than be silently swallowed.

- [ ] **Step 9: Run the full test suite**

Run: `uv run pytest -v`

Expected: all tests PASS.

- [ ] **Step 10: Compile-check**

Run: `uv run python -m compileall src tests`

Expected: no errors.

- [ ] **Step 11: Commit**

```bash
git add src/expresso_calib/calibration.py src/expresso_calib/server.py tests/test_calibration_accumulator.py
git commit -m "$(cat <<'EOF'
feat(calibration): return typed SolveOutcome from solve_snapshot

Distinguish "insufficient data" from "OpenCV refused to solve this" so
the server can stop swallowing cv2 exceptions as opaque last_error
strings. Server's _solver_loop now matches on SolveOk/SolveInsufficient
Data/SolveNumericalFailure instead of catching every Exception.
EOF
)"
```

---

### Task 2: Rejected-frame flag + rescoring against the refined model

After `_refine_outlier_views` drops outliers and recalibrates on the kept subset, today the rejected frames keep their pre-refine `per_view_error_px` ([calibration.py:351-376](src/expresso_calib/calibration.py:351)). Reports mix two model vintages. Fix: re-score every rejected frame via `cv2.solvePnP` against the refined intrinsics, and flag it with `rejected=True` so consumers know the solver excluded it from the final fit.

**Files:**
- Modify: `src/expresso_calib/calibration.py` (`CandidateFrame.rejected`, `CalibrationSolveResult.rejected*`, `_refine_outlier_views`, `commit_solve_result`, new `_project_per_view_error`)
- Test: `tests/test_calibration_accumulator.py` (new tests covering the flag and the rescore)

- [ ] **Step 1: Write a failing test that asserts rejected frames are flagged and rescored**

Add to `tests/test_calibration_accumulator.py`:

```python
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
        return 3.75  # rescored against refined model, lower than the pre-refine 4.20

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
    accumulator.observe(
        fake_detection(frame_index=1, center_x=0.5, center_y=0.5), image
    )
    assert accumulator.candidates[-1].rejected is False
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_calibration_accumulator.py::test_refine_outliers_marks_rejected_frames_and_rescores_them tests/test_calibration_accumulator.py::test_candidate_frame_default_rejected_is_false -v`

Expected: failures referencing `AttributeError: 'CandidateFrame' object has no attribute 'rejected'` or `AttributeError: 'CalibrationAccumulator' object has no attribute '_project_per_view_error'`.

- [ ] **Step 3: Add `rejected` to `CandidateFrame`**

In [calibration.py:20-27](src/expresso_calib/calibration.py:20), change:

```python
@dataclass
class CandidateFrame:
    detection: DetectionResult
    image_bgr: np.ndarray
    accepted_at: str
    image_signature: np.ndarray | None = None
    per_view_error_px: float | None = None
    selected: bool = False
    rejected: bool = False
```

- [ ] **Step 4: Extend `CalibrationSolveResult` with rejected frames + their rescored errors**

In [calibration.py:40-46](src/expresso_calib/calibration.py:40), change:

```python
@dataclass
class CalibrationSolveResult:
    calibration: CalibrationResult
    quality: dict[str, Any]
    selected: list[CandidateFrame]
    rejected: list[CandidateFrame]
    rejected_per_view_errors_px: list[float | None]
    candidate_count: int
```

- [ ] **Step 5: Add the per-view error projection helper**

Add as a new method on `CalibrationAccumulator`, placed right after `_compute_per_view_errors` (around [calibration.py:540](src/expresso_calib/calibration.py:540)):

```python
def _project_per_view_error(
    self,
    candidate: CandidateFrame,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float | None:
    detection = candidate.detection
    if detection.ids is None or detection.corners is None:
        return None
    object_corners = board_chessboard_corners(self.board)
    ids = np.asarray(detection.ids, dtype=np.int32).reshape(-1)
    object_points = object_corners[ids]
    image_points = np.asarray(detection.corners, dtype=np.float32).reshape(-1, 2)
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
    )
    if not success:
        return None
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    projected = projected.reshape(-1, 2)
    return float(
        np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1)))
    )
```

- [ ] **Step 6: Update `_refine_outlier_views` to return rejected frames + rescored errors**

Replace [calibration.py:351-376](src/expresso_calib/calibration.py:351) with:

```python
def _refine_outlier_views(
    self, selected: list[CandidateFrame], calibration: CalibrationResult
) -> tuple[
    list[CandidateFrame],
    CalibrationResult,
    list[CandidateFrame],
    list[float | None],
    float | None,
]:
    errors = calibration.per_view_errors_px
    if len(selected) < self.min_outlier_refine_frames or len(errors) != len(selected):
        return selected, calibration, [], [], None

    median_error = percentile(errors, 50)
    if median_error is None:
        return selected, calibration, [], [], None

    threshold = max(
        self.outlier_error_floor,
        median_error * self.outlier_error_median_factor,
    )
    kept: list[CandidateFrame] = []
    rejected: list[CandidateFrame] = []
    for item, error in zip(selected, errors):
        if float(error) <= threshold:
            kept.append(item)
        else:
            rejected.append(item)
    minimum_kept = max(4, min(self.min_outlier_refine_frames, len(selected)))
    if not rejected or len(kept) < minimum_kept:
        return selected, calibration, [], [], threshold

    refined = self._calibrate(kept, assign_per_view_errors=True)
    rejected_errors = [
        self._project_per_view_error(
            item, refined.camera_matrix, refined.distortion_coefficients
        )
        for item in rejected
    ]
    return kept, refined, rejected, rejected_errors, threshold
```

Update [calibration.py:251-265](src/expresso_calib/calibration.py:251) (inside `solve_snapshot`) to unpack the new tuple and feed the new `CalibrationSolveResult` fields:

```python
(
    selected,
    calibration,
    rejected,
    rejected_per_view_errors,
    outlier_threshold,
) = self._refine_outlier_views(selected, initial_calibration)
```

And in the same block, update the `quality["outlierRejection"]` build (around [calibration.py:261-265](src/expresso_calib/calibration.py:261)) and the `SolveOk` return at the end:

```python
quality["outlierRejection"] = {
    "rejectedFrames": len(rejected),
    "thresholdPx": outlier_threshold,
    "initialSelectedFrames": len(selected) + len(rejected),
}
if rejected:
    quality["warnings"].append(
        f"Excluded {len(rejected)} high-error selected frame"
        f"{'' if len(rejected) == 1 else 's'} before final solve."
    )
    if quality.get("verdict") == "GOOD":
        quality["verdict"] = "MARGINAL"
return SolveOk(
    solve=CalibrationSolveResult(
        calibration=calibration,
        quality=quality,
        selected=selected,
        rejected=rejected,
        rejected_per_view_errors_px=rejected_per_view_errors,
        candidate_count=len(candidates),
    )
)
```

- [ ] **Step 7: Update `commit_solve_result` to plumb the rejected list and its errors**

Replace [calibration.py:306-349](src/expresso_calib/calibration.py:306) with:

```python
def commit_solve_result(
    self,
    result: CalibrationSolveResult,
    *,
    consumed_new_frames: int | None = None,
) -> CalibrationResult:
    selected_ids = {id(item) for item in result.selected}
    rejected_ids = {id(item) for item in result.rejected}
    per_view_selected = {
        id(item): error
        for item, error in zip(result.selected, result.calibration.per_view_errors_px)
    }
    per_view_rejected = {
        id(item): error
        for item, error in zip(result.rejected, result.rejected_per_view_errors_px)
    }
    for item in self.candidates:
        item.selected = id(item) in selected_ids
        item.rejected = id(item) in rejected_ids
        if id(item) in per_view_selected:
            item.per_view_error_px = per_view_selected[id(item)]
        elif id(item) in per_view_rejected:
            item.per_view_error_px = per_view_rejected[id(item)]

    calibration = result.calibration
    self.last_calibration = calibration
    self.last_quality = result.quality
    self.k_history.append(calibration.camera_matrix.astype(float).tolist())
    self.k_history = self.k_history[-8:]
    k = calibration.camera_matrix
    self.solve_history.append(
        {
            "index": len(self.solve_history) + 1,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "candidateFrames": result.candidate_count,
            "selectedFrames": calibration.selected_count,
            "rmsReprojectionErrorPx": calibration.rms_reprojection_error_px,
            "fx": float(k[0, 0]),
            "fy": float(k[1, 1]),
            "cx": float(k[0, 2]),
            "cy": float(k[1, 2]),
        }
    )
    if consumed_new_frames is None:
        self.accepted_since_solve = 0
    else:
        self.accepted_since_solve = max(
            0, self.accepted_since_solve - consumed_new_frames
        )
    if self.auto_export:
        self.export()
    return calibration
```

- [ ] **Step 8: Update existing tests that construct `CalibrationSolveResult` directly**

The fake `_calibrate` returns inside `test_calibration_accumulator.py` use the real `_refine_outlier_views`; the new `rejected` / `rejected_per_view_errors_px` fields are populated by the accumulator, not the test. No fake `CalibrationSolveResult` instances exist in tests today (grep to confirm: `grep -n "CalibrationSolveResult(" tests/`). If any are added in Task 1's test updates, add `rejected=[]` and `rejected_per_view_errors_px=[]`.

In `test_snapshot_solve_rejects_high_error_selected_outlier` (currently asserts `result.quality["outlierRejection"]["rejectedFrames"] == 1`), add:

```python
assert len(result.rejected) == 1
assert result.rejected[0].detection.frame_index == 3
```

- [ ] **Step 9: Run all calibration tests**

Run: `uv run pytest tests/test_calibration_accumulator.py -v`

Expected: all tests PASS, including the two new ones from Step 1.

- [ ] **Step 10: Update CSV/JSON exports to surface the `rejected` flag**

In [calibration.py:766-783](src/expresso_calib/calibration.py:766) (`_candidate_json`), add `"rejected": item.rejected` to the payload:

```python
def _candidate_json(
    self, item: CandidateFrame, *, include_selected: bool = False
) -> dict[str, Any]:
    detection = item.detection
    payload = {
        "frame_index": detection.frame_index,
        "time_sec": detection.timestamp_sec,
        "charuco_corners": detection.charuco_count,
        "markers": detection.marker_count,
        "per_view_error_px": item.per_view_error_px,
        "area_fraction": detection.area_fraction,
        "center": [detection.center_x, detection.center_y],
        "sharpness": detection.sharpness,
        "accepted_at": item.accepted_at,
        "rejected": item.rejected,
    }
    if include_selected:
        payload["selected"] = item.selected
    return payload
```

In [calibration.py:785-825](src/expresso_calib/calibration.py:785) (`_write_detections_csv`), add a `rejected` column. Update the header row and each `writer.writerow(...)` to include `int(item.rejected)` right after `int(item.selected)`:

```python
writer.writerow(
    [
        "frame",
        "time_sec",
        "selected",
        "rejected",
        "markers",
        "charuco_corners",
        ...
    ]
)
for item in self.candidates:
    detection = item.detection
    writer.writerow(
        [
            detection.frame_index,
            f"{detection.timestamp_sec:.6f}",
            int(item.selected),
            int(item.rejected),
            detection.marker_count,
            detection.charuco_count,
            ...
        ]
    )
```

- [ ] **Step 11: Run the full suite + compile-check**

```bash
uv run pytest -v
uv run python -m compileall src tests
```

Expected: all PASS, no compile errors.

- [ ] **Step 12: Commit**

```bash
git add src/expresso_calib/calibration.py tests/test_calibration_accumulator.py
git commit -m "$(cat <<'EOF'
feat(calibration): rescore rejected outlier frames against refined model

After _refine_outlier_views drops high-error frames and recalibrates,
re-score the rejected frames via cv2.solvePnP against the refined
intrinsics. Add CandidateFrame.rejected so reports/exports can show
the dropped frames with errors that reflect the model actually shipped,
instead of the pre-refine model vintage.
EOF
)"
```

---

### Task 3: Extract export writers to `reports.py`

`CalibrationAccumulator` currently owns `export`, `_candidate_json`, `_write_detections_csv`, `_write_report`, `_write_debug_images` — ~170 lines of serialization mixed with the math+state class ([calibration.py:729-900](src/expresso_calib/calibration.py:729)). Extract them into a new module so the accumulator shrinks and Plan B's runtime split has a clean dependency edge.

**Files:**
- Create: `src/expresso_calib/reports.py`
- Modify: `src/expresso_calib/calibration.py` (delete extracted methods, keep `export()` as a one-line delegate)
- Test: existing tests cover the export path — verify they still pass.

- [ ] **Step 1: Create `src/expresso_calib/reports.py`**

```python
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2

from .board import BoardConfig
from .reference import MACBOOK_PRO_CAMERA_REFERENCE

if TYPE_CHECKING:
    from .calibration import CalibrationAccumulator, CandidateFrame


def build_calibration_payload(accumulator: "CalibrationAccumulator") -> dict[str, Any]:
    selected = accumulator.select_diverse(
        accumulator.candidates, accumulator.max_calib_frames
    )
    calibration = accumulator.last_calibration
    payload: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(accumulator.run_dir),
        "source": accumulator.source_id,
        "board": accumulator.board_config.manifest(),
        "target_metadata": accumulator.target_metadata,
        "macbook_reference": MACBOOK_PRO_CAMERA_REFERENCE,
        "quality": accumulator.last_quality,
        "calibration": None,
        "solve_history": accumulator.solve_history,
        "selected_frames": [
            _candidate_json(item, include_selected=True) for item in selected
        ],
    }
    if calibration is not None:
        payload["calibration"] = {
            "model": "opencv_plumb_bob",
            "rms_reprojection_error_px": calibration.rms_reprojection_error_px,
            "camera_matrix": calibration.camera_matrix.astype(float).tolist(),
            "distortion_coefficients": calibration.distortion_coefficients.astype(
                float
            ).tolist(),
            "flags": calibration.flags,
        }
    return payload


def _candidate_json(
    item: "CandidateFrame", *, include_selected: bool = False
) -> dict[str, Any]:
    detection = item.detection
    payload = {
        "frame_index": detection.frame_index,
        "time_sec": detection.timestamp_sec,
        "charuco_corners": detection.charuco_count,
        "markers": detection.marker_count,
        "per_view_error_px": item.per_view_error_px,
        "area_fraction": detection.area_fraction,
        "center": [detection.center_x, detection.center_y],
        "sharpness": detection.sharpness,
        "accepted_at": item.accepted_at,
        "rejected": item.rejected,
    }
    if include_selected:
        payload["selected"] = item.selected
    return payload


def write_calibration_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_detections_csv(candidates: list["CandidateFrame"], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "time_sec",
                "selected",
                "rejected",
                "markers",
                "charuco_corners",
                "per_view_error_px",
                "sharpness",
                "center_x",
                "center_y",
                "bbox_width",
                "bbox_height",
                "area_fraction",
                "angle_deg",
            ]
        )
        for item in candidates:
            detection = item.detection
            writer.writerow(
                [
                    detection.frame_index,
                    f"{detection.timestamp_sec:.6f}",
                    int(item.selected),
                    int(item.rejected),
                    detection.marker_count,
                    detection.charuco_count,
                    ""
                    if item.per_view_error_px is None
                    else f"{item.per_view_error_px:.6f}",
                    f"{detection.sharpness:.3f}",
                    f"{detection.center_x:.6f}",
                    f"{detection.center_y:.6f}",
                    f"{detection.bbox_width:.6f}",
                    f"{detection.bbox_height:.6f}",
                    f"{detection.area_fraction:.6f}",
                    f"{detection.angle_deg:.3f}",
                ]
            )


def write_report_md(
    payload: dict[str, Any], board_config: BoardConfig, path: Path
) -> None:
    quality = payload.get("quality") or {}
    calibration = payload.get("calibration") or {}
    lines = [
        "# Expresso Calib Intrinsic Report",
        "",
        f"Verdict: **{quality.get('verdict', 'PENDING')}**",
        "",
        f"Run: `{payload['run_dir']}`",
        f"Source: `{payload['source']}`",
        f"Board: `{board_config.squares_x}x{board_config.squares_y}` "
        f"{board_config.dictionary}, marker/square ratio "
        f"`{board_config.marker_length / board_config.square_length:.3f}`",
        "",
        "## Calibration",
        "",
    ]
    if calibration:
        lines.extend(
            [
                f"- RMS reprojection error: "
                f"`{calibration['rms_reprojection_error_px']:.4f} px`",
                f"- Selected frames: `{quality.get('selectedFrames', 0)}`",
                f"- Usable frames: `{quality.get('usableFrames', 0)}`",
                f"- Corner coverage: width "
                f"`{quality.get('coverage', {}).get('widthFraction', 0) * 100:.1f}%`, "
                f"height "
                f"`{quality.get('coverage', {}).get('heightFraction', 0) * 100:.1f}%`",
                "",
                "### Camera Matrix",
                "",
                "```json",
                json.dumps(calibration["camera_matrix"], indent=2),
                "```",
                "",
                "### Distortion Coefficients",
                "",
                "```json",
                json.dumps(calibration["distortion_coefficients"], indent=2),
                "```",
                "",
            ]
        )
    else:
        lines.extend(["Calibration has not run yet.", ""])

    if quality.get("redFlags"):
        lines.extend(["## Red Flags", ""])
        lines.extend(f"- {item}" for item in quality["redFlags"])
        lines.append("")
    if quality.get("warnings"):
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {item}" for item in quality["warnings"])
        lines.append("")
    lines.extend(
        [
            "## MacBook Reference",
            "",
            "The included MacBook intrinsic metadata is a manufacturer/sensor-space "
            "reference, not exact browser-frame ground truth.",
            "",
            "```json",
            json.dumps(MACBOOK_PRO_CAMERA_REFERENCE, indent=2),
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_debug_images(
    accumulator: "CalibrationAccumulator",
    selected: list["CandidateFrame"],
    debug_dir: Path,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    for item in selected:
        accumulator.write_candidate_screenshot(item, debug_dir)


def export_run(accumulator: "CalibrationAccumulator") -> Path:
    accumulator.run_dir.mkdir(parents=True, exist_ok=True)
    selected = accumulator.select_diverse(
        accumulator.candidates, accumulator.max_calib_frames
    )
    payload = build_calibration_payload(accumulator)
    write_calibration_json(payload, accumulator.run_dir / "calibration.json")
    write_detections_csv(accumulator.candidates, accumulator.run_dir / "detections.csv")
    write_report_md(payload, accumulator.board_config, accumulator.run_dir / "report.md")
    write_debug_images(accumulator, selected[:20], accumulator.run_dir / "debug")
    return accumulator.run_dir
```

- [ ] **Step 2: Replace `CalibrationAccumulator.export` with a delegating wrapper, delete the moved methods**

In `calibration.py`, replace the entire range [calibration.py:729-900](src/expresso_calib/calibration.py:729) (the `export`, `_candidate_json`, `_write_detections_csv`, `_write_report`, `_write_debug_images` methods) with:

```python
def export(self) -> Path:
    from .reports import export_run

    return export_run(self)
```

The `import cv2`, `import csv`, `import json` lines at the top of `calibration.py` should stay (cv2 is still used by `_calibrate` and `_project_per_view_error`; csv is no longer needed and json is no longer needed). Remove `import csv` and `import json` from the top of `calibration.py` after deleting the methods — they're now only used in `reports.py`.

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -v`

Expected: all tests PASS. Existing test `test_solve_does_not_export_artifacts_by_default` confirms export is not called automatically; no test calls `export()` directly today, so the rename of internal helpers should not break anything.

- [ ] **Step 4: Add a regression test that `export()` writes the expected files**

This guards the refactor itself. Add to `tests/test_calibration_accumulator.py`:

```python
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
    for index, center in enumerate(
        [(0.2, 0.2), (0.7, 0.2), (0.2, 0.7), (0.7, 0.7)], start=1
    ):
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
```

Add `import json` and `import pytest` to the test file if not already present (pytest is already there from Task 1; json may need adding).

- [ ] **Step 5: Run the new test**

Run: `uv run pytest tests/test_calibration_accumulator.py::test_export_writes_expected_files -v`

Expected: PASS.

- [ ] **Step 6: Full suite + compile-check**

```bash
uv run pytest -v
uv run python -m compileall src tests
```

Expected: all PASS, no errors.

- [ ] **Step 7: Verify line-count win**

Run: `wc -l src/expresso_calib/calibration.py src/expresso_calib/reports.py`

Expected: `calibration.py` drops from 900 to roughly 720; `reports.py` is roughly 200. (Approximate — confirm the totals make sense, not the exact numbers.)

- [ ] **Step 8: Commit**

```bash
git add src/expresso_calib/calibration.py src/expresso_calib/reports.py tests/test_calibration_accumulator.py
git commit -m "$(cat <<'EOF'
refactor(calibration): extract export writers to reports module

Move calibration.json / detections.csv / report.md / debug-image
writers out of CalibrationAccumulator into a new reports.py.
Accumulator.export() becomes a one-line delegate so callers
keep working. Adds a regression test that export() writes the
expected files.
EOF
)"
```

---

## Verification checklist (after all three tasks)

- [ ] `uv run pytest -v` — all green
- [ ] `uv run python -m compileall src tests` — no errors
- [ ] `node --check src/expresso_calib/web/operator.js` — no errors (no JS changes in this plan, but smoke-check)
- [ ] `wc -l src/expresso_calib/calibration.py` — should be ~720 (down from 900)
- [ ] `wc -l src/expresso_calib/reports.py` — should be ~200
- [ ] `git log --oneline -3` — three commits, one per task
- [ ] `grep -n "SolveOk\|SolveInsufficientData\|SolveNumericalFailure" src/expresso_calib/server.py` — confirms server consumes the typed result
- [ ] `grep -n "rejected" src/expresso_calib/calibration.py tests/test_calibration_accumulator.py` — confirms rejected field is plumbed
