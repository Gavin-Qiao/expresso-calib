from __future__ import annotations

import asyncio
import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .board import BoardConfig, board_chessboard_corners, create_board
from .detection import DetectionResult
from .reference import MACBOOK_PRO_CAMERA_REFERENCE, scaled_macbook_reference


@dataclass
class CandidateFrame:
    detection: DetectionResult
    image_bgr: np.ndarray
    accepted_at: str
    image_signature: np.ndarray | None = None
    per_view_error_px: float | None = None
    selected: bool = False
    rejected: bool = False


@dataclass
class CalibrationResult:
    rms_reprojection_error_px: float
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    per_view_errors_px: list[float]
    selected_count: int
    flags: int


@dataclass
class CalibrationSolveResult:
    calibration: CalibrationResult
    quality: dict[str, Any]
    selected: list[CandidateFrame]
    rejected: list[CandidateFrame]
    rejected_per_view_errors_px: list[float | None]
    candidate_count: int


@dataclass(frozen=True)
class SolveOk:
    solve: CalibrationSolveResult


@dataclass(frozen=True)
class SolveInsufficientData:
    reason: str


@dataclass(frozen=True)
class SolveNumericalFailure:
    reason: str


SolveOutcome = SolveOk | SolveInsufficientData | SolveNumericalFailure


def euclidean(a: Iterable[float], b: Iterable[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=False)))


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo))


def image_signature(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (24, 14), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)


def signature_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left - right)))


COVERAGE_CELLS_X = 8
COVERAGE_CELLS_Y = 6
COVERAGE_OCCUPANCY_REDO_BELOW = 0.30
COVERAGE_OCCUPANCY_MARGINAL_BELOW = 0.50
MAX_REFINEMENT_PASSES = 3

# Pose-diversity buckets. Calibration accuracy benefits from a wide spread of
# board orientations AND a wide spread of distances; we report bucket
# occupancy so the operator can see which combinations they haven't shown yet.
POSE_ANGLE_BUCKETS = 12  # 15-degree slices over [0, 180)
POSE_SCALE_BUCKET_LABELS = ("far", "mid", "near")
POSE_SCALE_BUCKET_EDGES = (0.04, 0.14)  # area_fraction boundaries: far<0.04<=mid<0.14<=near

# Convergence detector. "Converged" means K is stable AND RMS has plateaued
# AND the verdict is GOOD — i.e., the operator can stop. Other states tell
# them what's still wrong.
K_STABILITY_CONVERGED_PCT = 0.5  # < 0.5% relative span across last 5 solves' fx/fy/cx/cy
RMS_PLATEAU_RELATIVE_SPREAD = 0.05  # last-5 RMS spread within 5% of median = plateau
RMS_TREND_LOOKBACK = 5
MIN_SOLVES_FOR_CONVERGED = 5  # don't declare "converged" before this many solves


def compute_cell_occupancy_grid(
    selected: list[CandidateFrame],
    image_width: int,
    image_height: int,
    cells_x: int = COVERAGE_CELLS_X,
    cells_y: int = COVERAGE_CELLS_Y,
) -> set[tuple[int, int]]:
    occupied: set[tuple[int, int]] = set()
    if not selected or image_width <= 0 or image_height <= 0:
        return occupied
    for item in selected:
        corners = item.detection.corners
        if corners is None:
            continue
        pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        cell_x = np.clip((pts[:, 0] / image_width * cells_x).astype(int), 0, cells_x - 1)
        cell_y = np.clip((pts[:, 1] / image_height * cells_y).astype(int), 0, cells_y - 1)
        for cx, cy in zip(cell_x.tolist(), cell_y.tolist(), strict=False):
            occupied.add((int(cx), int(cy)))
    return occupied


def compute_pose_diversity(
    candidates: list[CandidateFrame],
    *,
    angle_buckets: int = POSE_ANGLE_BUCKETS,
    scale_edges: tuple[float, float] = POSE_SCALE_BUCKET_EDGES,
    scale_labels: tuple[str, str, str] = POSE_SCALE_BUCKET_LABELS,
) -> dict[str, Any]:
    angle_seen: set[int] = set()
    scale_seen: set[str] = set()
    for item in candidates:
        det = item.detection
        bucket = int((det.angle_deg % 180.0) / 180.0 * angle_buckets)
        angle_seen.add(min(bucket, angle_buckets - 1))
        if det.area_fraction < scale_edges[0]:
            scale_seen.add(scale_labels[0])
        elif det.area_fraction < scale_edges[1]:
            scale_seen.add(scale_labels[1])
        else:
            scale_seen.add(scale_labels[2])
    return {
        "angleBuckets": angle_buckets,
        "angleBucketsCovered": len(angle_seen),
        "scaleBuckets": len(scale_labels),
        "scaleBucketsCovered": len(scale_seen),
        "missingScale": [label for label in scale_labels if label not in scale_seen],
    }


def least_occupied_quadrant(
    occupied_cells: set[tuple[int, int]],
    cells_x: int = COVERAGE_CELLS_X,
    cells_y: int = COVERAGE_CELLS_Y,
) -> tuple[str, int] | None:
    if not occupied_cells:
        return None
    mid_x = cells_x // 2
    mid_y = cells_y // 2
    quads: dict[str, int] = {
        "upper-left": 0,
        "upper-right": 0,
        "lower-left": 0,
        "lower-right": 0,
    }
    for cx, cy in occupied_cells:
        key = ("upper" if cy < mid_y else "lower") + "-" + ("left" if cx < mid_x else "right")
        quads[key] += 1
    name = min(quads, key=lambda k: quads[k])
    return name, quads[name]


def compute_cell_occupancy(
    selected: list[CandidateFrame],
    image_width: int,
    image_height: int,
    cells_x: int = COVERAGE_CELLS_X,
    cells_y: int = COVERAGE_CELLS_Y,
) -> float:
    occupied = compute_cell_occupancy_grid(selected, image_width, image_height, cells_x, cells_y)
    return len(occupied) / (cells_x * cells_y)


class CalibrationAccumulator:
    def __init__(
        self,
        board_config: BoardConfig,
        runs_dir: Path,
        *,
        min_corners: int = 12,
        min_solve_frames: int = 15,
        solve_every_new_frames: int = 10,
        max_calib_frames: int = 80,
        max_candidates: int = 250,
        duplicate_pose_distance: float = 0.045,
        duplicate_image_pose_distance: float = 0.12,
        duplicate_image_distance: float = 2.0,
        outlier_error_floor: float = 1.25,
        outlier_error_median_factor: float = 2.5,
        min_outlier_refine_frames: int = 12,
        min_outlier_refine_keep: int | None = None,
        refinement_marginal_rejection_rate: float = 0.10,
        min_solve_corners: int | None = None,
        min_solve_area_fraction: float = 0.018,
        min_solve_sharpness: float = 25.0,
        auto_export: bool = False,
        create_run_dir: bool = True,
    ) -> None:
        self.board_config = board_config
        self.board = create_board(board_config)
        self.runs_dir = runs_dir
        self.min_corners = min_corners
        self.min_solve_frames = min_solve_frames
        self.solve_every_new_frames = solve_every_new_frames
        self.max_calib_frames = max_calib_frames
        self.max_candidates = max_candidates
        self.duplicate_pose_distance = duplicate_pose_distance
        self.duplicate_image_pose_distance = duplicate_image_pose_distance
        self.duplicate_image_distance = duplicate_image_distance
        self.outlier_error_floor = outlier_error_floor
        self.outlier_error_median_factor = outlier_error_median_factor
        self.min_outlier_refine_frames = min_outlier_refine_frames
        self.min_outlier_refine_keep = (
            min_outlier_refine_keep
            if min_outlier_refine_keep is not None
            else max(4, min_outlier_refine_frames // 2)
        )
        self.refinement_marginal_rejection_rate = refinement_marginal_rejection_rate
        self.min_solve_corners = min_solve_corners or max(
            min_corners, int(round(board_config.charuco_corners * 0.45))
        )
        self.min_solve_area_fraction = min_solve_area_fraction
        self.min_solve_sharpness = min_solve_sharpness
        self.auto_export = auto_export
        self.create_run_dir = create_run_dir
        self.run_dir = self._make_run_dir()
        self.candidates: list[CandidateFrame] = []
        self.lock = asyncio.Lock()
        self.last_detection: DetectionResult | None = None
        self.last_calibration: CalibrationResult | None = None
        self.last_quality: dict[str, Any] | None = None
        self.k_history: list[list[list[float]]] = []
        self.solve_history: list[dict[str, Any]] = []
        self.accepted_since_solve = 0
        self.total_frames_seen = 0
        self.duplicate_pose_rejections = 0
        self.duplicate_image_rejections = 0
        self.last_accept_reason = "waiting for frames"
        self.target_metadata: dict[str, Any] = {}
        self.source_id = "browser_upload"

    def reset(self) -> None:
        self.run_dir = self._make_run_dir()
        self.candidates = []
        self.last_detection = None
        self.last_calibration = None
        self.last_quality = None
        self.k_history = []
        self.solve_history = []
        self.accepted_since_solve = 0
        self.total_frames_seen = 0
        self.duplicate_pose_rejections = 0
        self.duplicate_image_rejections = 0
        self.last_accept_reason = "reset"

    def _make_run_dir(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.runs_dir / stamp
        if self.create_run_dir:
            run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def observe(self, detection: DetectionResult, image_bgr: np.ndarray) -> tuple[bool, str]:
        self.total_frames_seen += 1
        self.last_detection = detection

        if detection.charuco_count < self.min_corners:
            reason = "need more ChaRuCo corners"
            self.last_accept_reason = reason
            return False, reason
        if detection.area_fraction < 0.012:
            reason = "board is too small"
            self.last_accept_reason = reason
            return False, reason
        if detection.sharpness < 18.0:
            reason = "frame is too blurry"
            self.last_accept_reason = reason
            return False, reason

        feature = detection.feature_vector()
        nearest_pose = None
        if self.candidates:
            nearest_pose = min(
                euclidean(feature, item.detection.feature_vector()) for item in self.candidates
            )
            if nearest_pose < self.duplicate_pose_distance:
                reason = "duplicate pose"
                self.duplicate_pose_rejections += 1
                self.last_accept_reason = reason
                return False, reason

        signature = image_signature(image_bgr)
        if self.candidates and nearest_pose is not None:
            image_distances = [
                signature_distance(signature, item.image_signature)
                for item in self.candidates
                if item.image_signature is not None
            ]
            nearest_image = min(image_distances) if image_distances else None
            if (
                nearest_image is not None
                and nearest_pose < self.duplicate_image_pose_distance
                and nearest_image < self.duplicate_image_distance
            ):
                reason = "duplicate frame"
                self.duplicate_image_rejections += 1
                self.last_accept_reason = reason
                return False, reason

        self.candidates.append(
            CandidateFrame(
                detection=detection,
                image_bgr=image_bgr.copy(),
                accepted_at=datetime.now().isoformat(timespec="seconds"),
                image_signature=signature,
            )
        )
        self.accepted_since_solve += 1
        if len(self.candidates) > self.max_candidates:
            self.candidates = self.select_diverse(self.candidates, self.max_candidates)
        self.last_accept_reason = "accepted"
        return True, "accepted"

    def should_solve(self) -> bool:
        return (
            len(self.candidates) >= self.min_solve_frames
            and self.accepted_since_solve >= self.solve_every_new_frames
        )

    def solve_if_due(self, *, force: bool = False) -> CalibrationResult | None:
        if len(self.candidates) < self.min_solve_frames:
            return self.last_calibration
        if not force and not self.should_solve():
            return self.last_calibration

        outcome = self.solve_snapshot(list(self.candidates))
        if isinstance(outcome, SolveOk):
            return self.commit_solve_result(outcome.solve)
        return self.last_calibration

    def solve_snapshot(self, candidates: list[CandidateFrame]) -> SolveOutcome:
        if len(candidates) < self.min_solve_frames:
            return SolveInsufficientData(reason="too few candidates")

        solve_pool, solve_pool_stats = self._solve_pool(candidates)
        selected = self.select_diverse(solve_pool, self.max_calib_frames, mark_selected=False)
        if len(selected) < 4:
            return SolveInsufficientData(reason="solve pool too small")

        try:
            initial_calibration = self._calibrate(selected, assign_per_view_errors=False)
            (
                selected,
                calibration,
                rejected,
                rejected_per_view_errors,
                outlier_threshold,
            ) = self._refine_outlier_views(selected, initial_calibration)
        except (cv2.error, ValueError, RuntimeError) as exc:
            return SolveNumericalFailure(reason=str(exc))

        # Fallback safety net: a bug in summarize_quality / pose diversity /
        # convergence computation must NOT take down the worker. The K-matrix
        # solve already succeeded; emit it with a minimal quality dict so the
        # operator can still see RMS + export the calibration.
        try:
            quality = self.summarize_quality(selected, calibration, usable_frames=len(solve_pool))
            quality.update(solve_pool_stats)
            quality["initialRmsReprojectionErrorPx"] = initial_calibration.rms_reprojection_error_px
            outlier_rejection: dict[str, Any] = {"rejectedFrames": len(rejected)}
            if rejected:
                outlier_rejection["thresholdPx"] = outlier_threshold
                outlier_rejection["initialSelectedFrames"] = len(selected) + len(rejected)
            quality["outlierRejection"] = outlier_rejection
            if rejected:
                quality["warnings"].append(
                    f"Excluded {len(rejected)} high-error selected frame"
                    f"{'' if len(rejected) == 1 else 's'} before final solve."
                )
                rejection_rate = len(rejected) / max(1, len(selected) + len(rejected))
                if (
                    quality.get("verdict") == "GOOD"
                    and rejection_rate > self.refinement_marginal_rejection_rate
                ):
                    quality["verdict"] = "MARGINAL"
        except Exception as exc:
            quality = {
                "verdict": "UNKNOWN",
                "redFlags": [f"Quality computation failed: {exc}"],
                "warnings": [],
                "guidance": "Quality summary unavailable; calibration values are still valid.",
                "selectedFrames": len(selected),
            }
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

    def _solve_pool(
        self, candidates: list[CandidateFrame]
    ) -> tuple[list[CandidateFrame], dict[str, Any]]:
        strong = [
            item
            for item in candidates
            if item.detection.charuco_count >= self.min_solve_corners
            and item.detection.area_fraction >= self.min_solve_area_fraction
            and item.detection.sharpness >= self.min_solve_sharpness
        ]
        use_strong = len(strong) >= self.min_solve_frames
        pool = list(strong if use_strong else candidates)
        return pool, {
            "acceptedFrames": len(candidates),
            "solvePoolFrames": len(pool),
            "weakSolveFrames": max(0, len(candidates) - len(pool)),
            "minSolveCorners": self.min_solve_corners,
            "minSolveAreaFraction": self.min_solve_area_fraction,
            "minSolveSharpness": self.min_solve_sharpness,
            "usingStrongSolvePool": use_strong,
        }

    def solve_pool_stats(self) -> dict[str, Any]:
        _, stats = self._solve_pool(self.candidates)
        return stats

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
            for item, error in zip(
                result.selected, result.calibration.per_view_errors_px, strict=False
            )
        }
        per_view_rejected = {
            id(item): error
            for item, error in zip(
                result.rejected, result.rejected_per_view_errors_px, strict=False
            )
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
            self.accepted_since_solve = max(0, self.accepted_since_solve - consumed_new_frames)
        if self.auto_export:
            self.export()
        return calibration

    def _refine_outlier_views(
        self, selected: list[CandidateFrame], calibration: CalibrationResult
    ) -> tuple[
        list[CandidateFrame],
        CalibrationResult,
        list[CandidateFrame],
        list[float | None],
        float | None,
    ]:
        # Threshold = max(floor, median * factor). Works when outliers are a
        # minority; when outliers dominate (~50%+) the median sits among the
        # bad frames and the threshold balloons above them, suppressing
        # rejection. In the operator workflow that case is already a
        # "redo from scratch" situation that the verdict catches.
        #
        # Iterates up to MAX_REFINEMENT_PASSES: after refinement the new
        # model may surface frames that were borderline before. Stops as soon
        # as a pass finds no new outliers or would leave too few kept frames.
        if len(selected) < self.min_outlier_refine_frames:
            return selected, calibration, [], [], None

        current_selected = selected
        current_calibration = calibration
        rejected_all: list[CandidateFrame] = []
        last_threshold: float | None = None

        for _pass in range(MAX_REFINEMENT_PASSES):
            errors = current_calibration.per_view_errors_px
            if len(errors) != len(current_selected):
                break
            median_error = percentile(errors, 50)
            if median_error is None:
                break
            threshold = max(
                self.outlier_error_floor,
                median_error * self.outlier_error_median_factor,
            )
            last_threshold = threshold
            kept: list[CandidateFrame] = []
            pass_rejected: list[CandidateFrame] = []
            for item, error in zip(current_selected, errors, strict=False):
                if float(error) <= threshold:
                    kept.append(item)
                else:
                    pass_rejected.append(item)
            if not pass_rejected:
                break
            if len(kept) < self.min_outlier_refine_keep:
                break
            try:
                current_calibration = self._calibrate(kept, assign_per_view_errors=False)
            except (cv2.error, ValueError, RuntimeError):
                break
            current_selected = kept
            rejected_all.extend(pass_rejected)

        if not rejected_all:
            return selected, calibration, [], [], None
        rejected_errors = [
            self._project_per_view_error(
                item, current_calibration.camera_matrix, current_calibration.distortion_coefficients
            )
            for item in rejected_all
        ]
        return current_selected, current_calibration, rejected_all, rejected_errors, last_threshold

    def write_candidate_screenshot(self, item: CandidateFrame, screenshot_dir: Path) -> Path:
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        detection = item.detection
        image = item.image_bgr.copy()
        if detection.corners is not None and detection.ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(image, detection.corners, detection.ids)
        path = screenshot_dir / f"frame_{detection.frame_index:06d}.jpg"
        cv2.imwrite(str(path), image)
        return path

    def select_diverse(
        self,
        candidates: list[CandidateFrame],
        max_frames: int,
        *,
        mark_selected: bool = True,
    ) -> list[CandidateFrame]:
        if mark_selected:
            for item in candidates:
                item.selected = False
        if len(candidates) <= max_frames:
            if mark_selected:
                for item in candidates:
                    item.selected = True
            return list(candidates)

        # Normalize sharpness against the best-in-set, not a hardcoded constant.
        # Different cameras produce wildly different Laplacian-variance scales;
        # the old 450.0 cap saturated for any camera with naturally crisp focus
        # and under-represented soft-focus cameras.
        sharp_cap = max((c.detection.sharpness for c in candidates), default=1.0)
        sharp_cap = max(sharp_cap, 1.0)

        def quality(item: CandidateFrame) -> float:
            detection = item.detection
            corner_score = detection.charuco_count / max(1, self.board_config.charuco_corners)
            sharp_score = min(1.0, detection.sharpness / sharp_cap)
            area_score = min(1.0, detection.area_fraction / 0.25)
            edge_score = max(abs(detection.center_x - 0.5), abs(detection.center_y - 0.5))
            return corner_score * 0.55 + sharp_score * 0.12 + area_score * 0.23 + edge_score * 0.10

        remaining = list(candidates)
        first = max(remaining, key=quality)
        selected = [first]
        remaining.remove(first)

        # Diversity is primary; quality is a multiplicative tiebreaker. The old
        # additive `+ quality * 0.18` could swamp tight clusters (where the
        # diversity term is ~0.05) and pull near-duplicates ahead of truly
        # diverse-but-mediocre frames.
        while remaining and len(selected) < max_frames:
            next_item = max(
                remaining,
                key=lambda item: (
                    min(
                        euclidean(
                            item.detection.feature_vector(),
                            chosen.detection.feature_vector(),
                        )
                        for chosen in selected
                    )
                    * (1.0 + 0.2 * quality(item))
                ),
            )
            selected.append(next_item)
            remaining.remove(next_item)

        selected.sort(key=lambda item: item.detection.frame_index)
        if mark_selected:
            selected_ids = {id(item) for item in selected}
            for item in candidates:
                item.selected = id(item) in selected_ids
        return selected

    def _calibrate(
        self,
        selected: list[CandidateFrame],
        *,
        assign_per_view_errors: bool = True,
    ) -> CalibrationResult:
        for index, item in enumerate(selected):
            if item.detection.ids is None or item.detection.corners is None:
                raise ValueError(
                    f"_calibrate received selected[{index}] with no ids/corners; "
                    "caller must filter incomplete frames before solving."
                )
        object_points, image_points = self._calibration_points(selected)
        first = selected[0].detection
        image_size = (first.width, first.height)

        if hasattr(cv2, "calibrateCameraExtended"):
            result = cv2.calibrateCameraExtended(
                object_points,
                image_points,
                image_size,
                None,
                None,
                flags=0,
            )
            rms, camera_matrix, dist_coeffs, rvecs, tvecs = result[:5]
            per_view = result[7] if len(result) > 7 else None
        else:
            rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                object_points,
                image_points,
                image_size,
                None,
                None,
                flags=0,
            )
            per_view = None

        computed_per_view = self._compute_per_view_errors(
            selected, camera_matrix, dist_coeffs, rvecs, tvecs
        )
        if per_view is not None:
            flattened = [float(x) for x in np.asarray(per_view).reshape(-1)]
            if len(flattened) == len(selected):
                computed_per_view = flattened
        if assign_per_view_errors:
            for candidate, error in zip(selected, computed_per_view, strict=False):
                candidate.per_view_error_px = error

        return CalibrationResult(
            rms_reprojection_error_px=float(rms),
            camera_matrix=np.asarray(camera_matrix, dtype=float),
            distortion_coefficients=np.asarray(dist_coeffs, dtype=float).reshape(-1),
            per_view_errors_px=computed_per_view,
            selected_count=len(selected),
            flags=0,
        )

    def _calibration_points(
        self, selected: list[CandidateFrame]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        object_corners = board_chessboard_corners(self.board)
        object_points: list[np.ndarray] = []
        image_points: list[np.ndarray] = []
        for item in selected:
            detection = item.detection
            if detection.ids is None or detection.corners is None:
                continue
            ids = np.asarray(detection.ids, dtype=np.int32).reshape(-1)
            object_points.append(object_corners[ids].reshape(-1, 1, 3))
            image_points.append(np.asarray(detection.corners, dtype=np.float32).reshape(-1, 1, 2))
        return object_points, image_points

    def _compute_per_view_errors(
        self,
        selected: list[CandidateFrame],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        rvecs: Any,
        tvecs: Any,
    ) -> list[float]:
        object_corners = board_chessboard_corners(self.board)
        errors: list[float] = []
        for item, rvec, tvec in zip(selected, rvecs, tvecs, strict=False):
            detection = item.detection
            if detection.ids is None or detection.corners is None:
                continue
            ids = np.asarray(detection.ids, dtype=np.int32).reshape(-1)
            object_points = object_corners[ids]
            image_points = np.asarray(detection.corners, dtype=np.float32).reshape(-1, 2)
            projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
            projected = projected.reshape(-1, 2)
            error = np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1)))
            errors.append(float(error))
        return errors

    def _project_per_view_error(
        self,
        candidate: CandidateFrame,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> float | None:
        # Per-view error for an outlier-rejected frame: solve PnP for the best
        # rvec/tvec given the FINAL intrinsics, then reproject. This is a
        # lower bound on the joint-fit error the frame would have shown if it
        # were still in the solver pool — the joint solve constrains all
        # rvec/tvec to share K, while PnP gets to pick the best pose per frame.
        # Kept-frame errors (in `result.calibration.per_view_errors_px`) come
        # from the joint solve and are not directly comparable; treat the
        # rejected error as "what's the smallest residual we can achieve for
        # this frame against the shipped model?"
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
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        projected = projected.reshape(-1, 2)
        return float(np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))))

    def summarize_quality(
        self,
        selected: list[CandidateFrame],
        calibration: CalibrationResult,
        *,
        usable_frames: int | None = None,
    ) -> dict[str, Any]:
        first = selected[0].detection
        width = max(1, first.width)
        height = max(1, first.height)
        all_points = np.concatenate(
            [
                np.asarray(item.detection.corners, dtype=np.float32).reshape(-1, 2)
                for item in selected
                if item.detection.corners is not None
            ],
            axis=0,
        )
        coverage_width = float((all_points[:, 0].max() - all_points[:, 0].min()) / width)
        coverage_height = float((all_points[:, 1].max() - all_points[:, 1].min()) / height)
        occupied_cells = compute_cell_occupancy_grid(selected, width, height)
        cell_occupancy = len(occupied_cells) / (COVERAGE_CELLS_X * COVERAGE_CELLS_Y)
        pose_diversity = compute_pose_diversity(selected)
        weakest_quadrant = least_occupied_quadrant(occupied_cells)
        edge_margin = {
            "left": float(all_points[:, 0].min() / width),
            "right": float((width - all_points[:, 0].max()) / width),
            "top": float(all_points[:, 1].min() / height),
            "bottom": float((height - all_points[:, 1].max()) / height),
        }
        corner_counts = [item.detection.charuco_count for item in selected]
        areas = [item.detection.area_fraction for item in selected]
        per_view = calibration.per_view_errors_px

        red: list[str] = []
        yellow: list[str] = []
        if len(selected) < 15:
            red.append("Too few calibration frames; target at least 25 usable views.")
        elif len(selected) < 25:
            yellow.append("Usable frame count is low; 35-80 diverse views is better.")
        if calibration.rms_reprojection_error_px > 1.20:
            red.append("RMS reprojection error is above 1.20 px.")
        elif calibration.rms_reprojection_error_px > 0.80:
            yellow.append("RMS reprojection error is marginal; below 0.80 px is preferred.")
        p95 = percentile(per_view, 95)
        if p95 is not None:
            if p95 > 1.80:
                red.append("95th percentile per-view error is high.")
            elif p95 > 1.20:
                yellow.append("A few selected views have elevated reprojection error.")
        if cell_occupancy < COVERAGE_OCCUPANCY_REDO_BELOW:
            red.append(
                f"Image cell coverage is too low "
                f"({cell_occupancy * 100:.0f}%); spread the board across more of the frame."
            )
        elif cell_occupancy < COVERAGE_OCCUPANCY_MARGINAL_BELOW:
            yellow.append(
                f"Image cell coverage is limited "
                f"({cell_occupancy * 100:.0f}%); aim for >= "
                f"{int(COVERAGE_OCCUPANCY_MARGINAL_BELOW * 100)}%."
            )
        if areas:
            if max(areas) < 0.14:
                yellow.append("The board never gets very close; include close-up views.")
            area_ratio = max(areas) / max(min(areas), 1e-9)
            if area_ratio < 1.8:
                yellow.append("Board scale does not vary much; mix near and far views.")
        if corner_counts and statistics.median(corner_counts) < 18:
            yellow.append("Median ChaRuCo corner count is low; show more of the board.")
        # Pose diversity is the operator-trust signal: 25 same-pose frames can
        # otherwise produce GOOD without actually constraining the K matrix.
        scale_missing = pose_diversity.get("missingScale") or []
        if scale_missing:
            yellow.append(
                f"Missing distance band{'s' if len(scale_missing) > 1 else ''}: "
                f"{', '.join(scale_missing)}. Show the board at varied distances."
            )
        angle_covered = pose_diversity.get("angleBucketsCovered", 0)
        angle_total = pose_diversity.get("angleBuckets", 1)
        if angle_covered < angle_total * 0.25:
            red.append(
                f"Too few orientations sampled ({angle_covered}/{angle_total}); "
                "rotate the board more as you move it."
            )
        elif angle_covered < angle_total * 0.5:
            yellow.append(
                f"Orientation variety is limited ({angle_covered}/{angle_total}); "
                "add more tilted views."
            )

        verdict = "GOOD"
        if red:
            verdict = "REDO"
        elif yellow:
            verdict = "MARGINAL"

        convergence = self.compute_convergence(verdict=verdict)
        guidance = self._live_guidance(
            verdict=verdict,
            convergence=convergence,
            pose_diversity=pose_diversity,
            cell_occupancy=cell_occupancy,
            weakest_quadrant=weakest_quadrant,
        )

        return {
            "verdict": verdict,
            "redFlags": red,
            "warnings": yellow,
            "guidance": guidance,
            "convergence": convergence,
            "poseDiversity": pose_diversity,
            "selectedFrames": len(selected),
            "usableFrames": len(self.candidates) if usable_frames is None else usable_frames,
            "cornerCount": {
                "min": min(corner_counts) if corner_counts else 0,
                "median": float(statistics.median(corner_counts)) if corner_counts else 0.0,
                "max": max(corner_counts) if corner_counts else 0,
            },
            "perViewErrorPx": {
                "median": percentile(per_view, 50),
                "p95": p95,
                "max": max(per_view) if per_view else None,
            },
            "coverage": {
                "widthFraction": coverage_width,
                "heightFraction": coverage_height,
                "cellOccupancyFraction": cell_occupancy,
                "cellGrid": [COVERAGE_CELLS_X, COVERAGE_CELLS_Y],
                "weakestQuadrant": weakest_quadrant[0] if weakest_quadrant else None,
                "edgeMarginFraction": edge_margin,
            },
            "boardAreaFraction": {
                "min": min(areas) if areas else 0.0,
                "median": float(statistics.median(areas)) if areas else 0.0,
                "max": max(areas) if areas else 0.0,
            },
        }

    def _live_guidance(
        self,
        *,
        verdict: str,
        convergence: dict[str, Any],
        pose_diversity: dict[str, Any],
        cell_occupancy: float,
        weakest_quadrant: tuple[str, int] | None,
    ) -> str:
        # Ordered by what the operator should care about MOST right now.
        # Converged means stop; diverging means worry; otherwise prefer the
        # most-specific actionable gap (missing distance band > missing
        # quadrant > generic "keep going").
        state = convergence["state"]
        if state == "converged":
            return "Calibration has converged. You can stop and export."
        if state == "diverging":
            return "Reprojection error is trending up — hold steady or reset and start over."
        missing_scale = pose_diversity.get("missingScale") or []
        if "near" in missing_scale:
            return "Show the board closer to the camera."
        if "far" in missing_scale:
            return "Show the board farther from the camera."
        if cell_occupancy < COVERAGE_OCCUPANCY_MARGINAL_BELOW and weakest_quadrant:
            return f"Move the board toward the {weakest_quadrant[0]} quadrant."
        angle_buckets = pose_diversity.get("angleBuckets") or 1
        angle_covered = pose_diversity.get("angleBucketsCovered") or 0
        if angle_covered < angle_buckets * 0.5:
            return "Add more tilted orientations — rotate the board as you move it."
        if state == "improving":
            return "Camera matrix stable; refining error. Keep adding diverse frames."
        if state == "collecting":
            return "Camera matrix still drifting; keep moving the board to new poses."
        if verdict == "MARGINAL":
            return "Numbers stable; address the warnings to reach GOOD."
        return "Keep capturing varied poses through edges, corners, near, far, and tilted views."

    def guidance(self) -> str:
        detection = self.last_detection
        if detection is None:
            return "Connect a camera URL and show the iPad target."
        if detection.charuco_count < self.min_corners:
            return "Not enough ChaRuCo corners yet. Face the target toward the camera."
        if detection.sharpness < 18.0:
            return "Hold steady; the current frame is blurry."
        if detection.area_fraction < 0.04:
            return "Move the target closer."
        if len(self.candidates) < self.min_solve_frames:
            return "Good detection. Move slowly to collect diverse poses."
        quality = self.last_quality or {}
        coverage = quality.get("coverage", {})
        edges = coverage.get("edgeMarginFraction", {})
        if edges:
            widest_gap = max(edges, key=lambda key: edges[key])
            if edges[widest_gap] > 0.25:
                return f"Move the board toward the {widest_gap} edge."
        if self.last_quality and self.last_quality.get("verdict") == "GOOD":
            return "Coverage good. Add a few tilted near/far views or export."
        return "Keep moving through edges, corners, near, far, and tilted poses."

    def snapshot(self) -> dict[str, Any]:
        detection = self.last_detection
        calibration = self.last_calibration
        first_size = (
            {"width": detection.width, "height": detection.height}
            if detection is not None
            else None
        )
        payload: dict[str, Any] = {
            "runDir": str(self.run_dir),
            "totalFramesSeen": self.total_frames_seen,
            "candidateFrames": len(self.candidates),
            "acceptedSinceSolve": self.accepted_since_solve,
            "lastAcceptReason": self.last_accept_reason,
            "guidance": self.guidance(),
            "targetMetadata": self.target_metadata,
            "board": self.board_config.manifest(),
            "macbookReference": MACBOOK_PRO_CAMERA_REFERENCE,
            "detection": detection.to_public_dict() if detection else None,
            "calibration": None,
            "quality": self.last_quality,
            "trends": {
                "kHistory": self.k_history,
                "solveHistory": self.solve_history,
                "rmsHistory": [
                    {
                        "index": item["index"],
                        "candidateFrames": item["candidateFrames"],
                        "selectedFrames": item["selectedFrames"],
                        "rmsReprojectionErrorPx": item["rmsReprojectionErrorPx"],
                    }
                    for item in self.solve_history
                ],
                "kStabilityPct": self._k_stability_pct(),
            },
        }
        if calibration is not None:
            width = first_size["width"] if first_size else 0
            height = first_size["height"] if first_size else 0
            payload["calibration"] = {
                "rmsReprojectionErrorPx": calibration.rms_reprojection_error_px,
                "cameraMatrix": calibration.camera_matrix.astype(float).tolist(),
                "distortionCoefficients": calibration.distortion_coefficients.astype(
                    float
                ).tolist(),
                "selectedFrames": calibration.selected_count,
                "scaledMacbookReference": scaled_macbook_reference(width, height),
            }
        return payload

    def _k_stability_pct(self) -> float | None:
        if len(self.k_history) < 3:
            return None
        values = np.asarray(self.k_history[-5:], dtype=float)
        terms = np.array(
            [
                values[:, 0, 0],
                values[:, 1, 1],
                values[:, 0, 2],
                values[:, 1, 2],
            ]
        )
        means = np.maximum(np.abs(terms.mean(axis=1)), 1e-9)
        rel_span = (terms.max(axis=1) - terms.min(axis=1)) / means
        return float(np.max(rel_span) * 100.0)

    def _rms_trend(self) -> str | None:
        if len(self.solve_history) < 3:
            return None
        rms_values = [s["rmsReprojectionErrorPx"] for s in self.solve_history[-RMS_TREND_LOOKBACK:]]
        if len(rms_values) < 3:
            return None
        median = statistics.median(rms_values)
        if median <= 0:
            return None
        spread = (max(rms_values) - min(rms_values)) / median
        if spread < RMS_PLATEAU_RELATIVE_SPREAD:
            return "plateau"
        half = len(rms_values) // 2
        first = sum(rms_values[:half]) / half
        last = sum(rms_values[half:]) / (len(rms_values) - half)
        if last < first * 0.97:
            return "improving"
        if last > first * 1.03:
            return "worsening"
        return "plateau"

    def compute_convergence(self, *, verdict: str | None = None) -> dict[str, Any]:
        # The operator needs to know: keep going, or stop?
        #   - not_solved   : haven't solved yet
        #   - collecting   : K matrix still drifting; need more diverse poses
        #   - improving    : K stable, RMS still trending down — a few more frames will help
        #   - plateau      : K stable AND RMS flat, but verdict is not GOOD — fix what's flagged
        #   - converged    : K stable AND RMS flat AND verdict GOOD — you can stop
        #   - diverging    : RMS trending UP — recent frames may be noisy
        if not self.solve_history:
            return {
                "state": "not_solved",
                "kStabilityPct": None,
                "rmsTrend": None,
                "recommendation": "Capture more frames; the first solve runs once enough are collected.",
            }
        k_stability = self._k_stability_pct()
        rms_trend = self._rms_trend()
        k_stable = k_stability is not None and k_stability < K_STABILITY_CONVERGED_PCT
        enough_solves = len(self.solve_history) >= MIN_SOLVES_FOR_CONVERGED
        if rms_trend == "worsening":
            state = "diverging"
        elif k_stable and rms_trend == "plateau" and verdict == "GOOD" and enough_solves:
            state = "converged"
        elif k_stable and rms_trend == "plateau":
            state = "plateau"
        elif k_stable:
            state = "improving"
        else:
            state = "collecting"
        recommendations = {
            "not_solved": "Capture more frames; the first solve runs once enough are collected.",
            "collecting": "Camera matrix still drifting; keep adding diverse poses.",
            "improving": "Camera matrix is stable; refining residual error. A few more frames will help.",
            "plateau": "Numbers are stable but verdict is not yet GOOD. Address red flags and warnings.",
            "converged": "Calibration has converged. You can stop and export.",
            "diverging": "Error trending UP. Recent frames may be noisy; hold steady or reset.",
        }
        return {
            "state": state,
            "kStabilityPct": k_stability,
            "rmsTrend": rms_trend,
            "recommendation": recommendations[state],
        }

    def export(self) -> Path:
        from .reports import export_run

        return export_run(self)
