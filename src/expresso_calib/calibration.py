from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

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
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


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
        self.min_solve_corners = min_solve_corners or max(
            min_corners, int(round(board_config.charuco_corners * 0.45))
        )
        self.min_solve_area_fraction = min_solve_area_fraction
        self.min_solve_sharpness = min_solve_sharpness
        self.auto_export = auto_export
        self.create_run_dir = create_run_dir
        self.run_dir = self._make_run_dir()
        self.candidates: list[CandidateFrame] = []
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

    def observe(
        self, detection: DetectionResult, image_bgr: np.ndarray
    ) -> tuple[bool, str]:
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
                euclidean(feature, item.detection.feature_vector())
                for item in self.candidates
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
        selected = self.select_diverse(
            solve_pool, self.max_calib_frames, mark_selected=False
        )
        if len(selected) < 4:
            return SolveInsufficientData(reason="solve pool too small")

        try:
            initial_calibration = self._calibrate(selected, assign_per_view_errors=False)
            selected, calibration, rejected_outliers, outlier_threshold = (
                self._refine_outlier_views(selected, initial_calibration)
            )
        except (cv2.error, ValueError, RuntimeError) as exc:
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
        per_view_by_id = {
            id(item): error
            for item, error in zip(result.selected, result.calibration.per_view_errors_px)
        }
        for item in self.candidates:
            item.selected = id(item) in selected_ids
            if item.selected and id(item) in per_view_by_id:
                item.per_view_error_px = per_view_by_id[id(item)]

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

    def _refine_outlier_views(
        self, selected: list[CandidateFrame], calibration: CalibrationResult
    ) -> tuple[list[CandidateFrame], CalibrationResult, int, float | None]:
        errors = calibration.per_view_errors_px
        if len(selected) < self.min_outlier_refine_frames or len(errors) != len(selected):
            return selected, calibration, 0, None

        median_error = percentile(errors, 50)
        if median_error is None:
            return selected, calibration, 0, None

        threshold = max(
            self.outlier_error_floor,
            median_error * self.outlier_error_median_factor,
        )
        kept = [
            item
            for item, error in zip(selected, errors)
            if float(error) <= threshold
        ]
        minimum_kept = max(4, min(self.min_outlier_refine_frames, len(selected)))
        if len(kept) == len(selected) or len(kept) < minimum_kept:
            return selected, calibration, 0, threshold

        refined = self._calibrate(kept, assign_per_view_errors=False)
        return kept, refined, len(selected) - len(kept), threshold

    def write_candidate_screenshot(
        self, item: CandidateFrame, screenshot_dir: Path
    ) -> Path:
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

        def quality(item: CandidateFrame) -> float:
            detection = item.detection
            corner_score = detection.charuco_count / max(
                1, self.board_config.charuco_corners
            )
            sharp_score = min(1.0, detection.sharpness / 450.0)
            area_score = min(1.0, detection.area_fraction / 0.25)
            edge_score = max(abs(detection.center_x - 0.5), abs(detection.center_y - 0.5))
            return (
                corner_score * 0.55
                + sharp_score * 0.12
                + area_score * 0.23
                + edge_score * 0.10
            )

        remaining = list(candidates)
        first = max(remaining, key=quality)
        selected = [first]
        remaining.remove(first)

        while remaining and len(selected) < max_frames:
            next_item = max(
                remaining,
                key=lambda item: min(
                    euclidean(
                        item.detection.feature_vector(),
                        chosen.detection.feature_vector(),
                    )
                    for chosen in selected
                )
                + quality(item) * 0.18,
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
            for candidate, error in zip(selected, computed_per_view):
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
            image_points.append(
                np.asarray(detection.corners, dtype=np.float32).reshape(-1, 1, 2)
            )
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
        for item, rvec, tvec in zip(selected, rvecs, tvecs):
            detection = item.detection
            if detection.ids is None or detection.corners is None:
                continue
            ids = np.asarray(detection.ids, dtype=np.int32).reshape(-1)
            object_points = object_corners[ids]
            image_points = np.asarray(detection.corners, dtype=np.float32).reshape(-1, 2)
            projected, _ = cv2.projectPoints(
                object_points, rvec, tvec, camera_matrix, dist_coeffs
            )
            projected = projected.reshape(-1, 2)
            error = np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1)))
            errors.append(float(error))
        return errors

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
        coverage_height = float(
            (all_points[:, 1].max() - all_points[:, 1].min()) / height
        )
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
        if coverage_width < 0.45 or coverage_height < 0.32:
            red.append("Detected corners do not cover enough of the image.")
        elif coverage_width < 0.62 or coverage_height < 0.45:
            yellow.append("Image coverage is limited; include more edge and corner views.")
        if areas:
            if max(areas) < 0.14:
                yellow.append("The board never gets very close; include close-up views.")
            if max(areas) - min(areas) < 0.10:
                yellow.append("Board scale does not vary much; mix near and far views.")
        if corner_counts and statistics.median(corner_counts) < 18:
            yellow.append("Median ChaRuCo corner count is low; show more of the board.")

        verdict = "GOOD"
        if red:
            verdict = "REDO"
        elif yellow:
            verdict = "MARGINAL"

        return {
            "verdict": verdict,
            "redFlags": red,
            "warnings": yellow,
            "selectedFrames": len(selected),
            "usableFrames": len(self.candidates)
            if usable_frames is None
            else usable_frames,
            "cornerCount": {
                "min": min(corner_counts) if corner_counts else 0,
                "median": float(statistics.median(corner_counts))
                if corner_counts
                else 0.0,
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
                "edgeMarginFraction": edge_margin,
            },
            "boardAreaFraction": {
                "min": min(areas) if areas else 0.0,
                "median": float(statistics.median(areas)) if areas else 0.0,
                "max": max(areas) if areas else 0.0,
            },
        }

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

    def export(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        selected = self.select_diverse(self.candidates, self.max_calib_frames)
        calibration = self.last_calibration
        quality = self.last_quality
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(self.run_dir),
            "source": self.source_id,
            "board": self.board_config.manifest(),
            "target_metadata": self.target_metadata,
            "macbook_reference": MACBOOK_PRO_CAMERA_REFERENCE,
            "quality": quality,
            "calibration": None,
            "solve_history": self.solve_history,
            "selected_frames": [
                self._candidate_json(item, include_selected=True) for item in selected
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
        (self.run_dir / "calibration.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        self._write_detections_csv(self.run_dir / "detections.csv")
        self._write_report(self.run_dir / "report.md", payload)
        self._write_debug_images(selected[:20], self.run_dir / "debug")
        return self.run_dir

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
        }
        if include_selected:
            payload["selected"] = item.selected
        return payload

    def _write_detections_csv(self, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "frame",
                    "time_sec",
                    "selected",
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
            for item in self.candidates:
                detection = item.detection
                writer.writerow(
                    [
                        detection.frame_index,
                        f"{detection.timestamp_sec:.6f}",
                        int(item.selected),
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

    def _write_report(self, path: Path, payload: dict[str, Any]) -> None:
        quality = payload.get("quality") or {}
        calibration = payload.get("calibration") or {}
        lines = [
            "# Expresso Calib Intrinsic Report",
            "",
            f"Verdict: **{quality.get('verdict', 'PENDING')}**",
            "",
            f"Run: `{payload['run_dir']}`",
            f"Source: `{payload['source']}`",
            f"Board: `{self.board_config.squares_x}x{self.board_config.squares_y}` "
            f"{self.board_config.dictionary}, marker/square ratio "
            f"`{self.board_config.marker_length / self.board_config.square_length:.3f}`",
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

    def _write_debug_images(
        self, selected: list[CandidateFrame], debug_dir: Path
    ) -> None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        for item in selected:
            self.write_candidate_screenshot(item, debug_dir)
