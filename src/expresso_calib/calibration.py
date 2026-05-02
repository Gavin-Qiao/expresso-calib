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
    ) -> None:
        self.board_config = board_config
        self.board = create_board(board_config)
        self.runs_dir = runs_dir
        self.min_corners = min_corners
        self.min_solve_frames = min_solve_frames
        self.solve_every_new_frames = solve_every_new_frames
        self.max_calib_frames = max_calib_frames
        self.max_candidates = max_candidates
        self.run_dir = self._make_run_dir()
        self.candidates: list[CandidateFrame] = []
        self.last_detection: DetectionResult | None = None
        self.last_calibration: CalibrationResult | None = None
        self.last_quality: dict[str, Any] | None = None
        self.k_history: list[list[list[float]]] = []
        self.solve_history: list[dict[str, Any]] = []
        self.accepted_since_solve = 0
        self.total_frames_seen = 0
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
        self.last_accept_reason = "reset"

    def _make_run_dir(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.runs_dir / stamp
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
        if self.candidates:
            nearest = min(
                euclidean(feature, item.detection.feature_vector())
                for item in self.candidates
            )
            if nearest < 0.045:
                reason = "duplicate pose"
                self.last_accept_reason = reason
                return False, reason

        self.candidates.append(
            CandidateFrame(
                detection=detection,
                image_bgr=image_bgr.copy(),
                accepted_at=datetime.now().isoformat(timespec="seconds"),
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

        selected = self.select_diverse(self.candidates, self.max_calib_frames)
        if len(selected) < 4:
            return self.last_calibration
        calibration = self._calibrate(selected)
        quality = self.summarize_quality(selected, calibration)
        self.last_calibration = calibration
        self.last_quality = quality
        self.k_history.append(calibration.camera_matrix.astype(float).tolist())
        self.k_history = self.k_history[-8:]
        k = calibration.camera_matrix
        self.solve_history.append(
            {
                "index": len(self.solve_history) + 1,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "candidateFrames": len(self.candidates),
                "selectedFrames": calibration.selected_count,
                "rmsReprojectionErrorPx": calibration.rms_reprojection_error_px,
                "fx": float(k[0, 0]),
                "fy": float(k[1, 1]),
                "cx": float(k[0, 2]),
                "cy": float(k[1, 2]),
            }
        )
        self.accepted_since_solve = 0
        self.export()
        return calibration

    def select_diverse(
        self, candidates: list[CandidateFrame], max_frames: int
    ) -> list[CandidateFrame]:
        for item in candidates:
            item.selected = False
        if len(candidates) <= max_frames:
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
        selected_ids = {id(item) for item in selected}
        for item in candidates:
            item.selected = id(item) in selected_ids
        return selected

    def _calibrate(self, selected: list[CandidateFrame]) -> CalibrationResult:
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
        self, selected: list[CandidateFrame], calibration: CalibrationResult
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
            "usableFrames": len(self.candidates),
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
        aruco = cv2.aruco
        for item in selected:
            detection = item.detection
            image = item.image_bgr.copy()
            if detection.corners is not None and detection.ids is not None:
                aruco.drawDetectedCornersCharuco(image, detection.corners, detection.ids)
            cv2.imwrite(str(debug_dir / f"frame_{detection.frame_index:06d}.jpg"), image)
