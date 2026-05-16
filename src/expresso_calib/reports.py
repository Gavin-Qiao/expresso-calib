from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .board import BoardConfig
from .reference import MACBOOK_PRO_CAMERA_REFERENCE

if TYPE_CHECKING:
    from .calibration import CalibrationAccumulator, CandidateFrame


def build_calibration_payload(
    accumulator: CalibrationAccumulator,
    selected: list[CandidateFrame],
) -> dict[str, Any]:
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
        "selected_frames": [_candidate_json(item, include_selected=True) for item in selected],
    }
    if calibration is not None:
        payload["calibration"] = {
            "model": "opencv_plumb_bob",
            "rms_reprojection_error_px": calibration.rms_reprojection_error_px,
            "camera_matrix": calibration.camera_matrix.astype(float).tolist(),
            "distortion_coefficients": calibration.distortion_coefficients.astype(float).tolist(),
            "flags": calibration.flags,
        }
    return payload


def _candidate_json(item: CandidateFrame, *, include_selected: bool = False) -> dict[str, Any]:
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


def write_detections_csv(candidates: list[CandidateFrame], path: Path) -> None:
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
                    "" if item.per_view_error_px is None else f"{item.per_view_error_px:.6f}",
                    f"{detection.sharpness:.3f}",
                    f"{detection.center_x:.6f}",
                    f"{detection.center_y:.6f}",
                    f"{detection.bbox_width:.6f}",
                    f"{detection.bbox_height:.6f}",
                    f"{detection.area_fraction:.6f}",
                    f"{detection.angle_deg:.3f}",
                ]
            )


def write_report_md(payload: dict[str, Any], board_config: BoardConfig, path: Path) -> None:
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
                f"- RMS reprojection error: `{calibration['rms_reprojection_error_px']:.4f} px`",
                f"- Selected frames: `{quality.get('selectedFrames', 0)}`",
                f"- Usable frames: `{quality.get('usableFrames', 0)}`",
                f"- Corner coverage: width "
                f"`{quality.get('coverage', {}).get('widthFraction', 0) * 100:.1f}%`, "
                f"height "
                f"`{quality.get('coverage', {}).get('heightFraction', 0) * 100:.1f}%`",
                f"- Cell occupancy: "
                f"`{quality.get('coverage', {}).get('cellOccupancyFraction', 0) * 100:.1f}%` "
                f"({'x'.join(str(n) for n in quality.get('coverage', {}).get('cellGrid', [8, 6]))} grid)",
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
    accumulator: CalibrationAccumulator,
    selected: list[CandidateFrame],
    debug_dir: Path,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    for item in selected:
        accumulator.write_candidate_screenshot(item, debug_dir)


def export_run(accumulator: CalibrationAccumulator) -> Path:
    accumulator.run_dir.mkdir(parents=True, exist_ok=True)
    selected = accumulator.select_diverse(accumulator.candidates, accumulator.max_calib_frames)
    payload = build_calibration_payload(accumulator, selected)
    write_calibration_json(payload, accumulator.run_dir / "calibration.json")
    write_detections_csv(accumulator.candidates, accumulator.run_dir / "detections.csv")
    write_report_md(payload, accumulator.board_config, accumulator.run_dir / "report.md")
    write_debug_images(accumulator, selected[:20], accumulator.run_dir / "debug")
    return accumulator.run_dir
