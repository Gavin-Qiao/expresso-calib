from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .board import BoardConfig, create_board, create_dictionary, get_aruco_module


@dataclass
class Frame:
    index: int
    timestamp_sec: float
    image_bgr: np.ndarray
    source_id: str = "browser_upload"

    @property
    def width(self) -> int:
        return int(self.image_bgr.shape[1])

    @property
    def height(self) -> int:
        return int(self.image_bgr.shape[0])


@dataclass
class DetectionResult:
    frame_index: int
    timestamp_sec: float
    width: int
    height: int
    marker_count: int
    charuco_count: int
    sharpness: float
    corners: np.ndarray | None = None
    ids: np.ndarray | None = None
    center_x: float = 0.5
    center_y: float = 0.5
    bbox_width: float = 0.0
    bbox_height: float = 0.0
    area_fraction: float = 0.0
    angle_deg: float = 0.0
    board_polygon: list[list[float]] | None = None

    @property
    def detected(self) -> bool:
        return self.charuco_count > 0

    @property
    def enough_for_candidate(self) -> bool:
        return self.charuco_count >= 12

    def feature_vector(self) -> list[float]:
        angle_rad = (self.angle_deg % 180.0) * math.pi / 180.0
        return [
            self.center_x,
            self.center_y,
            min(1.0, self.area_fraction * 5.0),
            self.bbox_width,
            self.bbox_height,
            (math.cos(2.0 * angle_rad) + 1.0) / 2.0,
            (math.sin(2.0 * angle_rad) + 1.0) / 2.0,
            min(1.0, self.charuco_count / 40.0),
        ]

    def overlay_points(self) -> list[list[float]]:
        if self.corners is None:
            return []
        return [
            [float(point[0][0]), float(point[0][1])]
            for point in np.asarray(self.corners, dtype=np.float32)
        ]

    def to_public_dict(self) -> dict[str, object]:
        return {
            "detected": self.detected,
            "frameIndex": self.frame_index,
            "frameSize": {"width": self.width, "height": self.height},
            "markerCount": self.marker_count,
            "charucoCount": self.charuco_count,
            "sharpness": self.sharpness,
            "center": {"x": self.center_x, "y": self.center_y},
            "bbox": {"width": self.bbox_width, "height": self.bbox_height},
            "areaFraction": self.area_fraction,
            "angleDeg": self.angle_deg,
            "points": self.overlay_points(),
            "boardPolygon": self.board_polygon or [],
        }


def laplacian_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class CharucoDetector:
    def __init__(self, config: BoardConfig) -> None:
        self.config = config
        self.aruco = get_aruco_module()
        self.dictionary = create_dictionary(config.dictionary)
        self.board = create_board(config)
        self.params = self._create_detector_parameters()
        self.marker_detector = None
        if hasattr(self.aruco, "ArucoDetector"):
            self.marker_detector = self.aruco.ArucoDetector(self.dictionary, self.params)

    def _create_detector_parameters(self) -> Any:
        if hasattr(self.aruco, "DetectorParameters"):
            params = self.aruco.DetectorParameters()
        else:
            params = self.aruco.DetectorParameters_create()

        if hasattr(self.aruco, "CORNER_REFINE_SUBPIX") and hasattr(
            params, "cornerRefinementMethod"
        ):
            params.cornerRefinementMethod = self.aruco.CORNER_REFINE_SUBPIX
        for attr, value in {
            "cornerRefinementWinSize": 5,
            "cornerRefinementMaxIterations": 30,
            "cornerRefinementMinAccuracy": 0.01,
            "adaptiveThreshWinSizeMin": 3,
            "adaptiveThreshWinSizeMax": 33,
            "adaptiveThreshWinSizeStep": 10,
        }.items():
            if hasattr(params, attr):
                setattr(params, attr, value)
        return params

    def detect(self, frame: Frame) -> DetectionResult:
        gray = cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2GRAY)
        sharpness = laplacian_sharpness(gray)
        marker_corners, marker_ids = self._detect_markers(gray)
        marker_count = 0 if marker_ids is None else int(len(marker_ids))
        if marker_ids is None or marker_count <= 0:
            return DetectionResult(
                frame_index=frame.index,
                timestamp_sec=frame.timestamp_sec,
                width=frame.width,
                height=frame.height,
                marker_count=0,
                charuco_count=0,
                sharpness=sharpness,
            )

        charuco_corners, charuco_ids = self._interpolate_charuco(gray, marker_corners, marker_ids)
        if charuco_corners is None or charuco_ids is None or len(charuco_ids) <= 0:
            return DetectionResult(
                frame_index=frame.index,
                timestamp_sec=frame.timestamp_sec,
                width=frame.width,
                height=frame.height,
                marker_count=marker_count,
                charuco_count=0,
                sharpness=sharpness,
            )

        return detection_from_corners(
            frame=frame,
            marker_count=marker_count,
            charuco_corners=charuco_corners,
            charuco_ids=charuco_ids,
            sharpness=sharpness,
        )

    def _detect_markers(self, gray: np.ndarray) -> tuple[Any, Any]:
        if self.marker_detector is not None:
            corners, ids, _ = self.marker_detector.detectMarkers(gray)
        else:
            corners, ids, _ = self.aruco.detectMarkers(
                gray, self.dictionary, parameters=self.params
            )
        return corners, ids

    def _interpolate_charuco(
        self, gray: np.ndarray, marker_corners: Any, marker_ids: Any
    ) -> tuple[Any, Any]:
        if not hasattr(self.aruco, "interpolateCornersCharuco"):
            raise RuntimeError("OpenCV is missing interpolateCornersCharuco.")
        _, charuco_corners, charuco_ids = self.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, self.board
        )
        return charuco_corners, charuco_ids


def detection_from_corners(
    frame: Frame,
    marker_count: int,
    charuco_corners: Any,
    charuco_ids: Any,
    sharpness: float,
) -> DetectionResult:
    corners = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 1, 2)
    ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1, 1)
    points = corners.reshape(-1, 2)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    bbox_w_px = float(max(0.0, x_max - x_min))
    bbox_h_px = float(max(0.0, y_max - y_min))
    width = max(1, frame.width)
    height = max(1, frame.height)
    area_fraction = float((bbox_w_px * bbox_h_px) / (width * height))
    center_x = float(((x_min + x_max) / 2.0) / width)
    center_y = float(((y_min + y_max) / 2.0) / height)
    bbox_width = float(bbox_w_px / width)
    bbox_height = float(bbox_h_px / height)
    rect = cv2.minAreaRect(points.astype(np.float32))
    angle_deg = float(rect[2])
    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2)

    return DetectionResult(
        frame_index=frame.index,
        timestamp_sec=frame.timestamp_sec,
        width=frame.width,
        height=frame.height,
        marker_count=marker_count,
        charuco_count=int(len(ids)),
        sharpness=sharpness,
        corners=corners,
        ids=ids,
        center_x=center_x,
        center_y=center_y,
        bbox_width=bbox_width,
        bbox_height=bbox_height,
        area_fraction=area_fraction,
        angle_deg=angle_deg,
        board_polygon=[[float(x), float(y)] for x, y in hull],
    )


def decode_jpeg_frame(payload: bytes, index: int, timestamp_sec: float) -> Frame:
    encoded = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode JPEG frame payload.")
    return Frame(index=index, timestamp_sec=timestamp_sec, image_bgr=image)
