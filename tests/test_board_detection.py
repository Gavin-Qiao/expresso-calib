from __future__ import annotations

import cv2
import numpy as np

from expresso_calib.board import DEFAULT_BOARD, draw_board, target_png_bytes
from expresso_calib.detection import CharucoDetector, Frame


def test_target_png_uses_board_manifest() -> None:
    payload = target_png_bytes(DEFAULT_BOARD, 900, 600)
    assert payload.startswith(b"\x89PNG")
    assert DEFAULT_BOARD.dictionary == "DICT_4X4_50"
    assert DEFAULT_BOARD.squares_x == 9
    assert DEFAULT_BOARD.squares_y == 6
    assert DEFAULT_BOARD.legacy_pattern is True


def test_detector_finds_generated_charuco_board() -> None:
    board = draw_board(DEFAULT_BOARD, 900, 600)
    bgr = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    frame = Frame(index=1, timestamp_sec=0.0, image_bgr=bgr)
    detection = CharucoDetector(DEFAULT_BOARD).detect(frame)

    assert detection.marker_count > 0
    assert detection.charuco_count >= 30
    assert detection.detected is True
    assert detection.area_fraction > 0.45
    assert len(detection.overlay_points()) == detection.charuco_count


def test_detector_handles_blank_frame() -> None:
    image = np.full((540, 960, 3), 255, dtype=np.uint8)
    frame = Frame(index=1, timestamp_sec=0.0, image_bgr=image)
    detection = CharucoDetector(DEFAULT_BOARD).detect(frame)

    assert detection.detected is False
    assert detection.marker_count == 0
    assert detection.charuco_count == 0
