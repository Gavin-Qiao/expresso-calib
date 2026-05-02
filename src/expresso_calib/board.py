from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class BoardConfig:
    squares_x: int = 9
    squares_y: int = 6
    square_length: float = 1.0
    marker_length: float = 0.75
    dictionary: str = "DICT_4X4_50"
    legacy_pattern: bool = True

    @property
    def charuco_corners(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    @property
    def aspect_ratio(self) -> float:
        return self.squares_x / self.squares_y

    def manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["charuco_corners"] = self.charuco_corners
        payload["marker_square_ratio"] = self.marker_length / self.square_length
        return payload


DEFAULT_BOARD = BoardConfig()


def get_aruco_module() -> Any:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "This OpenCV build does not include cv2.aruco. "
            "Install opencv-contrib-python, not opencv-python."
        )
    return cv2.aruco


@lru_cache(maxsize=16)
def create_dictionary(name: str) -> Any:
    aruco = get_aruco_module()
    dictionary_name = name.strip().upper()
    if not dictionary_name.startswith("DICT_"):
        dictionary_name = "DICT_" + dictionary_name
    if not hasattr(aruco, dictionary_name):
        valid = sorted(item for item in dir(aruco) if item.startswith("DICT_"))
        raise RuntimeError(
            f"Unknown ArUco dictionary {name!r}. Examples: {', '.join(valid[:8])}"
        )
    dictionary_id = getattr(aruco, dictionary_name)
    if hasattr(aruco, "getPredefinedDictionary"):
        return aruco.getPredefinedDictionary(dictionary_id)
    return aruco.Dictionary_get(dictionary_id)


@lru_cache(maxsize=16)
def create_board(config: BoardConfig = DEFAULT_BOARD) -> Any:
    aruco = get_aruco_module()
    dictionary = create_dictionary(config.dictionary)
    if config.squares_x < 2 or config.squares_y < 2:
        raise RuntimeError("ChaRuCo board needs at least 2 squares in each direction.")
    if (
        config.square_length <= 0
        or config.marker_length <= 0
        or config.marker_length >= config.square_length
    ):
        raise RuntimeError("Expected 0 < marker_length < square_length.")

    board = None
    if hasattr(aruco, "CharucoBoard"):
        try:
            board = aruco.CharucoBoard(
                (config.squares_x, config.squares_y),
                config.square_length,
                config.marker_length,
                dictionary,
            )
        except Exception:
            board = None
    if board is None and hasattr(aruco, "CharucoBoard_create"):
        board = aruco.CharucoBoard_create(
            config.squares_x,
            config.squares_y,
            config.square_length,
            config.marker_length,
            dictionary,
        )
    if board is None:
        raise RuntimeError("This OpenCV build has no ChaRuCo board constructor.")

    if config.legacy_pattern and hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(True)
    return board


def board_chessboard_corners(board: Any) -> np.ndarray:
    if hasattr(board, "getChessboardCorners"):
        return np.asarray(board.getChessboardCorners(), dtype=np.float32)
    if hasattr(board, "chessboardCorners"):
        return np.asarray(board.chessboardCorners, dtype=np.float32)
    raise RuntimeError("Could not read ChaRuCo chessboard object points from board.")


def draw_board(config: BoardConfig, width: int, height: int) -> np.ndarray:
    board = create_board(config)
    if hasattr(board, "generateImage"):
        return board.generateImage((int(width), int(height)), marginSize=0, borderBits=1)

    image = np.full((int(height), int(width)), 255, dtype=np.uint8)
    if hasattr(board, "draw"):
        board.draw((int(width), int(height)), image, marginSize=0, borderBits=1)
        return image
    raise RuntimeError("This OpenCV board object cannot render a board image.")


def target_png_bytes(
    config: BoardConfig = DEFAULT_BOARD,
    canvas_width: int = 1800,
    canvas_height: int = 1200,
) -> bytes:
    canvas_width = max(400, min(int(canvas_width), 5000))
    canvas_height = max(300, min(int(canvas_height), 5000))
    margin = int(min(canvas_width, canvas_height) * 0.035)
    available_w = max(200, canvas_width - margin * 2)
    available_h = max(200, canvas_height - margin * 2)

    board_w = min(available_w, int(available_h * config.aspect_ratio))
    board_h = int(round(board_w / config.aspect_ratio))
    if board_h > available_h:
        board_h = available_h
        board_w = int(round(board_h * config.aspect_ratio))

    board_image = draw_board(config, board_w, board_h)
    canvas = np.full((canvas_height, canvas_width), 255, dtype=np.uint8)
    x0 = (canvas_width - board_w) // 2
    y0 = (canvas_height - board_h) // 2
    canvas[y0 : y0 + board_h, x0 : x0 + board_w] = board_image

    image = Image.fromarray(canvas, mode="L")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def target_pdf_bytes(
    config: BoardConfig = DEFAULT_BOARD,
    page_width_mm: float = 280.6,
    page_height_mm: float = 194.7,
    dpi: int = 144,
) -> bytes:
    page_width_mm = max(80.0, min(float(page_width_mm), 400.0))
    page_height_mm = max(80.0, min(float(page_height_mm), 400.0))
    dpi = max(72, min(int(dpi), 300))
    canvas_width = int(round(page_width_mm / 25.4 * dpi))
    canvas_height = int(round(page_height_mm / 25.4 * dpi))
    png = target_png_bytes(config, canvas_width, canvas_height)
    image = Image.open(BytesIO(png)).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="PDF", resolution=float(dpi))
    return buffer.getvalue()
