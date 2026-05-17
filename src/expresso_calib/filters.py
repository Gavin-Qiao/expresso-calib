from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

DEFAULT_BRIGHTNESS = 0
DEFAULT_CONTRAST = 100
DEFAULT_GAMMA = 1.0
DEFAULT_CLAHE = False
DEFAULT_CLAHE_CLIP = 2.0
DEFAULT_CLAHE_TILE = 8

BRIGHTNESS_RANGE = (-100, 100)
CONTRAST_RANGE = (50, 200)
GAMMA_RANGE = (0.5, 2.5)


@dataclass
class FilterSettings:
    brightness: int = DEFAULT_BRIGHTNESS
    contrast: int = DEFAULT_CONTRAST
    gamma: float = DEFAULT_GAMMA
    clahe: bool = DEFAULT_CLAHE

    def is_default(self) -> bool:
        return (
            self.brightness == DEFAULT_BRIGHTNESS
            and self.contrast == DEFAULT_CONTRAST
            and abs(self.gamma - DEFAULT_GAMMA) < 1e-6
            and not self.clahe
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "brightness": int(self.brightness),
            "contrast": int(self.contrast),
            "gamma": float(round(self.gamma, 3)),
            "clahe": bool(self.clahe),
        }


def clamp_settings(payload: Any) -> FilterSettings:
    if not isinstance(payload, dict):
        raise ValueError("filter payload must be a JSON object")
    return FilterSettings(
        brightness=_clamp_int(payload.get("brightness", DEFAULT_BRIGHTNESS), *BRIGHTNESS_RANGE),
        contrast=_clamp_int(payload.get("contrast", DEFAULT_CONTRAST), *CONTRAST_RANGE),
        gamma=_clamp_float(payload.get("gamma", DEFAULT_GAMMA), *GAMMA_RANGE),
        clahe=bool(payload.get("clahe", DEFAULT_CLAHE)),
    )


def _clamp_int(value: Any, lo: int, hi: int) -> int:
    try:
        v = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected integer in [{lo}, {hi}]") from exc
    return max(lo, min(hi, v))


def _clamp_float(value: Any, lo: float, hi: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected number in [{lo}, {hi}]") from exc
    if v != v:  # NaN guard
        raise ValueError(f"expected number in [{lo}, {hi}]")
    return max(lo, min(hi, v))


class FilterPipeline:
    """Per-camera image filter stack with cached LUT and CLAHE.

    ``apply`` returns the input frame unchanged when settings are at defaults,
    skipping both the copy and the work. Caches are rebuilt only when gamma
    changes or CLAHE is enabled for the first time.
    """

    def __init__(self) -> None:
        self.settings = FilterSettings()
        self._gamma_lut: np.ndarray | None = None
        self._gamma_lut_value: float | None = None
        self._clahe: Any | None = None

    def update(self, settings: FilterSettings) -> None:
        if self._gamma_lut_value != settings.gamma:
            self._gamma_lut = None
            self._gamma_lut_value = None
        self.settings = settings

    def apply(self, frame_bgr: Any) -> Any:
        settings = self.settings
        if settings.is_default() or frame_bgr is None:
            return frame_bgr

        out = frame_bgr

        alpha = settings.contrast / 100.0
        beta = float(settings.brightness)
        if abs(alpha - 1.0) > 1e-6 or beta != 0.0:
            out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

        if abs(settings.gamma - 1.0) > 1e-6:
            if self._gamma_lut is None or self._gamma_lut_value != settings.gamma:
                inv = 1.0 / settings.gamma
                self._gamma_lut = np.array(
                    [((i / 255.0) ** inv) * 255.0 for i in range(256)],
                    dtype=np.uint8,
                )
                self._gamma_lut_value = settings.gamma
            out = cv2.LUT(out, self._gamma_lut)

        if settings.clahe and out.ndim == 3 and out.shape[2] == 3:
            if self._clahe is None:
                self._clahe = cv2.createCLAHE(
                    clipLimit=DEFAULT_CLAHE_CLIP,
                    tileGridSize=(DEFAULT_CLAHE_TILE, DEFAULT_CLAHE_TILE),
                )
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            channels = list(cv2.split(lab))
            channels[0] = self._clahe.apply(channels[0])
            lab = cv2.merge(channels)
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return out
