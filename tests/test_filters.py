from __future__ import annotations

import numpy as np
import pytest

from expresso_calib.filters import (
    BRIGHTNESS_RANGE,
    CONTRAST_RANGE,
    GAMMA_RANGE,
    FilterPipeline,
    FilterSettings,
    clamp_settings,
)


def _bgr_gradient(width: int = 32, height: int = 24) -> np.ndarray:
    row = np.linspace(0, 255, width, dtype=np.uint8)
    plane = np.tile(row, (height, 1))
    return np.stack([plane, plane, plane], axis=-1)


def test_defaults_are_noop_and_return_same_array():
    pipe = FilterPipeline()
    frame = _bgr_gradient()
    result = pipe.apply(frame)
    assert result is frame, "default settings must skip allocation"


def test_brightness_only_shifts_values_uniformly():
    pipe = FilterPipeline()
    pipe.update(FilterSettings(brightness=50))
    frame = np.full((8, 8, 3), 100, dtype=np.uint8)
    result = pipe.apply(frame)
    assert int(result[0, 0, 0]) == 150


def test_contrast_only_scales_values():
    pipe = FilterPipeline()
    pipe.update(FilterSettings(contrast=200))
    frame = np.full((8, 8, 3), 50, dtype=np.uint8)
    result = pipe.apply(frame)
    assert int(result[0, 0, 0]) == 100


def test_gamma_above_one_darkens_midtones():
    pipe = FilterPipeline()
    pipe.update(FilterSettings(gamma=2.0))
    frame = np.full((8, 8, 3), 128, dtype=np.uint8)
    result = pipe.apply(frame)
    # gamma 2.0 inv 0.5 applied to 128/255 gives ~181 after rounding.
    assert 170 < int(result[0, 0, 0]) < 195


def test_gamma_lut_is_cached_until_value_changes():
    pipe = FilterPipeline()
    pipe.update(FilterSettings(gamma=1.5))
    pipe.apply(_bgr_gradient())
    cached = pipe._gamma_lut
    pipe.update(FilterSettings(gamma=1.5))
    pipe.apply(_bgr_gradient())
    assert pipe._gamma_lut is cached, "LUT must be reused when gamma is unchanged"
    pipe.update(FilterSettings(gamma=1.8))
    pipe.apply(_bgr_gradient())
    assert pipe._gamma_lut is not cached, "LUT must rebuild when gamma changes"


def test_clahe_changes_pixels_only_when_enabled():
    pipe = FilterPipeline()
    frame = _bgr_gradient(64, 48)

    pipe.update(FilterSettings(clahe=False))
    out_off = pipe.apply(frame)
    assert out_off is frame

    pipe.update(FilterSettings(clahe=True))
    out_on = pipe.apply(frame)
    assert out_on is not frame
    assert not np.array_equal(out_on, frame)


def test_apply_handles_none_frame():
    pipe = FilterPipeline()
    pipe.update(FilterSettings(brightness=30))
    assert pipe.apply(None) is None


def test_clamp_settings_clips_out_of_range_values():
    s = clamp_settings(
        {"brightness": 9999, "contrast": -50, "gamma": 100.0, "clahe": True}
    )
    assert s.brightness == BRIGHTNESS_RANGE[1]
    assert s.contrast == CONTRAST_RANGE[0]
    assert s.gamma == GAMMA_RANGE[1]
    assert s.clahe is True


def test_clamp_settings_uses_defaults_for_missing_keys():
    s = clamp_settings({})
    assert s.brightness == 0
    assert s.contrast == 100
    assert s.gamma == 1.0
    assert s.clahe is False


def test_clamp_settings_rejects_non_dict():
    with pytest.raises(ValueError):
        clamp_settings("not a dict")


def test_clamp_settings_rejects_non_numeric_brightness():
    with pytest.raises(ValueError):
        clamp_settings({"brightness": "foo"})


def test_clamp_settings_rejects_nan_gamma():
    with pytest.raises(ValueError):
        clamp_settings({"gamma": float("nan")})


def test_to_public_dict_round_trips_through_clamp():
    settings = FilterSettings(brightness=-20, contrast=140, gamma=1.6, clahe=True)
    public = settings.to_public_dict()
    again = clamp_settings(public)
    assert again.brightness == settings.brightness
    assert again.contrast == settings.contrast
    assert abs(again.gamma - settings.gamma) < 1e-3
    assert again.clahe is settings.clahe
