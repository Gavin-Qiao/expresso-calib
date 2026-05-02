from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


FOCUS_HOLD_SECONDS = 5.0


@dataclass(frozen=True)
class CameraConfig:
    id: str
    label: str
    url: str


class EphemeralCameraRegistry:
    def __init__(self) -> None:
        self._next_id = 1
        self._items: dict[str, CameraConfig] = {}

    def add(self, label: str, url: str) -> CameraConfig:
        camera_id = f"cam-{self._next_id}"
        self._next_id += 1
        config = CameraConfig(
            id=camera_id,
            label=clean_label(label, fallback=f"Camera {self._next_id - 1}"),
            url=url.strip(),
        )
        self._items[camera_id] = config
        return config

    def remove(self, camera_id: str) -> CameraConfig | None:
        return self._items.pop(camera_id, None)

    def list(self) -> list[CameraConfig]:
        return list(self._items.values())


class FocusTracker:
    def __init__(self, hold_seconds: float = FOCUS_HOLD_SECONDS) -> None:
        self.hold_seconds = hold_seconds
        self.focused_camera_id: str | None = None
        self.lost_at: float | None = None

    def update(self, cameras: list[dict[str, Any]], now: float) -> str | None:
        strongest = strongest_detecting_camera_id(cameras)
        if strongest is not None:
            self.focused_camera_id = strongest
            self.lost_at = None
            return self.focused_camera_id

        if self.focused_camera_id is None:
            self.lost_at = None
            return None

        if self.lost_at is None:
            self.lost_at = now
            return self.focused_camera_id

        if now - self.lost_at >= self.hold_seconds:
            self.focused_camera_id = None
            self.lost_at = None
        return self.focused_camera_id

    def clear_if_removed(self, camera_ids: set[str]) -> None:
        if self.focused_camera_id not in camera_ids:
            self.focused_camera_id = None
            self.lost_at = None


def strongest_detecting_camera_id(cameras: list[dict[str, Any]]) -> str | None:
    detecting = [camera for camera in cameras if camera.get("detectingCharuco")]
    if not detecting:
        return None
    return max(detecting, key=detection_score).get("id")


def detection_score(camera: dict[str, Any]) -> tuple[int, float, float]:
    detection = camera.get("detection") or {}
    return (
        int(detection.get("charucoCount") or 0),
        float(detection.get("areaFraction") or 0.0),
        float(detection.get("sharpness") or 0.0),
    )


def clean_label(label: str, *, fallback: str = "Camera") -> str:
    clean = " ".join(str(label or "").split())[:80]
    return clean or fallback


def slugify_label(label: str, fallback: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", label.strip()).strip("-._")
    return (clean or fallback)[:80]
