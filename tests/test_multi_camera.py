from __future__ import annotations

from expresso_calib.multi_camera import (
    EphemeralCameraRegistry,
    FocusTracker,
    slugify_label,
    strongest_detecting_camera_id,
)


def camera_payload(
    camera_id: str,
    *,
    detecting: bool,
    corners: int,
    area: float,
    sharpness: float,
) -> dict:
    return {
        "id": camera_id,
        "detectingCharuco": detecting,
        "detection": {
            "charucoCount": corners,
            "areaFraction": area,
            "sharpness": sharpness,
        },
    }


def test_registry_adds_and_removes_named_cameras_ephemerally() -> None:
    registry = EphemeralCameraRegistry()
    first = registry.add("Front Left", "http://10.39.86.11:1181/?action=stream")
    second = registry.add("Rear", "http://10.39.86.12:1181/?action=stream")

    assert [item.label for item in registry.list()] == ["Front Left", "Rear"]
    assert registry.remove(first.id) == first
    assert registry.list() == [second]


def test_focus_arbitration_chooses_strongest_detection() -> None:
    cameras = [
        camera_payload("a", detecting=True, corners=12, area=0.22, sharpness=100),
        camera_payload("b", detecting=True, corners=24, area=0.05, sharpness=80),
        camera_payload("c", detecting=False, corners=40, area=0.30, sharpness=200),
    ]

    assert strongest_detecting_camera_id(cameras) == "b"


def test_focus_tracker_holds_then_zooms_out_after_timeout() -> None:
    tracker = FocusTracker(hold_seconds=5.0)
    detecting = [camera_payload("a", detecting=True, corners=20, area=0.1, sharpness=90)]
    empty = [camera_payload("a", detecting=False, corners=0, area=0.0, sharpness=20)]

    assert tracker.update(detecting, now=10.0) == "a"
    assert tracker.update(empty, now=11.0) == "a"
    assert tracker.update(empty, now=15.9) == "a"
    assert tracker.update(empty, now=16.0) is None


def test_focus_tracker_switches_immediately_to_another_detector() -> None:
    tracker = FocusTracker(hold_seconds=5.0)
    assert tracker.update(
        [camera_payload("a", detecting=True, corners=20, area=0.1, sharpness=90)],
        now=10.0,
    ) == "a"
    assert tracker.update(
        [
            camera_payload("a", detecting=False, corners=0, area=0, sharpness=10),
            camera_payload("b", detecting=True, corners=18, area=0.2, sharpness=80),
        ],
        now=11.0,
    ) == "b"


def test_slugify_label_is_filesystem_safe() -> None:
    assert slugify_label("Front Left / Tag 1", "cam-1") == "Front-Left-Tag-1"
