from __future__ import annotations

MACBOOK_PRO_CAMERA_REFERENCE = {
    "device": "MacBook Pro Camera",
    "source": "Apple OriginalCameraIntrinsicMatrix sample-buffer attachment",
    "note": (
        "Manufacturer/sensor-space reference. The browser frame may be cropped "
        "or scaled, so this is a sanity check rather than exact output-frame truth."
    ),
    "reference_dimensions": {"width": 3040, "height": 2880},
    "camera_matrix": [
        [1503.3333, 0.0, 1509.9377],
        [0.0, 1503.3333, 1359.7521],
        [0.0, 0.0, 1.0],
    ],
}


def scaled_macbook_reference(width: int, height: int) -> dict[str, object]:
    ref = MACBOOK_PRO_CAMERA_REFERENCE
    ref_dims = ref["reference_dimensions"]
    sx = width / float(ref_dims["width"])
    sy = height / float(ref_dims["height"])
    k = ref["camera_matrix"]
    return {
        "note": "Naive scale only; does not compensate for unknown crop path.",
        "frame_dimensions": {"width": width, "height": height},
        "camera_matrix": [
            [k[0][0] * sx, 0.0, k[0][2] * sx],
            [0.0, k[1][1] * sy, k[1][2] * sy],
            [0.0, 0.0, 1.0],
        ],
    }
