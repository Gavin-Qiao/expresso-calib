# Expresso Calib

Expresso Calib is a live multi-camera ChaRuCo intrinsic-calibration console for
robot camera development. It lets you append named camera stream URLs, view all
streams at once, and automatically focus the camera that currently sees the
ChaRuCo target.

The operator workflow is deliberately visual and minimal:

- Add named camera URLs.
- Start all streams.
- Move the iPad ChaRuCo target through the view.
- Watch the focused stream and its centered RMS digit.
- Keep moving until the digit stops improving.

The app keeps calibration state in memory. It does not automatically write
calibration JSON, reports, or CSV files. The only permanent artifact during live
calibration is an auditable screenshot for each accepted ChaRuCo candidate frame.

## What It Does

Implemented:

- Multi-camera URL ingest with one independent calibration accumulator per
  camera.
- Responsive grid of all camera streams.
- Smooth focus/zoom to the camera that detects enough ChaRuCo corners.
- Five second hold before returning to the full grid after detection is lost.
- Immediate focus switch when another camera detects the target.
- Centered per-camera RMS overlay with pending, poor, marginal, and good color
  states.
- iPad target page and generated target PDF.
- MacBook camera helper that exposes the built-in camera as an MJPEG URL.
- Permanent screenshots of accepted ChaRuCo candidate frames.

Not implemented:

- Persisted camera URL profiles.
- Multi-camera extrinsic calibration.
- Robot camera discovery.
- Authentication or remote deployment.

## Trust Model

This app is intended for **local network use only** between an operator laptop,
the iPad target, and robot cameras on the same trusted LAN. The server has no
authentication, no TLS, and accepts arbitrary http/https/rtsp camera URLs which
it fetches from its own network position. A user with operator access to the
console can:

- Reach any host the server can reach by typing its URL into the camera input
  (no SSRF filtering by design — the operator is trusted).
- Trigger the server to download arbitrarily large MJPEG payloads (capped at
  8 MiB per frame and 64 KiB per JSON request, but still costs CPU).

**Do not expose the operator port (3987) or the MacBook camera bridge port
(3988) to the public internet.** Bind them to the LAN interface or put them
behind a VPN. If you need remote access for a co-located robot, use SSH port
forwarding rather than opening the port.

## Requirements

- Python 3.11 or newer.
- `uv`.
- macOS for the bundled MacBook camera helper app.
- A browser on the operator laptop.
- An iPad or tablet on the same network for the ChaRuCo target.

Install dependencies:

```sh
uv sync
```

Run the server:

```sh
uv run expresso-calib
```

Open:

```text
http://127.0.0.1:3987/operator
```

## MacBook Camera Source

Launch the helper app:

```sh
open tools/macos-camera-bridge/ExpressoCameraBridge.app
```

When macOS asks, grant Camera access to **Expresso Camera Bridge**.

The helper serves the MacBook camera at:

```text
http://127.0.0.1:3988/stream.mjpg
```

The operator page starts with this URL as a convenience camera row. It is not
persisted across server restarts.

The helper prefers the built-in MacBook camera over Continuity Camera devices.
To choose another camera deliberately:

```sh
EXPRESSO_CAMERA_NAME="iPhone" open tools/macos-camera-bridge/ExpressoCameraBridge.app
```

Helper logs are written to:

```text
/tmp/ExpressoCameraBridge.log
```

## Robot Camera URLs

Append one named URL per camera. Examples:

```text
http://10.39.86.11:1181/?action=stream
http://10.39.86.12:1181/?action=stream
rtsp://camera-host/stream
```

Each camera has its own:

- Stream reader.
- ChaRuCo detector state.
- Calibration accumulator.
- RMS history.
- Accepted-frame screenshot directory.

## iPad Target

The target endpoints are unchanged:

```text
GET /target
GET /api/target.png
GET /api/target.pdf
GET /api/qr.png
```

Typical workflow:

1. Open `/target` on the iPad.
2. Choose the iPad model or enter screen dimensions manually.
3. Tap **Show Board** or open the generated PDF.
4. Move the target around each camera until that camera’s RMS digit stabilizes.

The board defaults are:

- Dictionary: `DICT_4X4_50`
- Squares: `9 x 6`
- Marker/square ratio: `0.75`
- Legacy pattern enabled when supported by OpenCV

## Operator UX

The operator page has three zones:

- Source strip: add a camera name and URL, then start or stop all streams.
- Source list: current ephemeral camera rows with remove buttons.
- Stream grid: live camera tiles.

Each tile contains only:

- The mirrored camera stream.
- A centered RMS digit.

RMS color:

- Gray: no solve yet.
- Red: poor RMS, above `1.20 px`.
- Amber: marginal RMS, `0.80-1.20 px`.
- Green: good RMS, at or below `0.80 px`.

Focus behavior:

- A camera focuses when it detects enough ChaRuCo corners.
- If multiple cameras detect the target, the app picks the strongest detection:
  corner count, then board area, then sharpness.
- If the focused camera loses the target, it stays focused for 5 seconds.
- If another camera detects during that hold window, focus switches immediately.
- If no camera detects for 5 seconds, the UI returns to the grid.

## Persistence Model

Camera URLs, calibration accumulators, and solve history are ephemeral. They live
only for the current server process.

The app permanently saves only accepted ChaRuCo screenshots:

```text
runs/YYYYMMDD_HHMMSS/screenshots/<camera-label>/frame_000123.jpg
```

Screenshots are written only when a frame is accepted into that camera’s
calibration candidate set. They include the detected ChaRuCo corners drawn on
the image for auditability.

The live solve no longer writes these files automatically:

```text
calibration.json
report.md
detections.csv
```

## API Surface

Camera management:

```text
GET    /api/cameras
POST   /api/cameras
DELETE /api/cameras/{camera_id}
POST   /api/cameras/start-all
POST   /api/cameras/stop-all
```

Per-camera preview:

```text
GET /api/cameras/{camera_id}/latest.jpg
GET /api/cameras/{camera_id}/stream.mjpg
```

Metrics:

```text
WS /ws/metrics
```

The websocket emits per-camera snapshots (selected fields shown — see
`ManagedCamera.public_snapshot` for the full set):

```json
{
  "focusedCameraId": "cam-1",
  "serverTime": 1770000000.0,
  "cameras": [
    {
      "id": "cam-1",
      "label": "MacBook",
      "url": "http://127.0.0.1:3988/stream.mjpg",
      "running": true,
      "fps": 29.8,
      "detectingCharuco": true,
      "rms": 0.74,
      "rmsDisplay": "0.74",
      "errorGrade": "good",
      "errorColor": "rgb(22, 163, 74)",
      "rmsThresholds": {
        "goodMaxPx": 0.80,
        "marginalMaxPx": 1.20,
        "poorP95MaxPx": 1.80
      },
      "candidateFrames": 42,
      "selectedFrames": 38,
      "rejectedFrames": 4,
      "pipeline": {
        "captureRunning": true,
        "previewRunning": true,
        "detectionRunning": false,
        "solverRunning": false,
        "screenshotRunning": false
      }
    }
  ]
}
```

Session and target metadata:

```text
GET  /api/session
GET  /api/status
POST /api/reset
POST /api/target-metadata
```

## Development

Run tests:

```sh
uv run pytest
```

Compile-check Python:

```sh
uv run python -m compileall src tests
```

Check browser JavaScript:

```sh
node --check src/expresso_calib/web/operator.js
node --check src/expresso_calib/web/target.js
```

Rebuild the MacBook bridge app after editing its Swift source:

```sh
xcrun swiftc -O \
  -framework AVFoundation \
  -framework CoreImage \
  -framework Network \
  tools/macos-camera-bridge/ExpressoCameraBridge.swift \
  -o tools/macos-camera-bridge/ExpressoCameraBridge.app/Contents/MacOS/ExpressoCameraBridge
```

## Project Layout

```text
src/expresso_calib/
  board.py          ChaRuCo board generation and target rendering
  calibration.py    frame scoring, accumulation, solving
  detection.py      OpenCV ArUco/ChaRuCo detection
  multi_camera.py   camera registry and focus arbitration helpers
  reports.py        calibration.json / detections.csv / report.md writers
  server.py         FastAPI app and multi-camera stream runtime
  web/              operator and iPad target pages

tools/
  macos-camera-bridge/          MacBook camera to MJPEG helper app
  macos-camera-intrinsic-probe/ MacBook intrinsic metadata probe

tests/
  test_board_detection.py
  test_calibration_accumulator.py
  test_multi_camera.py
```

## Troubleshooting

If a tile is blank:

- Confirm the source app is running.
- Confirm the URL opens outside Expresso Calib.
- For the MacBook helper, check `http://127.0.0.1:3988/stream.mjpg`.
- Check `/tmp/ExpressoCameraBridge.log`.

If macOS blocks the camera:

- Open System Settings.
- Go to Privacy & Security > Camera.
- Enable camera access for **Expresso Camera Bridge**.
- Quit and reopen the helper app.

If RMS does not improve:

- Move the target through the full image, not only the center.
- Include close and far target scales.
- Include tilted views.
- Avoid glare and motion blur.

## Reference Intrinsic

The repo includes a MacBook camera reference matrix from the original probe:

```text
Reference dimensions: 3040 x 2880

[1503.3333, 0,       1509.9377]
[0,       1503.3333, 1359.7521]
[0,       0,          1]
```

This is manufacturer/sensor-space metadata, not exact pixel-space ground truth
for the stream frames.

## License

No license has been selected yet. Until a license is added, the public
repository is source-available but not explicitly licensed for reuse.
