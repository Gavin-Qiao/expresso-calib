# Expresso Calib

Expresso Calib is a live ChaRuCo intrinsic-calibration station for robot camera
development. It serves a laptop operator dashboard, an iPad calibration target,
and a source-agnostic OpenCV pipeline that can use the MacBook camera during
development and later swap to robot camera streams without changing the
detector, accumulator, metrics, or export format.

The MVP is intentionally small:

- Python backend with FastAPI and OpenCV.
- Plain HTML/CSS/JS operator and target pages.
- URL-based camera ingest for MJPEG/HTTP/RTSP streams.
- Native macOS helper app that exposes the MacBook camera as an MJPEG URL.
- Live ChaRuCo detection, frame scoring, calibration solve history, and report
  export.

## What It Does

Expresso Calib calibrates a single camera at a time.

The operator laptop shows:

- A QR code for the iPad target page.
- A live camera preview with mirrored movement for easier hand motion.
- ChaRuCo marker/corner detection status.
- Selected frame count, RMS reprojection error, coverage, board area range, and
  focal-length stability trends.
- Run export controls for local calibration artifacts.

The iPad target page shows:

- Manual or preset iPad screen metadata.
- A full-screen-ish browser ChaRuCo target.
- A generated PDF target sized from the chosen iPad screen dimensions for a
  cleaner iPad PDF viewer workflow.

## Current Scope

Implemented:

- Single-camera intrinsic calibration.
- ChaRuCo board generation and detection.
- Live URL camera ingest.
- MacBook camera MJPEG bridge for local testing.
- iPad target page and target PDF endpoint.
- Calibration export under timestamped `runs/` directories.
- Unit tests for board generation, detection, and accumulator behavior.

Not implemented yet:

- Multi-camera auto-lock.
- Robot camera discovery.
- Extrinsic calibration.
- Persistent run browser.
- Printed target workflows.

## Requirements

- macOS for the bundled MacBook camera helper app.
- Python 3.11 or newer.
- `uv`.
- A browser on the laptop for the operator page.
- An iPad or other tablet on the same Wi-Fi for the ChaRuCo target.

The Python dependencies are declared in `pyproject.toml` and locked in
`uv.lock`.

## Quick Start

Install dependencies:

```sh
uv sync
```

Start the calibration server:

```sh
uv run expresso-calib
```

Open the laptop dashboard:

```text
http://127.0.0.1:3987/operator
```

The operator page will show the iPad target QR code and a `Camera URL` field.

## MacBook Camera Development Source

The MacBook camera is exposed as a normal stream URL so it behaves like future
robot cameras.

In a second terminal, launch the helper app:

```sh
open tools/macos-camera-bridge/ExpressoCameraBridge.app
```

When macOS asks, grant Camera access to **Expresso Camera Bridge**.

The helper serves:

```text
http://127.0.0.1:3988/stream.mjpg
```

Use that URL in the operator page and press **Connect**.

To stop using the MacBook helper, press **Disconnect** in the operator page and
quit the helper app.

### Choosing a Different macOS Camera

The helper prefers the built-in MacBook camera over external or Continuity
Camera devices. To deliberately choose another camera, set
`EXPRESSO_CAMERA_NAME` before launching the app:

```sh
EXPRESSO_CAMERA_NAME="iPhone" open tools/macos-camera-bridge/ExpressoCameraBridge.app
```

Other helper settings:

```sh
EXPRESSO_STREAM_PORT=3988
EXPRESSO_STREAM_PATH=/stream.mjpg
EXPRESSO_CAMERA_FPS=30
EXPRESSO_UPLOAD_WIDTH=1280
EXPRESSO_UPLOAD_HEIGHT=720
EXPRESSO_JPEG_QUALITY=0.70
```

Logs are written to:

```text
/tmp/ExpressoCameraBridge.log
```

## Robot Camera Streams

Any camera source that OpenCV can open can be pasted into the operator page.
Examples:

```text
http://10.39.86.11:1181/?action=stream
http://10.39.86.11:1182/?action=stream
rtsp://camera-host/stream
```

The backend reads the URL, converts frames into the common `Frame` structure,
updates the preview, and samples frames into the same ChaRuCo detector used for
the MacBook helper.

Do not point the camera URL at this app's preview proxy:

```text
http://127.0.0.1:3987/api/latest-stream.mjpg
```

That URL is only for the operator page preview. The app rejects it to avoid a
self-referential stream loop.

## iPad Target Workflow

1. Open `/operator` on the laptop.
2. Scan the QR code with the iPad.
3. On the iPad target page, choose the iPad model or enter screen dimensions
   manually.
4. Either tap **Show Board** for the browser target or **Open PDF** for the PDF
   target.
5. Present the iPad to the camera and move it around the full image.

The PDF endpoint is:

```text
/api/target.pdf?width_mm=280.6&height_mm=194.7&landscape=true
```

The PNG endpoint is:

```text
/api/target.png?w=1800&h=1200
```

Both use the same board manifest as the detector.

## ChaRuCo Defaults

The current board configuration is:

- Dictionary: `DICT_4X4_50`
- Squares: `9 x 6`
- Square length: `1.0` normalized unit
- Marker length: `0.75` normalized unit
- Marker-to-square ratio: `0.75`
- Legacy pattern: enabled when supported by OpenCV

The calibration solve uses the relative board geometry. The iPad physical
screen metadata is recorded with the run so you can audit what target was shown.

## Reading The Metrics

`ChaRuCo` is the number of interpolated ChaRuCo corners detected in the current
frame. More corners generally means a stronger frame.

`Accepted` is the number of non-duplicate frames selected for calibration. The
accumulator rejects weak, blurry, too-small, and near-duplicate poses.

`RMS` is the current reprojection RMS error in pixels after a calibration solve.
Lower is better, but a low RMS with poor coverage can still be misleading.

`Coverage` is the span of detected ChaRuCo corners across selected frames as a
fraction of the camera image width and height. Good runs see the board near the
center, edges, and corners.

`Board Area` is the minimum-to-maximum detected board area as a fraction of the
image. A useful run has scale diversity: some far/small views and some
close/large views.

`Cumulative RMS` shows the RMS value from each calibration solve in the current
run.

`Focal Length Trend` shows `fx` and `fy` in pixels from each solve. A converging
run should show these estimates flattening out rather than drifting.

`K Stability` is the recent percent spread of estimated camera matrix terms.
Large movement means the selected frame set is still underconstrained.

## Capture Guidance

For a stronger intrinsic solve:

- Move the target to all image quadrants.
- Include left, right, top, bottom, and corner coverage.
- Capture both close and far board scales.
- Tilt the target in pitch and yaw.
- Avoid motion blur and glare.
- Do not collect every frame from one pose; move slowly but deliberately.

The MVP starts solving after enough accepted frames and keeps updating as more
useful frames arrive. A typical useful run should collect roughly 35 to 80
selected frames.

## Exports

Runs are written under:

```text
runs/YYYYMMDD_HHMMSS/
```

Generated files:

- `calibration.json`: camera matrix, distortion coefficients, quality metrics,
  target metadata, and solve history.
- `report.md`: human-readable calibration report with a verdict.
- `detections.csv`: selected-frame and detection metadata.
- `debug/`: selected frame snapshots with detected corners drawn.

Verdicts are:

- `GOOD`: enough coverage, diversity, stability, and reprojection quality.
- `MARGINAL`: usable but weak in one or more quality dimensions.
- `REDO`: insufficient or unstable data.

`runs/` is gitignored because it contains local calibration artifacts.

## API Surface

Operator and target pages:

```text
GET /operator
GET /target
```

Target assets:

```text
GET /api/target.png
GET /api/target.pdf
GET /api/qr.png
```

Session and status:

```text
GET  /api/session
GET  /api/status
POST /api/reset
POST /api/export
POST /api/target-metadata
```

URL camera source:

```text
POST /api/camera-source/start
POST /api/camera-source/stop
```

Preview:

```text
GET /api/latest-frame.jpg
GET /api/latest-stream.mjpg
WS  /ws/metrics
```

Legacy push ingest remains available for integrations that POST JPEG frames:

```text
POST /api/frames/jpeg
```

## Project Layout

```text
src/expresso_calib/
  board.py          ChaRuCo board generation and PDF/PNG target rendering
  calibration.py    frame scoring, accumulation, solve, export, reports
  detection.py      OpenCV ArUco/ChaRuCo detection
  reference.py      MacBook camera reference metadata
  server.py         FastAPI app, camera URL ingest, websockets, API routes
  sources.py        source boundary types
  web/              operator and iPad target pages

tools/
  macos-camera-bridge/          MacBook camera to MJPEG helper app
  macos-camera-intrinsic-probe/ MacBook intrinsic metadata probe

tests/
  test_board_detection.py
  test_calibration_accumulator.py
```

## Development

Run tests:

```sh
uv run pytest
```

Compile-check Python files:

```sh
uv run python -m compileall src tests
```

Check browser JavaScript syntax:

```sh
node --check src/expresso_calib/web/operator.js
node --check src/expresso_calib/web/target.js
```

Rebuild the MacBook bridge app after editing the Swift source:

```sh
xcrun swiftc -O \
  -framework AVFoundation \
  -framework CoreImage \
  -framework Network \
  tools/macos-camera-bridge/ExpressoCameraBridge.swift \
  -o tools/macos-camera-bridge/ExpressoCameraBridge.app/Contents/MacOS/ExpressoCameraBridge
```

Typecheck the bridge source:

```sh
xcrun swiftc -typecheck tools/macos-camera-bridge/ExpressoCameraBridge.swift
```

## Troubleshooting

If the operator preview is blank:

- Confirm the server is running at `http://127.0.0.1:3987/operator`.
- Confirm the camera URL is reachable.
- For the MacBook helper, confirm the app is open and listening on
  `http://127.0.0.1:3988/stream.mjpg`.
- Quit and reopen the helper after rebuilding it.
- Check `/tmp/ExpressoCameraBridge.log`.

If macOS blocks the camera:

- Open System Settings.
- Go to Privacy & Security > Camera.
- Enable camera access for **Expresso Camera Bridge**.
- Quit and reopen the helper app.

If an iPhone or Continuity Camera is selected instead of the MacBook camera:

- Quit the helper.
- Relaunch with `EXPRESSO_CAMERA_NAME` set to part of the desired camera name.

If calibration looks unstable:

- Add more edge and corner coverage.
- Add more near/far scale diversity.
- Include tilted views.
- Avoid only collecting center-facing, same-distance frames.

## Reference Intrinsic

The project includes a MacBook camera reference matrix from the previous probe:

```text
Reference dimensions: 3040 x 2880

[1503.3333, 0,       1509.9377]
[0,       1503.3333, 1359.7521]
[0,       0,          1]
```

This is treated as manufacturer/sensor-space metadata, not exact pixel-space
ground truth for browser or stream frames, which may be scaled or cropped.

## License

No license has been selected yet. Until a license is added, the public repository
is source-available but not explicitly licensed for reuse.
