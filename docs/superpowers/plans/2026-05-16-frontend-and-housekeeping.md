# Frontend & Housekeeping Implementation Plan (Plan C)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the JS↔server threshold drift, fix the frontend bug pile (MJPEG leak, WS reconnect, silent `addCamera`, `target.js` resize spam), document the LAN-only trust model in the README, and add the missing CI / dep-pinning gates the audit flagged.

**Architecture:** Server emits an authoritative `rmsThresholds` block in the WS payload; JS reads it instead of its hardcoded `0.6`/`1.8` magic numbers (which never matched the server's `0.80`/`1.20`). Frontend bugs are local cleanups in `operator.js` / `target.js`. CI added as a single GitHub Actions workflow. Deps pinned via `uv.lock` committed + upper bounds on the two attacker-surface deps (`opencv-contrib-python`, `fastapi`).

**Tech Stack:** Vanilla JS (no build step), `node --check` for syntax, GitHub Actions, `ruff` for lint.

**Out of scope:** Anything not in Plans A or B (those landed first).

---

## Task 1: Server-authoritative thresholds + JS consumption

The server's RMS verdict thresholds live at [server.py:874-881](src/expresso_calib/server.py:874) (`good ≤ 0.80`, `marginal ≤ 1.20`, else `poor`) and [calibration.py summarize_quality](src/expresso_calib/calibration.py) (`> 1.20` red, `> 0.80` yellow, p95 `> 1.80` red). The JS accuracy-meter at [operator.js:259-262](src/expresso_calib/web/operator.js:259) hardcodes `perfectError = 0.6` and `poorError = 1.8` — values that match nothing on the server. Tuning the server thresholds silently desynchronizes the meter.

**Files:**
- Modify: `src/expresso_calib/server.py` (emit `rmsThresholds` in `public_snapshot`)
- Modify: `src/expresso_calib/web/operator.js` (use server values, drop magic numbers and `||` fallbacks where server reliably provides)
- Test: `tests/test_camera_pipeline.py` (assert payload shape includes the new block)

- [ ] **Step 1: Add module-level threshold constants to `server.py`**

Near the other constants (around [server.py:40-48](src/expresso_calib/server.py:40)):

```python
RMS_GOOD_MAX_PX = 0.80
RMS_MARGINAL_MAX_PX = 1.20
RMS_POOR_P95_MAX_PX = 1.80
```

Replace the inline `0.80` / `1.20` literals in `rms_grade` / `rms_color` with the new constants.

- [ ] **Step 2: Emit `rmsThresholds` in `public_snapshot`**

Inside `ManagedCamera.public_snapshot` ([server.py:730-780](src/expresso_calib/server.py:730)), add a sibling key next to `errorGrade` / `errorColor`:

```python
"rmsThresholds": {
    "goodMaxPx": RMS_GOOD_MAX_PX,
    "marginalMaxPx": RMS_MARGINAL_MAX_PX,
    "poorP95MaxPx": RMS_POOR_P95_MAX_PX,
},
```

- [ ] **Step 3: Failing test for payload shape**

Add to `tests/test_camera_pipeline.py`:

```python
def test_public_snapshot_emits_rms_thresholds(tmp_path) -> None:
    from expresso_calib.server import MultiCameraCalibrationState, RMS_GOOD_MAX_PX

    live = MultiCameraCalibrationState()
    camera = live.add_camera("test", "http://example.invalid/stream.mjpg")
    snapshot = camera.public_snapshot(now=0.0)
    assert snapshot["rmsThresholds"]["goodMaxPx"] == RMS_GOOD_MAX_PX
    assert snapshot["rmsThresholds"]["marginalMaxPx"] > snapshot["rmsThresholds"]["goodMaxPx"]
    assert snapshot["rmsThresholds"]["poorP95MaxPx"] > snapshot["rmsThresholds"]["marginalMaxPx"]
```

- [ ] **Step 4: Run test**

```
uv run pytest tests/test_camera_pipeline.py::test_public_snapshot_emits_rms_thresholds -v
```

Expected: PASS (the constants are defined and the snapshot includes them).

- [ ] **Step 5: Update `operator.js` to use server-emitted thresholds**

Replace the `accuracyScore` function ([operator.js:256-263](src/expresso_calib/web/operator.js:256)) with one that takes the thresholds:

```javascript
function accuracyScore(rms, thresholds) {
  const value = Number(rms);
  if (!Number.isFinite(value)) return null;
  const perfect = thresholds?.goodMaxPx ?? 0.80;
  const poor = thresholds?.marginalMaxPx ?? 1.20;
  const score = ((poor - value) / (poor - perfect)) * 100;
  return Math.round(Math.max(0, Math.min(100, score)));
}
```

Note the fallback to `0.80 / 1.20` matches the server's published defaults — fallbacks are only there for the first metric frame before the payload arrives, not as a parallel source of truth.

Update the call site in `updateTile` (around [operator.js:219](src/expresso_calib/web/operator.js:219)):

```javascript
const score = hasCurrentTarget ? accuracyScore(camera.rms, camera.rmsThresholds) : null;
```

- [ ] **Step 6: Drop the `|| 12` / `|| 15` magic-number fallbacks where the server reliably provides**

In `updateTile` and `framesUsedText` and any other place that does `Number(camera.minCandidateCorners || 12)` / `Number(camera.minSolveFrames || 15)`, drop the `||` fallback — the server always emits them on a real connection. To keep first-frame robustness, use:

```javascript
const minCorners = camera.minCandidateCorners ?? 12;
const minSolve = camera.minSolveFrames ?? 15;
```

(`??` only falls back on `null`/`undefined`, not on zero — important because zero should be allowed.)

- [ ] **Step 7: `node --check` the JS files**

```
node --check src/expresso_calib/web/operator.js
node --check src/expresso_calib/web/target.js
```

Expected: no errors.

- [ ] **Step 8: Run full test suite**

```
uv run pytest -v
```

Expected: all PASS (one more test than Plan B's final count).

- [ ] **Step 9: Commit**

```bash
git add src/expresso_calib/server.py src/expresso_calib/web/operator.js tests/test_camera_pipeline.py
git commit -m "$(cat <<'EOF'
feat: emit RMS thresholds in WS payload and consume in operator UI

operator.js's accuracy meter hardcoded perfectError=0.6 and
poorError=1.8 — values that never matched the server's actual grade
thresholds (good <= 0.80, marginal <= 1.20). Tuning either side
silently desynchronized the visual feedback. Promote the three RMS
constants on the server, emit them as rmsThresholds in
public_snapshot, and rewrite accuracyScore to consume them. Drop the
|| magic-number fallbacks for minCandidateCorners / minSolveFrames
where the server reliably emits them; keep ?? for the first-frame
window before the payload arrives.
EOF
)"
```

---

## Task 2: Frontend bug pile

Four independent fixes in operator.js + target.js:
- **MJPEG `<img>` teardown on DELETE:** `removeCamera` calls `state.streamSrcById.delete(...)` then `refreshCameras()`; the WS re-render rebuilds and `renderCameraGrid` removes the tile via `tile.remove()` without first clearing `image.src` or removing the per-tile `image.addEventListener("error", …)`. The open MJPEG HTTP connection may stay in CLOSE_WAIT until GC; the error listener closes over a stale `camera.id`.
- **WS reconnect:** `connectMetrics` ([operator.js:111-119](src/expresso_calib/web/operator.js:111)) blindly reconnects after `close` with `setTimeout(..., 1000)`, no `onerror`, no jitter, no cap, never `removeEventListener`s the previous `state.ws`. A flap can spawn parallel sockets that stomp `state.cameras` / `state.focusedCameraId`.
- **Silent `addCamera`:** `await fetch(...)` then `response.json()` with no try/catch ([operator.js:54-63](src/expresso_calib/web/operator.js:54)); a network error or non-JSON 5xx throws and the form just freezes.
- **`refreshBoardImage` resize storm:** bound directly to `window resize` with cache-busting `Date.now()` ([target.js:107,226-232](src/expresso_calib/web/target.js:107)); iPad rotation fires many resize events, each forces a server PNG re-render.

**Files:**
- Modify: `src/expresso_calib/web/operator.js`
- Modify: `src/expresso_calib/web/target.js`
- No tests — there's no JS test infra. `node --check` is the verification gate.

- [ ] **Step 1: Fix MJPEG `<img>` teardown**

In `renderCameraGrid` ([operator.js:189-193](src/expresso_calib/web/operator.js:189)), change the existing-tile teardown loop:

```javascript
for (const [cameraId, tile] of existing.entries()) {
  if (!ids.has(cameraId)) {
    const image = tile.querySelector(".camera-feed");
    if (image) {
      image.removeAttribute("src");
      image.replaceWith(image.cloneNode(false));  // drops all listeners
    }
    state.streamSrcById.delete(cameraId);
    tile.remove();
  }
}
```

- [ ] **Step 2: Fix WS reconnect — jitter, cap, cleanup, no duplicate handlers**

Replace `connectMetrics` ([operator.js:111-119](src/expresso_calib/web/operator.js:111)) with:

```javascript
let metricsReconnectAttempts = 0;
const METRICS_RECONNECT_MAX_DELAY_MS = 15000;

function connectMetrics() {
  if (state.ws) {
    try { state.ws.close(); } catch (e) {}
    state.ws = null;
  }
  const ws = new WebSocket(wsUrl("/ws/metrics"));
  state.ws = ws;
  ws.addEventListener("open", () => {
    metricsReconnectAttempts = 0;
  });
  ws.addEventListener("message", (event) => {
    if (state.ws === ws) {
      render(JSON.parse(event.data));
    }
  });
  ws.addEventListener("close", () => {
    if (state.ws !== ws) return;
    state.ws = null;
    metricsReconnectAttempts++;
    const base = Math.min(
      METRICS_RECONNECT_MAX_DELAY_MS,
      1000 * Math.pow(2, Math.min(6, metricsReconnectAttempts))
    );
    const jitter = Math.random() * 0.5 + 0.75;
    setTimeout(connectMetrics, base * jitter);
  });
  ws.addEventListener("error", () => {
    try { ws.close(); } catch (e) {}
  });
}
```

- [ ] **Step 3: Make `addCamera` surface network errors**

Replace `addCamera` ([operator.js:48-66](src/expresso_calib/web/operator.js:48)) with:

```javascript
async function addCamera(event) {
  event.preventDefault();
  const label = elements.label.value.trim() || "Camera";
  const url = elements.url.value.trim();
  if (!url) return;

  try {
    const response = await fetch("/api/cameras", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label, url }),
    });
    let payload = null;
    try {
      payload = await response.json();
    } catch (e) {}
    if (!response.ok || !payload?.ok) {
      flashFormError(payload?.detail);
      return;
    }
    elements.label.value = "";
    await refreshCameras();
  } catch (err) {
    flashFormError(String(err?.message || err || "Network error"));
  }
}
```

If `flashFormError` doesn't accept a message today, extend it. (Check the existing signature first — `grep -n "function flashFormError" src/expresso_calib/web/operator.js`. If it's `()`, change to `(detail)` and append the message to the form's status element when present.)

- [ ] **Step 4: Debounce `refreshBoardImage` resize spam**

Replace `target.js:107` and the `refreshBoardImage` declaration:

```javascript
let pendingResizeId = null;
let lastBoardSizeKey = "";

function scheduleBoardRefresh() {
  if (pendingResizeId !== null) {
    clearTimeout(pendingResizeId);
  }
  pendingResizeId = setTimeout(() => {
    pendingResizeId = null;
    refreshBoardImage();
  }, 200);
}

function refreshBoardImage() {
  const rect = elements.stage.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(900, Math.round(rect.width * ratio));
  const height = Math.max(650, Math.round(rect.height * ratio));
  const key = `${width}x${height}`;
  if (key === lastBoardSizeKey) return;
  lastBoardSizeKey = key;
  elements.image.src = `/api/target.png?w=${width}&h=${height}&ts=${Date.now()}`;
}

window.addEventListener("resize", scheduleBoardRefresh);
window.addEventListener("orientationchange", () => setTimeout(scheduleBoardRefresh, 250));
```

Replace any other call site of `refreshBoardImage()` that the existing code uses for *deliberate* refresh (e.g., after applying a new preset) — those should still call `refreshBoardImage()` directly to bypass the size-key cache. To force a refresh, the caller can set `lastBoardSizeKey = ""` then call `refreshBoardImage()`. Grep `grep -n "refreshBoardImage" src/expresso_calib/web/target.js` to confirm only the resize/orientation paths are debounced.

- [ ] **Step 5: `node --check` both files**

```
node --check src/expresso_calib/web/operator.js
node --check src/expresso_calib/web/target.js
```

Expected: no errors.

- [ ] **Step 6: Manual smoke test** (mark complete after spot-checking; no automation)

Run the server (`uv run expresso-calib`), open `/operator` in a browser:
- Add a fake camera URL, then DELETE it — verify the tile disappears cleanly with no console errors.
- Open dev tools → Network, kill the server, restart — verify the WS reconnects with backoff (not a tight 1-second loop).
- POST a bad camera URL — verify the form shows an error message instead of freezing.
- Open `/target` on the iPad (or resize the browser repeatedly) — verify only one `/api/target.png` request per stable size, not per resize event.

- [ ] **Step 7: Commit**

```bash
git add src/expresso_calib/web/operator.js src/expresso_calib/web/target.js
git commit -m "$(cat <<'EOF'
fix: frontend bug pile (MJPEG teardown, WS reconnect, addCamera errors, resize debounce)

- removeCamera removed the tile via tile.remove() without first
  clearing the MJPEG <img>'s src or stripping its closure-captured
  error listener; the open HTTP connection could linger in CLOSE_WAIT.
  Clear src + clone the node to drop listeners before removal.
- connectMetrics had no jitter, no cap, no removeEventListener and
  could spawn parallel sockets on flap that stomped state.cameras.
  Rewrite with exponential backoff capped at 15s, ±25% jitter, and
  a guard that ignores messages from a stale socket.
- addCamera awaited fetch + response.json() with no try/catch; any
  network error or non-JSON 5xx threw and the form froze silently.
  Wrap in try/catch, surface the message via flashFormError.
- target.js bound refreshBoardImage directly to window resize, so
  iPad rotation fired many resize events each downloading a fresh
  PNG. Debounce with 200ms timeout and skip if the computed size key
  matches the last render.
EOF
)"
```

---

## Task 3: Ship gates — README, CI, dep pinning

Three independent housekeeping items the audit flagged:
- README disclaims `Authentication or remote deployment` but doesn't explicitly say "LAN-only trust model — do not expose to the internet." The SSRF discussion (Plan A decision #5) was deferred to README.
- No `.github/workflows/`, no `pre-commit`, no `ruff`/`mypy`. For a 3.3k-line asyncio FastAPI app, this is a gap.
- `pyproject.toml` uses only lower bounds; no `uv.lock` is committed. The exact package decoding attacker-supplied MJPEG (`opencv-contrib-python`) drifts on every `uv sync`.

**Files:**
- Modify: `README.md`
- Create: `.github/workflows/ci.yml`
- Modify: `pyproject.toml` (upper bounds on the security-sensitive deps + add `ruff` to dev)
- Commit: `uv.lock`

- [ ] **Step 1: README LAN-only scope note**

Add a new section right after the existing "Not implemented" bullet list at [README.md:36-41](README.md:36):

```markdown
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
```

- [ ] **Step 2: Add `ruff` to dev deps and a minimal config**

In `pyproject.toml`, extend the dev group:

```toml
[dependency-groups]
dev = [
  "pytest>=8",
  "pytest-asyncio>=0.23",
  "ruff>=0.6",
]
```

Add a `[tool.ruff]` block at the bottom:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"
extend-exclude = [".claude", "tools"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM"]
ignore = ["E501"]  # line length handled by formatter
```

Run `uv sync` to install.

- [ ] **Step 3: Add upper bounds on security-sensitive deps**

In `pyproject.toml` dependencies block, change:

```toml
"opencv-contrib-python>=4.8",
"fastapi>=0.115",
```

to:

```toml
"opencv-contrib-python>=4.8,<5",
"fastapi>=0.115,<1",
```

The others (`numpy`, `pillow`, `python-multipart`, `qrcode`, `uvicorn`) are lower-risk; leave them with only lower bounds. The `uv.lock` (next step) is the real defense.

- [ ] **Step 4: Commit `uv.lock`**

```
uv lock
ls uv.lock  # confirm it exists
```

(`uv lock` will create or update the lockfile.)

- [ ] **Step 5: Add `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          version: latest
      - name: Set up Python
        run: uv python install 3.11
      - name: Sync dependencies
        run: uv sync --all-extras --dev
      - name: Lint
        run: uv run ruff check src tests
      - name: Format check
        run: uv run ruff format --check src tests
      - name: Compile
        run: uv run python -m compileall src tests
      - name: Test
        run: uv run pytest -v
      - name: JS syntax check
        run: |
          node --check src/expresso_calib/web/operator.js
          node --check src/expresso_calib/web/target.js
```

- [ ] **Step 6: Make `ruff check` and `ruff format --check` pass locally**

```
uv run ruff check src tests --fix
uv run ruff format src tests
uv run ruff check src tests
uv run ruff format --check src tests
```

Expected: no errors. If ruff changes the code, review the diff before staging — should be whitespace / import-ordering only on a project this size.

- [ ] **Step 7: Run full verification**

```
uv run pytest -v
uv run python -m compileall src tests
node --check src/expresso_calib/web/operator.js
node --check src/expresso_calib/web/target.js
```

Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add README.md pyproject.toml uv.lock .github/workflows/ci.yml src/ tests/
git commit -m "$(cat <<'EOF'
chore: README LAN-trust note, CI workflow, ruff, dep pinning

Document that this app is for trusted LAN use only — no auth, no
SSRF filtering by design, do not expose ports 3987/3988 to the
internet. Add a GitHub Actions workflow that runs ruff lint+format
check, compileall, pytest, and node --check on both JS files. Add
ruff to dev deps with a minimal config. Pin upper bounds on the
two security-sensitive deps (opencv-contrib-python, fastapi) and
commit uv.lock as the real defense against drift.
EOF
)"
```

---

## Verification checklist (after all three tasks)

- [ ] `uv run pytest -v` — all green
- [ ] `uv run ruff check src tests` — no errors
- [ ] `uv run ruff format --check src tests` — clean
- [ ] `node --check src/expresso_calib/web/*.js` — no errors
- [ ] `git log --oneline c2dbc97..HEAD` — Plan A + Plan B + Plan C commits all present
- [ ] `cat .github/workflows/ci.yml` — exists
- [ ] `ls uv.lock` — exists
- [ ] `grep -n "LAN" README.md` — Trust Model section present
- [ ] Manual: open `/operator`, observe MJPEG teardown / WS reconnect / addCamera errors / target.js resize all behave per the bug-fix descriptions
