"use strict";

const elements = {
  form: document.getElementById("cameraForm"),
  label: document.getElementById("cameraLabel"),
  url: document.getElementById("cameraUrl"),
  showTargetQr: document.getElementById("showTargetQr"),
  targetQrDialog: document.getElementById("targetQrDialog"),
  closeTargetQr: document.getElementById("closeTargetQr"),
  targetQrImage: document.getElementById("targetQrImage"),
  targetQrUrl: document.getElementById("targetQrUrl"),
  toggleStreams: document.getElementById("toggleStreams"),
  list: document.getElementById("cameraList"),
  grid: document.getElementById("cameraGrid"),
  rowTemplate: document.getElementById("cameraRowTemplate"),
  tileTemplate: document.getElementById("cameraTileTemplate")
};

const state = {
  ws: null,
  cameras: [],
  focusedCameraId: null,
  targetUrl: "",
  streamsRunning: false,
  streamSrcById: new Map()
};

init();

async function init() {
  const session = await fetchJson("/api/session");
  elements.url.value = session.defaultCameraUrl || "http://127.0.0.1:3988/stream.mjpg";
  elements.url.placeholder = session.defaultCameraUrl || "http://127.0.0.1:3988/stream.mjpg";
  state.targetUrl = session.targetUrl || "/target";
  elements.targetQrUrl.href = state.targetUrl;
  elements.targetQrUrl.textContent = state.targetUrl;

  elements.form.addEventListener("submit", addCamera);
  elements.showTargetQr.addEventListener("click", showTargetQr);
  elements.closeTargetQr.addEventListener("click", closeTargetQr);
  elements.targetQrDialog.addEventListener("click", closeTargetQrOnBackdrop);
  elements.toggleStreams.addEventListener("click", toggleStreams);

  await refreshCameras();
  connectMetrics();
}

async function addCamera(event) {
  event.preventDefault();
  const label = elements.label.value.trim() || "Camera";
  const url = elements.url.value.trim();
  if (!url) return;

  try {
    const response = await fetch("/api/cameras", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label, url })
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

async function removeCamera(cameraId) {
  await fetch(`/api/cameras/${encodeURIComponent(cameraId)}`, { method: "DELETE" });
  state.streamSrcById.delete(cameraId);
  await refreshCameras();
}

async function toggleStreams() {
  const endpoint = state.streamsRunning
    ? "/api/cameras/stop-all"
    : "/api/cameras/start-all";
  elements.toggleStreams.disabled = true;
  try {
    await fetch(endpoint, { method: "POST" });
    await refreshCameras();
  } finally {
    elements.toggleStreams.disabled = false;
  }
}

function showTargetQr() {
  elements.targetQrImage.src = "/api/qr.png?ts=" + Date.now();
  if (typeof elements.targetQrDialog.showModal === "function") {
    elements.targetQrDialog.showModal();
  } else {
    elements.targetQrDialog.setAttribute("open", "");
  }
}

function closeTargetQr() {
  elements.targetQrDialog.close();
}

function closeTargetQrOnBackdrop(event) {
  if (event.target === elements.targetQrDialog) {
    closeTargetQr();
  }
}

async function refreshCameras() {
  const payload = await fetchJson("/api/cameras");
  render(payload);
}

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

function render(payload) {
  state.cameras = payload.cameras || [];
  state.focusedCameraId = payload.focusedCameraId || null;
  state.streamsRunning = state.cameras.some((camera) => camera.running);
  renderStreamToggle();
  renderCameraList(state.cameras);
  renderCameraGrid(state.cameras, state.focusedCameraId);
}

function renderStreamToggle() {
  elements.toggleStreams.textContent = state.streamsRunning ? "Stop All" : "Start All";
  elements.toggleStreams.classList.toggle("primary", !state.streamsRunning);
  elements.toggleStreams.classList.toggle("danger", state.streamsRunning);
}

function renderCameraList(cameras) {
  const existing = new Map(
    Array.from(elements.list.querySelectorAll(".camera-row")).map((row) => [
      row.dataset.cameraId,
      row
    ])
  );

  for (const camera of cameras) {
    let row = existing.get(camera.id);
    if (!row) {
      row = elements.rowTemplate.content.firstElementChild.cloneNode(true);
      row.dataset.cameraId = camera.id;
      row.querySelector(".camera-row-remove").addEventListener("click", () => {
        removeCamera(camera.id);
      });
      elements.list.appendChild(row);
    }
    row.querySelector(".camera-row-label").textContent = camera.label;
    row.querySelector(".camera-row-url").textContent = camera.url;
    row.classList.toggle("is-running", Boolean(camera.running));
    existing.delete(camera.id);
  }

  for (const row of existing.values()) {
    row.remove();
  }
}

function renderCameraGrid(cameras, focusedCameraId) {
  const existing = new Map(
    Array.from(elements.grid.querySelectorAll(".camera-tile")).map((tile) => [
      tile.dataset.cameraId,
      tile
    ])
  );
  const ids = new Set(cameras.map((camera) => camera.id));

  for (const camera of cameras) {
    let tile = existing.get(camera.id);
    if (!tile) {
      tile = elements.tileTemplate.content.firstElementChild.cloneNode(true);
      tile.dataset.cameraId = camera.id;
      const image = tile.querySelector(".camera-feed");
      image.addEventListener("error", () => {
        state.streamSrcById.delete(camera.id);
      });
      elements.grid.appendChild(tile);
    }
    updateTile(tile, camera, focusedCameraId);
    existing.delete(camera.id);
  }

  for (const [cameraId, tile] of existing.entries()) {
    if (!ids.has(cameraId)) {
      const image = tile.querySelector(".camera-feed");
      if (image) {
        image.removeAttribute("src");
        image.replaceWith(image.cloneNode(false));
      }
      state.streamSrcById.delete(cameraId);
      tile.remove();
    }
  }

  elements.grid.classList.toggle("focus-mode", Boolean(focusedCameraId));
}

function updateTile(tile, camera, focusedCameraId) {
  const image = tile.querySelector(".camera-feed");
  const placeholderDetail = tile.querySelector(".camera-placeholder-detail");
  const title = tile.querySelector(".camera-title");
  const accuracyPanel = tile.querySelector(".accuracy-panel");
  const accuracyValue = tile.querySelector(".accuracy-value");
  const accuracyRms = tile.querySelector(".accuracy-rms");
  const accuracyUsed = tile.querySelector(".accuracy-used");
  const accuracyFill = tile.querySelector(".accuracy-meter-fill");
  const badge = tile.querySelector(".detection-badge");
  const log = tile.querySelector(".camera-log");
  const hasLiveFrame = Boolean(camera.connected && camera.hasLatestFrame);
  const shouldStream = hasLiveFrame;
  const streamSrc = `/api/cameras/${encodeURIComponent(camera.id)}/stream.mjpg`;
  const streamRevision = Number(camera.generation || 0);
  const streamKey = `${streamSrc}?rev=${streamRevision}`;
  const detection = camera.detection || {};
  const charucoCount = Number(detection.charucoCount || 0);
  const markerCount = Number(detection.markerCount || 0);
  const minCorners = camera.minCandidateCorners ?? 12;
  const hasCurrentTarget = charucoCount > 0;
  const score = hasCurrentTarget ? accuracyScore(camera.rms, camera.rmsThresholds) : null;
  const grade = hasCurrentTarget ? camera.errorGrade || "pending" : "pending";
  const color = hasCurrentTarget
    ? camera.errorColor || "rgba(255,255,255,0.42)"
    : "rgba(255,255,255,0.36)";

  if (shouldStream && state.streamSrcById.get(camera.id) !== streamKey) {
    image.src = streamKey + "&ts=" + Date.now();
    state.streamSrcById.set(camera.id, streamKey);
  } else if (!shouldStream) {
    image.removeAttribute("src");
    state.streamSrcById.delete(camera.id);
  }

  image.alt = camera.label;
  title.textContent = camera.label;
  placeholderDetail.textContent = cameraPlaceholderText(camera);
  accuracyPanel.className = "accuracy-panel " + grade;
  accuracyPanel.style.setProperty("--accuracy-color", color);
  accuracyPanel.style.setProperty("--accuracy-fill", score === null ? "0%" : `${score}%`);
  accuracyValue.textContent = score === null ? "--" : `${score}%`;
  accuracyRms.textContent = accuracyRmsText(camera, hasCurrentTarget);
  accuracyUsed.textContent = framesUsedText(camera);
  accuracyFill.style.setProperty("--accuracy-color", color);
  badge.textContent = detectionBadgeText(charucoCount, markerCount, minCorners);
  log.textContent = cameraLogText(camera, charucoCount, markerCount, minCorners);
  tile.classList.toggle("is-focused", camera.id === focusedCameraId);
  tile.classList.toggle("is-detecting", Boolean(camera.detectingCharuco));
  tile.classList.toggle("is-partial", charucoCount > 0 && !camera.detectingCharuco);
  tile.classList.toggle("is-offline", Boolean(camera.lastError));
  tile.classList.toggle("has-live-frame", hasLiveFrame);
  tile.classList.toggle("accuracy-good", grade === "good");
  tile.classList.toggle("accuracy-marginal", grade === "marginal");
  tile.classList.toggle("accuracy-poor", grade === "poor");
  tile.classList.toggle("accuracy-pending", grade === "pending");
}

function accuracyScore(rms, thresholds) {
  const value = Number(rms);
  if (!Number.isFinite(value)) return null;
  const perfect = thresholds?.goodMaxPx ?? 0.80;
  const poor = thresholds?.marginalMaxPx ?? 1.20;
  const score = ((poor - value) / (poor - perfect)) * 100;
  return Math.round(Math.max(0, Math.min(100, score)));
}

function framesUsedText(camera) {
  const used = Number(camera.calculationFrames || camera.selectedFrames || 0);
  const accepted = Number(camera.candidateFrames || 0);
  const ready = Number(camera.solvePoolFrames || accepted);
  const minSolve = camera.minSolveFrames ?? 15;
  if (used > 0) {
    if (ready !== accepted) {
      return `${used}/${ready} solve-ready, ${accepted} accepted`;
    }
    return `${used}/${accepted} used`;
  }
  return `${ready}/${minSolve} solve-ready`;
}

function accuracyRmsText(camera, hasCurrentTarget) {
  if (!hasCurrentTarget) {
    return "No target";
  }
  return Number.isFinite(Number(camera.rms))
    ? `${Number(camera.rms).toFixed(2)} px RMS`
    : "No solve";
}

function detectionBadgeText(charucoCount, markerCount, minCorners) {
  if (charucoCount > 0) {
    return `${charucoCount}/${minCorners} corners`;
  }
  if (markerCount > 0) {
    return `${markerCount} marker${markerCount === 1 ? "" : "s"}`;
  }
  return "";
}

function cameraLogText(camera, charucoCount, markerCount, minCorners) {
  if (camera.lastError) {
    return camera.lastError;
  }
  if (camera.pipeline && camera.pipeline.solverRunning) {
    return "solving calibration";
  }
  if (camera.solveDue) {
    return `solve queued: ${Number(camera.acceptedSinceSolve || 0)} new`;
  }
  if (Number(camera.weakSolveFrames || 0) > 0) {
    return `${Number(camera.weakSolveFrames || 0)} weak accepted frames ignored`;
  }
  if (camera.quality && camera.quality.verdict === "REDO") {
    return "quality: redo recommended";
  }
  if (camera.quality && camera.quality.verdict === "MARGINAL") {
    return "quality: marginal";
  }
  if (camera.lastAcceptReason === "duplicate frame") {
    return `deduped still frame (${Number(camera.duplicateImageFrames || 0)})`;
  }
  if (camera.lastAcceptReason === "duplicate pose") {
    return `deduped pose (${Number(camera.duplicatePoseFrames || 0)})`;
  }
  if (!camera.running) {
    return "stopped";
  }
  if (!camera.connected) {
    return "waiting for stream";
  }
  if (charucoCount >= minCorners) {
    return `locked: ${charucoCount} corners`;
  }
  if (charucoCount > 0) {
    return `partial: ${charucoCount}/${minCorners} corners`;
  }
  if (markerCount > 0) {
    return `marker-only: ${markerCount} seen`;
  }
  return "no markers";
}

function cameraPlaceholderText(camera) {
  if (camera.lastError) {
    return camera.lastError;
  }
  if (!camera.running) {
    return "Start streams to connect";
  }
  if (!camera.hasLatestFrame) {
    return "Waiting for the first fresh frame";
  }
  return "Preview is catching up";
}

function wsUrl(path) {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return protocol + "//" + window.location.host + path;
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  return response.json();
}

function flashFormError(detail) {
  elements.form.classList.add("has-error");
  setTimeout(() => elements.form.classList.remove("has-error"), 900);
  let status = elements.form.querySelector(".camera-form-status");
  if (!status) {
    status = document.createElement("small");
    status.className = "camera-form-status";
    status.setAttribute("role", "alert");
    elements.form.appendChild(status);
  }
  status.textContent = detail ? String(detail) : "";
}
