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
  exportAll: document.getElementById("exportAll"),
  useSystemWebcam: document.getElementById("useSystemWebcam"),
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
  streamSrcById: new Map(),
  filterPostTimers: new Map(),
  filterUserTouchAt: new Map(),
  filterRestoredFor: new Set()
};

const FILTER_DEFAULTS = { brightness: 0, contrast: 100, gamma: 1.0, clahe: false };
const FILTER_DEBOUNCE_MS = 150;
const FILTER_TOUCH_GRACE_MS = 1200;
const FILTER_STORAGE_KEY = "expressoFilters_v1";

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
  elements.exportAll.addEventListener("click", exportAllCameras);
  elements.useSystemWebcam.addEventListener("click", () => {
    elements.url.value = "device://0";
    if (!elements.label.value.trim()) {
      elements.label.value = "Webcam";
    }
    elements.url.focus();
  });

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
    clearFormError();
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

async function exportAllCameras() {
  elements.exportAll.disabled = true;
  const original = elements.exportAll.textContent;
  elements.exportAll.textContent = "Exporting…";
  try {
    const response = await fetch("/api/cameras/export-all", { method: "POST" });
    let payload = null;
    try {
      payload = await response.json();
    } catch (e) {}
    if (!response.ok || !payload?.ok) {
      elements.exportAll.textContent = "Export failed";
      return;
    }
    const results = payload.results || [];
    const writes = results.filter((r) => r.ok);
    const skips = results.filter((r) => !r.ok);
    let summary = `Wrote ${writes.length}`;
    if (writes.length === 1) {
      summary += `: ${writes[0].runDir}`;
    } else if (writes.length > 1) {
      summary += " run dirs";
    }
    if (skips.length > 0) {
      summary += ` · skipped ${skips.length} (no solve yet)`;
    }
    elements.exportAll.textContent = summary;
  } catch (err) {
    elements.exportAll.textContent = `Export error: ${err?.message || err}`;
  } finally {
    setTimeout(() => {
      elements.exportAll.textContent = original;
      elements.exportAll.disabled = false;
    }, 4000);
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
  const seenIds = new Set();

  for (const camera of cameras) {
    let row = existing.get(camera.id);
    const isNew = !row;
    if (isNew) {
      row = elements.rowTemplate.content.firstElementChild.cloneNode(true);
      row.dataset.cameraId = camera.id;
      row.querySelector(".camera-row-remove").addEventListener("click", () => {
        removeCamera(camera.id);
      });
      attachFilterHandlers(row, camera.id);
      elements.list.appendChild(row);
    }
    row.querySelector(".camera-row-label").textContent = camera.label;
    row.querySelector(".camera-row-url").textContent = camera.url;
    row.classList.toggle("is-running", Boolean(camera.running));
    row.dataset.cameraLabel = camera.label || "";
    syncFilterValues(row, camera);
    if (isNew) {
      maybeRestoreFilters(camera);
    }
    seenIds.add(camera.id);
    existing.delete(camera.id);
  }

  for (const [id, row] of existing.entries()) {
    row.remove();
    state.filterPostTimers.delete(id);
    state.filterUserTouchAt.delete(id);
    state.filterRestoredFor.delete(id);
  }
}

function attachFilterHandlers(row, cameraId) {
  const toggle = row.querySelector(".camera-row-filters-toggle");
  const panel = row.querySelector(".camera-row-filters");
  toggle.addEventListener("click", () => {
    const open = panel.hasAttribute("hidden");
    if (open) {
      panel.removeAttribute("hidden");
      toggle.setAttribute("aria-expanded", "true");
    } else {
      panel.setAttribute("hidden", "");
      toggle.setAttribute("aria-expanded", "false");
    }
  });

  const onInput = () => {
    state.filterUserTouchAt.set(cameraId, Date.now());
    updateFilterReadouts(row);
    scheduleFilterPost(cameraId, row);
  };

  for (const cls of [".filter-brightness", ".filter-contrast", ".filter-gamma"]) {
    row.querySelector(cls).addEventListener("input", onInput);
  }
  row.querySelector(".filter-clahe").addEventListener("change", onInput);

  row.querySelector(".filter-reset").addEventListener("click", () => {
    setRowFilterValues(row, FILTER_DEFAULTS);
    state.filterUserTouchAt.set(cameraId, Date.now());
    scheduleFilterPost(cameraId, row);
  });
}

function syncFilterValues(row, camera) {
  const lastTouched = state.filterUserTouchAt.get(camera.id) || 0;
  if (Date.now() - lastTouched < FILTER_TOUCH_GRACE_MS) {
    updateFilterReadouts(row);
    return;
  }
  const filters = camera.filters || FILTER_DEFAULTS;
  setRowFilterValues(row, filters);
}

function setRowFilterValues(row, filters) {
  row.querySelector(".filter-brightness").value = String(filters.brightness ?? 0);
  row.querySelector(".filter-contrast").value = String(filters.contrast ?? 100);
  row.querySelector(".filter-gamma").value = String(filters.gamma ?? 1.0);
  row.querySelector(".filter-clahe").checked = Boolean(filters.clahe);
  updateFilterReadouts(row);
}

function updateFilterReadouts(row) {
  const b = row.querySelector(".filter-brightness").value;
  const c = row.querySelector(".filter-contrast").value;
  const g = Number(row.querySelector(".filter-gamma").value);
  row.querySelector(".filter-brightness-value").textContent = b;
  row.querySelector(".filter-contrast-value").textContent = c;
  row.querySelector(".filter-gamma-value").textContent = g.toFixed(2);
}

function readRowFilters(row) {
  return {
    brightness: Number(row.querySelector(".filter-brightness").value) | 0,
    contrast: Number(row.querySelector(".filter-contrast").value) | 0,
    gamma: Number(row.querySelector(".filter-gamma").value),
    clahe: row.querySelector(".filter-clahe").checked
  };
}

function scheduleFilterPost(cameraId, row) {
  const prev = state.filterPostTimers.get(cameraId);
  if (prev) clearTimeout(prev);
  const timer = setTimeout(() => {
    state.filterPostTimers.delete(cameraId);
    const payload = readRowFilters(row);
    saveFilterToStorage(row.dataset.cameraLabel || "", payload);
    fetch(`/api/cameras/${encodeURIComponent(cameraId)}/filters`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).catch(() => {});
  }, FILTER_DEBOUNCE_MS);
  state.filterPostTimers.set(cameraId, timer);
}

function loadAllFiltersFromStorage() {
  try {
    return JSON.parse(localStorage.getItem(FILTER_STORAGE_KEY) || "{}") || {};
  } catch (e) {
    return {};
  }
}

function saveFilterToStorage(label, filters) {
  if (!label) return;
  try {
    const all = loadAllFiltersFromStorage();
    all[label] = filters;
    localStorage.setItem(FILTER_STORAGE_KEY, JSON.stringify(all));
  } catch (e) {}
}

function maybeRestoreFilters(camera) {
  if (state.filterRestoredFor.has(camera.id)) return;
  state.filterRestoredFor.add(camera.id);
  const all = loadAllFiltersFromStorage();
  const saved = all[camera.label || ""];
  if (!saved) return;
  const current = camera.filters || FILTER_DEFAULTS;
  if (filtersEqual(saved, current)) return;
  fetch(`/api/cameras/${encodeURIComponent(camera.id)}/filters`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(saved)
  }).catch(() => {});
}

function filtersEqual(a, b) {
  return (
    Number(a.brightness) === Number(b.brightness) &&
    Number(a.contrast) === Number(b.contrast) &&
    Math.abs(Number(a.gamma) - Number(b.gamma)) < 1e-6 &&
    Boolean(a.clahe) === Boolean(b.clahe)
  );
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
  const convergenceState =
    (camera.quality && camera.quality.convergence && camera.quality.convergence.state) || null;
  tile.dataset.convergence = convergenceState || "";
  tile.classList.toggle("is-converged", convergenceState === "converged");
  tile.classList.toggle("is-diverging", convergenceState === "diverging");
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
  const convergence = camera.quality && camera.quality.convergence;
  const guidance = camera.quality && camera.quality.guidance;
  if (convergence && convergence.state === "converged") {
    return "converged — you can stop and export";
  }
  if (convergence && convergence.state === "diverging") {
    return "diverging — RMS trending up";
  }
  if (guidance) {
    return guidance;
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

function clearFormError() {
  elements.form.classList.remove("has-error");
  const status = elements.form.querySelector(".camera-form-status");
  if (status) status.textContent = "";
}
