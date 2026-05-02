"use strict";

const elements = {
  form: document.getElementById("cameraForm"),
  label: document.getElementById("cameraLabel"),
  url: document.getElementById("cameraUrl"),
  startAll: document.getElementById("startAll"),
  stopAll: document.getElementById("stopAll"),
  list: document.getElementById("cameraList"),
  grid: document.getElementById("cameraGrid"),
  rowTemplate: document.getElementById("cameraRowTemplate"),
  tileTemplate: document.getElementById("cameraTileTemplate")
};

const state = {
  ws: null,
  cameras: [],
  focusedCameraId: null,
  streamSrcById: new Map()
};

init();

async function init() {
  const session = await fetchJson("/api/session");
  elements.url.value = session.defaultCameraUrl || "http://127.0.0.1:3988/stream.mjpg";
  elements.url.placeholder = session.defaultCameraUrl || "http://127.0.0.1:3988/stream.mjpg";

  elements.form.addEventListener("submit", addCamera);
  elements.startAll.addEventListener("click", startAll);
  elements.stopAll.addEventListener("click", stopAll);

  await refreshCameras();
  connectMetrics();
}

async function addCamera(event) {
  event.preventDefault();
  const label = elements.label.value.trim() || "Camera";
  const url = elements.url.value.trim();
  if (!url) return;

  const response = await fetch("/api/cameras", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label, url })
  });
  const payload = await response.json();
  if (!response.ok || !payload.ok) {
    flashFormError();
    return;
  }
  elements.label.value = "";
  await refreshCameras();
}

async function removeCamera(cameraId) {
  await fetch(`/api/cameras/${encodeURIComponent(cameraId)}`, { method: "DELETE" });
  state.streamSrcById.delete(cameraId);
  await refreshCameras();
}

async function startAll() {
  await fetch("/api/cameras/start-all", { method: "POST" });
  await refreshCameras();
}

async function stopAll() {
  await fetch("/api/cameras/stop-all", { method: "POST" });
  await refreshCameras();
}

async function refreshCameras() {
  const payload = await fetchJson("/api/cameras");
  render(payload);
}

function connectMetrics() {
  state.ws = new WebSocket(wsUrl("/ws/metrics"));
  state.ws.addEventListener("message", (event) => {
    render(JSON.parse(event.data));
  });
  state.ws.addEventListener("close", () => {
    setTimeout(connectMetrics, 1000);
  });
}

function render(payload) {
  state.cameras = payload.cameras || [];
  state.focusedCameraId = payload.focusedCameraId || null;
  renderCameraList(state.cameras);
  renderCameraGrid(state.cameras, state.focusedCameraId);
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
      elements.grid.appendChild(tile);
    }
    updateTile(tile, camera, focusedCameraId);
    existing.delete(camera.id);
  }

  for (const [cameraId, tile] of existing.entries()) {
    if (!ids.has(cameraId)) {
      tile.remove();
    }
  }

  elements.grid.classList.toggle("focus-mode", Boolean(focusedCameraId));
}

function updateTile(tile, camera, focusedCameraId) {
  const image = tile.querySelector(".camera-feed");
  const digit = tile.querySelector(".error-digit");
  const shouldStream = camera.running || camera.hasLatestFrame;
  const streamSrc = `/api/cameras/${encodeURIComponent(camera.id)}/stream.mjpg`;

  if (shouldStream && state.streamSrcById.get(camera.id) !== streamSrc) {
    image.src = streamSrc + "?ts=" + Date.now();
    state.streamSrcById.set(camera.id, streamSrc);
  } else if (!shouldStream) {
    image.removeAttribute("src");
    state.streamSrcById.delete(camera.id);
  }

  image.alt = camera.label;
  digit.textContent = camera.rmsDisplay || "--";
  digit.className = "error-digit " + (camera.errorGrade || "pending");
  digit.style.setProperty("--error-color", camera.errorColor || "rgba(255,255,255,0.36)");
  tile.classList.toggle("is-focused", camera.id === focusedCameraId);
  tile.classList.toggle("is-detecting", Boolean(camera.detectingCharuco));
  tile.classList.toggle("is-offline", Boolean(camera.lastError));
}

function wsUrl(path) {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return protocol + "//" + window.location.host + path;
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  return response.json();
}

function flashFormError() {
  elements.form.classList.add("has-error");
  setTimeout(() => elements.form.classList.remove("has-error"), 900);
}
