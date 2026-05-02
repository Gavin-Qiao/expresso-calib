"use strict";

const elements = {
  cameraSourceForm: document.getElementById("cameraSourceForm"),
  cameraUrl: document.getElementById("cameraUrl"),
  cameraToggle: document.getElementById("cameraToggle"),
  resetRun: document.getElementById("resetRun"),
  exportRun: document.getElementById("exportRun"),
  preview: document.getElementById("preview"),
  backendPreview: document.getElementById("backendPreview"),
  videoStage: document.querySelector(".video-stage"),
  capture: document.getElementById("capture"),
  overlay: document.getElementById("overlay"),
  statusDot: document.getElementById("statusDot"),
  guidance: document.getElementById("guidance"),
  videoMeta: document.getElementById("videoMeta"),
  qr: document.getElementById("qr"),
  targetUrl: document.getElementById("targetUrl"),
  charucoCount: document.getElementById("charucoCount"),
  markerCount: document.getElementById("markerCount"),
  candidateFrames: document.getElementById("candidateFrames"),
  selectedFrames: document.getElementById("selectedFrames"),
  rms: document.getElementById("rms"),
  verdict: document.getElementById("verdict"),
  kStability: document.getElementById("kStability"),
  runDir: document.getElementById("runDir"),
  framesSeen: document.getElementById("framesSeen"),
  coverage: document.getElementById("coverage"),
  boardArea: document.getElementById("boardArea"),
  targetMeta: document.getElementById("targetMeta"),
  rmsTrend: document.getElementById("rmsTrend"),
  focalTrend: document.getElementById("focalTrend"),
  log: document.getElementById("log")
};

const state = {
  metricsWs: null,
  backendPreviewTimer: 0,
  mode: "idle",
  latestMetrics: null,
  rmsHistory: [],
  focalHistory: []
};

init();

async function init() {
  await loadSession();
  connectMetrics();
  elements.videoStage.classList.add("mirrored");
  elements.cameraSourceForm.addEventListener("submit", (event) => {
    event.preventDefault();
    toggleCameraStream();
  });
  elements.resetRun.addEventListener("click", resetRun);
  elements.exportRun.addEventListener("click", exportRun);
  window.addEventListener("resize", () => drawOverlay(state.latestMetrics));
}

async function loadSession() {
  const response = await fetch("/api/session");
  const data = await response.json();
  elements.targetUrl.textContent = data.targetUrl;
  elements.cameraUrl.value = data.defaultCameraUrl || "http://127.0.0.1:3988/stream.mjpg";
  elements.cameraUrl.placeholder = data.defaultCameraUrl || "http://127.0.0.1:3988/stream.mjpg";
  elements.qr.src = "/api/qr.png?ts=" + Date.now();
}

function wsUrl(path) {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return protocol + "//" + window.location.host + path;
}

function connectMetrics() {
  state.metricsWs = new WebSocket(wsUrl("/ws/metrics"));
  state.metricsWs.addEventListener("message", (event) => {
    const metrics = JSON.parse(event.data);
    state.latestMetrics = metrics;
    renderMetrics(metrics);
    drawOverlay(metrics);
  });
  state.metricsWs.addEventListener("close", () => {
    setLog("Metrics connection closed. Reconnecting...");
    setTimeout(connectMetrics, 1000);
  });
}

async function toggleCameraStream() {
  if (isCameraActive(state.latestMetrics)) {
    await stopCameraStream();
  } else {
    await startCameraStream();
  }
}

async function startCameraStream() {
  elements.cameraToggle.disabled = true;
  elements.cameraUrl.disabled = true;
  setDot("warn");
  setLog("Connecting camera URL...");
  try {
    const response = await fetch("/api/camera-source/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: elements.cameraUrl.value.trim() })
    });
    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || "Camera URL failed.");
    }
    state.latestMetrics = data.metrics || state.latestMetrics;
    updateCameraToggle(state.latestMetrics);
    setLog("Camera URL connected. Waiting for frames...");
  } catch (error) {
    setDot("error");
    setLog("Could not connect camera URL: " + error.message);
    elements.cameraUrl.disabled = false;
  } finally {
    elements.cameraToggle.disabled = false;
  }
}

async function stopCameraStream() {
  elements.cameraToggle.disabled = true;
  setLog("Disconnecting camera URL...");
  try {
    const stopPath = state.latestMetrics?.urlCamera?.running
      ? "/api/camera-source/stop"
      : "/api/native-bridge/stop";
    const response = await fetch(stopPath, { method: "POST" });
    const data = await response.json();
    state.latestMetrics = data.metrics || state.latestMetrics;
    stopPreviewPolling();
    state.mode = "idle";
    elements.videoStage.classList.remove("backend-mode");
    elements.backendPreview.removeAttribute("src");
    elements.cameraUrl.disabled = false;
    setDot("");
    setLog("Camera URL disconnected.");
    updateCameraToggle(state.latestMetrics);
  } catch (error) {
    setDot("error");
    setLog("Could not disconnect camera URL: " + error.message);
  } finally {
    elements.cameraToggle.disabled = false;
  }
}

function refreshLatestPreview() {
  if (state.mode !== "stream" || elements.backendPreview.src) return;
  elements.backendPreview.src = "/api/latest-stream.mjpg?ts=" + Date.now();
}

function startPreviewPolling() {
  refreshLatestPreview();
}

function stopPreviewPolling() {
  elements.backendPreview.removeAttribute("src");
}

async function resetRun() {
  const response = await fetch("/api/reset", { method: "POST" });
  const metrics = await response.json();
  state.rmsHistory = [];
  state.focalHistory = [];
  renderMetrics(metrics);
  drawOverlay(metrics);
  setLog("Run reset.");
}

async function exportRun() {
  const response = await fetch("/api/export", { method: "POST" });
  const data = await response.json();
  setLog("Report saved: " + data.runDir);
}

function renderMetrics(metrics) {
  const detection = metrics.detection || {};
  const calibration = metrics.calibration || null;
  const quality = metrics.quality || null;
  updateCameraToggle(metrics);
  elements.guidance.textContent = metrics.guidance || "Waiting for frames.";
  elements.charucoCount.textContent = detection.charucoCount || 0;
  elements.markerCount.textContent = (detection.markerCount || 0) + " markers";
  elements.candidateFrames.textContent = metrics.candidateFrames || 0;
  elements.selectedFrames.textContent = (quality?.selectedFrames || calibration?.selectedFrames || 0) + " selected";
  elements.framesSeen.textContent = metrics.totalFramesSeen || 0;
  elements.runDir.textContent = shortenPath(metrics.runDir || "--");
  const fps = metrics.urlCamera?.fps || metrics.nativeBridge?.fps;
  const fpsText = typeof fps === "number" && fps > 0 ? ` | ${fps.toFixed(1)} FPS` : "";
  elements.videoMeta.textContent = detection.frameSize
    ? `${detection.frameSize.width}x${detection.frameSize.height}${fpsText} | ${metrics.lastAcceptReason || "observing"}`
    : "idle";
  elements.targetMeta.textContent = metrics.targetMetadata?.model || "unknown/manual";

  if (calibration) {
    const rms = calibration.rmsReprojectionErrorPx;
    elements.rms.textContent = rms.toFixed(3);
    elements.verdict.textContent = quality?.verdict || "pending";
  } else {
    elements.rms.textContent = "--";
    elements.verdict.textContent = "pending";
  }
  updateTrendData(metrics, calibration);

  const stability = metrics.trends?.kStabilityPct;
  elements.kStability.textContent = typeof stability === "number" ? stability.toFixed(2) + "%" : "--";

  if (quality?.coverage) {
    elements.coverage.textContent =
      Math.round(quality.coverage.widthFraction * 100) + "% x " +
      Math.round(quality.coverage.heightFraction * 100) + "%";
  } else {
    elements.coverage.textContent = "--";
  }
  if (quality?.boardAreaFraction) {
    elements.boardArea.textContent =
      Math.round(quality.boardAreaFraction.min * 100) + "-" +
      Math.round(quality.boardAreaFraction.max * 100) + "%";
  } else {
    elements.boardArea.textContent = "--";
  }

  if (metrics.urlCamera?.url && metrics.urlCamera.url !== elements.cameraUrl.value) {
    elements.cameraUrl.value = metrics.urlCamera.url;
  }

  if (metrics.error) {
    setDot("error");
    setLog(metrics.error);
  } else if (metrics.urlCamera?.lastError) {
    setDot("error");
    setLog(metrics.urlCamera.lastError);
  } else if (isCameraActive(metrics)) {
    state.mode = "stream";
    elements.videoStage.classList.add("backend-mode");
    startPreviewPolling();
    setDot(detection.charucoCount >= 12 ? "live" : "warn");
    setLog("Camera URL running. Frames: " + (metrics.totalFramesSeen || 0));
  } else if (detection.charucoCount >= 12) {
    setDot("live");
  } else if (metrics.urlCamera === null || metrics.nativeBridge?.enabled === false) {
    state.mode = "idle";
    stopPreviewPolling();
    elements.videoStage.classList.remove("backend-mode");
    elements.backendPreview.removeAttribute("src");
    elements.cameraUrl.disabled = false;
  }

  drawTrend(elements.rmsTrend, state.rmsHistory, "#127f84");
  drawFocalTrend(elements.focalTrend, state.focalHistory);
}

function updateTrendData(metrics, calibration) {
  const solveHistory = metrics.trends?.solveHistory || [];
  if (solveHistory.length > 0) {
    state.rmsHistory = solveHistory.map((item) => item.rmsReprojectionErrorPx);
    state.focalHistory = solveHistory.map((item) => ({ fx: item.fx, fy: item.fy }));
    return;
  }
  const rmsHistory = metrics.trends?.rmsHistory || [];
  if (rmsHistory.length > 0) {
    state.rmsHistory = rmsHistory.map((item) => item.rmsReprojectionErrorPx);
  } else if (!calibration) {
    state.rmsHistory = [];
    state.focalHistory = [];
  }
}

function isCameraActive(metrics) {
  if (!metrics) return false;
  if (metrics.urlCamera?.running) return true;
  if (metrics.nativeBridge?.enabled === false) return false;
  if (metrics.sourceId === "native_camera_bridge") return true;
  const lastFrameAt = metrics.nativeBridge?.lastFrameAt;
  return typeof lastFrameAt === "number" && (Date.now() / 1000 - lastFrameAt) < 3;
}

function updateCameraToggle(metrics) {
  const active = isCameraActive(metrics);
  elements.cameraToggle.textContent = active ? "Disconnect" : "Connect";
  elements.cameraToggle.classList.toggle("danger", active);
  elements.cameraToggle.classList.toggle("primary", !active);
  elements.cameraUrl.disabled = active;
}

function drawOverlay(metrics) {
  const canvas = elements.overlay;
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const detection = metrics?.detection;
  if (!detection?.frameSize) return;
  const sx = canvas.width / detection.frameSize.width;
  const sy = canvas.height / detection.frameSize.height;
  const points = detection.points || [];
  ctx.lineWidth = 2 * ratio;
  ctx.strokeStyle = "rgba(18, 127, 132, 0.95)";
  ctx.fillStyle = "rgba(18, 127, 132, 0.95)";

  const polygon = detection.boardPolygon || [];
  if (polygon.length > 2) {
    ctx.beginPath();
    polygon.forEach((point, index) => {
      const x = point[0] * sx;
      const y = point[1] * sy;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();
  }

  for (const point of points) {
    ctx.beginPath();
    ctx.arc(point[0] * sx, point[1] * sy, 3.2 * ratio, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawTrend(canvas, values, color) {
  resizeTrendCanvas(canvas);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawGrid(ctx, canvas);
  if (values.length < 2) return;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(0.001, max - min);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2 * (window.devicePixelRatio || 1);
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = index / (values.length - 1) * canvas.width;
    const y = canvas.height - ((value - min) / span) * canvas.height;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function drawFocalTrend(canvas, values) {
  resizeTrendCanvas(canvas);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawGrid(ctx, canvas);
  if (values.length < 2) return;
  const all = values.flatMap((item) => [item.fx, item.fy]);
  const min = Math.min(...all);
  const max = Math.max(...all);
  const span = Math.max(0.001, max - min);
  drawSeries(ctx, canvas, values.map((item) => item.fx), min, span, "#3157a4");
  drawSeries(ctx, canvas, values.map((item) => item.fy), min, span, "#b7791f");
}

function drawSeries(ctx, canvas, values, min, span, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2 * (window.devicePixelRatio || 1);
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = index / (values.length - 1) * canvas.width;
    const y = canvas.height - ((value - min) / span) * canvas.height;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function resizeTrendCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * ratio));
  canvas.height = Math.max(1, Math.round(rect.height * ratio));
}

function drawGrid(ctx, canvas) {
  ctx.strokeStyle = "rgba(96,112,116,0.18)";
  ctx.lineWidth = window.devicePixelRatio || 1;
  for (let i = 1; i < 4; i++) {
    const y = i / 4 * canvas.height;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }
}

function setDot(mode) {
  elements.statusDot.className = "dot" + (mode ? " " + mode : "");
}

function setLog(message) {
  elements.log.textContent = message;
}

function shortenPath(path) {
  if (!path || path === "--") return "--";
  const parts = path.split("/");
  return parts.slice(-2).join("/");
}
