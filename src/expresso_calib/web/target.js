"use strict";

const elements = {
  stage: document.getElementById("targetStage"),
  image: document.getElementById("boardImage"),
  form: document.getElementById("targetForm"),
  modelSelect: document.getElementById("modelSelect"),
  widthMm: document.getElementById("widthMm"),
  heightMm: document.getElementById("heightMm"),
  fullscreen: document.getElementById("fullscreen"),
  pdfLink: document.getElementById("pdfLink"),
  installHint: document.getElementById("installHint"),
  status: document.getElementById("targetStatus")
};

const IPAD_PRESETS = [
  {
    id: "manual",
    label: "Unknown / manual",
    widthMm: "",
    heightMm: "",
    diagonalIn: ""
  },
  {
    id: "ipad-9-7",
    label: "iPad 9.7-inch",
    widthMm: "197.1",
    heightMm: "147.8",
    diagonalIn: "9.7"
  },
  {
    id: "ipad-10-2",
    label: "iPad 10.2-inch",
    widthMm: "207.2",
    heightMm: "155.4",
    diagonalIn: "10.2"
  },
  {
    id: "ipad-10-9",
    label: "iPad 10.9-inch / 11-inch Air",
    widthMm: "228.1",
    heightMm: "158.4",
    diagonalIn: "10.9"
  },
  {
    id: "ipad-pro-11",
    label: "iPad Pro 11-inch",
    widthMm: "232.0",
    heightMm: "159.9",
    diagonalIn: "11.0"
  },
  {
    id: "ipad-air-13",
    label: "iPad Air 13-inch",
    widthMm: "280.6",
    heightMm: "194.7",
    diagonalIn: "13.0"
  },
  {
    id: "ipad-pro-12-9",
    label: "iPad Pro 12.9-inch",
    widthMm: "262.3",
    heightMm: "196.7",
    diagonalIn: "12.9"
  },
  {
    id: "ipad-pro-13",
    label: "iPad Pro 13-inch",
    widthMm: "281.6",
    heightMm: "195.4",
    diagonalIn: "13.0"
  },
  {
    id: "ipad-mini-8-3",
    label: "iPad mini 8.3-inch",
    widthMm: "178.5",
    heightMm: "121.4",
    diagonalIn: "8.3"
  }
];

populateModelSelect();
loadSavedMetadata();
renderInstallHint();
refreshBoardImage();
refreshPdfLink();

elements.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await saveMetadata();
});

elements.modelSelect.addEventListener("change", () => {
  applySelectedPreset({ overwriteManual: true });
  refreshPdfLink();
});

elements.widthMm.addEventListener("input", refreshPdfLink);
elements.heightMm.addEventListener("input", refreshPdfLink);

elements.fullscreen.addEventListener("click", async () => {
  await saveMetadata();
  document.body.classList.add("target-fullscreen");
  lastBoardSizeKey = "";
  refreshBoardImage();
});

window.addEventListener("resize", scheduleBoardRefresh);
window.addEventListener("orientationchange", () => setTimeout(scheduleBoardRefresh, 250));

function populateModelSelect() {
  elements.modelSelect.replaceChildren();
  for (const preset of IPAD_PRESETS) {
    const option = document.createElement("option");
    option.value = preset.id;
    option.textContent = preset.label;
    elements.modelSelect.appendChild(option);
  }
}

function loadSavedMetadata() {
  const saved = JSON.parse(localStorage.getItem("expressoTargetMetadata") || "{}");
  const savedPreset = presetForSavedMetadata(saved);
  elements.modelSelect.value = savedPreset.id;
  elements.widthMm.value = saved.screen_width_mm || "";
  elements.heightMm.value = saved.screen_height_mm || "";
  applySelectedPreset({ overwriteManual: !saved.screen_width_mm || !saved.screen_height_mm });
}

function presetForSavedMetadata(saved) {
  if (saved.screen_preset_id) {
    const byId = IPAD_PRESETS.find((preset) => preset.id === saved.screen_preset_id);
    if (byId) return byId;
  }
  if (saved.model) {
    const byLabel = IPAD_PRESETS.find((preset) => preset.label === saved.model);
    if (byLabel) return byLabel;
  }
  return IPAD_PRESETS[0];
}

function selectedPreset() {
  return IPAD_PRESETS.find((preset) => preset.id === elements.modelSelect.value) || IPAD_PRESETS[0];
}

function applySelectedPreset({ overwriteManual }) {
  const preset = selectedPreset();
  if (preset.id !== "manual" && overwriteManual) {
    elements.widthMm.value = preset.widthMm;
    elements.heightMm.value = preset.heightMm;
  }
  if (preset.id === "manual") {
    elements.status.textContent = "Manual size selected. Enter physical display width and height if known.";
  } else {
    elements.status.textContent = `${preset.label} selected. Physical display size auto-filled.`;
  }
}

function refreshPdfLink() {
  const width = Number.parseFloat(elements.widthMm.value) || 280.6;
  const height = Number.parseFloat(elements.heightMm.value) || 194.7;
  const params = new URLSearchParams({
    width_mm: String(width),
    height_mm: String(height),
    landscape: "true"
  });
  elements.pdfLink.href = "/api/target.pdf?" + params.toString();
}

function renderInstallHint() {
  if (isStandalone()) {
    elements.installHint.textContent = "";
    elements.installHint.hidden = true;
    return;
  }
  elements.installHint.hidden = false;
  if (isAppleTouchDevice()) {
    elements.installHint.textContent =
      "For no Safari bars, add this page to the Home Screen and open the saved Expresso Target icon. Show Board avoids the browser fullscreen close button.";
  } else {
    elements.installHint.textContent =
      "Show Board uses the full page instead of browser fullscreen, avoiding the fullscreen close overlay.";
  }
}

function isStandalone() {
  return window.navigator.standalone === true ||
    window.matchMedia("(display-mode: standalone)").matches ||
    window.matchMedia("(display-mode: fullscreen)").matches;
}

function isAppleTouchDevice() {
  return /iPad|iPhone|iPod/.test(navigator.userAgent) ||
    (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
}

async function saveMetadata() {
  const metadata = currentMetadata();
  localStorage.setItem("expressoTargetMetadata", JSON.stringify(metadata));
  const response = await fetch("/api/target-metadata", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(metadata)
  });
  if (response.ok) {
    elements.status.textContent = "Metadata saved.";
  } else {
    elements.status.textContent = "Could not save metadata.";
  }
}

function currentMetadata() {
  const preset = selectedPreset();
  return {
    model: preset.id === "manual" ? "unknown/manual" : preset.label,
    screen_preset_id: preset.id,
    screen_diagonal_in: preset.diagonalIn,
    screen_width_mm: elements.widthMm.value.trim(),
    screen_height_mm: elements.heightMm.value.trim(),
    screen_width_px: Math.round(window.screen.width * window.devicePixelRatio),
    screen_height_px: Math.round(window.screen.height * window.devicePixelRatio),
    device_pixel_ratio: window.devicePixelRatio || 1,
    user_agent: navigator.userAgent
  };
}

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
