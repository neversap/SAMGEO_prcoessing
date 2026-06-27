const serviceStatus = document.querySelector("#serviceStatus");
const imagePreview = document.querySelector("#imagePreview");
const maskPreview = document.querySelector("#maskPreview");
const previewStage = document.querySelector("#previewStage");
const dropZone = document.querySelector("#dropZone");
const toast = document.querySelector("#toast");
const preprocessForm = document.querySelector("#preprocessForm");
const datasetPathInput = document.querySelector("#datasetPathInput");
const tileSizeInput = document.querySelector("#tileSizeInput");
const overlapInput = document.querySelector("#overlapInput");
const trainRatioInput = document.querySelector("#trainRatioInput");
const valRatioInput = document.querySelector("#valRatioInput");
const splitStrategyInput = document.querySelector("#splitStrategyInput");
const maskModeInput = document.querySelector("#maskModeInput");
const boundaryWidthInput = document.querySelector("#boundaryWidthInput");
const backgroundKeepInput = document.querySelector("#backgroundKeepInput");
const maxIgnoreInput = document.querySelector("#maxIgnoreInput");
const blackThresholdInput = document.querySelector("#blackThresholdInput");
const dropEmptyInput = document.querySelector("#dropEmptyInput");
const testProcessInput = document.querySelector("#testProcessInput");
const preprocessStatus = document.querySelector("#preprocessStatus");
const preprocessStage = document.querySelector("#preprocessStage");
const preprocessPercent = document.querySelector("#preprocessPercent");
const preprocessBar = document.querySelector("#preprocessBar");
const preprocessMessage = document.querySelector("#preprocessMessage");
const preprocessJobId = document.querySelector("#preprocessJobId");
const preprocessOutputs = document.querySelector("#preprocessOutputs");
const preprocessLogs = document.querySelector("#preprocessLogs");
const startPreprocessButton = document.querySelector("#startPreprocessButton");
const cancelPreprocessButton = document.querySelector("#cancelPreprocessButton");
const clearProcessedButton = document.querySelector("#clearProcessedButton");
const qaSplitInput = document.querySelector("#qaSplitInput");
const qaModeInput = document.querySelector("#qaModeInput");
const qaLimitInput = document.querySelector("#qaLimitInput");
const loadQaButton = document.querySelector("#loadQaButton");
const prevQaButton = document.querySelector("#prevQaButton");
const randomQaButton = document.querySelector("#randomQaButton");
const nextQaButton = document.querySelector("#nextQaButton");
const qaOverlayButton = document.querySelector("#qaOverlayButton");
const qaImageButton = document.querySelector("#qaImageButton");
const qaMaskButton = document.querySelector("#qaMaskButton");
const qaCounter = document.querySelector("#qaCounter");
const qaMeta = document.querySelector("#qaMeta");

let activePreprocessJobId = window.localStorage.getItem("samGeoPreprocessJobId") || "";
let preprocessPollTimer = 0;
let qaSamples = [];
let qaIndex = 0;
let qaImageObjectUrl = "";
let qaMaskObjectUrl = "";
let qaOverlayObjectUrl = "";
let qaViewMode = "overlay";

async function loadHealth() {
  try {
    const response = await fetch("/health");
    const health = await response.json();
    serviceStatus.textContent = `${health.backend} on ${health.device}`;
  } catch (error) {
    serviceStatus.textContent = "Service unavailable";
  }
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.remove("hidden");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    toast.classList.add("hidden");
  }, 4200);
}

function base64ToBlob(base64, contentType) {
  const bytes = atob(base64);
  const chunks = [];
  for (let offset = 0; offset < bytes.length; offset += 8192) {
    const slice = bytes.slice(offset, offset + 8192);
    const numbers = new Array(slice.length);
    for (let index = 0; index < slice.length; index += 1) {
      numbers[index] = slice.charCodeAt(index);
    }
    chunks.push(new Uint8Array(numbers));
  }
  return new Blob(chunks, { type: contentType });
}

preprocessForm.addEventListener("submit", startPreprocessJob);
cancelPreprocessButton.addEventListener("click", cancelPreprocessJob);
clearProcessedButton.addEventListener("click", clearProcessedData);
loadQaButton.addEventListener("click", loadQaSamples);
prevQaButton.addEventListener("click", () => showQaSample(qaIndex - 1));
nextQaButton.addEventListener("click", () => showQaSample(qaIndex + 1));
randomQaButton.addEventListener("click", showRandomQaSample);
qaOverlayButton.addEventListener("click", () => setQaViewMode("overlay"));
qaImageButton.addEventListener("click", () => setQaViewMode("image"));
qaMaskButton.addEventListener("click", () => setQaViewMode("mask"));

loadHealth();
resumePreprocessJob();

async function startPreprocessJob(event) {
  event.preventDefault();
  const datasetDir = datasetPathInput.value.trim();
  if (!datasetDir) {
    showToast("Enter a dataset path.");
    return;
  }

  const trainRatio = Number(trainRatioInput.value || 0.8);
  const valRatio = Number(valRatioInput.value || 0.1);
  const testRatio = Math.max(0, 1 - trainRatio - valRatio);
  const payload = {
    dataset_dir: datasetDir,
    tile_size: Number(tileSizeInput.value || 512),
    overlap: Number(overlapInput.value || 64),
    train_ratio: trainRatio,
    val_ratio: valRatio,
    test_ratio: Number(testRatio.toFixed(4)),
    split_strategy: splitStrategyInput.value,
    drop_empty: dropEmptyInput.checked,
    test_process: testProcessInput.checked,
    mask_mode: maskModeInput.value,
    boundary_width_pixels: Number(boundaryWidthInput.value || 2),
    background_keep_ratio: Number(backgroundKeepInput.value || 0.2),
    max_ignore_ratio: Number(maxIgnoreInput.value || 0.5),
    black_pixel_threshold: Number(blackThresholdInput.value || 0),
  };

  startPreprocessButton.disabled = true;
  startPreprocessButton.textContent = "Starting";
  let started = false;
  try {
    const response = await fetch("/preprocess/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const job = await response.json();
    if (!response.ok) {
      throw new Error(job.detail || "Failed to start preprocessing.");
    }
    activePreprocessJobId = job.job_id;
    started = true;
    window.localStorage.setItem("samGeoPreprocessJobId", activePreprocessJobId);
    renderPreprocessJob(job);
    pollPreprocessJob();
    showToast("Preprocessing job started.");
  } catch (error) {
    showToast(error.message);
  } finally {
    if (!started) {
      startPreprocessButton.disabled = false;
    }
    startPreprocessButton.textContent = "Start preprocess";
  }
}

async function cancelPreprocessJob() {
  if (!activePreprocessJobId) {
    return;
  }
  cancelPreprocessButton.disabled = true;
  try {
    const response = await fetch(`/preprocess/jobs/${activePreprocessJobId}/cancel`, {
      method: "POST",
    });
    const job = await response.json();
    if (!response.ok) {
      throw new Error(job.detail || "Failed to cancel preprocessing.");
    }
    renderPreprocessJob(job);
    showToast("Cancellation requested.");
  } catch (error) {
    showToast(error.message);
  }
}

async function clearProcessedData() {
  const datasetDir = datasetPathInput.value.trim();
  if (!datasetDir) {
    showToast("Enter a dataset path first.");
    return;
  }
  const confirmed = window.confirm(
    "Clear all files under processed/ and metadata/ for this dataset?"
  );
  if (!confirmed) {
    return;
  }
  clearProcessedButton.disabled = true;
  clearProcessedButton.textContent = "Clearing";
  try {
    const response = await fetch("/preprocess/clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_dir: datasetDir }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Failed to clear processed data.");
    }
    clearQaViewer();
    preprocessOutputs.innerHTML = "";
    preprocessLogs.innerHTML = "";
    showToast(`Cleared ${payload.cleared.length} processed entries.`);
  } catch (error) {
    showToast(error.message);
  } finally {
    clearProcessedButton.disabled = false;
    clearProcessedButton.textContent = "Clear processed data";
  }
}

async function resumePreprocessJob() {
  if (!activePreprocessJobId) {
    return;
  }
  try {
    const response = await fetch(`/preprocess/jobs/${activePreprocessJobId}`);
    const job = await response.json();
    if (!response.ok) {
      window.localStorage.removeItem("samGeoPreprocessJobId");
      activePreprocessJobId = "";
      return;
    }
    renderPreprocessJob(job);
    if (!isTerminalJob(job.status)) {
      pollPreprocessJob();
    }
  } catch (error) {
    preprocessMessage.textContent = "Could not restore previous job.";
  }
}

function pollPreprocessJob() {
  window.clearTimeout(preprocessPollTimer);
  if (!activePreprocessJobId) {
    return;
  }
  preprocessPollTimer = window.setTimeout(async () => {
    try {
      const response = await fetch(`/preprocess/jobs/${activePreprocessJobId}`);
      const job = await response.json();
      if (!response.ok) {
        throw new Error(job.detail || "Failed to load preprocess status.");
      }
      renderPreprocessJob(job);
      if (!isTerminalJob(job.status)) {
        pollPreprocessJob();
      }
    } catch (error) {
      preprocessMessage.textContent = error.message;
      pollPreprocessJob();
    }
  }, 1200);
}

function renderPreprocessJob(job) {
  const percent = Math.round(Number(job.progress || 0) * 100);
  preprocessStatus.textContent = job.status;
  preprocessStage.textContent = job.stage || job.status;
  preprocessPercent.textContent = `${percent}%`;
  preprocessBar.style.width = `${percent}%`;
  preprocessMessage.textContent = `${job.current}/${job.total} ${job.message || ""}`.trim();
  preprocessJobId.textContent = job.job_id ? `job: ${job.job_id}` : "";
  cancelPreprocessButton.disabled = isTerminalJob(job.status);
  startPreprocessButton.disabled = !isTerminalJob(job.status) && job.status !== "idle";
  renderPreprocessOutputs(job.output_paths || {});
  renderPreprocessLogs(job.logs || [], job.error);
  if (isTerminalJob(job.status)) {
    window.clearTimeout(preprocessPollTimer);
    if (job.status === "completed") {
      showToast("Preprocessing completed.");
    }
  }
}

function renderPreprocessOutputs(outputs) {
  const entries = Object.entries(outputs);
  preprocessOutputs.innerHTML = "";
  entries.forEach(([key, value]) => {
    const row = document.createElement("div");
    const label = document.createElement("span");
    const path = document.createElement("code");
    label.textContent = key;
    path.textContent = value;
    row.append(label, path);
    preprocessOutputs.append(row);
  });
}

function renderPreprocessLogs(logs, error) {
  preprocessLogs.innerHTML = "";
  logs.slice(-8).forEach((line) => {
    const row = document.createElement("div");
    row.textContent = line;
    preprocessLogs.append(row);
  });
  if (error) {
    const row = document.createElement("div");
    row.className = "error-line";
    row.textContent = error;
    preprocessLogs.append(row);
  }
}

function isTerminalJob(status) {
  return status === "completed" || status === "failed" || status === "cancelled";
}

async function loadQaSamples() {
  const datasetDir = datasetPathInput.value.trim();
  if (!datasetDir) {
    showToast("Enter a dataset path first.");
    return;
  }
  const params = new URLSearchParams({
    dataset_dir: datasetDir,
    split: qaSplitInput.value,
    mode: qaModeInput.value,
    limit: qaLimitInput.value || "12",
    seed: String(Date.now() % 1000000),
  });
  loadQaButton.disabled = true;
  loadQaButton.textContent = "Loading";
  try {
    const response = await fetch(`/preprocess/preview?${params.toString()}`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Failed to load QA samples.");
    }
    qaSamples = payload.samples || [];
    qaIndex = 0;
    if (!qaSamples.length) {
      clearQaViewer();
      showToast("No samples matched this QA filter.");
      return;
    }
    showQaSample(0);
    showToast(`Loaded ${payload.count} QA samples.`);
  } catch (error) {
    showToast(error.message);
  } finally {
    loadQaButton.disabled = false;
    loadQaButton.textContent = "Load samples";
  }
}

function showQaSample(index) {
  if (!qaSamples.length) {
    clearQaViewer();
    return;
  }
  qaIndex = (index + qaSamples.length) % qaSamples.length;
  const sample = qaSamples[qaIndex];
  setQaImageSources(sample);
  renderQaMeta(sample);
  qaCounter.textContent = `${qaIndex + 1}/${qaSamples.length}`;
  prevQaButton.disabled = false;
  nextQaButton.disabled = false;
  randomQaButton.disabled = false;
  previewStage.classList.remove("hidden");
  dropZone.querySelector(".empty-state").classList.add("hidden");
  setQaViewMode(qaViewMode);
}

function showRandomQaSample() {
  if (!qaSamples.length) {
    return;
  }
  showQaSample(Math.floor(Math.random() * qaSamples.length));
}

function setQaImageSources(sample) {
  revokeQaUrls();
  qaImageObjectUrl = URL.createObjectURL(base64ToBlob(sample.image_png_base64, "image/png"));
  qaMaskObjectUrl = URL.createObjectURL(base64ToBlob(sample.mask_png_base64, "image/png"));
  qaOverlayObjectUrl = URL.createObjectURL(base64ToBlob(sample.overlay_png_base64, "image/png"));
  imagePreview.src = qaImageObjectUrl;
}

function setQaViewMode(mode) {
  qaViewMode = mode;
  qaOverlayButton.classList.toggle("active", mode === "overlay");
  qaImageButton.classList.toggle("active", mode === "image");
  qaMaskButton.classList.toggle("active", mode === "mask");
  previewStage.classList.toggle("image-only", mode === "image");
  previewStage.classList.toggle("mask-only", mode === "mask");
  if (mode === "overlay") {
    maskPreview.src = qaOverlayObjectUrl;
  } else if (mode === "mask") {
    maskPreview.src = qaMaskObjectUrl;
  } else {
    maskPreview.removeAttribute("src");
  }
}

function renderQaMeta(sample) {
  const rows = [
    ["patch", sample.patch_name],
    ["source", sample.source_tif],
    ["split", sample.split],
    ["x,y", `${sample.x}, ${sample.y}`],
    ["size", `${sample.width} x ${sample.height}`],
    ["cropland", Number(sample.cropland_ratio * 100).toFixed(2) + "%"],
    ["interior", Number((sample.interior_ratio || 0) * 100).toFixed(2) + "%"],
    ["boundary", Number((sample.boundary_ratio || 0) * 100).toFixed(2) + "%"],
    ["background", Number((sample.background_ratio || 0) * 100).toFixed(2) + "%"],
    ["ignore", Number(sample.ignore_ratio * 100).toFixed(2) + "%"],
    ["type", sample.patch_type || ""],
    ["image", sample.image_path],
    ["mask", sample.mask_path],
  ];
  qaMeta.innerHTML = "";
  rows.forEach(([labelText, valueText]) => {
    const row = document.createElement("div");
    const label = document.createElement("span");
    const value = document.createElement("code");
    label.textContent = labelText;
    value.textContent = valueText;
    row.append(label, value);
    qaMeta.append(row);
  });
}

function clearQaViewer() {
  revokeQaUrls();
  qaSamples = [];
  qaIndex = 0;
  qaCounter.textContent = "0/0";
  qaMeta.innerHTML = "";
  prevQaButton.disabled = true;
  nextQaButton.disabled = true;
  randomQaButton.disabled = true;
  imagePreview.removeAttribute("src");
  maskPreview.removeAttribute("src");
  previewStage.classList.add("hidden");
  dropZone.querySelector(".empty-state").classList.remove("hidden");
}

function revokeQaUrls() {
  if (qaImageObjectUrl) {
    URL.revokeObjectURL(qaImageObjectUrl);
    qaImageObjectUrl = "";
  }
  if (qaMaskObjectUrl) {
    URL.revokeObjectURL(qaMaskObjectUrl);
    qaMaskObjectUrl = "";
  }
  if (qaOverlayObjectUrl) {
    URL.revokeObjectURL(qaOverlayObjectUrl);
    qaOverlayObjectUrl = "";
  }
}
