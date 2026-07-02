const inferenceJobForm = document.querySelector("#inferenceJobForm");
const checkpointInput = document.querySelector("#checkpointInput");
const configInput = document.querySelector("#configInput");
const ftwMetadataInput = document.querySelector("#ftwMetadataInput");
const inhouseDatasetInput = document.querySelector("#inhouseDatasetInput");
const ftwCountInput = document.querySelector("#ftwCountInput");
const inhouseCountInput = document.querySelector("#inhouseCountInput");
const seedInput = document.querySelector("#seedInput");
const inferenceJobStatus = document.querySelector("#inferenceJobStatus");
const inferenceJobStage = document.querySelector("#inferenceJobStage");
const inferenceJobPercent = document.querySelector("#inferenceJobPercent");
const inferenceJobBar = document.querySelector("#inferenceJobBar");
const inferenceJobMessage = document.querySelector("#inferenceJobMessage");
const inferenceJobId = document.querySelector("#inferenceJobId");
const inferenceOutputs = document.querySelector("#inferenceOutputs");
const inferenceLogs = document.querySelector("#inferenceLogs");
const summaryStatus = document.querySelector("#summaryStatus");
const inferenceGrid = document.querySelector("#inferenceGrid");
const startInferenceButton = document.querySelector("#startInferenceButton");
const cancelInferenceButton = document.querySelector("#cancelInferenceButton");
const toast = document.querySelector("#toast");
const tabButtons = Array.from(document.querySelectorAll(".inference-tabs .toggle"));
const imageDialog = document.querySelector("#imageDialog");
const imageDialogImage = document.querySelector("#imageDialogImage");
const imageDialogTitle = document.querySelector("#imageDialogTitle");
const imageDialogClose = document.querySelector("#imageDialogClose");

let activeInferenceJobId = window.localStorage.getItem("samGeoInferenceJobId") || "";
let inferencePollTimer = 0;
let currentSamples = [];
let currentSource = "all";

function showToast(message) {
  toast.textContent = message;
  toast.classList.remove("hidden");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    toast.classList.add("hidden");
  }, 4200);
}

async function startInferenceJob(event) {
  event.preventDefault();
  const payload = {
    checkpoint_path: checkpointInput.value.trim(),
    config_path: configInput.value.trim(),
    ftw_metadata_csv: ftwMetadataInput.value.trim(),
    inhouse_dataset_dir: inhouseDatasetInput.value.trim(),
    ftw_count: Number(ftwCountInput.value || 0),
    inhouse_count: Number(inhouseCountInput.value || 0),
    seed: Number(seedInput.value || 42),
  };
  startInferenceButton.disabled = true;
  startInferenceButton.textContent = "Starting";
  try {
    const response = await fetch("/inference/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const job = await response.json();
    if (!response.ok) {
      throw new Error(job.detail || "Failed to start inference.");
    }
    activeInferenceJobId = job.job_id;
    window.localStorage.setItem("samGeoInferenceJobId", activeInferenceJobId);
    currentSamples = [];
    renderInferenceGrid();
    renderInferenceJob(job);
    pollInferenceJob();
    showToast("Inference job started.");
  } catch (error) {
    showToast(error.message);
    startInferenceButton.disabled = false;
  } finally {
    startInferenceButton.textContent = "Start inference";
  }
}

async function cancelInferenceJob() {
  if (!activeInferenceJobId) {
    return;
  }
  cancelInferenceButton.disabled = true;
  try {
    const response = await fetch(`/inference/jobs/${activeInferenceJobId}/cancel`, {
      method: "POST",
    });
    const job = await response.json();
    if (!response.ok) {
      throw new Error(job.detail || "Failed to cancel inference.");
    }
    renderInferenceJob(job);
    showToast("Inference cancellation requested.");
  } catch (error) {
    showToast(error.message);
  }
}

async function resumeInferenceJob() {
  if (!activeInferenceJobId) {
    return;
  }
  try {
    const response = await fetch(`/inference/jobs/${activeInferenceJobId}`);
    const job = await response.json();
    if (!response.ok) {
      window.localStorage.removeItem("samGeoInferenceJobId");
      activeInferenceJobId = "";
      return;
    }
    renderInferenceJob(job);
    loadInferenceSummary();
    if (!isTerminalJob(job.status)) {
      pollInferenceJob();
    }
  } catch (error) {
    inferenceJobMessage.textContent = "Could not restore previous inference job.";
  }
}

function pollInferenceJob() {
  window.clearTimeout(inferencePollTimer);
  if (!activeInferenceJobId) {
    return;
  }
  inferencePollTimer = window.setTimeout(async () => {
    try {
      const response = await fetch(`/inference/jobs/${activeInferenceJobId}`);
      const job = await response.json();
      if (!response.ok) {
        throw new Error(job.detail || "Failed to load inference status.");
      }
      renderInferenceJob(job);
      if (job.status === "completed") {
        loadInferenceSummary();
      }
      if (!isTerminalJob(job.status)) {
        pollInferenceJob();
      }
    } catch (error) {
      inferenceJobMessage.textContent = error.message;
      pollInferenceJob();
    }
  }, 1800);
}

function renderInferenceJob(job) {
  const percent = Math.round(Number(job.progress || 0) * 100);
  inferenceJobStatus.textContent = job.status;
  inferenceJobStage.textContent = job.status;
  inferenceJobPercent.textContent = `${percent}%`;
  inferenceJobBar.style.width = `${percent}%`;
  inferenceJobMessage.textContent = `${job.current}/${job.total} ${job.message || ""}`.trim();
  inferenceJobId.textContent = job.job_id ? `job: ${job.job_id}` : "";
  cancelInferenceButton.disabled = isTerminalJob(job.status);
  startInferenceButton.disabled = !isTerminalJob(job.status) && job.status !== "idle";
  renderOutputs(job.output_paths || {});
  renderLogs(job.logs || [], job.error);
  if (isTerminalJob(job.status)) {
    window.clearTimeout(inferencePollTimer);
  }
}

function renderOutputs(outputs) {
  inferenceOutputs.innerHTML = "";
  Object.entries(outputs).forEach(([key, value]) => {
    const row = document.createElement("div");
    const label = document.createElement("span");
    const path = document.createElement("code");
    label.textContent = key;
    path.textContent = value;
    row.append(label, path);
    inferenceOutputs.append(row);
  });
}

function renderLogs(logs, error) {
  inferenceLogs.innerHTML = "";
  logs.slice(-18).forEach((line) => {
    const row = document.createElement("div");
    row.textContent = line;
    inferenceLogs.append(row);
  });
  if (error) {
    const row = document.createElement("div");
    row.className = "error-line";
    row.textContent = error;
    inferenceLogs.append(row);
  }
}

async function loadInferenceSummary() {
  if (!activeInferenceJobId) {
    return;
  }
  try {
    const response = await fetch(`/inference/jobs/${activeInferenceJobId}/summary`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Summary is not ready.");
    }
    currentSamples = payload.samples || [];
    summaryStatus.textContent = `${currentSamples.length} samples`;
    renderInferenceGrid();
  } catch (error) {
    summaryStatus.textContent = "waiting";
  }
}

function renderInferenceGrid() {
  inferenceGrid.innerHTML = "";
  const samples = currentSamples.filter((sample) => currentSource === "all" || sample.source === currentSource);
  if (!samples.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state inference-empty";
    empty.innerHTML = "<strong>No samples</strong><span>Run inference to inspect checkpoint outputs.</span>";
    inferenceGrid.append(empty);
    return;
  }
  samples.forEach((sample) => {
    inferenceGrid.append(renderSample(sample));
  });
}

function renderSample(sample) {
  const item = document.createElement("article");
  item.className = "inference-sample";

  const header = document.createElement("div");
  header.className = "training-sample-header";
  const title = document.createElement("strong");
  title.textContent = `${sample.id}. ${sample.sample_id}`;
  const source = document.createElement("span");
  source.textContent = sample.source;
  header.append(title, source);

  const images = document.createElement("div");
  images.className = "inference-image-grid";
  images.append(
    renderFigure(sample.image_url, "Image"),
    renderFigure(sample.gt_url, "GT mask"),
    renderFigure(sample.pred_url, "Prediction"),
    renderFigure(sample.overlay_url, "Overlay"),
  );

  const meta = document.createElement("div");
  meta.className = "training-meta";
  meta.textContent = [
    `split=${sample.split || "-"}`,
    `crop=${formatPercent(sample.cropland_ratio)}`,
    `mIoU=${formatMetric(sample.metrics.miou)}`,
    `Boundary F1=${formatMetric(sample.metrics.boundary_f1)}`,
    `Pixel Acc=${formatMetric(sample.metrics.pixel_accuracy)}`,
  ].join("  ");

  item.append(header, images, meta);
  return item;
}

function renderFigure(src, caption) {
  const figure = document.createElement("figure");
  const stack = document.createElement("div");
  stack.className = "training-image-stack inference-zoom-target";
  stack.tabIndex = 0;
  stack.role = "button";
  stack.setAttribute("aria-label", `Open ${caption}`);
  const image = document.createElement("img");
  image.src = src;
  image.loading = "lazy";
  image.alt = caption;
  const label = document.createElement("figcaption");
  label.textContent = caption;
  stack.append(image);
  figure.append(stack, label);
  stack.addEventListener("click", () => openImageDialog(src, caption));
  stack.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openImageDialog(src, caption);
    }
  });
  return figure;
}

function openImageDialog(src, title) {
  imageDialogImage.src = src;
  imageDialogImage.alt = title;
  imageDialogTitle.textContent = title;
  if (typeof imageDialog.showModal === "function") {
    imageDialog.showModal();
  } else {
    imageDialog.setAttribute("open", "open");
  }
}

function closeImageDialog() {
  imageDialog.close();
  imageDialogImage.removeAttribute("src");
}

function isTerminalJob(status) {
  return status === "completed" || status === "failed" || status === "cancelled";
}

function formatMetric(value) {
  const numeric = Number(value || 0);
  return numeric.toFixed(3);
}

function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

tabButtons.forEach((button) => {
  button.addEventListener("click", () => {
    currentSource = button.dataset.source || "all";
    tabButtons.forEach((item) => item.classList.toggle("active", item === button));
    renderInferenceGrid();
  });
});

imageDialogClose.addEventListener("click", closeImageDialog);
imageDialog.addEventListener("click", (event) => {
  if (event.target === imageDialog) {
    closeImageDialog();
  }
});
inferenceJobForm.addEventListener("submit", startInferenceJob);
cancelInferenceButton.addEventListener("click", cancelInferenceJob);
resumeInferenceJob();
renderInferenceGrid();
