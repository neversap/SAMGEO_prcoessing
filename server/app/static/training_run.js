const trainingJobForm = document.querySelector("#trainingJobForm");
const trainingConfigInput = document.querySelector("#trainingConfigInput");
const trainingDatasetPresetInput = document.querySelector("#trainingDatasetPresetInput");
const trainingLossPresetInput = document.querySelector("#trainingLossPresetInput");
const trainingInitCheckpointInput = document.querySelector("#trainingInitCheckpointInput");
const trainingStageInput = document.querySelector("#trainingStageInput");
const trainingEpochsInput = document.querySelector("#trainingEpochsInput");
const trainingBatchInput = document.querySelector("#trainingBatchInput");
const trainingMaxTrainInput = document.querySelector("#trainingMaxTrainInput");
const trainingMaxValInput = document.querySelector("#trainingMaxValInput");
const trainingJobStatus = document.querySelector("#trainingJobStatus");
const trainingJobStage = document.querySelector("#trainingJobStage");
const trainingJobPercent = document.querySelector("#trainingJobPercent");
const trainingJobBar = document.querySelector("#trainingJobBar");
const trainingJobMessage = document.querySelector("#trainingJobMessage");
const trainingJobId = document.querySelector("#trainingJobId");
const trainingOutputs = document.querySelector("#trainingOutputs");
const trainingLogs = document.querySelector("#trainingLogs");
const metricsStatus = document.querySelector("#metricsStatus");
const charts = {
  loss: document.querySelector("#lossChart"),
  miou: document.querySelector("#miouChart"),
  boundary: document.querySelector("#boundaryChart"),
  lr: document.querySelector("#lrChart"),
};
const startTrainingButton = document.querySelector("#startTrainingButton");
const cancelTrainingButton = document.querySelector("#cancelTrainingButton");
const toast = document.querySelector("#toast");
const trainingConfigByPreset = {
  ftw: {
    baseline: "/app/configs/pretrain/ftw_rgb_unet_effb3_pretrain_v1.yaml",
    prue_logcosh: "/app/configs/pretrain/ftw_rgb_unet_effb3_pretrain_prue_logcosh.yaml",
  },
  inhouse: {
    baseline: "/app/configs/finetune/inhouse_unet_effb3_finetune_v1.yaml",
    prue_logcosh: "/app/configs/finetune/inhouse_unet_effb3_finetune_prue_logcosh.yaml",
  },
};

let activeTrainingJobId = window.localStorage.getItem("samGeoTrainingJobId") || "";
let trainingPollTimer = 0;

function showToast(message) {
  toast.textContent = message;
  toast.classList.remove("hidden");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    toast.classList.add("hidden");
  }, 4200);
}

async function startTrainingJob(event) {
  event.preventDefault();
  const payload = {
    config_path: trainingConfigInput.value.trim(),
    stage: trainingStageInput.value,
    epochs: Number(trainingEpochsInput.value || 0),
    batch_size: Number(trainingBatchInput.value || 0),
    max_train_samples: Number(trainingMaxTrainInput.value || 0),
    max_val_samples: Number(trainingMaxValInput.value || 0),
    init_checkpoint: trainingInitCheckpointInput.value.trim(),
  };
  startTrainingButton.disabled = true;
  startTrainingButton.textContent = "Starting";
  try {
    const response = await fetch("/training/jobs", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    const job = await response.json();
    if (!response.ok) {
      throw new Error(job.detail || "Failed to start training.");
    }
    activeTrainingJobId = job.job_id;
    window.localStorage.setItem("samGeoTrainingJobId", activeTrainingJobId);
    renderTrainingJob(job);
    loadTrainingMetrics();
    pollTrainingJob();
    showToast("Training job started.");
  } catch (error) {
    showToast(error.message);
    startTrainingButton.disabled = false;
  } finally {
    startTrainingButton.textContent = "Start training";
  }
}

async function cancelTrainingJob() {
  if (!activeTrainingJobId) {
    return;
  }
  cancelTrainingButton.disabled = true;
  try {
    const response = await fetch(`/training/jobs/${activeTrainingJobId}/cancel`, {
      method: "POST",
    });
    const job = await response.json();
    if (!response.ok) {
      throw new Error(job.detail || "Failed to cancel training.");
    }
    renderTrainingJob(job);
    loadTrainingMetrics();
    showToast("Training cancellation requested.");
  } catch (error) {
    showToast(error.message);
  }
}

async function resumeTrainingJob() {
  if (!activeTrainingJobId) {
    return;
  }
  try {
    const response = await fetch(`/training/jobs/${activeTrainingJobId}`);
    const job = await response.json();
    if (!response.ok) {
      window.localStorage.removeItem("samGeoTrainingJobId");
      activeTrainingJobId = "";
      return;
    }
    renderTrainingJob(job);
    if (!isTerminalJob(job.status)) {
      pollTrainingJob();
    }
  } catch (error) {
    trainingJobMessage.textContent = "Could not restore previous training job.";
  }
}

function pollTrainingJob() {
  window.clearTimeout(trainingPollTimer);
  if (!activeTrainingJobId) {
    return;
  }
  trainingPollTimer = window.setTimeout(async () => {
    try {
      const response = await fetch(`/training/jobs/${activeTrainingJobId}`);
      const job = await response.json();
      if (!response.ok) {
        throw new Error(job.detail || "Failed to load training status.");
      }
      renderTrainingJob(job);
      loadTrainingMetrics();
      if (!isTerminalJob(job.status)) {
        pollTrainingJob();
      }
    } catch (error) {
      trainingJobMessage.textContent = error.message;
      pollTrainingJob();
    }
  }, 1800);
}

function renderTrainingJob(job) {
  const percent = Math.round(Number(job.progress || 0) * 100);
  trainingJobStatus.textContent = job.status;
  trainingJobStage.textContent = job.stage || job.status;
  trainingJobPercent.textContent = `${percent}%`;
  trainingJobBar.style.width = `${percent}%`;
  trainingJobMessage.textContent = `${job.current}/${job.total} ${job.message || ""}`.trim();
  trainingJobId.textContent = job.job_id ? `job: ${job.job_id}` : "";
  cancelTrainingButton.disabled = isTerminalJob(job.status);
  startTrainingButton.disabled = !isTerminalJob(job.status) && job.status !== "idle";
  renderOutputs(job.output_paths || {});
  renderLogs(job.logs || [], job.error);
  if (isTerminalJob(job.status)) {
    window.clearTimeout(trainingPollTimer);
    loadTrainingMetrics();
  }
}

function renderOutputs(outputs) {
  trainingOutputs.innerHTML = "";
  Object.entries(outputs).forEach(([key, value]) => {
    const row = document.createElement("div");
    const label = document.createElement("span");
    const path = document.createElement("code");
    label.textContent = key;
    path.textContent = value;
    row.append(label, path);
    trainingOutputs.append(row);
  });
}

function renderLogs(logs, error) {
  trainingLogs.innerHTML = "";
  logs.slice(-16).forEach((line) => {
    const row = document.createElement("div");
    row.textContent = line;
    trainingLogs.append(row);
  });
  if (error) {
    const row = document.createElement("div");
    row.className = "error-line";
    row.textContent = error;
    trainingLogs.append(row);
  }
}

function isTerminalJob(status) {
  return status === "completed" || status === "failed" || status === "cancelled";
}

async function loadTrainingMetrics() {
  if (!activeTrainingJobId) {
    return;
  }
  try {
    const response = await fetch(`/training/jobs/${activeTrainingJobId}/metrics`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Failed to load metrics.");
    }
    renderDashboard(payload.metrics || []);
    metricsStatus.textContent = payload.metrics.length
      ? `${payload.metrics.length} epochs`
      : "waiting";
  } catch (error) {
    metricsStatus.textContent = "unavailable";
  }
}

function renderDashboard(metrics) {
  drawChart(charts.loss, {
    title: "Loss",
    xLabel: "epoch",
    series: [
      { name: "train_loss", color: "#2563eb", values: metrics.map((row) => point(row, "train_loss")) },
      { name: "val_loss", color: "#d97706", values: metrics.map((row) => point(row, "val_loss")) },
    ],
  });
  drawChart(charts.miou, {
    title: "Validation mIoU",
    xLabel: "epoch",
    series: [
      { name: "val_miou", color: "#0f766e", values: metrics.map((row) => point(row, "val_miou")) },
    ],
  });
  drawChart(charts.boundary, {
    title: "Boundary F1",
    xLabel: "epoch",
    series: [
      { name: "val_boundary_f1", color: "#b42318", values: metrics.map((row) => point(row, "val_boundary_f1")) },
    ],
  });
  drawChart(charts.lr, {
    title: "Learning Rate",
    xLabel: "epoch",
    series: [
      { name: "lr", color: "#7c3aed", values: metrics.map((row) => point(row, "lr")) },
    ],
  });
}

function point(row, key) {
  return {
    x: Number(row.epoch || 0),
    y: Number(row[key] || 0),
  };
}

function drawChart(canvas, config) {
  if (!canvas) {
    return;
  }
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);

  const plot = { left: 62, right: width - 22, top: 34, bottom: height - 48 };
  const allPoints = config.series.flatMap((item) => item.values).filter((item) => Number.isFinite(item.x) && Number.isFinite(item.y));
  drawAxes(context, plot, width, height, config.title);
  if (!allPoints.length) {
    context.fillStyle = "#657084";
    context.font = "14px system-ui";
    context.fillText("Waiting for epoch metrics...", plot.left + 14, plot.top + 28);
    return;
  }

  const xMin = Math.min(...allPoints.map((item) => item.x));
  const xMax = Math.max(...allPoints.map((item) => item.x));
  let yMin = Math.min(...allPoints.map((item) => item.y));
  let yMax = Math.max(...allPoints.map((item) => item.y));
  if (yMax <= yMin) {
    yMax = yMin + 1;
  }
  const yPad = (yMax - yMin) * 0.08;
  yMin = Math.max(0, yMin - yPad);
  yMax += yPad;
  drawGrid(context, plot, xMin, xMax, yMin, yMax);

  config.series.forEach((series) => {
    const values = series.values.filter((item) => Number.isFinite(item.x) && Number.isFinite(item.y));
    drawLine(context, plot, values, xMin, xMax, yMin, yMax, series.color);
  });
  drawLegend(context, config.series, plot);
}

function drawAxes(context, plot, width, height, title) {
  context.strokeStyle = "#253246";
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(plot.left, plot.top);
  context.lineTo(plot.left, plot.bottom);
  context.lineTo(plot.right, plot.bottom);
  context.stroke();
  context.fillStyle = "#18202f";
  context.font = "700 15px system-ui";
  context.fillText(title, plot.left, 22);
}

function drawGrid(context, plot, xMin, xMax, yMin, yMax) {
  context.strokeStyle = "#e0e6ef";
  context.fillStyle = "#657084";
  context.font = "11px system-ui";
  for (let step = 0; step <= 4; step += 1) {
    const t = step / 4;
    const y = plot.bottom - t * (plot.bottom - plot.top);
    const value = yMin + t * (yMax - yMin);
    context.beginPath();
    context.moveTo(plot.left, y);
    context.lineTo(plot.right, y);
    context.stroke();
    context.fillText(formatMetric(value), 8, y + 4);
  }
  const xSteps = Math.max(1, Math.min(5, Math.ceil(xMax - xMin + 1)));
  for (let step = 0; step <= xSteps; step += 1) {
    const t = step / xSteps;
    const x = plot.left + t * (plot.right - plot.left);
    const value = xMin + t * Math.max(xMax - xMin, 1);
    context.beginPath();
    context.moveTo(x, plot.top);
    context.lineTo(x, plot.bottom);
    context.stroke();
    context.fillText(String(Math.round(value)), x - 5, plot.bottom + 20);
  }
}

function drawLine(context, plot, values, xMin, xMax, yMin, yMax, color) {
  if (!values.length) {
    return;
  }
  context.strokeStyle = color;
  context.fillStyle = color;
  context.lineWidth = 2;
  context.beginPath();
  values.forEach((pointValue, index) => {
    const x = scale(pointValue.x, xMin, xMax, plot.left, plot.right);
    const y = scale(pointValue.y, yMin, yMax, plot.bottom, plot.top);
    if (index === 0) {
      context.moveTo(x, y);
    } else {
      context.lineTo(x, y);
    }
  });
  context.stroke();
  values.forEach((pointValue) => {
    const x = scale(pointValue.x, xMin, xMax, plot.left, plot.right);
    const y = scale(pointValue.y, yMin, yMax, plot.bottom, plot.top);
    context.beginPath();
    context.arc(x, y, 3, 0, Math.PI * 2);
    context.fill();
  });
}

function drawLegend(context, series, plot) {
  let x = plot.left;
  const y = plot.bottom + 38;
  series.forEach((item) => {
    context.fillStyle = item.color;
    context.fillRect(x, y - 9, 18, 3);
    context.fillStyle = "#344256";
    context.font = "12px system-ui";
    context.fillText(item.name, x + 24, y - 4);
    x += 130;
  });
}

function scale(value, inputMin, inputMax, outputMin, outputMax) {
  if (inputMax <= inputMin) {
    return (outputMin + outputMax) / 2;
  }
  return outputMin + ((value - inputMin) / (inputMax - inputMin)) * (outputMax - outputMin);
}

function formatMetric(value) {
  if (Math.abs(value) < 0.001 && value !== 0) {
    return value.toExponential(1);
  }
  if (Math.abs(value) < 1) {
    return value.toFixed(3);
  }
  return value.toFixed(2);
}

trainingJobForm.addEventListener("submit", startTrainingJob);
cancelTrainingButton.addEventListener("click", cancelTrainingJob);
trainingDatasetPresetInput.addEventListener("change", updateTrainingConfigPreset);
trainingLossPresetInput.addEventListener("change", updateTrainingConfigPreset);
resumeTrainingJob();
renderDashboard([]);

function updateTrainingConfigPreset() {
  const dataset = trainingDatasetPresetInput.value;
  const loss = trainingLossPresetInput.value;
  trainingConfigInput.value = trainingConfigByPreset[dataset]?.[loss] || trainingConfigInput.value;
}
