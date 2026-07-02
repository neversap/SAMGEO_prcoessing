const serviceStatus = document.querySelector("#serviceStatus");
const trainingStatus = document.querySelector("#trainingStatus");
const trainingForm = document.querySelector("#trainingForm");
const sourceInput = document.querySelector("#sourceInput");
const rootPathInput = document.querySelector("#rootPathInput");
const ftwFields = document.querySelector("#ftwFields");
const countryInput = document.querySelector("#countryInput");
const windowInput = document.querySelector("#windowInput");
const maskTypeInput = document.querySelector("#maskTypeInput");
const metadataDirInput = document.querySelector("#metadataDirInput");
const trainRatioInput = document.querySelector("#trainRatioInput");
const valRatioInput = document.querySelector("#valRatioInput");
const maxIndexSamplesInput = document.querySelector("#maxIndexSamplesInput");
const buildIndexButton = document.querySelector("#buildIndexButton");
const splitInput = document.querySelector("#splitInput");
const modeInput = document.querySelector("#modeInput");
const limitInput = document.querySelector("#limitInput");
const seedInput = document.querySelector("#seedInput");
const hflipInput = document.querySelector("#hflipInput");
const vflipInput = document.querySelector("#vflipInput");
const rotateInput = document.querySelector("#rotateInput");
const scaleInput = document.querySelector("#scaleInput");
const noiseInput = document.querySelector("#noiseInput");
const brightnessInput = document.querySelector("#brightnessInput");
const contrastInput = document.querySelector("#contrastInput");
const loadAugmentButton = document.querySelector("#loadAugmentButton");
const indexOutputs = document.querySelector("#indexOutputs");
const trainingStats = document.querySelector("#trainingStats");
const trainingGrid = document.querySelector("#trainingGrid");
const toast = document.querySelector("#toast");

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

function updateSourceFields() {
  const isFtw = sourceInput.value === "ftw";
  ftwFields.classList.toggle("hidden", !isFtw);
  rootPathInput.placeholder = isFtw
    ? "/home/nvme1/datasets/ftw"
    : "/home/nvme1/datasets";
}

function defaultMetadataDir() {
  const root = rootPathInput.value.trim();
  if (!root) {
    return "";
  }
  return `${root.replace(/[\\/]+$/, "")}/metadata`;
}

async function buildFtwIndex() {
  const ftwRoot = rootPathInput.value.trim();
  if (!ftwRoot) {
    showToast("Enter the FTW root path first.");
    return;
  }
  const trainRatio = Number(trainRatioInput.value || 0.8);
  const valRatio = Number(valRatioInput.value || 0.1);
  const testRatio = Math.max(0, 1 - trainRatio - valRatio);
  const payload = {
    ftw_root: ftwRoot,
    metadata_dir: metadataDirInput.value.trim() || defaultMetadataDir(),
    country: countryInput.value.trim() || "all",
    window: windowInput.value,
    mask_type: maskTypeInput.value,
    train_ratio: trainRatio,
    val_ratio: valRatio,
    test_ratio: Number(testRatio.toFixed(4)),
    seed: Number(seedInput.value || 42),
    max_samples: Number(maxIndexSamplesInput.value || 0),
  };

  buildIndexButton.disabled = true;
  buildIndexButton.textContent = "Building";
  trainingStatus.textContent = "indexing";
  try {
    const response = await fetch("/training/index/ftw", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.detail || "Failed to build index.");
    }
    renderIndexOutputs(result);
    renderStats({
      count: result.count,
      buckets: result.buckets,
      splits: result.splits,
    });
    trainingStatus.textContent = "index ready";
    showToast(`FTW index built with ${result.count} samples.`);
  } catch (error) {
    trainingStatus.textContent = "failed";
    showToast(error.message);
  } finally {
    buildIndexButton.disabled = false;
    buildIndexButton.textContent = "Build FTW index";
  }
}

async function loadAugmentationPreview(event) {
  event.preventDefault();
  const root = rootPathInput.value.trim();
  if (!root) {
    showToast("Enter a root path first.");
    return;
  }
  const params = new URLSearchParams({
    source: sourceInput.value,
    root_path: root,
    country: countryInput.value.trim() || "all",
    window: windowInput.value === "both" ? "window_a" : windowInput.value,
    mask_type: maskTypeInput.value,
    split: splitInput.value,
    mode: modeInput.value,
    limit: limitInput.value || "6",
    seed: seedInput.value || "42",
    hflip: hflipInput.checked ? "true" : "false",
    vflip: vflipInput.checked ? "true" : "false",
    rotate90: rotateInput.checked ? "true" : "false",
    scale_jitter: scaleInput.value || "0",
    brightness: brightnessInput.value || "0",
    contrast: contrastInput.value || "0",
    noise: noiseInput.value || "0",
  });

  loadAugmentButton.disabled = true;
  loadAugmentButton.textContent = "Loading";
  trainingStatus.textContent = "previewing";
  try {
    const response = await fetch(`/training/augment-preview?${params.toString()}`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Failed to load augmentation preview.");
    }
    renderStats(payload.stats);
    renderSamples(payload.samples || []);
    trainingStatus.textContent = "ready";
    showToast(`Loaded ${payload.count} augmented samples.`);
  } catch (error) {
    trainingStatus.textContent = "failed";
    showToast(error.message);
  } finally {
    loadAugmentButton.disabled = false;
    loadAugmentButton.textContent = "Preview augmentation";
  }
}

function renderIndexOutputs(result) {
  indexOutputs.innerHTML = "";
  [
    ["index", result.index_path],
    ["stats", result.stats_path],
  ].forEach(([labelText, valueText]) => {
    const row = document.createElement("div");
    const label = document.createElement("span");
    const value = document.createElement("code");
    label.textContent = labelText;
    value.textContent = valueText;
    row.append(label, value);
    indexOutputs.append(row);
  });
}

function renderStats(stats) {
  trainingStats.innerHTML = "";
  const rows = [
    ["count", String(stats.count || 0)],
    ["buckets", formatDict(stats.buckets || {})],
    ["splits", formatDict(stats.splits || {})],
    ["countries", formatDict(stats.countries || {})],
  ];
  rows.forEach(([labelText, valueText]) => {
    const row = document.createElement("div");
    const label = document.createElement("span");
    const value = document.createElement("code");
    label.textContent = labelText;
    value.textContent = valueText || "-";
    row.append(label, value);
    trainingStats.append(row);
  });
}

function renderSamples(samples) {
  trainingGrid.innerHTML = "";
  if (!samples.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.innerHTML = "<strong>No samples</strong><span>Try another split or mode.</span>";
    trainingGrid.append(empty);
    return;
  }
  samples.forEach((sample) => {
    const item = document.createElement("article");
    item.className = "training-sample";

    const header = document.createElement("div");
    header.className = "training-sample-header";
    const title = document.createElement("strong");
    title.textContent = sample.patch_name || sample.sample_id;
    const bucket = document.createElement("span");
    bucket.textContent = sample.bucket || "-";
    header.append(title, bucket);

    const images = document.createElement("div");
    images.className = "training-pair";
    images.append(
      makeFigure("Original", sample.image_png_base64, sample.overlay_png_base64),
      makeFigure("Augmented", sample.augmented_image_png_base64, sample.augmented_overlay_png_base64)
    );

    const meta = document.createElement("div");
    meta.className = "training-meta";
    meta.textContent =
      `fg ${(sample.fg_ratio * 100).toFixed(2)}% | ` +
      `boundary ${(sample.boundary_ratio * 100).toFixed(2)}% | ` +
      `ignore ${(sample.ignore_ratio * 100).toFixed(2)}% | ` +
      `${sample.augmentation}`;

    item.append(header, images, meta);
    trainingGrid.append(item);
  });
}

function makeFigure(labelText, imageBase64, overlayBase64) {
  const figure = document.createElement("figure");
  const stack = document.createElement("div");
  stack.className = "training-image-stack";
  const image = document.createElement("img");
  image.src = `data:image/png;base64,${imageBase64}`;
  image.alt = `${labelText} image`;
  const overlay = document.createElement("img");
  overlay.src = `data:image/png;base64,${overlayBase64}`;
  overlay.alt = `${labelText} overlay`;
  stack.append(image, overlay);
  const caption = document.createElement("figcaption");
  caption.textContent = labelText;
  figure.append(stack, caption);
  return figure;
}

function formatDict(value) {
  return Object.entries(value)
    .map(([key, count]) => `${key}: ${count}`)
    .join(", ");
}

sourceInput.addEventListener("change", updateSourceFields);
rootPathInput.addEventListener("input", () => {
  if (!metadataDirInput.value.trim()) {
    metadataDirInput.placeholder = defaultMetadataDir() || "/home/nvme1/datasets/ftw/metadata";
  }
});
buildIndexButton.addEventListener("click", buildFtwIndex);
trainingForm.addEventListener("submit", loadAugmentationPreview);

loadHealth();
updateSourceFields();
