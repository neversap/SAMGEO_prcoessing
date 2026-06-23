const form = document.querySelector("#segmentForm");
const imageInput = document.querySelector("#imageInput");
const promptInput = document.querySelector("#promptInput");
const thresholdInput = document.querySelector("#thresholdInput");
const thresholdValue = document.querySelector("#thresholdValue");
const postprocessInput = document.querySelector("#postprocessInput");
const boxInput = document.querySelector("#boxInput");
const pointsInput = document.querySelector("#pointsInput");
const runButton = document.querySelector("#runButton");
const serviceStatus = document.querySelector("#serviceStatus");
const imagePreview = document.querySelector("#imagePreview");
const maskPreview = document.querySelector("#maskPreview");
const previewStage = document.querySelector("#previewStage");
const dropZone = document.querySelector("#dropZone");
const toast = document.querySelector("#toast");
const backendValue = document.querySelector("#backendValue");
const objectCountValue = document.querySelector("#objectCountValue");
const scoreValue = document.querySelector("#scoreValue");
const instancesList = document.querySelector("#instancesList");
const downloadMask = document.querySelector("#downloadMask");
const viewButtons = {
  overlay: document.querySelector("#showOverlay"),
  image: document.querySelector("#showImage"),
  semantic: document.querySelector("#showSemantic"),
  instances: document.querySelector("#showInstances"),
};

let imageObjectUrl = "";
let resultObjectUrl = "";
let semanticObjectUrl = "";

async function loadHealth() {
  try {
    const response = await fetch("/health");
    const health = await response.json();
    serviceStatus.textContent = `${health.backend} on ${health.device}`;
    backendValue.textContent = health.backend;
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

function setViewMode(mode) {
  previewStage.classList.toggle("image-only", mode === "image");
  previewStage.classList.toggle("mask-only", mode === "instances" || mode === "semantic");
  Object.entries(viewButtons).forEach(([key, button]) => {
    button.classList.toggle("active", key === mode);
  });
  if (mode === "semantic" && semanticObjectUrl) {
    maskPreview.src = semanticObjectUrl;
  }
  if ((mode === "overlay" || mode === "instances") && resultObjectUrl) {
    maskPreview.src = resultObjectUrl;
  }
}

function resetResult() {
  if (resultObjectUrl) {
    URL.revokeObjectURL(resultObjectUrl);
    resultObjectUrl = "";
  }
  if (semanticObjectUrl) {
    URL.revokeObjectURL(semanticObjectUrl);
    semanticObjectUrl = "";
  }
  maskPreview.removeAttribute("src");
  downloadMask.removeAttribute("href");
  downloadMask.classList.add("disabled");
  objectCountValue.textContent = "-";
  scoreValue.textContent = "-";
  instancesList.innerHTML = "";
}

function previewImage(file) {
  if (imageObjectUrl) {
    URL.revokeObjectURL(imageObjectUrl);
  }
  imageObjectUrl = URL.createObjectURL(file);
  imagePreview.src = imageObjectUrl;
  previewStage.classList.remove("hidden");
  dropZone.querySelector(".empty-state").classList.add("hidden");
  resetResult();
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

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (file) {
    previewImage(file);
  }
});

thresholdInput.addEventListener("input", () => {
  thresholdValue.textContent = Number(thresholdInput.value).toFixed(2);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = imageInput.files[0];
  if (!file) {
    showToast("Select an image first.");
    return;
  }

  const body = new FormData();
  body.append("image", file);
  body.append("prompt", promptInput.value.trim() || "object");
  body.append("threshold", thresholdInput.value);
  body.append("postprocess", postprocessInput.value);
  if (boxInput.value.trim()) {
    body.append("box", boxInput.value.trim());
  }
  if (pointsInput.value.trim()) {
    body.append("points", pointsInput.value.trim());
  }

  runButton.disabled = true;
  runButton.querySelector("span:last-child").textContent = "Running";

  try {
    const response = await fetch("/segment", {
      method: "POST",
      body,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Segmentation failed.");
    }
    const resultBlob = base64ToBlob(payload.instances_png_base64, "image/png");
    const semanticBlob = base64ToBlob(payload.semantic_png_base64, "image/png");
    if (resultObjectUrl) {
      URL.revokeObjectURL(resultObjectUrl);
    }
    if (semanticObjectUrl) {
      URL.revokeObjectURL(semanticObjectUrl);
    }
    resultObjectUrl = URL.createObjectURL(resultBlob);
    semanticObjectUrl = URL.createObjectURL(semanticBlob);
    maskPreview.src = resultObjectUrl;
    downloadMask.href = resultObjectUrl;
    downloadMask.classList.remove("disabled");
    backendValue.textContent = payload.backend;
    objectCountValue.textContent = String(payload.object_count);
    scoreValue.textContent = payload.masks.length
      ? Number(payload.masks[0].score).toFixed(4)
      : "-";
    renderInstances(payload.masks);
    setViewMode("overlay");
    showToast(`Segmentation complete. ${payload.object_count} objects found.`);
  } catch (error) {
    showToast(error.message);
  } finally {
    runButton.disabled = false;
    runButton.querySelector("span:last-child").textContent = "Run segmentation";
  }
});

viewButtons.overlay.addEventListener("click", () => setViewMode("overlay"));
viewButtons.image.addEventListener("click", () => setViewMode("image"));
viewButtons.semantic.addEventListener("click", () => setViewMode("semantic"));
viewButtons.instances.addEventListener("click", () => setViewMode("instances"));

loadHealth();

function renderInstances(masks) {
  instancesList.innerHTML = "";
  if (!masks.length) {
    const row = document.createElement("div");
    row.className = "instance-row muted";
    row.textContent = "No objects above threshold.";
    instancesList.append(row);
    return;
  }
  masks.forEach((mask, index) => {
    const row = document.createElement("div");
    row.className = "instance-row";
    const swatch = document.createElement("span");
    swatch.className = "instance-swatch";
    swatch.style.backgroundColor = instanceColor(index);
    const detail = document.createElement("span");
    const areaRatio = Number(mask.area_ratio * 100).toFixed(2);
    detail.textContent = `#${mask.id} score ${Number(mask.score).toFixed(4)} area ${mask.area} (${areaRatio}%) box [${mask.bbox.join(", ")}]`;
    row.append(swatch, detail);
    instancesList.append(row);
  });
}

function instanceColor(index) {
  return "rgba(127, 29, 29, 0.9)";
}
