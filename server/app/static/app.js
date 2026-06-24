const form = document.querySelector("#segmentForm");
const imageInput = document.querySelector("#imageInput");
const promptInput = document.querySelector("#promptInput");
const thresholdInput = document.querySelector("#thresholdInput");
const thresholdValue = document.querySelector("#thresholdValue");
const postprocessInput = document.querySelector("#postprocessInput");
const proposalInput = document.querySelector("#proposalInput");
const maxProposalsInput = document.querySelector("#maxProposalsInput");
const previewProposalsButton = document.querySelector("#previewProposalsButton");
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
  proposals: document.querySelector("#showProposals"),
  edges: document.querySelector("#showEdges"),
  preprocess: document.querySelector("#showPreprocess"),
};

let imageObjectUrl = "";
let resultObjectUrl = "";
let semanticObjectUrl = "";
let proposalObjectUrl = "";
let edgesObjectUrl = "";
let preprocessObjectUrl = "";

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
  previewStage.classList.toggle(
    "mask-only",
    mode === "instances" ||
      mode === "semantic" ||
      mode === "proposals" ||
      mode === "edges" ||
      mode === "preprocess"
  );
  Object.entries(viewButtons).forEach(([key, button]) => {
    button.classList.toggle("active", key === mode);
  });
  if (mode === "semantic" && semanticObjectUrl) {
    maskPreview.src = semanticObjectUrl;
  }
  if ((mode === "overlay" || mode === "instances") && resultObjectUrl) {
    maskPreview.src = resultObjectUrl;
  }
  if (mode === "proposals" && proposalObjectUrl) {
    maskPreview.src = proposalObjectUrl;
  }
  if (mode === "edges" && edgesObjectUrl) {
    maskPreview.src = edgesObjectUrl;
  }
  if (mode === "preprocess" && preprocessObjectUrl) {
    maskPreview.src = preprocessObjectUrl;
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
  if (proposalObjectUrl) {
    URL.revokeObjectURL(proposalObjectUrl);
    proposalObjectUrl = "";
  }
  if (edgesObjectUrl) {
    URL.revokeObjectURL(edgesObjectUrl);
    edgesObjectUrl = "";
  }
  if (preprocessObjectUrl) {
    URL.revokeObjectURL(preprocessObjectUrl);
    preprocessObjectUrl = "";
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

async function previewProposals() {
  const file = imageInput.files[0];
  if (!file) {
    showToast("Select an image first.");
    return;
  }

  const body = new FormData();
  body.append("image", file);
  body.append("max_proposals", maxProposalsInput.value || "30");

  previewProposalsButton.disabled = true;
  previewProposalsButton.textContent = "Previewing";

  try {
    const response = await fetch("/proposals", {
      method: "POST",
      body,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Proposal generation failed.");
    }
    updateProposalOverlay(payload);
    updateEdgesPreview(payload);
    updatePreprocessPreview(payload);
    renderProposals(payload.proposals);
    setViewMode("proposals");
    showToast(`Generated ${payload.proposal_count} proposals.`);
  } catch (error) {
    showToast(error.message);
  } finally {
    previewProposalsButton.disabled = false;
    previewProposalsButton.textContent = "Preview proposals";
  }
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
  const selectedMode = postprocessInput.value;
  body.append(
    "postprocess",
    selectedMode === "sam_cascade" ? "polygon" : selectedMode
  );
  body.append(
    "inference_mode",
    selectedMode === "sam_cascade" ? "sam_cascade" : "text"
  );
  body.append("use_opencv_proposals", proposalInput.checked ? "true" : "false");
  body.append("max_proposals", maxProposalsInput.value || "30");
  if (boxInput.value.trim()) {
    body.append("box", boxInput.value.trim());
  }
  if (selectedMode !== "sam_cascade" && pointsInput.value.trim()) {
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
    if (payload.proposals_png_base64) {
      updateProposalOverlay(payload);
    }
    if (payload.edges_png_base64) {
      updateEdgesPreview(payload);
    }
    if (payload.preprocess_png_base64) {
      updatePreprocessPreview(payload);
    }
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
viewButtons.proposals.addEventListener("click", () => {
  if (!proposalObjectUrl) {
    showToast("No proposals generated yet.");
    return;
  }
  setViewMode("proposals");
});
viewButtons.edges.addEventListener("click", () => {
  if (!edgesObjectUrl) {
    showToast("No edges preview generated yet.");
    return;
  }
  setViewMode("edges");
});
viewButtons.preprocess.addEventListener("click", () => {
  if (!preprocessObjectUrl) {
    showToast("No preprocessing preview generated yet.");
    return;
  }
  setViewMode("preprocess");
});
previewProposalsButton.addEventListener("click", previewProposals);

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

function renderProposals(proposals) {
  instancesList.innerHTML = "";
  if (!proposals.length) {
    const row = document.createElement("div");
    row.className = "instance-row muted";
    row.textContent = "No OpenCV proposals found.";
    instancesList.append(row);
    return;
  }
  proposals.forEach((proposal) => {
    const row = document.createElement("div");
    row.className = "instance-row";
    const swatch = document.createElement("span");
    swatch.className = "proposal-swatch";
    const detail = document.createElement("span");
    detail.textContent = `#${proposal.id} score ${Number(proposal.score).toFixed(3)} point [${proposal.point.join(", ")}] box [${proposal.bbox.join(", ")}]`;
    row.append(swatch, detail);
    instancesList.append(row);
  });
}

function updateProposalOverlay(payload) {
  const proposalBlob = base64ToBlob(payload.proposals_png_base64, "image/png");
  if (proposalObjectUrl) {
    URL.revokeObjectURL(proposalObjectUrl);
  }
  proposalObjectUrl = URL.createObjectURL(proposalBlob);
}

function updateEdgesPreview(payload) {
  const edgesBlob = base64ToBlob(payload.edges_png_base64, "image/png");
  if (edgesObjectUrl) {
    URL.revokeObjectURL(edgesObjectUrl);
  }
  edgesObjectUrl = URL.createObjectURL(edgesBlob);
}

function updatePreprocessPreview(payload) {
  const preprocessBlob = base64ToBlob(payload.preprocess_png_base64, "image/png");
  if (preprocessObjectUrl) {
    URL.revokeObjectURL(preprocessObjectUrl);
  }
  preprocessObjectUrl = URL.createObjectURL(preprocessBlob);
}

function instanceColor(index) {
  return "rgba(127, 29, 29, 0.9)";
}
