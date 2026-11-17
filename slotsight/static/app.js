const state = {
  slots: [],
  slotConfig: [],
  lotName: "Lot 01",
  reviewThreshold: 0.7,
  timer: null,
  editing: false,
  hoverId: null, // slot under the pointer, so the tooltip follows each refresh
};
// Dashboard poll interval. Sources sample every 2 s by default, so a change
// reaches the screen within about 2 s + processing + 2 s, under R-04's 5 s.
const POLL_MS = 2000;
let thresholdTimer = null;
const $ = (id) => document.getElementById(id);

function timeOnly(iso) {
  if (!iso) return "--:--:--";
  return new Date(iso).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

const SVG_NS = "http://www.w3.org/2000/svg";
const STATUSES = ["occupied", "vacant", "unknown"];
// A shape as well as a colour, so status reads without relying on colour alone.
const STATUS_GLYPHS = { occupied: "\u25CF", vacant: "\u25CB", unknown: "?" };

function statusOf(item) {
  return STATUSES.includes(item.status) ? item.status : "unknown";
}

function slotLabel(item) {
  const time = item.updated_at ? ` / ${timeOnly(item.updated_at)}` : "";
  return `${item.id} / ${statusOf(item)} / ${Math.round((item.confidence ?? 0) * 100)}% confidence${time}`;
}

// R-06: every slot is filled in its status colour, carries a pinned ID chip,
// and shows its status, confidence and last-update time on hover or focus.
function renderMap(data) {
  state.slots = data.slots || [];
  const layer = $("slotLayer");
  layer.replaceChildren();
  const shapes = document.createElementNS(SVG_NS, "svg");
  shapes.setAttribute("class", "slot-shapes");
  shapes.setAttribute("viewBox", "0 0 100 100");
  shapes.setAttribute("preserveAspectRatio", "none");
  shapes.setAttribute("aria-hidden", "true");
  layer.appendChild(shapes);
  state.slots.forEach((item, index) => {
    const points = state.slotConfig?.[index]?.points || item.points;
    if (!points?.length) return;
    const status = statusOf(item);

    const shape = document.createElementNS(SVG_NS, "polygon");
    shape.setAttribute("points", points.map(([x, y]) => `${x},${y}`).join(" "));
    shape.setAttribute("class", `slot-shape status-${status}`);
    shapes.appendChild(shape);

    const hit = document.createElement("button");
    hit.type = "button";
    hit.className = `slot-hit status-${status}`;
    hit.style.clipPath = `polygon(${points.map(([x, y]) => `${x}% ${y}%`).join(",")})`;
    hit.setAttribute("aria-label", slotLabel(item));
    hit.addEventListener("pointerenter", (event) => showConfidence(item, event));
    hit.addEventListener("pointermove", moveConfidence);
    hit.addEventListener("pointerleave", hideConfidence);
    hit.addEventListener("focus", () => showConfidence(item));
    hit.addEventListener("blur", hideConfidence);
    layer.appendChild(hit);

    const chip = document.createElement("span");
    chip.className = `slot-chip status-${status}`;
    chip.setAttribute("aria-hidden", "true");
    // Pin the chip at the polygon's centre, kept far enough from the edges to stay readable.
    const centre = (axis) => points.reduce((sum, point) => sum + Number(point[axis]), 0) / points.length;
    chip.style.left = `${Math.min(95, Math.max(5, centre(0)))}%`;
    chip.style.top = `${Math.min(96, Math.max(4, centre(1)))}%`;
    const glyph = document.createElement("b");
    glyph.textContent = STATUS_GLYPHS[status];
    chip.append(item.id, glyph);
    layer.appendChild(chip);
  });
  $("mapFootnote").textContent = `Polygon view / ${state.slots.length} slots`;
  // The slot buttons were rebuilt, so no new pointer event fires for a still pointer:
  // refresh an open tooltip from the new data instead of leaving the old reading.
  if (state.hoverId !== null) {
    const hovered = state.slots.find((item) => item.id === state.hoverId);
    if (hovered) $("confidenceTooltip").textContent = slotLabel(hovered);
    else hideConfidence();
  }
  if (state.editing) renderEditor();
}

function renderMapStamp(data) {
  const parts = [`Updated ${timeOnly(data.updated_at)}`];
  if (data.captured_at) parts.push(`captured ${timeOnly(data.captured_at)}`);
  else if (data.frame_time_s != null) parts.push(`video ${seconds(data.frame_time_s)}`);
  if (data.source === "demo") parts.push("demo data");
  $("mapStamp").textContent = parts.join(" / ");
}

function showConfidence(item, event) {
  const tooltip = $("confidenceTooltip");
  state.hoverId = item.id;
  tooltip.textContent = slotLabel(item);
  tooltip.hidden = false;
  moveConfidence(event);
}

function moveConfidence(event) {
  const map = $("lotMap").getBoundingClientRect();
  const tooltip = $("confidenceTooltip");
  if (!event) {
    // Keyboard focus: pin the tooltip to the top-left corner of the map.
    tooltip.style.left = "10px";
    tooltip.style.top = "10px";
    return;
  }
  tooltip.style.left = `${Math.min(map.width - tooltip.offsetWidth - 10, Math.max(10, event.clientX - map.left + 12))}px`;
  tooltip.style.top = `${Math.min(map.height - 34, Math.max(10, event.clientY - map.top - 36))}px`;
}

function hideConfidence() {
  state.hoverId = null;
  $("confidenceTooltip").hidden = true;
}

function mapPoint(event) {
  const bounds = $("lotMap").getBoundingClientRect();
  return [
    Math.max(
      0,
      Math.min(100, ((event.clientX - bounds.left) / bounds.width) * 100),
    ),
    Math.max(
      0,
      Math.min(100, ((event.clientY - bounds.top) / bounds.height) * 100),
    ),
  ];
}

function renderEditor() {
  const editor = $("editorLayer");
  editor.innerHTML = "";
  state.slotConfig.forEach((slot, index) => {
    const xs = slot.points.map((point) => point[0]);
    const ys = slot.points.map((point) => point[1]);
    const left = Math.min(...xs);
    const top = Math.min(...ys);
    const width = Math.max(...xs) - left;
    const height = Math.max(...ys) - top;
    const item = document.createElement("div");
    item.className = "editor-slot";
    item.style.left = `${left}%`;
    item.style.top = `${top}%`;
    item.style.width = `${width}%`;
    item.style.height = `${height}%`;
    const polygonPoints = slot.points
      .map(
        (point) =>
          `${((point[0] - left) / width) * 100},${((point[1] - top) / height) * 100}`,
      )
      .join(" ");
    item.innerHTML = `<svg class="editor-polygon" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true"><polygon points="${polygonPoints}"></polygon></svg><span>${slot.id}</span><button type="button" class="editor-delete" data-delete-slot="${index}" title="Delete ${slot.id}" aria-label="Delete ${slot.id}">x</button>`;
    const deleteButton = item.querySelector("button");
    deleteButton.style.left = "100%";
    deleteButton.style.top = "0%";
    slot.points.forEach((point, pointIndex) => {
      const handle = document.createElement("button");
      handle.type = "button";
      handle.className = "editor-handle";
      handle.dataset.slotIndex = index;
      handle.dataset.pointIndex = pointIndex;
      handle.title = `Move ${slot.id} corner ${pointIndex + 1}`;
      handle.setAttribute(
        "aria-label",
        `Move ${slot.id} corner ${pointIndex + 1}`,
      );
      handle.style.left = `${((point[0] - left) / width) * 100}%`;
      handle.style.top = `${((point[1] - top) / height) * 100}%`;
      item.appendChild(handle);
    });
    editor.appendChild(item);
  });
}

function setEditMode(enabled) {
  state.editing = enabled;
  $("lotMap").classList.toggle("is-editing", enabled);
  $("editorLayer").hidden = !enabled;
  $("editLayout").textContent = enabled ? "Save layout" : "Edit layout";
  if (enabled) {
    if (state.timer) clearInterval(state.timer);
    renderEditor();
  } else {
    setTimer();
  }
}

async function saveVisualConfig() {
  const response = await fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      slots: state.slotConfig,
      lot_name: state.lotName,
      review_threshold: state.reviewThreshold,
    }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Could not save layout.");
  setEditMode(false);
  await refresh();
}

function renderActivity(data) {
  const rows = (data.slots || []).map((item) => {
    const status = statusOf(item);
    const row = document.createElement("div");
    row.className = "activity-row";
    const name = document.createElement("span");
    const dot = document.createElement("i");
    dot.className = `status-dot dot-${status}`;
    name.append(dot, item.id);
    const label = document.createElement("span");
    label.className = "activity-status";
    label.textContent = status;
    const time = document.createElement("span");
    time.className = "activity-time";
    time.textContent = timeOnly(item.updated_at ?? data.updated_at);
    row.append(name, label, time);
    return row;
  });
  $("activityList").replaceChildren(...rows);
}

// System health card: the model in use, or what is happening to it.
function modelLabel(data) {
  const status = data.model_status;
  if (!status) return data.model || "Demo";
  if (status.state === "ready") return status.name;
  if (status.state === "downloading")
    return status.progress == null ? "Downloading..." : `Downloading ${Math.round(status.progress * 100)}%`;
  if (status.state === "loading") return "Loading model...";
  return "Demo (no model)";
}

function render(data) {
  const occupied = (data.slots || []).filter(
    (item) => item.status === "occupied",
  ).length;
  $("freeCount").textContent = data.free ?? "--";
  $("totalCount").textContent = data.total ?? "--";
  $("occupiedCount").textContent = occupied;
  $("unknownCount").textContent = data.unknown ?? 0;
  $("updatedAt").textContent = timeOnly(data.updated_at);
  const sourceLabels = { demo: "Demo source", none: "No frame yet" };
  $("sourceLabel").textContent = sourceLabels[data.source] ?? data.source;
  $("modelLabel").textContent = modelLabel(data);
  state.lotName = data.lot_name || state.lotName;
  state.reviewThreshold = data.review_threshold ?? state.reviewThreshold;
  $("lotNameLabel").textContent = state.lotName;
  $("lotNameHeading").textContent = state.lotName;
  renderMap(data);
  renderMapStamp(data);
  renderActivity(data);
  if ("job" in data) renderJob(data.job);
  if (data.preview_image) {
    const preview = $("framePreview");
    $("previewImage").src = data.preview_image;
    preview.classList.remove("is-empty");
    $("mapPreviewImage").src = data.preview_image;
    $("lotMap").classList.add("has-preview");
  }
}

async function refresh() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    const data = await response.json();
    state.slotConfig = data.slots.map((item) => ({
      id: item.id,
      points: item.points,
    }));
    state.lotName = data.lot_name || state.lotName;
    state.reviewThreshold = data.review_threshold ?? state.reviewThreshold;
    render(data);
  } catch (error) {
    $("formMessage").textContent = "Could not reach the monitor.";
  }
}

function setTimer() {
  if (state.timer) clearInterval(state.timer);
  if ($("autoRefresh").checked) state.timer = setInterval(refresh, POLL_MS);
}

function seconds(value) {
  return value == null ? "--" : `${Number(value).toFixed(2)} s`;
}

function renderJob(job) {
  const line = $("jobStatus");
  $("stopSource").disabled = !job || !["starting", "running"].includes(job.state);
  if (!job) {
    line.hidden = true;
    return;
  }
  const perFrame = job.processing_s ? `${seconds(job.processing_s.mean)} per frame` : "";
  const isFile = job.kind === "file";
  const progress =
    isFile && job.progress != null ? `${Math.round(job.progress * 100)}%` : "";
  const parts = {
    starting: [`Opening ${job.name}...`],
    running: isFile
      ? [`Processing ${job.name}`, progress, `${job.frames_processed} frames`, perFrame]
      : [`Live: ${job.name}`, `every ${job.interval_s} s`, `last ${timeOnly(job.last_frame_at)}`, perFrame],
    finished: [`Finished ${job.name}`, `${job.frames_processed} frames logged`],
    stopped: [`Stopped ${job.name}`, `${job.frames_processed} frames logged`],
    error: [`Source error: ${job.error}`],
  }[job.state] || [job.state];
  line.textContent = parts.filter(Boolean).join(" / ");
  line.classList.toggle("is-error", job.state === "error");
  line.hidden = false;
}

// Optional real-world start time of uploaded footage, sent as UTC ISO 8601.
// Rows for a video then carry start + frame offset as their timestamp.
function recordedAtIso() {
  const value = $("recordedAt").value;
  return value ? new Date(value).toISOString() : "";
}

async function postSource(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed.");
  return data;
}

$("refreshButton").addEventListener("click", refresh);
$("autoRefresh").addEventListener("change", setTimer);
$("reviewThresholdInput").addEventListener("input", (event) => {
  const threshold = Number(event.target.value) / 100;
  state.reviewThreshold = threshold;
  $("reviewThresholdValue").textContent = `${event.target.value}%`;
  clearTimeout(thresholdTimer);
  thresholdTimer = setTimeout(async () => {
    try {
      const response = await fetch("/api/review-threshold", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ review_threshold: threshold }),
      });
      const data = await response.json();
      if (!response.ok)
        throw new Error(data.error || "Could not update threshold.");
      render(data);
    } catch (error) {
      $("configMessage").textContent = error.message;
    }
  }, 180);
});
$("editLayout").addEventListener("click", async () => {
  if (!state.editing) {
    setEditMode(true);
    return;
  }
  try {
    await saveVisualConfig();
  } catch (error) {
    window.alert(error.message);
  }
});
$("editorLayer").addEventListener("click", (event) => {
  const button = event.target.closest("[data-delete-slot]");
  if (!button) return;
  const index = Number(button.dataset.deleteSlot);
  state.slotConfig.splice(index, 1);
  state.slots.splice(index, 1);
  renderEditor();
  renderMap({ slots: state.slots });
});
let draggingHandle = null;
$("editorLayer").addEventListener("pointerdown", (event) => {
  const handle = event.target.closest("[data-point-index]");
  if (!handle) return;
  draggingHandle = {
    slotIndex: Number(handle.dataset.slotIndex),
    pointIndex: Number(handle.dataset.pointIndex),
  };
  event.preventDefault();
});
document.addEventListener("pointermove", (event) => {
  if (!draggingHandle) return;
  const slot = state.slotConfig[draggingHandle.slotIndex];
  if (!slot) return;
  slot.points[draggingHandle.pointIndex] = mapPoint(event);
  renderEditor();
});
document.addEventListener("pointerup", () => {
  draggingHandle = null;
});
$("clearLayout").addEventListener("click", () => {
  if (!window.confirm("Clear every parking slot from this layout?")) return;
  if (!state.editing) setEditMode(true);
  state.slotConfig = [];
  state.slots = [];
  renderEditor();
  renderMap({ slots: [] });
});
let drawing = null;
$("lotMap").addEventListener("pointerdown", (event) => {
  if (
    !state.editing ||
    event.target.closest("button") ||
    event.target.closest(".editor-slot")
  )
    return;
  drawing = { start: mapPoint(event), preview: document.createElement("div") };
  drawing.preview.className = "draw-preview";
  $("editorLayer").appendChild(drawing.preview);
  event.currentTarget.setPointerCapture(event.pointerId);
});
$("lotMap").addEventListener("pointermove", (event) => {
  if (!drawing) return;
  const end = mapPoint(event);
  const left = Math.min(drawing.start[0], end[0]);
  const top = Math.min(drawing.start[1], end[1]);
  drawing.preview.style.left = `${left}%`;
  drawing.preview.style.top = `${top}%`;
  drawing.preview.style.width = `${Math.abs(end[0] - drawing.start[0])}%`;
  drawing.preview.style.height = `${Math.abs(end[1] - drawing.start[1])}%`;
});
$("lotMap").addEventListener("pointerup", (event) => {
  if (!drawing) return;
  const end = mapPoint(event);
  const left = Math.min(drawing.start[0], end[0]);
  const right = Math.max(drawing.start[0], end[0]);
  const top = Math.min(drawing.start[1], end[1]);
  const bottom = Math.max(drawing.start[1], end[1]);
  drawing.preview.remove();
  if (right - left >= 2 && bottom - top >= 2) {
    let suffix = state.slotConfig.length + 1;
    let id = `S${String(suffix).padStart(2, "0")}`;
    while (state.slotConfig.some((slot) => slot.id === id)) {
      suffix += 1;
      id = `S${String(suffix).padStart(2, "0")}`;
    }
    const points = [
      [left, top],
      [right, top],
      [right, bottom],
      [left, bottom],
    ];
    state.slotConfig.push({ id, points });
    state.slots.push({ id, status: "unknown", confidence: 0, points });
    renderEditor();
    renderMap({ slots: state.slots });
  }
  drawing = null;
});
$("sourceFile").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file)
    event.target.closest("label").querySelector("strong").textContent =
      file.name;
});
$("uploadForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = $("formMessage");
  const file = $("sourceFile").files[0];
  message.textContent = file
    ? "Processing frame..."
    : "Refreshing latest reading...";
  const formData = new FormData();
  if (file) formData.append("source", file);
  if (file && recordedAtIso()) formData.append("recorded_at", recordedAtIso());
  try {
    const response = await fetch("/api/process", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (response.status === 202) {
      // A video: it is sampled in the background and progress arrives with each poll.
      renderJob(data.job);
      message.textContent = data.message;
    } else if (response.ok) {
      render(data);
      message.textContent = file
        ? "Reading logged at " + timeOnly(data.updated_at)
        : "Showing the latest reading (nothing new logged).";
    } else message.textContent = data.error || "Processing failed.";
  } catch (error) {
    message.textContent = "Processing failed. Is the file under 100 MB?";
  }
});

$("liveSourceForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = $("formMessage");
  try {
    const data = await postSource("/api/source", {
      target: $("liveSourceInput").value,
      realtime: $("liveRealtime").checked,
      recorded_at: recordedAtIso(),
    });
    renderJob(data.job);
    message.textContent = data.message;
  } catch (error) {
    message.textContent = error.message;
  }
});

$("stopSource").addEventListener("click", async () => {
  try {
    const data = await postSource("/api/source/stop", {});
    renderJob(data.job);
  } catch (error) {
    $("formMessage").textContent = error.message;
  }
});

// --- Detection model picker (Configure dialog) ---
let modelTimer = null;

function modelState(model, job) {
  const running = job && job.state === "running" && job.id === model.id;
  if (running && job.action === "install")
    return job.progress == null ? "Downloading..." : `Downloading ${Math.round(job.progress * 100)}%`;
  if (running) return "Loading...";
  if (model.status.state === "error") return "Error";
  if (model.active && model.status.state === "ready") return "In use";
  if (model.active) return "Selected";
  if (!model.installed) return "Not installed";
  if (!model.package_available) return "Needs package";
  return "Installed";
}

function renderModels(data) {
  const busy = data.job && data.job.state === "running";
  $("modelPickerNote").textContent = data.overridden
    ? "Set by SLOTSIGHT_MODEL"
    : "COCO-trained; only car, motorcycle, bus and truck are used";
  const rows = data.models.map((model) => {
    const row = document.createElement("div");
    row.className = "model-row";
    row.classList.toggle("is-active", model.active);
    const info = document.createElement("div");
    info.className = "model-info";
    const name = document.createElement("strong");
    name.textContent = model.name;
    const meta = document.createElement("small");
    meta.textContent = `${model.provider} / ${model.licence} / ${model.size_mb ? `${model.size_mb} MB` : "local file"}`;
    info.append(name, meta);
    const detail = document.createElement("small");
    detail.className = "model-note";
    if (model.status.state === "error") {
      detail.textContent = model.status.message;
      detail.classList.add("is-error");
    } else if (!model.package_available) {
      detail.textContent = `Needs the Python package: pip install ${model.pip}`;
    } else {
      detail.textContent = model.note;
    }
    if (detail.textContent) info.append(detail);

    const state = document.createElement("span");
    state.className = "model-state";
    state.textContent = modelState(model, data.job);

    const actions = document.createElement("div");
    actions.className = "model-actions";
    if (!model.installed && model.size_mb) {
      actions.append(modelButton("Install", busy, () => modelAction("/api/models/install", model.id)));
    } else if (model.installed && !model.active) {
      actions.append(
        modelButton("Use", busy || data.overridden || !model.package_available, () =>
          modelAction("/api/models/select", model.id),
        ),
      );
    }
    row.append(info, state, actions);
    return row;
  });
  $("modelList").replaceChildren(...rows);
}

function modelButton(label, disabled, onClick) {
  const button = document.createElement("button");
  button.type = "button"; // inside a method="dialog" form: never submit or close it
  button.className = "map-edit-button";
  button.textContent = label;
  button.disabled = disabled;
  button.addEventListener("click", onClick);
  return button;
}

async function modelAction(url, id) {
  const message = $("modelMessage");
  try {
    const data = await postSource(url, { id });
    message.textContent = data.message || "";
  } catch (error) {
    message.textContent = error.message;
  }
  await loadModels();
  refresh();
}

async function loadModels() {
  clearTimeout(modelTimer);
  try {
    const response = await fetch("/api/models", { cache: "no-store" });
    const data = await response.json();
    renderModels(data);
    // Follow a running install or load while the dialog is open.
    if (data.job && data.job.state === "running" && $("configDialog").open)
      modelTimer = setTimeout(loadModels, 1000);
  } catch (error) {
    $("modelMessage").textContent = "Could not list the detection models.";
  }
}

$("openConfig").addEventListener("click", async () => {
  const response = await fetch("/api/status");
  const data = await response.json();
  const slots = data.slots.map((item, index) => ({
    id: item.id,
    points: state.slotConfig?.[index]?.points || [
      [10, 20],
      [20, 20],
      [20, 40],
      [10, 40],
    ],
  }));
  $("lotNameInput").value = data.lot_name || state.lotName;
  $("reviewThresholdInput").value = Math.round(
    (data.review_threshold ?? state.reviewThreshold) * 100,
  );
  $("reviewThresholdValue").textContent = `${$("reviewThresholdInput").value}%`;
  $("configInput").value = JSON.stringify(slots, null, 2);
  $("modelMessage").textContent = "";
  $("configDialog").showModal();
  loadModels();
});
$("saveConfig").addEventListener("click", async (event) => {
  event.preventDefault();
  const message = $("configMessage");
  try {
    const slots = JSON.parse($("configInput").value);
    const lotName = $("lotNameInput").value.trim();
    const reviewThreshold = Number($("reviewThresholdInput").value) / 100;
    const response = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        slots,
        lot_name: lotName,
        review_threshold: reviewThreshold,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    state.slotConfig = slots;
    state.lotName = data.lot_name;
    state.reviewThreshold = data.review_threshold;
    message.textContent = data.message;
    setTimeout(() => $("configDialog").close(), 350);
    refresh();
  } catch (error) {
    message.textContent = error.message || "Invalid JSON.";
  }
});

refresh();
setTimer();
