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

// How long a slot has had its status, measured to the reading's own time (for
// recorded footage that is the capture time, not now). "≥" when the status may
// have begun before monitoring started.
function statusDuration(item, readingTime) {
  if (!item.changed_at || !readingTime) return "";
  const minutes = Math.max(0, Math.floor((new Date(readingTime) - new Date(item.changed_at)) / 60000));
  const text = minutes >= 60 ? `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, "0")}m` : `${minutes}m`;
  return `${item.changed_since_start ? "≥" : ""}${text}`;
}

function slotLabel(item, readingTime = state.readingTime) {
  const parts = [item.id, statusOf(item)];
  const duration = statusDuration(item, readingTime);
  if (duration) parts.push(`for ${duration}`);
  if (item.raw_status && item.raw_status !== item.status) parts.push(`latest frame: ${item.raw_status}, confirming`);
  parts.push(`${Math.round((item.confidence ?? 0) * 100)}% confidence`);
  if (item.updated_at) parts.push(timeOnly(item.updated_at));
  return parts.join(" / ");
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
  editor.replaceChildren();
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
    // Built with DOM calls: slot ids come from config and must never be parsed as HTML.
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("class", "editor-polygon");
    svg.setAttribute("viewBox", "0 0 100 100");
    svg.setAttribute("preserveAspectRatio", "none");
    svg.setAttribute("aria-hidden", "true");
    const polygon = document.createElementNS(SVG_NS, "polygon");
    polygon.setAttribute("points", polygonPoints);
    svg.appendChild(polygon);
    const label = document.createElement("span");
    label.textContent = slot.id;
    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "editor-delete";
    deleteButton.dataset.deleteSlot = index;
    deleteButton.title = `Delete ${slot.id}`;
    deleteButton.setAttribute("aria-label", `Delete ${slot.id}`);
    deleteButton.textContent = "x";
    item.append(svg, label, deleteButton);
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
  const readingTime = data.captured_at || data.updated_at;
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
    const duration = statusDuration(item, readingTime);
    label.textContent = duration ? `${status} · ${duration}` : status;
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
  if (status.state === "missing") return "Demo (no model)";
  return "Model error";
}

// Explains, in plain words, why there are no real readings: no model installed
// (demo data) or a model that can't be used (an error, never demo data).
function renderModelAlert(data) {
  const alert = $("modelAlert");
  const status = data.model_status;
  let text = "";
  if (status && status.state === "missing") {
    text = `No detection model is installed, so the map shows demo data.${status.error ? ` Last attempt: ${status.error}` : ""} Install one under Configure slots > Detection model.`;
  } else if (status && ["error", "needs_package", "unknown"].includes(status.state)) {
    const reason = status.state === "needs_package" ? `${status.name} needs its Python package: ${status.message}` : status.message;
    text = `Detection model problem: ${reason}. No frames can be processed until it is fixed under Configure slots > Detection model.`;
  }
  alert.textContent = text;
  alert.hidden = !text;
  alert.classList.toggle("is-error", Boolean(status) && status.state !== "missing");
}

// Camera-shift check: a warning when the camera seems to have moved, otherwise a
// quiet note in the map footer.
function renderCamera(data) {
  const camera = data.camera || {};
  const alert = $("cameraAlert");
  const moved = camera.state === "moved";
  alert.hidden = !moved;
  alert.textContent = moved
    ? `The camera may have moved: slots look about ${camera.shift_pct ?? "?"}% of the frame width out of place (${camera.streak} checks in a row). Check the layout with Edit layout; saving it makes the current frame the new reference.`
    : "";
  const notes = {
    ok: `Camera check OK${camera.shift_pct != null ? ` (${camera.shift_pct}% shift)` : ""}`,
    suspect: `Camera check: possible shift of ${camera.shift_pct}% (${camera.streak} of 3 checks)`,
    unverified: `Camera check: can't verify (${camera.reason || "too few features"})`,
  };
  const note = notes[camera.state];
  $("mapFootnote").textContent = `Polygon view / ${state.slots.length} slots${note ? ` / ${note}` : ""}`;
}

function render(data) {
  state.readingTime = data.captured_at || data.updated_at;
  const occupied = (data.slots || []).filter(
    (item) => item.status === "occupied",
  ).length;
  $("freeCount").textContent = data.free ?? "--";
  $("totalCount").textContent = data.total ?? "--";
  $("occupiedCount").textContent = occupied;
  $("unknownCount").textContent = data.unknown ?? 0;
  $("updatedAt").textContent = timeOnly(data.updated_at);
  const sourceLabels = { demo: "Demo source", none: "No frame yet", "model error": "Model error" };
  $("sourceLabel").textContent = sourceLabels[data.source] ?? data.source;
  $("modelLabel").textContent = modelLabel(data);
  renderModelAlert(data);
  state.lotName = data.lot_name || state.lotName;
  state.reviewThreshold = data.review_threshold ?? state.reviewThreshold;
  $("lotNameLabel").textContent = state.lotName;
  $("lotNameHeading").textContent = state.lotName;
  renderMap(data);
  renderMapStamp(data);
  renderActivity(data);
  renderCamera(data);
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
    if (model.status.error) {
      detail.textContent = model.status.error;
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

// --- Trends: free slots over time and busiest hours (plain SVG) ---
const TREND_REFRESH_MS = 60000;
const trendState = { range: "24h", data: null, timer: null };

function svgEl(name, attributes = {}, parent = null) {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (parent) parent.appendChild(element);
  return element;
}

function trendTimeLabel(iso, range) {
  const date = new Date(iso);
  return range === "7d"
    ? date.toLocaleDateString([], { weekday: "short", day: "numeric" })
    : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function hourLabel(hour) {
  return `${String(hour).padStart(2, "0")}:00`;
}

function showChartTooltip(lines, x, y) {
  const tooltip = $("chartTooltip");
  const panel = tooltip.parentElement.getBoundingClientRect();
  tooltip.replaceChildren(
    ...lines.map((line, index) => {
      const element = document.createElement(index === 0 ? "strong" : "span");
      element.textContent = line;
      return element;
    }),
  );
  tooltip.hidden = false;
  const left = Math.min(panel.width - tooltip.offsetWidth - 8, Math.max(8, x - panel.left + 14));
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${Math.max(8, y - panel.top - tooltip.offsetHeight - 10)}px`;
}

function hideChartTooltip() {
  $("chartTooltip").hidden = true;
}

function emptyChart(box, text) {
  const empty = document.createElement("div");
  empty.className = "chart-empty";
  empty.textContent = text;
  box.replaceChildren(empty);
}

// Line + 10% area wash, one series: no legend (the caption names it). A crosshair
// snaps to the nearest interval; arrow keys move it when the chart has focus.
function renderFreeChart(data) {
  const box = $("freeChart");
  const points = data.series.map((point, index) => ({ ...point, index })).filter((point) => point.free != null);
  if (!points.length) {
    emptyChart(box, "No readings in this range yet. Process a video or start a live source.");
    return;
  }
  const width = Math.max(260, box.clientWidth);
  const height = 200;
  const margin = { top: 12, right: 44, bottom: 22, left: 30 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  // The axis ends at the number of slots: "free" can't exceed it.
  const yMax = Math.max(1, ...points.map((point) => point.total || point.free));
  const x = (index) => margin.left + (index / Math.max(1, data.series.length - 1)) * plotWidth;
  const y = (value) => margin.top + plotHeight - (value / yMax) * plotHeight;

  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, height, role: "img", tabindex: "0" });
  const latest = points[points.length - 1];
  svg.setAttribute(
    "aria-label",
    `Free slots over the ${data.range === "7d" ? "last 7 days" : "last 24 hours"} of readings; latest ${latest.free} of ${latest.total}. Use the arrow keys to read each interval, or open the table below.`,
  );
  const grid = svgEl("g", { class: "chart-grid" }, svg);
  const axis = svgEl("g", { class: "chart-axis" }, svg);
  const ticks = yMax <= 4 ? Array.from({ length: yMax + 1 }, (_, i) => i) : [0, Math.round(yMax / 2), yMax];
  for (const tick of ticks) {
    svgEl("line", { x1: margin.left, x2: margin.left + plotWidth, y1: y(tick), y2: y(tick) }, grid);
    svgEl("text", { x: margin.left - 6, y: y(tick) + 3, "text-anchor": "end" }, axis).textContent = Math.round(tick);
  }
  const labelCount = Math.min(5, data.series.length);
  for (let i = 0; i < labelCount; i += 1) {
    const index = Math.round((i / Math.max(1, labelCount - 1)) * (data.series.length - 1));
    const anchor = i === 0 ? "start" : i === labelCount - 1 ? "end" : "middle";
    svgEl("text", { x: x(index), y: height - 6, "text-anchor": anchor }, axis).textContent = trendTimeLabel(data.series[index].t, data.range);
  }

  // Join readings across short gaps (sparse sampling); break the line only where
  // more than MAX_GAP intervals in a row have no readings (an outage).
  const MAX_GAP = 3;
  const runs = [];
  let run = [];
  points.forEach((point) => {
    const previous = run[run.length - 1];
    if (previous && point.index - previous.index > MAX_GAP + 1) {
      runs.push(run);
      run = [];
    }
    run.push(point);
  });
  if (run.length) runs.push(run);
  for (const segment of runs) {
    const line = segment.map((point, i) => `${i ? "L" : "M"}${x(point.index).toFixed(1)},${y(point.free).toFixed(1)}`).join("");
    if (segment.length > 1) {
      const area = `${line}L${x(segment[segment.length - 1].index).toFixed(1)},${y(0)}L${x(segment[0].index).toFixed(1)},${y(0)}Z`;
      svgEl("path", { d: area, class: "chart-area" }, svg);
    }
    svgEl("path", { d: line, class: "chart-line" }, svg);
    if (segment.length === 1) svgEl("circle", { cx: x(segment[0].index), cy: y(segment[0].free), r: 3, class: "chart-dot" }, svg);
  }
  // Direct label at the end: the latest value only.
  svgEl("circle", { cx: x(latest.index), cy: y(latest.free), r: 4, class: "chart-dot" }, svg);
  svgEl("text", { x: x(latest.index) + 8, y: y(latest.free) + 3, class: "chart-label" }, svg).textContent = `${Math.round(latest.free)} free`;

  const crosshair = svgEl("line", { y1: margin.top, y2: margin.top + plotHeight, class: "chart-crosshair", visibility: "hidden" }, svg);
  const marker = svgEl("circle", { r: 4, class: "chart-dot", visibility: "hidden" }, svg);
  let current = points.length - 1;
  const show = (position, clientX, clientY) => {
    const point = points[position];
    current = position;
    crosshair.setAttribute("x1", x(point.index));
    crosshair.setAttribute("x2", x(point.index));
    crosshair.setAttribute("visibility", "visible");
    marker.setAttribute("cx", x(point.index));
    marker.setAttribute("cy", y(point.free));
    marker.setAttribute("visibility", "visible");
    const bounds = svg.getBoundingClientRect();
    const scale = bounds.width / width;
    showChartTooltip(
      [
        `${point.free} free of ${point.total}`,
        `${trendTimeLabel(point.t, "7d")} ${trendTimeLabel(point.t, "24h")}`,
        `lowest ${point.free_min} / unknown ${point.unknown} / ${point.frames} frames`,
      ],
      clientX ?? bounds.left + x(point.index) * scale,
      clientY ?? bounds.top + y(point.free) * scale,
    );
  };
  const hide = () => {
    crosshair.setAttribute("visibility", "hidden");
    marker.setAttribute("visibility", "hidden");
    hideChartTooltip();
  };
  const hit = svgEl("rect", { x: margin.left, y: margin.top, width: plotWidth, height: plotHeight, class: "chart-hit" }, svg);
  hit.addEventListener("pointermove", (event) => {
    const bounds = svg.getBoundingClientRect();
    const svgX = ((event.clientX - bounds.left) / bounds.width) * width;
    let best = 0;
    points.forEach((point, position) => {
      if (Math.abs(x(point.index) - svgX) < Math.abs(x(points[best].index) - svgX)) best = position;
    });
    show(best, event.clientX, event.clientY);
  });
  hit.addEventListener("pointerleave", hide);
  svg.addEventListener("focus", () => show(current));
  svg.addEventListener("blur", hide);
  svg.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      show(Math.min(points.length - 1, Math.max(0, current + (event.key === "ArrowRight" ? 1 : -1))));
    }
  });
  box.replaceChildren(svg);
}

function roundedTopBar(x, y, width, height) {
  const r = Math.min(4, height, width / 2);
  return `M${x},${y + height}V${y + r}Q${x},${y} ${x + r},${y}H${x + width - r}Q${x + width},${y} ${x + width},${y + r}V${y + height}Z`;
}

// 24 columns from one baseline, <= 24px thick with a 4px rounded top. Peak hours
// are a darker step and carry their value and a "Peak" label (never colour alone).
function renderHourChart(data) {
  const box = $("hourChart");
  if (!data.hours.some((hour) => hour.occupancy != null)) {
    emptyChart(box, "Busy hours appear once there are readings.");
    return;
  }
  const width = Math.max(260, box.clientWidth);
  const height = 200;
  const margin = { top: 26, right: 4, bottom: 22, left: 34 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const band = plotWidth / 24;
  const barWidth = Math.min(24, Math.max(3, band - 2)); // >= 2px surface gap between columns
  const y = (value) => margin.top + plotHeight - value * plotHeight;
  const peaks = new Set(data.peak_hours);
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, height, role: "group" });
  svg.setAttribute("aria-label", `Average occupancy by hour of day. Peak hours: ${data.peak_hours.map(hourLabel).join(", ") || "none yet"}.`);
  const grid = svgEl("g", { class: "chart-grid" }, svg);
  const axis = svgEl("g", { class: "chart-axis" }, svg);
  for (const tick of [0, 0.5, 1]) {
    svgEl("line", { x1: margin.left, x2: margin.left + plotWidth, y1: y(tick), y2: y(tick) }, grid);
    svgEl("text", { x: margin.left - 6, y: y(tick) + 3, "text-anchor": "end" }, axis).textContent = `${tick * 100}%`;
  }
  data.hours.forEach((hour) => {
    const center = margin.left + band * (hour.hour + 0.5);
    if (hour.hour % 3 === 0) svgEl("text", { x: center, y: height - 6, "text-anchor": "middle" }, axis).textContent = String(hour.hour).padStart(2, "0");
    if (hour.occupancy == null) return;
    const top = y(hour.occupancy);
    const barHeight = Math.max(1, y(0) - top);
    const group = svgEl("g", {}, svg);
    // The hit target is the whole column, bigger than the bar.
    const hit = svgEl("rect", { x: center - band / 2, y: margin.top, width: band, height: plotHeight, class: "chart-hit", tabindex: "0" }, group);
    const label = `${hourLabel(hour.hour)}-${hourLabel((hour.hour + 1) % 24)}: ${Math.round(hour.occupancy * 100)}% occupied${peaks.has(hour.hour) ? " (peak)" : ""}, ${hour.frames} frames`;
    hit.setAttribute("aria-label", label);
    svgEl("path", { d: roundedTopBar(center - barWidth / 2, top, barWidth, barHeight), class: `chart-bar${peaks.has(hour.hour) ? " is-peak" : ""}` }, group);
    if (hour.hour === data.peak_hours[0]) {
      // Only the busiest hour carries a value; the others are listed below the chart.
      svgEl("text", { x: center, y: top - 5, "text-anchor": "middle", class: "chart-label" }, group).textContent = `${Math.round(hour.occupancy * 100)}%`;
    }
    const lines = [`${Math.round(hour.occupancy * 100)}% occupied`, `${hourLabel(hour.hour)}-${hourLabel((hour.hour + 1) % 24)}${peaks.has(hour.hour) ? " / peak" : ""}`, `${hour.frames} frames`];
    hit.addEventListener("pointermove", (event) => showChartTooltip(lines, event.clientX, event.clientY));
    hit.addEventListener("pointerleave", hideChartTooltip);
    hit.addEventListener("focus", () => {
      const bounds = hit.getBoundingClientRect();
      showChartTooltip(lines, bounds.left + bounds.width / 2, bounds.top + 20);
    });
    hit.addEventListener("blur", hideChartTooltip);
  });
  const peakNote = document.createElement("p");
  peakNote.className = "chart-peaks";
  peakNote.textContent = data.peak_hours.length
    ? `Peak hours (darker): ${[...data.peak_hours].sort((a, b) => a - b).map((hour) => `${hourLabel(hour)}-${hourLabel((hour + 1) % 24)}`).join(", ")}`
    : "";
  box.replaceChildren(svg, peakNote);
}

function tableOf(caption, headings, rows) {
  const table = document.createElement("table");
  table.createCaption().textContent = caption;
  const head = table.createTHead().insertRow();
  for (const heading of headings) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = heading;
    head.appendChild(cell);
  }
  const body = table.createTBody();
  for (const values of rows) {
    const row = body.insertRow();
    for (const value of values) row.insertCell().textContent = value;
  }
  const scroll = document.createElement("div");
  scroll.className = "table-scroll";
  scroll.appendChild(table);
  return scroll;
}

function renderTrendTable(data) {
  const series = data.series
    .filter((point) => point.free != null)
    .map((point) => [`${trendTimeLabel(point.t, "7d")} ${trendTimeLabel(point.t, "24h")}`, point.free, point.free_min, point.unknown, point.frames]);
  const hours = data.hours
    .filter((hour) => hour.occupancy != null)
    .map((hour) => [hourLabel(hour.hour), `${Math.round(hour.occupancy * 100)}%`, data.peak_hours.includes(hour.hour) ? "peak" : "", hour.frames]);
  $("trendTable").replaceChildren(
    tableOf("Free slots per interval", ["Interval", "Free (avg)", "Lowest", "Unknown", "Frames"], series),
    tableOf("Occupancy by hour", ["Hour", "Occupied", "", "Frames"], hours),
  );
}

function renderTrends(data) {
  trendState.data = data;
  const span = data.range === "7d" ? "7 days" : "24 hours";
  $("trendNote").textContent = data.end
    ? `${data.frames} frames over the ${span} up to the latest reading (${new Date(data.end).toLocaleString()}). Statuses are the smoothed ones shown on the dashboard.`
    : "No readings yet.";
  renderFreeChart(data);
  renderHourChart(data);
  renderTrendTable(data);
}

async function loadTrends() {
  clearTimeout(trendState.timer);
  const grid = $("trendGrid");
  grid.classList.add("is-loading");
  try {
    const response = await fetch(`/api/trends?range=${encodeURIComponent(trendState.range)}`, { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not load trends.");
    renderTrends(data);
  } catch (error) {
    $("trendNote").textContent = error.message;
  } finally {
    grid.classList.remove("is-loading");
    trendState.timer = setTimeout(loadTrends, TREND_REFRESH_MS);
  }
}

document.querySelectorAll(".trend-range button").forEach((button) => {
  button.addEventListener("click", () => {
    trendState.range = button.dataset.range;
    document.querySelectorAll(".trend-range button").forEach((other) => other.setAttribute("aria-pressed", String(other === button)));
    loadTrends();
  });
});
let trendResizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(trendResizeTimer);
  trendResizeTimer = setTimeout(() => trendState.data && renderTrends(trendState.data), 150);
});

refresh();
setTimer();
loadTrends();
