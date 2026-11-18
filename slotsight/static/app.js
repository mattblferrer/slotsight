const state = {
  slots: [],
  slotConfig: [],
  lotName: "Lot 01",
  reviewThreshold: 0.7,
  timer: null,
  editing: false,
  hoverId: null, // slot under the pointer, so the tooltip follows each refresh
  lastData: null, // the last status payload, for the live-state indicator
  job: null, // the last known source job
  offline: false, // the last status poll failed
  lastUpdatedAt: null, // `updated_at` of the last reading seen
  lastChangeAt: Date.now(), // browser time when that reading (or a new job) arrived
  previewSrc: null, // the frame image currently shown
};
// Dashboard poll interval. Sources sample every 2 s by default, so a change
// reaches the screen within about 2 s + processing + 2 s, under R-04's 5 s.
const POLL_MS = 2000;
// A running source counts as stale after this long without a new reading
// (at least 10 s, or three sample intervals for slower sampling).
const STALE_MIN_S = 10;
// How long a slot that just changed status stays highlighted.
const CHANGE_MARK_MS = 2000;
const $ = (id) => document.getElementById(id);

function timeOnly(iso) {
  if (!iso) return "--:--:--";
  return new Date(iso).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

// Write text only when it changes, so live regions and alerts are not
// re-announced by screen readers on every poll.
function setText(element, text) {
  if (element.textContent !== text) element.textContent = text;
}

const SVG_NS = "http://www.w3.org/2000/svg";
const STATUSES = ["occupied", "vacant", "unknown"];
// A shape as well as a colour, so status reads without relying on colour alone.
const STATUS_GLYPHS = { occupied: "●", vacant: "○", unknown: "?" };
// The words people see; the data and exports keep vacant / occupied / unknown.
const STATUS_NAMES = { occupied: "Taken", vacant: "Free", unknown: "Not sure" };
const MODEL_ERROR_STATES = ["error", "needs_package", "unknown"];
const TONES = ["live", "warn", "error", "idle", "offline"];

function statusOf(item) {
  return STATUSES.includes(item.status) ? item.status : "unknown";
}

function setStatusClass(element, status) {
  element.classList.remove(...STATUSES.map((name) => `status-${name}`));
  element.classList.add(`status-${status}`);
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
  const parts = [item.id, STATUS_NAMES[statusOf(item)].toLowerCase()];
  const duration = statusDuration(item, readingTime);
  if (duration) parts.push(`for ${duration}`);
  if (item.raw_status && item.raw_status !== item.status)
    parts.push(`latest frame: ${(STATUS_NAMES[item.raw_status] || item.raw_status).toLowerCase()}, confirming`);
  if (item.reason) parts.push(item.reason);
  parts.push(`${Math.round((item.confidence ?? 0) * 100)}% confidence`);
  if (item.updated_at) parts.push(timeOnly(item.updated_at));
  return parts.join(" / ");
}

// R-06: every slot is filled in its status colour, carries a pinned ID chip,
// and shows its status, confidence and last-update time on hover or focus.
// The elements are kept between polls (keyed by slot id) and updated in place,
// so a status change fades to its new colour and the slot is marked once.
const mapEls = new Map();

function createSlotEls(id) {
  const shape = document.createElementNS(SVG_NS, "polygon");
  shape.setAttribute("class", "slot-shape");
  const hit = document.createElement("button");
  hit.type = "button";
  hit.className = "slot-hit";
  hit.addEventListener("pointerenter", (event) => showConfidence(id, event));
  hit.addEventListener("pointermove", moveConfidence);
  hit.addEventListener("pointerleave", hideConfidence);
  hit.addEventListener("focus", () => showConfidence(id));
  hit.addEventListener("blur", hideConfidence);
  const chip = document.createElement("span");
  chip.className = "slot-chip";
  chip.setAttribute("aria-hidden", "true");
  const glyph = document.createElement("b");
  return { shape, hit, chip, glyph, status: null, pointsKey: "", markTimer: null };
}

function removeSlotEls(els) {
  clearTimeout(els.markTimer);
  els.shape.remove();
  els.hit.remove();
  els.chip.remove();
}

// Mark a slot whose status just changed. Restarting the animation needs the
// class removed and a style flush before it is added again.
function markChanged(els) {
  for (const element of [els.shape, els.chip]) {
    element.classList.remove("is-changed");
    void element.getBoundingClientRect();
    element.classList.add("is-changed");
  }
  clearTimeout(els.markTimer);
  els.markTimer = setTimeout(() => {
    els.shape.classList.remove("is-changed");
    els.chip.classList.remove("is-changed");
  }, CHANGE_MARK_MS);
}

// Returns the ids of slots whose status changed since the last render.
function renderMap(data) {
  state.slots = data.slots || [];
  const layer = $("slotLayer");
  let shapes = layer.querySelector(".slot-shapes");
  if (!shapes) {
    shapes = document.createElementNS(SVG_NS, "svg");
    shapes.setAttribute("class", "slot-shapes");
    shapes.setAttribute("viewBox", "0 0 100 100");
    shapes.setAttribute("preserveAspectRatio", "none");
    shapes.setAttribute("aria-hidden", "true");
    layer.appendChild(shapes);
  }
  const seen = new Set();
  const changed = [];
  state.slots.forEach((item, index) => {
    const points = state.slotConfig?.[index]?.points || item.points;
    if (!points?.length) return;
    const status = statusOf(item);
    seen.add(item.id);
    let els = mapEls.get(item.id);
    if (!els) {
      els = createSlotEls(item.id);
      shapes.appendChild(els.shape);
      layer.append(els.hit, els.chip);
      mapEls.set(item.id, els);
    } else if (els.status !== status) {
      changed.push(item.id);
      markChanged(els);
    }
    if (els.status !== status) {
      setStatusClass(els.shape, status);
      setStatusClass(els.chip, status);
      els.status = status;
    }

    const pointsKey = JSON.stringify(points);
    if (els.pointsKey !== pointsKey) {
      els.pointsKey = pointsKey;
      els.shape.setAttribute("points", points.map(([x, y]) => `${x},${y}`).join(" "));
      els.hit.style.clipPath = `polygon(${points.map(([x, y]) => `${x}% ${y}%`).join(",")})`;
      // Pin the chip at the polygon's centre, kept far enough from the edges to stay readable.
      const centre = (axis) => points.reduce((sum, point) => sum + Number(point[axis]), 0) / points.length;
      els.chip.style.left = `${Math.min(95, Math.max(5, centre(0)))}%`;
      els.chip.style.top = `${Math.min(96, Math.max(4, centre(1)))}%`;
    }
    els.hit.setAttribute("aria-label", slotLabel(item));
    els.glyph.textContent = STATUS_GLYPHS[status];
    if (els.chip.firstChild?.nodeValue !== item.id || els.chip.childNodes.length !== 2) {
      els.chip.replaceChildren(item.id, els.glyph);
    }
  });
  for (const [id, els] of mapEls) {
    if (!seen.has(id)) {
      removeSlotEls(els);
      mapEls.delete(id);
    }
  }
  $("mapFootnote").textContent = `Polygon view / ${state.slots.length} slots`;
  // A still pointer gets no new pointer event when the reading changes:
  // refresh an open tooltip from the new data instead of leaving the old reading.
  if (state.hoverId !== null) {
    const hovered = state.slots.find((item) => item.id === state.hoverId);
    if (hovered) $("confidenceTooltip").textContent = slotLabel(hovered);
    else hideConfidence();
  }
  return changed;
}

function renderMapStamp(data, warning = "") {
  if (!data) return;
  const parts = [`Updated ${timeOnly(data.updated_at)}`];
  if (data.captured_at) parts.push(`captured ${timeOnly(data.captured_at)}`);
  else if (data.frame_time_s != null) parts.push(`video ${seconds(data.frame_time_s)}`);
  if (data.source === "demo") parts.push("demo data");
  if (warning) parts.unshift(warning);
  const stamp = $("mapStamp");
  setText(stamp, parts.join(" / "));
  stamp.classList.toggle("is-stale", Boolean(warning));
}

function showConfidence(id, event) {
  const item = state.slots.find((slot) => slot.id === id);
  if (!item) return;
  const tooltip = $("confidenceTooltip");
  state.hoverId = id;
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

const clampPercent = (value) => Math.round(Math.max(0, Math.min(100, value)) * 100) / 100;

function mapPoint(event) {
  const bounds = $("lotMap").getBoundingClientRect();
  return [
    clampPercent(((event.clientX - bounds.left) / bounds.width) * 100),
    clampPercent(((event.clientY - bounds.top) / bounds.height) * 100),
  ];
}

// --- Live state: what the dashboard is showing, said honestly ---
// One state drives the top-bar dot and text, the health card and the activity
// ring, so "live" only ever appears while a source is actually delivering readings.
function staleAfterS(job) {
  return Math.max(STALE_MIN_S, 3 * (Number(job.interval_s) || 0));
}

function liveState() {
  const data = state.lastData;
  if (state.offline) return { tone: "offline", text: "Reconnecting...", stale: true };
  if (!data) return { tone: "idle", text: "Connecting..." };
  if (state.editing) return { tone: "idle", text: "Paused while editing" };
  const model = data.model_status;
  if (data.source === "model error" || (model && MODEL_ERROR_STATES.includes(model.state)))
    return { tone: "error", text: "Model error" };
  if (data.source === "demo") return { tone: "warn", text: "Demo data" };
  if (data.source === "none")
    return model && ["downloading", "loading"].includes(model.state)
      ? { tone: "warn", text: "Model loading" }
      : { tone: "idle", text: "No frame yet" };
  if (!$("autoRefresh").checked) return { tone: "idle", text: "Auto refresh off" };
  const job = state.job;
  if (job && job.state === "starting") return { tone: "idle", text: "Starting source" };
  if (job && job.state === "running") {
    const quiet = Math.round((Date.now() - state.lastChangeAt) / 1000);
    if (quiet > staleAfterS(job)) return { tone: "warn", text: `No new reading for ${quiet} s`, stale: true };
    return { tone: "live", text: job.kind === "file" ? "Processing video" : "Live" };
  }
  return { tone: "idle", text: `Last reading ${timeOnly(data.updated_at)}` };
}

function renderLiveState() {
  const live = liveState();
  for (const id of ["liveDot", "healthPill", "activityPulse"]) {
    const element = $(id);
    element.classList.remove(...TONES.map((tone) => `tone-${tone}`));
    element.classList.add(`tone-${live.tone}`);
  }
  setText($("liveText"), live.text);
  setText($("healthPill"), live.text);
  $("lotMap").classList.toggle("is-stale", Boolean(live.stale));
  // The sign's LED only glows for a current, real reading: never for demo data,
  // no frame, a model error, a stale source or a lost connection.
  const source = state.lastData?.source;
  const current = !live.stale && !state.offline && source && !["demo", "none", "model error"].includes(source);
  $("stats").classList.toggle("is-dimmed", !current);
  renderMapStamp(state.lastData, live.stale ? live.text : "");
}

// Note when a new reading (or a new source) arrives, in browser time, so
// staleness never depends on the server's clock.
function trackFreshness(data) {
  if (data.updated_at !== state.lastUpdatedAt) {
    state.lastUpdatedAt = data.updated_at;
    state.lastChangeAt = Date.now();
  }
  if ("job" in data) {
    const startedAt = data.job?.started_at ?? null;
    if (startedAt !== (state.job?.started_at ?? null)) state.lastChangeAt = Date.now();
    state.job = data.job;
  }
}

const changedTime = (item) => Date.parse(item.changed_at) || 0;
const byId = new Intl.Collator(undefined, { numeric: true });

// Recent changes: every slot's current status and when it began, newest change
// first, so what just happened is at the top instead of somewhere in slot order.
function renderActivity(data, changed = []) {
  const readingTime = data.captured_at || data.updated_at;
  const slots = [...(data.slots || [])].sort(
    (a, b) => changedTime(b) - changedTime(a) || byId.compare(a.id, b.id),
  );
  const rows = slots.map((item) => {
    const status = statusOf(item);
    const row = document.createElement("div");
    row.className = "activity-row";
    row.classList.toggle("is-changed", changed.includes(item.id));
    const name = document.createElement("span");
    name.className = "activity-slot";
    const dot = document.createElement("i");
    dot.className = `status-dot dot-${status}`;
    name.append(dot, item.id);
    const label = document.createElement("span");
    label.className = "activity-status";
    const duration = statusDuration(item, readingTime);
    label.textContent = duration ? `${STATUS_NAMES[status]} · ${duration}` : STATUS_NAMES[status];
    const time = document.createElement("span");
    time.className = "activity-time";
    // When the status began; "since" when it may have begun before monitoring did.
    if (item.changed_at) {
      time.textContent = `${item.changed_since_start ? "since " : ""}${timeOnly(item.changed_at)}`;
    } else {
      time.textContent = timeOnly(item.updated_at ?? data.updated_at);
    }
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

// Health card: which per-slot decision is in use.
function classifierLabel(status) {
  if (!status || !status.enabled) return "detector only";
  if (status.state === "ready") return `slot classifier: ${status.kind}`;
  if (status.state === "error") return "slot classifier failed: detector only";
  return "no slot classifier: detector only";
}

// Explains, in plain words, why there are no real readings: no model installed
// (demo data) or a model that can't be used (an error, never demo data).
function renderModelAlert(data) {
  const alert = $("modelAlert");
  const status = data.model_status;
  let text = "";
  if (status && status.state === "missing") {
    text = `No detection model is installed, so the map shows demo data.${status.error ? ` Last attempt: ${status.error}` : ""} Install one under Settings > Detection model and resolution.`;
  } else if (status && MODEL_ERROR_STATES.includes(status.state)) {
    const reason = status.state === "needs_package" ? `${status.name} needs its Python package: ${status.message}` : status.message;
    text = `Detection model problem: ${reason}. No frames can be processed until it is fixed under Settings > Detection model and resolution.`;
  }
  setText(alert, text);
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
  setText(
    alert,
    moved
      ? `The camera may have moved: slots look about ${camera.shift_pct ?? "?"}% of the frame width out of place (${camera.streak} checks in a row). Check the layout with Edit layout; saving it makes the current frame the new reference.`
      : "",
  );
  const notes = {
    ok: `Camera check OK${camera.shift_pct != null ? ` (${camera.shift_pct}% shift)` : ""}`,
    suspect: `Camera check: possible shift of ${camera.shift_pct}% (${camera.streak} of 3 checks)`,
    unverified: `Camera check: can't verify (${camera.reason || "too few features"})`,
  };
  const note = notes[camera.state];
  $("mapFootnote").textContent = `Polygon view / ${state.slots.length} slots${note ? ` / ${note}` : ""}`;
}

function render(data) {
  state.lastData = data;
  state.readingTime = data.captured_at || data.updated_at;
  const occupied = (data.slots || []).filter(
    (item) => item.status === "occupied",
  ).length;
  setText($("freeCount"), String(data.free ?? "--"));
  setText($("totalCount"), String(data.total ?? "--"));
  setText($("ledTotal"), String(data.total ?? "--"));
  setText($("freeUnit"), data.free === 1 ? "space" : "spaces");
  setText($("topbarFreeCount"), `${data.free ?? "--"} of ${data.total ?? "--"}`);
  setText($("occupiedCount"), String(occupied));
  setText($("unknownCount"), String(data.unknown ?? 0));
  // The live state already says "Demo data", "No frame yet" or "Model error";
  // the source name is shown only when there is a real one.
  const realSource = !["demo", "none", "model error"].includes(data.source);
  $("sourceLabel").textContent = realSource ? data.source : "";
  $("sourceLabel").hidden = !realSource;
  $("sourceDivider").hidden = !realSource;
  $("modelLabel").textContent = modelLabel(data);
  renderModelAlert(data);
  $("classifierLabel").textContent = classifierLabel(data.classifier);
  state.lotName = data.lot_name || state.lotName;
  state.reviewThreshold = data.review_threshold ?? state.reviewThreshold;
  setText($("lotNameLabel"), state.lotName);
  document.title = `${state.lotName} · SlotSight`;
  const changed = renderMap(data);
  renderActivity(data, changed);
  renderCamera(data);
  if ("job" in data) renderJob(data.job);
  // The frame is a large data URL: only hand it to the image when it changes.
  if (data.preview_image && data.preview_image !== state.previewSrc) {
    state.previewSrc = data.preview_image;
    $("mapPreviewImage").src = data.preview_image;
    $("lotMap").classList.add("has-preview");
  }
  trackFreshness(data);
  renderLiveState();
}

// Size the map to the frame so it is shown undistorted (see .has-ratio).
$("mapPreviewImage").addEventListener("load", (event) => {
  const { naturalWidth: width, naturalHeight: height } = event.currentTarget;
  if (!width || !height) return;
  const map = $("lotMap");
  map.style.setProperty("--frame-ratio", String(width / height));
  map.classList.add("has-ratio");
});

async function refresh() {
  // Unsaved layout edits must not be replaced by the saved layout.
  if (state.editing) return;
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`Status ${response.status}`);
    const data = await response.json();
    if (state.editing) return;
    state.offline = false;
    state.slotConfig = data.slots.map((item) => ({
      id: item.id,
      points: item.points,
    }));
    state.lotName = data.lot_name || state.lotName;
    state.reviewThreshold = data.review_threshold ?? state.reviewThreshold;
    render(data);
  } catch (error) {
    // Shown where the live state is shown (top bar, health card, map), not in a form.
    state.offline = true;
    renderLiveState();
  }
}

function setTimer() {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  if ($("autoRefresh").checked && !state.editing) state.timer = setInterval(refresh, POLL_MS);
  renderLiveState();
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
$("refreshInterval").textContent = `${POLL_MS / 1000} s`;

// --- Map toast: short confirmations, with Undo where a change can be taken back ---
let toastTimer = null;
let toastUndo = null;

function showToast(text, onUndo = null) {
  $("mapToastText").textContent = text;
  toastUndo = onUndo;
  $("mapToastUndo").hidden = !onUndo;
  $("mapToast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(hideToast, onUndo ? 8000 : 3000);
}

function hideToast() {
  clearTimeout(toastTimer);
  $("mapToast").hidden = true;
  toastUndo = null;
}

$("mapToastUndo").addEventListener("click", () => {
  const undo = toastUndo;
  hideToast();
  if (undo) undo();
});

function setMapMessage(text, isError = false) {
  const message = $("mapMessage");
  message.textContent = text;
  message.classList.toggle("is-error", isError);
}

// --- Layout editor ---
// Edits stay in the browser until "Save layout". Every change can be undone
// (button or Ctrl+Z); Cancel or Esc discards the edit, with an Undo toast.
const editor = {
  undo: [], // layouts before each change, newest last
  els: [], // per slot: its outline, label, delete button and corner handles
  drag: null, // the corner being dragged
  drawing: null, // the slot being drawn
};
const UNDO_LIMIT = 50;

function cloneLayout() {
  return { slotConfig: structuredClone(state.slotConfig), slots: structuredClone(state.slots) };
}

function updateUndoButton() {
  $("undoLayout").disabled = editor.undo.length === 0;
}

function pushUndo(layout = cloneLayout()) {
  editor.undo.push(layout);
  if (editor.undo.length > UNDO_LIMIT) editor.undo.shift();
  hideToast(); // a toast's Undo would otherwise undo this newer change
  updateUndoButton();
}

function restoreLayout(layout) {
  state.slotConfig = layout.slotConfig;
  state.slots = layout.slots;
  renderMap({ slots: state.slots });
  renderEditor();
}

function undoLayout() {
  const previous = editor.undo.pop();
  if (!previous) return;
  hideToast();
  restoreLayout(previous);
  updateUndoButton();
}

function slotBounds(points) {
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  return { left: Math.min(...xs), top: Math.min(...ys), right: Math.max(...xs) };
}

function renderEditor() {
  const layer = $("editorLayer");
  // Built with DOM calls: slot ids come from config and must never be parsed as HTML.
  const shapes = document.createElementNS(SVG_NS, "svg");
  shapes.setAttribute("class", "editor-shapes");
  shapes.setAttribute("viewBox", "0 0 100 100");
  shapes.setAttribute("preserveAspectRatio", "none");
  shapes.setAttribute("aria-hidden", "true");
  const nodes = [shapes];
  editor.els = state.slotConfig.map((slot, index) => {
    const polygon = document.createElementNS(SVG_NS, "polygon");
    shapes.appendChild(polygon);
    const label = document.createElement("span");
    label.className = "editor-label";
    label.textContent = slot.id;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "editor-delete";
    remove.dataset.deleteSlot = index;
    remove.title = `Delete ${slot.id}`;
    remove.setAttribute("aria-label", `Delete ${slot.id}`);
    remove.textContent = "×";
    const handles = slot.points.map((point, pointIndex) => {
      const handle = document.createElement("button");
      handle.type = "button";
      handle.className = "editor-handle";
      handle.dataset.slotIndex = index;
      handle.dataset.pointIndex = pointIndex;
      handle.title = `Move ${slot.id} corner ${pointIndex + 1}`;
      handle.setAttribute("aria-label", `Move ${slot.id} corner ${pointIndex + 1}. Arrow keys nudge it.`);
      return handle;
    });
    nodes.push(label, remove, ...handles);
    return { polygon, label, remove, handles };
  });
  if (editor.drawing) nodes.push(editor.drawing.preview);
  layer.replaceChildren(...nodes);
  state.slotConfig.forEach((_, index) => positionEditorSlot(index));
}

// Move one slot's outline, label, delete button and handles to its points.
function positionEditorSlot(index) {
  const slot = state.slotConfig[index];
  const els = editor.els[index];
  if (!slot || !els) return;
  els.polygon.setAttribute("points", slot.points.map(([x, y]) => `${x},${y}`).join(" "));
  const box = slotBounds(slot.points);
  els.label.style.left = `${box.left}%`;
  els.label.style.top = `${box.top}%`;
  els.remove.style.left = `${box.right}%`;
  els.remove.style.top = `${box.top}%`;
  slot.points.forEach(([x, y], pointIndex) => {
    const handle = els.handles[pointIndex];
    if (!handle) return;
    handle.style.left = `${x}%`;
    handle.style.top = `${y}%`;
  });
}

function setEditMode(enabled) {
  state.editing = enabled;
  $("lotMap").classList.toggle("is-editing", enabled);
  $("editorLayer").hidden = !enabled;
  $("editLayout").textContent = enabled ? "Save layout" : "Edit layout";
  $("editLayout").classList.toggle("button-primary", enabled); // the action that ends the edit
  $("editLayout").classList.toggle("button-secondary", !enabled);
  $("cancelLayout").hidden = !enabled;
  $("undoLayout").hidden = !enabled;
  $("editorHint").hidden = !enabled;
  setMapMessage(""); // a save error belongs to the edit it came from
  if (enabled) {
    editor.undo = [];
    updateUndoButton();
    hideToast();
    hideConfidence();
    renderEditor();
  } else {
    cancelGesture();
    editor.els = [];
    $("editorLayer").replaceChildren();
  }
  setTimer(); // polling pauses while editing
}

function cancelGesture() {
  if (editor.drag) endHandleDrag(false);
  if (editor.drawing) {
    editor.drawing.preview.remove();
    editor.drawing = null;
  }
}

// Leave edit mode without saving. The discarded edit can be brought back from the toast.
function cancelEditing() {
  const discarded = { ...cloneLayout(), undo: editor.undo };
  const hadChanges = editor.undo.length > 0;
  setEditMode(false);
  refresh(); // show the saved layout again
  if (hadChanges)
    showToast("Layout changes discarded.", () => {
      setEditMode(true);
      editor.undo = discarded.undo;
      restoreLayout(discarded);
      updateUndoButton();
    });
}

async function saveVisualConfig() {
  const response = await fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ slots: state.slotConfig }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Could not save layout.");
  setEditMode(false);
  await refresh();
  showToast("Layout saved.");
}

$("editLayout").addEventListener("click", async () => {
  if (!state.editing) {
    setEditMode(true);
    return;
  }
  const button = $("editLayout");
  button.disabled = true;
  setMapMessage("Saving layout...");
  try {
    await saveVisualConfig();
    setMapMessage("");
  } catch (error) {
    setMapMessage(error.message, true);
  } finally {
    button.disabled = false;
  }
});
$("cancelLayout").addEventListener("click", cancelEditing);
$("undoLayout").addEventListener("click", undoLayout);

$("editorLayer").addEventListener("click", (event) => {
  const button = event.target.closest("[data-delete-slot]");
  if (!button) return;
  const index = Number(button.dataset.deleteSlot);
  const id = state.slotConfig[index]?.id;
  if (id == null) return;
  pushUndo();
  state.slotConfig.splice(index, 1);
  state.slots.splice(index, 1);
  renderMap({ slots: state.slots });
  renderEditor();
  showToast(`Deleted ${id}.`, undoLayout);
});

// Corner drag: the corner keeps the offset from where it was grabbed, follows the
// pointer 1:1 (captured, so it keeps tracking outside the map), and only its own
// slot is redrawn while it moves.
// Neighbouring slots share corners, so the 28 px targets can overlap: take the
// handle whose centre is nearest the pointer, not whichever is on top.
function nearestHandle(event, fallback) {
  let best = fallback;
  let bestDistance = Infinity;
  for (const els of editor.els) {
    for (const handle of els.handles) {
      const box = handle.getBoundingClientRect();
      const distance = Math.hypot(event.clientX - (box.left + box.width / 2), event.clientY - (box.top + box.height / 2));
      if (distance <= box.width / 2 && distance < bestDistance) {
        best = handle;
        bestDistance = distance;
      }
    }
  }
  return best;
}

$("editorLayer").addEventListener("pointerdown", (event) => {
  const target = event.target.closest("[data-point-index]");
  if (!target || event.button !== 0) return;
  const handle = nearestHandle(event, target);
  const slotIndex = Number(handle.dataset.slotIndex);
  const pointIndex = Number(handle.dataset.pointIndex);
  const point = state.slotConfig[slotIndex]?.points[pointIndex];
  if (!point) return;
  const at = mapPoint(event);
  editor.drag = {
    handle,
    slotIndex,
    pointIndex,
    pointerId: event.pointerId,
    offset: [point[0] - at[0], point[1] - at[1]],
    before: cloneLayout(),
    moved: false,
  };
  handle.setPointerCapture(event.pointerId);
  handle.classList.add("is-dragging");
  event.preventDefault();
});
$("editorLayer").addEventListener("pointermove", (event) => {
  const drag = editor.drag;
  if (!drag || event.pointerId !== drag.pointerId) return;
  const at = mapPoint(event);
  state.slotConfig[drag.slotIndex].points[drag.pointIndex] = [
    clampPercent(at[0] + drag.offset[0]),
    clampPercent(at[1] + drag.offset[1]),
  ];
  drag.moved = true;
  positionEditorSlot(drag.slotIndex);
});

function endHandleDrag(commit) {
  const drag = editor.drag;
  if (!drag) return;
  editor.drag = null;
  drag.handle.classList.remove("is-dragging");
  if (drag.handle.hasPointerCapture?.(drag.pointerId)) drag.handle.releasePointerCapture(drag.pointerId);
  if (!drag.moved) return;
  if (commit) {
    pushUndo(drag.before);
    renderMap({ slots: state.slots });
  } else {
    state.slotConfig = drag.before.slotConfig;
    state.slots = drag.before.slots;
    renderMap({ slots: state.slots });
    positionEditorSlot(drag.slotIndex);
  }
}
$("editorLayer").addEventListener("pointerup", () => endHandleDrag(true));
$("editorLayer").addEventListener("pointercancel", () => endHandleDrag(false));

// Keyboard: arrow keys nudge a focused corner (0.5 % of the frame, 2 % with Shift).
const NUDGE = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
$("editorLayer").addEventListener("keydown", (event) => {
  const handle = event.target.closest("[data-point-index]");
  const direction = NUDGE[event.key];
  if (!handle || !direction) return;
  event.preventDefault();
  const slotIndex = Number(handle.dataset.slotIndex);
  const pointIndex = Number(handle.dataset.pointIndex);
  const point = state.slotConfig[slotIndex]?.points[pointIndex];
  if (!point) return;
  if (!event.repeat) pushUndo(); // one undo step per key press, however long it is held
  const step = event.shiftKey ? 2 : 0.5;
  state.slotConfig[slotIndex].points[pointIndex] = [
    clampPercent(point[0] + direction[0] * step),
    clampPercent(point[1] + direction[1] * step),
  ];
  positionEditorSlot(slotIndex);
  renderMap({ slots: state.slots });
});

$("clearLayout").addEventListener("click", () => {
  // No confirmation dialog: nothing is saved until Save layout, and Undo brings it back.
  if (!state.editing) setEditMode(true);
  if (!state.slotConfig.length) return;
  pushUndo();
  state.slotConfig = [];
  state.slots = [];
  renderMap({ slots: [] });
  renderEditor();
  showToast("Cleared every slot (not saved yet).", undoLayout);
});

// Drawing a new slot: drag on an empty part of the frame.
$("lotMap").addEventListener("pointerdown", (event) => {
  if (
    !state.editing ||
    event.button !== 0 ||
    event.target.closest("button") ||
    event.target.closest(".map-toast")
  )
    return;
  editor.drawing = { start: mapPoint(event), preview: document.createElement("div"), pointerId: event.pointerId };
  editor.drawing.preview.className = "draw-preview";
  $("editorLayer").appendChild(editor.drawing.preview);
  event.currentTarget.setPointerCapture(event.pointerId);
});
$("lotMap").addEventListener("pointermove", (event) => {
  const drawing = editor.drawing;
  if (!drawing || event.pointerId !== drawing.pointerId) return;
  const end = mapPoint(event);
  const left = Math.min(drawing.start[0], end[0]);
  const top = Math.min(drawing.start[1], end[1]);
  drawing.preview.style.left = `${left}%`;
  drawing.preview.style.top = `${top}%`;
  drawing.preview.style.width = `${Math.abs(end[0] - drawing.start[0])}%`;
  drawing.preview.style.height = `${Math.abs(end[1] - drawing.start[1])}%`;
});
$("lotMap").addEventListener("pointerup", (event) => {
  const drawing = editor.drawing;
  if (!drawing || event.pointerId !== drawing.pointerId) return;
  editor.drawing = null;
  drawing.preview.remove();
  const end = mapPoint(event);
  const left = Math.min(drawing.start[0], end[0]);
  const right = Math.max(drawing.start[0], end[0]);
  const top = Math.min(drawing.start[1], end[1]);
  const bottom = Math.max(drawing.start[1], end[1]);
  if (right - left < 2 || bottom - top < 2) return;
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
  pushUndo();
  state.slotConfig.push({ id, points });
  state.slots.push({ id, status: "unknown", confidence: 0, points });
  renderMap({ slots: state.slots });
  renderEditor();
});
$("lotMap").addEventListener("pointercancel", () => {
  if (!editor.drawing) return;
  editor.drawing.preview.remove();
  editor.drawing = null;
});

// Esc cancels the gesture in progress, or else the whole edit (with an Undo toast).
// Ctrl+Z / Cmd+Z undoes the last change while editing.
document.addEventListener("keydown", (event) => {
  if (!state.editing || $("settingsDialog").open) return;
  if (event.key === "Escape") {
    event.preventDefault();
    if (editor.drag || editor.drawing) cancelGesture();
    else cancelEditing();
  } else if ((event.ctrlKey || event.metaKey) && !event.shiftKey && event.key.toLowerCase() === "z") {
    if (event.target.closest("input, textarea, select")) return;
    event.preventDefault();
    undoLayout();
  }
});

// --- Upload: pick a file, or drop one on the upload zone ---
const uploadZone = $("uploadZone");
const UPLOAD_PROMPT = uploadZone.querySelector("strong").textContent;

function showChosenFile(file) {
  uploadZone.querySelector("strong").textContent = file ? file.name : UPLOAD_PROMPT;
}

const draggingFiles = (event) => Boolean(event.dataTransfer?.types?.includes("Files"));

for (const type of ["dragenter", "dragover"]) {
  uploadZone.addEventListener(type, (event) => {
    if (!draggingFiles(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    uploadZone.classList.add("is-dragover");
  });
}
uploadZone.addEventListener("dragleave", (event) => {
  if (!uploadZone.contains(event.relatedTarget)) uploadZone.classList.remove("is-dragover");
});
uploadZone.addEventListener("drop", (event) => {
  if (!draggingFiles(event)) return;
  event.preventDefault();
  uploadZone.classList.remove("is-dragover");
  const file = event.dataTransfer.files[0];
  if (!file) return;
  // Some video types have no MIME type in the browser; the server checks every file.
  if (file.type && !/^(image|video)\//.test(file.type)) {
    $("formMessage").textContent = `${file.name} is not an image or video.`;
    return;
  }
  const chosen = new DataTransfer();
  chosen.items.add(file);
  $("sourceFile").files = chosen.files;
  showChosenFile(file);
  $("formMessage").textContent = "File ready. Press Check this file to read it.";
});
// A file dropped anywhere else would make the browser open it and leave the dashboard.
window.addEventListener("dragover", (event) => {
  if (!draggingFiles(event) || uploadZone.contains(event.target)) return;
  event.preventDefault();
  event.dataTransfer.dropEffect = "none";
});
window.addEventListener("drop", (event) => {
  if (draggingFiles(event) && !uploadZone.contains(event.target)) event.preventDefault();
});

$("sourceFile").addEventListener("change", (event) => {
  showChosenFile(event.target.files[0]);
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

// --- Detection model picker (Settings dialog) ---
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
  const active = data.models.find((model) => model.active);
  $("inferenceSizeNote").textContent =
    active && !active.supports_inference_size ? `(not used by ${active.name}: it has a fixed input size)` : "";
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
  button.className = "button button-secondary button-sm";
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
    if (data.job && data.job.state === "running" && $("settingsDialog").open)
      modelTimer = setTimeout(loadModels, 1000);
  } catch (error) {
    $("modelMessage").textContent = "Could not list the detection models.";
  }
}

// --- Settings dialog ---
// One rule for the whole dialog: each control is saved as soon as it changes
// (POST /api/settings, which never touches the slot layout). The polygon JSON is
// text that has to be valid first, so it has its own Apply button.
const settingsForm = { saved: {}, statusTimer: null, thresholdTimer: null, seq: 0 };

function settingsStatus(text, isError = false) {
  const status = $("settingsStatus");
  clearTimeout(settingsForm.statusTimer);
  status.textContent = text;
  status.classList.toggle("is-error", isError);
  if (text && !isError) settingsForm.statusTimer = setTimeout(() => (status.textContent = ""), 2500);
}

function showSettings(data) {
  settingsForm.saved = {
    lot_name: data.lot_name || state.lotName,
    review_threshold: data.review_threshold ?? state.reviewThreshold,
    inference_size: data.inference_size ?? 640,
    slot_classifier: data.classifier?.enabled ?? true,
  };
  const saved = settingsForm.saved;
  $("lotNameInput").value = saved.lot_name;
  $("reviewThresholdInput").value = Math.round(saved.review_threshold * 100);
  $("reviewThresholdValue").textContent = `${$("reviewThresholdInput").value}%`;
  $("inferenceSizeInput").value = String(saved.inference_size);
  $("classifierInput").checked = saved.slot_classifier;
  $("classifierNote").textContent =
    data.classifier?.state === "ready" ? `(${data.classifier.kind})` : "(no weights: the detector decides alone)";
}

// Save one or more settings. On failure the control goes back to the saved value
// (the lot name keeps what was typed, so it can be corrected).
async function saveSettings(updates, revert) {
  const seq = ++settingsForm.seq;
  settingsStatus("Saving...");
  try {
    const response = await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updates),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not save the setting.");
    Object.assign(settingsForm.saved, updates);
    if (!state.editing) render(data);
    if (seq === settingsForm.seq) settingsStatus("Saved");
  } catch (error) {
    if (revert) revert();
    settingsStatus(error.message, true);
  }
}

$("openSettings").addEventListener("click", async () => {
  let data = state.lastData;
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (response.ok) data = await response.json();
  } catch (error) {
    // Fall back to the last poll; saving will report a connection problem.
  }
  if (!data) return;
  showSettings(data);
  const slots = data.slots.map((item, index) => ({
    id: item.id,
    points: state.slotConfig?.[index]?.points || item.points,
  }));
  $("configInput").value = JSON.stringify(slots, null, 2);
  $("jsonMessage").textContent = "";
  $("modelMessage").textContent = "";
  settingsStatus("");
  // When the model needs attention, open its section straight away.
  const model = data.model_status;
  if (model && (model.state === "missing" || MODEL_ERROR_STATES.includes(model.state))) $("detectionGroup").open = true;
  $("settingsDialog").showModal();
  loadModels();
});

// Appearance: System, Light or Dark (theme.js applies it and keeps it in this browser).
const themeButtons = $("themeChoice").querySelectorAll("[data-theme-choice]");
function showThemeChoice(choice) {
  themeButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.themeChoice === choice)));
}
themeButtons.forEach((button) => button.addEventListener("click", () => window.slotsightTheme.set(button.dataset.themeChoice)));
window.slotsightTheme.onChange(showThemeChoice);
showThemeChoice(window.slotsightTheme.choice());

$("closeSettings").addEventListener("click", () => $("settingsDialog").close());
$("doneSettings").addEventListener("click", () => $("settingsDialog").close());
$("settingsDialog").addEventListener("close", () => clearTimeout(modelTimer));

function saveLotName() {
  const value = $("lotNameInput").value.trim();
  if (value === settingsForm.saved.lot_name) return;
  saveSettings({ lot_name: value });
}
$("lotNameInput").addEventListener("change", saveLotName);
$("lotNameForm").addEventListener("submit", (event) => {
  event.preventDefault(); // Enter saves the name and keeps the dialog open
  saveLotName();
});

// The slider previews on the map while it moves; the value is saved once it rests.
$("reviewThresholdInput").addEventListener("input", (event) => {
  $("reviewThresholdValue").textContent = `${event.target.value}%`;
  clearTimeout(settingsForm.thresholdTimer);
  const threshold = Number(event.target.value) / 100;
  settingsForm.thresholdTimer = setTimeout(() => {
    const previous = settingsForm.saved.review_threshold;
    saveSettings({ review_threshold: threshold }, () => {
      $("reviewThresholdInput").value = Math.round(previous * 100);
      $("reviewThresholdValue").textContent = `${$("reviewThresholdInput").value}%`;
    });
  }, 180);
});

$("classifierInput").addEventListener("change", (event) => {
  const previous = settingsForm.saved.slot_classifier;
  saveSettings({ slot_classifier: event.target.checked }, () => (event.target.checked = previous));
});

$("inferenceSizeInput").addEventListener("change", (event) => {
  const previous = settingsForm.saved.inference_size;
  saveSettings({ inference_size: Number(event.target.value) }, () => (event.target.value = String(previous)));
});

$("applyJson").addEventListener("click", async () => {
  const message = $("jsonMessage");
  message.classList.remove("is-error");
  let slots;
  try {
    slots = JSON.parse($("configInput").value);
  } catch (error) {
    message.textContent = `Not valid JSON: ${error.message}`;
    message.classList.add("is-error");
    return;
  }
  message.textContent = "Applying...";
  try {
    const response = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ slots }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not apply the outlines.");
    state.slotConfig = slots;
    message.textContent = data.message;
    refresh();
  } catch (error) {
    message.textContent = error.message;
    message.classList.add("is-error");
  }
});

// --- Top bar: the free count joins the bar once the summary has scrolled under
// it, and a hairline appears only while content runs underneath ---
const topbar = $("topbar");
if ("IntersectionObserver" in window) {
  new IntersectionObserver(
    ([entry]) => topbar.classList.toggle("show-free", !entry.isIntersecting && entry.boundingClientRect.top < 0),
    { rootMargin: "-64px 0px 0px 0px" },
  ).observe($("stats"));
}
const markScrolled = () => topbar.classList.toggle("is-scrolled", window.scrollY > 4);
window.addEventListener("scroll", markScrolled, { passive: true });
markScrolled();

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
    emptyChart(box, "No readings in this range yet. Check a video or connect a live camera.");
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
    const label = `${hourLabel(hour.hour)}-${hourLabel((hour.hour + 1) % 24)}: ${Math.round(hour.occupancy * 100)}% taken${peaks.has(hour.hour) ? " (peak)" : ""}, ${hour.frames} frames`;
    hit.setAttribute("aria-label", label);
    svgEl("path", { d: roundedTopBar(center - barWidth / 2, top, barWidth, barHeight), class: `chart-bar${peaks.has(hour.hour) ? " is-peak" : ""}` }, group);
    if (hour.hour === data.peak_hours[0]) {
      // Only the busiest hour carries a value; the others are listed below the chart.
      svgEl("text", { x: center, y: top - 5, "text-anchor": "middle", class: "chart-label" }, group).textContent = `${Math.round(hour.occupancy * 100)}%`;
    }
    const lines = [`${Math.round(hour.occupancy * 100)}% taken`, `${hourLabel(hour.hour)}-${hourLabel((hour.hour + 1) % 24)}${peaks.has(hour.hour) ? " / peak" : ""}`, `${hour.frames} frames`];
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
    tableOf("Free slots per interval", ["Interval", "Free (avg)", "Lowest", "Not sure", "Frames"], series),
    tableOf("Occupancy by hour", ["Hour", "Taken", "", "Frames"], hours),
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
