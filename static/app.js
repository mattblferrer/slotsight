const state = {
  slots: [],
  slotConfig: [],
  lotName: "Lot 01",
  reviewThreshold: 0.7,
  timer: null,
  editing: false,
};
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

function renderMap(data) {
  state.slots = data.slots || [];
  const layer = $("slotLayer");
  layer.innerHTML = "";
  state.slots.forEach((item, index) => {
    const points = state.slotConfig?.[index]?.points || item.points;
    if (!points?.length) return;
    const hit = document.createElement("button");
    hit.type = "button";
    hit.className = "slot-hit";
    hit.style.clipPath = `polygon(${points.map(([x, y]) => `${x}% ${y}%`).join(",")})`;
    hit.setAttribute(
      "aria-label",
      `${item.id}, ${item.status}, ${Math.round(item.confidence * 100)}% confidence`,
    );
    hit.addEventListener("pointerenter", (event) =>
      showConfidence(item, event),
    );
    hit.addEventListener("pointermove", moveConfidence);
    hit.addEventListener("pointerleave", hideConfidence);
    layer.appendChild(hit);
  });
  $("mapFootnote").textContent = `Polygon view / ${state.slots.length} slots`;
  if (state.editing) renderEditor();
}

function showConfidence(item, event) {
  const tooltip = $("confidenceTooltip");
  tooltip.textContent = `${item.id} / ${item.status} / ${Math.round(item.confidence * 100)}% confidence`;
  tooltip.hidden = false;
  moveConfidence(event);
}

function moveConfidence(event) {
  const map = $("lotMap").getBoundingClientRect();
  const tooltip = $("confidenceTooltip");
  tooltip.style.left = `${Math.min(map.width - 190, Math.max(10, event.clientX - map.left + 12))}px`;
  tooltip.style.top = `${Math.min(map.height - 34, Math.max(10, event.clientY - map.top - 36))}px`;
}

function hideConfidence() {
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
  const colors = {
    occupied: "dot-occupied",
    vacant: "dot-vacant",
    unknown: "dot-unknown",
  };
  $("activityList").innerHTML = (data.slots || [])
    .map(
      (item) => `
    <div class="activity-row"><span><i class="status-dot ${colors[item.status] || "dot-unknown"}"></i>${item.id}</span><span class="activity-status">${item.status}</span><span class="activity-time">${timeOnly(data.updated_at)}</span></div>
  `,
    )
    .join("");
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
  $("modelLabel").textContent = data.model === "YOLO" ? "YOLO live" : "Demo";
  state.lotName = data.lot_name || state.lotName;
  state.reviewThreshold = data.review_threshold ?? state.reviewThreshold;
  $("lotNameLabel").textContent = state.lotName;
  $("lotNameHeading").textContent = state.lotName;
  renderMap(data);
  renderActivity(data);
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
  if ($("autoRefresh").checked) state.timer = setInterval(refresh, 5000);
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
    state.slots.push({ id, status: "vacant", confidence: 1, points });
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
  const response = await fetch("/api/process", {
    method: "POST",
    body: formData,
  });
  const data = await response.json();
  if (response.ok) {
    render(data);
    message.textContent = file
      ? "Reading logged at " + timeOnly(data.updated_at)
      : "Showing the latest reading (nothing new logged).";
  } else message.textContent = data.error || "Processing failed.";
});

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
  $("configDialog").showModal();
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
