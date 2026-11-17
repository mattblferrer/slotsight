// Label page: mark each slot of each extracted frame as occupied or vacant (and
// partly hidden), saved to eval/labels.csv for the evaluation harness.
// Slots start unset: the model is never used to pre-fill labels, so they stay
// independent of what is being evaluated.
const $ = (id) => document.getElementById(id);
const SVG_NS = "http://www.w3.org/2000/svg";
const CYCLE = [null, "occupied", "vacant"];
const GLYPHS = { occupied: "●", vacant: "○" };

const label = {
  frames: [],
  slots: [],
  conditions: [],
  index: -1,
  marks: {}, // slot id -> { status: "occupied" | "vacant" | null, occluded: bool }
  dirty: false,
  confirmLeave: false,
  lastCondition: "",
};

function message(text) {
  $("labelMessage").textContent = text;
}

function frameUrl(path, frame) {
  return `${path}?frame=${encodeURIComponent(frame)}`;
}

function setDirty(value) {
  label.dirty = value;
  label.confirmLeave = false;
  $("saveNext").classList.toggle("is-dirty", value);
}

function centre(points, axis) {
  return points.reduce((sum, point) => sum + Number(point[axis]), 0) / points.length;
}

function renderOverlay() {
  const overlay = $("labelOverlay");
  const chips = $("labelChips");
  overlay.replaceChildren();
  chips.replaceChildren();
  for (const slot of label.slots) {
    const mark = label.marks[slot.id];
    const state = mark.status || "unset";
    const polygon = document.createElementNS(SVG_NS, "polygon");
    polygon.setAttribute("points", slot.points.map(([x, y]) => `${x},${y}`).join(" "));
    polygon.setAttribute("class", `label-shape status-${state}${mark.occluded ? " is-hidden-part" : ""}`);
    polygon.addEventListener("click", (event) => {
      if (event.shiftKey) toggleOccluded(slot.id);
      else cycle(slot.id);
    });
    overlay.appendChild(polygon);

    const chip = document.createElement("span");
    chip.className = `slot-chip label-chip status-${state === "unset" ? "unknown" : state}`;
    chip.style.left = `${Math.min(95, Math.max(5, centre(slot.points, 0)))}%`;
    chip.style.top = `${Math.min(96, Math.max(4, centre(slot.points, 1)))}%`;
    const glyph = document.createElement("b");
    glyph.textContent = `${GLYPHS[mark.status] || "?"}${mark.occluded ? " H" : ""}`;
    chip.append(slot.id, glyph);
    chips.appendChild(chip);
  }
}

function renderSlotList() {
  const rows = label.slots.map((slot) => {
    const mark = label.marks[slot.id];
    const row = document.createElement("div");
    row.className = "label-row";
    const name = document.createElement("strong");
    name.textContent = slot.id;
    const choices = document.createElement("div");
    choices.className = "label-choices";
    choices.setAttribute("role", "group");
    choices.setAttribute("aria-label", `${slot.id} status`);
    for (const status of ["occupied", "vacant"]) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = status;
      button.className = `label-choice choice-${status}`;
      button.setAttribute("aria-pressed", String(mark.status === status));
      button.addEventListener("click", () => setStatus(slot.id, mark.status === status ? null : status));
      choices.appendChild(button);
    }
    const hidden = document.createElement("label");
    hidden.className = "label-hidden";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = mark.occluded;
    box.addEventListener("change", () => toggleOccluded(slot.id));
    hidden.append(box, "partly hidden");
    row.append(name, choices, hidden);
    return row;
  });
  $("labelSlots").replaceChildren(...rows);
}

function render() {
  renderOverlay();
  renderSlotList();
  const done = label.frames.filter((frame) => frame.complete).length;
  const set = Object.values(label.marks).filter((mark) => mark.status).length;
  $("labelProgress").textContent = `Frame ${label.index + 1} of ${label.frames.length} / ${done} fully labeled / this frame: ${set} of ${label.slots.length} slots set`;
}

function setStatus(id, status) {
  label.marks[id].status = status;
  setDirty(true);
  render();
}

function cycle(id) {
  const current = CYCLE.indexOf(label.marks[id].status);
  setStatus(id, CYCLE[(current + 1) % CYCLE.length]);
}

function toggleOccluded(id) {
  label.marks[id].occluded = !label.marks[id].occluded;
  setDirty(true);
  render();
}

function setUnset(status) {
  for (const mark of Object.values(label.marks)) if (!mark.status) mark.status = status;
  setDirty(true);
  render();
}

async function show(index) {
  if (index < 0 || index >= label.frames.length) return;
  if (label.dirty && !label.confirmLeave) {
    label.confirmLeave = true;
    message("This frame has unsaved changes. Press again to leave without saving, or Save & next.");
    return;
  }
  label.index = index;
  const frame = label.frames[index];
  const response = await fetch(frameUrl("/api/label/frame", frame.frame), { cache: "no-store" });
  const data = await response.json();
  if (!response.ok) {
    message(data.error || "Could not load the frame.");
    return;
  }
  label.marks = {};
  for (const slot of label.slots) label.marks[slot.id] = { status: null, occluded: false };
  for (const row of data.labels) {
    if (label.marks[row.slot_id]) label.marks[row.slot_id] = { status: row.true_status, occluded: row.occluded === "1" };
  }
  const condition = data.labels[0]?.condition || frame.condition || data.session?.condition || label.lastCondition || label.conditions[0];
  $("labelCondition").value = condition;
  $("frameName").textContent = frame.frame.split("/").pop();
  $("frameSession").textContent = frame.frame.split("/").slice(0, -1).join(" / ");
  $("labelImage").src = frameUrl("/api/label/image", frame.frame);
  setDirty(false);
  message(frame.labeled ? "Already labeled: change anything and save again to replace it." : "");
  render();
}

function nextUnlabeled(from) {
  for (let step = 1; step <= label.frames.length; step += 1) {
    const index = (from + step) % label.frames.length;
    if (!label.frames[index].complete) return index;
  }
  return -1;
}

async function save() {
  const frame = label.frames[label.index];
  const condition = $("labelCondition").value;
  const slots = Object.entries(label.marks)
    .filter(([, mark]) => mark.status)
    .map(([slot_id, mark]) => ({ slot_id, true_status: mark.status, occluded: mark.occluded }));
  try {
    const response = await fetch("/api/label/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame: frame.frame, condition, slots }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not save.");
    frame.labeled = data.saved;
    frame.complete = data.saved === label.slots.length;
    frame.condition = condition;
    label.lastCondition = condition;
    setDirty(false);
    const next = nextUnlabeled(label.index);
    if (next === -1) {
      render();
      message(`${data.message} Every frame is fully labeled.`);
      return;
    }
    await show(next);
    message(data.message);
  } catch (error) {
    message(error.message);
  }
}

async function load() {
  const response = await fetch("/api/label/frames", { cache: "no-store" });
  const data = await response.json();
  label.frames = data.frames;
  label.slots = data.slots;
  label.conditions = data.conditions;
  $("labelCondition").replaceChildren(
    ...data.conditions.map((condition) => {
      const option = document.createElement("option");
      option.value = condition;
      option.textContent = condition;
      return option;
    }),
  );
  const empty = !label.frames.length || !label.slots.length;
  $("labelEmpty").hidden = !empty;
  $("labelWorkspace").hidden = empty;
  if (empty) {
    if (label.frames.length && !label.slots.length) $("labelEmpty").textContent = "There are frames, but no slots: draw the layout on the dashboard first.";
    return;
  }
  const start = label.frames.findIndex((frame) => !frame.complete);
  await show(start === -1 ? 0 : start);
}

$("saveNext").addEventListener("click", save);
$("prevFrame").addEventListener("click", () => show(label.index - 1));
$("nextFrame").addEventListener("click", () => show(label.index + 1));
$("nextUnlabeled").addEventListener("click", () => {
  const next = nextUnlabeled(label.index);
  if (next === -1) message("Every frame is fully labeled.");
  else show(next);
});
$("restVacant").addEventListener("click", () => setUnset("vacant"));
$("restOccupied").addEventListener("click", () => setUnset("occupied"));
$("clearAll").addEventListener("click", () => {
  for (const mark of Object.values(label.marks)) Object.assign(mark, { status: null, occluded: false });
  setDirty(true);
  render();
});
$("labelCondition").addEventListener("change", () => setDirty(true));
document.addEventListener("keydown", (event) => {
  if ($("labelWorkspace").hidden || event.target.closest("select, input, textarea")) return;
  if (event.key === "Enter" && !event.target.closest("button")) {
    event.preventDefault();
    save();
  } else if (event.key === "ArrowLeft") show(label.index - 1);
  else if (event.key === "ArrowRight") show(label.index + 1);
});

load();
