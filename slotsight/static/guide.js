// The guide's live sign: the lot as the dashboard shows it, read-only (GET /api/status).
// It follows the dashboard's rules (liveState and renderLiveState in app.js) in a
// shorter form: the number is lit only for a current, real reading.
(function () {
  const POLL_MS = 5000;
  const STALE_MIN_S = 10;
  const $ = (id) => document.getElementById(id);
  let lastUpdatedAt = null;
  let lastChangeAt = Date.now();

  if (new URLSearchParams(location.search).has("first")) $("guideWelcome").hidden = false;

  function timeOnly(iso) {
    return iso ? new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "--:--:--";
  }

  // What the numbers are based on, and whether they are current.
  function describe(data) {
    const model = data.model_status || {};
    const job = data.job;
    if (data.source === "model error" || ["error", "needs_package", "unknown"].includes(model.state))
      return { text: "Model error", tone: "error", current: false };
    if (data.source === "demo") return { text: "Demo data", tone: "warn", current: false };
    if (data.source === "none")
      return ["downloading", "loading"].includes(model.state)
        ? { text: "Model loading", tone: "warn", current: false }
        : { text: "No frame yet", tone: "idle", current: false };
    if (job && job.state === "running") {
      const quiet = Math.round((Date.now() - lastChangeAt) / 1000);
      if (quiet > Math.max(STALE_MIN_S, 3 * (Number(job.interval_s) || 0)))
        return { text: `No new reading for ${quiet} s`, tone: "warn", current: false };
      return { text: job.kind === "file" ? "Processing video" : "Live", tone: "live", current: true };
    }
    if (job && job.kind !== "file" && ["stopped", "error"].includes(job.state) && data.source === job.name)
      return { text: job.state === "error" ? "Camera lost" : "Camera disconnected", tone: job.state === "error" ? "error" : "idle", current: false };
    return { text: `Last reading ${timeOnly(data.updated_at)}`, tone: "idle", current: true };
  }

  function show(data) {
    if (data.updated_at !== lastUpdatedAt) {
      lastUpdatedAt = data.updated_at;
      lastChangeAt = Date.now();
    }
    const taken = (data.slots || []).filter((slot) => slot.status === "occupied").length;
    $("gLotName").textContent = data.lot_name || "Lot";
    $("gTaken").textContent = String(taken);
    $("gUnsure").textContent = String(data.unknown ?? 0);
    $("gTotal").textContent = String(data.total ?? "--");
    $("gLedTotal").textContent = String(data.total ?? "--");
    $("gFree").textContent = String(data.free ?? "--");
    $("gFreeUnit").textContent = data.free === 1 ? "space" : "spaces";
    const state = describe(data);
    const status = $("gStatus");
    status.textContent = state.text;
    status.className = `health-pill tone-${state.tone}`;
    $("guideSign").classList.toggle("is-dimmed", !state.current);
  }

  async function poll() {
    try {
      const response = await fetch("/api/status", { cache: "no-store" });
      if (!response.ok) throw new Error(`Status ${response.status}`);
      show(await response.json());
    } catch (error) {
      $("gStatus").textContent = "Reconnecting...";
      $("gStatus").className = "health-pill tone-idle";
      $("guideSign").classList.add("is-dimmed");
    }
  }

  // The top bar's hairline appears only while content runs under it.
  const topbar = $("topbar");
  const markScrolled = () => topbar.classList.toggle("is-scrolled", window.scrollY > 4);
  window.addEventListener("scroll", markScrolled, { passive: true });
  markScrolled();

  poll();
  setInterval(poll, POLL_MS);
})();
