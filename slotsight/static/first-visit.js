// The guide opens once: on the first visit to the dashboard in this browser, go to
// /guide before the dashboard is drawn. The visit is noted first, so the guide's
// "Open the dashboard" never loops back; if storage is blocked (some private
// windows), nothing is noted and the dashboard opens as usual.
(function () {
  const KEY = "slotsight-guide-seen";
  try {
    if (localStorage.getItem(KEY)) return;
    localStorage.setItem(KEY, "1");
    if (localStorage.getItem(KEY) !== "1") return;
    location.replace("/guide?first=1");
  } catch (error) {
    // No storage: never redirect.
  }
})();
