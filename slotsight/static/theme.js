// Theme: System (follow the operating system), Light or Dark. Loaded in <head>
// before the stylesheet applies, so the page is never drawn in the wrong theme.
// The choice is a viewer preference, so it is kept in this browser only, never in
// settings.txt (which is the lot's configuration, shared by everyone).
(function () {
  const KEY = "slotsight-theme";
  const CHOICES = ["system", "light", "dark"];
  const root = document.documentElement;
  const systemDark = window.matchMedia("(prefers-color-scheme: dark)");
  const listeners = [];

  function stored() {
    try {
      const value = localStorage.getItem(KEY);
      return CHOICES.includes(value) ? value : "system";
    } catch (error) {
      return "system"; // storage blocked (private mode, site data off): follow the system
    }
  }

  function apply(choice, animate) {
    const theme = choice === "dark" || (choice === "system" && systemDark.matches) ? "dark" : "light";
    if (animate && root.dataset.theme && root.dataset.theme !== theme) {
      // Ease the colours over instead of flashing the whole page (see .theme-changing).
      root.classList.add("theme-changing");
      clearTimeout(apply.timer);
      apply.timer = setTimeout(() => root.classList.remove("theme-changing"), 300);
    }
    root.dataset.theme = theme;
    root.dataset.themeChoice = choice;
    listeners.forEach((listener) => listener(choice, theme));
  }

  apply(stored(), false);
  systemDark.addEventListener("change", () => apply(stored(), true));
  // Another tab (the dashboard or the label page) changed the choice.
  window.addEventListener("storage", (event) => {
    if (event.key === KEY) apply(stored(), true);
  });

  window.slotsightTheme = {
    choice: stored,
    set(choice) {
      if (!CHOICES.includes(choice)) return;
      try {
        localStorage.setItem(KEY, choice);
      } catch (error) {
        // Not saved, but still applied for this page.
      }
      apply(choice, true);
    },
    onChange(listener) {
      listeners.push(listener);
    },
  };
})();
