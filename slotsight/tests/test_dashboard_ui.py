"""Static checks on the dashboard markup, script and styles (2026-11-18 UI passes:
honest live state, keyboard access, map updates in place, layout editor undo/cancel;
then hierarchy, the Settings dialog, type and buttons, top bar and themes)."""

import re
from pathlib import Path

SLOTSIGHT_DIR = Path(__file__).resolve().parent.parent
INDEX = (SLOTSIGHT_DIR / "templates" / "index.html").read_text(encoding="utf-8")
APP_JS = (SLOTSIGHT_DIR / "static" / "app.js").read_text(encoding="utf-8")
STYLE = (SLOTSIGHT_DIR / "static" / "style.css").read_text(encoding="utf-8")
LABEL = (SLOTSIGHT_DIR / "templates" / "label.html").read_text(encoding="utf-8")


def test_every_element_the_script_looks_up_exists_in_the_page():
    used = set(re.findall(r'\$\("([A-Za-z][\w-]*)"\)', APP_JS))
    defined = set(re.findall(r'\bid="([^"]+)"', INDEX))
    assert used, "the id pattern no longer matches app.js"
    assert used <= defined, f"missing in index.html: {sorted(used - defined)}"


def test_the_page_never_claims_to_be_online_or_live_on_its_own():
    # The live state comes from the status payload (liveState in app.js), never from fixed text.
    assert "Online" not in INDEX
    assert "Live monitor" not in INDEX


def test_form_controls_stay_reachable_by_keyboard():
    # display:none would take the file picker and the auto-refresh switch out of the tab order.
    assert not re.search(r"\.(switch|upload-zone) input\s*\{[^}]*display:\s*none", STYLE)
    assert 'id="sourceFile"\n                class="visually-hidden"' in INDEX.replace("\r\n", "\n")
    assert re.search(r'class="visually-hidden"\s+id="autoRefresh"', INDEX)


def test_layout_editor_has_no_blocking_dialogs():
    # Undo and Cancel replace confirm(); save errors are shown on the page.
    assert "window.confirm" not in APP_JS
    assert "window.alert" not in APP_JS


def test_pages_use_the_system_font_and_load_nothing_from_other_sites():
    # No Google Fonts: the dashboard works offline and sends no requests elsewhere.
    for page in (INDEX, LABEL):
        assert "fonts.googleapis" not in page and "fonts.gstatic" not in page
        assert 'name="color-scheme" content="light dark"' in page
    assert "system-ui" in STYLE


def test_styles_follow_the_contrast_motion_and_transparency_settings():
    for query in ("prefers-contrast: more", "prefers-reduced-motion: reduce", "prefers-reduced-transparency: reduce"):
        assert query in STYLE, query


def test_theme_choice_is_applied_before_the_page_is_drawn():
    # System / Light / Dark: theme.js sets data-theme (from the choice, or from the
    # system setting) in <head>, before the stylesheet, so there is no flash.
    theme_js = (SLOTSIGHT_DIR / "static" / "theme.js").read_text(encoding="utf-8")
    assert "prefers-color-scheme: dark" in theme_js and "localStorage" in theme_js
    assert '[data-theme="dark"]' in STYLE
    assert "prefers-color-scheme" not in STYLE  # one source of truth: theme.js
    for page in (INDEX, LABEL):
        head = page.split("</head>")[0]
        assert head.index("theme.js") < head.index("style.css")
    for choice in ("system", "light", "dark"):
        assert f'data-theme-choice="{choice}"' in INDEX


def test_one_button_system():
    # The old per-panel button classes are gone from the pages, the scripts and the styles.
    for text in (INDEX, LABEL, APP_JS, STYLE):
        assert "map-edit-button" not in text and "map-clear-button" not in text


def test_settings_dialog_applies_each_change_and_shows_one_frame():
    assert "Configure slots" not in INDEX and "Configure slots" not in APP_JS
    assert "Save configuration" not in INDEX  # each control saves itself (POST /api/settings)
    assert "/api/settings" in APP_JS
    # The annotated frame is shown once, on the map; the second copy in the source panel is gone.
    assert 'id="previewImage"' not in INDEX and 'id="framePreview"' not in INDEX


def test_fonts_are_self_hosted_with_their_licences():
    # Overpass and Overpass Mono (SIL OFL) are served from static/fonts, never a font CDN.
    fonts = SLOTSIGHT_DIR / "static" / "fonts"
    for name in ("overpass.woff2", "overpass-mono.woff2"):
        assert (fonts / name).read_bytes()[:4] == b"wOF2", name
        assert f'url("fonts/{name}")' in STYLE
    for name in ("OVERPASS-LICENSE", "OVERPASS-MONO-LICENSE"):
        assert "SIL Open Font License" in (fonts / name).read_text(encoding="utf-8")
