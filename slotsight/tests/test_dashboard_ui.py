"""Static checks on the dashboard markup and script (2026-11-18 UI pass: honest live
state, keyboard access, map updates in place, layout editor undo/cancel)."""

import re
from pathlib import Path

SLOTSIGHT_DIR = Path(__file__).resolve().parent.parent
INDEX = (SLOTSIGHT_DIR / "templates" / "index.html").read_text(encoding="utf-8")
APP_JS = (SLOTSIGHT_DIR / "static" / "app.js").read_text(encoding="utf-8")
STYLE = (SLOTSIGHT_DIR / "static" / "style.css").read_text(encoding="utf-8")


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
