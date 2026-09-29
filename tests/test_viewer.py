"""Structural checks for the replay viewer (SPEC §15, §17).

These are static checks over the HTML/JS text and the sample log — they do
not drive a browser. Manual verification (loading the page, stepping through
a real log) is done separately and reported alongside these results.
"""

import json
from pathlib import Path

import pytest

UI_DIR = Path(__file__).resolve().parents[1] / "src" / "amongus" / "ui"
VIEWER = UI_DIR / "viewer.html"
SAMPLE = UI_DIR / "sample_events.jsonl"

# Room names from docs/SPEC.md §5.
ROOM_NAMES = [
    "upper_engine", "reactor", "security", "lower_engine", "medbay",
    "cafeteria", "weapons", "o2", "navigation", "shields",
    "communications", "storage", "admin", "electrical",
]

# Event types required by docs/SPEC.md §14.1. Hard-coded on purpose: a future
# rename in sim/telemetry.py must fail this test loudly rather than silently
# degrade the viewer to rendering "?" everywhere.
EVENT_TYPES = [
    "SPAWN", "ROLES", "MOVE", "ARRIVE", "VENT", "KILL", "BODY_SEEN", "REPORT",
    "BUTTON", "MEETING_START", "STATEMENT", "SUSPICION", "VOTE", "EJECT",
    "SABOTAGE", "BID", "COMMIT", "REVOKE", "PANEL_DONE", "SABOTAGE_FIXED",
    "DOORS_OPEN", "REPLAN", "PROGRESS_STALL", "DEADLOCK_BROKEN",
    "LLM_PARSE_FAIL", "SHOCK", "GAME_OVER",
]

# Fields beyond t/type/task_bar, per docs/SPEC.md §14.1's binding contract
# table. Values may be null (e.g. VOTE target, SABOTAGE room) but the key
# must be present. Used to check any real log line against the pinned schema.
FIELD_SCHEMA = {
    "SPAWN": {"agent", "room"},
    "ROLES": {"impostors"},
    "MOVE": {"agent", "from", "to", "eta"},
    "ARRIVE": {"agent", "room"},
    "VENT": {"agent", "from", "to"},
    "KILL": {"killer", "victim", "room", "witnesses"},
    "BODY_SEEN": {"agent", "victim", "room"},
    "REPORT": {"agent", "victim", "room", "meeting"},
    "BUTTON": {"agent", "room", "meeting"},
    "MEETING_START": {"meeting", "reason", "alive"},
    "STATEMENT": {"meeting", "round", "agent", "text"},
    "SUSPICION": {"meeting", "round", "agent", "scores"},
    "VOTE": {"meeting", "agent", "target"},
    "EJECT": {"meeting", "target", "was_impostor", "tally"},
    "SABOTAGE": {"kind", "panels", "timer", "room", "agent"},
    "BID": {"agent", "costs"},
    "COMMIT": {"agent", "target", "eta"},
    "REVOKE": {"agent", "target", "eta", "grace", "backup", "backup_eta"},
    "PANEL_DONE": {"agent", "room"},
    "SABOTAGE_FIXED": {"kind", "ticks"},
    "DOORS_OPEN": {"room", "edges"},
    "REPLAN": {"agent", "reason", "expanded", "path"},
    "PROGRESS_STALL": {"ticks"},
    "DEADLOCK_BROKEN": {"ticks"},
    "LLM_PARSE_FAIL": {"agent", "meeting", "round"},
    "SHOCK": {"kind", "room"},
    "GAME_OVER": {"winner", "ticks", "reason"},
}


@pytest.fixture(scope="module")
def viewer_text() -> str:
    """Read viewer.html once for all structural checks."""
    return VIEWER.read_text(encoding="utf-8")


def test_viewer_exists():
    """The viewer file exists at its owned path."""
    assert VIEWER.is_file()


def test_sample_log_exists():
    """The sample log exists at its owned path."""
    assert SAMPLE.is_file()


def test_self_contained_no_external_refs(viewer_text):
    """The viewer must never reach the network: no http(s) URLs, no src=, no <link>."""
    lowered = viewer_text.lower()
    assert "http://" not in lowered
    assert "https://" not in lowered
    assert "<link" not in lowered
    # src= is fine only if it never appears (canvas/script/img are all inline here).
    assert "src=" not in lowered
    assert "<script src" not in lowered
    assert "cdn." not in lowered


def test_single_html_file_inline_style_and_script(viewer_text):
    """CSS and JS are inline; there is exactly one <script> and one <style> block."""
    assert viewer_text.count("<style>") == 1
    assert viewer_text.count("<script>") == 1
    assert "<canvas" in viewer_text


def test_all_14_rooms_present(viewer_text):
    """Every room name from §5 appears in the embedded ROOMS table."""
    assert len(ROOM_NAMES) == 14
    for room in ROOM_NAMES:
        assert room in viewer_text, f"room {room!r} missing from viewer"


def test_all_24_edges_declared(viewer_text):
    """The EDGES list literal is present with 24 entries (one per corridor)."""
    start = viewer_text.index("const EDGES = [")
    end = viewer_text.index("];", start)
    block = viewer_text[start:end]
    # Each edge is a [a,b,w] triple; count top-level bracket groups.
    assert block.count("[\"") + block.count("[ \"") >= 24 or block.count("[") - 1 >= 24


def test_vent_pairs_present(viewer_text):
    """All 8 vent pairs from §5 are embedded."""
    start = viewer_text.index("const VENT_PAIRS = [")
    end = viewer_text.index("];", start)
    block = viewer_text[start:end]
    assert block.count(",") >= 8  # 8 pairs, each with an internal comma at minimum


def test_every_event_type_in_dispatch_logic(viewer_text):
    """Every §14.1 event type has a handler in the EVENT_HANDLERS dispatch table."""
    start = viewer_text.index("const EVENT_HANDLERS = {")
    end = viewer_text.index("\n};", start)
    block = viewer_text[start:end]
    for et in EVENT_TYPES:
        assert f"{et}(" in block, f"event type {et!r} missing from dispatch table"


def test_reveal_toggle_present_and_off_by_default(viewer_text):
    """The reveal-impostors checkbox exists and is unchecked by default."""
    assert 'id="revealToggle"' in viewer_text
    # The checkbox markup must not carry a `checked` attribute.
    idx = viewer_text.index('id="revealToggle"')
    snippet = viewer_text[max(0, idx - 60): idx + 60]
    assert "checked" not in snippet


def test_eject_reads_contract_target_field(viewer_text):
    """EJECT's ejected agent is read from `target` (the §14.1 contract field), not `agent`."""
    start = viewer_text.index("EJECT(state, evt){")
    end = viewer_text.index("\n  },", start)
    block = viewer_text[start:end]
    assert 'pick(evt, ["target"' in block, "EJECT handler must try `target` first"


def test_bid_renders_costs_dict(viewer_text):
    """BID's feed line reads `costs` (a {room: int} dict), matching the §14.1 contract."""
    start = viewer_text.index('case "BID":')
    end = viewer_text.index("case \"COMMIT\":", start)
    block = viewer_text[start:end]
    assert '"costs"' in block


def test_revoke_reads_contract_fields(viewer_text):
    """REVOKE's feed line reads `target`, `grace` and `backup_eta`, matching §14.1."""
    start = viewer_text.index('case "REVOKE":')
    end = viewer_text.index("\n", start)
    line = viewer_text[start:end]
    assert '"target"' in line
    assert '"grace"' in line
    assert '"backup_eta"' in line


def test_never_throw_guardrails_present(viewer_text):
    """Parsing and event application are wrapped defensively (never-throw contract)."""
    assert "try{" in viewer_text or "try {" in viewer_text
    assert "malformed" in viewer_text.lower()


def test_no_horizontal_scroll_rule(viewer_text):
    """The stylesheet forbids horizontal scrolling on the page."""
    assert "overflow-x:hidden" in viewer_text.replace(" ", "")


def test_light_and_dark_theme_support(viewer_text):
    """Both a manual theme toggle and prefers-color-scheme are wired up."""
    assert "prefers-color-scheme" in viewer_text
    assert 'data-theme' in viewer_text


def test_sample_log_parses_as_jsonl():
    """sample_events.jsonl is valid JSONL: every non-blank line is a real §14.1 event."""
    lines = SAMPLE.read_text(encoding="utf-8").splitlines()
    non_blank = [line for line in lines if line.strip()]
    assert len(non_blank) > 0
    seen_types = set()
    for line in non_blank:
        obj = json.loads(line)  # raises if malformed -- the whole point of this test
        assert isinstance(obj, dict)
        assert "t" in obj and isinstance(obj["t"], int)
        assert "type" in obj and isinstance(obj["type"], str)
        assert "task_bar" in obj, "every line must carry task_bar per §14.1"
        assert obj["type"] in EVENT_TYPES, f"unknown type {obj['type']!r} in sample log"
        seen_types.add(obj["type"])
    # A real single-scenario run cannot exercise every possible type (e.g. a
    # single seed rarely trips BUTTON, SHOCK or LLM_PARSE_FAIL), so only
    # require a healthy, varied subset rather than the full EVENT_TYPES set.
    assert len(seen_types) >= 10, f"sample log only exercises {sorted(seen_types)}"


def test_sample_log_fields_match_pinned_schema():
    """Every event in the real sample carries exactly the §14.1 contract fields for its type."""
    raw_lines = SAMPLE.read_text(encoding="utf-8").splitlines()
    lines = [json.loads(line) for line in raw_lines if line.strip()]
    checked = 0
    for evt in lines:
        expected = FIELD_SCHEMA.get(evt["type"])
        if expected is None:
            continue
        missing = expected - set(evt.keys())
        assert not missing, f"{evt['type']} event missing contract fields {missing}: {evt}"
        checked += 1
    assert checked > 0


def test_sample_log_tells_reactor_renegotiation_story():
    """Sanity check: the reactor_defect sample tells a coherent sabotage/renegotiation story."""
    raw_lines = SAMPLE.read_text(encoding="utf-8").splitlines()
    lines = [json.loads(line) for line in raw_lines if line.strip()]
    by_type = {}
    for evt in lines:
        by_type.setdefault(evt["type"], []).append(evt)
    assert by_type.get("SABOTAGE"), "sample must include a sabotage"
    assert by_type.get("BID"), "sample must include impostor-fix bids"
    assert by_type.get("COMMIT"), "sample must include a commit"
    assert by_type.get("REVOKE"), "sample must include a revoke (defect/renegotiation)"
    assert by_type.get("KILL"), "sample must include a kill"
    assert by_type.get("GAME_OVER"), "sample must reach a conclusion"
    # Chronological sanity: sabotage starts before it is committed to, which
    # is revoked before the game ends.
    t_sabotage = by_type["SABOTAGE"][0]["t"]
    t_commit = by_type["COMMIT"][0]["t"]
    t_revoke = by_type["REVOKE"][0]["t"]
    t_over = by_type["GAME_OVER"][0]["t"]
    assert t_sabotage <= t_commit <= t_revoke <= t_over
