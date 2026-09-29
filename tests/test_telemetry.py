"""Tests for sim/telemetry.py (JSONL writer, terminal formatter) and sim/model.py (§17)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from amongus.config import SimConfig
from amongus.sim import telemetry
from amongus.sim.model import AmongUsModel
from amongus.sim.telemetry import TerminalWriter, format_event, write_events_jsonl
from amongus.types import Event
from amongus.world.engine import Engine

from .helpers import RandomWalkPolicy

# One synthetic Event per §14.1 type, each carrying task_bar like a real Engine.emit() call.
_SAMPLE_EVENTS: list[Event] = [
    Event(0, "SPAWN", {"task_bar": 0.0, "agent": "red", "room": "cafeteria"}),
    Event(0, "ROLES", {"task_bar": 0.0, "impostors": ["blue", "black"]}),
    Event(
        1, "MOVE", {"task_bar": 0.0, "agent": "red", "from": "cafeteria", "to": "storage", "eta": 4}
    ),
    Event(5, "ARRIVE", {"task_bar": 0.0, "agent": "red", "room": "storage"}),
    Event(5, "VENT", {"task_bar": 0.0, "agent": "blue", "from": "storage", "to": "electrical"}),
    Event(
        10,
        "KILL",
        {"task_bar": 0.1, "killer": "blue", "victim": "red", "room": "electrical", "witnesses": []},
    ),
    Event(
        15, "BODY_SEEN", {"task_bar": 0.1, "agent": "green", "victim": "red", "room": "electrical"}
    ),
    Event(
        15,
        "REPORT",
        {"task_bar": 0.1, "agent": "green", "victim": "red", "room": "electrical", "meeting": 1},
    ),
    Event(20, "BUTTON", {"task_bar": 0.1, "agent": "pink", "room": "cafeteria", "meeting": 2}),
    Event(
        15,
        "MEETING_START",
        {"task_bar": 0.1, "meeting": 1, "reason": "report", "alive": ["green", "blue"]},
    ),
    Event(
        15,
        "STATEMENT",
        {"task_bar": 0.1, "meeting": 1, "round": 1, "agent": "green", "text": "I found red."},
    ),
    Event(
        15,
        "SUSPICION",
        {"task_bar": 0.1, "meeting": 1, "round": 1, "agent": "green", "scores": {"blue": 0.8}},
    ),
    Event(15, "VOTE", {"task_bar": 0.1, "meeting": 1, "agent": "green", "target": "blue"}),
    Event(
        15,
        "EJECT",
        {
            "task_bar": 0.1,
            "meeting": 1,
            "target": "blue",
            "was_impostor": True,
            "tally": {"blue": 4, "skip": 1},
        },
    ),
    Event(
        20,
        "SABOTAGE",
        {
            "task_bar": 0.1,
            "kind": "reactor",
            "panels": ["reactor", "o2"],
            "timer": 30,
            "room": None,
            "agent": "black",
        },
    ),
    Event(21, "BID", {"task_bar": 0.1, "agent": "yellow", "costs": {"reactor": 8, "o2": 11}}),
    Event(21, "COMMIT", {"task_bar": 0.1, "agent": "yellow", "target": "reactor", "eta": 29}),
    Event(
        26,
        "REVOKE",
        {
            "task_bar": 0.1,
            "agent": "black",
            "target": "o2",
            "eta": 26,
            "backup": "white",
            "backup_eta": 35,
        },
    ),
    Event(29, "PANEL_DONE", {"task_bar": 0.1, "agent": "yellow", "room": "reactor"}),
    Event(35, "SABOTAGE_FIXED", {"task_bar": 0.1, "kind": "reactor", "ticks": 15}),
    Event(
        40,
        "DOORS_OPEN",
        {"task_bar": 0.1, "room": "storage", "edges": [["storage", "electrical"]]},
    ),
    Event(
        41,
        "REPLAN",
        {
            "task_bar": 0.1,
            "agent": "green",
            "reason": "topology",
            "expanded": 9,
            "path": ["a", "b"],
        },
    ),
    Event(60, "PROGRESS_STALL", {"task_bar": 0.1, "ticks": 20}),
    Event(60, "DEADLOCK_BROKEN", {"task_bar": 0.1, "ticks": 20}),
    Event(15, "LLM_PARSE_FAIL", {"task_bar": 0.1, "agent": "white", "meeting": 1, "round": 2}),
    Event(70, "SHOCK", {"task_bar": 0.1, "kind": "doors", "room": "storage"}),
    Event(
        100,
        "GAME_OVER",
        {"task_bar": 1.0, "winner": "crewmate", "ticks": 100, "reason": "all_tasks_done"},
    ),
]


def test_every_event_line_has_t_type_task_bar() -> None:
    """Every synthetic §14.1 event round-trips with the three mandatory keys present."""
    for event in _SAMPLE_EVENTS:
        line = telemetry.event_line(event)
        obj = json.loads(line)
        assert obj["t"] == event.tick
        assert obj["type"] == event.type
        assert "task_bar" in obj
        for key, value in event.data.items():
            assert obj[key] == value


def test_write_events_jsonl_creates_run_dir(tmp_path) -> None:
    """write_events_jsonl creates runs/<scenario>_<seed>/events.jsonl with one line per event."""
    path = write_events_jsonl(_SAMPLE_EVENTS, scenario="s1", seed=42, base_dir=tmp_path)
    assert path == tmp_path / "s1_42" / "events.jsonl"
    assert path.is_file()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(_SAMPLE_EVENTS)
    for line in lines:
        obj = json.loads(line)
        assert "t" in obj and "type" in obj and "task_bar" in obj


def test_jsonl_deterministic_same_seed(tmp_path) -> None:
    """Hashing events.jsonl twice for the same seed gives the same digest (§14.3)."""
    config = SimConfig(seed=7, n_players=4, n_impostors=1, max_ticks=30)
    ids = list(config.colors[: config.n_players])

    def run(base_dir) -> bytes:
        policies = {aid: RandomWalkPolicy(seed=100 + i) for i, aid in enumerate(ids)}
        engine = Engine(config, policies)
        engine.run(20)
        path = write_events_jsonl(
            engine.events, scenario="det", seed=config.seed, base_dir=base_dir
        )
        return path.read_bytes()

    blob1 = run(tmp_path / "run1")
    blob2 = run(tmp_path / "run2")
    assert hashlib.sha256(blob1).hexdigest() == hashlib.sha256(blob2).hexdigest()
    assert len(blob1) > 0


def test_unknown_event_type_writer_does_not_raise() -> None:
    """An unrecognised event type serializes to a JSON line instead of raising."""
    event = Event(5, "TOTALLY_UNKNOWN", {"task_bar": 0.5, "foo": "bar"})
    line = telemetry.event_line(event)
    obj = json.loads(line)
    assert obj["type"] == "TOTALLY_UNKNOWN"
    assert obj["foo"] == "bar"


def test_unknown_event_type_formatter_degrades_gracefully() -> None:
    """format_event never raises on an unknown type; it returns a generic line."""
    event = Event(5, "TOTALLY_UNKNOWN", {"task_bar": 0.5, "foo": "bar"})
    line = format_event(event, verbosity=1)
    assert line is not None
    assert "TOTALLY_UNKNOWN" in line
    assert "foo=bar" in line


def test_formatter_missing_field_degrades_instead_of_raising() -> None:
    """A known type with a missing expected field still produces a line, not an exception."""
    event = Event(5, "MOVE", {"task_bar": 0.5})  # missing agent/from/to/eta
    line = format_event(event, verbosity=1)
    assert line is not None
    assert "MOVE" in line


def test_verbosity_zero_is_silent() -> None:
    """verbosity=0 suppresses every event type."""
    for event in _SAMPLE_EVENTS:
        assert format_event(event, verbosity=0) is None


def test_suspicion_gated_to_verbosity_two() -> None:
    """SUSPICION lines only appear at verbosity>=2 ('events + suspicion')."""
    event = next(e for e in _SAMPLE_EVENTS if e.type == "SUSPICION")
    assert format_event(event, verbosity=1) is None
    line = format_event(event, verbosity=2)
    assert line is not None
    assert "sus" in line


def test_roles_gated_to_debug_verbosity() -> None:
    """ROLES (ground truth) is hidden below verbosity 3."""
    event = next(e for e in _SAMPLE_EVENTS if e.type == "ROLES")
    assert format_event(event, verbosity=1) is None
    assert format_event(event, verbosity=2) is None
    assert format_event(event, verbosity=3) is not None


def test_meeting_lines_have_their_own_shape() -> None:
    """STATEMENT/SUSPICION/EJECT render as '[Mn ...]' lines, not '[t=NNN] ...'."""
    statement = next(e for e in _SAMPLE_EVENTS if e.type == "STATEMENT")
    suspicion = next(e for e in _SAMPLE_EVENTS if e.type == "SUSPICION")
    eject = next(e for e in _SAMPLE_EVENTS if e.type == "EJECT")
    assert format_event(statement, 1).startswith("[M1 r1]")
    assert format_event(suspicion, 2).startswith("[M1 sus]")
    assert format_event(eject, 1).startswith("[M1 vote]")
    assert "EJECT blue (IMPOSTOR)" in format_event(eject, 1)


def test_terminal_writer_plain_text_when_rich_absent(monkeypatch, capsys) -> None:
    """With rich unimportable, TerminalWriter still prints plain text lines."""
    monkeypatch.setattr(telemetry, "_rich_console", lambda: None)
    writer = TerminalWriter(verbosity=1)
    assert writer._console is None
    move = next(e for e in _SAMPLE_EVENTS if e.type == "MOVE")
    writer.emit(move)
    out = capsys.readouterr().out
    assert "MOVE" in out
    assert "red" in out


def test_terminal_writer_emit_all_never_raises_on_mixed_stream() -> None:
    """emit_all runs over every sample event plus an unknown one without raising."""
    writer = TerminalWriter(verbosity=3)
    writer.emit_all([*_SAMPLE_EVENTS, Event(1, "MYSTERY", {"task_bar": 0.0})])


def test_mesa_model_advances_and_collects_series() -> None:
    """AmongUsModel.step() ticks the engine and DataCollector gathers a matching series."""
    config = replace(SimConfig(), n_players=4, n_impostors=1, max_ticks=50, seed=3)
    ids = list(config.colors[: config.n_players])
    policies = {aid: RandomWalkPolicy(seed=200 + i) for i, aid in enumerate(ids)}
    model = AmongUsModel(config, policies)
    n_steps = 10
    for _ in range(n_steps):
        model.step()
    data = model.datacollector.model_vars
    assert set(data) == {"alive", "task_bar", "sabotage_active", "meeting_count"}
    for series in data.values():
        assert len(series) == n_steps + 1  # initial collect at construction, plus one per step
    assert all(isinstance(v, int) for v in data["alive"])
    assert data["alive"][0] >= data["alive"][-1]  # alive count never increases
