"""§14 telemetry: deterministic events.jsonl writer plus a §14.2 terminal formatter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from amongus.types import Event

# Column widths for the terminal formatter; a display constant, not a simulation tunable.
_TYPE_COL = 9
_AGENT_COL = 8


def _json_default(obj: Any) -> Any:
    """Fallback JSON encoder: sets/frozensets become sorted lists, anything else becomes str."""
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


def event_line(event: Event) -> str:
    """Serialize one Event to a single deterministic JSON line per the §14.1 schema."""
    payload = {"t": event.tick, "type": event.type, **event.data}
    return json.dumps(payload, sort_keys=True, default=_json_default)


def write_events_jsonl(
    events: Iterable[Event], scenario: str, seed: int, base_dir: str | Path = "runs"
) -> Path:
    """Write a run's event log to <base_dir>/<scenario>_<seed>/events.jsonl, one JSON per line."""
    run_dir = Path(base_dir) / f"{scenario}_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "events.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(event_line(event))
            f.write("\n")
    return path


def _ag(agent: str) -> str:
    """Left-justify an agent id to the fixed agent column width."""
    return f"{agent:<{_AGENT_COL}}"


def _line(tick: int, type_: str, body: str) -> str:
    """Build one '[t=NNN] TYPE      body' terminal line."""
    return f"[t={tick:03d}] {type_:<{_TYPE_COL}}{body}"


def _score(value: float) -> str:
    """Format a 0..1 suspicion score as e.g. '.80' rather than '0.80'."""
    text = f"{value:.2f}"
    return text[1:] if text.startswith("0.") else text


def _scores(scores: dict[str, float]) -> str:
    """Render a suspicion score dict, highest first, ties broken alphabetically."""
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return " | ".join(f"{agent} {_score(v)}" for agent, v in ranked)


def _tally(tally: dict[str, int]) -> str:
    """Render a vote tally dict, highest count first, ties broken alphabetically."""
    ranked = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))
    return " ".join(f"{k}:{v}" for k, v in ranked)


def _f_kill(e: Event) -> str:
    """Format a KILL event, listing witnesses sorted for determinism."""
    d = e.data
    witnesses = ",".join(sorted(d.get("witnesses", [])))
    body = f"{_ag(d['killer'])}x {d['victim']} @{d['room']}   witnesses=[{witnesses}]"
    return _line(e.tick, "KILL", body)


def _f_eject(e: Event) -> str:
    """Format an EJECT event's vote tally and outcome."""
    d = e.data
    target = d["target"]
    if target is None:
        outcome = "-> no ejection"
    else:
        was_impostor = d.get("was_impostor")
        role = "IMPOSTOR" if was_impostor else "CREWMATE" if was_impostor is False else "UNKNOWN"
        outcome = f"-> EJECT {target} ({role})"
    return f"[M{d['meeting']} vote] {_tally(d.get('tally', {}))} {outcome}"


def _f_sabotage(e: Event) -> str:
    """Format a SABOTAGE event, including panels/room only when present."""
    d = e.data
    parts = [d["kind"]]
    if d.get("panels"):
        parts.append(f"panels=[{','.join(d['panels'])}]")
    if d.get("room") is not None:
        parts.append(f"room={d['room']}")
    parts.append(f"timer={d['timer']}")
    return _line(e.tick, "SABOTAGE", " ".join(parts))


def _f_revoke(e: Event) -> str:
    """Format a REVOKE event, showing the backup takeover only when one was named."""
    d = e.data
    backup = d.get("backup")
    tail = (
        f" -> BACKUP {backup} eta={d.get('backup_eta')}" if backup is not None else " -> no backup"
    )
    return _line(e.tick, "REVOKE", f"{d['agent']} {d['target']} (eta {d['eta']}){tail}")

def _costs(d: dict) -> str:
    """Render a BID event's per-room cost dict, sorted by room for determinism."""
    return ",".join(f"{r}:{c}" for r, c in sorted(d.get("costs", {}).items()))


def _edges(d: dict) -> str:
    """Render a DOORS_OPEN event's edge list."""
    return ",".join(f"({a}-{b})" for a, b in d.get("edges", []))


# Simple one-line-body event types: type -> function building the body string from data.
_SIMPLE: dict[str, Callable[[dict], str]] = {
    "MOVE": lambda d: f"{_ag(d['agent'])}{d['from']} -> {d['to']} (eta {d['eta']})",
    "ARRIVE": lambda d: f"{_ag(d['agent'])}@{d['room']}",
    "VENT": lambda d: f"{_ag(d['agent'])}{d['from']} -> {d['to']}",
    "BODY_SEEN": lambda d: f"{_ag(d['agent'])}{d['victim']} @{d['room']}",
    "REPORT": (
        lambda d: f"{_ag(d['agent'])}found {d['victim']} @{d['room']} -> MEETING #{d['meeting']}"
    ),
    "BUTTON": lambda d: f"{_ag(d['agent'])}@{d['room']} -> MEETING #{d['meeting']}",
    "BID": lambda d: f"{_ag(d['agent'])}costs={{{_costs(d)}}}",
    "COMMIT": lambda d: f"{d['agent']}->{d['target']} eta={d['eta']}",
    "PANEL_DONE": lambda d: f"{_ag(d['agent'])}@{d['room']}",
    "SABOTAGE_FIXED": lambda d: f"{d['kind']} ticks={d['ticks']}",
    "DOORS_OPEN": lambda d: f"{d['room']} edges=[{_edges(d)}]",
    "REPLAN": lambda d: f"{_ag(d['agent'])}reason={d['reason']} expanded={d['expanded']}",
    "PROGRESS_STALL": lambda d: f"ticks={d['ticks']}",
    "DEADLOCK_BROKEN": lambda d: f"ticks={d['ticks']}",
    "LLM_PARSE_FAIL": lambda d: f"agent={d['agent']} meeting={d['meeting']} round={d['round']}",
    "SHOCK": lambda d: f"kind={d['kind']} room={d.get('room') or '-'}",
    "GAME_OVER": lambda d: (
        f"winner={d['winner'] if d['winner'] is not None else 'draw'} "
        f"ticks={d['ticks']} reason={d['reason']}"
    ),
    "SPAWN": lambda d: f"{_ag(d['agent'])}@{d['room']}",
    "ROLES": lambda d: f"impostors=[{','.join(sorted(d.get('impostors', [])))}]",
}


def _statement_line(e: Event) -> str:
    """Format a STATEMENT event as a quoted '[Mn rN] agent \"text\"' line."""
    d = e.data
    return f"[M{d['meeting']} r{d['round']}] {_ag(d['agent'])}\"{d['text']}\""


def _suspicion_line(e: Event) -> str:
    """Format a SUSPICION event as a '[Mn sus] agent scores' line."""
    d = e.data
    return f"[M{d['meeting']} sus] {_ag(d['agent'])}{_scores(d.get('scores', {}))}"


# Meeting-scoped lines, shaped "[Mn ...]" instead of "[t=NNN] TYPE ...".
_MEETING: dict[str, Callable[[Event], str]] = {
    "MEETING_START": lambda e: (
        f"[M{e.data['meeting']}] MEETING_START reason={e.data['reason']} "
        f"alive=[{','.join(e.data.get('alive', []))}]"
    ),
    "STATEMENT": _statement_line,
    "SUSPICION": _suspicion_line,
    "VOTE": lambda e: (
        f"[M{e.data['meeting']}] {'VOTE':<{_TYPE_COL}}{e.data['agent']} -> "
        f"{e.data['target'] if e.data['target'] is not None else 'skip'}"
    ),
    "EJECT": _f_eject,
}

# type -> (formatter, minimum verbosity). ROLES/SPAWN/VOTE are debug-only (3): ROLES is ground
# truth no simulated agent may read (§14.1), and per-round VOTE is redundant with EJECT's tally.
_MIN_VERBOSITY: dict[str, int] = {"SUSPICION": 2, "VOTE": 3, "SPAWN": 3, "ROLES": 3}
_CUSTOM: dict[str, Callable[[Event], str]] = {
    "KILL": _f_kill,
    "SABOTAGE": _f_sabotage,
    "REVOKE": _f_revoke,
}

_STYLES: dict[str, str] = {
    "KILL": "bold red",
    "BODY_SEEN": "red",
    "REPORT": "yellow",
    "EJECT": "bold magenta",
    "GAME_OVER": "bold green",
    "SABOTAGE": "red",
    "SABOTAGE_FIXED": "green",
    "SHOCK": "bold red",
}


def _f_generic(e: Event) -> str:
    """Fallback line for any type not in the §14.1 table: never raises."""
    fields = " ".join(f"{k}={e.data[k]}" for k in sorted(e.data) if k != "task_bar")
    return _line(e.tick, e.type, fields)


def format_event(event: Event, verbosity: int) -> str | None:
    """Format one Event to a §14.2-style line, or None if it should be suppressed."""
    if verbosity <= 0:
        return None
    if verbosity < _MIN_VERBOSITY.get(event.type, 1):
        return None
    try:
        if event.type in _MEETING:
            return _MEETING[event.type](event)
        if event.type in _CUSTOM:
            return _CUSTOM[event.type](event)
        if event.type in _SIMPLE:
            return _line(event.tick, event.type, _SIMPLE[event.type](event.data))
    except (KeyError, TypeError):
        return _f_generic(event)
    return _f_generic(event)


def _rich_console() -> Any:
    """Return a rich Console, or None if rich is not importable. A seam for tests."""
    try:
        from rich.console import Console
    except ImportError:
        return None
    return Console(highlight=False)


class TerminalWriter:
    """Prints §14.2-style lines for a stream of events; plain text if rich is unavailable."""

    def __init__(self, verbosity: int) -> None:
        """Store the verbosity level and try to acquire a rich Console."""
        self.verbosity = verbosity
        self._console = _rich_console()

    def emit(self, event: Event) -> None:
        """Format one event and print it, doing nothing if suppressed by verbosity."""
        line = format_event(event, self.verbosity)
        if line is None:
            return
        if self._console is not None:
            self._console.print(line, style=_STYLES.get(event.type), markup=False)
        else:
            print(line)

    def emit_all(self, events: Iterable[Event]) -> None:
        """Format and print every event in order."""
        for event in events:
            self.emit(event)
