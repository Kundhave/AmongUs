"""Tests for §16 scenarios S1-S4 and the §14.1 event schema they must produce."""

import hashlib
import json
from dataclasses import asdict, replace

import pytest

from amongus.sim.scenarios import SCENARIOS, build
from amongus.types import Phase, Role

# §14.1's binding field table: type -> required fields beyond t/type/task_bar. MOVE/ARRIVE/VENT
# are intentionally included even though world/resolve.py currently emits them lowercase with
# different field names ("move_start"/"arrive"/"vent"/"weight") — see the telemetry-engineer
# milestone report for the flagged cross-team gap; this table stays spec-accurate so the check
# tightens automatically once that's fixed upstream.
_SCHEMA: dict[str, tuple[str, ...]] = {
    "SPAWN": ("agent", "room"),
    "ROLES": ("impostors",),
    "MOVE": ("agent", "from", "to", "eta"),
    "ARRIVE": ("agent", "room"),
    "VENT": ("agent", "from", "to"),
    "KILL": ("killer", "victim", "room", "witnesses"),
    "BODY_SEEN": ("agent", "victim", "room"),
    "REPORT": ("agent", "victim", "room", "meeting"),
    "BUTTON": ("agent", "room", "meeting"),
    "MEETING_START": ("meeting", "reason", "alive"),
    "STATEMENT": ("meeting", "round", "agent", "text"),
    "SUSPICION": ("meeting", "round", "agent", "scores"),
    "VOTE": ("meeting", "agent", "target"),
    "EJECT": ("meeting", "target", "was_impostor", "tally"),
    "SABOTAGE": ("kind", "panels", "timer", "room", "agent"),
    "BID": ("agent", "costs"),
    "COMMIT": ("agent", "target", "eta"),
    "REVOKE": ("agent", "target", "eta", "grace", "backup", "backup_eta"),
    "PANEL_DONE": ("agent", "room"),
    "SABOTAGE_FIXED": ("kind", "ticks"),
    "DOORS_OPEN": ("room", "edges"),
    "REPLAN": ("agent", "reason", "expanded", "path"),
    "PROGRESS_STALL": ("ticks",),
    "DEADLOCK_BROKEN": ("ticks",),
    "LLM_PARSE_FAIL": ("agent", "meeting", "round"),
    "SHOCK": ("kind", "room"),
    "GAME_OVER": ("winner", "ticks", "reason"),
}


def _event_hash(engine) -> str:
    """Hash a run's event list deterministically for the reproducibility test."""
    blob = json.dumps([asdict(e) for e in engine.events], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_reaches_terminal_state_offline(name: str) -> None:
    """Every scenario runs to a winner or a draw with no exception, fully offline by default."""
    engine, _scripted = build(SCENARIOS[name])
    engine.run(engine.config.max_ticks)
    assert engine.world.phase is Phase.OVER
    assert engine.config.deliberator == "template"


def test_scenario_deliberator_is_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    """`config_overrides={"deliberator": "gemini"}` overrides the scenario's template default.

    With no GEMINI_API_KEY set, GeminiDeliberator's own client still falls back to the
    template path internally with zero network calls, so this stays fully offline (§9.3).
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    base = SCENARIOS["baseline"]
    scenario = replace(base, config_overrides={**base.config_overrides, "deliberator": "gemini"})
    engine, _scripted = build(scenario)
    assert engine.config.deliberator == "gemini"
    engine.run(engine.config.max_ticks)
    assert engine.world.phase is Phase.OVER


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_reproducible(name: str) -> None:
    """The same scenario, built and run twice, produces a byte-identical event log."""
    e1, _s1 = build(SCENARIOS[name])
    e1.run(e1.config.max_ticks)
    e2, _s2 = build(SCENARIOS[name])
    e2.run(e2.config.max_ticks)
    assert _event_hash(e1) == _event_hash(e2)
    assert len(e1.events) > 0


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_event_matches_the_14_1_schema(name: str) -> None:
    """Every produced event carries t/type/task_bar, plus its type's required fields if known."""
    engine, _scripted = build(SCENARIOS[name])
    engine.run(engine.config.max_ticks)
    for event in engine.events:
        assert isinstance(event.tick, int)
        assert isinstance(event.type, str) and event.type
        assert 0.0 <= event.data["task_bar"] <= 1.0
        for field in _SCHEMA.get(event.type, ()):
            assert field in event.data, f"{event.type} event missing '{field}': {event.data}"


def test_worked_example_choreography_is_deterministic() -> None:
    """S2's script reliably produces: blue kills red in electrical at t=43, green reports t=47."""
    engine, _scripted = build(SCENARIOS["worked_example"])
    engine.run(engine.config.max_ticks)
    kills = [e for e in engine.events if e.type == "KILL"]
    reports = [e for e in engine.events if e.type == "REPORT"]
    assert kills and kills[0].tick == 43
    assert kills[0].data["killer"] == "blue"
    assert kills[0].data["victim"] == "red"
    assert kills[0].data["room"] == "electrical"
    assert reports and reports[0].tick == 47
    assert reports[0].data["agent"] == "green"
    assert reports[0].data["victim"] == "red"
    impostors = set(next(e for e in engine.events if e.type == "ROLES").data["impostors"])
    assert impostors == {"blue", "black"}


def test_reactor_defect_default_seed_reaches_sabotage_fixed() -> None:
    """The scenario's own default seed must reach the full commit->defect->revoke->fixed arc.

    Pinned to `SCENARIOS["reactor_defect"].seed` rather than a literal 3, so the demo can't
    silently regress to a seed where both impostors sweep both panels and burn the reactor
    timer instead (the protocol failing, not succeeding) without this test catching it.
    """
    engine, _scripted = build(SCENARIOS["reactor_defect"])
    engine.run(engine.config.max_ticks)
    assert any(e.type == "REVOKE" for e in engine.events)
    assert any(e.type == "SABOTAGE_FIXED" for e in engine.events)


def test_reactor_defect_produces_a_shared_negotiation() -> None:
    """S3 exercises one shared ReactorBoard: BIDs from multiple agents, then a real COMMIT.

    A per-agent (unshared) board would still emit BIDs but never a coherent COMMIT, since no
    agent would see any other agent's bid; asserting on COMMIT is what actually discriminates.
    """
    engine, _scripted = build(SCENARIOS["reactor_defect"])
    engine.run(engine.config.max_ticks)
    bidders = {e.data["agent"] for e in engine.events if e.type == "BID"}
    commits = [e for e in engine.events if e.type == "COMMIT"]
    assert len(bidders) >= 2
    assert commits
    assert any(e.type == "REVOKE" for e in engine.events)


@pytest.mark.parametrize("deadlock_protocol", [True, False])
def test_standoff_both_ablation_outcomes(deadlock_protocol: bool) -> None:
    """Same seed, one flag: recovers with the protocol on, stalls to max_ticks with it off."""
    base = SCENARIOS["standoff"]
    scenario = replace(
        base, config_overrides={**base.config_overrides, "deadlock_protocol": deadlock_protocol}
    )
    engine, _scripted = build(scenario)
    engine.run(engine.config.max_ticks)
    stalls = [e for e in engine.events if e.type == "PROGRESS_STALL"]
    breaks = [e for e in engine.events if e.type == "DEADLOCK_BROKEN"]
    if deadlock_protocol:
        assert stalls and breaks
        assert engine.world.task_bar == 1.0
        assert engine.world.winner is Role.CREWMATE
    else:
        assert not stalls
        assert not breaks
        assert engine.world.tick == engine.config.max_ticks
        assert engine.world.task_bar < 1.0
