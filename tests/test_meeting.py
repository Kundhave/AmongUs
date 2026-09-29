"""Tests for the §10 meeting protocol: speaking order, voting, ejection, and full runs."""

import hashlib
import json
from dataclasses import asdict, replace

import pytest

from amongus.agents.crewmate import CrewmatePolicy
from amongus.agents.impostor import ImpostorPolicy
from amongus.agents.memory import AgentMemory
from amongus.agents.protocol import MeetingProtocol
from amongus.agents.reactor import ReactorBoard
from amongus.config import SimConfig
from amongus.contracts import Statement
from amongus.types import Role, Wait
from amongus.world.engine import Engine


class _StubPolicy:
    """A do-nothing Policy that still carries an AgentMemory, like the real policies do."""

    def __init__(self) -> None:
        """Give this stub the same `.memory` shape MeetingProtocol expects to find."""
        self.memory = AgentMemory()

    def decide(self, obs) -> Wait:
        """Always wait; only used to keep the engine's tick pipeline happy between meetings."""
        return Wait()


class _FixedDeliberator:
    """Returns the same pre-scripted Statement for an agent regardless of round."""

    def __init__(self, statements: dict[str, Statement]) -> None:
        """Store one Statement per speaker id."""
        self.statements = statements

    def speak(self, ctx) -> Statement:
        """Return the scripted Statement for this speaker."""
        return self.statements[ctx.self_id]


class _RaisingDeliberator:
    """A Deliberator that always raises, to exercise the LLM_PARSE_FAIL safety net."""

    def speak(self, ctx):
        """Simulate a malformed/uncaught LLM failure."""
        raise ValueError("boom")


def _build_stub_engine(n_players: int = 4, seed: int = 1) -> Engine:
    """Build an engine with n_players stub policies (each carrying an AgentMemory)."""
    config = replace(SimConfig(), n_players=n_players, n_impostors=1, seed=seed)
    ids = list(config.colors[:n_players])
    return Engine(config, {aid: _StubPolicy() for aid in ids})


def _fire_meeting(
    engine: Engine, protocol: MeetingProtocol, reporter: str, victim: str
) -> None:
    """Simulate step 6's REPORT bookkeeping, then invoke the protocol directly."""
    engine.world.meeting_count += 1
    meeting_no = engine.world.meeting_count
    engine.emit("REPORT", agent=reporter, victim=victim, room="cafeteria", meeting=meeting_no)
    protocol(engine, "report", victim)


def test_meeting_emits_start_statement_suspicion_vote_eject() -> None:
    """A meeting logs one MEETING_START, 2 rounds of statements, votes, and one EJECT."""
    engine = _build_stub_engine(n_players=4)
    ids = list(engine.world.agents)
    stmt = {aid: Statement(speaker=aid, text="x", suspicion={}, vote=None) for aid in ids}
    protocol = MeetingProtocol(engine.config, deliberator=_FixedDeliberator(stmt))
    _fire_meeting(engine, protocol, reporter=ids[0], victim=ids[1])

    types = [e.type for e in engine.events]
    assert types.count("MEETING_START") == 1
    assert types.count("STATEMENT") == len(ids) * engine.config.meeting_rounds
    assert types.count("SUSPICION") == len(ids) * engine.config.meeting_rounds
    assert types.count("VOTE") == len(ids)  # round 2 only
    assert types.count("EJECT") == 1
    start = next(e for e in engine.events if e.type == "MEETING_START")
    assert start.data["reason"] == "report"
    assert set(start.data["alive"]) == set(ids)


def test_plurality_ejects_the_top_target() -> None:
    """A clear plurality vote ejects that agent and reveals its role."""
    engine = _build_stub_engine(n_players=4)
    ids = list(engine.world.agents)
    target = ids[1]
    engine.world.agents[target].role = Role.IMPOSTOR
    stmt = {
        ids[0]: Statement(ids[0], "x", {target: 0.9}, vote=target),
        ids[1]: Statement(ids[1], "x", {}, vote=None),
        ids[2]: Statement(ids[2], "x", {target: 0.9}, vote=target),
        ids[3]: Statement(ids[3], "x", {target: 0.9}, vote=target),
    }
    protocol = MeetingProtocol(engine.config, deliberator=_FixedDeliberator(stmt))
    _fire_meeting(engine, protocol, reporter=ids[0], victim=ids[2])

    eject = next(e for e in engine.events if e.type == "EJECT")
    assert eject.data["target"] == target
    assert eject.data["was_impostor"] is True
    assert engine.world.agents[target].alive is False


def test_tie_ejects_nobody() -> None:
    """Two agents tied for the top vote count ejects nobody."""
    engine = _build_stub_engine(n_players=4)
    ids = list(engine.world.agents)
    stmt = {
        ids[0]: Statement(ids[0], "x", {}, vote=ids[2]),
        ids[1]: Statement(ids[1], "x", {}, vote=ids[3]),
        ids[2]: Statement(ids[2], "x", {}, vote=None),
        ids[3]: Statement(ids[3], "x", {}, vote=None),
    }
    protocol = MeetingProtocol(engine.config, deliberator=_FixedDeliberator(stmt))
    _fire_meeting(engine, protocol, reporter=ids[0], victim=ids[1])

    eject = next(e for e in engine.events if e.type == "EJECT")
    assert eject.data["target"] is None
    assert eject.data["was_impostor"] is None
    assert all(a.alive for a in engine.world.agents.values())


def test_skips_at_or_above_top_count_ejects_nobody() -> None:
    """Skips >= the top target's vote count ejects nobody."""
    engine = _build_stub_engine(n_players=4)
    ids = list(engine.world.agents)
    stmt = {
        ids[0]: Statement(ids[0], "x", {}, vote=ids[1]),
        ids[1]: Statement(ids[1], "x", {}, vote=None),
        ids[2]: Statement(ids[2], "x", {}, vote=None),
        ids[3]: Statement(ids[3], "x", {}, vote=None),
    }
    protocol = MeetingProtocol(engine.config, deliberator=_FixedDeliberator(stmt))
    _fire_meeting(engine, protocol, reporter=ids[0], victim=ids[1])

    eject = next(e for e in engine.events if e.type == "EJECT")
    assert eject.data["target"] is None


def test_ejection_adds_meeting_note_for_survivors() -> None:
    """A resolved ejection appends a 'meeting'-kind note to every surviving policy's memory."""
    engine = _build_stub_engine(n_players=4)
    ids = list(engine.world.agents)
    target = ids[1]
    stmt = {
        ids[0]: Statement(ids[0], "x", {target: 0.9}, vote=target),
        ids[1]: Statement(ids[1], "x", {}, vote=None),
        ids[2]: Statement(ids[2], "x", {target: 0.9}, vote=target),
        ids[3]: Statement(ids[3], "x", {}, vote=None),
    }
    protocol = MeetingProtocol(engine.config, deliberator=_FixedDeliberator(stmt))
    _fire_meeting(engine, protocol, reporter=ids[0], victim=ids[2])

    survivor = ids[0]
    notes = engine.policies[survivor].memory.notes
    assert any(n.kind == "meeting" for n in notes)
    assert any(n.kind == "meeting" for n in engine.notes[survivor])


def test_impostor_vote_for_partner_is_sanitized_to_skip() -> None:
    """A Deliberator that votes an impostor's own partner is overridden to a skip."""
    engine = _build_stub_engine(n_players=4)
    ids = list(engine.world.agents)
    imp_a, imp_b = ids[0], ids[1]
    engine.world.agents[imp_a].role = Role.IMPOSTOR
    engine.world.agents[imp_b].role = Role.IMPOSTOR
    for aid in ids[2:]:
        engine.world.agents[aid].role = Role.CREWMATE
    stmt = {aid: Statement(aid, "x", {}, vote=None) for aid in ids}
    stmt[imp_a] = Statement(imp_a, "x", {}, vote=imp_b)  # votes its own partner
    protocol = MeetingProtocol(engine.config, deliberator=_FixedDeliberator(stmt))
    _fire_meeting(engine, protocol, reporter=ids[2], victim=ids[3])

    vote_event = next(e for e in engine.events if e.type == "VOTE" and e.data["agent"] == imp_a)
    assert vote_event.data["target"] is None


def test_llm_parse_fail_never_crashes_and_falls_back() -> None:
    """A raising Deliberator logs LLM_PARSE_FAIL and still produces a valid Statement."""
    engine = _build_stub_engine(n_players=4)
    ids = list(engine.world.agents)
    protocol = MeetingProtocol(engine.config, deliberator=_RaisingDeliberator())
    _fire_meeting(engine, protocol, reporter=ids[0], victim=ids[1])

    fail_events = [e for e in engine.events if e.type == "LLM_PARSE_FAIL"]
    assert len(fail_events) == len(ids) * engine.config.meeting_rounds
    assert any(e.type == "EJECT" for e in engine.events)


def _build_full_engine(seed: int, max_ticks: int = 200, **config_overrides) -> Engine:
    """Build an engine wired end-to-end with real crewmate/impostor policies and meetings.

    All 8 policies share one ReactorBoard, as real wiring must (§12.3): the protocol only
    produces one identical assignment per game if every agent negotiates over the same board.
    """
    config = replace(SimConfig(), seed=seed, max_ticks=max_ticks, **config_overrides)
    ids = list(config.colors[: config.n_players])
    protocol = MeetingProtocol(config)
    engine = Engine(config, {}, meeting_handler=protocol)
    board = ReactorBoard()
    policies = {}
    for aid in ids:
        role = engine.world.agents[aid].role
        if role is Role.CREWMATE:
            policies[aid] = CrewmatePolicy(aid, config, emit=engine.emit, reactor_board=board)
        else:
            policies[aid] = ImpostorPolicy(aid, config, emit=engine.emit, reactor_board=board)
    engine.policies = policies
    return engine


@pytest.mark.parametrize("seed", range(20))
def test_20_seeds_reach_terminal_state_offline(seed: int) -> None:
    """20 consecutive seeds each run to a winner or a draw with no exception, fully offline."""
    engine = _build_full_engine(seed)
    engine.run(engine.config.max_ticks)
    assert engine.world.phase.value == "over"
    winner = engine.world.winner
    assert winner is None or winner in (Role.CREWMATE, Role.IMPOSTOR)


def test_determinism_with_policies_and_meetings() -> None:
    """The same seed, run twice through real policies and meetings, is byte-identical."""

    def event_hash(engine: Engine) -> str:
        blob = json.dumps([asdict(e) for e in engine.events], sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    e1 = _build_full_engine(seed=42)
    e1.run(e1.config.max_ticks)
    e2 = _build_full_engine(seed=42)
    e2.run(e2.config.max_ticks)
    assert event_hash(e1) == event_hash(e2)
    assert len(e1.events) > 0
