"""Tests for lights/doors/reactor sabotage state machines and stall detection (§12, §13)."""

import hashlib
import json
from dataclasses import asdict, fields, replace

from amongus.config import SimConfig
from amongus.types import DoTask, HoldPanel, PressButton, Role, Sabotage, Transit, Wait
from amongus.world.engine import Engine
from amongus.world.map import GraphView
from amongus.world.observation import p_miss_eff

from .helpers import ScriptedPolicy, safe_engine


def _find(engine: Engine, role: Role) -> list:
    """Return alive agents of the given role, in fixed id order."""
    return [a for a in engine.world.agents.values() if a.role is role]


def test_lights_miss_rate_crewmate_vs_impostor() -> None:
    """During lights, a crewmate's miss rate is p_miss_lights, an impostor's stays p_miss."""
    engine, ids = safe_engine(seed=1)
    imp = _find(engine, Role.IMPOSTOR)[0]
    crew = _find(engine, Role.CREWMATE)[0]
    engine.policies[imp.id] = ScriptedPolicy([Sabotage(kind="lights")])
    engine.run(1)
    assert engine.world.sabotage is not None and engine.world.sabotage.kind == "lights"
    assert p_miss_eff(engine.world, crew, engine.config) == engine.config.p_miss_lights
    assert p_miss_eff(engine.world, imp, engine.config) == engine.config.p_miss


def test_lights_clears_after_panel_hold_ticks_in_electrical() -> None:
    """Lights stays active until electrical is held for panel_hold_ticks, then SABOTAGE_FIXED."""
    engine, ids = safe_engine(seed=2, panel_hold_ticks=2)
    imp = _find(engine, Role.IMPOSTOR)[0]
    crew = _find(engine, Role.CREWMATE)[0]
    crew.room = "electrical"
    engine.policies[imp.id] = ScriptedPolicy([Sabotage(kind="lights")])
    engine.policies[crew.id] = ScriptedPolicy(
        [Wait(), HoldPanel(room="electrical"), HoldPanel(room="electrical")]
    )
    engine.run(1)
    assert engine.world.sabotage is not None
    engine.run(1)  # first hold: 1/2
    assert engine.world.sabotage is not None
    engine.run(1)  # second hold: 2/2, fixed
    assert engine.world.sabotage is None
    fixed = [e for e in engine.events if e.type == "SABOTAGE_FIXED"]
    assert len(fixed) == 1
    assert fixed[0].data["kind"] == "lights"
    assert fixed[0].data["ticks"] == 2


def test_doors_close_exact_incident_edges_and_graphview_sees_it() -> None:
    """Doors closes exactly the corridors incident to the target room, visible the same tick."""
    engine, ids = safe_engine(seed=3, doors_duration=4)
    imp = _find(engine, Role.IMPOSTOR)[0]
    engine.policies[imp.id] = ScriptedPolicy([Sabotage(kind="doors", target="electrical")])
    engine.run(1)
    expected = {frozenset({"electrical", "storage"}), frozenset({"electrical", "lower_engine"})}
    assert engine.world.closed_edges == expected
    view = GraphView(closed_edges=frozenset(engine.world.closed_edges))
    assert view.neighbors("electrical") == []


def test_doors_vents_unaffected() -> None:
    """Vent links incident to a doors-closed room remain usable."""
    engine, ids = safe_engine(seed=4, doors_duration=4)
    imp = _find(engine, Role.IMPOSTOR)[0]
    engine.policies[imp.id] = ScriptedPolicy([Sabotage(kind="doors", target="electrical")])
    engine.run(1)
    view = GraphView(closed_edges=frozenset(engine.world.closed_edges), allow_vents=True)
    neighbors = dict(view.neighbors("electrical"))
    assert "medbay" in neighbors and "security" in neighbors


def test_doors_transit_continues_across_closed_edge() -> None:
    """An agent already in transit across a to-be-closed edge still arrives."""
    engine, ids = safe_engine(seed=5, doors_duration=4)
    imp = _find(engine, Role.IMPOSTOR)[0]
    crew = _find(engine, Role.CREWMATE)[0]
    crew.room = None
    crew.transit = Transit(src="storage", dst="electrical", remaining=2)
    engine.policies[imp.id] = ScriptedPolicy([Sabotage(kind="doors", target="electrical")])
    engine.run(1)  # trigger; crew's remaining 2 -> 1
    assert crew.room is None
    engine.run(1)  # crew's remaining 1 -> 0, arrives despite the closure
    assert crew.room == "electrical"


def test_doors_reopen_exactly_on_time() -> None:
    """Doors reopen exactly doors_duration ticks after the sabotage starts."""
    engine, ids = safe_engine(seed=6, doors_duration=4)
    imp = _find(engine, Role.IMPOSTOR)[0]
    engine.policies[imp.id] = ScriptedPolicy([Sabotage(kind="doors", target="electrical")])
    engine.run(3)
    assert engine.world.sabotage is not None
    assert len(engine.world.closed_edges) == 2
    engine.run(1)  # 4th tick: reopen
    assert engine.world.sabotage is None
    assert engine.world.closed_edges == set()
    opened = [e for e in engine.events if e.type == "DOORS_OPEN"]
    assert len(opened) == 1
    assert opened[0].data["room"] == "electrical"
    edge_set = {frozenset(e) for e in opened[0].data["edges"]}
    expected = {frozenset({"electrical", "storage"}), frozenset({"electrical", "lower_engine"})}
    assert edge_set == expected


def test_reactor_panel_holds_accumulate_noncontiguously_and_across_agents() -> None:
    """Panel holds accumulate across a gap tick and across different agents."""
    engine, ids = safe_engine(seed=7, panel_hold_ticks=2)
    imp = _find(engine, Role.IMPOSTOR)[0]
    crewA, crewB = _find(engine, Role.CREWMATE)
    crewA.room = "reactor"
    crewB.room = "reactor"
    engine.policies[imp.id] = ScriptedPolicy([Sabotage(kind="reactor")])
    # crewA holds tick2, nobody holds tick3 (gap), crewB holds tick4 -> still reaches 2/2.
    engine.policies[crewA.id] = ScriptedPolicy([Wait(), HoldPanel(room="reactor"), Wait(), Wait()])
    engine.policies[crewB.id] = ScriptedPolicy([Wait(), Wait(), Wait(), HoldPanel(room="reactor")])
    engine.run(4)
    assert engine.world.sabotage is not None  # o2 panel untouched, reactor not fully fixed
    assert engine.world.sabotage.panel_holds["reactor"] == 2
    assert engine.world.sabotage.done["reactor"] is True
    done_events = [e for e in engine.events if e.type == "PANEL_DONE"]
    assert len(done_events) == 1
    assert done_events[0].data["room"] == "reactor"
    assert done_events[0].data["agent"] == crewB.id


def test_reactor_timer_zero_is_impostor_win() -> None:
    """A reactor timer reaching zero before both panels are done ends the game for impostors."""
    engine, ids = safe_engine(seed=8, reactor_timer=1)
    imp = _find(engine, Role.IMPOSTOR)[0]
    engine.policies[imp.id] = ScriptedPolicy([Sabotage(kind="reactor")])
    engine.run(1)
    assert engine.world.winner is Role.IMPOSTOR
    game_over = next(e for e in engine.events if e.type == "GAME_OVER")
    assert game_over.data["reason"] == "reactor_timer"


def test_no_meeting_while_reactor_active_then_fires_when_fixed() -> None:
    """A queued meeting stays queued through an active reactor sabotage and fires once fixed."""
    config = replace(SimConfig(seed=9), n_players=4, n_impostors=1, panel_hold_ticks=2)
    ids = list(config.colors[: config.n_players])
    engine = Engine(config, {aid: ScriptedPolicy([]) for aid in ids})
    imp = _find(engine, Role.IMPOSTOR)[0]
    crew = _find(engine, Role.CREWMATE)
    presser, holder1, holder2 = crew[0], crew[1], crew[2]
    holder1.room = "reactor"
    holder2.room = "o2"
    engine.policies[imp.id] = ScriptedPolicy([Sabotage(kind="reactor")])
    engine.policies[presser.id] = ScriptedPolicy([Wait(), PressButton()])
    engine.policies[holder1.id] = ScriptedPolicy(
        [Wait(), HoldPanel(room="reactor"), HoldPanel(room="reactor")]
    )
    engine.policies[holder2.id] = ScriptedPolicy(
        [Wait(), HoldPanel(room="o2"), HoldPanel(room="o2")]
    )
    engine.run(1)  # tick1: reactor sabotage starts
    engine.run(1)  # tick2: button pressed, meeting queued but blocked
    assert engine.world.meeting_count == 0
    assert any(e.type == "BUTTON" for e in engine.events)
    engine.run(1)  # tick3: both panels done -> fixed -> queued meeting fires same tick
    assert engine.world.sabotage is None
    assert any(e.type == "SABOTAGE_FIXED" and e.data["kind"] == "reactor" for e in engine.events)
    assert engine.world.meeting_count == 1


def test_at_most_one_active_and_id_order_tiebreak() -> None:
    """Two same-tick Sabotage attempts: only the first agent in id order starts one."""
    config = replace(SimConfig(seed=10), n_players=5, n_impostors=2)
    ids = list(config.colors[: config.n_players])
    engine = Engine(config, {aid: ScriptedPolicy([]) for aid in ids})
    impostors = [
        aid for aid in engine.world.agents if engine.world.agents[aid].role is Role.IMPOSTOR
    ]
    first_id, second_id = impostors[0], impostors[1]
    engine.policies[first_id] = ScriptedPolicy([Sabotage(kind="lights")])
    engine.policies[second_id] = ScriptedPolicy([Sabotage(kind="reactor")])
    engine.run(1)
    assert engine.world.sabotage is not None
    assert engine.world.sabotage.kind == "lights"


def test_sabotage_cooldown_respected_then_released() -> None:
    """A new sabotage is rejected during the shared cooldown and allowed once it elapses."""
    engine, ids = safe_engine(seed=11, panel_hold_ticks=1, sabotage_cooldown=5)
    imp = _find(engine, Role.IMPOSTOR)[0]
    crew = _find(engine, Role.CREWMATE)[0]
    crew.room = "electrical"
    engine.policies[imp.id] = ScriptedPolicy(
        [
            Sabotage(kind="lights"),  # tick1: starts
            Wait(),  # tick2
            Sabotage(kind="doors", target="storage"),  # tick3: rejected, still on cooldown
            Wait(),  # tick4
            Wait(),  # tick5
            Sabotage(kind="doors", target="storage"),  # tick6: cooldown over, starts
        ]
    )
    engine.policies[crew.id] = ScriptedPolicy([Wait(), HoldPanel(room="electrical")])
    engine.run(2)  # tick2: lights fixed (panel_hold_ticks=1)
    assert engine.world.sabotage is None
    engine.run(1)  # tick3: cooldown still active, doors rejected
    assert engine.world.sabotage is None
    engine.run(3)  # ticks4-6
    assert engine.world.sabotage is not None
    assert engine.world.sabotage.kind == "doors"


def test_observation_never_exposes_saboteur_identity() -> None:
    """The public sabotage alarm carries no field identifying who started the sabotage."""
    engine, ids = safe_engine(seed=12)
    imp = _find(engine, Role.IMPOSTOR)[0]
    engine.policies[imp.id] = ScriptedPolicy([Sabotage(kind="lights")])
    engine.run(1)
    crew = _find(engine, Role.CREWMATE)[0]
    obs = engine._observations[crew.id]
    assert obs.sabotage_alarm is not None
    field_names = {f.name for f in fields(obs.sabotage_alarm)}
    assert field_names == {"kind", "panels", "timer", "panel_done"}
    assert imp.id not in str(asdict(obs.sabotage_alarm))


def test_stall_clause_progress_recent_suppresses_stall() -> None:
    """No PROGRESS_STALL while last_progress_tick stays within the deadlock window."""
    engine, ids = safe_engine(seed=13, deadlock_window=3)
    for _ in range(6):
        engine.run(1)
        engine.world.last_progress_tick = engine.world.tick  # simulate continuous progress
    assert not any(e.type == "PROGRESS_STALL" for e in engine.events)


def test_stall_clause_active_sabotage_suppresses_stall() -> None:
    """No PROGRESS_STALL while a sabotage is active, even past the deadlock window."""
    engine, ids = safe_engine(seed=14, deadlock_window=3)
    imp = _find(engine, Role.IMPOSTOR)[0]
    engine.policies[imp.id] = ScriptedPolicy([Sabotage(kind="lights")])
    engine.run(5)
    assert engine.world.sabotage is not None
    assert not any(e.type == "PROGRESS_STALL" for e in engine.events)


def test_stall_clause_recent_meeting_suppresses_stall() -> None:
    """No PROGRESS_STALL until deadlock_window ticks have passed since the last meeting too."""
    engine, ids = safe_engine(seed=15, deadlock_window=5)
    presser = ids[0]
    engine.policies[presser] = ScriptedPolicy([PressButton()])
    engine.run(5)  # meeting fires tick1; tick-last_progress(0)>=5 at tick5, but tick-meeting(1)=4<5
    assert not any(e.type == "PROGRESS_STALL" for e in engine.events)
    engine.run(1)  # tick6: 6-1=5, both clauses now satisfied
    assert any(e.type == "PROGRESS_STALL" for e in engine.events)


def test_stall_all_clauses_true_emits_and_breaks_on_progress() -> None:
    """PROGRESS_STALL fires once all three clauses hold; DEADLOCK_BROKEN fires on progress."""
    engine, ids = safe_engine(seed=16, deadlock_window=5)
    crew = _find(engine, Role.CREWMATE)[0]
    task = crew.tasks[0]
    crew.room = task.room
    engine.policies[crew.id] = ScriptedPolicy([Wait()] * 5 + [DoTask(task_id=task.task_id)])
    engine.run(5)
    stalls = [e for e in engine.events if e.type == "PROGRESS_STALL"]
    assert len(stalls) == 1
    assert stalls[0].tick == 5
    assert stalls[0].data["ticks"] == 5
    engine.run(1)
    broken = [e for e in engine.events if e.type == "DEADLOCK_BROKEN"]
    assert len(broken) == 1
    assert broken[0].tick == 6
    assert broken[0].data["ticks"] == 1


def test_deadlock_protocol_false_suppresses_progress_stall() -> None:
    """With deadlock_protocol=False the stall is never flagged, even when idle past the window."""
    engine, ids = safe_engine(seed=17, deadlock_window=3, deadlock_protocol=False)
    engine.run(6)
    assert not any(e.type == "PROGRESS_STALL" for e in engine.events)


def _event_hash(engine: Engine) -> str:
    """Serialize an engine's event log to a stable hash for determinism comparison."""
    blob = json.dumps([asdict(e) for e in engine.events], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def test_sabotage_determinism() -> None:
    """Two runs with the same seed and scripted sabotage actions replay byte-identically."""

    def build() -> Engine:
        engine, ids = safe_engine(seed=42, doors_duration=3, panel_hold_ticks=2)
        imp = _find(engine, Role.IMPOSTOR)[0]
        crew = _find(engine, Role.CREWMATE)[0]
        crew.room = "electrical"
        engine.policies[imp.id] = ScriptedPolicy(
            [Sabotage(kind="doors", target="electrical")]
        )
        engine.policies[crew.id] = ScriptedPolicy(
            [Wait(), Wait(), Wait(), Wait(), HoldPanel(room="electrical")]
        )
        return engine

    e1 = build()
    e1.run(6)
    e2 = build()
    e2.run(6)
    assert _event_hash(e1) == _event_hash(e2)
    assert len(e1.events) > 0
