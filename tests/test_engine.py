"""Tests for the §6.2 tick pipeline: setup, movement, vents, work, determinism."""

import hashlib
import json
from dataclasses import asdict, replace

from amongus.config import SimConfig
from amongus.rng import make_rng
from amongus.types import (
    Body,
    DoTask,
    Kill,
    Move,
    Phase,
    PressButton,
    Report,
    Role,
    Transit,
    Vent,
    Wait,
)
from amongus.world.engine import Engine
from amongus.world.map import TASK_POOL
from amongus.world.sabotage import SabotageState
from amongus.world.state import new_world

from .helpers import RandomWalkPolicy, ScriptedPolicy


def _event_hash(engine: Engine) -> str:
    """Serialize an engine's event log to a stable hash for determinism comparison."""
    blob = json.dumps([asdict(e) for e in engine.events], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def test_setup_roles_and_tasks() -> None:
    """Setup assigns n_impostors, deals distinct tasks, spawns everyone in cafeteria."""
    config = SimConfig(seed=1)
    policies = {c: ScriptedPolicy([]) for c in config.colors[: config.n_players]}
    engine = Engine(config, policies)
    impostors = [a for a in engine.world.agents.values() if a.role is Role.IMPOSTOR]
    assert len(impostors) == config.n_impostors
    for agent in engine.world.agents.values():
        assert agent.room == "cafeteria"
        assert agent.kill_cooldown == config.kill_cooldown
        assert len(agent.tasks) == config.tasks_per_crewmate
        assert len({t.task_id for t in agent.tasks}) == config.tasks_per_crewmate


def test_task_duration_noise_varies_and_floors_at_one() -> None:
    """Task durations differ from the pool base by at most task_duration_noise, never below 1."""
    base_by_id = {t: d for t, _room, d in TASK_POOL}
    config = replace(SimConfig(), task_duration_noise=3, tasks_per_crewmate=8)
    saw_nonzero_delta = False
    for seed in range(10):
        world = new_world(config, make_rng(seed))
        for agent in world.agents.values():
            for task in agent.tasks:
                assert task.duration >= 1
                delta = task.duration - base_by_id[task.task_id]
                assert abs(delta) <= config.task_duration_noise
                if delta != 0:
                    saw_nonzero_delta = True
    assert saw_nonzero_delta


def test_determinism() -> None:
    """Two runs with the same seed produce a byte-identical event log."""
    config = SimConfig(seed=7)
    ids = list(config.colors[: config.n_players])

    def build() -> Engine:
        policies = {aid: RandomWalkPolicy(seed=100 + i) for i, aid in enumerate(ids)}
        return Engine(config, policies)

    e1 = build()
    e1.run(25)
    e2 = build()
    e2.run(25)
    assert _event_hash(e1) == _event_hash(e2)
    assert len(e1.events) > 0


def test_transit_hides_agent_and_blocks_action() -> None:
    """An agent in transit is absent from occupants and its policy is not called."""
    # 3 players (2 crew, 1 impostor) so alive_impostors < alive_crew and the game
    # doesn't immediately end under the §6.2 step 8 win check.
    config = replace(SimConfig(), n_players=3, n_impostors=1, p_miss=0.0, seed=3)
    ids = list(config.colors[: config.n_players])
    calls: dict[str, int] = {aid: 0 for aid in ids}

    class CountingScript:
        def __init__(self, actions):
            self._actions = actions
            self._i = 0

        def decide(self, obs):
            calls[obs.self_id] += 1
            if self._i >= len(self._actions):
                return Wait()
            action = self._actions[self._i]
            self._i += 1
            return action

    mover, stayer, extra = ids[0], ids[1], ids[2]
    policies = {
        mover: CountingScript([Move(to="weapons")]),
        stayer: CountingScript([]),
        extra: CountingScript([]),
    }
    engine = Engine(config, policies)
    engine.tick()  # tick 1: mover starts a transit toward weapons (weight 3)
    assert engine.world.agents[mover].room is None
    assert calls[mover] == 1
    for _ in range(2):  # ticks 2-3: still in transit, must not be asked to decide
        engine.tick()
        stayer_obs = engine._observations[stayer]
        assert mover not in stayer_obs.occupants
    assert calls[mover] == 1  # unchanged while in transit
    engine.tick()  # tick 4: arrives (remaining hits 0)
    assert engine.world.agents[mover].room == "weapons"


def test_vent_rejected_for_crewmate() -> None:
    """A crewmate's Vent action is rejected silently: no move, and no VENT event logged.

    vent_rejected itself is internal-only (filtered before Engine.events, not a §14.1 type).
    """
    config = replace(SimConfig(), n_players=2, n_impostors=0, seed=5)
    ids = list(config.colors[: config.n_players])
    policies = {aid: ScriptedPolicy([Vent(to="admin")]) for aid in ids}
    engine = Engine(config, policies)
    engine.tick()
    for agent in engine.world.agents.values():
        assert agent.room == "cafeteria"
    assert not any(e.type == "VENT" for e in engine.events)


def test_vent_works_for_impostor() -> None:
    """An impostor's Vent arrives the same tick, no transit involved."""
    config = replace(SimConfig(), n_players=1, n_impostors=1, seed=9)
    policies = {config.colors[0]: ScriptedPolicy([Vent(to="admin")])}
    engine = Engine(config, policies)
    engine.tick()
    agent = engine.world.agents[config.colors[0]]
    assert agent.room == "admin"
    assert agent.transit is None


def test_dotask_outside_room_is_noop() -> None:
    """DoTask does not advance progress when the agent is not in the task's room."""
    config = replace(SimConfig(), n_players=1, n_impostors=0, seed=11)
    aid = config.colors[0]
    engine = Engine(config, {aid: ScriptedPolicy([])})
    task = engine.world.agents[aid].tasks[0]
    if task.room == "cafeteria":
        task = next(t for t in engine.world.agents[aid].tasks if t.room != "cafeteria")
    engine.policies[aid] = ScriptedPolicy([DoTask(task_id=task.task_id)])
    engine.tick()
    assert task.progress == 0
    assert engine.world.task_bar == 0.0


def test_task_bar_excludes_impostor_fake_tasks() -> None:
    """task_bar only reflects real crewmate task progress, never impostor fake tasks."""
    config = replace(SimConfig(), n_players=2, n_impostors=1, seed=13)
    ids = list(config.colors[: config.n_players])
    engine = Engine(config, {aid: ScriptedPolicy([]) for aid in ids})
    crew = next(a for a in engine.world.agents.values() if a.role is Role.CREWMATE)
    imp = next(a for a in engine.world.agents.values() if a.role is Role.IMPOSTOR)
    crew_task, imp_task = crew.tasks[0], imp.tasks[0]
    crew.room, imp.room = crew_task.room, imp_task.room  # place each in its own task's room
    engine.policies[crew.id] = ScriptedPolicy([DoTask(task_id=crew_task.task_id)])
    engine.policies[imp.id] = ScriptedPolicy([DoTask(task_id=imp_task.task_id)])
    engine.tick()
    progress, duration = 0, 0
    for a in engine.world.agents.values():
        if a.role is Role.CREWMATE:
            for t in a.tasks:
                progress += t.progress
                duration += t.duration
    assert engine.world.task_bar == progress / duration
    assert imp_task.progress == 1  # advanced physically, but excluded from task_bar


def _safe_engine(seed: int) -> tuple[Engine, list[str]]:
    """3 players (2 crew, 1 impostor), idle, so no win condition fires by itself."""
    config = replace(SimConfig(), n_players=3, n_impostors=1, seed=seed)
    ids = list(config.colors[: config.n_players])
    engine = Engine(config, {aid: ScriptedPolicy([]) for aid in ids})
    return engine, ids


def test_report_noop_without_body() -> None:
    """Report with no body in the agent's room is a no-op: no event, no meeting flagged."""
    engine, ids = _safe_engine(seed=20)
    engine.policies[ids[0]] = ScriptedPolicy([Report()])
    engine.tick()
    assert not any(e.type == "REPORT" for e in engine.events)
    assert engine.world.meeting_count == 0


def test_report_noop_when_already_reported() -> None:
    """Report against an already-reported body is a no-op."""
    engine, ids = _safe_engine(seed=21)
    room = engine.world.agents[ids[0]].room
    engine.world.bodies.append(Body(victim=ids[1], room=room, tick=0, reported=True))
    engine.policies[ids[0]] = ScriptedPolicy([Report()])
    engine.tick()
    assert not any(e.type == "REPORT" for e in engine.events)
    assert engine.world.meeting_count == 0


def test_report_success_flags_meeting_and_resets_state() -> None:
    """A valid Report emits REPORT, marks the body reported, and triggers the step-10 reset."""
    engine, ids = _safe_engine(seed=22)
    room = engine.world.agents[ids[0]].room
    engine.world.bodies.append(Body(victim=ids[1], room=room, tick=0, reported=False))
    mover = engine.world.agents[ids[2]]
    mover.room = None  # simulate an in-progress transit that must be canceled by the reset
    mover.transit = Transit(src="cafeteria", dst="weapons", remaining=2)
    engine.policies[ids[0]] = ScriptedPolicy([Report()])
    engine.tick()
    report_events = [e for e in engine.events if e.type == "REPORT"]
    assert len(report_events) == 1
    data = report_events[0].data
    assert data["agent"] == ids[0]
    assert data["victim"] == ids[1]
    assert data["room"] == room
    assert data["meeting"] == 1
    assert data["task_bar"] == engine.world.task_bar  # every event stamps task_bar (§14.1)
    assert engine.world.bodies[0].reported is True
    assert engine.world.meeting_count == 1
    for agent in engine.world.agents.values():
        if agent.alive:
            assert agent.room == "cafeteria"
            assert agent.transit is None
            assert agent.kill_cooldown == engine.config.kill_cooldown
    assert engine.world.phase is Phase.PLAY


def test_button_noop_outside_button_room() -> None:
    """PressButton outside config.button_room is a no-op."""
    engine, ids = _safe_engine(seed=23)
    engine.world.agents[ids[0]].room = "electrical"
    engine.policies[ids[0]] = ScriptedPolicy([PressButton()])
    engine.tick()
    assert not any(e.type == "BUTTON" for e in engine.events)
    assert engine.world.meeting_count == 0


def test_button_noop_when_used_twice() -> None:
    """A second PressButton by the same agent is a no-op once button_used is set."""
    engine, ids = _safe_engine(seed=24)
    engine.world.agents[ids[0]].button_used = True
    engine.policies[ids[0]] = ScriptedPolicy([PressButton()])
    engine.tick()
    assert not any(e.type == "BUTTON" for e in engine.events)


def test_meeting_queued_while_reactor_sabotage_active() -> None:
    """A meeting flag stays queued while a reactor sabotage is active, firing once it clears."""
    engine, ids = _safe_engine(seed=25)
    engine.world.sabotage = SabotageState(kind="reactor", timer=10)
    room = engine.world.agents[ids[0]].room
    engine.world.bodies.append(Body(victim=ids[1], room=room, tick=0, reported=False))
    engine.policies[ids[0]] = ScriptedPolicy([Report()])
    engine.tick()
    assert engine.world.meeting_count == 0  # queued, not fired
    engine.world.sabotage = None  # reactor fixed
    engine.tick()
    assert engine.world.meeting_count == 1


def test_win_all_real_tasks_done_crew_wins() -> None:
    """Crew wins once every real crewmate task is complete."""
    config = replace(SimConfig(), n_players=1, n_impostors=0, seed=30)
    aid = config.colors[0]
    engine = Engine(config, {aid: ScriptedPolicy([])})
    for task in engine.world.agents[aid].tasks:
        task.progress = task.duration
    engine.tick()
    assert engine.world.winner is Role.CREWMATE
    assert engine.world.phase is Phase.OVER
    game_over = next(e for e in engine.events if e.type == "GAME_OVER")
    assert game_over.data["winner"] == "crewmate"


def test_win_impostors_majority() -> None:
    """Impostors win once alive impostors reach or exceed alive crewmates."""
    config = replace(SimConfig(), n_players=2, n_impostors=1, seed=31)
    ids = list(config.colors[: config.n_players])
    engine = Engine(config, {aid: ScriptedPolicy([]) for aid in ids})
    engine.tick()
    assert engine.world.winner is Role.IMPOSTOR
    game_over = next(e for e in engine.events if e.type == "GAME_OVER")
    assert game_over.data["winner"] == "impostor"


def test_win_draw_at_max_ticks() -> None:
    """The game ends in a draw (winner None) once max_ticks is reached."""
    config = replace(SimConfig(), n_players=3, n_impostors=1, max_ticks=2, seed=32)
    ids = list(config.colors[: config.n_players])
    engine = Engine(config, {aid: ScriptedPolicy([]) for aid in ids})
    engine.run(5)  # run() stops early once phase is OVER
    assert engine.world.winner is None
    assert engine.world.phase is Phase.OVER
    assert engine.world.tick == 2
    game_over = next(e for e in engine.events if e.type == "GAME_OVER")
    assert game_over.data["winner"] is None
    assert game_over.data["reason"] == "max_ticks"


# §14.1's closed event-type set, hard-coded so a future silent rename fails loudly rather than
# being masked by a dict.get(..., ()) default. world/ only ever emits a subset of this (meeting
# events come from agents/protocol.py's meeting_handler, not the engine).
_EVENT_TYPES_14_1 = frozenset(
    {
        "SPAWN", "ROLES", "MOVE", "ARRIVE", "VENT", "KILL", "BODY_SEEN", "REPORT", "BUTTON",
        "MEETING_START", "STATEMENT", "SUSPICION", "VOTE", "EJECT", "SABOTAGE", "BID", "COMMIT",
        "REVOKE", "PANEL_DONE", "SABOTAGE_FIXED", "DOORS_OPEN", "REPLAN", "PROGRESS_STALL",
        "DEADLOCK_BROKEN", "LLM_PARSE_FAIL", "SHOCK", "GAME_OVER",
    }
)


def test_move_arrive_vent_stay_inside_the_14_1_closed_schema() -> None:
    """A generated log's MOVE/ARRIVE/VENT events are uppercase and carry their §14.1 fields."""
    config = replace(SimConfig(), n_players=3, n_impostors=1, seed=201)
    ids = list(config.colors[: config.n_players])
    engine = Engine(config, {aid: ScriptedPolicy([]) for aid in ids})
    impostor = next(a for a in engine.world.agents.values() if a.role is Role.IMPOSTOR)
    crew = next(a for a in engine.world.agents.values() if a.role is Role.CREWMATE)
    engine.policies[impostor.id] = ScriptedPolicy([Vent(to="admin")])
    engine.policies[crew.id] = ScriptedPolicy([Move(to="weapons")])
    engine.run(5)  # cafeteria->weapons has weight 3, so ARRIVE lands within 5 ticks

    for event in engine.events:
        assert event.type in _EVENT_TYPES_14_1, event.type
        assert "task_bar" in event.data

    move = next(e for e in engine.events if e.type == "MOVE")
    assert {"agent", "from", "to", "eta"} <= move.data.keys()
    assert move.data == {
        "task_bar": move.data["task_bar"],
        "agent": crew.id,
        "from": "cafeteria",
        "to": "weapons",
        "eta": 3,
    }

    arrive = next(e for e in engine.events if e.type == "ARRIVE")
    assert {"agent", "room"} <= arrive.data.keys()
    assert arrive.data["agent"] == crew.id
    assert arrive.data["room"] == "weapons"

    vent = next(e for e in engine.events if e.type == "VENT")
    assert vent.data == {
        "task_bar": vent.data["task_bar"],
        "agent": impostor.id,
        "from": "cafeteria",
        "to": "admin",
    }


def test_body_seen_emitted_once_per_agent_victim_pair() -> None:
    """A bystander who keeps re-seeing the same body logs BODY_SEEN only on first sighting."""
    config = replace(SimConfig(), n_players=3, n_impostors=1, p_miss=0.0, seed=202)
    ids = list(config.colors[: config.n_players])
    engine = Engine(config, {aid: ScriptedPolicy([]) for aid in ids})
    killer = next(a for a in engine.world.agents.values() if a.role is Role.IMPOSTOR)
    crew = [a for a in engine.world.agents.values() if a.role is Role.CREWMATE]
    target, witness = crew
    killer.kill_cooldown = 0
    engine.policies[killer.id] = ScriptedPolicy([Kill(target=target.id)])
    engine.run(6)  # everyone idle after the kill, so the body stays visible to witness each tick

    body_seen = [
        e
        for e in engine.events
        if e.type == "BODY_SEEN" and e.data["agent"] == witness.id and e.data["victim"] == target.id
    ]
    assert len(body_seen) == 1
    # the underlying perception is unaffected: the witness still perceives the body every tick.
    assert target.id in engine._observations[witness.id].bodies_here
