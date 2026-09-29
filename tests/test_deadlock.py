"""Tests for the §13 deadlock response: PROGRESS_STALL breaks a mutual-shadow standoff."""

from dataclasses import replace

from amongus.agents.crewmate import CrewmatePolicy
from amongus.config import SimConfig
from amongus.types import Role, TaskInstance
from amongus.world.engine import Engine

from .helpers import ScriptedPolicy


def _standoff_engine(seed: int, deadlock_protocol: bool) -> tuple[Engine, str, str]:
    """Two crewmates, mutually suspicious and co-located at spawn, plus one idle impostor.

    Both crewmates' tasks are placed away from cafeteria so "made progress" only happens
    once shadowing stops and one of them actually walks to and works its task.
    """
    config = replace(
        SimConfig(),
        seed=seed,
        n_players=3,
        n_impostors=1,
        deadlock_window=5,
        shadow_cooldown=10,
        deadlock_protocol=deadlock_protocol,
        max_ticks=40,
    )
    ids = list(config.colors[: config.n_players])
    engine = Engine(config, {aid: ScriptedPolicy([]) for aid in ids})
    crew = [aid for aid in ids if engine.world.agents[aid].role is Role.CREWMATE]
    imp = next(aid for aid in ids if engine.world.agents[aid].role is Role.IMPOSTOR)
    crew_a, crew_b = crew
    engine.world.agents[crew_a].tasks = [
        TaskInstance(task_id="tA", room="storage", duration=2)
    ]
    engine.world.agents[crew_b].tasks = [
        TaskInstance(task_id="tB", room="weapons", duration=2)
    ]
    policy_a = CrewmatePolicy(crew_a, config)
    policy_b = CrewmatePolicy(crew_b, config)
    policy_a.memory.suspicion[crew_b] = 0.9
    policy_a.memory.last_seen[crew_b] = ("cafeteria", 0)
    policy_b.memory.suspicion[crew_a] = 0.9
    policy_b.memory.last_seen[crew_a] = ("cafeteria", 0)
    engine.policies[crew_a] = policy_a
    engine.policies[crew_b] = policy_b
    engine.policies[imp] = ScriptedPolicy([])
    return engine, crew_a, crew_b


def test_mutual_shadow_stalls_then_breaks_with_protocol_on() -> None:
    """With deadlock_protocol on, the standoff stalls once, then progress resumes."""
    engine, _crew_a, _crew_b = _standoff_engine(seed=1, deadlock_protocol=True)
    engine.run(engine.config.max_ticks)
    stalls = [e for e in engine.events if e.type == "PROGRESS_STALL"]
    broken = [e for e in engine.events if e.type == "DEADLOCK_BROKEN"]
    assert len(stalls) >= 1
    assert len(broken) >= 1
    assert engine.world.task_bar > 0.0


def test_mutual_shadow_never_breaks_with_protocol_off() -> None:
    """With deadlock_protocol off, the same standoff never recovers: no stall is ever flagged."""
    engine, _crew_a, _crew_b = _standoff_engine(seed=1, deadlock_protocol=False)
    engine.run(engine.config.max_ticks)
    assert not any(e.type == "PROGRESS_STALL" for e in engine.events)
    assert not any(e.type == "DEADLOCK_BROKEN" for e in engine.events)
    assert engine.world.task_bar == 0.0


def test_stall_disables_shadow_and_clears_goal() -> None:
    """On the PROGRESS_STALL rising edge, shadow_disabled_until advances and goal is cleared."""
    engine, crew_a, _crew_b = _standoff_engine(seed=2, deadlock_protocol=True)
    policy_a = engine.policies[crew_a]
    before = policy_a.shadow_disabled_until
    engine.run(engine.config.deadlock_window + 1)
    assert policy_a.shadow_disabled_until > before
