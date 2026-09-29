"""Tests for §7 observation building: locality, the lights asymmetry, and no WorldState leak."""

import dataclasses
from dataclasses import replace

from amongus.config import SimConfig
from amongus.rng import make_rng
from amongus.types import AgentPhys, Phase, Role, WorldState
from amongus.world.observation import Observation, TickSignals, build_observation, p_miss_eff
from amongus.world.sabotage import SabotageState


def _two_agent_world(sabotage: SabotageState | None) -> WorldState:
    """Build a minimal two-agent world (one crewmate, one impostor) in the same room."""
    crew = AgentPhys(
        id="red", role=Role.CREWMATE, alive=True, room="electrical", transit=None,
        tasks=[], kill_cooldown=0, button_used=False,
    )
    imp = AgentPhys(
        id="blue", role=Role.IMPOSTOR, alive=True, room="electrical", transit=None,
        tasks=[], kill_cooldown=0, button_used=False,
    )
    return WorldState(
        tick=1, phase=Phase.PLAY, agents={"red": crew, "blue": imp}, bodies=[],
        sabotage=sabotage, closed_edges=set(), task_bar=0.0, last_progress_tick=0,
        meeting_count=0, winner=None,
    )


def test_no_worldstate_leak() -> None:
    """No field of Observation is, or transitively holds, a WorldState instance."""
    config = SimConfig()
    world = _two_agent_world(sabotage=None)
    rng = make_rng(0)
    obs = build_observation(world, "red", config, rng, TickSignals())
    seen: set[int] = set()

    def walk(value) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        assert not isinstance(value, WorldState)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            for f in dataclasses.fields(value):
                walk(getattr(value, f.name))
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for v in value:
                walk(v)

    walk(obs)
    assert isinstance(obs, Observation)


def test_lights_asymmetry_function() -> None:
    """p_miss_eff raises the crewmate's miss rate during lights, leaves the impostor's alone."""
    config = SimConfig()
    lit = SabotageState(kind="lights", timer=5)
    world = _two_agent_world(sabotage=lit)
    assert p_miss_eff(world, world.agents["red"], config) == config.p_miss_lights
    assert p_miss_eff(world, world.agents["blue"], config) == config.p_miss
    world_no_lights = _two_agent_world(sabotage=None)
    assert p_miss_eff(world_no_lights, world_no_lights.agents["red"], config) == config.p_miss


def test_lights_asymmetry_statistical() -> None:
    """Over many samples, crewmate sighting rate tracks p_miss_lights; impostor's tracks p_miss."""
    config = replace(SimConfig(), p_miss=0.02, p_miss_lights=0.5)
    lit = SabotageState(kind="lights", timer=5)
    world = _two_agent_world(sabotage=lit)
    rng = make_rng(42)
    n = 20000
    crew_sees = 0
    imp_sees = 0
    for _ in range(n):
        obs_crew = build_observation(world, "red", config, rng, TickSignals())
        if "blue" in obs_crew.occupants:
            crew_sees += 1
        obs_imp = build_observation(world, "blue", config, rng, TickSignals())
        if "red" in obs_imp.occupants:
            imp_sees += 1
    crew_rate = crew_sees / n
    imp_rate = imp_sees / n
    assert abs(crew_rate - (1 - config.p_miss_lights)) < 0.02
    assert abs(imp_rate - (1 - config.p_miss)) < 0.02


def test_occupants_independent_and_local_only_in_room() -> None:
    """An agent not in a room gets no local fields; occupants excludes self and dead agents."""
    config = SimConfig()
    world = _two_agent_world(sabotage=None)
    world.agents["red"].room = None  # simulate transit
    rng = make_rng(1)
    obs = build_observation(world, "red", config, rng, TickSignals())
    assert obs.room is None
    assert obs.occupants == []
    assert obs.bodies_here == []
    assert obs.local_events == []
