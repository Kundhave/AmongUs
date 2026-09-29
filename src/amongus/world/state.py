"""WorldState construction and mutation helpers (SPEC §4, §6.1 setup)."""

import numpy as np

from amongus.config import SimConfig
from amongus.types import AgentPhys, Phase, Role, TaskInstance, WorldState
from amongus.world.map import TASK_POOL

SPAWN_ROOM = "cafeteria"  # fixed by §6.1 setup / §6.2 step 10, not a scenario tunable


def new_world(config: SimConfig, rng: np.random.Generator) -> WorldState:
    """Build the initial WorldState: roles, dealt tasks, spawn, cooldowns (§6.1)."""
    ids = list(config.colors[: config.n_players])
    impostor_idx = rng.choice(len(ids), size=config.n_impostors, replace=False)
    impostor_ids = {ids[i] for i in impostor_idx}
    agents: dict[str, AgentPhys] = {}
    for agent_id in ids:
        role = Role.IMPOSTOR if agent_id in impostor_ids else Role.CREWMATE
        agents[agent_id] = AgentPhys(
            id=agent_id,
            role=role,
            alive=True,
            room=SPAWN_ROOM,
            transit=None,
            tasks=_deal_tasks(rng, config.tasks_per_crewmate, config.task_duration_noise),
            kill_cooldown=config.kill_cooldown,
            button_used=False,
        )
    return WorldState(
        tick=0,
        phase=Phase.PLAY,
        agents=agents,
        bodies=[],
        sabotage=None,
        closed_edges=set(),
        task_bar=0.0,
        last_progress_tick=0,
        meeting_count=0,
        winner=None,
    )


def _deal_tasks(rng: np.random.Generator, count: int, duration_noise: int) -> list[TaskInstance]:
    """Draw `count` distinct tasks; duration = pool base +/- uniform noise, floored at 1 (§6.1)."""
    idx = rng.choice(len(TASK_POOL), size=count, replace=False)
    tasks = []
    for i in idx:
        task_id, room, base = TASK_POOL[i]
        delta = int(rng.integers(-duration_noise, duration_noise + 1))
        tasks.append(TaskInstance(task_id=task_id, room=room, duration=max(1, base + delta)))
    return tasks


def real_task_progress(world: WorldState) -> tuple[int, int]:
    """Return (progress_sum, duration_sum) over real crewmate tasks only, for task_bar."""
    progress = 0
    duration = 0
    for agent in world.agents.values():
        if agent.role is not Role.CREWMATE:
            continue
        for task in agent.tasks:
            progress += task.progress
            duration += task.duration
    return progress, duration
