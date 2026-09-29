"""Pure resolution helpers: arrivals, kills, departures, reports/button, win checks (§6.2)."""

import numpy as np

from amongus.config import SimConfig
from amongus.types import (
    Action,
    AgentId,
    Body,
    Kill,
    Move,
    PressButton,
    Report,
    Role,
    Transit,
    Vent,
    WorldState,
)
from amongus.world.map import GraphView
from amongus.world.observation import p_miss_eff
from amongus.world.state import real_task_progress

EventTuple = tuple[str, dict]


def resolve_arrivals(
    world: WorldState,
    config: SimConfig,
    rng: np.random.Generator,
    actions: dict[AgentId, Action],
    vent_adj: dict[str, set[str]],
) -> tuple[list[EventTuple], list[tuple[AgentId, str, frozenset[AgentId]]]]:
    """Decrement transits, land arrivals, and resolve Vent (arrives this tick) — §6.2 step 3.

    Still-in-transit agents remain unkillable. Move is *not* handled here: it is deferred to
    resolve_departures (step 5), which runs after kills, so a target cannot dodge a same-tick
    kill just by deciding to walk. Each vent arrival's witness set is rolled once, here, and
    is the single source of truth reused by observation.py — never re-rolled at observe time.
    """
    events: list[EventTuple] = []
    for agent_id, agent in world.agents.items():
        if not agent.alive or agent.transit is None:
            continue
        agent.transit.remaining -= 1
        if agent.transit.remaining <= 0:
            agent.room = agent.transit.dst
            agent.transit = None
            events.append(("ARRIVE", {"agent": agent_id, "room": agent.room}))
    vent_events: list[tuple[AgentId, str, frozenset[AgentId]]] = []
    for agent_id, action in actions.items():
        agent = world.agents[agent_id]
        if not agent.alive or agent.transit is not None or not isinstance(action, Vent):
            continue
        event, dst = _try_vent(vent_adj, agent_id, agent, action.to)
        events.append(event)
        if dst is not None:
            witnesses = _witnesses_in_room(world, config, rng, dst, {agent_id})
            vent_events.append((agent_id, dst, witnesses))
    return events, vent_events


def resolve_departures(
    world: WorldState, actions: dict[AgentId, Action]
) -> list[EventTuple]:
    """Start new Move transits for surviving agents — §6.2 step 5, run after kills resolve.

    A killed agent's Move action (decided before it died this tick) is a no-op here: it is
    already dead by this point and cannot depart.
    """
    events: list[EventTuple] = []
    view = GraphView(closed_edges=frozenset(world.closed_edges))
    for agent_id, action in actions.items():
        agent = world.agents[agent_id]
        if not agent.alive or agent.transit is not None or not isinstance(action, Move):
            continue
        events.append(_start_move(view, agent_id, agent, action.to))
    return events


def _start_move(view: GraphView, agent_id: AgentId, agent, dest: str) -> EventTuple:
    """Start a transit toward dest if the edge from agent's room is open, else reject."""
    room = agent.room
    weight = dict(view.neighbors(room)).get(dest)
    if weight is None:
        return ("move_rejected", {"agent": agent_id, "from": room, "to": dest})
    agent.transit = Transit(src=room, dst=dest, remaining=weight)
    agent.room = None
    return ("MOVE", {"agent": agent_id, "from": room, "to": dest, "eta": weight})


def _try_vent(
    vent_adj: dict[str, set[str]], agent_id: AgentId, agent, dest: str
) -> tuple[EventTuple, str | None]:
    """Move an impostor through a vent this same tick; reject silently for a crewmate."""
    room = agent.room
    if agent.role is not Role.IMPOSTOR:
        return ("vent_rejected", {"agent": agent_id, "to": dest, "reason": "crewmate"}), None
    if dest not in vent_adj.get(room, set()):
        return ("vent_rejected", {"agent": agent_id, "to": dest, "reason": "no_vent"}), None
    agent.room = dest
    return ("VENT", {"agent": agent_id, "from": room, "to": dest}), dest


def _witnesses_in_room(
    world: WorldState,
    config: SimConfig,
    rng: np.random.Generator,
    room: str,
    exclude_ids: set[AgentId],
) -> frozenset[AgentId]:
    """Roll, once, which alive agents in room (other than exclude_ids) perceive an event (§7)."""
    witnesses = set()
    for other_id, other in world.agents.items():
        if other_id in exclude_ids or not other.alive or other.room != room:
            continue
        if rng.random() >= p_miss_eff(world, other, config):
            witnesses.add(other_id)
    return frozenset(witnesses)


def resolve_kills(
    world: WorldState, config: SimConfig, rng: np.random.Generator, actions: dict[AgentId, Action]
) -> tuple[list[EventTuple], list[tuple[AgentId, AgentId, str, frozenset[AgentId]]]]:
    """Resolve Kill actions (§6.2 step 4); a duplicate target goes to the first killer by id."""
    events: list[EventTuple] = []
    attempts: dict[AgentId, list[AgentId]] = {}
    for agent_id in world.agents:  # fixed id order
        action = actions.get(agent_id)
        if not isinstance(action, Kill):
            continue
        if _kill_valid(world, agent_id, action.target):
            attempts.setdefault(action.target, []).append(agent_id)
        else:
            events.append(("kill_rejected", {"agent": agent_id, "target": action.target}))
    kill_events: list[tuple[AgentId, AgentId, str, frozenset[AgentId]]] = []
    for target_id, killers in attempts.items():
        killer_id = killers[0]
        for extra in killers[1:]:
            events.append(
                ("kill_rejected", {"agent": extra, "target": target_id, "reason": "duplicate"})
            )
        event, kill_event = _resolve_kill(world, config, rng, killer_id, target_id)
        events.append(event)
        kill_events.append(kill_event)
    return events, kill_events


def _kill_valid(world: WorldState, killer_id: AgentId, target_id: AgentId) -> bool:
    """True if killer_id may legally kill target_id this tick, per §6.2 step 4."""
    killer = world.agents[killer_id]
    target = world.agents.get(target_id)
    if killer.role is not Role.IMPOSTOR or not killer.alive or killer.kill_cooldown != 0:
        return False
    if target is None or not target.alive or target.role is not Role.CREWMATE:
        return False
    return killer.room is not None and killer.room == target.room


def _resolve_kill(
    world: WorldState,
    config: SimConfig,
    rng: np.random.Generator,
    killer_id: AgentId,
    target_id: AgentId,
) -> tuple[EventTuple, tuple[AgentId, AgentId, str, frozenset[AgentId]]]:
    """Kill target_id, drop a Body, reset the killer's cooldown; return the KILL event.

    The witness set is rolled once, here, and is the single source of truth reused by
    observation.py's local_events (and thus notes) — never re-rolled at observe time (§7).
    The victim's incomplete tasks are reassigned to a survivor here too (§6.2 step 4).
    """
    killer = world.agents[killer_id]
    target = world.agents[target_id]
    room = killer.room
    target.alive = False
    world.bodies.append(Body(victim=target_id, room=room, tick=world.tick))
    killer.kill_cooldown = config.kill_cooldown
    _reassign_tasks(world, target_id)
    witnesses = _witnesses_in_room(world, config, rng, room, {killer_id, target_id})
    event = (
        "KILL",
        {
            "killer": killer_id,
            "victim": target_id,
            "room": room,
            "witnesses": sorted(witnesses),
        },
    )
    return event, (killer_id, target_id, room, witnesses)


def _remaining_tasks(agent) -> int:
    """Count an agent's not-yet-done tasks."""
    return sum(1 for t in agent.tasks if not t.done)


def _reassign_tasks(world: WorldState, victim_id: AgentId) -> None:
    """Move all of the dead victim's incomplete tasks onto one survivor (§6.2 step 4).

    The kill target is always a crewmate (§6.2 step 4 validity), so only real tasks ever
    move; there is nothing to reassign for an impostor's fake tasks. The recipient — the
    surviving crewmate with the fewest remaining tasks, ties broken by fixed id order (the
    same world.agents iteration order used to resolve duplicate kill targets) — is chosen
    once, before any transfer, so the victim's tasks all land on a single agent rather than
    scattering as each transfer changes who has the fewest remaining. No RNG. Each
    TaskInstance itself is moved, not copied, so its progress is preserved and the crew's
    total real task count (the task_bar denominator) never changes.
    """
    victim = world.agents[victim_id]
    incomplete = [t for t in victim.tasks if not t.done]
    if not incomplete:
        return
    id_order = {aid: i for i, aid in enumerate(world.agents)}
    survivors = [a for a in world.agents.values() if a.alive and a.role is Role.CREWMATE]
    if not survivors:
        return  # no crewmate left to inherit; the game is already over
    receiver = min(survivors, key=lambda a: (_remaining_tasks(a), id_order[a.id]))
    for task in incomplete:
        victim.tasks.remove(task)
        receiver.tasks.append(task)


def resolve_reports_and_button(
    world: WorldState, config: SimConfig, actions: dict[AgentId, Action], meeting_pending: bool
) -> tuple[list[EventTuple], tuple[str, AgentId | None] | None]:
    """Resolve Report/PressButton (§6.2 step 7); return events and a newly queued meeting."""
    events: list[EventTuple] = []
    queued: tuple[str, AgentId | None] | None = None
    for agent_id, action in actions.items():
        if meeting_pending or queued is not None:
            break
        agent = world.agents[agent_id]
        if isinstance(action, Report) and agent.room is not None:
            body = next((b for b in world.bodies if b.room == agent.room and not b.reported), None)
            if body is None:
                continue
            body.reported = True
            meeting_no = world.meeting_count + 1
            events.append(
                (
                    "REPORT",
                    {
                        "agent": agent_id,
                        "victim": body.victim,
                        "room": agent.room,
                        "meeting": meeting_no,
                    },
                )
            )
            queued = ("report", body.victim)
        elif isinstance(action, PressButton):
            if agent.room != config.button_room or agent.button_used:
                continue
            agent.button_used = True
            meeting_no = world.meeting_count + 1
            button_data = {"agent": agent_id, "room": agent.room, "meeting": meeting_no}
            events.append(("BUTTON", button_data))
            queued = ("button", None)
    return events, queued


def check_win(world: WorldState, config: SimConfig) -> EventTuple | None:
    """Evaluate §6.2 step 9's win conditions; return a GAME_OVER event tuple once one is met."""
    progress, duration = real_task_progress(world)
    alive_impostors = _alive_count(world, Role.IMPOSTOR)
    alive_crew = _alive_count(world, Role.CREWMATE)
    winner: Role | None = None
    reason = "max_ticks"
    if duration > 0 and progress >= duration:
        winner, reason = Role.CREWMATE, "all_tasks_done"
    elif alive_impostors == 0:
        winner, reason = Role.CREWMATE, "impostors_ejected"
    elif alive_impostors >= alive_crew:
        winner, reason = Role.IMPOSTOR, "impostors_majority"
    elif (
        world.sabotage is not None
        and world.sabotage.kind == "reactor"
        and world.sabotage.timer <= 0
    ):
        winner, reason = Role.IMPOSTOR, "reactor_timer"
    if winner is None and world.tick < config.max_ticks:
        return None
    world.winner = winner
    winner_str = winner.value if winner is not None else None
    return ("GAME_OVER", {"winner": winner_str, "ticks": world.tick, "reason": reason})


def _alive_count(world: WorldState, role: Role) -> int:
    """Count alive agents with the given role."""
    return sum(1 for a in world.agents.values() if a.alive and a.role is role)
