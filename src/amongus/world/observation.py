"""Observation and note generation: the only channel from world state to a policy (SPEC §7)."""

from dataclasses import dataclass, field

import numpy as np

from amongus.config import SimConfig
from amongus.types import AgentId, AgentPhys, Note, Role, RoomId, WorldState


@dataclass
class SabotageAlarm:
    """Public sabotage info shown to every agent — never the saboteur's identity (§7)."""

    kind: str
    panels: tuple[RoomId, ...]
    timer: int
    panel_done: dict[RoomId, bool]


@dataclass
class TickSignals:
    """Per-tick facts the engine collects during resolution, fed into observation building.

    vent_events and kill_events already carry the resolved witness set for each event —
    rolled once at resolution time in world/resolve.py — so observation-building only
    checks membership and never re-rolls the same probability (§7).
    """

    vent_events: list[tuple[AgentId, RoomId, frozenset[AgentId]]] = field(default_factory=list)
    kill_events: list[tuple[AgentId, AgentId, RoomId, frozenset[AgentId]]] = field(
        default_factory=list
    )
    radio: list[str] = field(default_factory=list)
    stalled: bool = False


@dataclass
class Observation:
    """What one agent perceives this tick; never carries a WorldState reference (§7)."""

    self_id: AgentId
    tick: int
    room: RoomId | None
    self_phys: AgentPhys | None
    occupants: list[AgentId]
    bodies_here: list[AgentId]
    local_events: list[dict]
    task_bar: float
    sabotage_alarm: SabotageAlarm | None
    closed_doors: frozenset[frozenset[RoomId]]
    radio: list[str]
    stall_flag: bool
    death_notices: list[AgentId]
    partner_id: AgentId | None
    partner_room: RoomId | None


def p_miss_eff(world: WorldState, agent: AgentPhys, config: SimConfig) -> float:
    """Return this agent's per-occupant miss probability this tick — the lights asymmetry."""
    lights_active = world.sabotage is not None and world.sabotage.kind == "lights"
    if lights_active and agent.role is Role.CREWMATE:
        return config.p_miss_lights
    return config.p_miss


def build_observation(
    world: WorldState,
    agent_id: AgentId,
    config: SimConfig,
    rng: np.random.Generator,
    signals: TickSignals,
) -> Observation:
    """Build agent_id's Observation for the current world.tick (§7)."""
    agent = world.agents[agent_id]
    p_eff = p_miss_eff(world, agent, config)
    lights_active = world.sabotage is not None and world.sabotage.kind == "lights"

    room: RoomId | None = None
    occupants: list[AgentId] = []
    bodies_here: list[AgentId] = []
    local_events: list[dict] = []
    if agent.room is not None:
        room = agent.room
        for other_id, other in world.agents.items():
            if other_id == agent_id or not other.alive or other.room != room:
                continue
            if rng.random() >= p_eff:
                occupants.append(other_id)
        body_p = config.p_body_visible_lights if lights_active else 1.0
        for body in world.bodies:
            if body.room == room and rng.random() < body_p:
                bodies_here.append(body.victim)
        for venter, vent_room, witnesses in signals.vent_events:
            if vent_room == room and agent_id in witnesses:
                local_events.append({"kind": "vent", "agent": venter, "room": room})
        for killer, victim, kill_room, witnesses in signals.kill_events:
            if kill_room == room and agent_id in witnesses:
                local_events.append(
                    {"kind": "kill", "killer": killer, "victim": victim, "room": room}
                )

    sabotage_alarm = None
    if world.sabotage is not None:
        sabotage_alarm = SabotageAlarm(
            kind=world.sabotage.kind,
            panels=world.sabotage.panels,
            timer=world.sabotage.timer,
            panel_done=dict(world.sabotage.done),
        )

    partner_id, partner_room = None, None
    if agent.role is Role.IMPOSTOR:
        for other_id, other in world.agents.items():
            if other_id != agent_id and other.role is Role.IMPOSTOR:
                partner_id, partner_room = other_id, other.room
                break

    death_notices = [b.victim for b in world.bodies if b.reported]

    return Observation(
        self_id=agent_id,
        tick=world.tick,
        room=room,
        self_phys=_copy_phys(agent),
        occupants=occupants,
        bodies_here=bodies_here,
        local_events=local_events,
        task_bar=world.task_bar,
        sabotage_alarm=sabotage_alarm,
        closed_doors=frozenset(world.closed_edges),
        radio=list(signals.radio),
        stall_flag=signals.stalled,
        death_notices=death_notices,
        partner_id=partner_id,
        partner_room=partner_room,
    )


def _copy_phys(agent: AgentPhys) -> AgentPhys:
    """Return a shallow copy of an agent's own physical state, safe to hand to a policy."""
    return AgentPhys(
        id=agent.id,
        role=agent.role,
        alive=agent.alive,
        room=agent.room,
        transit=agent.transit,
        tasks=list(agent.tasks),
        kill_cooldown=agent.kill_cooldown,
        button_used=agent.button_used,
    )


def _join_names(names: list[AgentId]) -> str:
    """Join agent ids for a note sentence: 'red', 'red and blue', 'red, blue and green'."""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def generate_notes(obs: Observation) -> list[Note]:
    """Turn an Observation into a few plain-English Notes via fixed templates (§7.1)."""
    notes: list[Note] = []
    if obs.room is None:
        return notes
    if obs.occupants:
        notes.append(
            Note(
                tick=obs.tick,
                kind="saw",
                text=f"t={obs.tick} I was in {obs.room} with {_join_names(obs.occupants)}.",
            )
        )
    else:
        notes.append(
            Note(tick=obs.tick, kind="alone", text=f"t={obs.tick} I was alone in {obs.room}.")
        )
    for body_id in obs.bodies_here:
        notes.append(
            Note(
                tick=obs.tick,
                kind="body",
                text=f"t={obs.tick} I found {body_id}'s body in {obs.room}.",
            )
        )
    for ev in obs.local_events:
        if ev["kind"] == "kill":
            notes.append(
                Note(
                    tick=obs.tick,
                    kind="kill",
                    text=f"t={obs.tick} I saw {ev['killer']} kill {ev['victim']} in {ev['room']}.",
                )
            )
        elif ev["kind"] == "vent":
            notes.append(
                Note(
                    tick=obs.tick,
                    kind="vent",
                    text=f"t={obs.tick} I saw {ev['agent']} climb out of a vent in {ev['room']}.",
                )
            )
    return notes
