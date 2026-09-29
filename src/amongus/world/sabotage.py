"""Lights/doors/reactor sabotage state machines and their per-tick resolution (§12, §13)."""

from dataclasses import dataclass, field

from amongus.config import SimConfig
from amongus.types import Action, AgentId, HoldPanel, Role, RoomId, Sabotage, WorldState
from amongus.world.map import ROOMS, GraphView

EventTuple = tuple[str, dict]


@dataclass
class SabotageState:
    """The one active sabotage's public state; the agent who started it is never stored here."""

    kind: str  # "lights" | "doors" | "reactor"
    timer: int
    panels: tuple[RoomId, ...] = ()
    panel_holds: dict[RoomId, int] = field(default_factory=dict)
    done: dict[RoomId, bool] = field(default_factory=dict)
    room: RoomId | None = None  # doors target only
    edges: tuple[tuple[RoomId, RoomId], ...] = ()  # doors: exact incident edges closed
    start_tick: int = 0


def start_lights(config: SimConfig, tick: int) -> SabotageState:
    """Begin a lights sabotage: fixed once electrical is held for panel_hold_ticks (§12.1)."""
    room = config.lights_fix_room
    return SabotageState(
        kind="lights", timer=0, panels=(room,), panel_holds={room: 0},
        done={room: False}, start_tick=tick,
    )


def start_reactor(config: SimConfig, tick: int) -> SabotageState:
    """Begin a reactor sabotage: two panels, timer counts down to an impostor win (§12.3)."""
    panels = config.reactor_panels
    return SabotageState(
        kind="reactor", timer=config.reactor_timer, panels=panels,
        panel_holds={p: 0 for p in panels}, done={p: False for p in panels}, start_tick=tick,
    )


def start_doors(
    config: SimConfig, tick: int, target: RoomId
) -> tuple[SabotageState, list[tuple[RoomId, RoomId]]]:
    """Begin a doors sabotage: close corridors incident to target for doors_duration (§12.2)."""
    view = GraphView()
    edges = [(target, other) for other, _w in view.neighbors(target)]
    state = SabotageState(
        kind="doors", timer=config.doors_duration, room=target, edges=tuple(edges), start_tick=tick
    )
    return state, edges


def _edge_set(edges) -> frozenset[frozenset[RoomId]]:
    """Turn a list of (room, room) pairs into the frozenset-of-frozenset form closed_edges uses."""
    return frozenset(frozenset(e) for e in edges)


def _start(
    world: WorldState, config: SimConfig, agent_id: AgentId, action: Sabotage
) -> EventTuple | None:
    """Instantiate the requested state machine and mutate world; return the SABOTAGE event."""
    tick = world.tick
    room: RoomId | None = None
    if action.kind == "lights":
        state = start_lights(config, tick)
    elif action.kind == "reactor":
        state = start_reactor(config, tick)
    elif action.kind == "doors":
        if action.target is None or action.target not in ROOMS:
            return None
        state, edges = start_doors(config, tick, action.target)
        world.closed_edges |= _edge_set(edges)
        room = action.target
    else:
        return None
    world.sabotage = state
    return (
        "SABOTAGE",
        {"kind": action.kind, "panels": list(state.panels), "timer": state.timer,
         "room": room, "agent": agent_id},
    )


def resolve_sabotage_action(
    world: WorldState, config: SimConfig, actions: dict[AgentId, Action], ready_tick: int
) -> tuple[list[EventTuple], int]:
    """Validate Sabotage actions, start at most one; id-order tie-break, shared cooldown (§6.2)."""
    events: list[EventTuple] = []
    active = world.sabotage is not None
    on_cooldown = world.tick < ready_tick
    candidates: list[tuple[AgentId, Sabotage]] = []
    for agent_id, action in actions.items():
        if not isinstance(action, Sabotage):
            continue
        agent = world.agents[agent_id]
        if active:
            reason = "active"
        elif on_cooldown:
            reason = "cooldown"
        elif not agent.alive or agent.role is not Role.IMPOSTOR:
            reason = "not_impostor"
        else:
            candidates.append((agent_id, action))
            continue
        events.append(
            ("sabotage_rejected", {"agent": agent_id, "kind": action.kind, "reason": reason})
        )
    if active or on_cooldown or not candidates:
        return events, ready_tick
    winner_id, winner_action = candidates[0]
    for loser_id, loser_action in candidates[1:]:
        events.append(
            ("sabotage_rejected", {"agent": loser_id, "kind": loser_action.kind, "reason": "tie"})
        )
    started = _start(world, config, winner_id, winner_action)
    if started is None:
        events.append(
            (
                "sabotage_rejected",
                {"agent": winner_id, "kind": winner_action.kind, "reason": "bad_target"},
            )
        )
        return events, ready_tick
    events.append(started)
    return events, world.tick + config.sabotage_cooldown


def resolve_panels(
    world: WorldState, config: SimConfig, actions: dict[AgentId, Action]
) -> list[EventTuple]:
    """Advance HoldPanel holds toward panel_hold_ticks; clear a fully-fixed sabotage (step 6)."""
    events: list[EventTuple] = []
    state = world.sabotage
    held_rooms: dict[RoomId, AgentId] = {}
    for agent_id, action in actions.items():
        if not isinstance(action, HoldPanel):
            continue
        agent = world.agents[agent_id]
        valid = (
            state is not None
            and agent.alive
            and agent.room == action.room
            and action.room in state.panels
            and not state.done.get(action.room, False)
        )
        if valid:
            held_rooms.setdefault(action.room, agent_id)
        else:
            events.append(("holdpanel_rejected", {"agent": agent_id, "room": action.room}))
    if state is None:
        return events
    for room, agent_id in held_rooms.items():
        state.panel_holds[room] += 1
        if state.panel_holds[room] >= config.panel_hold_ticks:
            state.done[room] = True
            events.append(("PANEL_DONE", {"agent": agent_id, "room": room}))
    if state.panels and all(state.done.values()):
        events.append(
            ("SABOTAGE_FIXED", {"kind": state.kind, "ticks": world.tick - state.start_tick})
        )
        world.sabotage = None
    return events


def resolve_timer(world: WorldState, config: SimConfig) -> list[EventTuple]:
    """Count down reactor/doors timers and reopen doors on expiry (step 8); lights has no timer."""
    state = world.sabotage
    if state is None:
        return []
    if state.kind == "reactor":
        state.timer = max(0, state.timer - 1)
        return []
    if state.kind == "doors":
        state.timer -= 1
        if state.timer <= 0:
            world.closed_edges -= _edge_set(state.edges)
            world.sabotage = None
            return [("DOORS_OPEN", {"room": state.room, "edges": [list(e) for e in state.edges]})]
    return []


def detect_stall(
    world: WorldState, config: SimConfig, stall_start_tick: int | None, last_meeting_end_tick: int
) -> tuple[list[EventTuple], int | None]:
    """Emit PROGRESS_STALL / DEADLOCK_BROKEN per the three-clause condition (§13)."""
    events: list[EventTuple] = []
    tick = world.tick
    progressed = world.last_progress_tick == tick
    if stall_start_tick is not None and progressed:
        events.append(("DEADLOCK_BROKEN", {"ticks": tick - stall_start_tick}))
        stall_start_tick = None
    if config.deadlock_protocol and stall_start_tick is None:
        no_sabotage = world.sabotage is None
        no_recent_meeting = (tick - last_meeting_end_tick) >= config.deadlock_window
        long_enough = (tick - world.last_progress_tick) >= config.deadlock_window
        if long_enough and no_sabotage and no_recent_meeting:
            stall_start_tick = tick
            events.append(("PROGRESS_STALL", {"ticks": tick - world.last_progress_tick}))
    return events, stall_start_tick
