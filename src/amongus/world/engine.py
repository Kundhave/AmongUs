"""Tick pipeline for the simulation: setup plus the exact eleven-step §6.2 resolution loop."""

from typing import Callable

from amongus.config import SimConfig
from amongus.contracts import Policy
from amongus.rng import make_rng
from amongus.types import (
    Action,
    AgentId,
    DoTask,
    Event,
    Note,
    Phase,
    Role,
    WorldState,
)
from amongus.world import sabotage as sab
from amongus.world.map import VENT_PAIRS
from amongus.world.observation import Observation, TickSignals, build_observation, generate_notes
from amongus.world.resolve import (
    check_win,
    resolve_arrivals,
    resolve_departures,
    resolve_kills,
    resolve_reports_and_button,
)
from amongus.world.state import SPAWN_ROOM, new_world, real_task_progress

# Debug-only rejections/progress ticks from world/resolve.py and world/sabotage.py that are
# not part of the §14.1 schema; the engine drops them from the log rather than inventing new
# uppercase types.
_INTERNAL_ONLY = frozenset(
    {
        "sabotage_rejected",
        "holdpanel_rejected",
        "move_rejected",
        "vent_rejected",
        "kill_rejected",
    }
)

_VENT_ADJ: dict[str, set[str]] = {}
for _a, _b in VENT_PAIRS:
    _VENT_ADJ.setdefault(_a, set()).add(_b)
    _VENT_ADJ.setdefault(_b, set()).add(_a)

# Public seam for agents/protocol.py (§10, agent-engineer): the engine calls
# meeting_handler(engine, reason, victim) once a meeting fires ("report"|"button" reason;
# victim is the reported body's AgentId, else None). The handler should use engine.emit(...)
# to log MEETING_START/STATEMENT/SUSPICION/VOTE/EJECT and engine.world for state it needs;
# the engine calls engine.reset_after_meeting() itself right after the handler returns.
MeetingHandler = Callable[["Engine", str, "AgentId | None"], None]


class Engine:
    """Drives WorldState through the §6.2 tick pipeline, one Policy per agent."""

    def __init__(
        self,
        config: SimConfig,
        policies: dict[AgentId, Policy],
        meeting_handler: MeetingHandler | None = None,
    ) -> None:
        """Build the initial world and tick-0 observations for the given per-agent policies."""
        self.config = config
        self.policies = policies
        self.meeting_handler = meeting_handler
        self.rng = make_rng(config.seed)
        self.world: WorldState = new_world(config, self.rng)
        self.events: list[Event] = []
        self.notes: dict[AgentId, list[Note]] = {aid: [] for aid in self.world.agents}
        self._observations: dict[AgentId, Observation] = {}
        self._meeting_pending = False
        self._meeting_reason: str | None = None
        self._meeting_victim: AgentId | None = None
        self._sabotage_ready_tick = 0
        self._stall_start_tick: int | None = None
        self._last_meeting_end_tick = 0
        self._body_seen: set[tuple[AgentId, AgentId]] = set()
        self._observe_all(TickSignals())

    def run(self, n_ticks: int) -> None:
        """Advance the simulation up to n_ticks, stopping early once the game is over."""
        for _ in range(n_ticks):
            if self.world.phase is Phase.OVER:
                break
            self.tick()

    def tick(self) -> None:
        """Run one tick of the §6.2 pipeline in exact order."""
        if self.world.phase is Phase.OVER:
            return
        self.world.tick += 1
        actions = self._decide()  # 1 decide
        self._sabotage(actions)  # 2 sabotage
        vent_events = self._arrivals(actions)  # 3 arrivals
        kill_events = self._kills(actions)  # 4 kills
        # 5 departures: after kills, so Move can't dodge a same-tick kill (§6.2).
        self._departures(actions)
        self._work(actions)  # 6 work
        self._reports_and_button(actions)  # 7 reports/button
        self._timers()  # 8 timers
        self._win_check()  # 9 win check
        stalled = self._stall_start_tick is not None
        signals = TickSignals(vent_events=vent_events, kill_events=kill_events, stalled=stalled)
        self._observe_all(signals)  # 10 observe
        self._meeting()  # 11 meeting

    def emit(self, type_: str, **fields) -> None:
        """Public §14.1 event API: append one Event, stamping tick and task_bar automatically."""
        data = {"task_bar": self.world.task_bar, **fields}
        self.events.append(Event(tick=self.world.tick, type=type_, data=data))

    def _decide(self) -> dict[AgentId, Action]:
        """Collect one action per alive, non-transiting agent from the same pre-tick obs."""
        actions: dict[AgentId, Action] = {}
        for agent_id, agent in self.world.agents.items():
            if not agent.alive or agent.transit is not None:
                continue  # in transit: cannot act (§6.2 step 3)
            obs = self._observations.get(agent_id)
            if obs is None:
                continue
            actions[agent_id] = self.policies[agent_id].decide(obs)
        return actions

    def _sabotage(self, actions: dict[AgentId, Action]) -> None:
        """Validate and start at most one sabotage via world/sabotage.py (§6.2 step 2)."""
        events, self._sabotage_ready_tick = sab.resolve_sabotage_action(
            self.world, self.config, actions, self._sabotage_ready_tick
        )
        self._emit_filtered(events)

    def _arrivals(
        self, actions: dict[AgentId, Action]
    ) -> list[tuple[AgentId, str, frozenset[AgentId]]]:
        """Land transit arrivals and resolve Vent via world/resolve.py; emit events (step 3)."""
        events, vent_events = resolve_arrivals(
            self.world, self.config, self.rng, actions, _VENT_ADJ
        )
        self._emit_filtered(events)
        return vent_events

    def _departures(self, actions: dict[AgentId, Action]) -> None:
        """Start new Move transits via world/resolve.py and emit events (step 5, after kills)."""
        events = resolve_departures(self.world, actions)
        self._emit_filtered(events)

    def _kills(
        self, actions: dict[AgentId, Action]
    ) -> list[tuple[AgentId, AgentId, str, frozenset[AgentId]]]:
        """Resolve Kill actions via world/resolve.py and emit the resulting events (step 4)."""
        events, kill_events = resolve_kills(self.world, self.config, self.rng, actions)
        self._emit_filtered(events)
        return kill_events

    def _work(self, actions: dict[AgentId, Action]) -> None:
        """Advance DoTask progress and reactor/lights panel holds (§6.2 step 6)."""
        for agent_id, action in actions.items():
            agent = self.world.agents[agent_id]
            if not isinstance(action, DoTask) or agent.room is None:
                continue
            task = next((t for t in agent.tasks if t.task_id == action.task_id), None)
            if task is None or task.room != agent.room or task.done:
                continue
            task.progress += 1  # not logged: task_progress is not a §14.1 event type
            if agent.role is Role.CREWMATE:
                self.world.last_progress_tick = self.world.tick
        self._emit_filtered(sab.resolve_panels(self.world, self.config, actions))
        progress, duration = real_task_progress(self.world)
        self.world.task_bar = progress / duration if duration else 0.0

    def _reports_and_button(self, actions: dict[AgentId, Action]) -> None:
        """Resolve Report/PressButton, queuing a meeting if one fired (step 7)."""
        events, queued = resolve_reports_and_button(
            self.world, self.config, actions, self._meeting_pending
        )
        for type_, data in events:
            self.emit(type_, **data)
        if queued is not None:
            self._meeting_pending = True
            self._meeting_reason, self._meeting_victim = queued

    def _timers(self) -> None:
        """Decrement cooldowns, sabotage/door timers, and run stall detection (§6.2 step 8)."""
        for agent in self.world.agents.values():
            if agent.kill_cooldown > 0:
                agent.kill_cooldown -= 1
        self._emit_filtered(sab.resolve_timer(self.world, self.config))
        stall_events, self._stall_start_tick = sab.detect_stall(
            self.world, self.config, self._stall_start_tick, self._last_meeting_end_tick
        )
        self._emit_filtered(stall_events)

    def _emit_filtered(self, events: list[tuple[str, dict]]) -> None:
        """Emit each event tuple, dropping debug-only types not in the §14.1 schema."""
        for type_, data in events:
            if type_ in _INTERNAL_ONLY:
                continue
            self.emit(type_, **data)

    def _win_check(self) -> None:
        """Evaluate §6.2 step 9's win conditions via world/resolve.py and end the game if met."""
        event = check_win(self.world, self.config)
        if event is None:
            return
        self.world.phase = Phase.OVER
        type_, data = event
        self.emit(type_, **data)

    def _observe_all(self, signals: TickSignals) -> None:
        """Build obs_t (step 10), emit BODY_SEEN once per (agent, victim), append notes (§7).

        A re-seen body still perceives normally (obs.bodies_here, notes are unaffected); only
        the BODY_SEEN log entry is suppressed on repeat so the event log logs each first
        sighting once instead of once per tick the body remains visible.
        """
        self._observations = {}
        for agent_id, agent in self.world.agents.items():
            if not agent.alive:
                continue
            obs = build_observation(self.world, agent_id, self.config, self.rng, signals)
            self._observations[agent_id] = obs
            for victim_id in obs.bodies_here:
                pair = (agent_id, victim_id)
                if pair in self._body_seen:
                    continue
                self._body_seen.add(pair)
                self.emit("BODY_SEEN", agent=agent_id, victim=victim_id, room=obs.room)
            self.notes[agent_id].extend(generate_notes(obs))

    def _meeting(self) -> None:
        """Run the queued meeting once no reactor sabotage blocks it, then reset (§6.2 step 11)."""
        if not self._meeting_pending:
            return
        if self.world.phase is Phase.OVER:
            return  # step 9's win check already ended the game this tick; it takes priority
        if self.world.sabotage is not None and self.world.sabotage.kind == "reactor":
            return  # stays queued until the reactor is fixed
        reason, victim = self._meeting_reason, self._meeting_victim
        self._meeting_pending = False
        self._meeting_reason = None
        self._meeting_victim = None
        self.world.meeting_count += 1
        self.world.phase = Phase.MEETING
        if self.meeting_handler is not None:
            self.meeting_handler(self, reason, victim)
        self.reset_after_meeting()
        if self.world.phase is Phase.MEETING:
            self.world.phase = Phase.PLAY

    def reset_after_meeting(self) -> None:
        """Public §6.2 step-11 reset (respawn/cooldowns/sabotage/bodies); called after a meeting."""
        for agent in self.world.agents.values():
            if not agent.alive:
                continue
            agent.room = SPAWN_ROOM
            agent.transit = None
            agent.kill_cooldown = self.config.kill_cooldown
        self.world.closed_edges = set()
        self.world.sabotage = None
        for body in self.world.bodies:
            body.reported = True
        self._last_meeting_end_tick = self.world.tick
