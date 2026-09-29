"""Per-agent memory: notes, suspicion, last-seen sightings, plan state (SPEC §9.1, §8.4)."""

from dataclasses import dataclass, field
from typing import Callable

from amongus.contracts import CostFn, Planner
from amongus.types import AgentId, Note, RoomId
from amongus.world.map import GraphView
from amongus.world.observation import Observation, generate_notes

# §8.4 replan triggers, in the priority order used to pick one REPLAN "reason" when several
# fire on the same tick.
TRIGGER_PRIORITY: tuple[str, ...] = (
    "arrival",
    "suspicion_shift",
    "topology",
    "new_goal",
    "meeting_end",
    "blocked",
)


def _pick_reason(triggers: set[str]) -> str:
    """Return the highest-priority fired trigger name, per TRIGGER_PRIORITY."""
    for name in TRIGGER_PRIORITY:
        if name in triggers:
            return name
    return "new_goal"  # fallback: goal changed with no other trigger recorded


@dataclass
class AgentMemory:
    """One agent's private notes, suspicion beliefs, sightings and route plan (§9.1)."""

    notes: list[Note] = field(default_factory=list)
    suspicion: dict[AgentId, float] = field(default_factory=dict)
    last_seen: dict[AgentId, tuple[RoomId, int]] = field(default_factory=dict)
    goal: RoomId | None = None
    path: list[RoomId] = field(default_factory=list)

    _prev_room: RoomId | None = field(default=None, repr=False)
    _prev_closed: frozenset = field(default_factory=frozenset, repr=False)
    _prev_sabotage: str | None = field(default=None, repr=False)
    _prev_bodies: int = field(default=0, repr=False)
    _notes_consumed: int = field(default=0, repr=False)
    _prev_suspicion_snapshot: dict = field(default_factory=dict, repr=False)

    def recent(self, n: int) -> list[Note]:
        """Return the last n notes, most-recent-last, for the LLM prompt (§9.1)."""
        return self.notes[-n:]

    def observe(self, obs: Observation) -> set[str]:
        """Absorb one Observation: update last_seen, append its §7.1 notes, return §8.4 triggers.

        Note generation happens here (not in the engine) so every note this agent ever
        perceives — saw/alone/body/kill/vent — lands in the one list the Deliberator reads
        (§9.1's `recent(n)`) and that §11.1 step 4 scans for unreported witness evidence.
        """
        if obs.room is not None:
            for agent_id in obs.occupants:
                self.last_seen[agent_id] = (obs.room, obs.tick)

        triggers: set[str] = set()
        if obs.room is not None and self._prev_room is None:
            triggers.add("arrival")
        if obs.closed_doors != self._prev_closed:
            triggers.add("topology")
        alarm_kind = obs.sabotage_alarm.kind if obs.sabotage_alarm is not None else None
        if alarm_kind is not None and alarm_kind != self._prev_sabotage:
            triggers.add("new_goal")
        if obs.bodies_here and self._prev_bodies == 0:
            triggers.add("new_goal")
        if any(n.kind == "meeting" for n in self.notes[self._notes_consumed :]):
            triggers.add("meeting_end")
        if self._suspicion_shifted():
            triggers.add("suspicion_shift")

        self.notes.extend(generate_notes(obs))
        self._notes_consumed = len(self.notes)
        self._prev_room = obs.room
        self._prev_closed = obs.closed_doors
        self._prev_sabotage = alarm_kind
        self._prev_bodies = len(obs.bodies_here)
        return triggers

    def _suspicion_shifted(self) -> bool:
        """True if suspicion moved by L1 > 0.1 since the last check; resets the baseline."""
        keys = set(self.suspicion) | set(self._prev_suspicion_snapshot)
        l1 = sum(
            abs(self.suspicion.get(k, 0.0) - self._prev_suspicion_snapshot.get(k, 0.0))
            for k in keys
        )
        self._prev_suspicion_snapshot = dict(self.suspicion)
        return l1 > 0.1

    def set_plan(self, goal: RoomId | None, path: list[RoomId]) -> None:
        """Store a freshly computed route as the current plan."""
        self.goal = goal
        self.path = path

    def route(
        self,
        agent_id: AgentId,
        planner: Planner,
        view: GraphView,
        cost_fn: CostFn,
        dest: RoomId,
        current_room: RoomId,
        triggers: set[str],
        emit: Callable[..., None] | None,
    ) -> RoomId | None:
        """Ensure the plan targets dest, replanning (and emitting REPLAN) if needed.

        The stored path is not consumed as the agent advances along it, so "on track" is
        checked by locating current_room's index within it, not by comparing path[0].
        Returns the next room to Move to, or None if current_room already is dest.
        """
        if current_room == dest:
            self.set_plan(dest, [dest])
            return None
        idx = self.path.index(current_room) if current_room in self.path else None
        stale = self.goal != dest or not self.path or self.path[-1] != dest or idx is None
        blocked = (
            not stale
            and idx is not None
            and idx < len(self.path) - 1
            and frozenset((current_room, self.path[idx + 1])) in view.closed_edges
        )
        if stale or triggers or blocked:
            fired = set(triggers)
            if blocked:
                fired.add("blocked")
            if not fired:
                fired.add("new_goal")
            reason = _pick_reason(fired)
            path, _cost, stats = planner.path(view, current_room, dest, cost_fn)
            self.set_plan(dest, path)
            if emit is not None:
                emit("REPLAN", agent=agent_id, reason=reason, expanded=stats.expanded, path=path)
            idx = 0
        if idx is None or idx >= len(self.path) - 1:
            return None
        return self.path[idx + 1]
