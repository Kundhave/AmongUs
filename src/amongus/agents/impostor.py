"""Impostor Policy: the strict priority order of SPEC §11.2."""

from typing import Callable

import numpy as np

from amongus.agents.memory import AgentMemory
from amongus.agents.reactor import ReactorBoard, ReactorNegotiator
from amongus.config import SimConfig
from amongus.rng import make_rng
from amongus.search.astar import AStarPlanner
from amongus.search.costs import plain_cost_fn
from amongus.types import (
    Action,
    AgentId,
    DoTask,
    HoldPanel,
    Kill,
    Move,
    Note,
    RoomId,
    Sabotage,
    Vent,
    Wait,
)
from amongus.world.map import VENT_PAIRS, GraphView
from amongus.world.observation import Observation

_SABOTAGE_KINDS = ("lights", "reactor", "doors")
_SABOTAGE_WEIGHTS = (0.4, 0.4, 0.2)


def _build_vent_adj() -> dict[RoomId, set[RoomId]]:
    """Mirror engine.py's vent adjacency: undirected pairs from the static map (§5)."""
    adj: dict[RoomId, set[RoomId]] = {}
    for a, b in VENT_PAIRS:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


_VENT_ADJ = _build_vent_adj()


class ImpostorPolicy:
    """Kill, sabotage, defect on a won commitment, hunt, else blend in (§11.2)."""

    def __init__(
        self,
        agent_id: AgentId,
        config: SimConfig,
        planner: AStarPlanner | None = None,
        emit: Callable[..., None] | None = None,
        reactor_board: ReactorBoard | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Build an impostor policy with its own memory, planner and optional telemetry sink.

        Meeting deliberation is not owned here: MeetingProtocol (agents/protocol.py) calls
        one shared Deliberator for every speaker each round (§10), keyed off this policy's
        `.memory` (duck-typed, read for notes/route-risk and written with final suspicion).

        `reactor_board` is the shared §12.3 public bulletin (see CrewmatePolicy). `rng`
        drives sabotage-kind and defect-intent draws (§11.2 step 2-3); if not supplied it is
        built deterministically from `config.seed` and this agent's fixed index in
        `config.colors`, so two policies never draw from the same stream by accident while
        the whole run stays reproducible from `config.seed` alone (SPEC §0).
        """
        self.agent_id = agent_id
        self.config = config
        self.planner = planner or AStarPlanner()
        self.memory = AgentMemory()
        self._emit = emit
        self._just_killed_tick: int | None = None
        offset = config.colors.index(agent_id) if agent_id in config.colors else 0
        self.rng = rng if rng is not None else make_rng(config.seed * 1_009 + offset)
        self.reactor_commitment: RoomId | None = None
        self._reactor = ReactorNegotiator(
            agent_id, config, self.planner, reactor_board or ReactorBoard(), emit
        )
        self._next_sabotage_ready_tick = 0
        self._defect_intent: bool | None = None
        self._known_dead: set[AgentId] = set()

    def decide(self, obs: Observation) -> Action:
        """Return this tick's action, in the exact §11.2 priority order."""
        triggers = self.memory.observe(obs)
        if obs.room is None or obs.self_phys is None:
            return Wait()
        self._track_known_dead(obs)

        if self._just_killed_tick == obs.tick:
            self._just_killed_tick = None
            return self._escape(obs)

        # 1. Kill: cooldown 0, exactly one non-partner alive occupant observed.
        target = self._kill_target(obs)
        if target is not None:
            self._just_killed_tick = obs.tick + 1
            self.memory.notes.append(
                Note(
                    tick=obs.tick + 1,
                    kind="kill",
                    text=f"t={obs.tick + 1} I killed {target} in {obs.room}.",
                )
            )
            return Kill(target=target)

        # 2. Sabotage: off cooldown, >=3 crewmates alive, per-tick probability p_sabotage.
        sabotage_action = self._maybe_sabotage(obs)
        if sabotage_action is not None:
            return sabotage_action

        if self.config.commit_protocol:
            self.reactor_commitment = self._reactor.step(obs, self.memory, self._bid_multiplier())

        # 3. Defect: reactor active and it won a commitment.
        defect_action = self._reactor_defect_or_fix(obs, triggers)
        if defect_action is not None:
            return defect_action

        # 4. Hunt: walk toward the freshest-known lone crewmate, preferring vent rooms.
        hunt_action = self._hunt(obs, triggers)
        if hunt_action is not None:
            return hunt_action

        # 5. Blend: work a fake task like a crewmate would.
        return self._blend(obs, triggers)

    def _track_known_dead(self, obs: Observation) -> None:
        """Approximate what this impostor knows is dead: reports, own kills, witnessed kills."""
        self._known_dead.update(obs.death_notices)
        for ev in obs.local_events:
            if ev["kind"] == "kill":
                self._known_dead.add(ev["victim"])

    def _alive_crewmates_estimate(self) -> int:
        """This impostor's lower-bound estimate of living crewmates (§11.2 step 2's gate)."""
        total_crew = self.config.n_players - self.config.n_impostors
        return total_crew - len(self._known_dead)

    def _kill_target(self, obs: Observation) -> AgentId | None:
        """A lone non-partner occupant is killable iff the kill cooldown is 0."""
        if obs.self_phys.kill_cooldown != 0:
            return None
        crew_here = [a for a in obs.occupants if a != obs.partner_id]
        if len(crew_here) != 1:
            return None
        return crew_here[0]

    def _escape(self, obs: Observation) -> Action:
        """The tick after a kill: vent out if possible, else walk to the quietest neighbor."""
        vents = sorted(_VENT_ADJ.get(obs.room, set()))
        if vents:
            return Vent(to=vents[0])
        view = GraphView(closed_edges=obs.closed_doors)
        neighbors = view.neighbors(obs.room)
        if not neighbors:
            return Wait()

        def occupancy(room_weight: tuple[RoomId, int]) -> tuple[int, RoomId]:
            room = room_weight[0]
            count = sum(
                1
                for _agent, (seen_room, seen_tick) in self.memory.last_seen.items()
                if seen_room == room and obs.tick - seen_tick <= self.config.last_seen_decay
            )
            return (count, room)

        best_room = min(neighbors, key=occupancy)[0]
        return Move(to=best_room)

    def _maybe_sabotage(self, obs: Observation) -> Action | None:
        """§11.2 step 2: pick a sabotage kind and fire it, gated by cooldown/headcount/roll."""
        if obs.sabotage_alarm is not None:
            return None
        if obs.tick < self._next_sabotage_ready_tick:
            return None
        if self._alive_crewmates_estimate() < 3:
            return None
        if self.rng.random() >= self.config.p_sabotage:
            return None
        kind = str(self.rng.choice(_SABOTAGE_KINDS, p=_SABOTAGE_WEIGHTS))
        if kind == "doors":
            target = self._nearest_crewmate_room(obs)
            if target is None:
                return None  # no known crewmate to target this tick; try again later
            action: Action = Sabotage(kind="doors", target=target)
        else:
            action = Sabotage(kind=kind)
        self._next_sabotage_ready_tick = obs.tick + self.config.sabotage_cooldown
        return action

    def _nearest_crewmate_room(self, obs: Observation) -> RoomId | None:
        """The last-known room of the most recently sighted non-partner crewmate (doors target)."""
        candidates = [
            (seen_tick, room)
            for agent, (room, seen_tick) in self.memory.last_seen.items()
            if agent != obs.partner_id and agent not in self._known_dead
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda c: (-c[0], c[1]))
        return candidates[0][1]

    def _bid_multiplier(self) -> float:
        """Draw (once per reactor window) whether to defect, and this tick's BID multiplier."""
        if self._defect_intent is None:
            self._defect_intent = bool(self.rng.random() < self.config.p_defect)
        return self.config.impostor_underbid if self._defect_intent else 1.0

    def _reactor_defect_or_fix(self, obs: Observation, triggers: set[str]) -> Action | None:
        """§11.2 step 3: stall on a won commitment if defecting, else fix exactly like a crew."""
        alarm = obs.sabotage_alarm
        if alarm is None or alarm.kind != "reactor":
            self._defect_intent = None  # window closed: redraw intent next time one opens
            return None
        if not self.config.commit_protocol or self.reactor_commitment is None:
            return None
        target = self.reactor_commitment
        if alarm.panel_done.get(target, False):
            return None
        if self._defect_intent:
            return Wait()
        if obs.room == target:
            return HoldPanel(room=target)
        view = GraphView(closed_edges=obs.closed_doors)
        next_hop = self.memory.route(
            self.agent_id, self.planner, view, plain_cost_fn, target, obs.room, triggers,
            self._emit,
        )
        return Move(to=next_hop) if next_hop is not None else Wait()

    def _alone_rooms(self, obs: Observation) -> set[RoomId]:
        """Rooms where, as far as this impostor knows, exactly one live non-partner crew stands."""
        counts: dict[RoomId, int] = {}
        for agent, (room, _seen_tick) in self.memory.last_seen.items():
            if agent == obs.partner_id or agent in self._known_dead:
                continue
            counts[room] = counts.get(room, 0) + 1
        return {room for room, count in counts.items() if count == 1}

    def _hunt(self, obs: Observation, triggers: set[str]) -> Action | None:
        """§11.2 step 4: A*-nearest lone crewmate as far as known, preferring vent rooms."""
        candidate_rooms = {
            room
            for agent, (room, _seen_tick) in self.memory.last_seen.items()
            if agent != obs.partner_id and agent not in self._known_dead
        }
        if not candidate_rooms:
            return None
        view = GraphView(closed_edges=obs.closed_doors)
        alone_rooms = self._alone_rooms(obs)

        def score(room: RoomId) -> tuple[bool, float, bool, RoomId]:
            _path, cost, _stats = self.planner.path(view, obs.room, room, plain_cost_fn)
            return (room not in alone_rooms, cost, room not in _VENT_ADJ, room)

        target_room = min(candidate_rooms, key=score)
        if obs.room == target_room:
            return Wait()
        next_hop = self.memory.route(
            self.agent_id,
            self.planner,
            view,
            plain_cost_fn,
            target_room,
            obs.room,
            triggers,
            self._emit,
        )
        return Move(to=next_hop) if next_hop is not None else None

    def _blend(self, obs: Observation, triggers: set[str]) -> Action:
        """Walk to and work a fake task room, nearest-first, exactly like a crewmate would."""
        tasks = [t for t in obs.self_phys.tasks if not t.done]
        if not tasks:
            return Wait()
        rooms = sorted({t.room for t in tasks})
        view = GraphView(closed_edges=obs.closed_doors)
        if self.memory.goal not in rooms:
            best_room, best_cost = None, float("inf")
            for room in rooms:
                _path, cost, _stats = self.planner.path(view, obs.room, room, plain_cost_fn)
                if cost < best_cost:
                    best_room, best_cost = room, cost
            if best_room is None:
                return Wait()  # every fake-task room is currently unreachable
            self.memory.goal = best_room
        goal = self.memory.goal
        if goal == obs.room:
            task = next(t for t in tasks if t.room == obs.room)
            return DoTask(task_id=task.task_id)
        next_hop = self.memory.route(
            self.agent_id, self.planner, view, plain_cost_fn, goal, obs.room, triggers, self._emit
        )
        return Move(to=next_hop) if next_hop is not None else Wait()
