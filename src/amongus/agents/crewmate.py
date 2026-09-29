"""Crewmate Policy: the strict priority order of SPEC §11.1."""

from typing import Callable

from amongus.agents.memory import AgentMemory
from amongus.agents.reactor import ReactorBoard, ReactorNegotiator
from amongus.config import SimConfig
from amongus.search.astar import AStarPlanner
from amongus.search.costs import make_cost_fn, plain_cost_fn
from amongus.types import (
    Action,
    AgentId,
    DoTask,
    HoldPanel,
    Move,
    PressButton,
    Report,
    RoomId,
    Wait,
)
from amongus.world.map import GraphView
from amongus.world.observation import Observation


def _witnessed_since_last_meeting(memory: AgentMemory) -> bool:
    """True if a kill/vent note was recorded strictly after the most recent meeting note."""
    last_meeting_tick = 0
    for note in reversed(memory.notes):
        if note.kind == "meeting":
            last_meeting_tick = note.tick
            break
    return any(
        note.kind in ("kill", "vent") and note.tick > last_meeting_tick for note in memory.notes
    )


def _risk(room: RoomId, memory: AgentMemory, tick: int, cfg: SimConfig) -> float:
    """Replicate §8.2's risk_i(v) for a single room, to rank immediate neighbors (step 7)."""
    total = 0.0
    for agent, (seen_room, seen_tick) in memory.last_seen.items():
        if seen_room != room or tick - seen_tick > cfg.last_seen_decay:
            continue
        total += memory.suspicion.get(agent, 0.0)
    return total


class CrewmatePolicy:
    """Report bodies, hold panels, fix lights, call meetings, shadow, then do tasks (§11.1)."""

    def __init__(
        self,
        agent_id: AgentId,
        config: SimConfig,
        planner: AStarPlanner | None = None,
        emit: Callable[..., None] | None = None,
        reactor_board: ReactorBoard | None = None,
    ) -> None:
        """Build a crewmate policy with its own memory, planner and optional telemetry sink.

        Meeting deliberation is not owned here: MeetingProtocol (agents/protocol.py) calls
        one shared Deliberator for every speaker each round (§10), keyed off this policy's
        `.memory` (duck-typed, read for notes/route-risk and written with final suspicion).

        `reactor_board` is the one §12.3 public bulletin every policy in a game must share;
        callers wiring up a full 8-agent game must construct one ReactorBoard and pass the
        same instance to every CrewmatePolicy/ImpostorPolicy, or the protocol degrades to
        each agent negotiating against an empty board of its own.
        """
        self.agent_id = agent_id
        self.config = config
        self.planner = planner or AStarPlanner()
        self.memory = AgentMemory()
        self._emit = emit
        self.reactor_commitment: RoomId | None = None  # set by the renegotiation protocol
        self._reactor = ReactorNegotiator(
            agent_id, config, self.planner, reactor_board or ReactorBoard(), emit
        )
        self.shadow_disabled_until = 0  # §13: pushed forward on PROGRESS_STALL
        self._was_stalled = False

    def decide(self, obs: Observation) -> Action:
        """Return this tick's action, in the exact §11.1 priority order."""
        triggers = self.memory.observe(obs)
        if obs.room is None or obs.self_phys is None:
            return Wait()

        self._handle_stall(obs)
        if self.config.commit_protocol:
            self.reactor_commitment = self._reactor.step(obs, self.memory)

        # 1. An unreported body is in the room, or a kill was witnessed this tick -> Report.
        # A victim already in death_notices has been reported (§7): re-reporting it is
        # rejected by the engine every tick forever, which livelocks the whole crew in
        # cafeteria after a meeting. Fall through instead.
        unreported_bodies = [b for b in obs.bodies_here if b not in obs.death_notices]
        if unreported_bodies or any(e["kind"] == "kill" for e in obs.local_events):
            return Report()

        # 2. Committed to a reactor panel (or, with commit_protocol off, the nearest one).
        panel_action = self._reactor_panel_action(obs, triggers)
        if panel_action is not None:
            return panel_action

        # 3. Lights active and within 6 ticks of electrical -> go hold that panel.
        lights_action = self._lights_action(obs, triggers)
        if lights_action is not None:
            return lights_action

        # 4. Emergency meeting: witnessed a kill/vent since the last meeting -> press the button.
        button_action = self._call_meeting(obs, triggers)
        if button_action is not None:
            return button_action

        # 5. Shadowing: co-located with, or tracking, a highly suspicious agent.
        shadow = self._shadow(obs, triggers)
        if shadow is not None:
            return shadow

        # 6. Tasks remain: visit task rooms nearest-first by A* distance.
        task_action = self._do_tasks(obs, triggers)
        if task_action is not None:
            return task_action

        # 7. Tasks done: patrol toward the lowest-risk neighboring room.
        return self._patrol(obs)

    def _handle_stall(self, obs: Observation) -> None:
        """§13: on the PROGRESS_STALL rising edge, drop shadowing and force a task replan."""
        if obs.stall_flag and not self._was_stalled and self.config.deadlock_protocol:
            self.shadow_disabled_until = obs.tick + self.config.shadow_cooldown
            self.memory.goal = None
        self._was_stalled = obs.stall_flag

    def _reactor_panel_action(self, obs: Observation, triggers: set[str]) -> Action | None:
        """§11.1 step 2: route (no risk term) to a committed panel, or hold it on arrival."""
        alarm = obs.sabotage_alarm
        if alarm is None or alarm.kind != "reactor":
            return None
        view = self._view(obs)
        if self.config.commit_protocol:
            target = self.reactor_commitment
            if target is None or alarm.panel_done.get(target, False):
                return None
        else:
            target = self._nearest_open_panel(obs, alarm, view)
            if target is None:
                return None
        return self._route_and_hold(obs, triggers, view, target)

    def _nearest_open_panel(self, obs: Observation, alarm, view: GraphView) -> RoomId | None:
        """Ablation fallback (commit_protocol=False): the nearest not-yet-done panel."""
        open_panels = [p for p in alarm.panels if not alarm.panel_done.get(p, False)]
        if not open_panels:
            return None
        best, best_cost = None, float("inf")
        for panel in open_panels:
            _path, cost, _stats = self.planner.path(view, obs.room, panel, plain_cost_fn)
            if cost < best_cost:
                best, best_cost = panel, cost
        return best

    def _lights_action(self, obs: Observation, triggers: set[str]) -> Action | None:
        """§11.1 step 3: within 6 ticks of electrical during lights, go hold that panel."""
        alarm = obs.sabotage_alarm
        if alarm is None or alarm.kind != "lights":
            return None
        target = self.config.lights_fix_room
        if alarm.panel_done.get(target, False):
            return None
        view = self._view(obs)
        _path, cost, _stats = self.planner.path(view, obs.room, target, plain_cost_fn)
        if cost > 6:
            return None
        return self._route_and_hold(obs, triggers, view, target)

    def _route_and_hold(
        self, obs: Observation, triggers: set[str], view: GraphView, target: RoomId
    ) -> Action:
        """Shared step-2/step-3 tail: HoldPanel on arrival, else A* there with plain_cost_fn."""
        if obs.room == target:
            return HoldPanel(room=target)
        next_hop = self.memory.route(
            self.agent_id, self.planner, view, plain_cost_fn, target, obs.room, triggers,
            self._emit,
        )
        return Move(to=next_hop) if next_hop is not None else Wait()

    def _call_meeting(self, obs: Observation, triggers: set[str]) -> Action | None:
        """§11.1 step 4: route to and press the button on unreported witness evidence."""
        if obs.self_phys.button_used:
            return None
        if obs.sabotage_alarm is not None and obs.sabotage_alarm.kind == "reactor":
            return None
        if not _witnessed_since_last_meeting(self.memory):
            return None
        if obs.room == self.config.button_room:
            return PressButton()
        next_hop = self.memory.route(
            self.agent_id,
            self.planner,
            self._view(obs),
            self._cost_fn(obs.tick),
            self.config.button_room,
            obs.room,
            triggers,
            self._emit,
        )
        return Move(to=next_hop) if next_hop is not None else Wait()

    def _view(self, obs: Observation) -> GraphView:
        """Build a GraphView reflecting this tick's known closed doors."""
        return GraphView(closed_edges=obs.closed_doors)

    def _cost_fn(self, tick: int):
        """Risk-weighted cost (§8.2), or plain distance when `planner=astar_no_risk` (ablation)."""
        if self.config.planner == "astar_no_risk":
            return plain_cost_fn
        return make_cost_fn(self.memory.suspicion, self.memory.last_seen, tick, self.config)

    def _shadow(self, obs: Observation, triggers: set[str]) -> Action | None:
        """Move toward the most-suspicious tracked agent if its suspicion clears theta_shadow."""
        if obs.tick < self.shadow_disabled_until or not self.memory.suspicion:
            return None
        suspect = max(self.memory.suspicion, key=lambda a: self.memory.suspicion[a])
        if self.memory.suspicion[suspect] < self.config.theta_shadow:
            return None
        last = self.memory.last_seen.get(suspect)
        if last is None:
            return None
        target_room, _seen_tick = last
        if obs.room == target_room:
            return Wait()
        next_hop = self.memory.route(
            self.agent_id,
            self.planner,
            self._view(obs),
            self._cost_fn(obs.tick),
            target_room,
            obs.room,
            triggers,
            self._emit,
        )
        return Move(to=next_hop) if next_hop is not None else Wait()

    def _do_tasks(self, obs: Observation, triggers: set[str]) -> Action | None:
        """Route nearest-first (by A* distance) to a remaining task room and work it."""
        tasks = [t for t in obs.self_phys.tasks if not t.done]
        if not tasks:
            return None
        rooms = sorted({t.room for t in tasks})
        cost_fn = self._cost_fn(obs.tick)
        view = self._view(obs)
        if self.memory.goal not in rooms:
            best_room, best_cost = None, float("inf")
            for room in rooms:
                _path, cost, _stats = self.planner.path(view, obs.room, room, cost_fn)
                if cost < best_cost:
                    best_room, best_cost = room, cost
            if best_room is None:
                return None  # every task room is currently unreachable (e.g. doors sabotage)
            self.memory.goal = best_room
        goal = self.memory.goal
        if goal == obs.room:
            task = next(t for t in tasks if t.room == obs.room)
            return DoTask(task_id=task.task_id)
        next_hop = self.memory.route(
            self.agent_id, self.planner, view, cost_fn, goal, obs.room, triggers, self._emit
        )
        return Move(to=next_hop) if next_hop is not None else Wait()

    def _patrol(self, obs: Observation) -> Action:
        """All tasks done: step to the adjacent room with the lowest risk, and keep watching."""
        view = self._view(obs)
        neighbors = view.neighbors(obs.room)
        if not neighbors:
            return Wait()

        def score(room_weight: tuple[RoomId, int]) -> tuple[float, RoomId]:
            room = room_weight[0]
            return (_risk(room, self.memory, obs.tick, self.config), room)

        best_room = min(neighbors, key=score)[0]
        return Move(to=best_room)
