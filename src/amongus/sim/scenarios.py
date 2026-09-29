"""§16 scenarios S1-S4: scripted setups layered on top of a normal Engine wiring."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from amongus.agents.crewmate import CrewmatePolicy
from amongus.agents.impostor import ImpostorPolicy
from amongus.agents.protocol import MeetingProtocol
from amongus.agents.reactor import ReactorBoard
from amongus.config import SimConfig
from amongus.contracts import Policy
from amongus.types import Action, AgentId, Kill, Move, Report, Role, RoomId, Sabotage, Wait
from amongus.world.engine import Engine
from amongus.world.observation import Observation


@dataclass
class Scenario:
    """One §16 scenario: seed, config overrides, forced roles/spawns, and scripted actions."""

    name: str
    seed: int
    config_overrides: dict = field(default_factory=dict)
    roles: tuple[AgentId, AgentId] | None = None
    spawns: dict[AgentId, RoomId] | None = None
    scripted: dict[int, dict[AgentId, Action]] = field(default_factory=dict)
    sabotage_schedule: dict[int, Sabotage] = field(default_factory=dict)


class _ScriptedPolicy:
    """Wraps a real Policy, substituting a scripted Action for one tick when one is scheduled."""

    def __init__(
        self, inner: Policy, agent_id: AgentId, scripted: dict[int, dict[AgentId, Action]]
    ) -> None:
        """Store the wrapped policy, this agent's id, and the mutable tick->action script."""
        self._inner = inner
        self._agent_id = agent_id
        self._scripted = scripted
        self.memory = getattr(inner, "memory", None)  # duck-typed passthrough (§10, §16)

    def decide(self, obs: Observation) -> Action:
        """Always run the wrapped policy for its side effects, then swap in a scripted action.

        "A scripted action replaces that agent's decision for that tick only; everything
        else runs normally" (§16) means the real Policy still observes (so memory/notes stay
        populated for the meeting), replans and negotiates as usual — only the *returned*
        Action is substituted when one is scheduled for this tick.

        `obs.tick` is built at the *end* of the previous tick (§6.2 step 10), one behind the
        `world.tick` this decision actually executes at (§6.2 step 1 of the next tick) — so
        scripted keys are matched against `obs.tick + 1`, the tick that will actually run.
        """
        real_action = self._inner.decide(obs)
        overrides = self._scripted.get(obs.tick + 1)
        if overrides is not None and self._agent_id in overrides:
            return overrides[self._agent_id]
        return real_action


def _apply_roles(engine: Engine, roles: tuple[AgentId, AgentId] | None) -> None:
    """Force these two agents to be the impostors, swapping with whichever pair rng picked."""
    if roles is None:
        return
    agents = engine.world.agents
    desired = set(roles)
    current = {aid for aid, a in agents.items() if a.role is Role.IMPOSTOR}
    for aid in desired - current:
        agents[aid].role = Role.IMPOSTOR
    for aid in current - desired:
        agents[aid].role = Role.CREWMATE


def _apply_spawns(engine: Engine, spawns: dict[AgentId, RoomId] | None) -> None:
    """Force these agents' tick-0 room, overriding the default cafeteria spawn."""
    if not spawns:
        return
    for aid, room in spawns.items():
        engine.world.agents[aid].room = room


def _apply_sabotage_schedule(
    scripted: dict[int, dict[AgentId, Action]],
    sabotage_schedule: dict[int, Sabotage],
    config: SimConfig,
    impostors: list[AgentId],
) -> None:
    """Assign each scheduled Sabotage to the lowest-`config.colors`-index alive impostor.

    §16's `Scenario.sabotage_schedule` maps tick -> Sabotage without naming an actor; a
    scenario picks its `roles` order so the intended saboteur is that agent (documented per
    scenario below). The saboteur is held with Wait() on every earlier tick so it cannot
    drift into transit and become non-decidable (skipped entirely, §6.2 step 1) right when
    its scheduled action is due.
    """
    if not sabotage_schedule or not impostors:
        return
    saboteur = min(impostors, key=lambda a: config.colors.index(a))
    for tick, action in sabotage_schedule.items():
        _wait_until(scripted, saboteur, range(1, tick))
        scripted.setdefault(tick, {})[saboteur] = action


def build_engine(scenario: Scenario) -> tuple[Engine, dict[int, dict[AgentId, Action]]]:
    """Wire an Engine exactly like `tests/test_meeting.py`'s `_build_full_engine`, then script it.

    One shared ReactorBoard for every policy (§12.3) — the canonical wiring pattern required
    so the renegotiation protocol computes one identical assignment instead of degrading to
    isolated per-agent boards. Returns `(engine, scripted)`; `scripted` is the live, mutable
    tick->action map every `_ScriptedPolicy` reads, so a caller (runner.py's live shock
    injection) can add further entries after construction and have them take effect.
    """
    # Scenarios default to the offline template Deliberator (so every existing scenario, test
    # and default CLI path stays quota-free); `scenario.config_overrides` — and runner.py's
    # `--deliberator` flag, which folds into it — may override this to "gemini".
    overrides = {"deliberator": "template", **scenario.config_overrides}
    config = replace(SimConfig(), seed=scenario.seed, **overrides)
    protocol = MeetingProtocol(config)
    engine = Engine(config, {}, meeting_handler=protocol)
    _apply_roles(engine, scenario.roles)
    _apply_spawns(engine, scenario.spawns)

    scripted: dict[int, dict[AgentId, Action]] = {
        tick: dict(actions) for tick, actions in scenario.scripted.items()
    }
    impostors = [aid for aid, a in engine.world.agents.items() if a.role is Role.IMPOSTOR]
    _apply_sabotage_schedule(scripted, scenario.sabotage_schedule, config, impostors)

    board = ReactorBoard()
    ids = list(config.colors[: config.n_players])
    policies: dict[AgentId, Policy] = {}
    for agent_id in ids:
        role = engine.world.agents[agent_id].role
        base: Policy
        if role is Role.CREWMATE:
            base = CrewmatePolicy(agent_id, config, emit=engine.emit, reactor_board=board)
        else:
            base = ImpostorPolicy(agent_id, config, emit=engine.emit, reactor_board=board)
        policies[agent_id] = _ScriptedPolicy(base, agent_id, scripted)
    engine.policies = policies

    for agent_id in ids:
        engine.emit("SPAWN", agent=agent_id, room=engine.world.agents[agent_id].room)
    engine.emit("ROLES", impostors=sorted(impostors))
    return engine, scripted


def _wait_until(scripted: dict[int, dict[AgentId, Action]], agent_id: AgentId, ticks) -> None:
    """Schedule Wait() for agent_id on every tick in `ticks` (an iterable of ints)."""
    for tick in ticks:
        scripted.setdefault(tick, {})[agent_id] = Wait()


def _worked_example_scripted() -> dict[int, dict[AgentId, Action]]:
    """§16 S2 choreography: blue and red alone in electrical ~t=41, kill t=43, report t=47.

    All timings are derived from the engine's exact §6.2 movement arithmetic (a Move decided
    at tick t along a weight-w edge arrives at tick t+w and is next decidable at t+w+1): the
    cafeteria->storage->electrical path costs 4+4 ticks each leg. Bystanders are held at
    cafeteria with Wait() for the whole window so nothing else disturbs the sequence.
    """
    scripted: dict[int, dict[AgentId, Action]] = {}
    bystanders = ("green", "pink", "orange", "yellow", "white", "black")
    for aid in bystanders:
        _wait_until(scripted, aid, range(1, 47))  # green gets its own script from tick 37 on

    for aid in ("blue", "red"):
        _wait_until(scripted, aid, range(1, 32))
        scripted.setdefault(32, {})[aid] = Move(to="storage")
        scripted.setdefault(37, {})[aid] = Move(to="electrical")
    scripted.setdefault(42, {})["blue"] = Wait()
    scripted.setdefault(42, {})["red"] = Wait()
    scripted.setdefault(43, {})["blue"] = Kill(target="red")

    _wait_until(scripted, "green", range(1, 37))
    scripted.setdefault(37, {})["green"] = Move(to="storage")
    scripted.setdefault(42, {})["green"] = Move(to="electrical")
    scripted.setdefault(47, {})["green"] = Report()
    return scripted


def _standoff_scenario(seed: int, deadlock_protocol: bool = True) -> Scenario:
    """§16 S4: pink and orange lock into mutual shadowing, freezing their share of the task bar.

    `tasks_per_crewmate=1` makes pink/orange literally "hold the last tasks": every other
    crewmate finishes its single task quickly, so the task bar depends only on the standoff
    pair. `max_ticks` is shortened so the ablation (stall persisting forever) doesn't need a
    long run to demonstrate.
    """
    return Scenario(
        name="standoff",
        seed=seed,
        config_overrides={
            "deadlock_protocol": deadlock_protocol,
            "tasks_per_crewmate": 1,
            "max_ticks": 150,
        },
    )


def _prime_standoff(engine: Engine, scripted: dict[int, dict[AgentId, Action]]) -> None:
    """Set up the pure deadlock demo: neutralize both impostors, lock pink/orange in mutual
    suspicion from tick 0 (§11.1 step 5) so nothing but the standoff can end the game early.
    """
    a, b = "pink", "orange"
    room = engine.world.agents[a].room
    for x, y in ((a, b), (b, a)):
        policy = engine.policies[x]
        policy.memory.suspicion[y] = 0.9
        policy.memory.last_seen[y] = (room, engine.world.tick)
    impostors = [aid for aid, ag in engine.world.agents.items() if ag.role is Role.IMPOSTOR]
    for aid in impostors:
        for tick in range(1, engine.config.max_ticks + 1):
            scripted.setdefault(tick, {})[aid] = Wait()


S1_BASELINE = Scenario(name="baseline", seed=7)

S2_WORKED_EXAMPLE = Scenario(
    name="worked_example",
    seed=3,
    roles=("blue", "black"),
    scripted=_worked_example_scripted(),
)

# roles=(white, black): sabotage_schedule assigns to the lower-`colors`-index alive impostor
# (black, index 6, precedes white, index 7), and black spawns at the reactor so it is the
# nearest bidder and reliably wins that panel's commitment before defecting (p_defect=1).
# seed=3 is load-bearing: verified end to end to produce the full commit -> defect -> revoke
# -> backup -> SABOTAGE_FIXED arc plus the payoff (black's suspicion rising next meeting).
# Other seeds (e.g. 11) can let both impostors sweep both panels and burn the reactor timer
# before any crewmate gets a turn, demonstrating the protocol failing instead of working.
S3_REACTOR_DEFECT = Scenario(
    name="reactor_defect",
    seed=3,
    config_overrides={"p_defect": 1.0},
    roles=("white", "black"),
    spawns={"black": "reactor"},
    sabotage_schedule={3: Sabotage(kind="reactor")},
)

S4_STANDOFF = _standoff_scenario(seed=4)

SCENARIOS: dict[str, Scenario] = {
    s.name: s for s in (S1_BASELINE, S2_WORKED_EXAMPLE, S3_REACTOR_DEFECT, S4_STANDOFF)
}


def build(scenario: Scenario) -> tuple[Engine, dict[int, dict[AgentId, Action]]]:
    """Build a fully wired, scripted Engine for `scenario`, applying scenario-specific priming.

    Returns `(engine, scripted)` — see `build_engine`. `scripted` is the handle `runner.py`
    mutates to inject live/CLI shocks at arbitrary future ticks.
    """
    engine, scripted = build_engine(scenario)
    if scenario.name == "standoff":
        _prime_standoff(engine, scripted)
    return engine, scripted
