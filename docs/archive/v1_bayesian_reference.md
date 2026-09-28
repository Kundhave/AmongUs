# AI Among Us — Implementation Reference (v1.0)

This file is the single source of truth for the build. Every agent reads only the sections it is pointed at (§ numbers). If code and this document disagree, this document wins, unless the deviation is recorded in `docs/DECISIONS.md` with a reason.

---

## §0 How to use this document

| Section | Owner agent | Read by |
|---|---|---|
| §1–§4 Overview, stack, config, contracts | world-engineer | everyone |
| §5 Map | world-engineer | search-engineer, agent-engineer |
| §6 Engine, §7 Observation | world-engineer | agent-engineer, belief-engineer |
| §8 Search & planning | search-engineer | agent-engineer |
| §9 Belief model | belief-engineer | agent-engineer |
| §10 Communication, §11 Meetings | agent-engineer | belief-engineer |
| §12 Policies | agent-engineer | — |
| §13 Sabotage & commitment protocol | agent-engineer + world-engineer | belief-engineer |
| §14 Deadlock | agent-engineer | — |
| §15 Telemetry, §16 Scenarios, §17 Metrics | telemetry-engineer | everyone (formats) |
| §18 Tests | each implementer for own module | test-runner |
| §19 Milestones | orchestrator (main session) | everyone |
| §20 Out of scope | everyone | everyone |

Rules that apply to all code:

- Python ≥ 3.11, type hints everywhere, dataclasses for data, `typing.Protocol` for module contracts.
- **All randomness** goes through one seeded `numpy.random.Generator` created in `rng.py`. No `random` module, no unseeded calls. Same seed + same config ⇒ byte-identical event log.
- No LLM calls anywhere in the decision loop (see §21).
- Every public function has a one-line docstring. Modules stay under ~400 lines; split if larger.

---

## §1 System summary

Eight agents (6 crewmates, 2 impostors) act on a 14-room weighted graph in discrete ticks. Each agent observes only its own room. Crewmates maintain an **exact Bayesian posterior over the 28 possible impostor pairs** and use it to plan risk-aware routes (**uniform-cost search**), order tasks (**Held–Karp DP**), share testimony, and vote. Impostors plan kills on a **time-expanded DAG** that minimises expected witnesses, lie in meetings, sabotage, and can defect from coordination commitments. A reactor sabotage exercises a **commit → defect → revoke → backup** renegotiation protocol. A **surveillance-standoff deadlock** is detected and broken by a zero-progress timeout.

The whole system is algorithmic. It must run end-to-end with zero network access.

---

## §2 Stack and project layout

### Dependencies (`pyproject.toml`)

| Package | Version | Used for | Required |
|---|---|---|---|
| numpy | ≥1.26 | belief vectors, RNG | yes |
| networkx | ≥3.2 | graph container, map drawing, test oracle for shortest paths | yes |
| mesa | ≥3.0,<4 | thin model wrapper + `DataCollector` | yes (wrapper only) |
| matplotlib | ≥3.8 | plots, optional map frames | yes |
| pytest, ruff | latest | tests, lint | dev |
| rich | latest | coloured terminal telemetry | optional |
| anthropic | latest | narration layer only (§21) | optional extra `[narrate]` |

**Mesa rule:** the core engine (`world/`) must not import Mesa. Mesa is used only in `sim/model.py` to wrap the engine and collect data. Mesa 3 removed the old scheduler classes, so do not use `RandomActivation`/`SimultaneousActivation`; the engine implements its own simultaneous resolution (§6). Before writing `sim/model.py`, check the installed Mesa version and its API (`python -c "import mesa; print(mesa.__version__)"`).

No pandas. Use `csv` and plain dicts.

### Layout

```
ai-among-us/
  pyproject.toml
  CLAUDE.md
  README.md
  docs/
    IMPLEMENTATION_REFERENCE.md   # this file
    DECISIONS.md                  # deviations log
  src/amongus/
    __init__.py
    config.py            # §3 SimConfig
    rng.py               # seeded generator factory
    types.py             # §4 shared dataclasses and enums
    contracts.py         # §4 Protocols: Planner, BeliefModel, Policy
    world/
      map.py             # §5 rooms, edges, vents, coords, tasks
      state.py           # WorldState and mutation helpers
      engine.py          # §6 tick pipeline, resolution, win check
      observation.py     # §7 observation builder
      sabotage.py        # §13 sabotage state machines
    search/
      ucs.py             # §8.1 uniform-cost search + all-pairs
      held_karp.py       # §8.3 task ordering
      time_expanded.py   # §8.4 impostor DAG planner
      baselines.py       # §8.6 BFS, A*, greedy order (benchmarks only)
      costs.py           # §8.2 risk-weighted edge cost
      stats.py           # SearchStats
    belief/
      hypotheses.py      # §9.1 hypothesis enumeration and indexing
      facts.py           # §9.3 fact store, trust rules, contradictions
      evidence.py        # §9.4 likelihood functions
      belief.py          # §9.5 update, marginals, entropy
    comms/
      messages.py        # §10 message schema
      meeting.py         # §11 meeting protocol and voting
      protocol.py        # §13 reactor commitment protocol
    agents/
      base.py            # shared agent memory, replan triggers
      crewmate.py        # §12.1
      impostor.py        # §12.2
      deadlock.py        # §14
    sim/
      model.py           # Mesa wrapper
      runner.py          # CLI: single game
      scenarios.py       # §16
      telemetry.py       # §15
    analysis/
      experiments.py     # §17 batch runs and ablations
      plots.py           # §17 figures
    narration/           # §21 optional
      narrate.py
  tests/
  scripts/
    run_demo.py
    run_batch.py
    bench_search.py
  runs/                  # gitignored outputs
```

The six swappable modules (the "scalability" claim on the slides) are: `world`, `search`, `belief`, `comms`, `agents`, `sim`. Belief model, planner and policy are selected by config string, so ablations swap them without code edits.

---

## §3 Configuration

`config.py` defines one frozen dataclass `SimConfig`. Every tunable lives here; no magic numbers elsewhere. Scenarios override fields with `dataclasses.replace`.

| Field | Default | Meaning |
|---|---|---|
| `seed` | 0 | master seed |
| `n_players` | 8 | total agents |
| `n_impostors` | 2 | impostors |
| `colors` | red, blue, green, pink, orange, yellow, black, white | agent ids, in this order |
| `max_ticks` | 600 | draw if reached |
| `tasks_per_crewmate` | 4 | real tasks per crewmate (k ≤ 5 for Held–Karp) |
| `fake_tasks_per_impostor` | 4 | blending targets |
| `task_duration_noise` | 1 | ± uniform integer ticks |
| `kill_cooldown` | 20 | ticks, also set after each meeting |
| `buttons_per_agent` | 1 | emergency meetings |
| `button_room` | cafeteria | |
| `p_fn` | 0.02 | per-occupant miss probability |
| `p_fn_lights` | 0.50 | crewmate miss probability during lights sabotage |
| `sabotage_cooldown` | 30 | shared by both impostors |
| `reactor_timer` | 30 | ticks to fix or impostors win |
| `reactor_panels` | (reactor, o2) | two rooms, deliberately far apart |
| `panel_hold_ticks` | 2 | ticks an agent must stay on a panel |
| `eta_grace` | 3 | ticks past ETA before revocation |
| `lights_fix_room` | electrical | |
| `doors_duration` | 10 | ticks edges stay removed |
| `claims_per_meeting` | 2 | message budget per agent per meeting |
| `meeting_rounds` | 2 | one claim per round |
| `ejection_reveals_role` | True | |
| `theta_vote` | 0.35 | min marginal to vote for someone (else skip) |
| `theta_call` | 0.75 | min marginal to press the button |
| `theta_shadow` | 0.50 | min marginal to start shadowing a co-located agent |
| `alpha_risk` | 2.0 | risk weight in edge cost |
| `last_seen_decay` | 15 | ticks after which a sighting stops localising an agent |
| `replan_belief_eps` | 0.10 | L1 marginal shift that triggers replanning |
| `eps_opportunity` | 0.02 | opportunity value for an alibied suspect |
| `p_noise` | 0.05 | likelihood of a contradiction between two honest sources |
| `eps_witness` | 0.001 | likelihood of a witnessed kill under a hypothesis excluding the killer |
| `p_miss_honest` | 0.10 | P(missed commitment \| honest) |
| `p_miss_impostor` | 0.60 | P(missed commitment \| impostor) |
| `likelihood_floor` | 1e-6 | numerical floor |
| `impostor_horizon` | 15 | time-expanded DAG horizon in ticks |
| `beta_isolation` | 3.0 | reward weight for isolated target |
| `gamma_witness` | 1.0 | cost weight per expected witness |
| `p_defect` | 0.5 | P(impostor defects on a won reactor commitment) |
| `impostor_underbid` | 0.5 | multiplier impostor applies to its bid when planning to defect |
| `alibi_style` | solo | `solo` or `name_other` (§12.2) |
| `deadlock_window` | 20 | ticks with zero global task progress |
| `shadow_cooldown` | 15 | ticks shadowing is disabled after a stall |
| `deadlock_protocol` | True | ablation switch |
| `commit_protocol` | True | ablation switch (False = everyone rushes nearest panel) |
| `belief_model` | exact | `exact`, `uniform`, `heuristic` |
| `planner` | ucs | `ucs` or `ucs_no_risk` |
| `verbosity` | 1 | 0 silent, 1 events, 2 events + beliefs, 3 debug |

---

## §4 Core types and module contracts

`types.py` (sketch, extend as needed; keep names):

```python
RoomId = str
AgentId = str

class Role(Enum): CREWMATE = "crewmate"; IMPOSTOR = "impostor"

@dataclass
class Transit:          # agent is inside a corridor
    src: RoomId; dst: RoomId; remaining: int; via_vent: bool = False

@dataclass
class AgentPhys:
    id: AgentId; role: Role; alive: bool
    room: RoomId | None            # None while in transit
    transit: Transit | None
    tasks: list["TaskInstance"]    # real for crew, fake for impostors
    kill_cooldown: int; buttons_left: int

@dataclass
class TaskInstance:
    task_id: str; room: RoomId; duration: int; progress: int = 0
    @property
    def done(self) -> bool: return self.progress >= self.duration

@dataclass
class Body:
    victim: AgentId; room: RoomId; tick: int; reported: bool = False

class Phase(Enum): PLAY = "play"; MEETING = "meeting"; OVER = "over"

@dataclass
class WorldState:
    tick: int; phase: Phase
    agents: dict[AgentId, AgentPhys]
    bodies: list[Body]
    sabotage: "SabotageState | None"
    closed_edges: set[frozenset[RoomId]]
    task_bar: float                 # public, 0..1
    last_progress_tick: int
    meeting_count: int
    winner: Role | None

# Actions (one per agent per tick)
@dataclass
class Move:     to: RoomId                    # next room on planned path
@dataclass
class Vent:     to: RoomId                    # impostor only
@dataclass
class DoTask:   task_id: str
@dataclass
class HoldPanel: room: RoomId                 # reactor panel or lights fix
@dataclass
class Kill:     target: AgentId               # impostor only
@dataclass
class Report:   pass                          # body in room
@dataclass
class PressButton: pass                       # must be in button_room
@dataclass
class Sabotage: kind: str; target: RoomId | None = None   # "reactor"|"lights"|"doors"
@dataclass
class Wait:     pass
Action = Move | Vent | DoTask | HoldPanel | Kill | Report | PressButton | Sabotage | Wait

@dataclass
class Event:
    tick: int; type: str; data: dict        # e.g. type="KILL", data={"killer":..,"victim":..,"room":..}
```

`contracts.py`:

```python
class Planner(Protocol):
    def distances(self, graph_view: "GraphView", cost_fn: CostFn) -> "DistMatrix": ...
    def path(self, graph_view, src: RoomId, dst: RoomId, cost_fn) -> tuple[list[RoomId], float, SearchStats]: ...
    def task_order(self, start: RoomId, rooms: list[RoomId], dist: DistMatrix) -> tuple[list[RoomId], float, SearchStats]: ...

class BeliefModel(Protocol):
    def update(self, evidence: "Evidence") -> None: ...
    def marginals(self) -> dict[AgentId, float]: ...
    def pair_mass(self, pair: frozenset[AgentId]) -> float: ...
    def entropy(self) -> float: ...

class Policy(Protocol):
    def decide(self, obs: "Observation") -> Action: ...
    def meeting_message(self, ctx: "MeetingContext", round_idx: int) -> "Message | None": ...
    def vote(self, ctx: "MeetingContext") -> AgentId | None: ...   # None = skip
    def radio(self, obs: "Observation") -> list["Message"]: ...      # sabotage coordination only
```

`GraphView` is a read-only view of the map with `closed_edges` applied and a flag `allow_vents`.

---

## §5 Map specification

Fourteen rooms, loosely based on a well-known spaceship layout. Coordinates are only for drawing and the A* benchmark heuristic.

```python
ROOMS: dict[str, tuple[float, float]] = {
    "upper_engine": (1.0, 5.0), "reactor": (0.0, 3.0), "security": (2.0, 3.0),
    "lower_engine": (1.0, 1.0), "medbay": (4.0, 4.0), "cafeteria": (6.0, 5.0),
    "weapons": (9.0, 5.0), "o2": (8.0, 4.0), "navigation": (11.0, 3.0),
    "shields": (9.0, 1.0), "communications": (7.0, 0.0), "storage": (6.0, 1.0),
    "admin": (7.0, 2.5), "electrical": (4.0, 1.5),
}

# Undirected corridors, weight = travel ticks
EDGES: list[tuple[str, str, int]] = [
    ("cafeteria", "weapons", 3), ("cafeteria", "medbay", 3), ("cafeteria", "upper_engine", 5),
    ("cafeteria", "storage", 4), ("cafeteria", "admin", 4), ("weapons", "o2", 2),
    ("weapons", "navigation", 4), ("o2", "navigation", 3), ("o2", "shields", 4),
    ("navigation", "shields", 4), ("shields", "communications", 2), ("shields", "storage", 4),
    ("communications", "storage", 3), ("storage", "admin", 2), ("storage", "electrical", 4),
    ("storage", "lower_engine", 6), ("electrical", "lower_engine", 5), ("lower_engine", "security", 3),
    ("lower_engine", "reactor", 3), ("lower_engine", "upper_engine", 5), ("security", "reactor", 2),
    ("security", "upper_engine", 3), ("reactor", "upper_engine", 3), ("upper_engine", "medbay", 4),
]  # 24 edges

# Vents: impostor-only, 1 tick, stored as DIRECTED edges (both directions listed at load time)
VENT_PAIRS: list[tuple[str, str]] = [
    ("reactor", "upper_engine"), ("reactor", "lower_engine"), ("electrical", "medbay"),
    ("electrical", "security"), ("medbay", "security"), ("cafeteria", "admin"),
    ("navigation", "weapons"), ("navigation", "shields"),
]

TASK_POOL: list[tuple[str, str, int]] = [   # (task_id, room, base_duration)
    ("t01", "upper_engine", 4), ("t02", "lower_engine", 4), ("t03", "reactor", 5),
    ("t04", "security", 3), ("t05", "medbay", 5), ("t06", "cafeteria", 3),
    ("t07", "weapons", 4), ("t08", "o2", 3), ("t09", "navigation", 4),
    ("t10", "navigation", 3), ("t11", "shields", 4), ("t12", "communications", 3),
    ("t13", "storage", 4), ("t14", "admin", 3), ("t15", "electrical", 4),
    ("t16", "electrical", 5),
]
```

Load-time validation (tested): 14 rooms, 24 edges, graph connected, all edge weights ≥ 1, every vent endpoint is a room, every task room exists, `button_room` and panel rooms exist.

The map is static. Dynamic state (closed edges, bodies, sabotage) lives in `WorldState`.

---

## §6 Engine

### 6.1 Setup

1. `rng = make_rng(config.seed)`.
2. Roles: choose `n_impostors` ids uniformly without replacement (scenario may override).
3. Tasks: each crewmate draws `tasks_per_crewmate` distinct tasks from the pool; duration = base + uniform integer in [−noise, +noise], minimum 1. Impostors draw fake tasks the same way (never counted in the task bar).
4. Spawn: all agents in `cafeteria` at tick 0 (scenario may override).
5. Kill cooldowns set to `kill_cooldown`.

### 6.2 Tick pipeline (phase PLAY), in this exact order

1. **Decide.** For every alive agent in fixed id order, `action = policy.decide(obs_t)` (or a scripted action from the scenario, §16). Radio messages from `policy.radio(obs_t)` are collected too. Decisions are simultaneous: no agent sees another's action this tick.
2. **Sabotage triggers.** Valid `Sabotage` actions start a sabotage (§13). At most one active sabotage; if two impostors trigger on the same tick, the first in id order wins.
3. **Movement.** Agents in transit decrement `remaining`; at 0 they arrive. `Move(to)` from a room starts a transit along an open edge (`remaining = weight`); an agent with `weight = 1` arrives this tick. `Vent(to)` moves an impostor to the destination in 1 tick (arrives this tick). Agents in transit are not in any room and cannot be observed, killed, or act.
4. **Kills.** A `Kill(target)` is valid if killer is an alive impostor, cooldown is 0, killer and target are in the same room after movement, target is alive and not an impostor. If two kills target the same victim, the first in id order resolves, the other is void. Resolution: victim dies, a `Body` is created, killer cooldown resets. Witnesses are other alive agents in that room; each observes the kill with probability `1 − p_fn_effective` (§7).
5. **Work.** `DoTask` increments progress if the agent is in the task room. `HoldPanel` increments the panel hold counter (§13). Crewmate task progress updates `task_bar` and `last_progress_tick`.
6. **Reports and button.** `Report` is valid if an unreported body is in the agent's room. `PressButton` is valid in `button_room` with `buttons_left > 0`. Either flags a meeting. **No meetings while a reactor sabotage is active**: the flag is queued and fires the tick the reactor is fixed.
7. **Timers.** Decrement cooldowns, sabotage timers, door timers (reopen edges at 0). Run the deadlock detector (§14).
8. **Win check.** Crew wins if all real tasks are done or all impostors are ejected. Impostors win if alive impostors ≥ alive crewmates, or the reactor timer reaches 0. Draw at `max_ticks`.
9. **Observe.** Build `obs_{t+1}` for every alive agent (§7) including this tick's events.
10. **Meeting.** If flagged, run §11, then respawn all alive agents in `cafeteria`, cancel transits, reset kill cooldowns to `kill_cooldown`, clear lights and doors (not reactor, which blocks meetings anyway), and mark all bodies as reported.

Every state change emits an `Event` to the event log (§15).

---

## §7 Observation model

This is what makes the environment partially observable. Get it exactly right.

An `Observation` for agent `i` at tick `t` contains:

**Local (own room only, only if in a room):**
- `room`, `tick`, own `AgentPhys` (role, tasks, cooldowns, buttons).
- `occupants`: each other alive agent in the room is included independently with probability `1 − p_fn_eff`.
- `bodies_here`: bodies in the room (always visible; during lights, visible with probability 0.7).
- `local_events`: kills and vent entries/exits that happened in this room this tick, each seen with probability `1 − p_fn_eff`. A vent exit is attributed to the agent who emerged.

`p_fn_eff = p_fn_lights` for crewmates during lights sabotage, else `p_fn`. Impostors are unaffected by lights.

**Global (every alive agent):**
- `task_bar` (0..1).
- Sabotage alarms: kind, panel rooms, timer remaining, panel hold status (reactor), fix status. **The saboteur's identity is never revealed.**
- Door closures: which room is sealed.
- Radio messages from this tick (sabotage coordination only, §10).
- `PROGRESS_STALL` event (§14).
- Meeting transcripts, votes and ejection results.
- Death notices only for bodies reported (unreported deaths are unknown).

**Impostor extra:** partner identity, partner's current room, private partner messages (`TARGET_CLAIM`).

Agents never receive `WorldState`. Tests must enforce that a policy only ever receives an `Observation`.

The three-way distinction used on the slides, reflected in code: hidden state = roles, other rooms, unreported bodies, truth of claims. Future actions and strategic choices are not modelled as hidden state.

---

## §8 Search and planning

A* is **not** the production route planner. With |V| = 14, no admissible heuristic prunes meaningfully; A* is kept only as a benchmark baseline. Real search difficulty lives in task ordering and impostor planning.

### 8.1 Uniform-cost search (`search/ucs.py`)

```
ucs(view, src, dst, cost_fn) -> (path, cost, stats)
  frontier = min-heap of (g, tie_id, room); g[src] = 0
  while frontier:
      pop (g, _, u); if u already closed: continue; close u; stats.expanded += 1
      if u == dst: reconstruct and return
      for (v, w) in view.neighbors(u):          # respects closed_edges and allow_vents
          c = g + cost_fn(u, v, w)
          if c < best[v]: best[v] = c; parent[v] = u; push (c, next_tie, v)
  return ([], inf, stats)
```

Ties break deterministically on a monotonically increasing counter, then room name. `all_pairs(view, cost_fn)` runs UCS from every source (14 runs) and returns a `DistMatrix` plus next-hop table. This matrix is rebuilt on every replan trigger (§8.5); it is cheap enough that incremental repair (D* Lite) is not worth its complexity.

### 8.2 Risk-weighted cost (`search/costs.py`)

For crewmate `i`: `cost(u, v, w) = w + alpha_risk · risk_i(v)`.

`risk_i(v) = Σ_s m_i(s) · loc_i(s, v)`, where `m_i(s)` is the belief marginal that `s` is an impostor and `loc_i(s, v)` is 1 if `i` saw `s` in `v` within the last `last_seen_decay` ticks, spread uniformly over rooms within graph distance ≤ (ticks since sighting) otherwise, and 0 once older than `last_seen_decay`. Costs are non-negative, so UCS remains correct. With `planner = ucs_no_risk`, `alpha_risk = 0`.

### 8.3 Held–Karp task ordering (`search/held_karp.py`)

Open-path TSP from the agent's current room over the distinct rooms of remaining tasks (k ≤ 5), using the `DistMatrix`.

```
dp[mask][j] = min cost to start at s, visit exactly the set mask, end at room j
dp[{j}][j] = d(s, j)
dp[mask][j] = min over i in mask\{j} of dp[mask\{j}][i] + d(i, j)
answer = min_j dp[full][j]; reconstruct order via parent table
```

Complexity O(k²·2ᵏ), about 800 operations at k = 5. Count state evaluations in `SearchStats`. Multiple tasks in one room collapse into one visit.

### 8.4 Impostor time-expanded DAG (`search/time_expanded.py`)

Nodes are `(room, τ)` for τ ∈ [t, t + H], `H = impostor_horizon`. Edges: wait `(v, τ) → (v, τ+1)`; walk `(u, τ) → (v, τ+w)` over open corridors; vent `(u, τ) → (v, τ+1)`. Because τ only increases, the graph is a DAG, so a single forward pass in τ order is optimal.

Predicted occupancy: for each non-partner agent `a`, `P(a at v, τ)` is uniform over rooms within graph distance ≤ (τ − last_seen_tick) of its last sighting (all rooms if never seen), capped at a full uniform distribution.

```
E_w(v, τ)   = Σ_a P(a at v, τ)
P_iso(v, τ) = max_a [ P(a at v, τ) · Π_{b ≠ a} (1 − P(b at v, τ)) ]
node_cost(v, τ) = gamma_witness · E_w(v, τ)
terminal_value(v, τ) = −beta_isolation · P_iso(v, τ)
best = argmin over nodes of [ path_cost(root → node) + terminal_value(node) ]
```

Return the first action on the best path and the target agent. Partner coordination: if the partner has broadcast `TARGET_CLAIM(a)`, exclude `a` from `P_iso`.

### 8.5 Replanning triggers (`agents/base.py`)

An agent recomputes its `DistMatrix`, task order and path only when one fires:

| Trigger | Condition |
|---|---|
| `arrival` | reached the current goal room |
| `belief_shift` | L1 change of marginals since last plan > `replan_belief_eps` |
| `topology` | an edge closed or reopened |
| `new_goal` | sabotage alarm, body seen, commitment won or revoked, shadow start or stop |
| `meeting_end` | respawn after a meeting |
| `blocked` | next hop is on a closed edge |

Each replan emits a `REPLAN` event with reason and `nodes_expanded`.

### 8.6 Baselines (`search/baselines.py`, benchmarks only)

- `bfs(view, src, dst)`: minimises hops; used to show BFS returns non-optimal travel time.
- `astar(view, src, dst, cost_fn)`: heuristic `h(u) = euclid(u, dst) / s`, where `s = max over edges of euclid(u, v) / w` computed at load time, which makes `h` admissible. Used to show node expansions ≈ UCS on this graph.
- `greedy_order(start, rooms, dist)`: nearest-next task ordering; used to measure Held–Karp's optimality gap.
- Brute-force permutation ordering: test oracle for Held–Karp.

`scripts/bench_search.py` runs all pairs and random task sets and writes `runs/bench/search.csv`.

---

## §9 Belief model

### 9.1 Hypotheses

`H` = all 2-element subsets of the 8 agent ids, in lexicographic order: 28 hypotheses. Store the belief as a `numpy` array of log-probabilities of length 28, plus a precomputed boolean matrix `contains[h, agent]`.

### 9.2 Prior and hard constraints

- Crewmate `i`: uniform over hypotheses not containing `i` (21 of 28); hypotheses containing `i` are `−inf`.
- Reported victim `v`: hypotheses containing `v` become `−inf` (bodies are always crewmates).
- Ejection with reveal: if `x` revealed impostor, hypotheses not containing `x` become `−inf`; if revealed crewmate, hypotheses containing `x` become `−inf`.
- Impostors know the truth. They additionally run a **public belief**: the same update using only public information (meeting transcripts, public events) from a generic crewmate's view. It is used to choose accusations and to judge which lies are safe.

### 9.3 Facts and trust (`belief/facts.py`)

A `Fact` is `(subject, room, t_from, t_to, source, kind)` where `kind ∈ {present, kill, vent}`. Sources are the agent itself (direct observation) or a speaker (testimony).

**Trust rule, per hypothesis:** a fact with `source == self` is always trusted. A fact from speaker `j` is trusted under `h` iff `j ∉ h`. This single rule is the deception model: the same claim counts as evidence under hypotheses where the speaker is honest and is ignored where the speaker would be lying.

**Contradiction:** two facts about the same subject, from different sources, whose intervals are closer in time than the travel distance between their rooms allows (using the vent-inclusive distance if the subject would be an impostor under the hypothesis being scored, else the corridor distance). Facts from the same source never contradict each other.

### 9.4 Evidence types and likelihoods (`belief/evidence.py`)

All likelihoods are computed per hypothesis `h`, floored at `likelihood_floor`, and multiplied in log space.

| Evidence | L(e \| h) |
|---|---|
| Witnessed kill or vent by `s` (own observation) | 1 if `s ∈ h`, else `eps_witness` |
| Witnessed kill or vent by `s` (testimony from `j`) | if `j ∉ h`: as above; if `j ∈ h`: 1 (uninformative) |
| Body found (see opportunity below) | `max(eps_opportunity, 1 − Π_{s ∈ h, alive at kill} (1 − opp_h(s)))` |
| Contradiction between sources `j` and `k` | `p_noise` if `j ∉ h` and `k ∉ h`; 1 otherwise |
| Contradiction between own observation and speaker `j` | `p_noise` if `j ∉ h`; 1 otherwise |
| Missed commitment by `j` (§13) | `p_miss_honest` if `j ∉ h`; `p_miss_impostor` if `j ∈ h` |

**Opportunity `opp_h(s)` for a body found at tick `t_f` in room `r`:**
1. Kill window `W = [t_ls, t_f]`, where `t_ls` is the latest trusted sighting of the victim alive (own or testimony trusted under `h`), else the last meeting end.
2. For each trusted-under-`h` fact placing `s` in room `x` at tick `τ`, exclude the interval `(τ − d(x, r), τ + d(x, r))` from `W`, where `d` is the vent-inclusive shortest distance (`s ∈ h` in every case where this matters).
3. If any part of `W` remains, `opp_h(s) = 1`; otherwise `opp_h(s) = eps_opportunity`.

This reuses the search module's distance matrix inside belief, which is worth mentioning in the presentation.

### 9.5 Update procedure (`belief/belief.py`)

```
update(evidence):
    for h in active hypotheses: logb[h] += log(max(L(evidence | h), floor))
    logb -= logsumexp(logb[active])
marginals: m(s) = Σ_{h ∋ s} exp(logb[h])        # sums to 2 across agents
pair_mass(pair) = exp(logb[index(pair)])
entropy = −Σ p log p over active hypotheses
```

Updates happen at observation time (own sightings), at body discovery, after each meeting round, on commitment failures, and on ejections. Body-opportunity likelihoods are recomputed after each meeting round because new testimony changes which facts are trusted; implement this by storing body evidence and re-scoring it from the prior plus all non-body evidence (simplest correct approach: recompute the full posterior from the evidence list; 28 hypotheses make this trivial).

Ablations: `belief_model = uniform` (never updates beyond hard constraints) and `heuristic` (suspicion score = count of accusations + contradictions, normalised). Both implement `BeliefModel`.

---

## §10 Communication

### 10.1 Channels

| Channel | When | Who hears |
|---|---|---|
| Meeting | only during meetings | all alive agents |
| Radio | only while a sabotage is active, only protocol messages | all alive agents |
| Partner | any time | the other impostor |

There is no free chat during play. This is the communication bottleneck on the slides: an agent accumulates many private facts between meetings but may voice at most `claims_per_meeting` of them.

### 10.2 Message schema (`comms/messages.py`)

```python
@dataclass(frozen=True)
class Message:
    type: str; sender: AgentId; tick: int; payload: dict
```

| Type | Payload | Channel |
|---|---|---|
| `CLAIM_LOCATION` | `room, t_from, t_to, with: [agent]` | meeting |
| `REPORT_SIGHTING` | `subject, room, tick, event: present\|kill\|vent` | meeting |
| `ACCUSE` | `subject, reason: witnessed\|contradiction\|no_alibi\|missed_commit\|high_posterior` | meeting |
| `DEFEND` | `claim: <restated CLAIM_LOCATION payload>` | meeting |
| `BID_FIX` | `costs: {panel_room: ticks}` | radio |
| `COMMIT_FIX` | `panel, eta` | radio |
| `REVOKE_FIX` | `agent, panel` | radio (emitted by protocol) |
| `TARGET_CLAIM` | `target` | partner |
| `VOTE` | `target \| None` | meeting |

Every `CLAIM_LOCATION` and `REPORT_SIGHTING` becomes `Fact`s for listeners: the sender's own location, and one `present` fact per agent in `with`. Crewmates only emit facts they actually observed. Impostors may fabricate (§12.2).

### 10.3 Crewmate claim selection

Score candidates and emit the best unused one per round:

| Candidate | Score | Round |
|---|---|---|
| `REPORT_SIGHTING` of a witnessed kill or vent | 100 | 1 or 2 |
| `REPORT_SIGHTING` that contradicts a round-1 claim | 80 | 2 |
| `DEFEND` if accused in round 1 | 60 | 2 |
| `CLAIM_LOCATION` covering the kill window, with co-occupants | 50 | 1 or 2 |
| `ACCUSE` top suspect if its marginal > 0.4 | 40 | 1 or 2 |

---

## §11 Meeting protocol (`comms/meeting.py`)

1. Triggered by `Report` (reporter speaks first) or `PressButton` (presser speaks first). Remaining speakers follow in an RNG-shuffled order.
2. **Round 1:** each alive agent emits at most one message. After the round, every listener converts messages to facts and evidence and updates belief.
3. **Round 2:** same, and agents may respond to round-1 content (contradictions, defences).
4. **Vote:** crewmates vote for `argmax m(s)` over alive others if that marginal ≥ `theta_vote`, else skip. Impostors vote for the non-partner with the highest public-belief marginal if the current plurality leader is a crewmate (join the crowd); otherwise skip.
5. **Resolution:** plurality over targets. Ties, or skip ≥ top count, eject nobody. If ejected and `ejection_reveals_role`, the role is announced and applied as a hard constraint (§9.2).
6. Log every message, the belief snapshot of each crewmate after each round (top 3 marginals and true-pair mass), votes and outcome.

The meeting is the only point where the world clock pauses.

---

## §12 Agent policies

### 12.1 Crewmate (`agents/crewmate.py`), priority order

1. Witnessed a kill this tick or a body is in the room → `Report`.
2. Committed to a reactor panel → route to it with UCS (no risk term) and `HoldPanel`.
3. Lights active and graph distance to `electrical` ≤ 6 → go fix (`HoldPanel` in electrical for `panel_hold_ticks`).
4. `max m(s) ≥ theta_call` and `buttons_left > 0` and no reactor → route to `cafeteria` and `PressButton`.
5. Shadowing allowed (not in cooldown), a co-located agent `s` has `m(s) ≥ theta_shadow` → **shadow**: each tick move toward `s`'s last seen room; stay if already together.
6. Remaining tasks → Held–Karp order, risk-weighted UCS route to the next task room, `DoTask`.
7. Tasks done → move to the adjacent room with lowest risk and keep observing.

### 12.2 Impostor (`agents/impostor.py`), priority order

1. Can kill: cooldown 0, a crewmate target in the room, and no other observed non-partner alive occupant → `Kill`. Next tick: `Vent` away if a vent exists in the room, else move along the lowest-witness path.
2. Sabotage off cooldown, ≥ 3 crewmates alive, per-tick probability 0.1 → pick one: `reactor` (weight 0.4), `lights` (0.4, preferred when a kill is planned within 5 ticks), `doors` (0.2, target = room holding the intended victim).
3. Reactor active and the impostor won a commitment → with probability `p_defect` it stalls (moves elsewhere or waits); otherwise it fixes honestly. When planning to defect, its `BID_FIX` costs are multiplied by `impostor_underbid` to win the commitment.
4. Hunt: run §8.4, send `TARGET_CLAIM` to partner, take the first step.
5. Blend: walk to fake task rooms and stand for the task duration.

**Meeting behaviour:**
- `CLAIM_LOCATION` for the kill window: with `alibi_style = solo`, claim a room adjacent to its true trajectory where no alive non-partner could have seen it (the impostor assumes anyone it saw also saw it). With `name_other`, also name a crewmate in `with`, which produces a checkable alibi that the named agent may deny (used by scenario S2).
- `ACCUSE`: if its own public-belief marginal is the highest, counter-accuse the reporter or its accuser; otherwise accuse the crewmate with the highest public marginal.
- `DEFEND` if accused: restate its claim.
- Never self-reports bodies (baseline).

---

## §13 Sabotage and the commitment protocol

`world/sabotage.py` holds the state machines; `comms/protocol.py` holds the reactor negotiation.

### 13.1 Lights
Active until an agent holds the `electrical` panel for `panel_hold_ticks`. Effects: crewmate `p_fn_eff = p_fn_lights`, bodies visible with 0.7.

### 13.2 Doors
Closes every corridor incident to the target room for `doors_duration` ticks (vents unaffected). Agents already in transit continue. Emits a `topology` replan trigger for all agents.

### 13.3 Reactor (the headline demo)

Timer `reactor_timer`. Two panels at `reactor_panels`. Each panel is done after one agent holds it for `panel_hold_ticks` (not necessarily simultaneously). Both done ⇒ fixed. Timer 0 ⇒ impostors win.

**Protocol (no central coordinator; every agent runs the same deterministic function on the same public messages):**

1. Alarm tick `t`: every alive agent broadcasts `BID_FIX` with UCS travel costs to each panel. Impostors planning to defect underbid.
2. Tick `t + 1`: every agent computes the same assignment: the pair `(a, b)`, `a ≠ b`, minimising `max(cost_a[P1], cost_b[P2])`, ties broken by agent id. The two winners broadcast `COMMIT_FIX(panel, eta = t + 1 + cost)`.
3. Monitoring: panel hold status is public. If `tick > eta + eta_grace` and that panel is not done, the commitment is **revoked** (`REVOKE_FIX` event), a `missed_commitment` fact is recorded against the agent, and the next-best bidder for that panel (excluding revoked agents) commits with a new ETA.
4. Every crewmate applies the missed-commitment likelihood (§9.4) immediately, so the defector's marginal rises before the next meeting.

With `commit_protocol = False` (ablation baseline), every agent rushes the nearest panel with no commitments or revocations.

Honest agents can also miss (doors, long paths); `p_miss_honest` covers that.

---

## §14 Deadlock detection (`agents/deadlock.py`)

**The deadlock:** two crewmates who each suspect the other both choose to shadow (§12.1 step 5). Each waits for the other to move; neither does tasks; if they hold the remaining tasks, global progress stops.

**Detection (engine):** if `tick − last_progress_tick ≥ deadlock_window`, no meeting occurred in that window and no sabotage is active, emit a global `PROGRESS_STALL` event (the task bar is public, so every agent can see it has stalled).

**Resolution (agents):** on `PROGRESS_STALL`, every crewmate stops shadowing, sets `shadow_disabled_until = tick + shadow_cooldown`, and replans its task route (§8.5 `new_goal`). Log `DEADLOCK_BROKEN` when progress resumes.

With `deadlock_protocol = False` the stall event is not emitted (ablation).

---

## §15 Telemetry and logging (`sim/telemetry.py`)

### 15.1 Event log
Every game writes `runs/<scenario>_<seed>/events.jsonl`, one JSON object per line: `{"t": int, "type": str, ...data}`. Required types: `SPAWN, ROLES (hidden, for analysis only), MOVE, ARRIVE, VENT, KILL, BODY_SEEN, REPORT, BUTTON, MEETING_START, MESSAGE, BELIEF, VOTE, EJECT, SABOTAGE, BID, COMMIT, REVOKE, PANEL_DONE, SABOTAGE_FIXED, DOORS_OPEN, REPLAN, PROGRESS_STALL, DEADLOCK_BROKEN, GAME_OVER`.

`BELIEF` records: `agent, true_pair_mass, entropy, top: [[id, marginal], x3]`. Emit at every meeting round and every 10 ticks.

Reproducibility test: hashing `events.jsonl` for the same seed twice gives the same digest.

### 15.2 Terminal output (verbosity 1–2)

```
[t=041] MOVE     blue    storage -> electrical (eta 4)
[t=043] KILL     blue x red @electrical   witnesses=[]
[t=047] REPORT   green found red @electrical -> MEETING #1
[M1 r1] green    REPORT_SIGHTING blue present @electrical t=41
[M1 r1] blue     CLAIM_LOCATION navigation t=38-47 with=[pink]
[M1 r2] pink     CLAIM_LOCATION shields t=38-47 with=[]      ! contradicts blue
[M1 bel] green   blue .44 | black .21 | white .12   true-pair .31
[M1 vote] blue:4 green:1 skip:1 -> EJECT blue (IMPOSTOR)
[t=120] SABOTAGE reactor panels=[reactor,o2] timer=30
[t=121] COMMIT   yellow->reactor eta=129 | black->o2 eta=126
[t=130] REVOKE   black o2 (eta 126 + 3) -> BACKUP white eta=135
[t=133] REPLAN   green reason=topology expanded=58
```

Use `rich` colours if installed, plain text otherwise.

### 15.3 Map rendering (optional, M6)
`analysis/plots.py::draw_map(state)` draws the room graph with networkx at the §5 coordinates, closed edges dashed, agents as coloured dots offset around their room. `--frames` writes one PNG per tick for a demo GIF.

---

## §16 Scenarios (`sim/scenarios.py`)

```python
@dataclass
class Scenario:
    name: str; seed: int
    config_overrides: dict = field(default_factory=dict)
    roles: tuple[AgentId, AgentId] | None = None
    spawns: dict[AgentId, RoomId] | None = None
    scripted: dict[int, dict[AgentId, Action]] = field(default_factory=dict)  # tick -> agent -> action
    sabotage_schedule: dict[int, Sabotage] = field(default_factory=dict)
    policy_overrides: dict[AgentId, dict] = field(default_factory=dict)
```

A scripted action replaces the policy decision for that agent and tick; everything else runs normally.

| Id | Name | Setup | Acceptance (observable in the log) |
|---|---|---|---|
| S1 | `baseline` | seed 7, defaults | Game ends with a winner; at least one meeting; belief lines printed |
| S2 | `worked_example` | roles (blue, black); script blue and red into electrical alone around t=41, blue kills at t=43, green enters and reports at t=47; blue `alibi_style=name_other` naming pink | Pink contradicts blue in round 2; green's marginal on blue rises across rounds; blue is ejected with the default seed. Put the actual numbers from this run on the slide |
| S3 | `reactor_defect` | reactor at a tick when black (impostor) is nearest a panel; `p_defect=1` for black | `COMMIT` to black, `REVOKE` at eta+3, backup commits, reactor fixed; black's marginal rises before the next meeting |
| S4 | `lights_out` | lights at t=60, a kill scheduled soon after | Crewmate belief entropy stays higher / true-pair mass grows slower than S1 over the same window |
| S5 | `doors` | doors on a room on a crewmate's planned path | `REPLAN reason=topology` with a different path and logged expansions |
| S6 | `standoff` | two crewmates with the last tasks, mutually suspicious via scripted early evidence | `PROGRESS_STALL` after 20 ticks, `DEADLOCK_BROKEN`, tasks complete; with `deadlock_protocol=False` the stall persists |
| S7 | `stress` | 100 seeds, all shocks enabled | No crashes; summary table written |

CLI: `python -m amongus.sim.runner --scenario reactor_defect --seed 3 --verbosity 2`.

---

## §17 Metrics and experiments

`analysis/experiments.py` runs batches (seeded ranges, config overrides) and writes `runs/batch/<name>/summary.csv`. Report means with 95% intervals (normal approximation is fine).

| Metric | Definition |
|---|---|
| Win rate | crew / impostor / draw over seeds |
| True-pair mass | mean over alive crewmates of `pair_mass(true_pair)` at each meeting, and over time |
| Ejection precision | ejected impostors / all ejections |
| Ejection recall | ejected impostors / impostors |
| Information throughput | distinct private facts voiced in meetings / distinct private facts collected |
| Sabotage response | reactor fix rate and mean ticks to fix, protocol vs `commit_protocol=False` |
| Search cost | expansions and path cost: BFS vs UCS vs A*; Held–Karp vs greedy vs brute force |
| Risk effect | crewmate survival rate, `alpha_risk` 2.0 vs 0 |
| Deadlock | stall incidence and task-completion rate, protocol on vs off |

Ablations to run: `belief_model ∈ {exact, heuristic, uniform}`, `planner ∈ {ucs, ucs_no_risk}`, `commit_protocol ∈ {True, False}`, `deadlock_protocol ∈ {True, False}`.

`analysis/plots.py` produces: true-pair mass over time with meeting markers (the headline figure), search-expansion bars, win rate by ablation, sabotage response comparison.

---

## §18 Tests (pytest, `tests/`)

Each implementer writes tests for its module in the same milestone. Minimum set:

- **map:** 14 rooms, 24 edges, connected, vents valid, tasks valid.
- **ucs:** equals `networkx.dijkstra_path_length` on all pairs; respects closed edges; vents only when allowed; deterministic tie-breaking.
- **baselines:** A* cost equals UCS cost on all pairs; heuristic admissible on all pairs; BFS cost ≥ UCS cost.
- **held_karp:** equals brute-force permutations on 200 random task sets (k ≤ 5); ≤ greedy.
- **time_expanded:** equals brute-force enumeration on a 4-room toy graph with H = 4.
- **engine:** determinism by event-log hash; transit agents invisible and unkillable; vent only by impostors; kill cooldown respected; no meetings during reactor; reactor timeout ends game; doors reopen.
- **observation:** policies receive `Observation` only; lights change crewmate miss rate, not impostors'.
- **belief:** 28 hypotheses; crewmate self-exclusion gives 21 active; normalisation to 1; marginals sum to 2; witnessed kill makes killer marginal > 0.95; own alibi drives `opp` to eps; testimony from a speaker counts under honest hypotheses only; contradiction lowers the joint-honest mass; reveal constraints applied.
- **meeting:** at most `claims_per_meeting` messages per agent; tie → no ejection; reveal applied.
- **protocol:** identical assignment computed by all agents; revocation exactly at eta + grace; backup commits; missed-commit evidence recorded.
- **deadlock:** S6 stalls without protocol and recovers with it.

---

## §19 Milestones

Each milestone ends with: tests green (via test-runner), a short entry in `docs/PROGRESS.md`, and a spec-review pass for M3, M4 and M6.

| Milestone | Owner(s) | Deliverables | Acceptance |
|---|---|---|---|
| **M0 Scaffold** | world-engineer | `pyproject.toml`, config, rng, types, contracts, map + validation, `WorldState`, random-walk stub policy, runner printing ticks | `pytest` green on map tests; `runner --seed 1 --ticks 20` deterministic |
| **M1a Search** | search-engineer | ucs, all_pairs, costs (risk hook stubbed), held_karp, baselines, time_expanded, stats, bench script | §18 search tests green; `bench_search.py` writes CSV |
| **M1b Movement & tasks** (parallel with M1a) | world-engineer | transit, vents, tasks, task bar, observation builder (§7) | engine and observation tests green |
| **M2 Kills & meetings skeleton** | world-engineer, then agent-engineer | kills, bodies, reports, button, win checks, meeting loop with `uniform` belief voting, simple crewmate (tasks via Held–Karp + UCS) and impostor (hunt via DAG) | full game runs to a winner on 20 seeds without errors |
| **M3 Belief** | belief-engineer | hypotheses, facts, trust, contradictions, opportunity, update, marginals, public belief; wire into votes and risk cost | belief tests green; over 50 seeds, mean true-pair mass at the final meeting > 0.15 (uniform baseline ≈ 0.048) |
| **M4 Comms, sabotage, protocols** | agent-engineer, world-engineer (sabotage state) | claim selection, lies, accusations, radio, reactor protocol, lights, doors, shadowing, deadlock | S3, S5, S6 acceptance met |
| **M5 Telemetry & experiments** | telemetry-engineer | telemetry formats, scenarios S1–S7, metrics, experiments, plots | all scenarios run from CLI; plots written to `runs/` |
| **M6 Demo polish** | telemetry-engineer, orchestrator | `scripts/run_demo.py` (S2 then S3 then S4, paced output), README, optional map frames, optional narration | 5-minute demo runs start to finish; spec review passes |

Parallelism: M1a ∥ M1b. M3 can start once M2's contracts and meeting skeleton exist. M5's telemetry formatting can start after M2.

---

## §20 Out of scope (do not build)

Pygame or web UI, continuous movement or collision physics, multi-agent pathfinding (CBS etc.; rooms have no capacity), reinforcement learning or any training, ghosts, cameras/admin table/vitals, multiple maps, free-text chat, LLM decision-making, pandas, a database. If an agent believes one of these is needed, it stops and asks the orchestrator.

---

## §21 Optional LLM narration

`narration/narrate.py` converts a finished meeting's structured messages into short readable sentences for the demo transcript. It runs after the meeting is resolved, never influences state, is disabled by default, reads the API key from the environment, and falls back to template strings if the key or package is missing.

Why the LLM stays out of the loop (for the viva): belief update is exact Bayesian inference against an explicit message-generating model. If an LLM generated the claims, the true generating process would no longer match the likelihoods, and the update would be unsound.
