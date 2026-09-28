# AI Among Us — Spec (v2, minimal MVP)

Single source of truth. If code and this document disagree, this document wins unless the deviation is recorded in `docs/DECISIONS.md`.

**Design goal: every line must be explainable under questioning.** When choosing between a clever solution and an obvious one, take the obvious one. The v1 Bayesian spec is archived at `docs/archive/v1_bayesian_reference.md` and is **not** authoritative — do not read it.

---

## §0 Global rules

- Python ≥ 3.11. Type hints everywhere, `dataclass` for data, `typing.Protocol` for contracts.
- **All world randomness** goes through one seeded `numpy.random.Generator` from `rng.py`. No `random` module, no unseeded calls.
- **The LLM is called only during meetings** (§9). Never inside the tick loop. Every call is cached by prompt hash, so a replay is deterministic.
- Every public function gets a one-line docstring. Modules stay under **~200 lines**. Split if larger.
- Every tunable lives in `SimConfig` (§3). No magic numbers elsewhere.
- Agents receive only an `Observation`, never `WorldState` (§7).
- The simulation must run end-to-end **with no API key** using template fallbacks (§9.4).

| § | Topic | Owner |
|---|---|---|
| §1–§4 | Summary, stack, config, contracts | world-engineer |
| §5–§7 | Map, engine, observation | world-engineer |
| §8 | Search: A\* + baselines | search-engineer |
| §9 | LLM deliberation | agent-engineer |
| §10–§11 | Meetings, voting, agent policies | agent-engineer |
| §12–§13 | Sabotage, renegotiation, deadlock | agent-engineer (protocol), world-engineer (state) |
| §14, §16 | Telemetry, scenarios, shocks | telemetry-engineer |
| §15 | UI replay viewer | ui-engineer |
| §17 | Tests | each implementer |
| §18 | Milestones | orchestrator |
| §19 | Out of scope | everyone |

---

## §1 System summary

Eight agents (6 crewmates, 2 impostors) move on a 14-room weighted graph in discrete ticks. Each agent sees only its own room, so the environment is partially observable and stochastic.

**The architecture in one sentence: search handles space, the LLM handles people.**

- **Spatial reasoning is algorithmic.** Crewmates route with **A\*** over a risk-weighted room graph, do tasks, and report bodies. Impostors kill when isolated, vent, and sabotage. Zero LLM calls in the tick loop.
- **Social reasoning is delegated to an LLM.** Each agent accumulates plain-English *notes* about what it observed. At a meeting, it sends its notes plus the public transcript to Gemini Flash and gets back a short statement, a suspect ranking, and a vote. Impostors are told to lie.
- The suspect ranking feeds back into A\* as an edge-cost penalty, so crewmates physically route around agents they distrust.

Two coordination mechanisms are exercised on purpose: a **reactor renegotiation protocol** (commit → defect → revoke → backup) and a **shadowing deadlock** broken by a zero-progress timeout.

### Why an LLM instead of exact Bayesian inference

Exact inference needs a hand-written likelihood for every possible message and assumes the message-generating process is known. That is tractable only if agents speak a fixed schema. The moment testimony is natural language the observation space is unbounded and there is no likelihood to specify. We wanted open testimony and deception, so we traded exact inference over a fixed schema for approximate reasoning over free text.

The cost is real and must be stated, not hidden: we lose the correctness guarantee and per-call determinism. That is why the world simulation stays fully seeded and every LLM call is cached by prompt hash — a recorded run replays identically.

---

## §2 Stack and layout

| Package | Used for |
|---|---|
| `numpy` ≥1.26 | seeded RNG |
| `networkx` ≥3.2 | graph container, Dijkstra test oracle |
| `mesa` ≥3.0,<4 | thin model wrapper + `DataCollector` |
| `google-genai` | Gemini Flash client (§9) |
| `rich` | terminal telemetry |
| `pytest`, `ruff` | dev |

No pandas, no matplotlib. The UI is a self-contained HTML page (§15).

**Mesa rule:** `world/` must never import Mesa. Mesa appears only in `sim/model.py` as a ~60-line wrapper. Mesa 3 removed `RandomActivation`/`SimultaneousActivation`; the engine does its own simultaneous resolution (§6). Check the installed version before writing the wrapper.

```
pyproject.toml
CLAUDE.md
README.md
docs/  SPEC.md  PEAS.md  DECISIONS.md  PROGRESS.md  archive/
src/amongus/
  config.py          # §3 SimConfig
  rng.py             # seeded generator factory
  types.py           # §4 dataclasses and enums
  contracts.py       # §4 Planner, Deliberator, Policy
  world/
    map.py           # §5 rooms, edges, vents, tasks
    state.py         # WorldState + mutation helpers
    engine.py        # §6 tick pipeline
    observation.py   # §7 observation + note generation
    sabotage.py      # §12 lights / doors / reactor state machines
  search/
    astar.py         # §8.1 THE algorithm
    baselines.py     # §8.3 bfs, dijkstra, greedy (benchmark only)
    costs.py         # §8.2 risk-weighted edge cost
  agents/
    memory.py        # §9.1 private notes
    crewmate.py      # §11.1
    impostor.py      # §11.2
    protocol.py      # §12.3 renegotiation + §13 deadlock
  llm/
    client.py        # §9.3 Gemini call, cache, fallback
    deliberate.py    # §9.2 prompt build, JSON parse
  sim/
    model.py         # Mesa wrapper
    runner.py        # §16 CLI + shock injection
    scenarios.py     # §16 S1-S4
    telemetry.py     # §14 events.jsonl + rich terminal
  ui/
    viewer.html      # §15 self-contained replay viewer
scripts/  bench.py  run_demo.py
tests/
runs/                # gitignored
```

The swappable modules, for the scalability mark: `world`, `search`, `agents`, `llm`, `sim`, `ui`. Planner and deliberator are chosen by config string.

---

## §3 Configuration

`config.py` defines one frozen dataclass `SimConfig`. Scenarios override via `dataclasses.replace`.

| Field | Default | Meaning |
|---|---|---|
| `seed` | 0 | master seed |
| `n_players` / `n_impostors` | 8 / 2 | |
| `colors` | red, blue, green, pink, orange, yellow, black, white | agent ids, this order |
| `max_ticks` | 400 | draw if reached |
| `tasks_per_crewmate` | 4 | |
| `kill_cooldown` | 20 | ticks; reset after each meeting |
| `button_room` | cafeteria | |
| `p_miss` | 0.02 | per-occupant observation miss probability |
| `p_miss_lights` | 0.50 | crewmate miss probability during lights sabotage |
| `alpha_risk` | 2.0 | risk weight in A\* edge cost (§8.2) |
| `last_seen_decay` | 15 | ticks after which a sighting stops localising an agent |
| `theta_vote` | 0.35 | min suspicion to vote rather than skip |
| `theta_shadow` | 0.50 | min suspicion to start shadowing |
| `meeting_rounds` | 2 | |
| `notes_in_prompt` | 12 | most recent notes sent to the LLM |
| `ejection_reveals_role` | True | |
| `sabotage_cooldown` | 30 | shared by both impostors |
| `p_sabotage` | 0.08 | per-tick probability an eligible impostor sabotages |
| `reactor_timer` | 30 | ticks to fix or impostors win |
| `reactor_panels` | (reactor, o2) | |
| `panel_hold_ticks` | 2 | |
| `eta_grace` | 3 | ticks past ETA before revocation |
| `lights_fix_room` | electrical | |
| `doors_duration` | 10 | |
| `p_defect` | 0.5 | P(impostor defects on a won commitment) |
| `impostor_underbid` | 0.5 | bid multiplier when planning to defect |
| `deadlock_window` | 20 | ticks of zero task progress |
| `shadow_cooldown` | 15 | ticks shadowing is disabled after a stall |
| `commit_protocol` | True | ablation switch |
| `deadlock_protocol` | True | ablation switch |
| `planner` | astar | `astar` or `astar_no_risk` |
| `deliberator` | gemini | `gemini` or `template` |
| `llm_model` | gemini-2.0-flash | |
| `llm_cache` | runs/llm_cache.jsonl | |
| `verbosity` | 1 | 0 silent, 1 events, 2 events + suspicion, 3 debug |

---

## §4 Types and contracts

`types.py`:

```python
RoomId = str; AgentId = str

class Role(Enum): CREWMATE = "crewmate"; IMPOSTOR = "impostor"
class Phase(Enum): PLAY = "play"; MEETING = "meeting"; OVER = "over"

@dataclass
class Transit:
    src: RoomId; dst: RoomId; remaining: int; via_vent: bool = False

@dataclass
class TaskInstance:
    task_id: str; room: RoomId; duration: int; progress: int = 0
    @property
    def done(self) -> bool: return self.progress >= self.duration

@dataclass
class AgentPhys:
    id: AgentId; role: Role; alive: bool
    room: RoomId | None            # None while in transit
    transit: Transit | None
    tasks: list[TaskInstance]
    kill_cooldown: int; button_used: bool

@dataclass
class Body:
    victim: AgentId; room: RoomId; tick: int; reported: bool = False

@dataclass
class WorldState:
    tick: int; phase: Phase
    agents: dict[AgentId, AgentPhys]
    bodies: list[Body]
    sabotage: "SabotageState | None"
    closed_edges: set[frozenset[RoomId]]
    task_bar: float                # public, 0..1
    last_progress_tick: int
    meeting_count: int
    winner: Role | None

# Actions — exactly one per agent per tick
@dataclass
class Move: to: RoomId
@dataclass
class Vent: to: RoomId
@dataclass
class DoTask: task_id: str
@dataclass
class HoldPanel: room: RoomId
@dataclass
class Kill: target: AgentId
@dataclass
class Report: pass
@dataclass
class PressButton: pass
@dataclass
class Sabotage: kind: str; target: RoomId | None = None   # "lights"|"doors"|"reactor"
@dataclass
class Wait: pass
Action = Move | Vent | DoTask | HoldPanel | Kill | Report | PressButton | Sabotage | Wait

@dataclass
class Note:                        # §9.1 — one plain-English memory
    tick: int; text: str; kind: str   # "saw"|"kill"|"body"|"alone"|"missed_commit"|"meeting"

@dataclass
class Event:
    tick: int; type: str; data: dict
```

`contracts.py` — three protocols, that's all:

```python
class Planner(Protocol):
    """Routes an agent through the ship."""
    def path(self, view: "GraphView", src: RoomId, dst: RoomId,
             cost_fn: CostFn) -> tuple[list[RoomId], float, "SearchStats"]: ...

class Deliberator(Protocol):
    """Turns an agent's notes into a meeting statement, suspicion ranking and vote."""
    def speak(self, ctx: "MeetingContext") -> "Statement": ...

class Policy(Protocol):
    """Chooses one action per tick from an Observation."""
    def decide(self, obs: "Observation") -> Action: ...
```

```python
@dataclass
class Statement:
    speaker: AgentId
    text: str                        # <= 2 sentences, shown in the UI
    suspicion: dict[AgentId, float]  # 0..1 per other alive agent
    vote: AgentId | None             # None = skip

@dataclass
class SearchStats:
    expanded: int; frontier_max: int
```

`GraphView` is a read-only map view with `closed_edges` applied and an `allow_vents` flag.

---

## §5 Map

Coordinates are used for drawing (§15) and the A\* heuristic (§8.1).

```python
ROOMS: dict[str, tuple[float, float]] = {
    "upper_engine": (1.0, 5.0), "reactor": (0.0, 3.0), "security": (2.0, 3.0),
    "lower_engine": (1.0, 1.0), "medbay": (4.0, 4.0), "cafeteria": (6.0, 5.0),
    "weapons": (9.0, 5.0), "o2": (8.0, 4.0), "navigation": (11.0, 3.0),
    "shields": (9.0, 1.0), "communications": (7.0, 0.0), "storage": (6.0, 1.0),
    "admin": (7.0, 2.5), "electrical": (4.0, 1.5),
}

EDGES: list[tuple[str, str, int]] = [   # undirected, weight = travel ticks
    ("cafeteria", "weapons", 3), ("cafeteria", "medbay", 3), ("cafeteria", "upper_engine", 5),
    ("cafeteria", "storage", 4), ("cafeteria", "admin", 4), ("weapons", "o2", 2),
    ("weapons", "navigation", 4), ("o2", "navigation", 3), ("o2", "shields", 4),
    ("navigation", "shields", 4), ("shields", "communications", 2), ("shields", "storage", 4),
    ("communications", "storage", 3), ("storage", "admin", 2), ("storage", "electrical", 4),
    ("storage", "lower_engine", 6), ("electrical", "lower_engine", 5), ("lower_engine", "security", 3),
    ("lower_engine", "reactor", 3), ("lower_engine", "upper_engine", 5), ("security", "reactor", 2),
    ("security", "upper_engine", 3), ("reactor", "upper_engine", 3), ("upper_engine", "medbay", 4),
]  # 24 edges

VENT_PAIRS: list[tuple[str, str]] = [   # impostor only, 1 tick, bidirectional at load
    ("reactor", "upper_engine"), ("reactor", "lower_engine"), ("electrical", "medbay"),
    ("electrical", "security"), ("medbay", "security"), ("cafeteria", "admin"),
    ("navigation", "weapons"), ("navigation", "shields"),
]

TASK_POOL: list[tuple[str, str, int]] = [   # (task_id, room, duration)
    ("t01", "upper_engine", 4), ("t02", "lower_engine", 4), ("t03", "reactor", 5),
    ("t04", "security", 3), ("t05", "medbay", 5), ("t06", "cafeteria", 3),
    ("t07", "weapons", 4), ("t08", "o2", 3), ("t09", "navigation", 4),
    ("t10", "navigation", 3), ("t11", "shields", 4), ("t12", "communications", 3),
    ("t13", "storage", 4), ("t14", "admin", 3), ("t15", "electrical", 4),
    ("t16", "electrical", 5),
]
```

Validated at load (tested): 14 rooms, 24 edges, connected, all weights ≥ 1, vent and task rooms exist. The map is static; dynamic state lives in `WorldState`.

---

## §6 Engine

### 6.1 Setup
1. `rng = make_rng(config.seed)`.
2. Pick `n_impostors` ids uniformly without replacement (a scenario may override).
3. Each crewmate draws `tasks_per_crewmate` distinct tasks from the pool. Impostors get the same count as fake tasks, never counted in `task_bar`.
4. All agents spawn in `cafeteria` at tick 0.
5. Kill cooldowns set to `kill_cooldown`.

### 6.2 Tick pipeline (phase PLAY), in this exact order

1. **Decide.** For each alive agent in fixed id order, `action = policy.decide(obs_t)`, or a scripted action from the scenario. Decisions are simultaneous — no agent sees another's action this tick.
2. **Sabotage.** Valid `Sabotage` actions start one (§12). At most one active; on a tie the first agent in id order wins.
3. **Movement.** Agents in transit decrement `remaining` and arrive at 0. `Move(to)` along an open edge starts a transit with `remaining = weight`. `Vent(to)` arrives this tick. **Agents in transit are in no room: they cannot be seen, killed, or act.**
4. **Kills.** `Kill(target)` is valid if the killer is an alive impostor with cooldown 0, both are in the same room after movement, and the target is an alive crewmate. On a duplicate target the first killer in id order resolves. The victim dies, a `Body` is created, the cooldown resets. Other alive agents in the room are witnesses, each seeing it with probability `1 − p_miss_eff` (§7).
5. **Work.** `DoTask` advances progress if the agent is in the task room. `HoldPanel` advances the panel counter. Crewmate progress updates `task_bar` and `last_progress_tick`.
6. **Reports and button.** `Report` needs an unreported body in the room; `PressButton` needs `button_room` and `not button_used`. Either flags a meeting. **No meetings while a reactor sabotage is active** — the flag queues and fires the tick the reactor is fixed.
7. **Timers.** Decrement cooldowns, the sabotage timer, and door timers (reopen at 0). Run the deadlock detector (§13).
8. **Win check.** Crew wins if all real tasks are done or all impostors are ejected. Impostors win if alive impostors ≥ alive crewmates, or the reactor timer hits 0. Draw at `max_ticks`.
9. **Observe.** Build `obs_{t+1}` for each alive agent (§7) and append notes.
10. **Meeting.** If flagged, run §10. Then respawn all alive agents in `cafeteria`, cancel transits, reset kill cooldowns, clear lights and doors, mark all bodies reported.

Every state change emits an `Event` (§14).

---

## §7 Observation and notes

An `Observation` for agent `i` holds:

**Local** (only if in a room): `room`, `tick`, own `AgentPhys`; `occupants` — each other alive agent in the room included independently with probability `1 − p_miss_eff`; `bodies_here` — visible always, or with probability 0.7 during lights; `local_events` — kills and vent exits in this room this tick, each seen with probability `1 − p_miss_eff`.

`p_miss_eff = p_miss_lights` for crewmates during a lights sabotage, else `p_miss`. **Impostors are unaffected by lights** — that asymmetry is the point of the sabotage.

**Global** (every alive agent): `task_bar`; sabotage alarm (kind, panels, timer, panel status — **never the saboteur's identity**); closed doors; radio messages this tick; `PROGRESS_STALL`; meeting transcripts, votes and ejections; death notices only for *reported* bodies.

**Impostor extra:** partner identity and partner's room.

Agents never receive `WorldState`. A test enforces this.

### 7.1 Note generation

`observation.py` turns each observation into at most a few plain-English `Note`s via fixed templates. These notes are the *only* thing the LLM ever sees about the past, which keeps the prompt honest and small.

```
kind=saw    "t=41 I was in electrical with red and blue."
kind=alone  "t=44 I was alone in navigation."
kind=kill   "t=43 I saw blue kill red in electrical."
kind=vent   "t=45 I saw blue climb out of a vent in medbay."
kind=body   "t=47 I found red's body in electrical."
kind=missed_commit "t=130 black committed to fix o2 by t=126 and never did."
kind=meeting "Meeting 1: blue was ejected and was an impostor."
```

Notes are append-only per agent. Impostors get notes for their own kills too, so they know what they must explain away.

---

## §8 Search

### 8.1 A\* (`search/astar.py`) — the algorithm to defend

```
astar(view, src, dst, cost_fn) -> (path, cost, stats)
  open = min-heap of (f, tie, room); g[src] = 0; f = h(src)
  while open:
      pop (f, _, u); if u closed: continue
      close u; stats.expanded += 1
      if u == dst: reconstruct path and return
      for (v, w) in view.neighbors(u):        # respects closed_edges, allow_vents
          g2 = g[u] + cost_fn(u, v, w)
          if g2 < g.get(v, inf):
              g[v] = g2; parent[v] = u; push (g2 + h(v), next_tie, v)
  return ([], inf, stats)
```

Ties break on a monotonically increasing counter, then room name, so runs are reproducible.

**Heuristic and its admissibility proof.** Let `s = max over edges (u,v) of euclid(u,v) / weight(u,v)` — the fastest straight-line distance any single edge covers per tick, computed once at load. Then

```
h(u) = euclid(u, dst) / s
```

No path can cover ground faster than `s` per tick, so the true remaining cost is at least `euclid(u, dst) / s`. Therefore `h` never overestimates: it is admissible, and A\* returns an optimal path. `h` is also consistent, so no node needs reopening.

Note the heuristic ignores the risk term, which only ever *adds* cost — so `h` stays admissible under risk-weighted costs too. Say this before you are asked.

`all_pairs(view, cost_fn)` runs A\* from every source and returns a distance matrix plus next-hop table, rebuilt on each replan.

### 8.2 Risk-weighted cost (`search/costs.py`)

```
cost(u, v, w) = w + alpha_risk * risk_i(v)

risk_i(v) = sum over agents s of  suspicion_i(s) * seen_recently_i(s, v)
```

`suspicion_i(s)` is agent `i`'s latest LLM suspicion score (§9); `seen_recently_i(s, v)` is 1 if `i` saw `s` in `v` within `last_seen_decay` ticks, else 0. Costs stay non-negative so A\* remains correct. With `planner = astar_no_risk`, `alpha_risk = 0` — that is the ablation.

This is the one place where social reasoning feeds spatial reasoning, and it is the best single thing to point at in the demo.

### 8.3 Baselines (`search/baselines.py`) — benchmark only, never used by agents

- `bfs(view, src, dst)` — minimises hop count, so its tick cost is ≥ A\*'s.
- `dijkstra(view, src, dst, cost_fn)` — same optimal cost as A\*, more expansions (no heuristic).
- `greedy(view, src, dst)` — expands fewest, not optimal.

### 8.4 Replanning triggers (`agents/memory.py`)

Recompute the route only when one fires: `arrival`, `suspicion_shift` (L1 change > 0.1), `topology` (an edge closed or reopened), `new_goal` (sabotage alarm, body seen, commitment won or revoked), `meeting_end`, `blocked`. Each emits a `REPLAN` event with the reason and `expanded` count.

### 8.5 Bench (`scripts/bench.py`)

Runs all 182 ordered room pairs through all four algorithms and writes `runs/bench/search.csv` plus a Markdown table to stdout: algorithm, mean path cost, mean expansions, optimal yes/no. **This table is P1 slide material** — it must be generated from real runs, not typed by hand.

---

## §9 LLM deliberation

### 9.1 Memory (`agents/memory.py`)
`AgentMemory` holds the append-only `list[Note]` (§7.1), the last suspicion dict, and current plan state. `recent(n)` returns the last `n` notes for the prompt.

### 9.2 Prompt and response (`llm/deliberate.py`)

One call per agent per round. The prompt is assembled from four fixed blocks — keep it short and auditable:

1. **Rules:** the game, the room list, who is alive, that agents only see their own room.
2. **Identity:** `You are <color>.` For an impostor, add: `You are an impostor. Your partner is <color>. You killed <who, where, when>. You must not be caught. You may lie, but your story must fit what others can check.`
3. **Notes:** the agent's own `notes_in_prompt` most recent notes verbatim.
4. **Transcript:** every statement already made this meeting, in order.

Required response — strict JSON, requested via the model's JSON mode:

```json
{
  "statement": "one or two sentences you say out loud",
  "suspicion": {"blue": 0.7, "pink": 0.1},
  "vote": "blue"
}
```

`vote` may be `null` to skip. Parsing: strip code fences, `json.loads`, validate keys, clamp suspicion to `[0,1]`, drop unknown or dead agent ids. On failure retry once; on a second failure use the template fallback (§9.4) and log `LLM_PARSE_FAIL`. **A malformed response must never crash a run.**

### 9.3 Client and cache (`llm/client.py`)

- `GEMINI_API_KEY` read from the environment. Never stored in the repo.
- Cache key `sha256(model + "\n" + prompt)`; store `{key, response}` lines in `llm_cache.jsonl`. A cache hit skips the network entirely.
- The 8 per-round calls go out concurrently (`asyncio` or a thread pool) so a round costs about one call's latency.
- One retry on a transport error, then fall back. Never raise into the engine.

Consequence worth stating in the demo: **the world is seeded and the LLM is cached, so a recorded run replays byte-identically.** Nondeterminism is confined to the first uncached run.

### 9.4 Template fallback (`deliberator = template`)

A pure-Python `Deliberator` with no network, used for tests, offline demos, and any parse failure:

- statement = the agent's highest-priority note rendered as speech (`kill` > `body` > `saw` > `alone`).
- suspicion: `0.9` for an agent it saw kill or vent; `0.5` for an agent it saw near a body within 10 ticks; `0.1` otherwise. An impostor instead assigns `0.6` to the reporter and `0.0` to its partner.
- vote = argmax suspicion if ≥ `theta_vote`, else `None`.

This is also the honest ablation: **template vs gemini** is the experiment that shows what the LLM actually buys you.

---

## §10 Meetings and voting (`agents/protocol.py`)

1. Triggered by `Report` (reporter speaks first) or `PressButton` (presser first). Remaining speakers follow in an RNG-shuffled order.
2. **Round 1:** each alive agent produces one `Statement` via its `Deliberator`. All statements are appended to the public transcript.
3. **Round 2:** same, with round 1 visible, so agents can contradict or defend.
4. **Vote:** each agent's round-2 `vote`. A crewmate skips if its top suspicion is below `theta_vote`. An impostor never votes for its partner.
5. **Resolution:** plurality. A tie, or skips ≥ the top count, ejects nobody. If someone is ejected and `ejection_reveals_role`, the role is announced and becomes a `meeting` note for everyone.
6. Each agent's final suspicion dict is stored for the A\* risk term (§8.2).
7. Log every statement, every suspicion snapshot, votes and the outcome (§14).

The meeting is the only point where the world clock pauses.

---

## §11 Agent policies

### 11.1 Crewmate (`agents/crewmate.py`) — strict priority order
1. A body is in the room, or it witnessed a kill this tick → `Report`.
2. Committed to a reactor panel (§12.3) → A\* to it (no risk term), then `HoldPanel`.
3. Lights active and within 6 ticks of `electrical` → go hold that panel.
4. Shadowing enabled and a co-located agent has suspicion ≥ `theta_shadow` → **shadow** it: move toward its last seen room, or `Wait` if already together.
5. Tasks remain → visit task rooms **nearest-first** by A\* distance, `DoTask` on arrival.
6. Tasks done → move to the adjacent room with the lowest risk and keep watching.

Nearest-first is a deliberate choice over exact TSP ordering: with 4 task rooms the optimal tour saves a handful of ticks and costs a subset-DP recurrence to defend. Be ready to say that Held–Karp would give the optimal order in `O(k²·2^k)` and that we judged it not worth the complexity here.

### 11.2 Impostor (`agents/impostor.py`) — strict priority order
1. **Kill:** cooldown 0, a crewmate in the room, and no other non-partner alive occupant observed → `Kill`. Next tick, `Vent` out if the room has a vent, else walk to the lowest-occupancy neighbour.
2. **Sabotage:** off cooldown, ≥3 crewmates alive, probability `p_sabotage` per tick → pick `lights` (0.4), `reactor` (0.4), `doors` (0.2, targeting the room of the nearest crewmate).
3. **Defect:** reactor active and it won a commitment → with probability `p_defect` stall (`Wait` or walk away); otherwise fix honestly. When it intends to defect it multiplies its bid by `impostor_underbid` to win the assignment.
4. **Hunt:** walk toward the nearest crewmate that is currently alone as far as it knows, preferring rooms with vents.
5. **Blend:** walk to a fake task room and stand there for the duration.

Impostors never self-report a body.

---

## §12 Sabotage

`world/sabotage.py` holds the state machines; `agents/protocol.py` holds the reactor negotiation.

### 12.1 Lights — *the "sensor failure" shock*
Active until an agent holds the `electrical` panel for `panel_hold_ticks`. Effect: crewmate `p_miss_eff = p_miss_lights`, bodies visible with probability 0.7. Impostors unaffected.

### 12.2 Doors — *the "grid barrier" shock*
Closes every corridor incident to the target room for `doors_duration` ticks. Vents are unaffected. Agents already in transit continue. Fires a `topology` replan for every agent, which is what makes A\* visibly reroute in the demo.

### 12.3 Reactor and the renegotiation protocol — the headline

Timer `reactor_timer`, two panels at `reactor_panels`, each done once an agent holds it for `panel_hold_ticks` (not necessarily at the same time). Both done ⇒ fixed. Timer 0 ⇒ impostors win. There is **no central coordinator**: every agent runs the same deterministic function over the same public messages.

1. **Bid** (tick `t`): every alive agent broadcasts `BID` with its A\* travel cost to each panel. A defecting impostor underbids.
2. **Commit** (tick `t+1`): every agent computes the same assignment — the pair `(a, b)`, `a ≠ b`, minimising `max(cost_a[P1], cost_b[P2])`, ties broken by agent id. The two winners broadcast `COMMIT(panel, eta = t + 1 + cost)`.
3. **Revoke:** panel status is public. If `tick > eta + eta_grace` and the panel is not done, the commitment is revoked (`REVOKE` event), a `missed_commit` note is recorded against that agent **by every agent**, and the next-best unrevoked bidder commits with a fresh ETA.
4. The `missed_commit` note goes straight into the next meeting's prompt, so the defector has to explain itself in natural language. That is the whole payoff: a broken commitment becomes social evidence.

With `commit_protocol = False` every agent just rushes the nearest panel — the ablation that shows the protocol matters. Honest agents can miss too (doors, long paths), which is why a revocation is evidence rather than proof.

---

## §13 Deadlock

**The deadlock:** two crewmates who each suspect the other both choose to shadow (§11.1 step 4). Each follows the other, neither does tasks, and if they hold the remaining tasks global progress stops.

**Detection** (engine, §6.2 step 7): if `tick − last_progress_tick ≥ deadlock_window` with no active sabotage and no meeting in that window, emit a global `PROGRESS_STALL`. This is legitimate for agents to act on because `task_bar` is public.

**Resolution** (agents): on `PROGRESS_STALL` every crewmate stops shadowing, sets `shadow_disabled_until = tick + shadow_cooldown`, and replans its task route. Emit `DEADLOCK_BROKEN` when progress resumes.

With `deadlock_protocol = False` the event is never emitted and the stall persists — the ablation.

---

## §14 Telemetry

### 14.1 Event log
Each run writes `runs/<scenario>_<seed>/events.jsonl`, one JSON object per line: `{"t": int, "type": str, ...}`. Required types:

`SPAWN, ROLES, MOVE, ARRIVE, VENT, KILL, BODY_SEEN, REPORT, BUTTON, MEETING_START, STATEMENT, SUSPICION, VOTE, EJECT, SABOTAGE, BID, COMMIT, REVOKE, PANEL_DONE, SABOTAGE_FIXED, DOORS_OPEN, REPLAN, PROGRESS_STALL, DEADLOCK_BROKEN, LLM_PARSE_FAIL, SHOCK, GAME_OVER`

`ROLES` is written for analysis and the UI's reveal toggle only; no agent ever reads it. `SUSPICION` records `{agent, scores, round}`. The log must be complete enough for §15 to replay a whole game without the simulator.

### 14.2 Terminal (verbosity 1–2, `rich` if available)

```
[t=041] MOVE     blue    storage -> electrical (eta 4)
[t=043] KILL     blue x red @electrical   witnesses=[]
[t=047] REPORT   green found red @electrical -> MEETING #1
[M1 r1] green    "I found red's body in electrical. Blue was walking out as I came in."
[M1 r1] blue     "I was in navigation with pink the whole time."
[M1 r2] pink     "I was never in navigation. Blue is lying."
[M1 sus] green   blue .80 | black .20 | white .10
[M1 vote] blue:4 green:1 skip:1 -> EJECT blue (IMPOSTOR)
[t=120] SABOTAGE reactor panels=[reactor,o2] timer=30
[t=121] COMMIT   yellow->reactor eta=129 | black->o2 eta=126
[t=130] REVOKE   black o2 (eta 126 +3) -> BACKUP white eta=135
[t=133] REPLAN   green reason=topology expanded=9
```

### 14.3 Reproducibility
Hashing `events.jsonl` twice for the same seed with a warm LLM cache gives the same digest. This is a test.

---

## §15 UI — replay viewer (`src/amongus/ui/viewer.html`)

**One self-contained HTML file.** No build step, no CDN, no framework: inline CSS and vanilla JS, `<canvas>` for the map. Opening it in a browser and dropping an `events.jsonl` on it must Just Work, because a demo that depends on a running server is a demo that can fail live.

Layout:
- **Map canvas** — the 14 rooms at their §5 coordinates, corridors as lines with their weights, closed doors dashed red, vents as dotted links. Agents are coloured dots offset around their room; bodies are `✕`; a room under sabotage pulses.
- **Header** — tick, task bar, alive count, active sabotage and its timer.
- **Transport** — play / pause / step / scrub slider / speed 1× 2× 4×. Scrubbing rebuilds state from the log up to that tick, so seeking is exact.
- **Event feed** — the same lines as §14.2, auto-scrolling, current tick highlighted.
- **Meeting overlay** — when a `MEETING_START` is reached, dim the map and show the transcript as chat bubbles in speaking order, then the suspicion bars, then the vote tally and result. This panel is the part that sells the project; give it the most polish.
- **Path overlay** — on `REPLAN`, briefly draw the chosen route so A\* is visible.
- **Reveal toggle** — off by default; on, it outlines the true impostors from `ROLES` so a viewer can see who was lying.

Load either by drag-and-drop or `?log=<relative path>`. Theme both light and dark. Degrade gracefully on an unknown event type (ignore it, do not throw).

---

## §16 Scenarios and shocks (`sim/scenarios.py`, `sim/runner.py`)

```python
@dataclass
class Scenario:
    name: str; seed: int
    config_overrides: dict = field(default_factory=dict)
    roles: tuple[AgentId, AgentId] | None = None
    spawns: dict[AgentId, RoomId] | None = None
    scripted: dict[int, dict[AgentId, Action]] = field(default_factory=dict)
    sabotage_schedule: dict[int, Sabotage] = field(default_factory=dict)
```

A scripted action replaces that agent's decision for that tick only; everything else runs normally.

| Id | Name | Setup | Acceptance, visible in the log |
|---|---|---|---|
| S1 | `baseline` | seed 7, defaults | Runs to a winner; ≥1 meeting; statements and suspicion logged |
| S2 | `worked_example` | roles (blue, black); script blue and red alone into electrical ~t=41, blue kills t=43, green enters and reports t=47 | Green's statement names blue; blue gives an alibi naming pink; pink contradicts it in round 2; blue ejected. **These exact lines go on the slide** |
| S3 | `reactor_defect` | reactor fires while black (impostor) is nearest a panel; `p_defect=1` for black | `COMMIT` black → `REVOKE` at eta+3 → backup `COMMIT` → `SABOTAGE_FIXED`; black's suspicion rises at the next meeting |
| S4 | `standoff` | two crewmates hold the last tasks and are scripted into mutual suspicion | `PROGRESS_STALL` at ~20 ticks, then `DEADLOCK_BROKEN`, tasks complete. With `deadlock_protocol=False` the stall persists to `max_ticks` |

### 16.1 Live shock injection — the Exceeds criterion

`runner.py` reads stdin without blocking the tick loop. Single keypresses inject a shock at the current tick, logged as `SHOCK`:

| Key | Shock | Rubric wording it satisfies |
|---|---|---|
| `l` | lights sabotage | "sudden sensor failure" |
| `d` | doors on a random occupied room | "grid barrier additions" |
| `r` | reactor sabotage | forces live renegotiation |
| `k` | clear both impostors' kill cooldowns | stress |
| `b` | force an emergency meeting | stress |

Also available non-interactively as `--shock 60:lights --shock 120:doors:storage` for a scripted run, and that path is what the tests use.

CLI: `python -m amongus.sim.runner --scenario worked_example --seed 3 --verbosity 2`

---

## §17 Tests (`tests/`, pytest)

Each implementer writes its own in the same milestone. Minimum set:

- **map:** 14 rooms, 24 edges, connected, vent and task rooms valid.
- **astar:** cost equals `networkx.dijkstra_path_length` on all 182 pairs; **heuristic admissible on all pairs** (`h(u) ≤ true_cost(u, dst)`); expansions ≤ Dijkstra's; deterministic tie-breaking; respects closed edges; vents only when `allow_vents`.
- **baselines:** BFS tick-cost ≥ A\* cost on all pairs; Dijkstra cost equals A\* cost.
- **costs:** risk term is non-negative; `alpha_risk=0` reproduces plain distances.
- **engine:** event-log hash identical across two runs of the same seed; transit agents invisible and unkillable; `Vent` rejected for crewmates; kill cooldown enforced; no meeting while reactor is active; reactor timeout ends the game; doors reopen on time.
- **observation:** a policy only ever receives an `Observation`; lights raise the crewmate miss rate and not the impostor's.
- **llm:** malformed, fenced, and truncated JSON all fall back without raising; a cache hit returns the stored response and makes no network call; `deliberator=template` needs no key.
- **protocol:** every agent computes the same assignment from the same bids; revocation happens exactly at `eta + eta_grace`; the backup commits; a `missed_commit` note is recorded by all agents.
- **deadlock:** S4 stalls with `deadlock_protocol=False` and recovers with it True.
- **shocks:** each `--shock` kind applies at the right tick and the run still completes.

Every test runs offline with `deliberator=template` or a warm cache. **No test may require an API key.**

---

## §18 Milestones

| M | Owner | Deliverables | Acceptance |
|---|---|---|---|
| **M0** | orchestrator | This spec, CLAUDE.md, agent definitions, DECISIONS entry | done |
| **M1** | search-engineer ∥ world-engineer | **1a:** `astar.py`, `costs.py`, `baselines.py`, `scripts/bench.py`. **1b:** `pyproject.toml`, config, rng, types, contracts, map, state, engine skeleton, observation + notes | A\* and map tests green; `bench.py` prints the comparison table; `runner --seed 1 --ticks 20` deterministic |
| **M2** | world-engineer → agent-engineer | Kills, bodies, reports, button, win checks; crewmate and impostor policies with the `template` deliberator | 20 seeds run to a winner with no errors, fully offline |
| **M3** | agent-engineer | `llm/client.py`, `llm/deliberate.py`, meetings, voting, suspicion → risk cost | S2 produces a coherent transcript; llm tests green; a warm-cache replay is byte-identical |
| **M4** | world-engineer (state) → agent-engineer (protocol) | Lights, doors, reactor; renegotiation; deadlock; shock injection | S3 and S4 acceptance met; all five shock keys work mid-run |
| **M5** | ui-engineer ∥ telemetry-engineer | `viewer.html`; scenarios S1–S4; `run_demo.py`; `docs/PEAS.md`; README | All scenarios run from the CLI; the viewer replays each log; demo runs start to finish |

Parallel only where files do not overlap: M1a ∥ M1b, and M5's UI ∥ telemetry. Everything else is sequential.

`docs/PEAS.md` is P1 slide material — PEAS table, environment classification, the state-space count (`14^8 ≈ 1.5×10^9` position states × `C(8,2) = 28` role assignments × task progress), the communication bottleneck, and the §8.5 search comparison table. **Pull it forward to M1 if Presentation 1 is due first.**

---

## §19 Out of scope

Pygame, a live web server, continuous movement or collision physics, multi-agent pathfinding (CBS etc.), reinforcement learning or any training, ghosts, cameras/vitals, multiple maps, exact Bayesian belief, Held–Karp, time-expanded planning, pandas, a database, LLM calls inside the tick loop. An agent that thinks one of these is needed stops and asks the orchestrator.
