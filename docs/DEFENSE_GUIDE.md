# AI Among Us — Defense Guide

**Purpose of this document: prepare *you* to be cross-questioned, not to make slides.**
For slide content, numbers, and PEAS tables, use `docs/PROJECT_REFERENCE.md`. This document goes one layer deeper: for every mechanism, it teaches the *concept* properly (as if from a textbook), then shows you the *exact code* that implements it (file, line numbers, real snippets — verified against the current codebase as of writing this), then drills you with the follow-up questions a sharp examiner asks next.

**How to use it:** don't read it once. Read one section, close the laptop, explain it out loud to an empty room in your own words, then open the code file it points to and find the line yourself without looking at this doc. If you can do that for every section below, you cannot be cornered.

Every code snippet here is copied verbatim from the actual files — if `sir` asks you to open the file and it doesn't match, tell me, because something has drifted and needs fixing before you present.

---

## Table of contents

1. [The one-sentence architecture, and why you must lead with it](#1)
2. [PEAS and environment classification — the theory you're actually being examined on](#2)
3. [Search — A\* in full depth, the part you said you're worried about](#3)
4. [The tick engine — simultaneous multi-agent resolution](#4)
5. [Partial observability — the Observation model](#5)
6. [The LLM deliberation layer — and why not Bayes](#6)
7. [The reactor renegotiation protocol — distributed consensus without a coordinator](#7)
8. [Deadlock — an emergent multi-agent failure mode](#8)
9. [How determinism and reproducibility actually work](#9)
10. [The testing philosophy: what code review misses, that measurement catches](#10)
11. [Rapid-fire cross-questioning drill](#11)
12. [If you get asked to run something live](#12)

---

<a name="1"></a>
## 1. The one-sentence architecture, and why you must lead with it

> **Search handles space. An LLM handles people.**

Concretely: every tick, all 8 agents move, do tasks, kill, and sabotage using **pure algorithmic logic** — a fixed priority list per role, executed with A\* for pathing. Zero network calls happen in this loop. Only when a meeting is triggered (someone reports a body, or presses the emergency button) does the simulation pause and ask a language model, once per agent per round, to produce a spoken statement, a suspicion ranking over the other players, and a vote.

**Why this split, and not one AI end-to-end?** Because the two problems are fundamentally different in kind:

- *Where should I walk to minimize risk and finish my tasks* is a **well-defined optimization problem** over a known, static graph. It has an exact, provably optimal algorithm (A\*). Using an LLM for this would be strictly worse — slower, non-deterministic, and unable to guarantee optimality.
- *Should I believe what blue just said* is a problem of **natural-language reasoning under deception**, where the space of possible things an agent could say is unbounded. There is no exact algorithm for this — you cannot write a formula that scores the plausibility of an arbitrary English sentence. This is exactly the class of problem LLMs are suited to and classical AI (search, logic, exact inference) is not.

So the architecture is a deliberate **decomposition by problem type**, not an arbitrary implementation choice. State that framing before anything else — it pre-empts "why didn't you just use an LLM for everything?" and "why didn't you just hand-code the social reasoning?" in one move.

**The single mechanism that ties the two halves together**, and the best thing to point at if asked "so does the social reasoning actually *affect* anything, or is it decorative?":

> Suspicion produced at a meeting is stored per-agent, and it becomes an extra cost term on the A\* graph (`alpha_risk × suspicion`) for the rest of the game. A crewmate who now distrusts `blue`, last seen in `electrical`, will pay more to walk through `electrical` — so it detours. **Belief changes geometry.** Code: `src/amongus/search/costs.py:31-34`, consumed by `src/amongus/agents/crewmate.py:214-218` (`_cost_fn`).

---

<a name="2"></a>
## 2. PEAS and environment classification — the theory you're actually being examined on

Full tables live in `PROJECT_REFERENCE.md` §5–§7. Here is the conceptual grounding underneath them, because Presentation 1 marks are about whether you understand *why* something is classified the way it is, not whether you can recite the classification.

### PEAS — what each letter actually means

- **Performance measure** — how you *judge* success, from the outside. Not "what the agent does," but "how well did it do it." For crewmates: win rate, ejection precision (of the people we ejected, how many were actually impostors), ejection recall (of the actual impostors, how many did we catch). Precision and recall matter separately because a crew that ejects everyone eventually gets 100% recall but terrible precision (they ejected innocents too) — a naive "did they win" metric hides that distinction.
- **Environment** — everything the agent doesn't fully control but must act within: the map, other agents (whose intentions are hidden), the task bar, active sabotage, closed doors.
- **Actuators** — the *outputs* of the agent: the things it can *do* to change the world. `Move`, `DoTask`, `Kill`, `Sabotage`, `Report`, `PressButton`.
- **Sensors** — the *inputs*: what the agent can *perceive*. Its own room's occupants (probabilistically), bodies, task bar, sabotage alarms.

**The classic trap, and why it's a trap:** students often list `Report` as a sensor because "the agent reports what it sees." Wrong — *seeing the body* is the sensor event (it arrives via `Observation.bodies_here`); *choosing to announce it via the `Report` action* is the actuator. The distinction is "what comes in" vs. "what goes out." If sir asks you to classify something on the spot, ask yourself: does this change the external world, or does it inform the agent about the external world? First = actuator. Second = sensor.

**Code proof that this distinction is real, not cosmetic:** `src/amongus/types.py` defines `Observation` (sensor data — read-only, delivered *to* a policy) as a completely separate type from the `Action` union (`Move | DoTask | Kill | Report | PressButton | Sabotage | ...` — actuator commands, returned *by* a policy). A `Policy.decide(obs: Observation) -> Action` (see `contracts.py:56-61`) can only ever consume the former and produce the latter — the type system itself enforces the PEAS boundary.

### Environment classification — define each term precisely, then justify

Don't just say "partially observable, stochastic, sequential, dynamic, discrete, multi-agent." Sir will ask you to *define* at least one of these and justify it. Here's the actual AI-textbook definition of each, paired with our justification:

| Property | Textbook definition | Our justification |
|---|---|---|
| **Partially observable** | An agent's percept does not give it the complete state of the environment. | An agent's `Observation` contains only its own room's contents, probabilistically. It cannot see other rooms, cannot see roles, cannot see unreported bodies. `WorldState` (the true, complete state) is never passed to any policy — see `contracts.py`'s `Policy.decide(obs: Observation)` signature; there is no path from `Observation` back to `WorldState`. |
| **Stochastic** | The next state is not a deterministic function of the current state and the agent's action — there's randomness. | Observation misses are probabilistic (`p_miss` per-occupant roll, `src/amongus/world/observation.py:88`), role assignment is randomized at game start, and every random draw goes through **one seeded `numpy.random.Generator`** (never Python's `random` module) — so the environment is stochastic *and* fully reproducible given a seed. |
| **Sequential** (vs. episodic) | The current decision can affect all future decisions — the agent must think ahead. | A kill at tick 43 causes a report at tick 47, which causes a meeting, which causes an ejection or not, which changes who's alive for the rest of the game. Nothing here is "one-shot." |
| **Dynamic** (vs. static) | The environment can change while the agent is deliberating. | The world advances every tick regardless of what any single agent does; `Wait` is a real, sometimes costly, choice — time doesn't pause for you to think. |
| **Discrete** | Finite, countable states, times, and percepts. | Integer ticks, 14 named rooms (not continuous coordinates), one action per agent per tick from a fixed action-type union. |
| **Multi-agent** | More than one agent whose actions affect each other's outcomes. | 8 agents, and — the subtle part — it isn't simply "cooperative" or "competitive," see below. |

### The three-layer multi-agent structure — this is worth a full minute of explanation, not a one-liner

Don't say "6 crewmates cooperate, 2 impostors compete." That's only two of three layers:

1. **Cooperative within the crew** — 6 crewmates share one win condition and (in principle) benefit from pooling information.
2. **Zero-sum across the two teams** — impostors win exactly when crewmates lose. Classic adversarial game.
3. **Cooperative-but-defectable within the impostor pair** — this is the one people miss. The two impostors share a goal and coordinate (a private partner channel, joint sabotage planning), *but* one impostor can **underbid in the reactor-repair auction to win a commitment it never intends to honor**, stalling the timer while its partner does the real work of killing. This is a coordination game *with* a defection incentive *inside* an otherwise cooperative relationship — game-theoretically distinct from both pure cooperation and pure zero-sum competition. It's why the renegotiation protocol (§7 below) exists at all: it's built specifically to *detect* this kind of defection.

If sir asks "is this cooperative or competitive," the strong answer names all three layers and explains why layer 3 is the interesting one.

### State-space size — the number, and the point it's actually making

- Joint position states: `14^8 ≈ 1.5 × 10^9` (8 agents, 14 rooms each, ignoring in-transit states which add more).
- Role assignments: `C(8,2) = 28` (choose 2 impostors from 8).
- Multiply in alive/dead subsets, task progress per agent, sabotage state, door closures: the full joint state space is **≳ 10³⁰**.

**The point is not the number — it's what you do about it.** No agent enumerates or searches this space. Instead:

- **Spatially**, every agent reduces its planning problem to the **14-node room graph** — searched exactly and cheaply with A\*.
- **Socially**, every agent reasons over a **small, bounded set of other agents' suspicion scores** in natural language, rather than a joint probability distribution over 28 role-assignment hypotheses.

So the actual argument for the algorithmic-modelling mark is: *we never search the joint space; we project it onto two much smaller, tractable sub-problems and solve each with the tool suited to it.* That sentence is worth memorizing verbatim.

---

<a name="3"></a>
## 3. Search — A\* in full depth, the part you said you're worried about

This is the one algorithm you must be able to derive on a whiteboard, prove correct, and defend against every "why not X" alternative, unprompted. Take your time on this section.

### 3.1 What problem is being solved

Every crewmate, every tick, needs to answer: *"given I'm in room `u`, and I want to get to room `dst`, what is the cheapest sequence of rooms to walk through, and what's the very next room I should step into?"* The map (`src/amongus/world/map.py:10-52`) is a weighted undirected graph: 14 rooms as nodes, 24 corridors as edges, each edge carrying an integer weight (2–6) representing travel time in ticks.

This is **single-source shortest path** — a classical, solved problem. The question is which algorithm to use, and that choice is graded.

### 3.2 The concept ladder — know where A\* sits and why each rung below it is worse

Work through these in order; each one motivates the next.

**Breadth-First Search (BFS).** Explores level by level, treats every edge as cost 1. Finds the path with the **fewest edges**, not the cheapest. On an *unweighted* graph, fewest edges = cheapest, so BFS is optimal there. Our graph is weighted (edges cost 2–6 ticks, not 1), so BFS optimizes the wrong quantity. **Concrete counterexample from our own map** (verified, not hypothetical): `storage → upper_engine` has two routes of equal hop-count (2 hops each) — via `cafeteria` costs 9 ticks, via `lower_engine` costs 11 ticks. BFS has no way to tell these apart; it might return either, and 17 of the 182 room-pairs in our map have a case where BFS's hop-minimal choice is *provably worse in ticks* than the truly cheapest path — worst case `o2↔security`, 13 ticks optimal vs. 17 via BFS.

**Uniform-Cost Search / Dijkstra.** Fixes BFS's flaw: instead of exploring level-by-level by hop count, it always expands the frontier node with the **lowest accumulated cost so far** (`g(n)`), using a priority queue (min-heap) instead of a plain queue. This *is* optimal on any non-negative-weight graph — it will always find the true cheapest path. But it is **uninformed**: it has no notion of "am I getting closer to the destination," so it expands outward in every direction equally, including straight away from the goal, until it happens to reach it. Wasteful.

**Greedy Best-First Search.** The opposite mistake: expands whichever frontier node the heuristic `h(n)` claims is closest to the goal, **completely ignoring** the cost already spent to get there (`g(n)`). Fast (fewest expansions, `3.34` mean on our map) but **not optimal** — it can walk into an expensive dead-end because it looked promising locally. Worst case on our map: `weapons → admin`, optimal cost 7, greedy finds a path costing 12.

**A\* Search.** The synthesis: expand the frontier node minimizing `f(n) = g(n) + h(n)` — accumulated cost so far, *plus* an estimate of remaining cost. This is Dijkstra with a "look-ahead" bonus. If the heuristic never overestimates (**admissible**, defined below), A\* is *guaranteed optimal*, exactly like Dijkstra, but it typically expands far fewer nodes because the heuristic prunes directions that are obviously wrong.

**The one-sentence summary you should give:** *"A\* is the only algorithm in this table that is simultaneously optimal and informed — Dijkstra is optimal but blind, greedy is informed but unsound, BFS is neither on a weighted graph."*

### 3.3 The algorithm itself, and the actual code

```
astar(view, src, dst, cost_fn) -> (path, cost, stats)
    open = min-heap of (f, tie_counter, room);  g[src] = 0;  f[src] = h(src, dst)
    while open is not empty:
        pop the (f, _, u) with smallest f
        if u already closed: skip (a cheaper route already finalized it)
        mark u closed;  count this as an expansion
        if u == dst: reconstruct the path via parent pointers and return it
        for each neighbor (v, w) of u:
            g2 = g[u] + cost_fn(u, v, w)
            if g2 < g.get(v, infinity):
                g[v] = g2;  parent[v] = u
                push (g2 + h(v, dst), next_tie_counter, v) onto open
    return (no path found, infinite cost)
```

That's `src/amongus/search/astar.py:34-65`, verbatim in structure. Read the real file if asked to show it — it's short enough to walk through line by line live. Two implementation details worth calling out proactively:

- **Deterministic tie-breaking.** The heap entries are `(f, tie_counter, room)` where `tie_counter` is a monotonically increasing integer (`itertools.count()`, line 39). Two entries with identical `f` break the tie by *insertion order*, never by Python's arbitrary object comparison or dict iteration order. This is why the same seed always produces the exact same path — a correctness requirement for the whole project's reproducibility story (§9 below).
- **Closed-set re-check** (`if u in closed: continue`, line 48–49): a room can be pushed onto the heap multiple times with different costs before it's ever popped; once popped and closed, later stale entries for it are silently skipped.

### 3.4 The heuristic, and the admissibility proof you must be able to say out loud

```python
def heuristic(u: RoomId, dst: RoomId) -> float:
    return _euclid(u, dst) / HEURISTIC_SCALE
```

`HEURISTIC_SCALE` (computed once, `map.py:132-134`) is:

```
s = max over all 24 edges (u, v) of  euclidean_distance(u, v) / weight(u, v)
```

— i.e., **the single fastest rate, in drawing-units per tick, that any one corridor in the whole map ever covers ground.** It's a constant of the map, computed once at import time.

**The proof, step by step — practice saying this exactly:**

1. `s` is, by construction, the maximum "speed" (distance per tick) achieved by any edge in the graph.
2. Therefore, **no path through the graph — however cleverly chosen — can cover straight-line distance faster than `s` per tick.** Any real path is a sequence of edges, each of which advances ground no faster than `s`/tick; concatenating slower-or-equal segments cannot beat the fastest possible rate.
3. So the *true* minimum number of ticks to get from `u` to `dst` is **at least** `euclidean_distance(u, dst) / s`.
4. Our heuristic `h(u) = euclidean_distance(u, dst) / s` is *exactly* that lower bound.
5. A heuristic that never overestimates the true remaining cost is called **admissible**, by definition.
6. **A\* with an admissible heuristic is guaranteed to find the optimal-cost path.** (This is a general theorem, not something we invented — the proof sketch: if A\* returned a suboptimal path, the truly optimal path's frontier node would still have had `f = g + h ≤ true optimal cost < the suboptimal cost found`, so A\* would have expanded it first — contradiction.)

**The two follow-up questions sir will almost certainly ask, and the pre-written answers:**

> *"Is it also consistent?"* — Yes. Consistency means `h(u) ≤ cost(u,v) + h(v)` for every edge, which is exactly the triangle inequality on Euclidean distance divided by a constant — a scaled Euclidean metric always satisfies the triangle inequality. Consistency's practical payoff: once a node is closed, it's *guaranteed* never worth reopening, which is why the algorithm never revisits a closed node (line 48–49) and that's safe.

> *"You add a risk penalty to the edge cost for crewmate routing — doesn't that break admissibility?"* — No, and this is the best answer in the whole document if you get this question, because it shows you understand *why*, not just *that*: `cost(u,v,w) = w + alpha_risk × risk(v)`, and `risk(v) ≥ 0` always (it's built from non-negative suspicion scores — see `costs.py:31-34`, `assert c >= 0.0`). The heuristic `h` is computed purely from the *raw edge weight* `w`, completely ignorant of the risk term. Since the risk term only ever **increases** the true cost above what `h` already accounts for, `h` remains a valid *lower bound* — an underestimate of a smaller number is still an underestimate of the (now larger) real number. Admissibility is preserved *by construction*, not by luck.

### 3.5 Measured comparison — real numbers, not claims

Run `python scripts/bench.py` yourself before presenting — it regenerates this from all 182 ordered room-pairs, so you can show it live:

| Algorithm | Mean path cost | Mean expansions | Optimal on all 182 pairs? |
|---|---|---|---|
| **A\*** | 7.868 | **4.90** | ✅ |
| Dijkstra | 7.868 | 8.00 | ✅ |
| BFS | 8.082 | 8.00 | ❌ (fails on 17/182) |
| Greedy | 8.088 | **3.34** | ❌ (fails on 20/182) |

**The number to lead with:** A\* finds the *identical* optimal cost as Dijkstra (both 7.868) while touching **39% fewer nodes** (4.90 vs 8.00 expansions). That's the heuristic earning its keep.

**The honest caveat you must volunteer, unprompted, because it's the difference between a defensible answer and an oversold one:** on a 14-node graph, this speed difference is real but modest — you will not see A\* finish in microseconds versus Dijkstra's seconds. The argument for A\* here is *not* "it's dramatically faster on this map." It's: (a) it is provably optimal *and* informed, which is strictly better than either Dijkstra (optimal, uninformed) or greedy (informed, not optimal) alone; (b) the admissibility proof is airtight and derivable in 30 seconds; (c) the exact same code scales unchanged to a much larger map, where the expansion-count gap would widen substantially. Say this before sir points it out — it's much stronger coming from you.

### 3.6 "Why not multi-agent pathfinding / CBS / Held–Karp / other fancier search?" — have the rejection ready

- **Held–Karp (exact TSP via bitmask DP)** would give the *provably optimal order* to visit multiple task rooms, instead of our simpler "nearest-first" greedy ordering. We deliberately didn't build it: with only 4 task rooms per crewmate, the optimal tour saves at most a handful of ticks over nearest-first, and the DP costs a nontrivial recurrence (`O(k² · 2^k)`, about 800 operations at k=5) to explain and defend. This is a **documented, deliberate trade-off** (see `docs/DECISIONS.md`), not an oversight — know the complexity figure if asked.
- **Conflict-Based Search / multi-agent pathfinding** solves the problem of multiple agents needing to avoid *colliding* in shared space. We don't need it because **rooms have unlimited capacity** — any number of agents can occupy the same room simultaneously; there is no collision to avoid. The problem CBS solves doesn't exist in our model.
- **A\* itself, "isn't it overkill on 14 nodes?"** — see §3.5's honest caveat above. Don't dodge this; answer it directly and move on.

---

<a name="4"></a>
## 4. The tick engine — simultaneous multi-agent resolution

### 4.1 The concept: why can't you just loop over agents one at a time?

If you resolved agent A's action fully (move it, maybe kill someone) before even asking agent B what it wants to do, agent B's decision would be based on stale information for some agents and fresh information for others, depending on turn order — physically nonsensical for a "everyone acts at once" game, and it would make the outcome depend on the arbitrary order agents happen to be iterated in.

The correct model is **simultaneous resolution**: every agent's `decide()` is called against the *exact same* snapshot of the world (the pre-tick observations), *then* all the resulting actions are resolved together, in a fixed, documented order. No agent can react within the same tick to what another agent just chose.

### 4.2 The actual 11-step pipeline (verified from `engine.py`, and this is worth memorizing in order)

```
1  Decide       every alive, non-transiting agent, same pre-tick Observation, fixed id order
2  Sabotage     at most one starts; id-order tie-break on same-tick collision
3  Arrivals     transiting agents who reach 0 remaining ticks land in their destination room
4  Kills        resolved between arrivals and departures (see 4.3 below — this ordering is load-bearing)
5  Departures   agents who chose to Move start a new transit
6  Work         DoTask progress, HoldPanel progress, task_bar updates
7  Reports      Report / PressButton validated; queues a meeting
8  Timers       cooldowns tick down, sabotage/door timers tick down, stall detection runs
9  Win check    all tasks done? all impostors ejected? impostors ≥ crew? reactor timer hit 0? max_ticks?
10 Observe      build next tick's Observation for every alive agent; generate their notes
11 Meeting      if one is queued and no reactor is active, run it; then reset (respawn, clear cooldowns)
```

Code: `engine.py:90-108` (`tick()`), each numbered step called out in a comment on its own line.

### 4.3 Why the arrivals/kills/departures split exists — the best "how do you know your system is correct" story in the whole project

**Tell this one in full if asked how you validated correctness — it's a genuine finding, not a rehearsed answer.**

The first working version of this engine resolved *all* movement (both arrivals and new departures) as a single step, entirely before kills were resolved. This seemed like the natural way to write it. It made kills **almost impossible**: a `Move` action puts an agent into transit *instantly* within that same tick, and an agent in transit cannot be killed (it's "between rooms"). Since crewmates issue a `Move` on almost every tick while doing their rounds, a target could simply *decide to walk away* in the very same tick an impostor decided to kill it, and the kill would silently fail — the impostor and the victim were "in the same room" when both decided, but the victim had already left by the time the kill resolved.

This was **found by instrumentation, not code review**: wrapping the impostor's `decide()` and counting outcomes showed **31 kill decisions where the target was genuinely co-located at decision time, of which only 2 actually resolved into a kill**. Across 20 games, that meant essentially no bodies, no meetings, no testimony, no votes — the LLM deliberation layer, the whole second half of the architecture, was never once being exercised, because the trigger for a meeting (finding a body) almost never occurred.

**The fix:** split movement into two steps around kills — *arrivals* land first (so nobody is mid-transit anymore this tick), *then kills resolve against who is now in each room*, *then departures* start fresh transits for anyone who chose to move. A target can no longer dodge a kill by choosing to leave in the same tick it's caught — it is already caught by the time it would leave, and it only gets to depart if it survives.

Say explicitly: *"This is exactly why the ordering of a simultaneous-resolution pipeline is a real design decision with observable consequences, not an arbitrary implementation detail — the 'obvious' ordering was actively wrong, and we only found out by measuring outcomes, not by reading the code."*

### 4.4 Task inheritance on death — the second subtlety in step 4

When a kill resolves, the victim's **incomplete tasks transfer** to the surviving crewmate currently holding the fewest remaining tasks (ties broken by fixed agent-id order), with progress-so-far preserved.

**Why this exists:** there are no ghosts in this game (deliberately out of scope). Without reassignment, a dead crewmate's unfinished tasks would be permanently stranded, and the crew's win condition ("all real tasks done") would become **mathematically unreachable** the moment the first kill happens. Measured on the pre-fix version: 19 stranded tasks across a batch of games, all ending in a draw at the 400-tick cap.

**The alternative you should proactively reject if asked "why not just remove the dead agent's tasks from the goal instead":** that creates a perverse incentive — killing a crewmate would *reduce* the total remaining work and help the crew finish sooner, which is backwards. Reassignment keeps the total task count constant across the whole game regardless of how many kills happen, so a kill costs the crew labor without ever *helping* them.

---

<a name="5"></a>
## 5. Partial observability — the Observation model

### 5.1 The concept: belief states vs. full states

In a partially observable environment, an agent cannot act on the true state of the world — only on its **belief state**, built from whatever its sensors report. The design challenge is making sure that belief state is (a) genuinely all the agent gets, with no accidental leakage of hidden information, and (b) rich enough to act sensibly on.

### 5.2 What's actually in an `Observation`, and the asymmetry that makes lights sabotage meaningful

From `observation.py:38-56`, an `Observation` carries: the agent's own room and physical state; a probabilistic list of **occupants** (each other alive agent in the room is included *independently*, with probability `1 − p_miss_eff`); any **bodies** in the room (visible with probability 1, or reduced during lights sabotage); **local events** (kills/vents witnessed this tick); the public task bar; a **sabotage alarm** that names the *kind* of sabotage and its timer but never who started it; and, if the agent is an impostor, its partner's identity and room.

**The one asymmetry worth memorizing exactly, because it's the mechanical payoff of the lights sabotage:**

```python
def p_miss_eff(world, agent, config) -> float:
    lights_active = world.sabotage is not None and world.sabotage.kind == "lights"
    if lights_active and agent.role is Role.CREWMATE:
        return config.p_miss_lights   # 0.50 during lights
    return config.p_miss              # 0.02 normally
```

(`observation.py:59-64`) — **impostors are structurally unaffected by their own sabotage.** This is what makes lights an actual weapon rather than a symmetric fog: it degrades only the crew's perception, creating a real window for an unseen kill.

### 5.3 Why the saboteur's identity is never in the Observation — and how it's provably true, not just claimed

`SabotageAlarm` (lines 12–18) has fields for `kind`, `panels`, `timer`, `panel_done` — no `agent` field. The `agent` who started the sabotage exists only in the ground-truth `SABOTAGE` log event (used for post-hoc analysis and the UI's optional reveal toggle) and is never constructed into anything a policy receives. If asked to *prove* this rather than just assert it: point at the dataclass definition itself — the field simply does not exist on the type a policy can see, so there is no code path by which it could leak, by construction rather than by discipline.

### 5.4 The perception-consistency bug — a second genuine "how do you know it's correct" story

**A second real bug worth having ready, distinct from the ordering one in §4.3.** The original implementation rolled the random "did you witness this kill" probability **twice, independently** — once when building the ground-truth `KILL` event's `witnesses` list (for the log), and again, separately, when building each individual agent's `Observation.local_events` for that same tick. Since both rolls used the same probability but were statistically independent draws, they could disagree: the log could claim agent `green` witnessed a kill, while `green`'s own `Observation` (and therefore its notes, and therefore what it could ever say in a meeting) showed nothing.

**The fix, and the invariant now enforced by a test:** perception is rolled **exactly once**, at the moment of resolution in `world/resolve.py`, producing a single `frozenset` of witnessing agents. That one set is threaded through as `TickSignals.kill_events`/`vent_events` (see `observation.py:21-34`'s docstring: *"kill_events already carry the resolved witness set... rolled once at resolution time... so observation-building only checks membership and never re-rolls"*) and used both for the log's `witnesses` field and for deciding each agent's `local_events`. A test now asserts these three views — the log, the observations, and the generated notes — always name the exact same set of agents.

Use this if asked "what kinds of subtle bugs does partial observability introduce that a fully-observable system wouldn't have?" — the answer is precisely this: **any single piece of information that's supposed to reach multiple, separately-computed views of the world must be computed once and shared, never independently re-derived**, or the views can silently diverge in a way unit tests on any one view, in isolation, will never catch.

---

<a name="6"></a>
## 6. The LLM deliberation layer — and why not Bayes

### 6.1 The alternative you rejected, stated fairly, before you explain why

The natural "proper AI" alternative to an LLM here is **exact Bayesian inference**: maintain a probability distribution over the `C(8,2) = 28` possible impostor-pair hypotheses, and update it via Bayes' rule every time new evidence arrives (`P(hypothesis | evidence) ∝ P(evidence | hypothesis) × P(hypothesis)`). This is more "principled" in the sense that it gives you calibrated, provably correct probabilities — *if* your model of the evidence-generating process is correct.

**That "if" is exactly where it breaks, and this is the crux of the whole justification:**

Exact Bayesian updating requires you to write down `P(evidence | hypothesis)` — a **likelihood function** — for every possible piece of evidence you might observe. That's tractable if evidence comes from a small, fixed schema: e.g., `CLAIM_LOCATION(room, t_from, t_to)`, where you can hand-write "if the speaker is honest under this hypothesis, this claim is true with probability X; if lying, Y." But the moment you let agents say **anything in natural language** — which is what makes deception interesting and realistic — the space of possible utterances is **unbounded**. You cannot write `P("I walked into electrical right as green was standing near red's body..." | blue-and-black-are-impostors)`. There is no likelihood function for an arbitrary sentence. The formal machinery simply has no input slot for open-ended text.

**The trade, stated honestly (this is the sentence to say word for word if asked "so which is better?"):**

> *"We traded a correctness guarantee — exact, provably-calibrated posteriors — for the ability to reason over open-ended natural language and produce genuine deception. That's a real cost, and we don't hide it: our suspicion scores are not calibrated probabilities, they're an LLM's best-effort estimate, and a single call is not even reproducible on its own. We contained that cost architecturally rather than pretending it doesn't exist — see §9 below."*

### 6.2 The architecture, end to end, with exact file references

**The `Deliberator` protocol** (`contracts.py:46-52`) is the single interface: `speak(ctx: MeetingContext) -> Statement`. Two implementations exist behind it, swappable by one config string (`config.deliberator = "template"` or `"gemini"`), with **zero code changes** required to switch — this is the ablation.

**`TemplateDeliberator`** (`llm/template.py`) — a deliberately simple, rule-based, zero-network baseline. It exists for three reasons, all of which you should state if asked "why bother building a fake LLM": (1) it's the **offline path** — the whole game must run with no API key, ever; (2) it's the **test path** — no test in the suite may require network access or a key; (3) it's the **honest ablation** — comparing `template` vs `gemini` output is literally the experiment that demonstrates what the LLM buys you.

**`GeminiDeliberator`** (`llm/deliberate.py`) — builds a structured prompt, calls the model, parses strict JSON, with fallback.

**The prompt — four fixed blocks, in this exact order** (`deliberate.py:22-81`, functions `_rules_block`, `_identity_block`, `_notes_block`, `_transcript_block`):

1. **Rules**: the room list, who's alive, the fact that sight is room-local — general game knowledge every agent shares.
2. **Identity**: `"You are {self_id}, a crewmate."` — or, for an impostor: who its partner is, what it actually did (`"You did this: I killed red in electrical."`, built from its own genuine `kill` notes), and the explicit instruction *"You must not be caught. You may lie, but your story must fit what others can check."*
3. **Notes**: the speaker's own most recent observations, verbatim, nothing else's.
4. **Transcript**: every statement made *so far this meeting*, in speaking order — including earlier speakers in the *same* round, not just prior rounds (this was a real bug, fixed — see below).

**Response format — strict JSON, requested via the model's native JSON mode** (`{"statement": "...", "suspicion": {"agent": 0.0-1.0, ...}, "vote": "agent" or null}`), parsed defensively (`deliberate.py:84-109`): strip markdown code fences, validate required keys exist, clamp every suspicion score into `[0,1]`, silently drop any agent id that's unknown or already dead, coerce an invalid vote target to `null`.

### 6.3 Why the intra-round transcript bug mattered, and is worth mentioning unprompted

Originally, the transcript passed into each speaker's prompt was **frozen at the start of the round** — so a speaker in round 1 could never see what an *earlier* speaker in that same round had just said. This silently defeated the whole point of a speaking order: §10's mechanic of "the reporter speaks first, then others can react" only works if later speakers can actually see and respond to earlier ones. Fixed by growing the transcript live as each statement is produced (`protocol.py:104-129` — `transcript.append(statement)` happens inside the per-speaker loop, and `build_ctx` is called fresh for each speaker with `list(transcript)` as it stands *at that moment*). This is also mechanically what allows one agent to contradict another within the same round — the specific behavior the flagship demo scenario relies on.

### 6.4 A subtler bug: the impostor confessing its own crime — good example of "template deliberator as a naive baseline, done wrong"

The original `TemplateDeliberator` picked a single "highest priority" note to speak (`kill > body > saw > alone`) using the *same* logic for both roles. An impostor's own kill is recorded as a `kill`-kind note in its own memory (it has to know what it needs to explain away — `deliberate.py:33-35`, `_own_kills`). Since `kill` was the highest priority, an impostor would **announce its own murder out loud in the meeting**: `"t=42 I killed red in electrical."` Found by actually reading a generated transcript, not by unit tests (every existing unit test happened to check *some* statement was produced, not *what* it said). Fixed by branching the template's statement-selection on role: an impostor only ever voices a `saw`/`alone` note (a location claim), never a `kill`/`vent`/`body` note about itself. A regression test now sweeps many seeds asserting no impostor statement, ever, contains "I killed" or "I vented."

**The teaching point if asked "what's the hardest kind of bug in a system like this to catch":** anything where the *type signature* is fine (a `Statement` was produced, it has valid fields) but the *semantic content* is catastrophically wrong. Pure unit tests on structure won't catch it; you have to actually read what came out.

### 6.5 Demo safety — the engineering that makes "what if the API is down" a non-issue

- **Missing key** → `LLMClient.__init__` (`client.py:63-73`) detects no `GEMINI_API_KEY` in the environment, logs one clear line, and the client behaves identically to `template` for the rest of the run. No exception anywhere.
- **A malformed / truncated / non-JSON response** → `GeminiDeliberator.speak` (`deliberate.py:127-143`) tries to parse it; on failure, retries **once** with a forced cache-bypass; on a second failure, raises `LLMFailure`. That exception is caught one layer up in `MeetingProtocol._speak` (`protocol.py:148-155`), which logs an `LLM_PARSE_FAIL` event and falls back to a `TemplateDeliberator` instance for that one agent's statement. **The rest of the meeting is unaffected.**
- **Caching** — every call is keyed by `sha256(model + "\n" + prompt)` (`client.py:80-82`). A cache hit *never* touches the network (there's a dedicated test proving this using a fake client that raises an exception if it's ever called — if the cache logic were broken, that test would fail loudly). This is also what makes a *recorded* run replay byte-for-byte identical on a second playback — see §9.
- **Retry with backoff** — a transient failure (503 "model busy," 429 rate limit) retries up to `llm_max_attempts` (default 4) times with **exponential backoff plus jitter** (`client.py:140-175`, `_retry.py`), bounded to roughly 10.5 seconds worst case before giving up and falling back — chosen specifically so a live demo can never hang. A **permanent** failure (404 model not found, 401/403 bad credentials) is detected and **not retried at all** — retrying a dead model name for 10 seconds during a live demo would be strictly worse than failing fast. We validated this distinction against a real quota-exhaustion event during development, not a simulated one: dozens of consecutive live calls failed with `429 RESOURCE_EXHAUSTED`, and every single one degraded cleanly to the template path with the run completing normally.

**If asked "what if the model changes its API or gets deprecated mid-project" — this actually happened to us**, and it's a great real anecdote: `gemini-2.0-flash` (our original default model string) was retired mid-development and started returning `404 NOT_FOUND` with a message pointing us at `gemini-3.8-flash`. Config is one string in `SimConfig` (`llm_model`); updating it took one line and no other code changed, because the model name is a configuration value, never hardcoded into the calling logic.

---

<a name="7"></a>
## 7. The reactor renegotiation protocol — distributed consensus without a coordinator

This is one of your two named "Exceeds Expectations" mechanisms (rubric explicitly names "dynamic renegotiation"). Know it cold.

### 7.1 The concept: distributed agreement with no central authority

The problem: when the reactor is sabotaged, two panels (in far-apart rooms) must each be held by *some* agent for a fixed duration, or the impostors win when the timer expires. There is no single "coordinator" agent assigning tasks — every agent must independently arrive at the *same* conclusion about who's responsible for what, using only information that's been made **public**. This is a **distributed consensus problem**: the same computation, run by every party on the same shared input, produces the same output everywhere, without any message beyond "here's my public data" ever needing to be sent between deciding parties.

**Why a `ReactorBoard` object exists at all** (`agents/reactor.py:44-84`) — the `Observation` type is deliberately room-local and has no generic "broadcast channel" field (that would let information leak in ways that break partial observability elsewhere). The `ReactorBoard` is a **stand-in for a public radio channel specifically for this one protocol** — every agent's policy holds a reference to the *same* board instance, reads and writes to it directly, and no agent ever reaches into another agent's private state to do so. This is explicitly called out as a wiring requirement: whoever constructs the 8 policies for a game must pass **one shared board instance** to all of them, or the protocol silently degrades into 8 isolated boards that never communicate.

### 7.2 The four phases, and the exact function for each

**1. Bid** (`reactor.py:191-199`, `_bid`) — once the reactor alarm sounds, every alive agent computes its own A\* travel cost (with `plain_cost_fn`, no risk term — bidding is about honest logistics) to each of the two panels, and writes it onto the shared board under its own id. An impostor planning to defect **multiplies its bid by `impostor_underbid` (<1)** before broadcasting — a strictly smaller number wins auctions, so it deliberately looks like the best candidate for a job it doesn't intend to finish.

**2. Commit — the actual consensus function** (`reactor.py:97-124`, `compute_assignment`, a *pure function*):

```python
def compute_assignment(bids, panels):
    p1, p2 = panels
    best_key = None
    for a in sorted(bids):
        for b in sorted(bids):
            if a == b: continue
            key = (max(bids[a][p1], bids[b][p2]), a, b)   # minimize the WORSE of the two arrival times
            if best_key is None or key < best_key:
                best_key = key
    return {p1: best_key[1], p2: best_key[2]}
```

It picks the pair `(a, b)` minimizing the **later** of the two arrivals (`max(...)`, not `sum(...)`) — because both panels must be done, so what matters is when the *slower* of the two assignees finishes, not the total travel distance. Ties break lexicographically by agent id, which is deterministic and never depends on dict-iteration order.

**Why this is genuinely "no central coordinator" and not just a claim:** every single one of the 8 agents runs this exact same function, independently, over the exact same `bids` dictionary (because it's read off the shared board, and — critically — every negotiator *waits until every alive agent's bid has arrived* before computing anything, `reactor.py:201-225`, `_commit`, specifically to guarantee the input set is complete and identical for everyone rather than order-dependent). Since the function is pure and the input is identical, every agent arrives at the identical output with zero messages exchanged beyond the public bids. There's a direct test asserting all 8 agents compute the identical assignment from the same bid set.

**3. Revoke** (`reactor.py:227-273`, `_check_revokes`) — panel completion status is public (it's on the observation's sabotage alarm). If `now > eta + eta_grace` and the panel still isn't done, the commitment is revoked, the defector's id is added to a `revoked` set (so it can never again be selected as anyone's backup for the rest of this window — a genuine bug we found and fixed: an earlier version let a just-revoked agent be handed a fresh assignment seconds later), and the **next-cheapest still-eligible bidder** is assigned as backup with a **freshly recomputed** ETA (not a stale bid from many ticks earlier — also a genuine bug we found: reusing an old bid as a hard deadline could assign an honest agent an arithmetically impossible ETA and get it wrongly revoked).

**4. The payoff — a broken commitment becomes social evidence** (`reactor.py:347-361`, `_record_missed_commits`): every revocation is converted into a plain-English note — `"t=130 black committed to fix o2 by t=126 and never did."` — appended to **every** agent's memory. That note flows straight into the next meeting's LLM prompt (§6.2's notes block). **This is the single best sentence in the whole project to say if asked "how do the search/engine layer and the LLM layer actually connect": a coordination-protocol failure is mechanically converted into testimony the social-reasoning layer can act on.**

### 7.3 The honest nuance — say this unprompted, it shows depth

Honest agents can also miss a deadline (doors sabotage blocking their route, a genuinely long path) — so a revocation is **evidence of suspicious behavior, not proof of guilt**. That ambiguity is intentional: it's what makes the ensuing accusation in the meeting a real judgment call for the LLM, rather than a free "gotcha."

### 7.4 Measured — a real run, seed 3, `reactor_defect` scenario

```
t=3    SABOTAGE  black starts reactor (timer 30)
t=4-6  BID x8    every agent bids its A* cost to each panel
t=6    COMMIT    black -> reactor (eta=6)   white -> o2 (eta=8)
t=10   REVOKE    black (eta 6 + grace 3, now past) -> backup orange
t=12   REVOKE    white (eta 8 + grace 3, now past) -> backup green
t=19   PANEL_DONE green @ o2
t=20   PANEL_DONE orange @ reactor
t=20   SABOTAGE_FIXED   (took 17 ticks total)
t=34   MEETING #1 — black's suspicion sits at 0.50, vs 0.10 baseline for everyone else
```

That last line is the demonstration: black's defection at t=10 became a `missed_commit` note, and by the next meeting, every agent's baseline suspicion of black is already elevated *before anyone has said a word*.

---

<a name="8"></a>
## 8. Deadlock — an emergent multi-agent failure mode

Your second named "Exceeds" mechanism.

### 8.1 The concept: an emergent failure with no single agent at fault

This is the interesting one to explain well: **the deadlock isn't a bug and isn't caused by an impostor** — it's an emergent consequence of two individually rational rules interacting badly. §11.1 step 5 says: if a co-located agent's suspicion clears `theta_shadow`, stop doing your task and follow them instead (a locally sensible "keep an eye on the person I distrust" behavior). If **two crewmates each cross that threshold for the other, simultaneously**, both abandon their tasks to follow each other, neither one ever actually catches the other (each is reactively tracking the other's last-known position), and if between them they're holding the last uncompleted tasks, **global progress stops completely** — with zero impostor involvement. Two perfectly reasonable individual rules produce a collectively pathological outcome. This is a genuine, well-known category of multi-agent systems failure (related to livelock in distributed systems), not a contrived scenario.

### 8.2 Detection — why it must use only public information

`world/sabotage.py:185-202`, `detect_stall`, requires **all three** of these conditions simultaneously:

1. `tick - last_progress_tick >= deadlock_window` (no task progress for 20+ ticks)
2. no sabotage is currently active (a reactor lockdown legitimately halts task progress too — that's not a deadlock, don't confuse the two)
3. no meeting has happened recently (a meeting also legitimately pauses everything)

**Why the engine, not an individual agent, detects this:** the trigger condition (`world.last_progress_tick`) is drawn from the **publicly visible task bar** — every agent can already see the task bar isn't moving, so this isn't a hidden-state leak; it's a globally observable fact the engine is simply allowed to notice and broadcast as a `PROGRESS_STALL` event.

**A genuine bug we found in this detector, worth mentioning if asked "was this hard to get right":** the very first implementation only checked condition 1 — because the engineer building it hadn't yet been given the §13 spec text describing conditions 2 and 3. Missing condition 2, in particular, would have made a reactor sabotage spuriously look like a deadlock every time it ran long enough, incorrectly suspending shadowing during a completely unrelated, legitimate lockdown.

### 8.3 Resolution — and a *third* genuine bug that only measurement caught

On `PROGRESS_STALL`, every crewmate stops shadowing, disables it for `shadow_cooldown` further ticks, and forces a fresh route plan (`crewmate.py:127-132`, `_handle_stall`).

But even *after* that fix was implemented and tested, live gameplay measurement still showed occasional 15–20 tick freezes. The actual root cause, found by instrumenting live games rather than by re-reading the stall-detection code: the shadowing logic itself (`crewmate.py:220-255`, `_shadow`) picked its target purely by "highest suspicion score in memory," with **no check that the sighting was still fresh** — so a crewmate could get stuck "shadowing" the last known position of an agent who had since been **ejected or killed**, chasing a ghost that could structurally never be caught, for up to a full `deadlock_window` before the stall detector even noticed something was wrong. The fix restricts shadow candidates to sightings within `last_seen_decay` ticks (the same freshness window already used for the A\* risk cost, so it's reusing an existing concept rather than inventing a new one).

**Why this is worth telling in full if asked "how do you validate multi-agent emergent behavior":** the mechanism (`_handle_stall`) was individually correct and individually unit-tested, and the *system* still misbehaved, because the actual bug lived in a completely different function (`_shadow`) whose interaction with a rare timing condition (an ejection happening while being shadowed) was never exercised by any single unit test. Only running real games at scale and looking for anomalous idle periods surfaced it.

### 8.4 The ablation — same seed, one flag, two outcomes, and this is the demo

`config.deadlock_protocol = False` suppresses `detect_stall`'s ability to ever emit the event (`sabotage.py:195`, gated on `config.deadlock_protocol`). Same random seed, same scripted mutual-suspicion setup, one config flag flipped: with the protocol on, the stall is detected and broken, tasks resume, crew wins. With it off, the same two agents freeze permanently and the game runs out the clock to a draw at `max_ticks`. **That contrast, replayed side by side in the viewer, is the single strongest visual demonstration in the entire project** — it needs zero narration to understand.

---

<a name="9"></a>
## 9. How determinism and reproducibility actually work

**The claim:** given the same seed, the same scenario runs byte-for-byte identically every time — including with the LLM in the loop, once its responses are cached.

**How, mechanically:**

1. **All world randomness passes through exactly one seeded generator.** `rng.py` creates one `numpy.random.Generator(seed)`; there is a project-wide rule (checked by discipline, not by a runtime guard) that nothing in `world/` ever calls Python's built-in `random` module or an unseeded `numpy` call. Role assignment, observation misses, task duration noise, meeting speaking order — all draw from this single stream, in a fixed order dictated by the tick pipeline, so the *sequence* of random numbers consumed is itself deterministic given the seed.
2. **Every LLM call is cached by `sha256(model + "\n" + prompt)`.** A cache hit is a pure dictionary lookup — no network round-trip, no fresh randomness from the model's own sampling temperature. So a *second* run of the exact same scenario and seed, once the cache is warm, produces the exact same prompts (deterministic game state) which hit the exact same cache entries (deterministic model output) — full end-to-end reproducibility, LLM included.
3. **Event-log serialization is order-independent-proof.** Every log line carries `t`, `type`, and a fixed set of named fields (never a raw dict dump whose key order could vary) — verified by a test that hashes the same scenario+seed's `events.jsonl` twice and asserts the SHA-256 digests match exactly.

**The one place nondeterminism genuinely exists, and you should say this plainly rather than overclaim perfection:** the *very first, uncached* call to a live LLM is not reproducible on its own — the model itself may sample differently between calls even with identical input (temperature/sampling is outside our control). We don't hide this; the caching strategy exists *specifically* to contain that nondeterminism to a one-time cost, after which every replay is exact.

---

<a name="10"></a>
## 10. The testing philosophy: what code review misses, that measurement catches

This section exists because it answers, comprehensively and with concrete receipts, the single most dangerous question you can be asked: **"how do you actually know your system works correctly?"**

The honest, memorable answer: *"Every module has unit tests, and we treat those as necessary but not sufficient — five of the most serious bugs in this project were found only by running real games and measuring outcomes, because they were bugs of interaction between individually-correct components, invisible to any single unit test."* Then walk through this table (all verified against `docs/DECISIONS.md`):

| Bug | Each involved component was individually tested and correct | What measurement actually caught |
|---|---|---|
| Movement resolved fully before kills | Movement logic: correct. Kill validity logic: correct. | Wiring an impostor's `decide()` with a counter showed 31 kill *decisions* against a co-located target, only 2 *resolved* — an emergent interaction between two correct pieces |
| Perception rolled twice (log vs. observation) | Both random rolls, individually, used the documented probability correctly | Directly diffing the log's `witnesses` field against what agents' own notes contained, across many seeds, revealed disagreement |
| `PressButton` unreachable in practice | The action type, the validity check, the engine handling — all correct | Counting `BUTTON` events across 20 live games: exactly zero, despite dozens of kills |
| Dead crewmates' tasks stranded | Kill resolution: correct. Win-condition check: correct. | Counting incomplete tasks held by *dead* agents at game-end: 19, across a batch — win became mathematically unreachable |
| `Report` livelock (paralyzed 20% of games) | The Report action's validity check was correct in isolation; the crewmate's priority-1 rule was correct in isolation | Instrumenting `decide()` during a frozen-looking game showed four agents returning `Report` on *every single tick* for 29+ consecutive ticks — the engine silently rejected the invalid re-report every time, and that rejection was filtered out of the event log (by design, to keep the log clean of debug noise), so the failure was invisible as an *absence* of any other event, not a visible error |

**The sharpest one, if you only have time to tell one:** the `Report` livelock. The engine did exactly the right thing — it correctly rejected an invalid repeated action. But *because* the rejection was silently filtered rather than logged, a systemic failure looked, from the outside, like nothing happening at all rather than like an error. That's a genuinely subtle lesson: **silently-correct rejection of bad input can itself hide a design flaw**, if nothing is watching for the *absence* of expected events, not just the presence of error events.

**If pressed further — "so is your test suite actually any good, then?"** — yes, and say why: the unit test suite (328+ tests) is what makes *refactoring* safe and catches regressions the moment they're introduced; it's necessary. But it structurally cannot catch *emergent, interaction-level* bugs between correctly-implemented components, because by definition a unit test isolates one component. That's why every one of the five bugs above was followed by a **new regression test targeting the specific emergent behavior** (e.g., "no game should show 15+ consecutive ticks with zero movement while still in progress") rather than just a fix — converting a measurement finding into a permanent guard.

---

<a name="11"></a>
## 11. Rapid-fire cross-questioning drill

Practice these out loud, from memory, without looking at the answer first. If you fumble one, go re-read the relevant section above, then come back and try again cold.

1. **"Define admissible heuristic, then prove ours is one."** → §3.4.
2. **"Prove your risk-weighted cost doesn't break that proof."** → §3.4, second follow-up.
3. **"Why is BFS wrong here specifically, with a real example from your own map?"** → §3.2, `storage↔upper_engine`, 9 vs 11 ticks.
4. **"You said A\* is optimal — is Dijkstra not also optimal? Then why prefer A\*?"** → §3.2's one-sentence summary: both optimal, A\* is also informed.
5. **"Isn't A\* pointless on a 14-node graph?"** → §3.5's honest caveat, said unprompted before they finish the question ideally.
6. **"Why not exact Bayesian inference for suspicion?"** → §6.1, word-for-word the trade-off sentence.
7. **"So your suspicion scores aren't real probabilities?"** → Correct, say so plainly — they're an LLM's best-effort estimate, not calibrated posteriors, and that's the acknowledged cost of the trade in #6.
8. **"Prove there's no way for an agent to see another room."** → §5.3 — point at the `Observation`/`WorldState` type separation; there's no field, so there's no path, not just a policy of not using one.
9. **"Is this cooperative or competitive?"** → §2, all three layers, emphasize layer 3 (the defection incentive inside the impostor pair).
10. **"How do you know the renegotiation protocol really has no central coordinator, versus you just calling it that?"** → §7.2 — it's a pure function, every agent runs the identical instance of it over identical (complete, waited-for) public input, and there's a direct test asserting the outputs match across all 8 agents.
11. **"What happens if the API is down mid-demo?"** → §6.5 — and mention the real quota-exhaustion event that already happened and was survived cleanly during development, not a hypothetical.
12. **"How is the deadlock not just a scripted animation?"** → §8.1 — it's the product of two individually-reasonable rules (§11.1 step 5's shadowing threshold) interacting; walk through why two agents each independently crossing `theta_shadow` for each other produces the freeze, with no scripted "deadlock behavior" anywhere in the code.
13. **"What's the hardest bug you found, and how?"** → §10, the `Report` livelock — emphasize it was found by *instrumenting and measuring*, not by reading code.
14. **"Why Mesa, if your engine doesn't use its scheduler?"** → Mesa 3 removed the old `RandomActivation`/`SimultaneousActivation` scheduler classes, so we couldn't have used them even if we wanted to; our engine does its own simultaneous resolution (§4.2) precisely because that's now a requirement, not a choice. Mesa contributes `DataCollector` and standard model scaffolding in `sim/model.py` (~60 lines) — don't claim it does more than that.
15. **"How would this scale to 20 agents or a 50-room map?"** → A\* stays `O(E log V)`, essentially untouched by more rooms. Impostor-pair hypotheses (if you were doing exact inference, which you're not) would grow as `C(n,2)` — but since suspicion is LLM-produced, not enumerated, the real cost driver is prompt length and per-meeting LLM call count, which grows *linearly* in agent count, not combinatorially.
16. **"Why 2 meeting rounds, not more?"** → It's a deliberate model of a real information bottleneck: agents accumulate many private observations between meetings but can only voice a couple of statements. More rounds would let the crew converge on the truth too easily and remove the impostors' realistic chance of surviving a vote.

---

<a name="12"></a>
## 12. If you get asked to run something live

Keep this exact sequence ready — you've verified all of it works:

```bash
cd /home/kundhave/Projects/Among_us

# A concrete, honestly-picked crewmate win with real drama (3 kills, 2 meetings, sabotage):
.venv/bin/python -m amongus.sim.runner --scenario baseline --seed 4 --verbosity 2

# A concrete impostor win (majority achieved via coordinated kills + doors sabotage):
.venv/bin/python -m amongus.sim.runner --scenario baseline --seed 24 --verbosity 2

# The scripted headline: a kill, a discovery, a full two-round meeting with contradiction:
.venv/bin/python -m amongus.sim.runner --scenario worked_example --seed 3 --verbosity 2

# The renegotiation protocol end-to-end (bid -> commit -> defect -> revoke -> backup -> fixed):
.venv/bin/python -m amongus.sim.runner --scenario reactor_defect --verbosity 2

# Then open src/amongus/ui/viewer.html in a browser and drag the resulting
# runs/<scenario>_<seed>/events.jsonl onto it for the visual replay.

# Live shock injection (press keys while it runs, no --no-interactive):
.venv/bin/python -m amongus.sim.runner --scenario baseline --seed 7
#   press: l = lights   d = doors   r = reactor   k = clear cooldowns   b = force a meeting

# The full paced demo, all four scenarios back to back:
.venv/bin/python scripts/run_demo.py

# The A* comparison table, regenerated live from all 182 room-pairs:
.venv/bin/python scripts/bench.py

# The full test suite, if asked to prove it's all green:
.venv/bin/pytest -q
```

**If sir wants to see genuine live Gemini output** rather than the offline template: add `--deliberator gemini` to any command above, with `GEMINI_API_KEY` set in your shell environment. Be upfront that the free tier caps this at 20 requests/day, and mention that this constraint is exactly why the offline `template` path exists as a first-class, fully-tested implementation rather than an afterthought.
