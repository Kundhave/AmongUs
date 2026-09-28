# AI Among Us

## What this is

Eight agents (6 crewmates, 2 impostors) move on a 14-room weighted graph in discrete ticks, each seeing only its own room. **Search handles space, the LLM handles people:** crewmates route with **A\*** over a risk-weighted graph, do tasks and report bodies with zero LLM calls in the tick loop, while at meetings each agent sends its plain-English notes to Gemini Flash and gets back a statement, a suspicion ranking and a vote. That ranking feeds back into the A\* edge cost, so crewmates physically route around agents they distrust. Two coordination mechanisms are exercised deliberately: a reactor **renegotiation protocol** (commit → defect → revoke → backup) and a shadowing **deadlock** broken by a zero-progress timeout. Sabotages double as environmental shocks and can be injected live by keypress.

Full spec: `docs/SPEC.md`. If code and that document disagree, the document wins unless the deviation is recorded in `docs/DECISIONS.md` with a reason.

`docs/archive/v1_bayesian_reference.md` is the superseded Bayesian design. **It is not authoritative — never point a subagent at it.**

## Two constraints that outrank everything

1. **Explainability.** This is built to be defended in a viva. Given a clever solution and an obvious one, take the obvious one. Modules stay under ~200 lines. No algorithm goes in that cannot be derived on a whiteboard in under a minute.
2. **The demo must never fail live.** No API key, no network, a malformed LLM response, an unknown event type in the viewer — all of these degrade, none of them crash. Every test runs offline.

## You are the orchestrator

This main session plans, delegates, integrates and verifies. It writes **docs and specs**, plus glue or fixes under ~20 lines. It does **not** write module code — that goes to a subagent. Subagents cannot spawn subagents, so every delegation happens from here.

## Delegation rules

Every task sent to a subagent names, explicitly:

1. the milestone (M1–M5);
2. the **exact § sections** of `docs/SPEC.md` to read — never "read the spec";
3. the files it may create or edit;
4. the acceptance checks from §17 (tests) and §18 (milestone acceptance);
5. the closing line: *"report back in ≤15 lines: files changed, tests added, test result, open issues"*.

Further rules:

- Do not paste file contents into delegation prompts. Give paths; the agent reads them.
- Run subagents in parallel **only** where owned files do not overlap: **M1a search ∥ M1b world**, and **M5 UI ∥ telemetry**. Everything else is sequential.
- After each implementer finishes, send `test-runner`. A milestone closes only when tests are green.
- After **M3, M4 and M5**, send `spec-reviewer` over the relevant sections. Fix confirmed mismatches before closing.

### Agents

| Agent | Owns | Sections |
|---|---|---|
| `world-engineer` | `config.py`, `rng.py`, `types.py`, `contracts.py`, `world/*`, `pyproject.toml` | §2–§7, §12.1–12.2, §13 detection |
| `search-engineer` | `search/*`, `scripts/bench.py` | §4, §5, §8 |
| `agent-engineer` | `agents/*`, `llm/*` | §4, §7, §8.2, §8.4, §9–§13 |
| `telemetry-engineer` | `sim/*`, `scripts/run_demo.py`, `README.md` | §14, §16 |
| `ui-engineer` | `src/amongus/ui/*` | §5, §14.1, §15 |
| `test-runner` | nothing (read-only) | §17 |
| `spec-reviewer` | nothing (read-only) | whole spec, one section at a time |

## Project rules (§0)

- Python ≥ 3.11. Type hints everywhere, `dataclass` for data, `typing.Protocol` for contracts.
- **All world randomness** goes through one seeded `numpy.random.Generator` from `rng.py`. No `random` module, no unseeded calls.
- **The LLM is called only during meetings.** Never inside the tick loop. Every call is cached by prompt hash, so a warm-cache replay is byte-identical.
- `GEMINI_API_KEY` comes from the environment. It is never written to a file or committed.
- Every tunable lives in `SimConfig` (§3). No magic numbers elsewhere.
- Every public function gets a one-line docstring. Modules under ~200 lines — split if larger.
- Mesa only in `sim/model.py`; `world/` must never import it. Mesa 3 has no `RandomActivation`/`SimultaneousActivation` — the engine does its own simultaneous resolution (§6).
- No pandas, no matplotlib. The UI is one self-contained HTML file — no build step, no CDN, no framework.
- Modules talk through `contracts.py` (`Planner`, `Deliberator`, `Policy`). Planner and deliberator are chosen by config string so ablations need no code edits.
- Agents receive only an `Observation`, never `WorldState` (§7).
- No test may require an API key.

### Out of scope (§19) — do not build

Pygame, a live web server, continuous movement or collision physics, multi-agent pathfinding (CBS etc.), reinforcement learning or any training, ghosts, cameras/vitals, multiple maps, exact Bayesian belief, Held–Karp, time-expanded planning, pandas, a database, LLM calls inside the tick loop. A subagent that thinks one of these is needed stops and asks the orchestrator.

## Why these choices (for the viva)

- **A\* over BFS/Dijkstra/greedy:** the rubric asks for informed search. `h(u) = euclid(u, dst) / s` where `s` is the fastest distance-per-tick of any edge — admissible because no path covers ground faster than `s`. The risk term only adds cost, so `h` stays admissible.
- **LLM over exact Bayesian inference:** exact inference needs a hand-written likelihood per message and a known generating process, which only works for a fixed schema. Natural-language testimony has an unbounded observation space, so there is no likelihood to write. We traded the correctness guarantee for open testimony and deception — and confined the resulting nondeterminism by seeding the world and caching every call.
- **Nearest-first task ordering over Held–Karp:** with 4 task rooms the optimal tour saves a few ticks and costs a subset-DP recurrence to defend. Know that Held–Karp is `O(k²·2^k)` and that this was a judgement call, not an oversight.

## Tracking

- `docs/PROGRESS.md` — milestone checklist with dates and one-line notes. Update after each milestone closes.
- `docs/DECISIONS.md` — every deviation from the spec, with the reason.

A subagent that deviates must say so in its report. The orchestrator decides whether to accept it, then records it. Do not let a deviation pass unrecorded.

## Commands

```bash
pip install -e ".[dev]"
pytest -q
ruff check src tests
python scripts/bench.py                     # search comparison table
python -m amongus.sim.runner --scenario worked_example --seed 3 --verbosity 2
python -m amongus.sim.runner --scenario baseline --seed 7 --shock 60:lights
```
