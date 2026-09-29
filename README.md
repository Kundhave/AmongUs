# AI Among Us

Eight agents (6 crewmates, 2 impostors) act on a 14-room weighted graph in discrete ticks.
Crewmates route with A\*, do tasks, and vote; impostors kill, sabotage, and lie in meetings.
Search handles space; an LLM (or an offline template fallback) handles people. Full design in
`docs/SPEC.md`.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Python >= 3.11. No GPU, no database, no build step for the UI.

## Run a scenario

```bash
.venv/bin/python -m amongus.sim.runner --scenario worked_example --seed 3 --verbosity 2
```

Flags:

| Flag | Meaning |
|---|---|
| `--scenario {baseline,worked_example,reactor_defect,standoff}` | which §16 scenario to run |
| `--seed N` | override the scenario's default seed |
| `--ticks N` | override `max_ticks` |
| `--verbosity {0,1,2,3}` | 0 silent, 1 events, 2 + suspicion, 3 + debug |
| `--out DIR` | base directory for `events.jsonl` (default `runs/`) |
| `--shock tick:kind[:room]` | schedule a shock at a tick; repeatable, e.g. `--shock 60:lights --shock 120:doors:storage` |
| `--no-interactive` | never poll stdin for live keypresses |
| `--deliberator {template,gemini}` | §9 ablation switch, default `template` |

Every run writes `<out>/<scenario>_<seed>/events.jsonl` — the single, deterministic contract
the replay viewer reads (§14). The same scenario and seed always hash to the same log.

`--deliberator` defaults to `template`, so every default path, every scenario and every test
is fully offline and needs no key. Gemini is optional: pass `--deliberator gemini` with
`GEMINI_API_KEY` set in the environment (never in a file or on the command line) to route
meeting deliberation through Gemini Flash instead — "search handles space, the LLM handles
people" (§9). If the key is missing or a call fails, the run falls back to the template
Deliberator automatically and still completes; it never crashes for lack of a key.

## The four scenarios

| Name | What it shows |
|---|---|
| `baseline` | A full unscripted game to a winner, with meetings and suspicion evolving |
| `worked_example` | `blue` and `red` are scripted alone into `electrical`; `blue` kills at t=43; `green` walks in and reports at t=47 — the deception/testimony headline |
| `reactor_defect` | `black` spawns at the reactor and wins its panel bid, then defects (`p_defect=1`): bid -> commit -> **revoke** -> backup, all off one shared `ReactorBoard` |
| `standoff` | Two crewmates are primed into mutual suspicion and freeze in shadow; `PROGRESS_STALL` fires, `deadlock_protocol` decides whether it ever breaks |

Run all of them back-to-back, paced for a live audience:

```bash
.venv/bin/python scripts/run_demo.py
```

It plays `worked_example` -> `reactor_defect` -> `standoff` (both with and without
`deadlock_protocol`) and prints where each resulting `events.jsonl` landed. It also accepts
`--deliberator {template,gemini}` (default `template`); the demo runs start-to-finish with
no API key either way.

## Live shock injection (§16.1)

While `runner.py` is running interactively (stdin is a real terminal), single keypresses
inject a shock at the current tick without blocking the simulation:

| Key | Effect |
|---|---|
| `l` | lights sabotage |
| `d` | doors sabotage on a random occupied room |
| `r` | reactor sabotage |
| `k` | clear both impostors' kill cooldowns |
| `b` | force an emergency meeting |

Each keypress logs a `SHOCK` event. Piped or redirected stdin (including under pytest)
degrades silently to no polling — the tick loop never blocks or slows down. The same
mechanism is available non-interactively for scripted/reproducible demos via repeated
`--shock tick:kind[:room]` flags.

## The replay viewer

`src/amongus/ui/viewer.html` is one self-contained HTML file — open it in a browser and
drag an `events.jsonl` onto it, or load `viewer.html?log=<path>`. No server, no build step.
It shows the map, task bar, event feed, a meeting overlay with the transcript and vote
tally, and a reveal toggle (off by default) that outlines the true impostors from the
`ROLES` event.

## Bench (search comparison, §8.5)

```bash
.venv/bin/python scripts/bench.py
```

Runs A\*, BFS, Dijkstra and greedy over all 182 ordered room pairs and writes
`runs/bench/search.csv` plus a Markdown table to stdout.

## Ablations

Every ablation is a `SimConfig` string field — no code edits needed:

| Field | Values | Effect |
|---|---|---|
| `planner` | `astar` \| `astar_no_risk` | whether A\*'s edge cost includes the suspicion-weighted risk term (§8.2) |
| `deliberator` | `gemini` \| `template` | LLM meeting deliberation vs. the offline rule-based fallback (§9.4) |
| `commit_protocol` | `True` \| `False` | distributed reactor bid/commit/revoke/backup vs. every agent rushing the nearest panel |
| `deadlock_protocol` | `True` \| `False` | whether a `PROGRESS_STALL` ever breaks a shadowing standoff |

Pass overrides via `dataclasses.replace(SimConfig(), ...)`, or (for a scenario) through its
`config_overrides` dict in `src/amongus/sim/scenarios.py`.

## Offline by design

No test in this repo makes a live network call, and no scenario does either unless you pass
`--deliberator gemini` yourself. `deliberator="template"` is a pure-Python, no-network
`Deliberator`; it is the default for every scenario, `run_demo.py`, and every test. The
world is driven by one seeded `numpy.random.Generator` (`rng.py`), so a recorded run with a
warm LLM cache replays byte-identically — hashing `events.jsonl` twice for the same seed is
one of the tests (`tests/test_scenarios.py`, `tests/test_runner.py`).

## Tests

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
```
