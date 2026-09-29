# Progress

Milestones per `docs/SPEC.md` §18. A milestone closes only when `test-runner` reports green — and for M3, M4 and M5, when `spec-reviewer` has passed.

| Done | Milestone | Owner(s) | Acceptance (§18) | Date | Note |
|---|---|---|---|---|---|
| [x] | **M0 Re-spec** | orchestrator | SPEC.md v2, CLAUDE.md, agent set, DECISIONS entries | 2026-09-28 | Pivoted off the v1 Bayesian design; v1 archived |
| [x] | **M1a Search** | search-engineer | A\* tests green (incl. admissibility on all 182 pairs); `bench.py` prints the comparison table | 2026-09-28 | A\* 4.90 expansions vs Dijkstra 8.00, same optimal cost |
| [x] | **M1b World** | world-engineer | map tests green; deterministic engine | 2026-09-28 | 11-step pipeline; observation + note generation |
| [x] | **M2 Game loop** | world-engineer → agent-engineer | 20 seeds run to a winner with no errors, fully offline | 2026-09-29 | Movement split around kills; task inheritance |
| [x] | **M3 LLM meetings** | agent-engineer | llm tests green; warm-cache replay byte-identical | 2026-09-29 | Gemini + cache + template fallback; `--deliberator` flag |
| [x] | **M4 Sabotage & protocols** | world-engineer → agent-engineer | S3 and S4 acceptance met; all five shock keys work | 2026-09-29 | Lights/doors/reactor; bid→commit→revoke→backup; deadlock |
| [x] | **M5 UI, scenarios, docs** | ui-engineer ∥ telemetry-engineer | All scenarios run from the CLI; viewer replays each log | 2026-09-29 | Browser-verified on all four real logs |
| [x] | **Spec review** | spec-reviewer | §5–§11, §13, §14, §16 audited | 2026-09-29 | 11 findings; engine core clean, gaps in policy detail |
| [ ] | **Final** | orchestrator | One live Gemini run; fill measured numbers | | Awaiting last policy fixes |

## Deliverables for the two presentations

| Needed for | Artefact | Status |
|---|---|---|
| P1 PEAS + Environment + algorithmic rationale | `docs/PROJECT_REFERENCE.md` §5–§8 | ✅ ready |
| P1 Algorithmic Modeling numbers | `runs/bench/search.csv` | ✅ measured — A\* 4.90 vs Dijkstra 8.00 expansions |
| P2 Interaction | S3 renegotiation log, S4 deadlock log | ✅ S3 reaches `SABOTAGE_FIXED` at t=20 |
| P2 Demo | `viewer.html` + `run_demo.py` + live shock keys | ✅ verified in browser |
| P2 Tools | Mesa 3.5.1 wrapper, README | ✅ |

## What measurement caught that review did not

Five bugs in this build were found by running the simulation and reading logs, not by reading code. Worth knowing for the viva — "how do you know it's correct?" has a concrete answer.

| Bug | How it was found | Impact if shipped |
|---|---|---|
| Movement resolved before kills | Instrumented the decision path: 31 kill decisions, 2 resolved | No bodies, no meetings, LLM layer never invoked |
| Perception rolled twice | Compared `KILL.witnesses` against observers' notes | Log names a witness whose own notes have no kill |
| `PressButton` never called | Counted events across 20 games: 0 presses | Half of all killings never became testimony |
| Dead crewmates stranded tasks | Counted incomplete tasks held by corpses: 19 | `all_tasks_done` unreachable after the first kill |
| Infinite `Report` livelock | Instrumented `decide()` during a frozen window | **20% of games ended in total paralysis** |

The last one is the sharpest example: the engine correctly *rejected* the invalid action, but rejections are filtered out of the event log, so the failure was visible only as an **absence** of events. Nothing was asserting on absence. A regression test now does.

## Notes

- **2026-09-28** — M0 closed. Re-specced to the minimal MVP: A\* as the single defended algorithm, Gemini Flash for meeting deliberation only, HTML replay viewer, renegotiation + deadlock retained as the rubric's named "Exceeds" examples.
- **2026-09-29** — Deadlock resolution was the highest-impact fix: before it, every config drove games to the 400-tick cap (20/20 draws); after, mean 108 ticks and a crew 9 / impostor 9 / draw 2 split.
- **2026-09-29** — Spec review returned 11 findings. The engine core (pipeline order, shared perception, A\* admissibility, stall clauses, §14.1 event schema) audited clean; every finding was in policy detail or config wiring.
