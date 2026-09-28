# Progress

Milestones per `docs/SPEC.md` §18. A milestone closes only when `test-runner` reports green — and for M3, M4 and M5, when `spec-reviewer` has passed.

| Done | Milestone | Owner(s) | Acceptance (§18) | Date | Note |
|---|---|---|---|---|---|
| [x] | **M0 Re-spec** | orchestrator | SPEC.md v2, CLAUDE.md, agent set, DECISIONS entries | 2026-09-28 | Pivoted off the v1 Bayesian design; v1 archived |
| [ ] | **M1a Search** | search-engineer | A\* tests green (incl. admissibility on all 182 pairs); `bench.py` prints the comparison table | | |
| [ ] | **M1b World** | world-engineer | map tests green; `runner --seed 1 --ticks 20` deterministic | | |
| [ ] | **M2 Game loop** | world-engineer → agent-engineer | 20 seeds run to a winner with no errors, fully offline | | |
| [ ] | **M3 LLM meetings** | agent-engineer | S2 transcript coherent; llm tests green; warm-cache replay byte-identical; spec-review §9–§11 | | |
| [ ] | **M4 Sabotage & protocols** | world-engineer → agent-engineer | S3 and S4 acceptance met; all five shock keys work mid-run; spec-review §12–§13, §16 | | |
| [ ] | **M5 UI, scenarios, docs** | ui-engineer ∥ telemetry-engineer | All scenarios run from the CLI; viewer replays each log; demo runs start to finish; full spec review | | |

Parallel only where owned files do not overlap: **M1a ∥ M1b**, and **M5 UI ∥ telemetry**. Everything else is sequential.

## Deliverables for the two presentations

| Needed for | Artefact | Ready at |
|---|---|---|
| P1 Algorithmic Modeling | `runs/bench/search.csv` + printed table | M1a |
| P1 PEAS + Environment | `docs/PEAS.md` | M5 — **pull forward to M1 if P1 is due first** |
| P2 Interaction | S3 renegotiation log, S4 deadlock log | M4 |
| P2 Demo | `viewer.html` + `run_demo.py` + live shock keys | M5 |

## Notes

- **2026-09-28** — M0 closed. Project re-specced to the minimal MVP: A\* as the single defended algorithm, Gemini Flash for meeting deliberation only, HTML replay viewer, renegotiation + deadlock retained as the rubric's named "Exceeds" examples. Seven rubric criteria mapped to concrete artefacts above.
