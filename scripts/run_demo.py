"""§16 demo: play S2 -> S3 -> S4 (both ablation settings) at a readable pace, then list logs."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amongus.sim.scenarios import SCENARIOS, build  # noqa: E402
from amongus.sim.telemetry import format_event, write_events_jsonl  # noqa: E402

_PACE_SECONDS = 0.015  # per printed line, only applied when stdout is a real terminal


def _banner(title: str) -> None:
    """Print a section banner so the audience can tell scenarios apart."""
    line = "=" * 64
    print(f"\n{line}\n{title}\n{line}")


def _play(name: str, scenario, verbosity: int = 2) -> Path:
    """Run one scenario, printing its formatted event log at a readable pace, and save it.

    `name` (not `scenario.name`) picks the output directory, so an ablation variant of the
    same named Scenario (e.g. standoff with deadlock_protocol=False) doesn't overwrite the
    other run's log.
    """
    engine, _scripted = build(scenario)
    engine.run(engine.config.max_ticks)
    paced = sys.stdout.isatty()
    for event in engine.events:
        line = format_event(event, verbosity)
        if line is not None:
            print(line)
            if paced:
                time.sleep(_PACE_SECONDS)
    return write_events_jsonl(engine.events, name, scenario.seed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the demo's one flag: which Deliberator (§9) every scenario below is built with."""
    parser = argparse.ArgumentParser(description="Play the four-part AI Among Us demo.")
    parser.add_argument(
        "--deliberator",
        choices=("template", "gemini"),
        default="template",
        help="§9 ablation switch; 'gemini' needs GEMINI_API_KEY or it falls back to template",
    )
    return parser.parse_args(argv)


def _with_deliberator(scenario, deliberator: str):
    """Return `scenario` with `deliberator` folded into its config_overrides."""
    overrides = {**scenario.config_overrides, "deliberator": deliberator}
    return replace(scenario, config_overrides=overrides)


def main(argv: list[str] | None = None) -> None:
    """Run the four-part demo and print where each resulting events.jsonl landed."""
    args = parse_args(argv)
    paths: list[Path] = []

    _banner("S2 worked_example -- deception, testimony, contradiction")
    scenario = _with_deliberator(SCENARIOS["worked_example"], args.deliberator)
    paths.append(_play("worked_example", scenario))

    _banner("S3 reactor_defect -- bid, commit, revoke, backup renegotiation")
    scenario = _with_deliberator(SCENARIOS["reactor_defect"], args.deliberator)
    paths.append(_play("reactor_defect", scenario))

    _banner("S4 standoff -- deadlock detected and broken (deadlock_protocol=True)")
    scenario = _with_deliberator(SCENARIOS["standoff"], args.deliberator)
    paths.append(_play("standoff", scenario, verbosity=1))

    _banner("S4 standoff -- same seed, deadlock_protocol=False: the stall persists")
    off = _with_deliberator(SCENARIOS["standoff"], args.deliberator)
    off = replace(off, config_overrides={**off.config_overrides, "deadlock_protocol": False})
    paths.append(_play("standoff_off", off, verbosity=1))

    _banner("Event logs (drop onto src/amongus/ui/viewer.html)")
    for path in paths:
        print(f"  {path}")

    _banner("Keyboard shocks: run runner.py interactively for l/d/r/k/b live injection")
    print("  python -m amongus.sim.runner --scenario baseline --seed 7 --verbosity 2")


if __name__ == "__main__":
    main()
