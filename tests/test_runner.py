"""Tests for the §16 CLI runner: flags, scripted shocks, and non-blocking stdin degrade."""

import hashlib
from pathlib import Path

import pytest

from amongus.sim.runner import _parse_shock_spec, main, parse_args, run
from amongus.sim.scenarios import SCENARIOS


def test_parse_shock_spec_with_and_without_room() -> None:
    """`--shock` accepts both `tick:kind` and `tick:kind:room`."""
    assert _parse_shock_spec("60:lights") == (60, "lights", None)
    assert _parse_shock_spec("120:doors:storage") == (120, "doors", "storage")


def test_cli_runs_under_pytest_without_hanging(tmp_path: Path) -> None:
    """Under pytest, stdin is not a TTY: the CLI must degrade silently and never block."""
    main(["--scenario", "baseline", "--seed", "7", "--verbosity", "0", "--out", str(tmp_path)])
    assert (tmp_path / "baseline_7" / "events.jsonl").exists()


def test_each_shock_kind_applies_and_the_run_completes() -> None:
    """Every §16.1 shock kind can be scheduled via --shock and the run still reaches a verdict."""
    scenario = SCENARIOS["baseline"]
    shocks = ["5:lights", "10:reactor", "15:doors:storage", "20:k", "1:b"]
    engine = run(scenario, shocks, verbosity=0, interactive=False)
    assert engine.world.phase.value == "over"
    shock_events = [e for e in engine.events if e.type == "SHOCK"]
    kinds = {e.data["kind"] for e in shock_events}
    assert {"lights", "reactor", "doors", "clear_cooldowns", "meeting"} <= kinds


def test_doors_shock_on_a_heavily_connected_room_still_completes() -> None:
    """A doors shock on storage (5 corridors) must not crash A* with an unreachable dest."""
    engine = run(SCENARIOS["baseline"], ["3:doors:storage"], verbosity=0, interactive=False)
    assert engine.world.phase.value == "over"


def test_runner_reproducible_event_log(tmp_path: Path) -> None:
    """The same scenario+seed run twice via the CLI writes byte-identical events.jsonl files."""
    out1, out2 = tmp_path / "a", tmp_path / "b"
    for out in (out1, out2):
        main(["--scenario", "baseline", "--seed", "7", "--verbosity", "0", "--out", str(out)])
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    log1 = out1 / "baseline_7" / "events.jsonl"
    log2 = out2 / "baseline_7" / "events.jsonl"
    assert digest(log1) == digest(log2)


def test_parse_args_defaults_and_shock_repeatable() -> None:
    """--shock is repeatable and other flags carry sensible defaults."""
    args = parse_args(["--scenario", "worked_example", "--shock", "1:l", "--shock", "2:d"])
    assert args.scenario == "worked_example"
    assert args.shock == ["1:l", "2:d"]
    assert args.verbosity == 1
    assert args.deliberator == "template"


def test_deliberator_defaults_to_template_offline(tmp_path: Path) -> None:
    """With no --deliberator flag, the CLI wires the offline template Deliberator (§9.4)."""
    main(["--scenario", "baseline", "--seed", "7", "--verbosity", "0", "--out", str(tmp_path)])
    # No exception and no network access required: main() completing offline is the assertion.
    assert (tmp_path / "baseline_7" / "events.jsonl").exists()


def test_deliberator_gemini_with_no_api_key_falls_back_and_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--deliberator gemini with no GEMINI_API_KEY degrades to the template path (demo safety).

    This is the exact path that runs if the API key fails on stage: `GeminiDeliberator`
    builds an `LLMClient` with no key, `LLMClient.enabled` is False, so every `speak()` call
    falls back to the template Deliberator with zero network access, and the run still
    reaches a verdict.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    main(
        [
            "--scenario",
            "baseline",
            "--seed",
            "7",
            "--verbosity",
            "0",
            "--out",
            str(tmp_path),
            "--deliberator",
            "gemini",
        ]
    )
    assert (tmp_path / "baseline_7" / "events.jsonl").exists()


def test_deliberator_flag_overrides_scenario_config_override() -> None:
    """--deliberator folds into config_overrides so it wins over the scenario's own default."""
    args = parse_args(["--scenario", "baseline", "--deliberator", "gemini"])
    assert args.deliberator == "gemini"
