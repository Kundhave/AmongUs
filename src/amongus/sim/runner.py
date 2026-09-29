"""§16 CLI: run a scenario, optionally with scripted or live keypress shock injection."""

from __future__ import annotations

import argparse
import select
import sys
from dataclasses import replace

from amongus.sim.scenarios import SCENARIOS, Scenario, build
from amongus.sim.telemetry import TerminalWriter, write_events_jsonl
from amongus.types import Action, AgentId, PressButton, Role, RoomId, Sabotage

_KEY_KINDS = {"l": "lights", "d": "doors", "r": "reactor", "k": "clear_cooldowns", "b": "meeting"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the §16 CLI flags: scenario, seed, ticks, verbosity, shocks, output dir."""
    parser = argparse.ArgumentParser(description="Run one AI Among Us scenario (SPEC §16).")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="baseline")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ticks", type=int, default=None, help="override max_ticks")
    parser.add_argument("--verbosity", type=int, default=1, choices=(0, 1, 2, 3))
    parser.add_argument("--out", default="runs", help="base directory for events.jsonl")
    parser.add_argument(
        "--shock",
        action="append",
        default=[],
        metavar="tick:kind[:room]",
        help="e.g. 60:lights or 120:doors:storage; repeatable",
    )
    parser.add_argument(
        "--no-interactive", action="store_true", help="never poll stdin for live keypresses"
    )
    parser.add_argument(
        "--deliberator",
        choices=("template", "gemini"),
        default="template",
        help="§9 ablation switch; 'gemini' needs GEMINI_API_KEY or it falls back to template",
    )
    return parser.parse_args(argv)


def _parse_shock_spec(spec: str) -> tuple[int, str, RoomId | None]:
    """Parse one `tick:kind[:room]` --shock argument."""
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"bad --shock spec {spec!r}, want tick:kind[:room]")
    tick = int(parts[0])
    kind = parts[1]
    room = parts[2] if len(parts) == 3 else None
    return tick, kind, room


def _decidable_impostor(engine) -> AgentId | None:
    """The lowest-`config.colors`-index alive, *not currently transiting* impostor.

    A transiting agent is skipped entirely at decide time (§6.2 step 1), so scheduling a
    Sabotage for one silently drops it; preferring a decidable impostor keeps a scripted or
    live shock from vanishing just because the first-choice impostor is mid-walk.
    """
    impostors = [
        aid
        for aid, a in engine.world.agents.items()
        if a.role is Role.IMPOSTOR and a.alive and a.transit is None
    ]
    if not impostors:
        return None
    return min(impostors, key=lambda a: engine.config.colors.index(a))


def _random_occupied_room(engine) -> RoomId | None:
    """Pick a uniformly random currently-occupied room, off the engine's own seeded RNG."""
    rooms = sorted({a.room for a in engine.world.agents.values() if a.alive and a.room})
    if not rooms:
        return None
    return rooms[int(engine.rng.integers(len(rooms)))]


class _ShockInjector:
    """Applies scripted and live-keypress shocks (§16.1), never blocking the tick loop."""

    def __init__(
        self, engine, scripted: dict[int, dict[AgentId, Action]], interactive: bool
    ) -> None:
        """Store the engine/scripted handles and whether stdin polling is enabled."""
        self.engine = engine
        self.scripted = scripted
        self.interactive = interactive
        self._pending_button = False
        self._pending_sabotage: tuple[str, RoomId | None] | None = None

    def poll_key(self) -> str | None:
        """Return a single pending stdin key with zero blocking, or None (§16.1)."""
        if not self.interactive:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return None
        line = sys.stdin.readline()
        return line.strip()[:1] or None

    def apply(self, kind: str, room: RoomId | None, tick: int) -> None:
        """Apply one shock so it takes effect at `tick`, then log the SHOCK event immediately."""
        if kind == "doors":
            room = room or _random_occupied_room(self.engine)
        if kind in ("lights", "reactor", "doors"):
            if not self._try_schedule_sabotage(kind, room, tick):
                self._pending_sabotage = (kind, room)  # retried each tick until decidable
        elif kind == "clear_cooldowns":
            for a in self.engine.world.agents.values():
                if a.role is Role.IMPOSTOR:
                    a.kill_cooldown = 0
        elif kind == "meeting":
            self._pending_button = True
        self.engine.emit("SHOCK", kind=kind, room=room)

    def _try_schedule_sabotage(self, kind: str, room: RoomId | None, tick: int) -> bool:
        """Schedule a Sabotage(kind) for a decidable impostor at `tick`; False if none is free."""
        saboteur = _decidable_impostor(self.engine)
        if saboteur is None:
            return False
        self.scripted.setdefault(tick, {})[saboteur] = Sabotage(kind=kind, target=room)
        return True

    def retry_pending(self, tick: int) -> None:
        """Retry a still-pending 'force meeting' or sabotage shock once it becomes possible."""
        if self._pending_button:
            room = self.engine.config.button_room
            for aid, a in self.engine.world.agents.items():
                if a.alive and a.room == room and not a.button_used:
                    self.scripted.setdefault(tick, {})[aid] = PressButton()
                    self._pending_button = False
                    break
        if self._pending_sabotage is not None:
            kind, room = self._pending_sabotage
            if self._try_schedule_sabotage(kind, room, tick):
                self._pending_sabotage = None


def run(scenario: Scenario, shocks: list[str], verbosity: int, interactive: bool) -> object:
    """Run one scenario to completion, applying scripted and (if interactive) live shocks."""
    engine, scripted = build(scenario)
    writer = TerminalWriter(verbosity)
    writer.emit_all(engine.events)
    injector = _ShockInjector(engine, scripted, interactive)
    scheduled = sorted(_parse_shock_spec(s) for s in shocks)

    while engine.world.phase.value != "over" and engine.world.tick < engine.config.max_ticks:
        next_tick = engine.world.tick + 1
        while scheduled and scheduled[0][0] == next_tick:
            _, kind, room = scheduled.pop(0)
            injector.apply(_KEY_KINDS.get(kind, kind), room, next_tick)
        key = injector.poll_key()
        if key in _KEY_KINDS:
            injector.apply(_KEY_KINDS[key], None, next_tick)
        injector.retry_pending(next_tick)
        before = len(engine.events)
        engine.tick()
        writer.emit_all(engine.events[before:])
    return engine


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: build, run, and persist one scenario's event log."""
    args = parse_args(argv)
    scenario = SCENARIOS[args.scenario]
    overrides = dict(scenario.config_overrides)
    if args.ticks is not None:
        overrides["max_ticks"] = args.ticks
    overrides["deliberator"] = args.deliberator
    seed = args.seed if args.seed is not None else scenario.seed
    scenario = replace(scenario, seed=seed, config_overrides=overrides)

    interactive = not args.no_interactive and sys.stdin.isatty()
    engine = run(scenario, args.shock, args.verbosity, interactive)

    path = write_events_jsonl(engine.events, scenario.name, scenario.seed, base_dir=args.out)
    print(f"[{scenario.name}] wrote {len(engine.events)} events to {path}")


if __name__ == "__main__":
    main()
