"""Offline template Deliberator: the honest ablation against Gemini (SPEC §9.4)."""

import re

from amongus.contracts import Statement
from amongus.llm import MeetingContext
from amongus.types import Note, Role

_KIND_PRIORITY = ("kill", "body", "saw", "alone")
_KILL_RE = re.compile(r"I saw (\S+) kill (\S+) in")
_VENT_RE = re.compile(r"I saw (\S+) climb out of a vent")
_BODY_RE = re.compile(r"I found (\S+)'s body")
_SAW_RE = re.compile(r"with (.+)\.$")
_NEAR_BODY_WINDOW = 10  # ticks; fixed by §9.4's wording, not a scenario tunable
_SUS_SEEN_KILL_OR_VENT = 0.9
_SUS_NEAR_BODY = 0.5
_SUS_BASELINE = 0.1
_SUS_IMPOSTOR_REPORTER = 0.6
_SUS_IMPOSTOR_PARTNER = 0.0


def _extract_names(saw_text: str) -> list[str]:
    """Reverse-parse a 'saw' note's 'with X, Y and Z.' clause into a name list."""
    match = _SAW_RE.search(saw_text)
    if match is None:
        return []
    joined = match.group(1).replace(" and ", ", ")
    return [name.strip() for name in joined.split(", ") if name.strip()]


def _pick_statement(notes: list[Note]) -> str:
    """Render the highest-priority note (kill > body > saw > alone) as speech."""
    by_kind: dict[str, Note] = {}
    for note in notes:
        by_kind[note.kind] = note  # last one wins: most recent of that kind
    for kind in _KIND_PRIORITY:
        if kind in by_kind:
            return by_kind[kind].text
    return "I have nothing to report."


def _crewmate_suspicion(ctx: MeetingContext) -> dict[str, float]:
    """0.9 for a seen killer/venter, 0.5 for an agent near a body within 10 ticks, else 0.1."""
    killers_or_venters: set[str] = set()
    body_ticks: list[int] = []
    sightings: list[tuple[int, list[str]]] = []
    for note in ctx.notes:
        if note.kind == "kill" and (m := _KILL_RE.search(note.text)):
            killers_or_venters.add(m.group(1))
        elif note.kind == "vent" and (m := _VENT_RE.search(note.text)):
            killers_or_venters.add(m.group(1))
        elif note.kind == "body":
            body_ticks.append(note.tick)
        elif note.kind == "saw":
            sightings.append((note.tick, _extract_names(note.text)))
    near_body: set[str] = set()
    for body_tick in body_ticks:
        for seen_tick, names in sightings:
            if abs(seen_tick - body_tick) <= _NEAR_BODY_WINDOW:
                near_body.update(names)
    suspicion = {}
    for agent in ctx.alive:
        if agent == ctx.self_id:
            continue
        if agent in killers_or_venters:
            suspicion[agent] = _SUS_SEEN_KILL_OR_VENT
        elif agent in near_body:
            suspicion[agent] = _SUS_NEAR_BODY
        else:
            suspicion[agent] = _SUS_BASELINE
    return suspicion


def _impostor_suspicion(ctx: MeetingContext) -> dict[str, float]:
    """0.6 for the reporter, 0.0 for the partner, 0.1 baseline otherwise."""
    suspicion = {}
    for agent in ctx.alive:
        if agent == ctx.self_id:
            continue
        if agent == ctx.partner_id:
            suspicion[agent] = _SUS_IMPOSTOR_PARTNER
        elif agent == ctx.reporter:
            suspicion[agent] = _SUS_IMPOSTOR_REPORTER
        else:
            suspicion[agent] = _SUS_BASELINE
    return suspicion


class TemplateDeliberator:
    """Deliberator implementation with no network: offline path, test path, and ablation."""

    def __init__(self, theta_vote: float) -> None:
        """Store the minimum suspicion required to vote rather than skip."""
        self.theta_vote = theta_vote

    def speak(self, ctx: MeetingContext) -> Statement:
        """Produce a fixed-template Statement: highest-priority note, rule-based suspicion."""
        text = _pick_statement(ctx.notes)
        if ctx.role is Role.IMPOSTOR:
            suspicion = _impostor_suspicion(ctx)
        else:
            suspicion = _crewmate_suspicion(ctx)
        vote = None
        if suspicion:
            top = max(suspicion, key=lambda a: suspicion[a])
            if suspicion[top] >= self.theta_vote:
                vote = top
        return Statement(speaker=ctx.self_id, text=text, suspicion=suspicion, vote=vote)
