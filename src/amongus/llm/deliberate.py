"""GeminiDeliberator: the §9.2 prompt, strict-JSON parsing, and the one-retry seam."""

import json
import re

from amongus.config import SimConfig
from amongus.contracts import Statement
from amongus.llm import MeetingContext
from amongus.llm.client import LLMClient
from amongus.llm.template import TemplateDeliberator
from amongus.types import Role
from amongus.world.map import ROOMS

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.MULTILINE)
_OWN_KILL_RE = re.compile(r"I killed \S+ in \S+\.$")


class LLMFailure(Exception):
    """Raised when Gemini produced no usable Statement after the one allowed retry (§9.2)."""


def _rules_block(ctx: MeetingContext) -> str:
    """Block 1 (§9.2): the game, the room list, who is alive, that sight is room-local."""
    rooms = ", ".join(sorted(ROOMS))
    alive = ", ".join(sorted(ctx.alive))
    return (
        "You are a player in a social deduction game on a 14-room spaceship: "
        f"{rooms}. Players currently alive: {alive}. Each player sees only their own room "
        "and must vote to eject whoever they believe is sabotaging the crew."
    )


def _own_kills(ctx: MeetingContext) -> list[str]:
    """This impostor's own 'I killed X in Y' notes, verbatim, for the identity block."""
    return [n.text for n in ctx.notes if n.kind == "kill" and _OWN_KILL_RE.search(n.text)]


def _identity_block(ctx: MeetingContext) -> str:
    """Block 2 (§9.2): who the speaker is; impostors get the lying-but-consistent extension."""
    if ctx.role is not Role.IMPOSTOR:
        return f"You are {ctx.self_id}, a crewmate."
    lines = [f"You are {ctx.self_id}. You are an impostor."]
    if ctx.partner_id is not None:
        lines.append(f"Your partner is {ctx.partner_id}.")
    kills = _own_kills(ctx)
    if kills:
        lines.append("You did this: " + " ".join(kills))
    lines.append(
        "You must not be caught. You may lie, but your story must fit what others can check."
    )
    return " ".join(lines)


def _notes_block(ctx: MeetingContext) -> str:
    """Block 3 (§9.2): the speaker's own recent notes, verbatim, nothing else's."""
    if not ctx.notes:
        return "Your notes: you have observed nothing worth recording yet."
    return "Your notes:\n" + "\n".join(n.text for n in ctx.notes)


def _transcript_block(ctx: MeetingContext) -> str:
    """Block 4 (§9.2): every statement already made this meeting, in speaking order."""
    if not ctx.transcript:
        return "Transcript so far: no one has spoken yet."
    lines = (f"{s.speaker}: {s.text}" for s in ctx.transcript)
    return "Transcript so far:\n" + "\n".join(lines)


_RESPONSE_SPEC = (
    'Respond with strict JSON only (no markdown fences, no commentary): '
    '{"statement": "one or two sentences", "suspicion": {"<agent>": 0.0-1.0, ...}, '
    '"vote": "<agent>" or null}'
)


def build_prompt(ctx: MeetingContext) -> str:
    """Assemble the four fixed §9.2 blocks, in order, plus the strict-JSON response spec."""
    return "\n\n".join(
        [_rules_block(ctx), _identity_block(ctx), _notes_block(ctx), _transcript_block(ctx),
         _RESPONSE_SPEC]
    )


def parse_response(raw: str, ctx: MeetingContext) -> Statement:
    """Parse raw model text into a Statement (§9.2); raises ValueError on anything invalid."""
    cleaned = _FENCE_RE.sub("", raw).strip()
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")
    if not {"statement", "suspicion", "vote"} <= data.keys():
        raise ValueError("response is missing a required key")
    raw_suspicion = data["suspicion"]
    if not isinstance(raw_suspicion, dict):
        raise ValueError("suspicion is not a JSON object")
    suspicion: dict[str, float] = {}
    for agent, score in raw_suspicion.items():
        if agent not in ctx.alive or agent == ctx.self_id:
            continue  # drop unknown or dead agent ids (§9.2)
        try:
            suspicion[agent] = max(0.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            continue
    vote = data["vote"]
    if vote is not None:
        vote = str(vote)
        if vote not in ctx.alive or vote == ctx.self_id:
            vote = None
    return Statement(speaker=ctx.self_id, text=str(data["statement"]), suspicion=suspicion,
                      vote=vote)


class GeminiDeliberator:
    """Deliberator backed by Gemini Flash: cached/concurrent calls, one retry, then raises."""

    def __init__(self, config: SimConfig, client: LLMClient | None = None) -> None:
        """Build (or accept) the cached LLMClient and an offline fallback for a missing key."""
        self.config = config
        self.client = client or LLMClient(
            model=config.llm_model,
            cache_path=config.llm_cache,
            max_workers=config.llm_max_workers,
            max_attempts=config.llm_max_attempts,
            backoff_base=config.llm_backoff_base,
        )
        self._fallback = TemplateDeliberator(theta_vote=config.theta_vote)

    def speak(self, ctx: MeetingContext) -> Statement:
        """No key: behave exactly as `template`. Otherwise call, parse, retry once, else raise.

        MeetingProtocol._speak already catches any exception, emits LLM_PARSE_FAIL, and
        falls back to its own TemplateDeliberator (§9.2) — raising here reuses that seam
        instead of duplicating the fallback-and-log logic in this class too.
        """
        if not self.client.enabled:
            return self._fallback.speak(ctx)
        prompt = build_prompt(ctx)
        statement = self._try_parse(self.client.call(prompt), ctx)
        if statement is not None:
            return statement
        statement = self._try_parse(self.client.call(prompt, force_refresh=True), ctx)
        if statement is not None:
            return statement
        raise LLMFailure(f"{ctx.self_id}: no valid Gemini response after one retry")

    def prefetch(self, ctxs: list[MeetingContext]) -> None:
        """Warm the cache for a whole round concurrently (§9.3), so speak() below is instant.

        Best-effort only: any failure here is silently absorbed, because speak()'s own
        retry-then-raise path is the single source of correctness (and of LLM_PARSE_FAIL).
        This method exists purely so "a round costs roughly one call's latency" (§9.3).
        """
        if not self.client.enabled:
            return
        prompts = {ctx.self_id: build_prompt(ctx) for ctx in ctxs}
        self.client.call_many(prompts)

    def _try_parse(self, raw: str | None, ctx: MeetingContext) -> Statement | None:
        """Return the parsed Statement, or None if raw is absent or fails to parse (§9.2)."""
        if raw is None:
            return None
        try:
            return parse_response(raw, ctx)
        except (ValueError, TypeError, json.JSONDecodeError):
            return None
