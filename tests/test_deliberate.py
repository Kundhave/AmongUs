"""Tests for llm/deliberate.py: prompt safety and every §9.2 parse-failure path (SPEC §17)."""

import json

import pytest

from amongus.agents.protocol import MeetingProtocol
from amongus.config import SimConfig
from amongus.contracts import Statement
from amongus.llm import MeetingContext
from amongus.llm.client import LLMClient
from amongus.llm.deliberate import GeminiDeliberator, build_prompt, parse_response
from amongus.types import Note, Role

_ALIVE = ["red", "blue", "green", "pink"]


def _ctx(
    self_id: str = "red",
    role: Role = Role.CREWMATE,
    notes: list[Note] | None = None,
    transcript: list[Statement] | None = None,
    partner_id: str | None = None,
    alive: list[str] | None = None,
) -> MeetingContext:
    """Build a minimal MeetingContext for parse/prompt tests."""
    return MeetingContext(
        self_id=self_id,
        role=role,
        alive=alive if alive is not None else list(_ALIVE),
        notes=notes or [],
        transcript=transcript or [],
        round=1,
        partner_id=partner_id,
        reporter=None,
        meeting=1,
    )


# ---------------------------------------------------------------------------
# parse_response: table-driven failure and sanitization cases (§9.2).
# ---------------------------------------------------------------------------

_RAISING_CASES = {
    "malformed_json": "{statement: not quoted}",
    "truncated": '{"statement": "hi", "suspicion": {"blue"',
    "empty_string": "",
    "non_json_prose": "I think blue did it, honestly.",
    "json_array_not_object": '["statement", "suspicion", "vote"]',
    "missing_required_key": '{"statement": "hi", "vote": null}',
    "suspicion_not_an_object": '{"statement": "hi", "suspicion": "blue", "vote": null}',
}


@pytest.mark.parametrize("raw", _RAISING_CASES.values(), ids=_RAISING_CASES.keys())
def test_parse_response_raises_on_unusable_input(raw: str) -> None:
    """Every case in this table is unusable and must raise, never silently produce garbage."""
    with pytest.raises((ValueError, TypeError, json.JSONDecodeError)):
        parse_response(raw, _ctx())


def test_parse_response_strips_markdown_fences() -> None:
    """A fence-wrapped response (```json ... ```) parses exactly like the bare JSON inside."""
    raw = '```json\n{"statement": "hi", "suspicion": {"blue": 0.5}, "vote": "blue"}\n```'
    stmt = parse_response(raw, _ctx())
    assert stmt.text == "hi"
    assert stmt.suspicion == {"blue": 0.5}
    assert stmt.vote == "blue"


def test_parse_response_clamps_out_of_range_suspicion() -> None:
    """Suspicion values outside [0, 1] are clamped, not rejected."""
    raw = '{"statement": "hi", "suspicion": {"blue": 5.0, "green": -3.0}, "vote": null}'
    stmt = parse_response(raw, _ctx())
    assert stmt.suspicion == {"blue": 1.0, "green": 0.0}


def test_parse_response_drops_non_numeric_suspicion_values() -> None:
    """A suspicion entry that can't convert to float is dropped, not a parse failure."""
    raw = '{"statement": "hi", "suspicion": {"blue": "very much", "green": 0.4}, "vote": null}'
    stmt = parse_response(raw, _ctx())
    assert stmt.suspicion == {"green": 0.4}


def test_parse_response_drops_unknown_agent_ids() -> None:
    """A suspicion/vote entry naming an agent id that doesn't exist is dropped."""
    raw = '{"statement": "hi", "suspicion": {"purple": 0.9, "blue": 0.2}, "vote": "purple"}'
    stmt = parse_response(raw, _ctx())
    assert stmt.suspicion == {"blue": 0.2}
    assert stmt.vote is None


def test_parse_response_drops_dead_agent_ids() -> None:
    """An id that used to be alive but was ejected before this meeting is dropped too."""
    raw = '{"statement": "hi", "suspicion": {"black": 0.9, "blue": 0.2}, "vote": "black"}'
    stmt = parse_response(raw, _ctx(alive=["red", "blue"]))  # "black" no longer alive
    assert stmt.suspicion == {"blue": 0.2}
    assert stmt.vote is None


def test_parse_response_drops_self_from_suspicion_and_vote() -> None:
    """A speaker naming itself in suspicion or as its own vote is sanitized away."""
    raw = '{"statement": "hi", "suspicion": {"red": 0.9, "blue": 0.2}, "vote": "red"}'
    stmt = parse_response(raw, _ctx(self_id="red"))
    assert stmt.suspicion == {"blue": 0.2}
    assert stmt.vote is None


def test_parse_response_accepts_null_vote_as_skip() -> None:
    """vote: null is a legitimate skip, not a parse failure."""
    raw = '{"statement": "hi", "suspicion": {}, "vote": null}'
    stmt = parse_response(raw, _ctx())
    assert stmt.vote is None


def test_parse_response_coerces_non_string_vote() -> None:
    """A vote that isn't a JSON string (e.g. a bare number) is coerced, not fatal."""
    raw = '{"statement": "hi", "suspicion": {}, "vote": 42}'
    stmt = parse_response(raw, _ctx())
    assert stmt.vote is None  # "42" isn't an alive agent id, so it's dropped


# ---------------------------------------------------------------------------
# GeminiDeliberator.speak: the retry-then-fallback contract must never raise into the caller.
# ---------------------------------------------------------------------------


class _StubModels:
    """Fake google.genai `client.models`: replays a fixed queue of raw texts."""

    def __init__(self, texts: list[str]) -> None:
        """Store the scripted response queue, repeating the last entry once exhausted."""
        self._texts = texts
        self.calls = 0

    def generate_content(self, *, model, contents, config):
        """Return the next scripted text as a fake response object."""
        idx = min(self.calls, len(self._texts) - 1)
        self.calls += 1
        text = self._texts[idx]
        return type("Resp", (), {"text": text})()


class _StubGenAI:
    """Fake google.genai.Client exposing only `.models`."""

    def __init__(self, models: _StubModels) -> None:
        """Wrap the stub models object."""
        self.models = models


def _deliberator(tmp_path, texts: list[str]) -> GeminiDeliberator:
    """Build a GeminiDeliberator wired to a stub client that replays `texts`."""
    client = LLMClient(
        model="m",
        cache_path=str(tmp_path / "cache.jsonl"),
        genai_client=_StubGenAI(_StubModels(texts)),
    )
    return GeminiDeliberator(SimConfig(), client=client)


@pytest.mark.parametrize("raw", _RAISING_CASES.values(), ids=_RAISING_CASES.keys())
def test_gemini_deliberator_raises_llmfailure_after_two_bad_responses(tmp_path, raw) -> None:
    """Two consecutive unusable responses (initial + one retry) raise a caught, typed failure."""
    from amongus.llm.deliberate import LLMFailure

    deliberator = _deliberator(tmp_path, texts=[raw, raw])
    with pytest.raises(LLMFailure):
        deliberator.speak(_ctx())


def test_meeting_protocol_never_raises_and_logs_llm_parse_fail(tmp_path) -> None:
    """The full pipeline: a bad response falls back to the template and logs LLM_PARSE_FAIL."""
    from amongus.agents.memory import AgentMemory
    from amongus.world.engine import Engine
    from amongus.world.observation import Observation

    class _StubPolicy:
        def __init__(self) -> None:
            self.memory = AgentMemory()

        def decide(self, obs: Observation):
            from amongus.types import Wait

            return Wait()

    config = SimConfig(n_players=3, n_impostors=1)
    ids = list(config.colors[: config.n_players])
    deliberator = _deliberator(tmp_path, texts=["not json at all", "still not json"])
    protocol = MeetingProtocol(config, deliberator=deliberator)
    engine = Engine(config, {aid: _StubPolicy() for aid in ids})
    protocol(engine, "button", None)  # must not raise
    fails = [e for e in engine.events if e.type == "LLM_PARSE_FAIL"]
    assert len(fails) == len(ids) * config.meeting_rounds
    assert any(e.type == "EJECT" for e in engine.events)


def test_gemini_deliberator_recovers_on_a_good_retry(tmp_path) -> None:
    """A bad first response followed by a good retry succeeds without falling back."""
    good = '{"statement": "it was blue", "suspicion": {"blue": 0.8}, "vote": "blue"}'
    deliberator = _deliberator(tmp_path, texts=["garbage", good])
    stmt = deliberator.speak(_ctx())
    assert stmt.text == "it was blue"
    assert stmt.vote == "blue"


# ---------------------------------------------------------------------------
# Prompt safety: no API key, no other agent's notes, no role beyond the speaker's own.
# ---------------------------------------------------------------------------


def test_prompt_never_contains_the_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even with a real-looking key in the environment, build_prompt never reads or emits it."""
    monkeypatch.setenv("GEMINI_API_KEY", "sk-super-secret-do-not-leak-12345")
    ctx = _ctx(notes=[Note(tick=1, kind="saw", text="t=1 I was in cafeteria with blue.")])
    prompt = build_prompt(ctx)
    assert "sk-super-secret-do-not-leak-12345" not in prompt


def test_prompt_contains_only_the_speakers_own_notes() -> None:
    """Another agent's private note text never leaks into this speaker's prompt."""
    own_notes = [Note(tick=5, kind="saw", text="t=5 I was in medbay with green.")]
    ctx = _ctx(self_id="red", notes=own_notes)
    prompt = build_prompt(ctx)
    assert "I was in medbay with green" in prompt
    assert "a private note only blue should ever see" not in prompt


def test_prompt_reveals_no_role_beyond_the_speakers_own_for_a_crewmate() -> None:
    """A crewmate's prompt never states any other agent (or itself) is an impostor."""
    ctx = _ctx(self_id="red", role=Role.CREWMATE)
    prompt = build_prompt(ctx)
    assert "impostor" not in prompt.lower()


def test_prompt_reveals_only_the_impostor_and_its_partner_not_a_third_agent() -> None:
    """An impostor's prompt names its own role and its partner's identity, nothing more."""
    ctx = _ctx(self_id="red", role=Role.IMPOSTOR, partner_id="blue")
    prompt = build_prompt(ctx)
    assert "You are red. You are an impostor." in prompt
    assert "Your partner is blue." in prompt
    for other in ("green", "pink"):
        assert f"{other} is an impostor" not in prompt
        assert f"{other} is a crewmate" not in prompt
