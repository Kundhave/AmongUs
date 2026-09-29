"""LLM deliberation package: MeetingContext plus the offline `template` Deliberator (SPEC §9)."""

from dataclasses import dataclass

from amongus.config import SimConfig
from amongus.contracts import Deliberator
from amongus.types import AgentId, Note, Role


@dataclass
class MeetingContext:
    """Everything a Deliberator needs to produce one Statement (§9.2, §10).

    `contracts.Deliberator.speak` forward-references this type by name; it lives here
    (rather than in the not-yet-built `llm/deliberate.py`) so the offline `template`
    Deliberator and the future Gemini client share one importable definition.
    """

    self_id: AgentId
    role: Role
    alive: list[AgentId]
    notes: list[Note]
    transcript: list["Statement"]  # noqa: F821 — amongus.contracts.Statement, avoids a cycle
    round: int
    partner_id: AgentId | None
    reporter: AgentId | None
    meeting: int


def build_deliberator(config: SimConfig) -> Deliberator:
    """Select the Deliberator named by `config.deliberator` (§3, §9.4) — an ablation switch.

    Import of GeminiDeliberator is local so a `template`-only run never loads the Gemini SDK.
    """
    from amongus.llm.template import TemplateDeliberator

    if config.deliberator == "gemini":
        from amongus.llm.deliberate import GeminiDeliberator

        return GeminiDeliberator(config)
    return TemplateDeliberator(theta_vote=config.theta_vote)
