"""Tests for llm/template.py: the honest offline ablation must never let an impostor confess."""

from amongus.llm import MeetingContext
from amongus.llm.template import TemplateDeliberator
from amongus.types import Note, Role

_ALIVE = ["red", "blue", "green", "pink"]


def _ctx(
    self_id: str = "blue",
    role: Role = Role.IMPOSTOR,
    notes: list[Note] | None = None,
    partner_id: str | None = None,
    reporter: str | None = None,
) -> MeetingContext:
    """Build a minimal MeetingContext for TemplateDeliberator.speak tests."""
    return MeetingContext(
        self_id=self_id,
        role=role,
        alive=list(_ALIVE),
        notes=notes or [],
        transcript=[],
        round=1,
        partner_id=partner_id,
        reporter=reporter,
        meeting=1,
    )


def test_impostor_never_voices_its_own_kill_note() -> None:
    """An impostor holding only a self-kill note must not confess it out loud (§12.2)."""
    notes = [Note(tick=42, kind="kill", text="t=42 I killed red in electrical.")]
    deliberator = TemplateDeliberator(theta_vote=0.35)
    stmt = deliberator.speak(_ctx(notes=notes))
    assert "I killed" not in stmt.text
    assert "I vented" not in stmt.text
    assert "red" not in stmt.text


def test_impostor_falls_back_to_its_most_recent_saw_or_alone_note() -> None:
    """With a kill note and a saw note, the impostor speaks the location claim, not the kill."""
    notes = [
        Note(tick=40, kind="saw", text="t=40 I was in cafeteria with green and pink."),
        Note(tick=42, kind="kill", text="t=42 I killed red in electrical."),
    ]
    deliberator = TemplateDeliberator(theta_vote=0.35)
    stmt = deliberator.speak(_ctx(notes=notes))
    assert stmt.text == "t=40 I was in cafeteria with green and pink."


def test_impostor_with_no_location_note_gets_a_neutral_claim_not_silence() -> None:
    """No saw/alone note at all still produces a plausible non-confessing statement."""
    notes = [Note(tick=42, kind="kill", text="t=42 I killed red in electrical.")]
    deliberator = TemplateDeliberator(theta_vote=0.35)
    stmt = deliberator.speak(_ctx(notes=notes))
    assert stmt.text
    assert "killed" not in stmt.text
    assert "vented" not in stmt.text


def test_impostor_witnessed_kill_note_is_also_withheld() -> None:
    """A witnessed (third-person) kill/vent note is withheld too, not just the self-confession."""
    notes = [
        Note(tick=40, kind="alone", text="t=40 I was alone in navigation."),
        Note(tick=44, kind="kill", text="t=44 I saw green kill pink in medbay."),
    ]
    deliberator = TemplateDeliberator(theta_vote=0.35)
    stmt = deliberator.speak(_ctx(notes=notes))
    assert stmt.text == "t=40 I was alone in navigation."


def test_crewmate_statement_is_unaffected() -> None:
    """A crewmate still uses the original kill > body > saw > alone priority order."""
    notes = [
        Note(tick=40, kind="saw", text="t=40 I was in cafeteria with green."),
        Note(tick=44, kind="kill", text="t=44 I saw blue kill red in electrical."),
    ]
    deliberator = TemplateDeliberator(theta_vote=0.35)
    stmt = deliberator.speak(_ctx(self_id="green", role=Role.CREWMATE, notes=notes))
    assert stmt.text == "t=44 I saw blue kill red in electrical."
