"""Meeting protocol: speaking order, two-round deliberation, voting (SPEC §10).

The §12.3 reactor renegotiation protocol lives in `agents/reactor.py`.
"""

from amongus.config import SimConfig
from amongus.contracts import Deliberator, Statement
from amongus.llm import MeetingContext, build_deliberator
from amongus.llm.template import TemplateDeliberator
from amongus.types import AgentId, Note, Role


def _find_caller(engine, reason: str, meeting_no: int) -> AgentId | None:
    """Recover who reported or pressed the button, from the event this same tick already logged."""
    key = "REPORT" if reason == "report" else "BUTTON"
    for event in reversed(engine.events):
        if event.type == key and event.data.get("meeting") == meeting_no:
            return event.data["agent"]
    return None


def _partner_of(world) -> dict[AgentId, AgentId]:
    """Map each impostor id to its partner id (§7's partner_id, recomputed for the protocol)."""
    impostors = [aid for aid, a in world.agents.items() if a.role is Role.IMPOSTOR]
    if len(impostors) != 2:
        return {}
    a, b = impostors
    return {a: b, b: a}


def _agent_notes(engine, agent_id: AgentId, limit: int) -> list[Note]:
    """Pull an agent's own recent notes from its policy's memory, if it tracks one."""
    policy = engine.policies.get(agent_id)
    memory = getattr(policy, "memory", None)
    return memory.recent(limit) if memory is not None else []


def _sanitize(statement: Statement, ctx: MeetingContext) -> Statement:
    """Clamp suspicion to [0,1], drop unknown/self ids, and veto an impostor voting its partner."""
    suspicion = {
        agent: max(0.0, min(1.0, float(score)))
        for agent, score in statement.suspicion.items()
        if agent in ctx.alive and agent != ctx.self_id
    }
    vote = statement.vote
    if vote is not None and (
        vote not in ctx.alive or vote == ctx.self_id or vote == ctx.partner_id
    ):
        vote = None
    return Statement(speaker=ctx.self_id, text=statement.text, suspicion=suspicion, vote=vote)


class MeetingProtocol:
    """Callable `meeting_handler`: runs §10's two-round deliberation, then votes and ejects."""

    def __init__(self, config: SimConfig, deliberator: Deliberator | None = None) -> None:
        """Store config and the Deliberator every agent speaks through this meeting.

        With no explicit override, the Deliberator is chosen by `config.deliberator` alone
        (§3) — swapping `gemini` for `template` needs no code edit anywhere.
        """
        self.config = config
        self.deliberator = deliberator or build_deliberator(config)
        self._fallback = TemplateDeliberator(theta_vote=config.theta_vote)

    def __call__(self, engine, reason: str, victim: AgentId | None) -> None:
        """Run one full meeting: MEETING_START, two speaking rounds, VOTE, EJECT (§10)."""
        world = engine.world
        cfg = self.config
        meeting_no = world.meeting_count
        alive = [aid for aid, a in world.agents.items() if a.alive]
        engine.emit("MEETING_START", meeting=meeting_no, reason=reason, alive=list(alive))

        first = _find_caller(engine, reason, meeting_no)
        rest = [a for a in alive if a != first]
        engine.rng.shuffle(rest)
        order = ([first] if first is not None else []) + list(rest)
        partner_of = _partner_of(world)

        transcript: list[Statement] = []
        round2: dict[AgentId, Statement] = {}
        for round_idx in (1, 2):
            # Frozen at round start: a concurrent round can only see fully completed prior
            # rounds, never a same-round speaker who hasn't been resolved yet (§9.3, §10).
            round_transcript = list(transcript)
            ctxs = {
                speaker: MeetingContext(
                    self_id=speaker,
                    role=world.agents[speaker].role,
                    alive=list(alive),
                    notes=_agent_notes(engine, speaker, cfg.notes_in_prompt),
                    transcript=list(round_transcript),
                    round=round_idx,
                    partner_id=partner_of.get(speaker),
                    reporter=first,
                    meeting=meeting_no,
                )
                for speaker in order
            }
            self._prefetch_round(list(ctxs.values()))
            for speaker in order:
                ctx = ctxs[speaker]
                statement = self._speak(engine, ctx, meeting_no, round_idx)
                transcript.append(statement)
                engine.emit(
                    "STATEMENT", meeting=meeting_no, round=round_idx, agent=speaker,
                    text=statement.text,
                )
                engine.emit(
                    "SUSPICION", meeting=meeting_no, round=round_idx, agent=speaker,
                    scores=dict(statement.suspicion),
                )
                if round_idx == 2:
                    round2[speaker] = statement

        self._vote_and_eject(engine, world, alive, round2, meeting_no)

    def _prefetch_round(self, ctxs: list[MeetingContext]) -> None:
        """Best-effort: if the Deliberator can warm a whole round concurrently, let it.

        This is the seam that makes "a round costs roughly one call's latency" (§9.3) true
        without changing any correctness path: the sequential `_speak` loop right after this
        reads from the now-warm cache. Any failure here is silently absorbed — `_speak`'s
        own per-agent retry-then-fallback is what actually has to be correct.
        """
        prefetch = getattr(self.deliberator, "prefetch", None)
        if prefetch is None:
            return
        try:
            prefetch(ctxs)
        except Exception:  # noqa: BLE001 — a failed warmup must never crash a run
            pass

    def _speak(self, engine, ctx: MeetingContext, meeting_no: int, round_idx: int) -> Statement:
        """Call the Deliberator; any exception falls back to the template and logs (§9.4)."""
        try:
            statement = self.deliberator.speak(ctx)
        except Exception:
            engine.emit("LLM_PARSE_FAIL", agent=ctx.self_id, meeting=meeting_no, round=round_idx)
            statement = self._fallback.speak(ctx)
        return _sanitize(statement, ctx)

    def _vote_and_eject(self, engine, world, alive: list[AgentId], round2, meeting_no: int) -> None:
        """Tally round-2 votes, eject on a clear plurality, and store each agent's suspicion."""
        for speaker in alive:
            statement = round2.get(speaker)
            policy = engine.policies.get(speaker)
            if statement is not None and hasattr(policy, "memory"):
                policy.memory.suspicion = dict(statement.suspicion)
            target = statement.vote if statement is not None else None
            engine.emit("VOTE", meeting=meeting_no, agent=speaker, target=target)

        tally: dict[str, int] = {}
        for speaker in alive:
            statement = round2.get(speaker)
            key = statement.vote if statement is not None and statement.vote is not None else "skip"
            tally[key] = tally.get(key, 0) + 1

        target_counts = {k: c for k, c in tally.items() if k != "skip"}
        skip_count = tally.get("skip", 0)
        ejected: AgentId | None = None
        if target_counts:
            top_count = max(target_counts.values())
            top = [k for k, c in target_counts.items() if c == top_count]
            if len(top) == 1 and top_count > skip_count:
                ejected = top[0]

        was_impostor = None
        if ejected is not None:
            agent = world.agents[ejected]
            agent.alive = False
            if self.config.ejection_reveals_role:
                was_impostor = agent.role is Role.IMPOSTOR
            self._log_meeting_note(engine, alive, meeting_no, ejected, was_impostor)
        else:
            self._log_meeting_note(engine, alive, meeting_no, None, None)

        engine.emit(
            "EJECT", meeting=meeting_no, target=ejected, was_impostor=was_impostor, tally=tally
        )

    def _log_meeting_note(self, engine, alive, meeting_no, ejected, was_impostor) -> None:
        """Append the §7.1 `meeting` note to every alive agent's log and per-policy memory."""
        if ejected is not None:
            role_word = "impostor" if was_impostor else "crewmate"
            text = f"Meeting {meeting_no}: {ejected} was ejected and was an {role_word}."
        else:
            text = f"Meeting {meeting_no}: no one was ejected."
        note = Note(tick=engine.world.tick, kind="meeting", text=text)
        for agent_id in alive:
            engine.notes.setdefault(agent_id, []).append(note)
            policy = engine.policies.get(agent_id)
            memory = getattr(policy, "memory", None)
            if memory is not None:
                memory.notes.append(note)

