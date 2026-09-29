"""Tests for §6.2 step 4 kill resolution: validity, duplicate targets, witnesses."""

from dataclasses import replace

from amongus.config import SimConfig
from amongus.types import DoTask, Kill, Move, Role, TaskInstance
from amongus.world.engine import Engine
from amongus.world.state import real_task_progress

from .helpers import ScriptedPolicy


def _build(seed: int, n_players: int = 4, n_impostors: int = 2, **overrides) -> Engine:
    """Build a 4-player engine (2 impostors) with all agents idle by default."""
    config = replace(
        SimConfig(), n_players=n_players, n_impostors=n_impostors, seed=seed, **overrides
    )
    ids = list(config.colors[:n_players])
    return Engine(config, {aid: ScriptedPolicy([]) for aid in ids})


def _place(engine: Engine, agent_id: str, room: str | None) -> None:
    """Force an agent's room directly, bypassing movement."""
    engine.world.agents[agent_id].room = room


def test_kill_resolves_despite_targets_same_tick_move() -> None:
    """A ready kill resolves even though the target chose to Move away this same tick.

    Regression test for the §6.2 pipeline-ordering bug: kills must resolve against
    post-arrival rooms, before departures start, so a target cannot dodge a certain kill
    just by deciding to walk (fixed by splitting movement into arrivals/departures).
    """
    engine = _build(seed=50)
    killer = next(a for a in engine.world.agents.values() if a.role is Role.IMPOSTOR)
    target = next(a for a in engine.world.agents.values() if a.role is Role.CREWMATE)
    killer.kill_cooldown = 0
    assert killer.room == target.room == "cafeteria"  # both start co-located
    engine.policies[killer.id] = ScriptedPolicy([Kill(target=target.id)])
    engine.policies[target.id] = ScriptedPolicy([Move(to="weapons")])
    engine.tick()
    assert target.alive is False
    assert len(engine.world.bodies) == 1
    assert engine.world.bodies[0].victim == target.id
    assert engine.world.bodies[0].room == "cafeteria"
    assert target.transit is None  # dead: the Move never starts a departure
    assert any(e.type == "KILL" for e in engine.events)
    assert not any(e.type == "MOVE" and e.data["agent"] == target.id for e in engine.events)


def test_kill_rejected_cooldown() -> None:
    """A Kill is rejected while the killer's cooldown is nonzero."""
    engine = _build(seed=1)
    killer = next(a for a in engine.world.agents.values() if a.role is Role.IMPOSTOR)
    target = next(a for a in engine.world.agents.values() if a.role is Role.CREWMATE)
    killer.kill_cooldown = 5
    _place(engine, killer.id, "electrical")
    _place(engine, target.id, "electrical")
    engine.policies[killer.id] = ScriptedPolicy([Kill(target=target.id)])
    engine.tick()
    assert target.alive is True
    assert len(engine.world.bodies) == 0
    # kill_rejected is internal-only (filtered before Engine.events, not a §14.1 type).
    assert not any(e.type == "KILL" for e in engine.events)


def test_kill_rejected_target_in_transit() -> None:
    """A Kill is rejected when the target is mid-transit (room is None)."""
    engine = _build(seed=2)
    killer = next(a for a in engine.world.agents.values() if a.role is Role.IMPOSTOR)
    target = next(a for a in engine.world.agents.values() if a.role is Role.CREWMATE)
    killer.kill_cooldown = 0
    _place(engine, killer.id, "electrical")
    target.room = None
    engine.policies[killer.id] = ScriptedPolicy([Kill(target=target.id)])
    engine.tick()
    assert target.alive is True


def test_kill_rejected_target_is_partner_impostor() -> None:
    """A Kill against the partner impostor is rejected (target must be a crewmate)."""
    engine = _build(seed=3)
    impostors = [a for a in engine.world.agents.values() if a.role is Role.IMPOSTOR]
    killer, partner = impostors
    killer.kill_cooldown = 0
    _place(engine, killer.id, "electrical")
    _place(engine, partner.id, "electrical")
    engine.policies[killer.id] = ScriptedPolicy([Kill(target=partner.id)])
    engine.tick()
    assert partner.alive is True


def test_kill_rejected_killer_is_crewmate() -> None:
    """A Kill issued by a crewmate is rejected."""
    engine = _build(seed=4)
    crew = [a for a in engine.world.agents.values() if a.role is Role.CREWMATE]
    killer, target = crew[0], crew[1]
    _place(engine, killer.id, "electrical")
    _place(engine, target.id, "electrical")
    engine.policies[killer.id] = ScriptedPolicy([Kill(target=target.id)])
    engine.tick()
    assert target.alive is True


def test_kill_rejected_different_rooms() -> None:
    """A Kill is rejected when killer and target are not in the same room."""
    engine = _build(seed=5)
    killer = next(a for a in engine.world.agents.values() if a.role is Role.IMPOSTOR)
    target = next(a for a in engine.world.agents.values() if a.role is Role.CREWMATE)
    killer.kill_cooldown = 0
    _place(engine, killer.id, "electrical")
    _place(engine, target.id, "medbay")
    engine.policies[killer.id] = ScriptedPolicy([Kill(target=target.id)])
    engine.tick()
    assert target.alive is True


def test_kill_duplicate_target_resolves_first_killer_only() -> None:
    """When two impostors target the same crewmate, only the first by id order resolves."""
    engine = _build(seed=6)
    # world.agents iterates in fixed id order, so this preserves killer priority order.
    killer_a, killer_b = (a for a in engine.world.agents.values() if a.role is Role.IMPOSTOR)
    target = next(a for a in engine.world.agents.values() if a.role is Role.CREWMATE)
    killer_a.kill_cooldown = 0
    killer_b.kill_cooldown = 0
    _place(engine, killer_a.id, "electrical")
    _place(engine, killer_b.id, "electrical")
    _place(engine, target.id, "electrical")
    engine.policies[killer_a.id] = ScriptedPolicy([Kill(target=target.id)])
    engine.policies[killer_b.id] = ScriptedPolicy([Kill(target=target.id)])
    engine.tick()
    assert target.alive is False
    assert len(engine.world.bodies) == 1
    kill_events = [e for e in engine.events if e.type == "KILL"]
    assert len(kill_events) == 1
    assert kill_events[0].data["killer"] == killer_a.id
    # killer_b's duplicate is rejected internally (kill_rejected is not a §14.1 event type);
    # its cooldown staying untouched at 0 is the observable proof it never resolved a kill.
    # cooldown resets at step 4, then step 7 decrements every cooldown by one tick.
    assert killer_a.kill_cooldown == engine.config.kill_cooldown - 1
    assert killer_b.kill_cooldown == 0


def test_kill_success_creates_body_and_witness() -> None:
    """A valid Kill creates a Body, resets cooldown, and reports witnesses at p_miss=0."""
    engine = _build(seed=7, p_miss=0.0)
    killer = next(a for a in engine.world.agents.values() if a.role is Role.IMPOSTOR)
    target = next(a for a in engine.world.agents.values() if a.role is Role.CREWMATE)
    witness = next(
        a
        for a in engine.world.agents.values()
        if a.id not in (killer.id, target.id) and a.role is Role.CREWMATE
    )
    killer.kill_cooldown = 0
    for agent in (killer, target, witness):
        _place(engine, agent.id, "electrical")
    engine.policies[killer.id] = ScriptedPolicy([Kill(target=target.id)])
    engine.tick()
    assert target.alive is False
    assert engine.world.bodies[0].victim == target.id
    assert engine.world.bodies[0].room == "electrical"
    assert killer.kill_cooldown == engine.config.kill_cooldown - 1
    kill_event = next(e for e in engine.events if e.type == "KILL")
    assert witness.id in kill_event.data["witnesses"]


def test_kill_witness_invariant_matches_observation_and_notes() -> None:
    """KILL.witnesses, each bystander's local_events, and their notes must agree exactly.

    Regression test for the double-roll bug: the witness set used to be rolled once for the
    KILL event and again, independently, while building each agent's Observation — so the
    log could say green witnessed a kill while green's own notes never mentioned it. Both
    consumers must now derive from the single witness set resolved at kill time (§7).
    """
    for seed in range(15):  # many seeds/bystanders so a re-introduced double roll would show
        engine = _build(seed=100 + seed, n_players=6, n_impostors=1, p_miss=0.4)
        killer = next(a for a in engine.world.agents.values() if a.role is Role.IMPOSTOR)
        others = [a for a in engine.world.agents.values() if a.id != killer.id]
        target, *bystanders = others
        killer.kill_cooldown = 0
        for agent in (killer, target, *bystanders):
            _place(engine, agent.id, "electrical")
        engine.policies[killer.id] = ScriptedPolicy([Kill(target=target.id)])
        engine.tick()
        kill_event = next(e for e in engine.events if e.type == "KILL")
        witness_ids = set(kill_event.data["witnesses"])
        for bystander in bystanders:
            obs = engine._observations[bystander.id]
            saw_kill_in_obs = any(
                ev["kind"] == "kill" and ev["victim"] == target.id for ev in obs.local_events
            )
            saw_kill_in_notes = any(
                n.kind == "kill" and n.tick == engine.world.tick
                for n in engine.notes[bystander.id]
            )
            expected = bystander.id in witness_ids
            assert saw_kill_in_obs == expected, (seed, bystander.id)
            assert saw_kill_in_notes == expected, (seed, bystander.id)


def test_kill_task_total_duration_invariant() -> None:
    """A kill never changes the crew's total real task duration (the task_bar denominator)."""
    engine = _build(seed=70, n_players=4, n_impostors=1)
    killer = next(a for a in engine.world.agents.values() if a.role is Role.IMPOSTOR)
    victim = next(a for a in engine.world.agents.values() if a.role is Role.CREWMATE)
    _, total_before = real_task_progress(engine.world)
    killer.kill_cooldown = 0
    engine.policies[killer.id] = ScriptedPolicy([Kill(target=victim.id)])
    engine.tick()
    assert victim.alive is False
    _, total_after = real_task_progress(engine.world)
    assert total_after == total_before


def test_kill_reassigns_to_survivor_with_fewest_remaining_tasks() -> None:
    """The victim's incomplete task goes to whichever survivor holds the fewest tasks."""
    engine = _build(seed=71, n_players=4, n_impostors=1)
    killer = next(a for a in engine.world.agents.values() if a.role is Role.IMPOSTOR)
    victim, fewer, more = (
        a for a in engine.world.agents.values() if a.role is Role.CREWMATE
    )
    victim_task = victim.tasks[0]
    victim_task.progress = 1
    assert victim_task.progress < victim_task.duration  # still incomplete
    # give `more` an extra task so it strictly outnumbers `fewer`'s remaining count.
    more.tasks.append(TaskInstance(task_id="extra1", room=more.room, duration=3, progress=0))
    killer.kill_cooldown = 0
    engine.policies[killer.id] = ScriptedPolicy([Kill(target=victim.id)])
    engine.tick()
    assert victim_task not in victim.tasks
    assert victim_task in fewer.tasks
    assert victim_task not in more.tasks
    assert victim_task.progress == 1  # progress preserved, not reset


def test_kill_reassign_ties_broken_by_fixed_id_order() -> None:
    """When remaining-task counts tie, the task goes to the survivor earlier in id order."""
    engine = _build(seed=72, n_players=4, n_impostors=1)
    killer = next(a for a in engine.world.agents.values() if a.role is Role.IMPOSTOR)
    # world.agents iterates in fixed id order, so `first` precedes `second`.
    victim, first, second = (
        a for a in engine.world.agents.values() if a.role is Role.CREWMATE
    )
    victim_task = victim.tasks[0]
    killer.kill_cooldown = 0
    engine.policies[killer.id] = ScriptedPolicy([Kill(target=victim.id)])
    engine.tick()
    assert victim_task in first.tasks
    assert victim_task not in second.tasks


def test_kill_reassign_with_no_survivors_does_not_raise() -> None:
    """Reassignment is a no-op, not an error, when the kill leaves no crewmate alive."""
    engine = _build(seed=73, n_players=2, n_impostors=1)
    killer = next(a for a in engine.world.agents.values() if a.role is Role.IMPOSTOR)
    victim = next(a for a in engine.world.agents.values() if a.role is Role.CREWMATE)
    killer.kill_cooldown = 0
    engine.policies[killer.id] = ScriptedPolicy([Kill(target=victim.id)])
    engine.tick()  # must not raise
    assert victim.alive is False


def test_game_reaches_all_tasks_done_after_a_kill() -> None:
    """A game with at least one kill can still reach all_tasks_done via task inheritance."""
    config = replace(
        SimConfig(),
        n_players=4,
        n_impostors=1,
        tasks_per_crewmate=1,
        task_duration_noise=0,
        seed=74,
    )
    ids = list(config.colors[:4])
    engine = Engine(config, {aid: ScriptedPolicy([]) for aid in ids})
    killer = next(a for a in engine.world.agents.values() if a.role is Role.IMPOSTOR)
    victim = next(a for a in engine.world.agents.values() if a.role is Role.CREWMATE)
    assert killer.room == victim.room  # both start in cafeteria
    killer.kill_cooldown = 0
    engine.policies[killer.id] = ScriptedPolicy([Kill(target=victim.id)])
    engine.tick()  # tick 1: the kill, and task inheritance
    assert victim.alive is False
    survivors = [a for a in engine.world.agents.values() if a.alive and a.role is Role.CREWMATE]
    assert len(survivors) == 2
    assert sum(len(a.tasks) for a in survivors) == 3  # 2 own + 1 inherited
    for agent in survivors:
        for task in list(agent.tasks):
            agent.room = task.room
            script = [DoTask(task_id=task.task_id)] * task.duration
            engine.policies[agent.id] = ScriptedPolicy(script)
            for _ in range(task.duration):
                engine.tick()
    assert engine.world.winner is Role.CREWMATE
    game_over = next(e for e in engine.events if e.type == "GAME_OVER")
    assert game_over.data["reason"] == "all_tasks_done"
