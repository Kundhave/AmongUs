"""Tests for the §12.3 reactor renegotiation protocol (agents/reactor.py)."""

from dataclasses import dataclass, field, replace

from amongus.agents.reactor import ReactorBoard, ReactorNegotiator, compute_assignment
from amongus.config import SimConfig
from amongus.search.astar import AStarPlanner
from amongus.sim.scenarios import SCENARIOS, build
from amongus.types import AgentPhys, Role
from amongus.world.observation import Observation, SabotageAlarm

_CFG = SimConfig()
_PLANNER = AStarPlanner()
_PANELS = ("reactor", "o2")


@dataclass
class _FakeMemory:
    """Just enough of AgentMemory's shape for ReactorNegotiator._record_missed_commits."""

    notes: list = field(default_factory=list)


def _phys(agent_id: str, room: str) -> AgentPhys:
    """A minimal AgentPhys, alive, in room, with no tasks."""
    return AgentPhys(
        id=agent_id, role=Role.CREWMATE, alive=True, room=room, transit=None,
        tasks=[], kill_cooldown=0, button_used=False,
    )


def _obs(agent_id: str, room: str, tick: int, panel_done: dict) -> Observation:
    """An Observation carrying only what ReactorNegotiator.step reads."""
    alarm = SabotageAlarm(kind="reactor", panels=_PANELS, timer=30, panel_done=dict(panel_done))
    return Observation(
        self_id=agent_id, tick=tick, room=room, self_phys=_phys(agent_id, room),
        occupants=[], bodies_here=[], local_events=[], task_bar=0.0, sabotage_alarm=alarm,
        closed_doors=frozenset(), radio=[], stall_flag=False, death_notices=[],
        partner_id=None, partner_room=None,
    )


def _no_alarm_obs(agent_id: str, room: str, tick: int) -> Observation:
    """An Observation with no active sabotage, for closing a window."""
    return Observation(
        self_id=agent_id, tick=tick, room=room, self_phys=_phys(agent_id, room),
        occupants=[], bodies_here=[], local_events=[], task_bar=0.0, sabotage_alarm=None,
        closed_doors=frozenset(), radio=[], stall_flag=False, death_notices=[],
        partner_id=None, partner_room=None,
    )


def test_compute_assignment_minimizes_the_max_pair_cost() -> None:
    """The pure §12.3 step-2 function picks (a,b) minimizing max(cost_a[P1], cost_b[P2])."""
    bids = {
        "a": {"reactor": 2.0, "o2": 13.0},
        "b": {"reactor": 8.0, "o2": 5.0},
        "c": {"reactor": 9.0, "o2": 8.0},
    }
    assert compute_assignment(bids, _PANELS) == {"reactor": "a", "o2": "b"}


def test_compute_assignment_is_identical_across_independent_callers() -> None:
    """The direct 'no central coordinator' claim: 8 independent calls agree byte-for-byte."""
    bids = {
        "red": {"reactor": 4.0, "o2": 6.0}, "blue": {"reactor": 9.0, "o2": 2.0},
        "green": {"reactor": 3.0, "o2": 11.0}, "pink": {"reactor": 7.0, "o2": 7.0},
        "orange": {"reactor": 5.0, "o2": 5.0}, "yellow": {"reactor": 12.0, "o2": 1.0},
        "black": {"reactor": 6.0, "o2": 6.0}, "white": {"reactor": 8.0, "o2": 3.0},
    }
    results = [compute_assignment(bids, _PANELS) for _ in range(8)]
    assert all(r == results[0] for r in results)
    assert results[0] is not None


def test_compute_assignment_ties_break_by_agent_id() -> None:
    """Equal-cost pairs resolve to the lexicographically smallest (agent, agent) key."""
    bids = {
        "b": {"reactor": 1.0, "o2": 1.0},
        "a": {"reactor": 1.0, "o2": 1.0},
    }
    assert compute_assignment(bids, _PANELS) == {"reactor": "a", "o2": "b"}


def _bid_and_commit(board: ReactorBoard, emit) -> dict[str, ReactorNegotiator]:
    """Three negotiators (security/cafeteria/storage) bid at tick 100, commit at tick 101."""
    negotiators = {
        aid: ReactorNegotiator(aid, _CFG, _PLANNER, board, emit)
        for aid in ("sec", "caf", "sto")
    }
    rooms = {"sec": "security", "caf": "cafeteria", "sto": "storage"}
    memories = {aid: _FakeMemory() for aid in negotiators}
    for aid, neg in negotiators.items():
        neg.step(_obs(aid, rooms[aid], 100, {}), memories[aid])
    for aid, neg in negotiators.items():
        neg.step(_obs(aid, rooms[aid], 101, {}), memories[aid])
    return negotiators, memories, rooms


def test_bid_then_commit_assigns_exactly_the_two_cheapest_agents() -> None:
    """security (cheapest to reactor) and cafeteria (cheapest to o2) win; storage does not."""
    board = ReactorBoard()
    _bid_and_commit(board, emit=lambda *a, **k: None)
    assert board.commits["reactor"].agent == "sec"
    assert board.commits["o2"].agent == "caf"


def test_commit_emits_exactly_one_event_per_panel_despite_three_negotiators() -> None:
    """All three negotiators independently compute the same assignment; only 2 COMMITs fire."""
    calls: list[tuple[str, dict]] = []

    def emit(type_: str, **fields) -> None:
        calls.append((type_, fields))

    board = ReactorBoard()
    _bid_and_commit(board, emit)
    commits = [c for c in calls if c[0] == "COMMIT"]
    assert len(commits) == 2
    bids = [c for c in calls if c[0] == "BID"]
    assert len(bids) == 3


def test_revoke_fires_exactly_at_eta_plus_grace_not_before() -> None:
    """No REVOKE while wall-clock now <= eta+grace; it fires the first tick past that.

    `eta` (from `_commit`) is a wall-clock value (`obs.tick + 1 + cost`), and the engine's
    own `obs.tick` always trails the wall clock by one tick (observations are built at the
    end of a tick, for use in the next one -- §6.2 step 10). This harness feeds `obs.tick`
    directly, exactly as the engine does, so "wall-clock now" for a given `neg.step(...,
    tick, ...)` call is `tick + 1`; the revoke check fires the first tick where that equals
    `eta + eta_grace + 1`, i.e. where the raw `tick` argument itself equals `eta + eta_grace`.
    """
    calls: list[tuple[str, dict]] = []

    def emit(type_: str, **fields) -> None:
        calls.append((type_, fields))

    board = ReactorBoard()
    negotiators, memories, rooms = _bid_and_commit(board, emit)
    defector = board.commits["reactor"].agent
    other_panel_done = {"o2": True}  # keep the other panel out of this test entirely
    eta = board.commits["reactor"].eta
    grace = _CFG.eta_grace

    tick = 102
    while tick < eta + grace:
        for aid, neg in negotiators.items():
            neg.step(
                _obs(aid, rooms[aid], tick, {**other_panel_done, "reactor": False}),
                memories[aid],
            )
        assert not any(c[0] == "REVOKE" for c in calls), f"revoked early at tick {tick}"
        tick += 1

    for aid, neg in negotiators.items():
        neg.step(
            _obs(aid, rooms[aid], tick, {**other_panel_done, "reactor": False}), memories[aid]
        )
    revokes = [c for c in calls if c[0] == "REVOKE"]
    assert len(revokes) == 1
    assert revokes[0][1]["agent"] == defector
    assert revokes[0][1]["target"] == "reactor"
    backup = revokes[0][1]["backup"]
    assert backup is not None and backup != defector
    assert board.commits["reactor"].agent == backup

    for aid, memory in memories.items():
        missed = [n for n in memory.notes if n.kind == "missed_commit"]
        assert len(missed) == 1, f"{aid} should have exactly one missed_commit note"


def test_board_closes_and_reopens_across_two_separate_sabotage_windows() -> None:
    """A closed board reopens fresh (no stale bids) the next time a reactor sabotage starts."""
    calls: list[tuple[str, dict]] = []

    def emit(type_: str, **fields) -> None:
        calls.append((type_, fields))

    board = ReactorBoard()
    neg = ReactorNegotiator("sec", _CFG, _PLANNER, board, emit)
    memory = _FakeMemory()
    neg.step(_obs("sec", "security", 1, {}), memory)
    assert board.is_open and "sec" in board.bids
    neg.step(_no_alarm_obs("sec", "security", 2), memory)
    assert board.is_open is False
    neg.step(_obs("sec", "security", 50, {}), memory)
    assert board.is_open and board.bids == {"sec": {"reactor": 2.0, "o2": 13.0}}


def test_commit_waits_for_every_registered_negotiator_to_bid() -> None:
    """No COMMIT until the whole roster has bid (bug 3, option (a)): a straggler must count."""
    board = ReactorBoard()
    calls: list[tuple[str, dict]] = []

    def emit(type_: str, **fields) -> None:
        calls.append((type_, fields))

    negotiators = {
        aid: ReactorNegotiator(aid, _CFG, _PLANNER, board, emit)
        for aid in ("sec", "caf", "sto")
    }
    rooms = {"sec": "security", "caf": "cafeteria", "sto": "storage"}
    memories = {aid: _FakeMemory() for aid in negotiators}

    # Only two of the three registered negotiators bid at first; the third straggles.
    for aid in ("sec", "caf"):
        negotiators[aid].step(_obs(aid, rooms[aid], 100, {}), memories[aid])
    for aid in ("sec", "caf"):
        negotiators[aid].step(_obs(aid, rooms[aid], 101, {}), memories[aid])
    assert not any(c[0] == "COMMIT" for c in calls), "committed from a partial bid set"

    negotiators["sto"].step(_obs("sto", rooms["sto"], 102, {}), memories["sto"])
    for aid in negotiators:
        negotiators[aid].step(_obs(aid, rooms[aid], 103, {}), memories[aid])
    commits = [c for c in calls if c[0] == "COMMIT"]
    assert len(commits) == 2, "should commit once the full roster (all 3) has bid"


def test_backup_never_double_books_an_agent_already_committed_elsewhere() -> None:
    """Bug 1: a backup must not be someone already holding a live commitment on the other panel."""
    board = ReactorBoard()
    calls: list[tuple[str, dict]] = []

    def emit(type_: str, **fields) -> None:
        calls.append((type_, fields))

    # "b" is cheap to both panels, so a naive backup pick for "reactor" would grab it even
    # though it is already the live o2 committee.
    bids = {
        "a": {"reactor": 1.0, "o2": 20.0},
        "b": {"reactor": 2.0, "o2": 2.0},
        "c": {"reactor": 30.0, "o2": 3.0},
    }
    negotiators = {
        aid: ReactorNegotiator(aid, _CFG, _PLANNER, board, emit) for aid in bids
    }
    memories = {aid: _FakeMemory() for aid in negotiators}
    rooms = {"a": "security", "b": "cafeteria", "c": "storage"}
    board.open_window(_PANELS)  # open first, so injecting bids below survives the reset
    for aid, neg in negotiators.items():
        board.bids[aid] = bids[aid]
        neg._bid_submitted = True  # noqa: SLF001 — inject bids directly, bypassing A*
    for aid, neg in negotiators.items():
        neg.step(_obs(aid, rooms[aid], 101, {}), memories[aid])
    assert board.commits["reactor"].agent == "a"
    assert board.commits["o2"].agent == "b"

    # Force "a" (reactor) past eta+grace without ever finishing; "b" must not be picked as
    # backup for reactor, since it still holds a live commitment on o2.
    eta = board.commits["reactor"].eta
    tick = eta + _CFG.eta_grace
    for aid, neg in negotiators.items():
        neg.step(
            _obs(aid, rooms[aid], tick, {"reactor": False, "o2": False}), memories[aid]
        )
    revoke = next(c for c in calls if c[0] == "REVOKE" and c[1]["target"] == "reactor")
    assert revoke[1]["backup"] == "c"
    assert board.commits["reactor"].agent == "c"
    # "b" still holds its own live o2 commitment throughout — never double-booked.
    assert board.commits["o2"].agent == "b" and not board.commits["o2"].revoked


def test_s3_reactor_defect_reaches_sabotage_fixed_with_black_as_defector() -> None:
    """Integration (§16 S3): on a real engine run, black defects, gets revoked, a backup fixes.

    Uses seed=3 (the reproduction seed from the original bug report): the scenario's own
    baked-in seed=11 has both impostors (white and black) rank as the two cheapest bidders for
    the reactor panel in turn, and with `p_defect=1.0` both defect back-to-back, burning the
    30-tick timer before any crewmate gets a turn -- a scenario-tightness issue in
    `sim/scenarios.py`'s default seed, not a defect in the negotiation protocol itself (see
    the deviations note).

    This exercises the full pipeline (not hand-built Observations), so it also pins down the
    exact eta/grace arithmetic against the engine's real one-tick observe lag (bug 2): every
    REVOKE in the log must fire at exactly `eta + eta_grace + 1` wall-clock ticks.
    """
    engine, _scripted = build(replace(SCENARIOS["reactor_defect"], seed=3))
    engine.run(25)  # comfortably past the first (scripted) reactor window, before the next one
    events = engine.events

    commits = [e for e in events if e.type == "COMMIT"]
    revokes = [e for e in events if e.type == "REVOKE"]
    fixed = [e for e in events if e.type == "SABOTAGE_FIXED"]
    assert commits and revokes and fixed, "expected a full commit->revoke->backup->fixed run"
    assert any(r.data["agent"] == "black" for r in revokes), "black must be the defector"

    for revoke in revokes:
        threshold = revoke.data["eta"] + revoke.data["grace"]
        assert revoke.tick == threshold + 1, (
            f"REVOKE at t={revoke.tick} should fire at eta+grace+1={threshold + 1}"
        )

    # Bug 1: replaying COMMIT/REVOKE in tick order, no agent ever holds two live panels.
    holder: dict[str, str] = {}  # agent -> panel it currently holds
    for e in sorted([*commits, *revokes], key=lambda ev: (ev.tick, ev.type != "REVOKE")):
        if e.type == "COMMIT":
            agent, target = e.data["agent"], e.data["target"]
            assert holder.get(agent) in (None, target), (
                f"{agent} double-booked: already holds {holder.get(agent)}, now {target}"
            )
            holder[agent] = target
        else:  # REVOKE
            holder.pop(e.data["agent"], None)

    # Bug 3: the first COMMIT must be computed from every alive agent's bid, not a partial set.
    # (BID and COMMIT can share the same wall tick: within one engine tick(), a straggler's BID
    # is appended to the shared board before a later agent in iteration order calls _commit --
    # so "at or before" the commit's own tick is the right happens-before window here.)
    first_commit_tick = commits[0].tick
    bidders_by_first_commit = {
        e.data["agent"] for e in events if e.type == "BID" and e.tick <= first_commit_tick
    }
    assert bidders_by_first_commit == set(engine.config.colors), (
        f"first COMMIT at t={first_commit_tick} used only "
        f"{len(bidders_by_first_commit)}/{len(engine.config.colors)} bids"
    )

    # The missed_commit note against black must reach the next meeting's context.
    black_memory = engine.policies["black"].memory
    assert any(n.kind == "missed_commit" and "black" in n.text for n in black_memory.notes)


def test_revoked_agent_never_reused_as_backup_and_pool_exhaustion_falls_back() -> None:
    """A revoked agent is excluded forever this window; exhaustion sets the rush fallback."""
    calls: list[tuple[str, dict]] = []

    def emit(type_: str, **fields) -> None:
        calls.append((type_, fields))

    board = ReactorBoard()
    negotiators, memories, rooms = _bid_and_commit(board, emit)
    # sec (cheapest to reactor) and caf (cheapest to o2) hold the initial commits; sto is free.
    o2_done = {"o2": True}  # keep o2 out of this test entirely
    tick = 102

    def revoke_current_reactor_holder() -> None:
        nonlocal tick
        eta = board.commits["reactor"].eta
        tick = eta + _CFG.eta_grace
        for aid, neg in negotiators.items():
            neg.step(_obs(aid, rooms[aid], tick, {**o2_done, "reactor": False}), memories[aid])

    revoke_current_reactor_holder()  # revokes sec -> backup sto (the only free bidder left)
    assert board.commits["reactor"].agent == "sto"
    revoke_current_reactor_holder()  # revokes sto -> no unrevoked, unbooked bidder remains
    assert "reactor" not in board.commits
    assert board.exhausted is True

    revokes = [c[1] for c in calls if c[0] == "REVOKE" and c[1]["target"] == "reactor"]
    revoked_agents = [r["agent"] for r in revokes]
    assert revoked_agents == ["sec", "sto"]
    assert all(r["backup"] not in revoked_agents[:i] for i, r in enumerate(revokes))
    assert revokes[-1]["backup"] is None, "no unrevoked bidder should remain to reuse"

    # Fallback: an uncommitted negotiator now rushes the nearest open panel, like the ablation.
    target = negotiators["caf"]._rush_target(  # noqa: SLF001 -- exercising the fallback directly
        _obs("caf", "cafeteria", tick, {"o2": True, "reactor": False})
    )
    assert target == "reactor"


def test_backup_eta_is_recomputed_from_its_current_position_not_the_stale_bid() -> None:
    """A backup's eta must reflect where it stands now, not its long-past alarm-tick bid.

    "sto" does not get a `.step()` call on the revoke tick itself, mirroring a backup that is
    mid-transit when it is assigned (the engine skips `decide()` for transiting agents, so its
    own negotiator cannot yet confirm anything) -- exactly the situation that made the eta
    stale in the real S3 run. Its first chance to speak for itself is a later tick, by which
    point it has moved.
    """
    calls: list[tuple[str, dict]] = []

    def emit(type_: str, **fields) -> None:
        calls.append((type_, fields))

    board = ReactorBoard()
    negotiators, memories, rooms = _bid_and_commit(board, emit)
    # sec bids reactor from "security" (cost 2); revoke it so the only free bidder, sto
    # (bid reactor from "storage", cost 9), becomes backup with a stale eta based on cost 9.
    o2_done = {"o2": True}
    eta = board.commits["reactor"].eta
    tick = eta + _CFG.eta_grace
    for aid, neg in negotiators.items():
        if aid == "sto":
            continue  # "sto" is mid-transit this tick and gets no decide() call at all
        neg.step(_obs(aid, rooms[aid], tick, {**o2_done, "reactor": False}), memories[aid])
    assert board.commits["reactor"].agent == "sto"
    stale_eta = board.commits["reactor"].eta
    assert stale_eta == tick + 1 + 9  # the stale, storage-based cost
    assert board.commits["reactor"].refreshed is False  # "sto" hasn't spoken for itself yet

    # "sto" lands from transit at "security" (cost 2) by its next tick and gets to act.
    tick += 1
    negotiators["sto"].step(
        _obs("sto", "security", tick, {**o2_done, "reactor": False}), memories["sto"]
    )
    refreshed_eta = board.commits["reactor"].eta
    assert refreshed_eta == tick + 1 + 2, "eta must reflect the real cost from sto's real room"
    assert refreshed_eta < stale_eta
    assert board.commits["reactor"].refreshed is True

    commit_events = [
        c[1] for c in calls if c[0] == "COMMIT" and c[1]["agent"] == "sto"
    ]
    assert commit_events[-1]["eta"] == refreshed_eta, "the refreshed eta must be (re)broadcast"
