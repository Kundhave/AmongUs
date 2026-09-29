"""§12.3 reactor renegotiation: commit -> defect -> revoke -> backup, no central coordinator.

Every CrewmatePolicy/ImpostorPolicy owns a ReactorNegotiator that reads and writes the same
ReactorBoard instance (the one shared "public radio" for one game), and every negotiator
calls the identical pure functions below (bid_costs, compute_assignment) over that shared,
public data. No agent ever reads another agent's private state to do this -- the wiring code
that builds all 8 policies for one game must construct one ReactorBoard and pass it to each.
"""

from dataclasses import dataclass, field
from typing import Callable

from amongus.config import SimConfig
from amongus.contracts import Planner
from amongus.search.costs import plain_cost_fn
from amongus.types import AgentId, Note, RoomId
from amongus.world.map import GraphView
from amongus.world.observation import Observation


@dataclass
class Commitment:
    """One panel's current committed holder and its ETA (§12.3 step 2)."""

    agent: AgentId
    target: RoomId
    eta: int
    revoked: bool = False
    refreshed: bool = False  # has the holder recomputed eta from its own current position?


@dataclass
class RevokeRecord:
    """A logged commitment failure, replayed into every agent's memory as a missed_commit note."""

    agent: AgentId
    target: RoomId
    eta: int
    grace: int
    backup: AgentId | None
    backup_eta: int | None


@dataclass
class ReactorBoard:
    """The shared public bulletin board of bids, commits and revokes for one reactor window.

    One instance must be shared by every policy in a game (see module docstring); each reads
    and writes it directly instead of going through a coordinator. `revoke_log` is permanent
    across the whole game (it is the record every agent's missed_commit notes are drawn
    from); `bids`/`commits` are cleared each time a fresh reactor sabotage opens.
    """

    is_open: bool = False
    panels: tuple[RoomId, ...] = ()
    bids: dict[AgentId, dict[RoomId, float]] = field(default_factory=dict)
    commits: dict[RoomId, Commitment] = field(default_factory=dict)
    revoke_log: list[RevokeRecord] = field(default_factory=list)
    roster: set[AgentId] = field(default_factory=set)
    dead: set[AgentId] = field(default_factory=set)
    revoked: set[AgentId] = field(default_factory=set)
    exhausted: bool = False  # true once no unrevoked bidder remains for some panel (§12.3)

    def open_window(self, panels: tuple[RoomId, ...]) -> None:
        """Start a fresh bidding window: clear bids/commits/revoked (revoke_log stays permanent)."""
        self.is_open = True
        self.panels = panels
        self.bids = {}
        self.commits = {}
        self.revoked = set()
        self.exhausted = False

    def expected_bidders(self) -> set[AgentId]:
        """Every registered negotiator minus agents publicly known (reported) dead (§12.3 step 1).

        `roster` is populated once per game, by every ReactorNegotiator's constructor, so it
        is the full 8-agent participant list regardless of bidding order.
        """
        return self.roster - self.dead

    def close_window(self) -> None:
        """Mark the window closed once the sabotage clears; the next reactor start reopens it."""
        self.is_open = False


def bid_costs(
    planner: Planner, view: GraphView, room: RoomId, panels: tuple[RoomId, ...]
) -> dict[RoomId, float]:
    """This agent's A* travel cost with no risk term to each panel — the BID payload (§12.3)."""
    costs: dict[RoomId, float] = {}
    for panel in panels:
        _path, cost, _stats = planner.path(view, room, panel, plain_cost_fn)
        costs[panel] = cost
    return costs


def compute_assignment(
    bids: dict[AgentId, dict[RoomId, float]], panels: tuple[RoomId, ...]
) -> dict[RoomId, AgentId] | None:
    """The pure §12.3 step-2 function: the pair (a,b) minimizing max(cost_a[P1], cost_b[P2]).

    Ties are broken by comparing (cost, a, b) lexicographically, i.e. by agent id. Every
    policy calls this same function over the same public `bids`, so every caller derives the
    identical assignment -- this is the direct, testable form of "no central coordinator".
    Returns None if fewer than two agents can be paired across the two distinct panels.
    """
    if len(panels) != 2:
        raise ValueError("compute_assignment only supports the two-panel reactor case")
    p1, p2 = panels
    agents = sorted(bids)
    best_key: tuple[float, AgentId, AgentId] | None = None
    for a in agents:
        if p1 not in bids[a]:
            continue
        for b in agents:
            if a == b or p2 not in bids[b]:
                continue
            key = (max(bids[a][p1], bids[b][p2]), a, b)
            if best_key is None or key < best_key:
                best_key = key
    if best_key is None:
        return None
    _cost, a, b = best_key
    return {p1: a, p2: b}


class ReactorNegotiator:
    """One agent's bid/commit/revoke bookkeeping against a shared ReactorBoard (§12.3)."""

    def __init__(
        self,
        agent_id: AgentId,
        config: SimConfig,
        planner: Planner,
        board: ReactorBoard,
        emit: Callable[..., None] | None,
    ) -> None:
        """Store this agent's identity, its planner and the shared board it negotiates over."""
        self.agent_id = agent_id
        self.config = config
        self.planner = planner
        self.board = board
        self.emit = emit
        self._bid_submitted = False
        self._committed = False
        self._noted = 0
        board.roster.add(agent_id)

    def step(self, obs: Observation, memory, bid_multiplier: float = 1.0) -> RoomId | None:
        """Advance this agent's negotiation state by one tick; return its committed panel.

        `_record_missed_commits` runs last, not first: a revoke this very call (via
        `_check_revokes`, below) must already be on `board.revoke_log` before this method
        returns, so the agent whose own commitment just got revoked still records its own
        missed_commit note on this same tick instead of one tick late.
        """
        result = self._negotiate(obs, bid_multiplier)
        self._record_missed_commits(memory, obs.tick)
        return result

    def _negotiate(self, obs: Observation, bid_multiplier: float) -> RoomId | None:
        """Bid, commit or check revokes, whichever this tick calls for; return the commitment."""
        self.board.dead.update(obs.death_notices)
        alarm = obs.sabotage_alarm
        if alarm is None or alarm.kind != "reactor":
            if self.board.is_open:
                self.board.close_window()
            self._bid_submitted = False
            self._committed = False
            return None
        if not self.board.is_open:
            self.board.open_window(alarm.panels)
            self._bid_submitted = False
            self._committed = False
        if obs.room is None:
            return self._my_commitment()
        if not self._bid_submitted:
            self._bid(obs, bid_multiplier)
            return self._my_commitment()
        if not self._committed:
            self._commit(obs)
        self._check_revokes(obs)
        commitment_panel = self._my_commitment()
        if commitment_panel is not None:
            self._maybe_refresh_eta(obs, commitment_panel)
            return commitment_panel
        if self.board.exhausted:
            return self._rush_target(obs)
        return None

    def _bid(self, obs: Observation, bid_multiplier: float) -> None:
        """Compute and broadcast this agent's BID; fires at most once per open window."""
        view = GraphView(closed_edges=obs.closed_doors)
        costs = bid_costs(self.planner, view, obs.room, self.board.panels)
        costs = {panel: cost * bid_multiplier for panel, cost in costs.items()}
        self.board.bids[self.agent_id] = costs
        if self.emit is not None:
            self.emit("BID", agent=self.agent_id, costs={p: round(c) for p, c in costs.items()})
        self._bid_submitted = True

    def _commit(self, obs: Observation) -> None:
        """Once every expected agent has bid, compute the shared assignment (§12.3 step 2).

        Waiting for the full roster (option (a), see module docstring) is what makes "every
        agent computes the same assignment from the same public data" literally true: a
        straggler still in transit when the alarm sounds cannot act that tick (the engine
        skips `decide()` for transiting agents), so committing as soon as any two bids exist
        would silently use a partial, order-dependent input set. `self._committed` is only
        set once an assignment is actually made, so every negotiator keeps retrying each tick
        until the roster completes.
        """
        if not self.board.expected_bidders().issubset(self.board.bids):
            return
        assignment = compute_assignment(self.board.bids, self.board.panels)
        self._committed = True
        if assignment is None:
            return
        for panel, winner in assignment.items():
            if panel in self.board.commits:
                continue
            cost = self.board.bids[winner][panel]
            eta = obs.tick + 1 + round(cost)
            self.board.commits[panel] = Commitment(agent=winner, target=panel, eta=eta)
            if self.emit is not None:
                self.emit("COMMIT", agent=winner, target=panel, eta=eta)

    def _check_revokes(self, obs: Observation) -> None:
        """Revoke any commitment past eta+grace whose panel is not done; assign a backup.

        `eta` (from `_commit`, below) is `obs.tick + 1 + cost`: the *wall-clock* tick the
        committer expects to arrive, because the engine observes at the end of a tick for use
        in the next one, so `obs.tick` always trails the wall clock by one (§6.2 step 10). The
        comparison below must use that same wall-clock "now" (`obs.tick + 1`), not the raw,
        trailing `obs.tick` -- using the raw value made every revoke fire one wall-tick later
        than `eta + eta_grace` actually calls for.
        """
        alarm = obs.sabotage_alarm
        now = obs.tick + 1
        for panel, commitment in list(self.board.commits.items()):
            if commitment.revoked:
                continue
            done = alarm.panel_done.get(panel, False) if alarm is not None else True
            if done or now <= commitment.eta + self.config.eta_grace:
                continue
            commitment.revoked = True
            self.board.revoked.add(commitment.agent)
            held_elsewhere = {
                c.agent for p, c in self.board.commits.items() if p != panel and not c.revoked
            }
            excluded = {commitment.agent} | held_elsewhere | self.board.revoked
            backup_id, backup_eta = self._pick_backup(panel, excluded, obs.tick)
            if backup_id is not None and backup_eta is not None:
                self.board.commits[panel] = Commitment(
                    agent=backup_id, target=panel, eta=backup_eta
                )
            else:
                # §12.3 fallback: the unrevoked-bidder pool is exhausted for this panel. Give
                # up on the protocol for the rest of this window rather than silently
                # re-admitting a revoked agent -- every remaining negotiator now rushes
                # whichever open panel is nearest, exactly like `commit_protocol=False`.
                del self.board.commits[panel]
                self.board.exhausted = True
            self.board.revoke_log.append(
                RevokeRecord(
                    agent=commitment.agent, target=panel, eta=commitment.eta,
                    grace=self.config.eta_grace, backup=backup_id, backup_eta=backup_eta,
                )
            )
            if self.emit is not None:
                self.emit(
                    "REVOKE", agent=commitment.agent, target=panel, eta=commitment.eta,
                    grace=self.config.eta_grace, backup=backup_id, backup_eta=backup_eta,
                )

    def _pick_backup(
        self, panel: RoomId, excluded: set[AgentId], tick: int
    ) -> tuple[AgentId | None, int | None]:
        """Return (agent, eta) of the cheapest still-eligible bidder for panel, or (None, None).

        `excluded` already covers the just-revoked agent, anyone with a live commitment on
        the *other* panel, and every agent ever revoked in this window (§12.3: "excluding
        revoked agents" -- a revocation is the protocol's own signal that an agent did not
        deliver, on *either* panel, so it is never re-admitted as a backup this window).
        """
        candidates = [
            (cost[panel], agent)
            for agent, cost in self.board.bids.items()
            if agent not in excluded and panel in cost
        ]
        if not candidates:
            return None, None
        candidates.sort(key=lambda c: (c[0], c[1]))
        cost, agent = candidates[0]
        return agent, tick + 1 + round(cost)

    def _maybe_refresh_eta(self, obs: Observation, panel: RoomId) -> None:
        """Once, replace a stale alarm-tick bid cost with this holder's real current cost.

        §12.3 step 3 says a backup "commits with a *new* ETA", not a re-derivation of a bid
        placed possibly many ticks earlier from a room the agent has since left (the initial,
        non-backup commit is already fresh, since it follows its own bid by ~1 tick, so
        refreshing it too is harmless -- same room, same number, no-op). The stale bid is
        still exactly right for *ranking* candidates in `_pick_backup`, per spec; only the
        deadline itself must reflect where the holder actually stands. Only the holder's own
        negotiator instance can do this (nobody else knows its current room, per §7).
        """
        commitment = self.board.commits.get(panel)
        if commitment is None or commitment.agent != self.agent_id or commitment.refreshed:
            return
        view = GraphView(closed_edges=obs.closed_doors)
        _path, cost, _stats = self.planner.path(view, obs.room, panel, plain_cost_fn)
        new_eta = obs.tick + 1 + round(cost)
        commitment.refreshed = True
        if new_eta == commitment.eta:
            return  # already accurate (e.g. the original, still-fresh commit) -- no re-emit
        commitment.eta = new_eta
        if self.emit is not None:
            self.emit("COMMIT", agent=self.agent_id, target=panel, eta=new_eta)

    def _rush_target(self, obs: Observation) -> RoomId | None:
        """§12.3 exhaustion fallback: behave like `commit_protocol=False` for this window.

        Once `_check_revokes` finds no unrevoked bidder left for some panel, the protocol has
        nothing left to offer; every negotiator instead heads for whichever open panel is
        nearest to it right now, same as the ablation.
        """
        alarm = obs.sabotage_alarm
        if alarm is None:
            return None
        view = GraphView(closed_edges=obs.closed_doors)
        best, best_cost = None, float("inf")
        for panel in self.board.panels:
            if alarm.panel_done.get(panel, False):
                continue
            _path, cost, _stats = self.planner.path(view, obs.room, panel, plain_cost_fn)
            if cost < best_cost:
                best, best_cost = panel, cost
        return best

    def _my_commitment(self) -> RoomId | None:
        """Return the panel this agent currently, actively holds a commitment for, if any."""
        for panel, commitment in self.board.commits.items():
            if commitment.agent == self.agent_id and not commitment.revoked:
                return panel
        return None

    def _record_missed_commits(self, memory, tick: int) -> None:
        """Append any not-yet-seen RevokeRecord as a missed_commit Note (§7.1, §12.3 step 4)."""
        while self._noted < len(self.board.revoke_log):
            record = self.board.revoke_log[self._noted]
            self._noted += 1
            memory.notes.append(
                Note(
                    tick=tick, kind="missed_commit",
                    text=(
                        f"t={tick} {record.agent} committed to fix {record.target} "
                        f"by t={record.eta} and never did."
                    ),
                )
            )
