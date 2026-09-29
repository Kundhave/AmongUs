"""Risk-weighted edge costs for A* — pure functions of plain data (SPEC §8.2)."""

from amongus.config import SimConfig
from amongus.contracts import CostFn
from amongus.types import AgentId, RoomId


def plain_cost_fn(u: RoomId, v: RoomId, w: int) -> float:
    """Cost equal to the raw edge weight, no risk term (used by astar_no_risk)."""
    return float(w)


def make_cost_fn(
    suspicion: dict[AgentId, float],
    last_seen: dict[AgentId, tuple[RoomId, int]],
    tick: int,
    cfg: SimConfig,
) -> CostFn:
    """Build cost(u, v, w) = w + alpha_risk * risk(v), risk from recent sightings."""

    def risk(room: RoomId) -> float:
        total = 0.0
        for agent, (seen_room, seen_tick) in last_seen.items():
            if seen_room != room:
                continue
            if tick - seen_tick > cfg.last_seen_decay:
                continue
            total += suspicion.get(agent, 0.0)
        return total

    def cost(u: RoomId, v: RoomId, w: int) -> float:
        c = w + cfg.alpha_risk * risk(v)
        assert c >= 0.0, "cost must stay non-negative for A* correctness"
        return c

    return cost
