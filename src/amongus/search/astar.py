"""A* planner over the risk-weighted room graph — the algorithm to defend (SPEC §8.1)."""

import itertools
import math
from heapq import heappop, heappush

from amongus.contracts import CostFn, SearchStats
from amongus.types import RoomId
from amongus.world.map import HEURISTIC_SCALE, ROOMS, GraphView


def _euclid(u: RoomId, v: RoomId) -> float:
    """Straight-line distance between two rooms' drawing coordinates."""
    ux, uy = ROOMS[u]
    vx, vy = ROOMS[v]
    return math.hypot(ux - vx, uy - vy)


def heuristic(u: RoomId, dst: RoomId) -> float:
    """Admissible A* heuristic: euclid(u, dst) / HEURISTIC_SCALE.

    Admissibility: HEURISTIC_SCALE = s is the fastest straight-line distance any
    single edge covers per tick (max over edges of euclid(edge) / weight(edge)),
    computed once at load. No path can cover ground faster than s per tick, so the
    true remaining cost from u to dst is at least euclid(u, dst) / s = h(u). Hence h
    never overestimates and is admissible (and consistent, so nodes never reopen).
    The risk term in cost_fn (§8.2) only ever adds non-negative cost on top of edge
    weight, so this bound — derived from raw edge weights alone — still holds under
    risk-weighted costs; admissibility survives risk weighting.
    """
    return _euclid(u, dst) / HEURISTIC_SCALE


def astar(
    view: GraphView, src: RoomId, dst: RoomId, cost_fn: CostFn
) -> tuple[list[RoomId], float, SearchStats]:
    """Find the least-cost path from src to dst under cost_fn, per SPEC §8.1."""
    stats = SearchStats(expanded=0, frontier_max=0)
    counter = itertools.count()
    open_heap: list[tuple[float, int, RoomId]] = [(heuristic(src, dst), next(counter), src)]
    g: dict[RoomId, float] = {src: 0.0}
    parent: dict[RoomId, RoomId] = {}
    closed: set[RoomId] = set()

    while open_heap:
        stats.frontier_max = max(stats.frontier_max, len(open_heap))
        f, _, u = heappop(open_heap)
        if u in closed:
            continue
        closed.add(u)
        stats.expanded += 1
        if u == dst:
            path = [u]
            while path[-1] != src:
                path.append(parent[path[-1]])
            path.reverse()
            return path, g[u], stats
        for v, w in view.neighbors(u):
            g2 = g[u] + cost_fn(u, v, w)
            if g2 < g.get(v, math.inf):
                g[v] = g2
                parent[v] = u
                heappush(open_heap, (g2 + heuristic(v, dst), next(counter), v))

    return [], math.inf, stats


def all_pairs(
    view: GraphView, cost_fn: CostFn
) -> tuple[dict[RoomId, dict[RoomId, float]], dict[RoomId, dict[RoomId, RoomId | None]]]:
    """Run A* from every room to every other room; return distances and next-hop table."""
    dist: dict[RoomId, dict[RoomId, float]] = {}
    next_hop: dict[RoomId, dict[RoomId, RoomId | None]] = {}
    for src in ROOMS:
        dist[src] = {}
        next_hop[src] = {}
        for dst in ROOMS:
            path, cost, _stats = astar(view, src, dst, cost_fn)
            dist[src][dst] = cost
            next_hop[src][dst] = path[1] if len(path) > 1 else None
    return dist, next_hop


class AStarPlanner:
    """Planner protocol implementation backed by astar()."""

    def path(
        self, view: GraphView, src: RoomId, dst: RoomId, cost_fn: CostFn
    ) -> tuple[list[RoomId], float, SearchStats]:
        """Return (path, total cost, search stats) from src to dst under cost_fn."""
        return astar(view, src, dst, cost_fn)
