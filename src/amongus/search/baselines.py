"""Benchmark-only baselines: BFS, Dijkstra, greedy — never called by an agent (SPEC §8.3)."""

import itertools
import math
from collections import deque
from heapq import heappop, heappush

from amongus.contracts import CostFn, SearchStats
from amongus.search.astar import heuristic
from amongus.types import RoomId
from amongus.world.map import GraphView


def _path_cost(view: GraphView, path: list[RoomId], cost_fn: CostFn) -> float:
    """Sum cost_fn over consecutive edges of an already-found path."""
    total = 0.0
    for u, v in zip(path, path[1:]):
        w = dict(view.neighbors(u))[v]
        total += cost_fn(u, v, w)
    return total


def bfs(
    view: GraphView, src: RoomId, dst: RoomId, cost_fn: CostFn
) -> tuple[list[RoomId], float, SearchStats]:
    """Minimise hop count; report the actual cost_fn cost of the hop-minimal path found."""
    stats = SearchStats(expanded=0, frontier_max=0)
    parent: dict[RoomId, RoomId] = {}
    visited: set[RoomId] = {src}
    queue: deque[RoomId] = deque([src])
    while queue:
        stats.frontier_max = max(stats.frontier_max, len(queue))
        u = queue.popleft()
        stats.expanded += 1
        if u == dst:
            path = [u]
            while path[-1] != src:
                path.append(parent[path[-1]])
            path.reverse()
            return path, _path_cost(view, path, cost_fn), stats
        for v, _w in view.neighbors(u):
            if v not in visited:
                visited.add(v)
                parent[v] = u
                queue.append(v)
    return [], math.inf, stats


def dijkstra(
    view: GraphView, src: RoomId, dst: RoomId, cost_fn: CostFn
) -> tuple[list[RoomId], float, SearchStats]:
    """Uniform-cost search with no heuristic; same optimal cost as A*, more expansions."""
    stats = SearchStats(expanded=0, frontier_max=0)
    counter = itertools.count()
    open_heap: list[tuple[float, int, RoomId]] = [(0.0, next(counter), src)]
    g: dict[RoomId, float] = {src: 0.0}
    parent: dict[RoomId, RoomId] = {}
    closed: set[RoomId] = set()
    while open_heap:
        stats.frontier_max = max(stats.frontier_max, len(open_heap))
        d, _, u = heappop(open_heap)
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
                heappush(open_heap, (g2, next(counter), v))
    return [], math.inf, stats


def greedy(
    view: GraphView, src: RoomId, dst: RoomId, cost_fn: CostFn
) -> tuple[list[RoomId], float, SearchStats]:
    """Best-first search on heuristic alone; expands fewest nodes but is not optimal."""
    stats = SearchStats(expanded=0, frontier_max=0)
    counter = itertools.count()
    open_heap: list[tuple[float, int, RoomId]] = [(heuristic(src, dst), next(counter), src)]
    parent: dict[RoomId, RoomId] = {}
    visited: set[RoomId] = set()
    while open_heap:
        stats.frontier_max = max(stats.frontier_max, len(open_heap))
        _h, _, u = heappop(open_heap)
        if u in visited:
            continue
        visited.add(u)
        stats.expanded += 1
        if u == dst:
            path = [u]
            while path[-1] != src:
                path.append(parent[path[-1]])
            path.reverse()
            return path, _path_cost(view, path, cost_fn), stats
        for v, _w in view.neighbors(u):
            if v not in visited:
                parent.setdefault(v, u)
                heappush(open_heap, (heuristic(v, dst), next(counter), v))
    return [], math.inf, stats
