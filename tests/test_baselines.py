"""Tests for benchmark-only baselines: BFS, Dijkstra, greedy (SPEC §8.3, §17)."""

import itertools
import math

from amongus.search.astar import astar
from amongus.search.baselines import bfs, dijkstra, greedy
from amongus.search.costs import plain_cost_fn
from amongus.world.map import ROOMS, GraphView

_PAIRS = list(itertools.permutations(ROOMS, 2))


def test_bfs_tick_cost_ge_astar_on_all_pairs():
    """BFS's hop-minimal path never costs less (in ticks) than A*'s optimal path."""
    view = GraphView()
    for src, dst in _PAIRS:
        _pa, a_cost, _sa = astar(view, src, dst, plain_cost_fn)
        _pb, b_cost, _sb = bfs(view, src, dst, plain_cost_fn)
        assert b_cost >= a_cost - 1e-9, (src, dst, a_cost, b_cost)


def test_dijkstra_matches_astar_on_all_pairs():
    """Dijkstra finds the same optimal cost as A* on every pair."""
    view = GraphView()
    for src, dst in _PAIRS:
        _pa, a_cost, _sa = astar(view, src, dst, plain_cost_fn)
        _pd, d_cost, _sd = dijkstra(view, src, dst, plain_cost_fn)
        assert math.isclose(a_cost, d_cost), (src, dst)


def test_storage_to_upper_engine_bfs_indifference():
    """storage -> upper_engine has two 2-hop routes (9 and 11 ticks); BFS is indifferent."""
    view = GraphView()
    path, cost, _stats = bfs(view, "storage", "upper_engine", plain_cost_fn)
    assert len(path) == 3  # a 2-hop route
    assert cost in (9.0, 11.0)


def test_greedy_src_equals_dst():
    """Greedy returns a zero-cost, single-room path when src == dst."""
    view = GraphView()
    path, cost, stats = greedy(view, "storage", "storage", plain_cost_fn)
    assert path == ["storage"]
    assert cost == 0.0
    assert stats.expanded == 1


def test_greedy_expands_no_more_than_dijkstra():
    """Greedy (best-first on heuristic alone) never expands more nodes than Dijkstra."""
    view = GraphView()
    for src, dst in _PAIRS:
        _pg, _cg, g_stats = greedy(view, src, dst, plain_cost_fn)
        _pd, _cd, d_stats = dijkstra(view, src, dst, plain_cost_fn)
        assert g_stats.expanded <= d_stats.expanded, (src, dst)


def test_baselines_deterministic():
    """Two identical calls to each baseline return identical paths and expansion counts."""
    view = GraphView()
    for fn in (bfs, dijkstra, greedy):
        p1, c1, s1 = fn(view, "cafeteria", "shields", plain_cost_fn)
        p2, c2, s2 = fn(view, "cafeteria", "shields", plain_cost_fn)
        assert p1 == p2
        assert c1 == c2
        assert s1.expanded == s2.expanded


def test_baselines_respect_closed_edges():
    """A closed edge is never used by any baseline's returned path."""
    closed = frozenset({frozenset({"storage", "electrical"})})
    view = GraphView(closed_edges=closed)
    for fn in (bfs, dijkstra, greedy):
        path, _cost, _stats = fn(view, "storage", "electrical", plain_cost_fn)
        assert ("storage", "electrical") not in zip(path, path[1:])
