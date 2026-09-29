"""Tests for the A* planner and heuristic (SPEC §8.1, §17)."""

import itertools
import math

import networkx as nx

from amongus.search.astar import AStarPlanner, astar, heuristic
from amongus.search.baselines import dijkstra
from amongus.search.costs import plain_cost_fn
from amongus.world.map import ROOMS, GraphView, to_networkx

_PAIRS = list(itertools.permutations(ROOMS, 2))
_NX_GRAPH = to_networkx()


def test_182_ordered_pairs():
    """There are exactly 182 ordered room pairs (14 * 13)."""
    assert len(_PAIRS) == 182


def test_astar_matches_networkx_dijkstra():
    """A* cost equals networkx.dijkstra_path_length on all 182 ordered pairs."""
    view = GraphView()
    for src, dst in _PAIRS:
        _path, cost, _stats = astar(view, src, dst, plain_cost_fn)
        expected = nx.dijkstra_path_length(_NX_GRAPH, src, dst)
        assert math.isclose(cost, expected), (src, dst, cost, expected)


def test_heuristic_admissible_on_all_pairs():
    """h(u, dst) <= true_cost(u, dst) for every one of the 182 ordered pairs."""
    for src, dst in _PAIRS:
        true_cost = nx.dijkstra_path_length(_NX_GRAPH, src, dst)
        assert heuristic(src, dst) <= true_cost + 1e-9, (src, dst)


def test_heuristic_zero_at_destination():
    """h(dst, dst) is exactly zero."""
    for room in ROOMS:
        assert heuristic(room, room) == 0.0


def test_astar_expands_no_more_than_dijkstra():
    """A* expansions never exceed Dijkstra's on any pair."""
    view = GraphView()
    for src, dst in _PAIRS:
        _p, _c, a_stats = astar(view, src, dst, plain_cost_fn)
        _p2, _c2, d_stats = dijkstra(view, src, dst, plain_cost_fn)
        assert a_stats.expanded <= d_stats.expanded, (src, dst)


def test_astar_deterministic_repeat_calls():
    """Two identical calls return the identical path and expansion count."""
    view = GraphView()
    p1, c1, s1 = astar(view, "storage", "navigation", plain_cost_fn)
    p2, c2, s2 = astar(view, "storage", "navigation", plain_cost_fn)
    assert p1 == p2
    assert c1 == c2
    assert s1.expanded == s2.expanded


def test_astar_src_equals_dst():
    """Zero cost, single-room path when src == dst."""
    view = GraphView()
    path, cost, stats = astar(view, "cafeteria", "cafeteria", plain_cost_fn)
    assert path == ["cafeteria"]
    assert cost == 0.0
    assert stats.expanded == 1


def test_astar_unreachable_destination():
    """Isolating dst via closed_edges makes it unreachable: empty path, infinite cost."""
    closed = frozenset(frozenset({"cafeteria", n}) for n, _w in GraphView().neighbors("cafeteria"))
    view = GraphView(closed_edges=closed)
    path, cost, _stats = astar(view, "cafeteria", "navigation", plain_cost_fn)
    assert path == []
    assert cost == math.inf


def test_astar_respects_closed_edges():
    """A closed edge is never used by A*'s returned path."""
    closed = frozenset({frozenset({"storage", "electrical"})})
    view = GraphView(closed_edges=closed)
    path, _cost, _stats = astar(view, "storage", "electrical", plain_cost_fn)
    assert ("storage", "electrical") not in zip(path, path[1:])
    assert ("electrical", "storage") not in zip(path, path[1:])


def test_astar_uses_vent_shortcut_when_allowed():
    """With allow_vents=True, A* may use a 1-tick vent hop with no equivalent regular edge."""
    no_vents = GraphView(allow_vents=False)
    with_vents = GraphView(allow_vents=True)
    # electrical <-> security is vent-only: no regular edge connects them directly.
    _p1, cost_no_vent, _s1 = astar(no_vents, "electrical", "security", plain_cost_fn)
    _p2, cost_vent, _s2 = astar(with_vents, "electrical", "security", plain_cost_fn)
    assert cost_vent < cost_no_vent
    assert cost_vent == 1.0  # direct vent pair, 1 tick


def test_astar_planner_matches_function():
    """AStarPlanner.path() delegates to astar() and returns the same result."""
    view = GraphView()
    planner = AStarPlanner()
    expected = astar(view, "medbay", "shields", plain_cost_fn)
    actual = planner.path(view, "medbay", "shields", plain_cost_fn)
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
