"""Tests for the static map data and GraphView (SPEC §5, §17)."""

import math

import networkx as nx

from amongus.world.map import (
    EDGES,
    HEURISTIC_SCALE,
    ROOMS,
    TASK_POOL,
    VENT_PAIRS,
    GraphView,
    to_networkx,
)


def test_room_count():
    """The map has exactly 14 rooms."""
    assert len(ROOMS) == 14


def test_edge_count():
    """The map has exactly 24 edges."""
    assert len(EDGES) == 24


def test_graph_connected():
    """The map graph is fully connected, verified against networkx."""
    g = to_networkx()
    assert nx.is_connected(g)


def test_weights_at_least_one():
    """Every edge weight is at least 1 tick."""
    for _u, _v, w in EDGES:
        assert w >= 1


def test_vent_endpoints_valid():
    """Every vent pair references real rooms."""
    for a, b in VENT_PAIRS:
        assert a in ROOMS
        assert b in ROOMS


def test_task_rooms_valid():
    """Every task in the pool references a real room."""
    for _task_id, room, _duration in TASK_POOL:
        assert room in ROOMS


def test_neighbors_respects_closed_edges():
    """A closed edge is absent from both endpoints' neighbor lists."""
    open_view = GraphView()
    closed = frozenset({frozenset({"cafeteria", "weapons"})})
    closed_view = GraphView(closed_edges=closed)

    open_neighbors = dict(open_view.neighbors("cafeteria"))
    closed_neighbors = dict(closed_view.neighbors("cafeteria"))

    assert "weapons" in open_neighbors
    assert "weapons" not in closed_neighbors


def test_vents_only_when_allowed():
    """Vent-linked rooms appear as neighbors only when allow_vents=True."""
    no_vents = GraphView(allow_vents=False)
    with_vents = GraphView(allow_vents=True)

    no_vent_neighbors = dict(no_vents.neighbors("electrical"))
    vent_neighbors = dict(with_vents.neighbors("electrical"))

    assert "lower_engine" in no_vent_neighbors  # regular edge, always present
    assert "medbay" not in no_vent_neighbors  # vent-only, absent without vents
    assert "medbay" in vent_neighbors  # vent-linked, present with vents
    assert "security" in vent_neighbors  # vent-linked, present with vents


def test_neighbor_order_deterministic():
    """Repeated calls return neighbors in the same, sorted-by-name order."""
    view = GraphView(allow_vents=True)
    first = view.neighbors("storage")
    second = view.neighbors("storage")
    assert first == second
    names = [name for name, _w in first]
    assert names == sorted(names)


def test_heuristic_scale_positive_finite():
    """HEURISTIC_SCALE is a positive, finite constant computed once at import."""
    assert HEURISTIC_SCALE > 0
    assert math.isfinite(HEURISTIC_SCALE)
