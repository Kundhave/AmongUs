"""Static 14-room map: rooms, edges, vents, tasks, and a read-only graph view (SPEC §5)."""

import math

import networkx as nx

from amongus.config import SimConfig
from amongus.types import RoomId

ROOMS: dict[str, tuple[float, float]] = {
    "upper_engine": (1.0, 5.0),
    "reactor": (0.0, 3.0),
    "security": (2.0, 3.0),
    "lower_engine": (1.0, 1.0),
    "medbay": (4.0, 4.0),
    "cafeteria": (6.0, 5.0),
    "weapons": (9.0, 5.0),
    "o2": (8.0, 4.0),
    "navigation": (11.0, 3.0),
    "shields": (9.0, 1.0),
    "communications": (7.0, 0.0),
    "storage": (6.0, 1.0),
    "admin": (7.0, 2.5),
    "electrical": (4.0, 1.5),
}

EDGES: list[tuple[str, str, int]] = [  # undirected, weight = travel ticks
    ("cafeteria", "weapons", 3),
    ("cafeteria", "medbay", 3),
    ("cafeteria", "upper_engine", 5),
    ("cafeteria", "storage", 4),
    ("cafeteria", "admin", 4),
    ("weapons", "o2", 2),
    ("weapons", "navigation", 4),
    ("o2", "navigation", 3),
    ("o2", "shields", 4),
    ("navigation", "shields", 4),
    ("shields", "communications", 2),
    ("shields", "storage", 4),
    ("communications", "storage", 3),
    ("storage", "admin", 2),
    ("storage", "electrical", 4),
    ("storage", "lower_engine", 6),
    ("electrical", "lower_engine", 5),
    ("lower_engine", "security", 3),
    ("lower_engine", "reactor", 3),
    ("lower_engine", "upper_engine", 5),
    ("security", "reactor", 2),
    ("security", "upper_engine", 3),
    ("reactor", "upper_engine", 3),
    ("upper_engine", "medbay", 4),
]  # 24 edges

VENT_PAIRS: list[tuple[str, str]] = [  # impostor only, 1 tick, bidirectional at load
    ("reactor", "upper_engine"),
    ("reactor", "lower_engine"),
    ("electrical", "medbay"),
    ("electrical", "security"),
    ("medbay", "security"),
    ("cafeteria", "admin"),
    ("navigation", "weapons"),
    ("navigation", "shields"),
]

TASK_POOL: list[tuple[str, str, int]] = [  # (task_id, room, duration)
    ("t01", "upper_engine", 4),
    ("t02", "lower_engine", 4),
    ("t03", "reactor", 5),
    ("t04", "security", 3),
    ("t05", "medbay", 5),
    ("t06", "cafeteria", 3),
    ("t07", "weapons", 4),
    ("t08", "o2", 3),
    ("t09", "navigation", 4),
    ("t10", "navigation", 3),
    ("t11", "shields", 4),
    ("t12", "communications", 3),
    ("t13", "storage", 4),
    ("t14", "admin", 3),
    ("t15", "electrical", 4),
    ("t16", "electrical", 5),
]

_VENT_TRAVEL_TICKS = 1  # fixed map data (SPEC §5), not a scenario tunable


def _euclid(u: str, v: str) -> float:
    """Straight-line distance between two rooms' drawing coordinates."""
    ux, uy = ROOMS[u]
    vx, vy = ROOMS[v]
    return math.hypot(ux - vx, uy - vy)


def validate_map() -> None:
    """Raise ValueError if the static map data violates any invariant in §5."""
    if len(ROOMS) != 14:
        raise ValueError(f"expected 14 rooms, got {len(ROOMS)}")
    if len(EDGES) != 24:
        raise ValueError(f"expected 24 edges, got {len(EDGES)}")
    for u, v, w in EDGES:
        if u not in ROOMS or v not in ROOMS:
            raise ValueError(f"edge references unknown room: {u}-{v}")
        if w < 1:
            raise ValueError(f"edge weight below 1: {u}-{v} = {w}")
    g = nx.Graph()
    g.add_nodes_from(ROOMS)
    g.add_weighted_edges_from(EDGES)
    if not nx.is_connected(g):
        raise ValueError("map graph is not connected")
    for a, b in VENT_PAIRS:
        if a not in ROOMS or b not in ROOMS:
            raise ValueError(f"vent references unknown room: {a}-{b}")
    for task_id, room, _duration in TASK_POOL:
        if room not in ROOMS:
            raise ValueError(f"task {task_id} references unknown room: {room}")
    cfg = SimConfig()
    if cfg.button_room not in ROOMS:
        raise ValueError(f"button_room not a real room: {cfg.button_room}")
    for panel in cfg.reactor_panels:
        if panel not in ROOMS:
            raise ValueError(f"reactor panel not a real room: {panel}")


def to_networkx() -> nx.Graph:
    """Build a weighted nx.Graph from EDGES, for use as an independent test oracle."""
    g = nx.Graph()
    g.add_nodes_from(ROOMS)
    g.add_weighted_edges_from(EDGES)
    return g


def _compute_heuristic_scale() -> float:
    """Compute s = max over edges of euclid(u, v) / weight(u, v)."""
    return max(_euclid(u, v) / w for u, v, w in EDGES)


class GraphView:
    """Read-only map view with closed_edges applied and an allow_vents flag."""

    def __init__(
        self,
        closed_edges: frozenset[frozenset[RoomId]] = frozenset(),
        allow_vents: bool = False,
    ) -> None:
        """Build a view of the static map with some edges closed and vents toggled."""
        self.closed_edges = closed_edges
        self.allow_vents = allow_vents
        self._adjacency: dict[str, dict[str, int]] = {r: {} for r in ROOMS}
        for u, v, w in EDGES:
            self._adjacency[u][v] = w
            self._adjacency[v][u] = w
        self._vent_adjacency: dict[str, dict[str, int]] = {r: {} for r in ROOMS}
        for a, b in VENT_PAIRS:
            self._vent_adjacency[a][b] = _VENT_TRAVEL_TICKS
            self._vent_adjacency[b][a] = _VENT_TRAVEL_TICKS

    def neighbors(self, room: RoomId) -> list[tuple[RoomId, int]]:
        """Return (neighbor, weight) pairs reachable from room, sorted by room name."""
        result: dict[str, int] = {}
        for v, w in self._adjacency[room].items():
            if frozenset((room, v)) in self.closed_edges:
                continue
            result[v] = w
        if self.allow_vents:
            for v, w in self._vent_adjacency[room].items():
                result[v] = min(result[v], w) if v in result else w
        return sorted(result.items())


HEURISTIC_SCALE: float = _compute_heuristic_scale()
"""Fastest straight-line distance (drawing units) any single edge covers per tick.

Used as the admissible A* heuristic scale: h(u) = euclid(u, dst) / HEURISTIC_SCALE
never overestimates true remaining cost, since no path can outrun this rate (§8.1).
"""

validate_map()
