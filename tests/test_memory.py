"""Tests for AgentMemory: notes, last_seen, and the §8.4 replan triggers."""

from amongus.agents.memory import AgentMemory
from amongus.search.astar import AStarPlanner
from amongus.search.costs import plain_cost_fn
from amongus.types import Note
from amongus.world.map import GraphView
from amongus.world.observation import Observation


def _obs(
    room: str | None,
    tick: int = 1,
    occupants: list[str] | None = None,
    bodies_here: list[str] | None = None,
    closed_doors: frozenset = frozenset(),
    sabotage_alarm=None,
) -> Observation:
    """Build a minimal Observation with only the fields AgentMemory.observe reads."""
    return Observation(
        self_id="red",
        tick=tick,
        room=room,
        self_phys=None,
        occupants=occupants or [],
        bodies_here=bodies_here or [],
        local_events=[],
        task_bar=0.0,
        sabotage_alarm=sabotage_alarm,
        closed_doors=closed_doors,
        radio=[],
        stall_flag=False,
        death_notices=[],
        partner_id=None,
        partner_room=None,
    )


def test_recent_trims_to_last_n() -> None:
    """recent(n) returns only the last n notes, in order."""
    mem = AgentMemory()
    for i in range(5):
        mem.notes.append(Note(tick=i, text=f"note {i}", kind="saw"))
    assert [n.text for n in mem.recent(2)] == ["note 3", "note 4"]


def test_observe_updates_last_seen() -> None:
    """observe() records each occupant's room and tick into last_seen."""
    mem = AgentMemory()
    mem.observe(_obs("electrical", tick=5, occupants=["blue"]))
    assert mem.last_seen["blue"] == ("electrical", 5)


def test_arrival_trigger_fires_from_in_transit() -> None:
    """Arrival fires the tick a room becomes known after being unknown (in transit)."""
    mem = AgentMemory()
    mem.observe(_obs(None, tick=1))  # in transit: no room yet
    triggers = mem.observe(_obs("storage", tick=2))
    assert "arrival" in triggers


def test_topology_trigger_on_closed_doors_change() -> None:
    """Topology fires when the set of closed doors differs from the previous tick."""
    mem = AgentMemory()
    mem.observe(_obs("storage", tick=1))
    closed = frozenset({frozenset({"storage", "admin"})})
    triggers = mem.observe(_obs("storage", tick=2, closed_doors=closed))
    assert "topology" in triggers


def test_new_goal_trigger_on_sabotage_alarm() -> None:
    """new_goal fires the tick a sabotage alarm first appears."""
    from amongus.world.observation import SabotageAlarm

    mem = AgentMemory()
    mem.observe(_obs("storage", tick=1))
    alarm = SabotageAlarm(kind="lights", panels=(), timer=10, panel_done={})
    triggers = mem.observe(_obs("storage", tick=2, sabotage_alarm=alarm))
    assert "new_goal" in triggers


def test_new_goal_trigger_on_body_seen() -> None:
    """new_goal fires the tick a body newly appears in the agent's room."""
    mem = AgentMemory()
    mem.observe(_obs("storage", tick=1))
    triggers = mem.observe(_obs("storage", tick=2, bodies_here=["blue"]))
    assert "new_goal" in triggers


def test_meeting_end_trigger_on_meeting_note() -> None:
    """meeting_end fires once a new 'meeting'-kind note has been appended to memory."""
    mem = AgentMemory()
    mem.observe(_obs("storage", tick=1))
    mem.notes.append(Note(tick=2, kind="meeting", text="Meeting 1: blue was ejected."))
    triggers = mem.observe(_obs("storage", tick=3))
    assert "meeting_end" in triggers


def test_suspicion_shift_trigger() -> None:
    """suspicion_shift fires only once L1 change since the last check exceeds 0.1."""
    mem = AgentMemory()
    mem.observe(_obs("storage", tick=1))  # baseline snapshot: empty
    mem.suspicion["blue"] = 0.05
    triggers = mem.observe(_obs("storage", tick=2))
    assert "suspicion_shift" not in triggers  # 0.05 <= 0.1
    mem.suspicion["blue"] = 0.5
    triggers = mem.observe(_obs("storage", tick=3))
    assert "suspicion_shift" in triggers


def test_route_emits_replan_with_exact_fields() -> None:
    """A fresh route() call replans and emits REPLAN with the exact §14.1 field names."""
    mem = AgentMemory()
    planner = AStarPlanner()
    view = GraphView()
    events = []

    def emit(type_: str, **fields) -> None:
        events.append((type_, fields))

    hop = mem.route("red", planner, view, plain_cost_fn, "electrical", "storage", set(), emit)
    assert hop is not None
    assert len(events) == 1
    type_, kwargs = events[0]
    assert type_ == "REPLAN"
    assert set(kwargs) == {"agent", "reason", "expanded", "path"}
    assert kwargs["agent"] == "red"
    assert kwargs["path"][0] == "storage"
    assert kwargs["path"][-1] == "electrical"


def test_route_does_not_replan_when_plan_is_stable() -> None:
    """A second route() call with no triggers and an unchanged goal does not replan again."""
    mem = AgentMemory()
    planner = AStarPlanner()
    view = GraphView()
    events = []

    def emit(type_: str, **fields) -> None:
        events.append((type_, fields))

    mem.route("red", planner, view, plain_cost_fn, "security", "storage", set(), emit)
    assert len(events) == 1
    mid_room = mem.path[1]
    assert mid_room != "security"  # still en route, not yet at the destination
    hop = mem.route("red", planner, view, plain_cost_fn, "security", mid_room, set(), emit)
    assert hop is not None
    assert len(events) == 1  # no second REPLAN


def test_route_reason_priority_prefers_topology_over_new_goal() -> None:
    """When multiple triggers fire, the reported reason follows TRIGGER_PRIORITY order."""
    mem = AgentMemory()
    planner = AStarPlanner()
    view = GraphView()
    events = []

    def emit(type_: str, **fields) -> None:
        events.append((type_, fields))

    mem.route(
        "red",
        planner,
        view,
        plain_cost_fn,
        "electrical",
        "storage",
        {"new_goal", "topology"},
        emit,
    )
    assert events[0][1]["reason"] == "topology"


def test_route_returns_none_at_destination() -> None:
    """route() returns None once the agent has already reached dest."""
    mem = AgentMemory()
    planner = AStarPlanner()
    view = GraphView()
    hop = mem.route("red", planner, view, plain_cost_fn, "storage", "storage", set(), None)
    assert hop is None
