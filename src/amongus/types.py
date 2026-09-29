"""Dataclasses and enums shared across the simulation (SPEC §4)."""

from dataclasses import dataclass
from enum import Enum

RoomId = str
AgentId = str


class Role(Enum):
    """Whether an agent is a crewmate or an impostor."""

    CREWMATE = "crewmate"
    IMPOSTOR = "impostor"


class Phase(Enum):
    """The engine's current phase."""

    PLAY = "play"
    MEETING = "meeting"
    OVER = "over"


@dataclass
class Transit:
    """An agent mid-move between two rooms."""

    src: RoomId
    dst: RoomId
    remaining: int
    via_vent: bool = False


@dataclass
class TaskInstance:
    """One task assigned to one agent."""

    task_id: str
    room: RoomId
    duration: int
    progress: int = 0

    @property
    def done(self) -> bool:
        """True once progress has reached the task's duration."""
        return self.progress >= self.duration


@dataclass
class AgentPhys:
    """An agent's physical state in the world."""

    id: AgentId
    role: Role
    alive: bool
    room: RoomId | None
    transit: Transit | None
    tasks: list[TaskInstance]
    kill_cooldown: int
    button_used: bool


@dataclass
class Body:
    """A dead agent's remains, visible until reported."""

    victim: AgentId
    room: RoomId
    tick: int
    reported: bool = False


@dataclass
class WorldState:
    """The full authoritative simulation state; never given to a policy."""

    tick: int
    phase: Phase
    agents: dict[AgentId, AgentPhys]
    bodies: list[Body]
    sabotage: "SabotageState | None"  # noqa: F821 (world/sabotage.py, future milestone)
    closed_edges: set[frozenset[RoomId]]
    task_bar: float
    last_progress_tick: int
    meeting_count: int
    winner: Role | None


# Actions — exactly one per agent per tick.


@dataclass
class Move:
    """Walk toward a neighboring room along an open edge."""

    to: RoomId


@dataclass
class Vent:
    """Impostor-only: teleport to a vent-linked room in one tick."""

    to: RoomId


@dataclass
class DoTask:
    """Advance progress on a task in the agent's current room."""

    task_id: str


@dataclass
class HoldPanel:
    """Hold a reactor panel to advance the reactor fix counter."""

    room: RoomId


@dataclass
class Kill:
    """Impostor-only: kill a crewmate in the same room."""

    target: AgentId


@dataclass
class Report:
    """Report an unreported body in the agent's room, flagging a meeting."""

    pass


@dataclass
class PressButton:
    """Call an emergency meeting from the button room."""

    pass


@dataclass
class Sabotage:
    """Start a sabotage of the given kind, optionally targeting a room."""

    kind: str
    target: RoomId | None = None


@dataclass
class Wait:
    """Do nothing this tick."""

    pass


Action = Move | Vent | DoTask | HoldPanel | Kill | Report | PressButton | Sabotage | Wait


@dataclass
class Note:
    """One plain-English memory entry generated from an observation (SPEC §7.1)."""

    tick: int
    text: str
    kind: str


@dataclass
class Event:
    """One recorded state change, emitted every tick for telemetry (SPEC §14)."""

    tick: int
    type: str
    data: dict
