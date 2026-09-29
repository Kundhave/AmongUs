"""Stable protocols other modules code against: Planner, Deliberator, Policy (SPEC §4)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

from amongus.types import Action, AgentId, RoomId

if TYPE_CHECKING:
    from amongus.world.map import GraphView
    # MeetingContext (§9/§10, agent-engineer) and Observation (§7, world/observation.py)
    # are defined in later milestones; forward-referenced here by name only.

# cost_fn(u, v, weight) -> non-negative edge cost, used by search (§8.2).
CostFn = Callable[[RoomId, RoomId, int], float]


@dataclass
class SearchStats:
    """Diagnostics returned alongside a search result."""

    expanded: int
    frontier_max: int


@dataclass
class Statement:
    """One agent's spoken output at a meeting round."""

    speaker: AgentId
    text: str
    suspicion: dict[AgentId, float]
    vote: AgentId | None


@runtime_checkable
class Planner(Protocol):
    """Routes an agent through the ship."""

    def path(
        self, view: "GraphView", src: RoomId, dst: RoomId, cost_fn: CostFn
    ) -> tuple[list[RoomId], float, SearchStats]:
        """Return (path, total cost, search stats) from src to dst under cost_fn."""
        ...


@runtime_checkable
class Deliberator(Protocol):
    """Turns an agent's notes into a meeting statement, suspicion ranking and vote."""

    def speak(self, ctx: "MeetingContext") -> Statement:  # noqa: F821
        """Produce this agent's Statement for the current meeting round."""
        ...


@runtime_checkable
class Policy(Protocol):
    """Chooses one action per tick from an Observation."""

    def decide(self, obs: "Observation") -> Action:  # noqa: F821
        """Return the single action this agent takes this tick."""
        ...
