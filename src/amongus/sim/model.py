"""Mesa 3.5 wrapper around Engine (SPEC §2).

Mesa 3 removed RandomActivation/SimultaneousActivation, so there is no scheduler for this
class to delegate to: Engine already does its own simultaneous tick resolution (§6). Mesa's
only real contributions here are model scaffolding (a seeded `mesa.Model`, whose `step()`
is driven through Mesa's own time-advance machinery) and `DataCollector`, which gathers a
per-tick series for later plotting. Do not read more into Mesa's role than that.
"""

from __future__ import annotations

import mesa

from amongus.config import SimConfig
from amongus.contracts import Policy
from amongus.types import AgentId
from amongus.world.engine import Engine, MeetingHandler


class AmongUsModel(mesa.Model):
    """Wraps an Engine so Mesa's DataCollector can gather per-tick series."""

    def __init__(
        self,
        config: SimConfig,
        policies: dict[AgentId, Policy],
        meeting_handler: MeetingHandler | None = None,
    ) -> None:
        """Build the underlying Engine and register the per-tick model reporters."""
        super().__init__(rng=config.seed)
        self.engine = Engine(config, policies, meeting_handler)
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "alive": lambda m: sum(1 for a in m.engine.world.agents.values() if a.alive),
                "task_bar": lambda m: m.engine.world.task_bar,
                "sabotage_active": lambda m: m.engine.world.sabotage is not None,
                "meeting_count": lambda m: m.engine.world.meeting_count,
            }
        )
        self.datacollector.collect(self)

    def step(self) -> None:
        """Advance the wrapped engine by one tick and collect this tick's data point."""
        self.engine.tick()
        self.datacollector.collect(self)
