"""A random-walk stub Policy for driving the engine in tests."""

from dataclasses import replace

from amongus.config import SimConfig
from amongus.rng import make_rng
from amongus.types import Action, DoTask, Move, Wait
from amongus.world.engine import Engine
from amongus.world.map import GraphView
from amongus.world.observation import Observation

_view = GraphView()


class RandomWalkPolicy:
    """Does an owned task if standing in its room, else moves to a random open neighbor."""

    def __init__(self, seed: int) -> None:
        """Seed this policy's private RNG, independent of the engine's own generator."""
        self._rng = make_rng(seed)

    def decide(self, obs: Observation) -> Action:
        """Return DoTask if possible, else Move to a uniformly random neighboring room."""
        if obs.room is None:
            return Wait()
        if obs.self_phys is not None:
            for task in obs.self_phys.tasks:
                if task.room == obs.room and not task.done:
                    return DoTask(task_id=task.task_id)
        options = [r for r, _w in _view.neighbors(obs.room)]
        if not options:
            return Wait()
        return Move(to=options[self._rng.integers(len(options))])


class ScriptedPolicy:
    """Replays a fixed list of actions in order, then Waits forever."""

    def __init__(self, actions: list[Action]) -> None:
        """Store the scripted action sequence."""
        self._actions = actions
        self._i = 0

    def decide(self, obs: Observation) -> Action:
        """Return the next scripted action, or Wait once the script is exhausted."""
        if self._i >= len(self._actions):
            return Wait()
        action = self._actions[self._i]
        self._i += 1
        return action


def safe_engine(seed: int, **overrides) -> tuple[Engine, list[str]]:
    """Build a 3-player (2 crew, 1 impostor), idle Engine so no win condition fires by itself."""
    config = replace(SimConfig(seed=seed, n_players=3, n_impostors=1), **overrides)
    ids = list(config.colors[: config.n_players])
    engine = Engine(config, {aid: ScriptedPolicy([]) for aid in ids})
    return engine, ids
