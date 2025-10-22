import jax

from brax.base import System
from brax.envs.base import PipelineEnv, State


class LavaGolf(PipelineEnv):
    """Trains a robot arm to push a ball to a target while avoiding obstacles and floor hazards."""

    def __init__(self, **kwargs):
        sys = System()
        super().__init__(sys, **kwargs)

    def reset(self, rng: jax.Array) -> State:
        pass

    def step(self, state: State, action: jax.Array) -> State:
        pass
