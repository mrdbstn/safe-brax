import jax
import numpy as np
from brax.envs.SafePointGoal import SafePointGoal
import imageio.v3 as iio


def run_random_episode(env, jit_step, jit_reset, steps: int = 100):
    seed = jax.random.PRNGKey(4242)
    action = jax.numpy.ones(env.action_size)
    state = jit_reset(seed)

    print(f"Starting episode")

    states = [state]
    for i in range(steps):
        state = jit_step(state, action)
        states.append(state)
        print("Step", i)

    print(f"Finished episode")
    return states

def render_states(env, states):
    # 2. Render the states
    pipeline_states = [s.pipeline_state for s in states]

    frames = env.render(
        pipeline_states,
        width=320,
        height=240,
        camera="fixedfar"  # or camera ID
    )

    # 3. Save as video
    iio.imwrite("output.mp4", np.stack(frames), fps=100)


def _init_safe_goal():
    env = SafePointGoal()
    step_jit = jax.jit(env.step)
    reset_jit = jax.jit(env.reset)
    return env, step_jit, reset_jit


if __name__ == '__main__':
    env, step, reset = _init_safe_goal()
    states = run_random_episode(env, step, reset)
    render_states(env, states)

    from brax.envs.env_utils import get_action_dimensions
    print(get_action_dimensions("brax/envs/assets/hopper.xml"))