# Essential imports for velocity-constrained ant training
from datetime import datetime
import functools
import time
import os
import subprocess
import dataclasses

import jax
import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt
import wandb
import mujoco

import brax.envs
from brax import envs
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State as BraxState
from brax.training.agents.ppo_lagrange_v2 import train as ppo_lagrange_v2
from brax.training.agents.ppo import train as ppo
from brax.io import model as brax_model
from brax.io import json as brax_json

@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    # Environment
    env_name: str = "ant_velocity_constrained"
    max_velocity: float = 0.5  # Maximum allowed velocity (m/s) - set below ant's natural max (~0.81 m/s)
    velocity_cost_weight: float = 1.0  # Weight for velocity violation cost
    ctrl_cost_weight: float = 0.5

    # Training Settings
    num_timesteps: int = 10_000_000
    num_evals: int = 10
    episode_length: int = 1000
    num_envs: int = 2048
    batch_size: int = 512
    learning_rate: float = 3e-4
    entropy_cost: float = 1e-3
    unroll_length: int = 10
    num_minibatches: int = 32
    num_updates_per_batch: int = 8
    discounting: float = 0.99
    normalize_observations: bool = True
    reward_scaling: float = 1.0
    action_repeat: int = 1
    seed: int = 42
    max_devices_per_host: int = None
    # PPO Lagrange specific
    safety_bound: float = 0.1  # Maximum allowed velocity violation
    lagrangian_coef_rate: float = 0.01
    initial_lambda_lagr: float = 0.0
    # Training metrics logging
    log_training_metrics: bool = True

def training_metric_steps(config: ExperimentConfig) -> int:
    return config.episode_length * config.num_envs // 5

def validate_gpu_connection():
    """Check if NVIDIA GPU connection is available."""
    try:
        if subprocess.run('nvidia-smi', capture_output=True).returncode:
            print("Warning: Cannot communicate with GPU")
        else:
            print("✓ GPU detected")
    except:
        print("Warning: nvidia-smi not available")

def set_mujoco_rendering_backend():
    """Configure MuJoCo to use the EGL rendering backend (requires GPU)"""
    os.environ['MUJOCO_GL'] = 'egl'

def validate_mujoco_install():
    """Check if MuJoCo is installed and can run"""
    try:
        print('Checking MuJoCo installation...')
        mujoco.MjModel.from_xml_string('<mujoco/>')
        print('✓ MuJoCo installation successful')
    except Exception as e:
        raise RuntimeError(f'MuJoCo installation failed: {e}')

def configure_triton():
    """Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs"""
    xla_flags = os.environ.get('XLA_FLAGS', '')
    xla_flags += ' --xla_gpu_triton_gemm_any=True'
    os.environ['XLA_FLAGS'] = xla_flags

def compute_environment_setup():
    """Set up simulator (Mujoco) and GPU connection"""
    validate_gpu_connection()
    set_mujoco_rendering_backend()
    validate_mujoco_install()
    configure_triton()

def initialize_simulation_environment(config: ExperimentConfig) -> brax.envs.Env:
    """Set up the experiment for training"""
    ## TODO: Figure out why there are two environments.
    ##  Answer: There is an option to use a separate eval env, but
    ##  using the same settings is the same as not specifying it.
    train_env = envs.get_environment(
        env_name=config.env_name,
        max_velocity=config.max_velocity,
        velocity_cost_weight=config.velocity_cost_weight,
        ctrl_cost_weight=config.ctrl_cost_weight,
    )
    return train_env

def train_model(train_env: brax.envs.Env, config: ExperimentConfig):
    make_inference_fn, params, final_metrics = ppo_lagrange_v2(
        environment=train_env,
        num_timesteps=config.num_timesteps,
        num_evals=config.num_evals,
        reward_scaling=config.reward_scaling,
        episode_length=config.episode_length,
        normalize_observations=config.normalize_observations,
        action_repeat=config.action_repeat,
        unroll_length=config.unroll_length,
        num_minibatches=config.num_minibatches,
        num_updates_per_batch=config.num_updates_per_batch,
        learning_rate=config.learning_rate,
        entropy_cost=config.entropy_cost,
        discounting=config.discounting,
        num_envs=config.num_envs,
        batch_size=config.batch_size,
        max_devices_per_host=config.max_devices_per_host,
        seed=config.seed,
        # Enable Brax's internal episode metrics logging
        log_training_metrics=config.log_training_metrics,
        training_metrics_steps=training_metric_steps(config),
        # PPO-Lagrange specific parameters
        safety_bound=config.safety_bound,
        lagrangian_coef_rate=config.lagrangian_coef_rate,
        initial_lambda_lagr=config.initial_lambda_lagr,
        progress_fn=bound_progress_fn
    )

if __name__ == '__main__':
    # compute_environment_setup()
    conf = ExperimentConfig()
    env_train = initialize_simulation_environment(conf)
    print(env_train)
