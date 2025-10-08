# Essential imports for velocity-constrained ant training
from datetime import datetime
import functools
import time
import os
import subprocess
import dataclasses
import warnings
from typing import Dict, Any, List, Optional

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
    model_save_dir = "models"

def training_metric_steps(config: ExperimentConfig) -> int:
    return config.episode_length * config.num_envs // 5

def validate_gpu_connection():
    """Check if NVIDIA GPU connection is available."""
    try:
        if subprocess.run('nvidia-smi', capture_output=True).returncode:
            warnings.warn("Warning: Cannot communicate with GPU")
        else:
            print("✓ GPU detected")
    except:
        warnings.warn("Warning: nvidia-smi not available")

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

def initialize_wandb_logging(config: ExperimentConfig):
    run_name = f"{config.env_name}_ppo_lag_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    wandb.init(
        project=config.env_name,
        name=run_name,
        config=config.__dict__,
    )

def wandb_log_final_train_params(final_metrics: dict, training_time: float, config: ExperimentConfig):
    if final_metrics and wandb.run is not None:
        final_wandb_metrics = {}
        for key, value in final_metrics.items():
            if isinstance(value, (int, float, np.number)):
                final_wandb_metrics[f"final/{key}"] = float(value)

        # Add training summary
        final_wandb_metrics.update({
            'final/training_time_seconds': training_time,
            'final/training_steps': config.num_timesteps,
            'final/max_velocity_limit': config.max_velocity,
            'final/safety_bound': config.safety_bound,
            'final/lagrangian_coef_rate': config.lagrangian_coef_rate,
        })

        wandb.log(final_wandb_metrics, step=config.num_timesteps)
        print("✓ Final metrics logged to W&B")

def wandb_logging(num_steps: int, metrics: Dict[str, Any]) -> None:
    """ Log metrics to Weights & Biases. """
    if wandb.run is None:
        warnings.warn('No wandb run configured, skipping wandb logging')
        return

    wandb_log_data = {}

    for key, value in metrics.items():
        # Extract scalar value from tensors if needed
        log_value = value.item() if hasattr(value, 'item') else value

        # Categorize metrics for better organization in W&B dashboard
        # Metrics without prefixes are assumed to be training batch metrics
        if not (key.startswith("episode/") or key.startswith("eval/") or key.startswith("training/")):
            wandb_log_data[f"training_batch/{key}"] = log_value
        else:
            wandb_log_data[key] = log_value

    if wandb_log_data:
        wandb.log(wandb_log_data, step=int(num_steps))

def terminal_logging(num_steps: int, metrics: Dict[str, Any]) -> None:
    """ Print formatted metrics to terminal with hierarchical organization. """
    print(f"Step {num_steps:,}:")

    # Primary performance metrics - these show how well the agent is doing
    perf_metrics = [
        'eval/episode_reward',
        'eval/episode_forward_reward',
        'episode/reward',
        'episode/forward_reward'
    ]

    for key in perf_metrics:
        if key in metrics:
            value = metrics[key].item() if hasattr(metrics[key], 'item') else metrics[key]
            # Strip prefixes for cleaner terminal output
            display_name = key.replace('eval/', '').replace('episode/', '')
            print(f"  {display_name}: {value:.2f}")

    # Safety constraint metrics - these show constraint satisfaction
    # These are logged prominently as they're critical for safe RL
    constraint_metrics = [
        'eval/episode_cost',
        'eval/episode_velocity_cost',
        'episode/cost',
        'training/lambda_lagr',  # Lagrange multiplier for constrained optimization
        'training/cost_violation',
        'training/mean_cost',
        'eval/episode_velocity_magnitude',
        'eval/episode_velocity_violation'
    ]

    for key in constraint_metrics:
        if key in metrics:
            value = metrics[key].item() if hasattr(metrics[key], 'item') else metrics[key]

            # Apply specialized formatting based on metric type
            if "lambda" in key:
                # Lagrange multipliers need high precision
                print(f"  {key}: {value:.6f}")
            elif "velocity_magnitude" in key:
                # Velocity in m/s with 2 decimal places
                print(f"  {key}: {value:.2f} m/s")
            else:
                # General constraint metrics with moderate precision
                print(f"  {key}: {value:.4f}")

    print("")  # Empty line for readability between steps

def logging_callback(
        num_steps: int,
        metrics: Dict[str, Any],
        use_wandb: bool = True,
        use_terminal: bool = True
) -> None:
    """ Main logging callback that orchestrates terminal and W&B logging.

    This is the function you pass to your training loop. It coordinates
    both terminal output and remote logging to Weights & Biases.

    Args:
        num_steps: Current training step number
        metrics: Dictionary of metric names to values (from Brax's
                 EpisodeMetricsLogger or Evaluator)
        use_wandb: Whether to log to Weights & Biases (default: True)
        use_terminal: Whether to print to terminal (default: True)
    """
    # Terminal logging - always helpful during training
    if use_terminal:
        terminal_logging(num_steps, metrics)

    # Remote logging to W&B - for experiment tracking
    if use_wandb:
        wandb_logging(num_steps, metrics)

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

def save_model(params, config: ExperimentConfig) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(config.model_save_dir, exist_ok=True)

    model_path = f'{config.model_save_dir}/ant_velocity_constrained_{timestamp}'
    brax_model.save_params(model_path, params)
    print(f"Model saved to: {model_path}")

def train_model(train_env: brax.envs.Env, config: ExperimentConfig):
    configured_callback_function = functools.partial(logging_callback, use_wandb=True)

    start_time = time.time()
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
        progress_fn=configured_callback_function
    )

    training_time = time.time() - start_time
    wandb_log_final_train_params(final_metrics, training_time, config)
    save_model(params, config)

if __name__ == '__main__':
    compute_environment_setup()
    conf = ExperimentConfig()
    initialize_wandb_logging(conf)
    env_train = initialize_simulation_environment(conf)
    print(env_train)
    train_model(env_train, conf)
