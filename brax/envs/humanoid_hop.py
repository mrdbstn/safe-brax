# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Humanoid one-legged hopping locomotion environment with contact constraints."""

import jax
from jax import numpy as jp
from brax.envs.humanoidstandup import HumanoidStandup
from brax.envs.base import State


class HumanoidHop(HumanoidStandup):
  """Humanoid locomotion environment with one-legged hopping constraint.
  
  This environment extends the standard HumanoidStandup environment by adding a
  contact-based safety constraint: the humanoid must hop on ONE designated leg,
  keeping the opposite leg off the ground. This creates a challenging locomotion
  task that tests contact pattern constraints and dynamic balance.
  
  The cost (constraint violation) is binary: 1.0 when the non-hopping foot
  touches the ground, 0.0 otherwise. This tests safe RL algorithms' ability to
  handle sparse, discontinuous constraint signals.
  """

  def __init__(
      self,
      hopping_leg: str = 'left',
      contact_threshold: float = 0.5,  # Newtons, threshold for foot contact
      cost_weight: float = 1.0,
      **kwargs,
  ):
    """Initialize the one-legged hopping humanoid environment.
    
    Args:
      hopping_leg: Which leg to hop on ('left' or 'right').
      contact_threshold: Minimum contact force (Newtons) to count as violation.
      cost_weight: Weight for contact constraint violation cost.
      **kwargs: Additional arguments passed to parent HumanoidStandup class.
    """
    super().__init__(**kwargs)
    
    assert hopping_leg in ['left', 'right'], f"hopping_leg must be 'left' or 'right', got {hopping_leg}"
    self._hopping_leg = hopping_leg
    self._contact_threshold = contact_threshold
    self._cost_weight = cost_weight
    
    # Body indices for feet (will be set after sys is loaded)
    # In humanoidstandup.xml:
    # Body 0: torso (root)
    # We need to find indices for left_foot and right_foot bodies
    self._left_foot_body_idx = None
    self._right_foot_body_idx = None

  def reset(self, rng: jax.Array) -> State:
    """Reset the environment with contact constraint metrics."""
    state = super().reset(rng)
    
    # Find foot body indices on first reset (cached for future resets)
    if self._left_foot_body_idx is None:
      self._find_foot_indices()
    
    # Initialize contact-related metrics with consistent shapes
    reward_shape = jp.shape(state.reward)
    zero = jp.zeros(reward_shape)
    
    # Add contact-specific metrics to the existing humanoidstandup metrics
    state.metrics.update(
        forward_reward=zero,
        x_position=zero,
        y_position=zero,
        distance_from_origin=zero,
        x_velocity=zero,
        y_velocity=zero,
        left_foot_contact_force=zero,
        right_foot_contact_force=zero,
        contact_violation=zero,
        contact_cost=zero,
        cost=zero,  # Initialize cost for PPO Lagrange
    )
    
    # Initialize info dictionary with cost (required for PPO Lagrange v2)
    # Note: Don't store strings (like hopping_leg) as they're not JAX types
    info = {
        "cost": zero,
        "left_foot_contact_force": zero,
        "right_foot_contact_force": zero,
        "contact_violation": zero,
        "step_count": 0,
    }
    
    return state.replace(info=info)

  def step(self, state: State, action: jax.Array) -> State:
    """Run one timestep of the environment's dynamics with contact constraints."""
    # Scale action from [-1,1] to actuator limits
    action_min = self.sys.actuator.ctrl_range[:, 0]
    action_max = self.sys.actuator.ctrl_range[:, 1]
    action = (action + 1) * (action_max - action_min) * 0.5 + action_min

    pipeline_state = self.pipeline_step(state.pipeline_state, action)

    # Calculate center-of-mass (CoM) velocity for reward
    pipeline_state0 = state.pipeline_state
    if pipeline_state0 is not None:
        com_before, *_ = self._com(pipeline_state0)
        com_after, *_ = self._com(pipeline_state)
        velocity = (com_after - com_before) / self.dt
        forward_reward = velocity[0]  # Reward forward movement
    else:
        forward_reward = 0.0
        velocity = jp.zeros(3)
    
    # Control cost
    ctrl_cost = 0.01 * jp.sum(jp.square(action))

    # Extract contact forces for feet
    left_foot_force, right_foot_force = self._get_foot_contact_forces(pipeline_state)
    
    # Compute contact violation cost
    contact_cost, violation = self._compute_contact_cost(left_foot_force, right_foot_force)

    obs = self._get_obs(pipeline_state, action)
    
    # Reward structure: encourage forward movement, base reward, control penalty
    reward = forward_reward + 1.0 - ctrl_cost
    done = 0.0
    
    # Ensure all metrics have consistent shapes
    reward_shape = jp.shape(reward)
    
    # Update metrics with contact-related information
    state.metrics.update(
        reward_linup=forward_reward,
        reward_quadctrl=-ctrl_cost,
        forward_reward=forward_reward,
        x_position=pipeline_state.x.pos[0, 0],
        y_position=pipeline_state.x.pos[0, 1],
        distance_from_origin=jp.linalg.norm(pipeline_state.x.pos[0, :2]),
        x_velocity=velocity[0],
        y_velocity=velocity[1],
        left_foot_contact_force=jp.broadcast_to(left_foot_force, reward_shape),
        right_foot_contact_force=jp.broadcast_to(right_foot_force, reward_shape),
        contact_violation=jp.broadcast_to(violation, reward_shape),
        contact_cost=jp.broadcast_to(contact_cost, reward_shape),
        cost=jp.broadcast_to(contact_cost, reward_shape),
    )
    
    # Update info dictionary with cost (required for PPO Lagrange v2)
    current_info = getattr(state, 'info', {})
    step_count = current_info.get('step_count', 0) + 1
    
    # Update info dictionary (copy existing and update)
    # Note: Don't store strings (like hopping_leg) as they're not JAX types
    new_info = current_info.copy() if isinstance(current_info, dict) else {}
    new_info.update({
        "cost": contact_cost,
        "left_foot_contact_force": left_foot_force,
        "right_foot_contact_force": right_foot_force,
        "contact_violation": violation,
        "step_count": step_count,
    })
    
    return state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, info=new_info
    )

  def _find_foot_indices(self):
    """Find body indices for shin bodies (which contain foot geoms).
    
    Note: In the humanoidstandup model, the foot geoms are attached to the 
    shin bodies, not separate foot bodies. So we track shin contacts for
    foot contact detection.
    """
    # Use link_names from Brax System
    # The feet geoms are part of the shin bodies in humanoidstandup.xml
    if hasattr(self.sys, 'link_names'):
      link_names = self.sys.link_names
      try:
        # In Brax, indices start from 0 for first body
        # We need shin indices since foot geoms are attached to shins
        self._left_foot_body_idx = link_names.index('left_shin')
        self._right_foot_body_idx = link_names.index('right_shin')
        return
      except ValueError:
        pass
    
    # Fallback: try MuJoCo model directly
    mj_model = self.sys.mj_model
    if mj_model is not None:
      try:
        self._left_foot_body_idx = mj_model.body('left_shin').id
        self._right_foot_body_idx = mj_model.body('right_shin').id
        return
      except Exception as e:
        print(f"Warning: Could not find shin bodies: {e}")
    
    # If all methods fail, raise error
    raise RuntimeError(
        f"Could not find 'left_shin' and 'right_shin' bodies in model. "
        f"Available link_names: {getattr(self.sys, 'link_names', 'N/A')}"
    )

  def _get_foot_contact_forces(self, pipeline_state) -> tuple[jax.Array, jax.Array]:
    """Extract contact forces for left and right feet.
    
    Args:
      pipeline_state: Current physics pipeline state (mjx.Data).
    
    Returns:
      Tuple of (left_foot_force_magnitude, right_foot_force_magnitude).
    """
    # Find foot indices if not yet cached
    if self._left_foot_body_idx is None:
      self._find_foot_indices()
    
    # cfrc_ext shape: (nbody, 6) - external forces and torques on each body
    # First 3 elements are forces [fx, fy, fz], last 3 are torques
    # We're interested in the magnitude of the contact force (primarily vertical)
    cfrc_ext = pipeline_state.cfrc_ext
    
    # Extract contact forces for feet
    left_foot_force_vec = cfrc_ext[self._left_foot_body_idx, :3]  # [fx, fy, fz]
    right_foot_force_vec = cfrc_ext[self._right_foot_body_idx, :3]
    
    # Use vertical (z) component as primary indicator of ground contact
    # Take absolute value since force direction can vary
    left_foot_force = jp.abs(left_foot_force_vec[2])  # z-component
    right_foot_force = jp.abs(right_foot_force_vec[2])
    
    return left_foot_force, right_foot_force

  def _compute_contact_cost(
      self, 
      left_foot_force: jax.Array,
      right_foot_force: jax.Array
  ) -> tuple[jax.Array, jax.Array]:
    """Compute cost based on which foot should not touch the ground.
    
    Args:
      left_foot_force: Contact force magnitude on left foot.
      right_foot_force: Contact force magnitude on right foot.
    
    Returns:
      Tuple of (cost, binary_violation_flag).
    """
    # Determine which foot should NOT touch the ground
    if self._hopping_leg == 'left':
      # Hopping on left, so right foot should NOT touch
      violation_force = right_foot_force
    else:
      # Hopping on right, so left foot should NOT touch
      violation_force = left_foot_force
    
    # Binary violation: 1.0 if contact force exceeds threshold, 0.0 otherwise
    violation = (violation_force > self._contact_threshold).astype(jp.float32)
    
    # Cost is weighted binary violation
    cost = self._cost_weight * violation
    
    return cost, violation

