import os
import tempfile

import jax
from jax import numpy as jp
from ml_collections import config_dict

from brax.envs.builder import XMLBuilder
from brax.envs.goals import GoalManager
from brax.envs.hazards import HazardManager


# -------------------------------- World building --------------------------------

def create_hazard_manager_from_config(hazards_cfg) -> HazardManager:
    """Create a HazardManager from new nested config.

    hazards_cfg: ConfigDict with field:
      - specs: list of dicts {type, count, size, height, collidable}
    """
    manager = HazardManager()
    specs = list(hazards_cfg.specs) if hazards_cfg and hasattr(hazards_cfg, 'specs') else []
    for spec in specs:
        manager.add_hazards(
            hazard_type=spec.get('type', 'cube'),
            count=spec.get('count', 0),
            size=spec.get('size', None),
            height=spec.get('height', None),
            collidable=spec.get('collidable', True),
            movable=spec.get('movable', False),
            density=spec.get('density', None),
        )
    return manager


def create_goal_manager_from_config(goals_cfg) -> GoalManager:
    """Create a GoalManager from new nested config.

    goals_cfg: ConfigDict with fields:
      - type ('cube'|'cylinder'), count, size, height,
        positions (optional list), collidable (bool)
    """
    manager = GoalManager()
    g = goals_cfg
    manager.add_goals(goal_type=g.type, count=g.count, positions=g.positions, size=g.size, height=g.height)
    return manager


def generate_point_goal_xml(goal_manager: GoalManager, hazard_manager: HazardManager,
                            env_name="point_goal_hazard") -> str:
    """Generate XML file content with the specified goals and hazards.

    Args:
        goal_manager: GoalManager containing the goals
        hazard_manager: HazardManager containing the hazards

    Returns:
        Path to the generated temporary XML file
    """
    # Construct absolute path to point.xml
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_xml_path = os.path.join(current_dir, 'assets', 'point.xml')

    # Use XMLBuilder to generate complete XML
    builder = XMLBuilder(env_name)
    xml_content = builder.build_xml(base_xml_path, goal_manager, hazard_manager)

    # Create temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.xml', prefix='safe_point_goal_')
    try:
        with os.fdopen(temp_fd, 'w') as f:
            f.write(xml_content)
    except:
        os.close(temp_fd)
        raise

    return temp_path


# -------------------------------- Config overwriting --------------------------------

def shallow_merge(base: config_dict.ConfigDict, over: config_dict.ConfigDict | None):
    if over is None:
        return base
    for k, v in over.items():
        base[k] = v
    return base


def expand_hazard_specs(cfg: config_dict.ConfigDict):
    if not hasattr(cfg, "hazards"):
        return cfg
    td = dict(getattr(cfg.hazards, "type_defaults", {}))
    specs = list(getattr(cfg.hazards, "specs", []))
    resolved = []
    for s in specs:
        if not isinstance(s, dict):
            resolved.append(s)
            continue
        t = s.get("type")
        base = dict(td.get(t, {}))
        base.update(s)  # override only what the spec sets
        resolved.append(base)
    cfg.hazards.specs = resolved
    return cfg


# -------------------------------- Per-goal signed distance helpers (2D) --------------------------------

def sdf_cylinder(agent_xy, center_xy, radius):
    """
    Signed distance from a 2D point to a circle.
      < 0 : inside the circle
      = 0 : on the boundary
      > 0 : outside the circle
    """
    offset = agent_xy - center_xy
    distance_to_center = jp.linalg.norm(offset, ord=2)
    return distance_to_center - radius


def sdf_cube(agent_xy, center_xy, half_sizes_xy, yaw=0.0):
    """
    Signed distance from a 2D point to a yaw-rotated axis-aligned box (rectangle).

    half_sizes_xy = (half_width, half_height)
    yaw = rotation of the box in radians, positive CCW.

      < 0 : inside the box
      = 0 : on the boundary
      > 0 : outside the box
    """
    # Rotate the point into the box's local frame
    cos_yaw, sin_yaw = jp.cos(yaw), jp.sin(yaw)
    rotation = jp.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    local_xy = (agent_xy - center_xy) @ rotation

    # Distance to the rectangle in local coordinates
    # q is how far we are from the inner rectangle [-half_sizes, +half_sizes]
    q = jp.abs(local_xy) - half_sizes_xy

    # Outside part: Euclidean norm of the positive components (how far outside)
    outside_dist = jp.linalg.norm(jp.maximum(q, 0.0), ord=2)

    # Inside part: if inside, max(q) is negative; clamp to <= 0 to keep sign
    inside_dist = jp.minimum(jp.maximum(q[0], q[1]), 0.0)

    # Combine: positive outside distance or negative depth inside
    return outside_dist + inside_dist


# -------------------------------- Object placement --------------------------------

def sample_candidate_positions(rng_key: jp.ndarray, num_candidates: int, keepout: float,
                               placement_extents) -> jp.ndarray:
    """ Sample many candidates within extents"""
    min_x, min_y, max_x, max_y = placement_extents
    min_x = min_x + keepout
    min_y = min_y + keepout
    max_x = max_x - keepout
    max_y = max_y - keepout
    rng_key, sk1, sk2 = jax.random.split(rng_key, 3)
    xs = jax.random.uniform(sk1, (num_candidates,), minval=min_x, maxval=max_x)
    ys = jax.random.uniform(sk2, (num_candidates,), minval=min_y, maxval=max_y)
    zs = jp.full((num_candidates,), 0.09)
    return jp.stack([xs, ys, zs], axis=1)


def sample_position_in_extents(rng_key: jp.ndarray, placement_extents, keepout: float = 0.0) -> jp.ndarray:
    """Sample a position within the constrained placement extents."""
    min_x, min_y, max_x, max_y = placement_extents

    # Apply keepout to reduce available area
    min_x = min_x + keepout
    min_y = min_y + keepout
    max_x = max_x - keepout
    max_y = max_y - keepout

    # Sample position
    pos_x = jax.random.uniform(rng_key, minval=min_x, maxval=max_x)
    rng_key, subkey = jax.random.split(rng_key)
    pos_y = jax.random.uniform(subkey, minval=min_y, maxval=max_y)

    return jp.array([pos_x, pos_y, 0.09])  # Fixed z-height


def choose_valid_position(
        rng_key: jp.ndarray,
        existing_positions_xy: jp.ndarray,
        existing_keepouts: jp.ndarray,
        active_count: jp.ndarray,
        candidate_keepout: float,
        num_candidates: int,
        placement_extents: tuple[float, float, float, float],
        placement_margin: float,
) -> tuple[jp.ndarray, jp.ndarray]:
    """
    Pick a valid (x, y, z) position for a new object given already-placed objects
    and keepout radii. We sample multiple candidates, take the one that maximizes
    its worst (minimum) clearance to all active objects, and if that candidate is
    still invalid we nudge it away from the nearest active object just enough to
    satisfy the clearance constraint, then clip to the placement extents.

    Clearance rule:
        ||candidate_xy - existing_xy[j]|| >= existing_keepouts[j] + candidate_keepout + placement_margin

    Args:
        rng_key: JAX PRNGKey.
        existing_positions_xy: (N, 2) array of existing object centers in XY.
        existing_keepouts: (N,) keepout radius per existing object, aligned with positions.
        active_count: scalar int32; number of active entries in the arrays (<= N).
        candidate_keepout: keepout radius for the object we’re placing.
        num_candidates: number of random candidate samples to evaluate.
        placement_extents: (min_x, min_y, max_x, max_y) sampling region.
        placement_margin: extra spacing added on top of both keepouts.

    Returns:
        (chosen_position_xyz, next_rng_key)
          chosen_position_xyz: (3,) array [x, y, z]. z is carried from the sampled candidate.
          next_rng_key: PRNGKey advanced once to keep scans deterministic.

    Notes:
        - Only the first `active_count` rows of existing arrays are considered active.
        - We use the circumscribed “required clearance” per active neighbor and
          maximize the candidate’s minimum slack over neighbors.
        - On violation, we push along the direction away from the nearest active neighbor
          by exactly the missing clearance plus a tiny epsilon, then clip to extents.
    """
    # Sample K candidates uniformly inside extents minus candidate keepout.
    candidates = sample_candidate_positions(rng_key, num_candidates, candidate_keepout, placement_extents)
    candidate_xy = candidates[:, :2]  # (K, 2)

    # Active mask for first `active_count` entries
    total_slots = existing_positions_xy.shape[0]
    indices_row = jp.arange(total_slots)[None, :]  # (1, N)
    active_mask = (indices_row < active_count)  # (1, N), broadcast over K

    # Pairwise distances candidate k to existing j: (K, N)
    deltas = candidate_xy[:, None, :] - existing_positions_xy[None, :, :]
    distances = jp.sqrt(jp.sum(jp.square(deltas), axis=-1) + 1e-8)

    # Required separation per neighbor j for this candidate
    required = existing_keepouts[None, :] + candidate_keepout + placement_margin  # (1, N)

    # Slack: positive means candidate meets clearance against that neighbor
    slack = distances - required  # (K, N)
    # Ignore inactive rows by giving them huge positive slack
    slack = jp.where(active_mask, slack, jp.full_like(slack, 1e9))

    # For each candidate: worst slack across active neighbors
    worst_slack_per_candidate = jp.min(slack, axis=1)  # (K,)
    best_index = jp.argmax(worst_slack_per_candidate)
    best_candidate = candidates[best_index]
    best_candidate_xy = best_candidate[:2]

    # Valid if the best candidate’s worst slack is nonnegative
    is_valid = (worst_slack_per_candidate[best_index] >= 0.0)

    def _accept(pos_xyz):
        return pos_xyz

    def _nudge(pos_xyz):
        # Move away from nearest active neighbor just enough to satisfy clearance
        distances_best = distances[best_index]  # (N,)
        distances_best = jp.where(active_mask[0], distances_best, jp.full_like(distances_best, 1e9))
        nearest_j = jp.argmin(distances_best)

        required_sep = required[0, nearest_j]
        current_sep = distances_best[nearest_j]

        # Direction from neighbor to candidate
        direction = best_candidate_xy - existing_positions_xy[nearest_j]
        direction_unit = direction / (jp.sqrt(jp.sum(jp.square(direction))) + 1e-8)

        # Move delta so new distance == required + epsilon
        delta_xy = (required_sep - current_sep + 1e-3) * direction_unit
        nudged_xy = best_candidate_xy + delta_xy

        # Clip to extents minus own keepout
        min_x, min_y, max_x, max_y = placement_extents
        nudged_xy = jp.array([
            jp.clip(nudged_xy[0], min_x + candidate_keepout, max_x - candidate_keepout),
            jp.clip(nudged_xy[1], min_y + candidate_keepout, max_y - candidate_keepout),
        ])

        return jp.array([nudged_xy[0], nudged_xy[1], pos_xyz[2]])

    chosen_position = jax.lax.cond(is_valid, _accept, _nudge, best_candidate)

    # Advance RNG once per call so scan order is deterministic
    rng_key, _ = jax.random.split(rng_key)
    return chosen_position, rng_key


def place_objects(rng_key: jp.ndarray, positions_xy: jp.ndarray, keepouts_array: jp.ndarray, placed_count: jp.ndarray,
                  per_item_keepouts: jp.ndarray, num_items: int, num_candidates: int, placement_extents,
                  placement_margin: float):
    """
    Place `num_items` objects using per-item keepout radii, updating
    positions_xy and keepouts_array in-place, and returning the new positions.

    Args:
        rng_key: JAX PRNGKey
        positions_xy: (N, 2) array of obstacle centers (agent + already placed)
        keepouts_array: (N,) array of keepout radii aligned with positions_xy
        placed_count: scalar int32, number of active entries in positions_xy/keepouts_array
        per_item_keepouts: (num_items,) keepout radius per object being placed
        num_items: number of objects to place
        num_candidates: number of candidate samples per placement
        placement_extents: (min_x, min_y, max_x, max_y)
        placement_margin: extra spacing added in placement checks

    Returns:
        rng_key, positions_xy, keepouts_array, placed_count, placed_positions (num_items, 3)
    """

    def place_one(carry, i):
        rng_key_i, positions_xy_i, keepouts_array_i, placed_count_i = carry
        keepout_radius_i = per_item_keepouts[i]

        position_i, rng_key_i = choose_valid_position(
            rng_key_i,
            positions_xy_i,
            keepouts_array_i,
            placed_count_i,
            keepout_radius_i,
            num_candidates,
            placement_extents,
            placement_margin,
        )

        positions_xy_i = positions_xy_i.at[placed_count_i].set(position_i[:2])
        keepouts_array_i = keepouts_array_i.at[placed_count_i].set(keepout_radius_i)
        placed_count_i = placed_count_i + 1

        return (rng_key_i, positions_xy_i, keepouts_array_i, placed_count_i), position_i

    (rng_key, positions_xy, keepouts_array, placed_count), placed_positions = jax.lax.scan(
        place_one,
        (rng_key, positions_xy, keepouts_array, placed_count),
        xs=jp.arange(num_items),
    )
    return rng_key, positions_xy, keepouts_array, placed_count, placed_positions


# -------------------------------- Math --------------------------------

def safe_norm(x, axis=None, keepdims=False, eps=1e-8):
    """Safely compute the norm with a small epsilon to avoid NaN."""
    return jp.sqrt(jp.sum(jp.square(x), axis=axis, keepdims=keepdims) + eps)
