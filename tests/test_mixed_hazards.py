import os
from pathlib import Path

from brax.envs.SafePointGoal import SafePointGoal, default_config
from train_from_config import record_episode_video
from utils import make_multi_directional_policy


def test_record_video_mixed_hazards():
    """Test recording a video of circular motion with cylinder hazards."""
    out_path = Path(__file__).parent / "videos" / f"mixed_hazards.mp4"

    print("\nRecording an episode with mixed hazards...")
    print(f"Output path: {out_path}")
    start_time = os.times()

    cfg = default_config()
    # cfg.placement.extents = (-3.0, -3.0, 3.0, 3.0)
    cfg.placement.extents = (-2.0, -2.0, 2.0, 2.0)
    cfg.goals.type = 'cylinder'
    cfg.goals.count = 2
    cfg.goals.size = 0.1
    cfg.goals.height = 0.2
    cfg.hazards.specs = [
        # 1) 3 cube hazards, size 0.3, height 0.3, non-collidable
        {"type": "cube", "count": 3, "size": 0.3, "height": 0.3, "collidable": False},

        # 2) 2 cube hazards, size 0.2, height 0.5, collidable
        {"type": "cube", "count": 2, "size": 0.2, "height": 0.5, "collidable": True},

        # 3) 4 cylinder hazards, radius 0.3, height 0.01, non-collidable
        {"type": "cylinder", "count": 4, "size": 0.3, "height": 0.01, "collidable": False},

        # 4) 3 cylinder hazards, radius 0.3, height 0.4, collidable
        {"type": "cylinder", "count": 3, "size": 0.3, "height": 0.4, "collidable": True},
    ]

    # Create an environment with cylinder hazards
    env = SafePointGoal(cfg)
    cube_hazards = env._hazard_manager.get_hazards_by_type("cube")
    cylinder_hazards = env._hazard_manager.get_hazards_by_type("cylinder")
    print(
        f"✓ Environment created in {os.times()[4] - start_time[4]:.2f} seconds with {env._num_hazards} hazards ({len(cube_hazards)} cubes, {len(cylinder_hazards)} cylinders)")
    start_time = os.times()
    steps = 5000

    # Make a deterministic policy
    make_infer = make_multi_directional_policy(
        action_dim=env.action_size,
        thrust=1.0,  # forward push along agent x
        thrust_idx=0,  # actuator 0 = site motor along x
        yaw_idx=1,  # actuator 1 = velocity actuator on hinge 'z'
        steps=steps,  # total steps in episode
        num_yaw_values=10,  # number of different yaw values to sample
        rng_seed=8,
    )
    print(f"✓ Policy created {os.times()[4] - start_time[4]:.2f} seconds")
    start_time = os.times()

    # Record the episode
    record_episode_video(
        env=env,
        make_inference_fn=make_infer,
        params=None,  # policy ignores params
        steps=steps,
        camera="fixedfar",
        width=640,
        height=480,
        fps=50,
        frame_stride=10,
        out_name=str(out_path.name),
        log_to_wandb=False,
        seed=6,
        show_metrics=True,  # Enable cost display on video
    )
    print(f"✓ Test completed in {os.times()[4] - start_time[4]:.2f} seconds")

    # Verify the video was created
    saved = os.path.join("videos", out_path.name)
    assert os.path.exists(saved), f"Expected video at {saved}"
    assert os.path.getsize(saved) > 0, "Video file is empty"
    print(f"✓ Video saved successfully at: {saved}")
    print(f"  File size: {os.path.getsize(saved)} bytes")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    test_record_video_mixed_hazards()
    print("\n" + "=" * 60)
    print("✓ Video successfully recorded with mixed hazards.")
