import os
from pathlib import Path

from brax.envs.SafePointGoal import SafePointGoal
from train_from_config import record_episode_video
from utils import make_multi_directional_policy


def test_record_video_circular_cylinders():
    """Test recording a video of circular motion with cylinder hazards."""
    num_hazards = 12
    out_path = Path(__file__).parent / "videos" / f"cylinder_hazards_{num_hazards}.mp4"

    print(f"\nRecording an episode with {num_hazards} cylinder hazards...")
    print(f"Output path: {out_path}")

    # Create environment with cylinder hazards
    env = SafePointGoal(num_hazards=num_hazards, hazard_type="cylinders")
    print(f"✓ Environment created with {env._num_hazards} {env._hazard_type} hazards")
    steps = 2500

    # Make a deterministic policy
    make_infer = make_multi_directional_policy(
        action_dim=env.action_size,
        thrust=1.0,  # forward push along agent x
        thrust_idx=0,  # actuator 0 = site motor along x
        yaw_idx=1,  # actuator 1 = velocity actuator on hinge 'z'
        steps=steps,  # total steps in episode
        num_yaw_values=5,  # number of different yaw values to sample
        rng_seed=8,
    )
    print("✓ Policy created")

    # Record the episode
    record_episode_video(
        env=env,
        make_inference_fn=make_infer,
        params=None,  # policy ignores params
        steps=steps,
        camera="fixedfar",
        width=640,
        height=480,
        fps=500,
        out_name=str(out_path.name),
        log_to_wandb=False,
        seed=0,
        show_metrics=True,  # Enable cost display on video
    )
    print("✓ Video recording completed")

    # Verify the video was created
    saved = os.path.join("videos", out_path.name)
    assert os.path.exists(saved), f"Expected video at {saved}"
    assert os.path.getsize(saved) > 0, "Video file is empty"
    print(f"✓ Video saved successfully at: {saved}")
    print(f"  File size: {os.path.getsize(saved)} bytes")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    test_record_video_circular_cylinders()
    print("\n" + "=" * 60)
    print("✓ Video successfully recorded with cylinder hazards.")
