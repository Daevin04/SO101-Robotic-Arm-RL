#!/usr/bin/env python3
"""
Quick example: Train Stage 1 for 5000 steps (fast demo).

This example demonstrates the basic training workflow.
For production training, use 25K+ steps per stage.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.stage_1_task import Stage1Task


def main():
    print("="*60)
    print("Quick Training Demo - Stage 1")
    print("="*60)
    print("\nThis will train for 5000 steps (demo purposes)")
    print("For real training, use 25000+ steps\n")

    # Create environment
    env = Stage1Task()

    # Create SAC agent
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        verbose=1,
    )

    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="./demo_checkpoints/",
        name_prefix="demo_stage_1",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=5000,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    # Save final model
    model.save("demo_checkpoints/demo_stage_1_final")
    print("\nTraining complete!")
    print(f"Model saved to: demo_checkpoints/demo_stage_1_final.zip")

    # Quick evaluation
    print("\nRunning quick evaluation (10 episodes)...")
    successes = 0
    for episode in range(10):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        success = info.get("is_success", False)
        if success:
            successes += 1

        print(f"Episode {episode + 1}/10 | Reward: {episode_reward:.1f} | Success: {success}")

    print(f"\nSuccess rate: {successes}/10 ({successes * 10}%)")
    print("\nNote: 5000 steps is too short for good performance.")
    print("For real training, use: python scripts/train.py --stage 1 --timesteps 25000")

    env.close()


if __name__ == "__main__":
    main()
