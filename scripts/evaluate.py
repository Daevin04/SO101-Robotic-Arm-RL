#!/usr/bin/env python3
"""
Evaluation script for trained SO-101 models.

Runs a trained model for multiple episodes and reports statistics:
- Success rate
- Average episode reward
- Average episode length

Usage:
    python scripts/evaluate.py \
        --model checkpoints/stage_1/stage_1_task_25000_steps.zip \
        --env stage_1 \
        --n-episodes 100
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.stage_1_task import Stage1Task
from envs.stage_2_task import Stage2Task
from envs.stage_3_task import Stage3Task
from envs.stage_4_task import Stage4Task
from envs.stage_5_task import Stage5Task
from envs.stage_6_task import Stage6Task

ENV_MAP = {
    "stage_1": Stage1Task,
    "stage_2": Stage2Task,
    "stage_3": Stage3Task,
    "stage_4": Stage4Task,
    "stage_5": Stage5Task,
    "stage_6": Stage6Task,
}


def evaluate_model(model_path, env_name, n_episodes=100, render=False):
    """
    Evaluate a trained model.

    Args:
        model_path: Path to trained model (.zip file)
        env_name: Name of environment (e.g., "stage_1")
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment

    Returns:
        Dictionary with evaluation statistics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path}")
    print(f"Environment: {env_name}")
    print(f"Episodes: {n_episodes}")
    print(f"{'='*60}\n")

    # Load model
    model = SAC.load(model_path)

    # Create environment
    env_class = ENV_MAP.get(env_name)
    if env_class is None:
        raise ValueError(f"Unknown environment: {env_name}. Available: {list(ENV_MAP.keys())}")

    render_mode = "human" if render else None
    env = env_class(render_mode=render_mode)

    # Run evaluation
    episode_rewards = []
    episode_lengths = []
    successes = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            if render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Check for success (if info provides it)
        success = info.get("is_success", False)
        successes.append(1 if success else 0)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Length: {episode_length} | "
                  f"Success: {success}")

    env.close()

    # Compute statistics
    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "success_rate": np.mean(successes) * 100,
        "n_episodes": n_episodes,
    }

    # Print results
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Mean reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"Mean length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
    print(f"Success rate: {stats['success_rate']:.1f}%")
    print(f"{'='*60}\n")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained SO-101 models")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.zip file)",
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=list(ENV_MAP.keys()),
        help="Environment name",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate (default: 100)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during evaluation",
    )

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    # Run evaluation
    stats = evaluate_model(
        model_path=args.model,
        env_name=args.env,
        n_episodes=args.n_episodes,
        render=args.render,
    )

    return stats


if __name__ == "__main__":
    main()
