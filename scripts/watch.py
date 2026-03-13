#!/usr/bin/env python3
"""
Watch the trained SO-101 agent in action with live visualization.

Opens a window showing the robot performing the task in real-time.

Usage:
    python watch.py                    # Interactive mode with menu
    python watch.py --model PATH       # Direct mode with specific model
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")  # Must set BEFORE importing mujoco

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import SAC

from envs.stage_1_task import Stage1Task
try:
    from envs.stage_2_task import Stage2Task
except ImportError:
    Stage2Task = None  # Stage 2 not implemented yet


def watch_random_agent(env, n_episodes=2, auto_start=False):
    """Demonstrate random agent behavior."""
    print("\n" + "=" * 50)
    print("RANDOM AGENT (no training)")
    print("Watch how it moves randomly!")
    print("=" * 50)
    if not auto_start:
        input("Press Enter to start...")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        steps_taken = 0

        for step in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            env.render()
            time.sleep(0.02)
            steps_taken = step + 1

            if terminated or truncated:
                break

        xy_dist = info.get("xy_distance", 0) * 100
        print(f"  Episode {ep+1}: reward={ep_reward:.1f}, XY dist={xy_dist:.1f}cm (random)")

    print("\nNotice how it doesn't know what to do?\n")


def watch_trained_agent(env, model, n_episodes=3, auto_start=False):
    """Demonstrate trained agent behavior."""
    print("\n" + "=" * 50)
    print("TRAINED AGENT (after learning)")
    print("Watch how it aligns with the object!")
    print("=" * 50)
    if not auto_start:
        input("Press Enter to start...")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        steps_taken = 0

        for step in range(150):  # Max 150 steps per episode
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            env.render()
            time.sleep(0.02)
            steps_taken = step + 1

            if terminated or truncated:
                break

        # Show detailed stats
        status = "SUCCESS!" if info.get("is_success", False) else "failed"
        xy_dist = info.get("xy_distance", 0) * 100  # Convert to cm
        above_steps = info.get("above_steps", 0)

        print(f"  Episode {ep+1}: {status}")
        print(f"    Reward: {ep_reward:.1f}")
        print(f"    XY distance: {xy_dist:.1f}cm")
        print(f"    Steps aligned: {above_steps}/10")
        print(f"    Total steps: {steps_taken}")

    print("\nSee the difference? The agent learned to align with the object!")


def find_checkpoints():
    """Scan for all available checkpoint files."""
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"

    if not checkpoint_dir.exists():
        return []

    # Find all .zip files in checkpoints subdirectories
    checkpoints = []
    for stage_dir in sorted(checkpoint_dir.glob("*/")):
        if stage_dir.is_dir():
            for ckpt_file in sorted(stage_dir.glob("*.zip")):
                checkpoints.append(ckpt_file)

    return checkpoints


def select_checkpoint_interactive():
    """Show menu to select a checkpoint."""
    checkpoints = find_checkpoints()

    if not checkpoints:
        print("No checkpoints found in checkpoints/ directory")
        print("Train a model first with: python train_stage_1.py")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("AVAILABLE CHECKPOINTS")
    print("=" * 70)

    for i, ckpt in enumerate(checkpoints, 1):
        # Extract stage and step info from path
        stage = ckpt.parent.name
        filename = ckpt.name
        print(f"  {i}. {stage}/{filename}")

    print("=" * 70)

    while True:
        try:
            choice = input(f"\nSelect checkpoint (1-{len(checkpoints)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                return checkpoints[idx]
            else:
                print(f"Please enter a number between 1 and {len(checkpoints)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nCancelled")
            sys.exit(0)


def get_env_for_checkpoint(checkpoint_path):
    """Determine which environment to use based on checkpoint path."""
    path_str = str(checkpoint_path)

    if "stage_1" in path_str:
        return Stage1Task(render_mode="human"), "Stage 1"
    elif "stage_2" in path_str:
        if Stage2Task is None:
            print("Warning: Stage 2 not implemented yet, using Stage 1 environment")
            return Stage1Task(render_mode="human"), "Stage 2 (using Stage 1 env)"
        return Stage2Task(render_mode="human"), "Stage 2"
    else:
        # Default to Stage1
        return Stage1Task(render_mode="human"), "Stage 1"


def main():
    parser = argparse.ArgumentParser(description="Watch SO-101 agent with live visualization")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model (.zip). If not provided, shows menu.")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of episodes to watch")
    parser.add_argument("--skip-random", action="store_true",
                        help="Skip random agent demonstration")

    args = parser.parse_args()

    # Interactive mode if no model specified
    if args.model is None:
        print("\n" + "=" * 70)
        print("SO-101 AGENT LIVE VISUALIZATION")
        print("=" * 70)

        # Select checkpoint
        checkpoint_path = select_checkpoint_interactive()

        # Ask for number of episodes if not specified
        if args.episodes is None:
            while True:
                try:
                    episodes_input = input("\nHow many episodes to watch? [default: 3]: ").strip()
                    if episodes_input == "":
                        n_episodes = 3
                        break
                    n_episodes = int(episodes_input)
                    if n_episodes > 0:
                        break
                    else:
                        print("Please enter a positive number")
                except ValueError:
                    print("Please enter a valid number")
                except KeyboardInterrupt:
                    print("\nCancelled")
                    sys.exit(0)
        else:
            n_episodes = args.episodes

        # Ask about random agent demo
        if not args.skip_random:
            show_random = input("\nShow random agent comparison? [Y/n]: ").strip().lower()
            skip_random = show_random == 'n'
        else:
            skip_random = True

        model_path = checkpoint_path
    else:
        # Direct mode with specified model
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"Error: Model not found at {args.model}")
            print("Train a model first with: python train_stage_1.py")
            return

        n_episodes = args.episodes if args.episodes else 3
        skip_random = args.skip_random

    # Detect if running on headless system
    is_headless = os.environ.get("DISPLAY") is None
    auto_start = is_headless

    if is_headless:
        print("\n" + "=" * 70)
        print("HEADLESS MODE DETECTED")
        print("=" * 70)
        print("No display available - rendering to offscreen buffer")
        print("Note: cv2 window display not available on headless systems")
        print("Tip: Use 'python evaluate.py --record' to create MP4 videos instead")
        print("=" * 70)

    print("\n" + "=" * 70)
    print("STARTING VISUALIZATION")
    print("=" * 70)
    print(f"Model: {model_path.name}")
    print(f"Episodes: {n_episodes}")
    print("=" * 70)

    # Create environment with rendering
    print("\nCreating environment with visualization...")
    env, stage_name = get_env_for_checkpoint(model_path)
    print(f"Environment: {stage_name}")

    # Load model
    print("Loading trained model...")
    model = SAC.load(str(model_path))
    print("Model loaded!")

    # Watch random agent first (if requested)
    if not skip_random:
        watch_random_agent(env, n_episodes=2, auto_start=auto_start)

    # Watch trained agent
    watch_trained_agent(env, model, n_episodes=n_episodes, auto_start=auto_start)

    env.close()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print("\nKey takeaway:")
    print("  - Random agent: moves aimlessly")
    print(f"  - Trained agent: performs {stage_name} task")
    print(f"\nTo record a video: python evaluate.py --model {model_path} --record")


if __name__ == "__main__":
    main()
