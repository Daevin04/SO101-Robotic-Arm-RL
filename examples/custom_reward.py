#!/usr/bin/env python3
"""
Example: Create a custom training stage with custom reward function.

This demonstrates how to extend the base environment for new tasks.
"""

import sys
from pathlib import Path
import numpy as np
import gymnasium

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.so101_base_env import SO101BaseEnv


class CustomReachTask(SO101BaseEnv):
    """
    Custom task: Reach a target position in 3D space.

    This is a simplified example showing how to:
    - Inherit from SO101BaseEnv
    - Define custom reward function
    - Set custom success criteria
    """

    def __init__(self, render_mode=None, target_position=None):
        """
        Initialize custom reach task.

        Args:
            render_mode: Rendering mode (None or "human")
            target_position: Target 3D position [x, y, z] (optional)
        """
        super().__init__(render_mode=render_mode)

        # Set target position (default: 20cm forward, 25cm away, 5cm up)
        if target_position is None:
            self.target_position = np.array([0.2, 0.25, 0.05])
        else:
            self.target_position = np.array(target_position)

    def _compute_reward(self, action):
        """
        Compute reward based on distance to target.

        Reward components:
        1. Distance reward: Negative distance to target (encourages proximity)
        2. Success bonus: +100 if within 2cm of target
        3. Action penalty: Small penalty for large actions (encourages smoothness)
        """
        # Get end-effector position
        ee_position = self._get_ee_position()

        # Compute distance to target
        distance = np.linalg.norm(ee_position - self.target_position)

        # Reward components
        distance_reward = -distance * 10  # Negative distance scaled by 10

        success_bonus = 0.0
        if distance < 0.02:  # Within 2cm
            success_bonus = 100.0

        action_penalty = -0.01 * np.sum(np.square(action))  # Small action penalty

        # Total reward
        reward = distance_reward + success_bonus + action_penalty

        return reward

    def _is_success(self):
        """Check if task is successful (within 2cm of target)."""
        ee_position = self._get_ee_position()
        distance = np.linalg.norm(ee_position - self.target_position)
        return distance < 0.02

    def _is_terminated(self):
        """Episode ends if success achieved."""
        return self._is_success()

    def _is_truncated(self):
        """Episode truncates after max steps (defined in base class)."""
        return self.step_count >= 500


def train_custom_task():
    """Train the custom reach task."""
    from stable_baselines3 import SAC

    print("="*60)
    print("Training Custom Reach Task")
    print("="*60)

    # Create environment
    env = CustomReachTask()

    # Create SAC agent
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        verbose=1,
    )

    # Train
    print("\nTraining for 10000 steps...")
    model.learn(total_timesteps=10000, progress_bar=True)

    # Save
    model.save("custom_reach_model")
    print("\nModel saved to: custom_reach_model.zip")

    # Evaluate
    print("\nEvaluating for 10 episodes...")
    successes = 0
    for i in range(10):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        success = env._is_success()
        if success:
            successes += 1

        print(f"Episode {i+1}: Reward={episode_reward:.1f}, Success={success}")

    print(f"\nSuccess rate: {successes}/10")
    env.close()


def main():
    """Main function."""
    print("Custom Reward Example\n")
    print("This example shows how to:")
    print("1. Create a custom task by inheriting from SO101BaseEnv")
    print("2. Define a custom reward function")
    print("3. Set custom success criteria")
    print("4. Train the custom task\n")

    # Train the custom task
    train_custom_task()


if __name__ == "__main__":
    main()
