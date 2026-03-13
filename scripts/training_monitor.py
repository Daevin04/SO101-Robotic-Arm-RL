"""
Custom callback to monitor stage-specific metrics during training.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback


class StageMonitorCallback(BaseCallback):
    """
    Callback to monitor and log stage-specific metrics during training.
    Shows metrics every N episodes to catch hacking/exploits early.
    Saves metrics to file for later analysis.
    """

    def __init__(self, check_freq=100, log_dir="logs", stage_name="stage_1", verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq  # Log every N steps
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_metrics = {}

        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{stage_name}_training_{timestamp}.jsonl"
        self.csv_file = self.log_dir / f"{stage_name}_training_{timestamp}.csv"

        # Initialize CSV
        self.csv_initialized = False

        if self.verbose > 0:
            print(f"📊 Logging training metrics to: {self.log_file}")
            print(f"📊 CSV metrics saved to: {self.csv_file}")

    def _on_step(self) -> bool:
        # Check if episode just finished
        if self.locals.get("dones")[0]:
            self.episode_count += 1

            # Get info from the episode
            info = self.locals.get("infos")[0]

            # Store reward
            episode_reward = self.locals.get("rewards")[0]
            self.episode_rewards.append(episode_reward)

            # Store stage-specific metrics
            for key, value in info.items():
                if isinstance(value, (int, float, np.number)):
                    if key not in self.episode_metrics:
                        self.episode_metrics[key] = []
                    self.episode_metrics[key].append(float(value))

            # Log every check_freq episodes
            if self.episode_count % self.check_freq == 0:
                self._print_summary()

        return True

    def _print_summary(self):
        """Print summary of last N episodes and save to log file."""
        if not self.episode_rewards:
            return

        n = min(self.check_freq, len(self.episode_rewards))
        recent_rewards = self.episode_rewards[-n:]

        # Collect metrics for logging
        log_data = {
            "episode": self.episode_count,
            "timestep": self.num_timesteps,
            "mean_reward": float(np.mean(recent_rewards)),
            "std_reward": float(np.std(recent_rewards)),
        }

        # Console output
        print("\n" + "=" * 70)
        print(f"TRAINING MONITOR - Episode {self.episode_count}")
        print("=" * 70)
        print(f"Mean reward (last {n}):  {np.mean(recent_rewards):7.2f}")

        # Show stage-specific metrics
        for key, values in self.episode_metrics.items():
            recent_values = values[-n:]
            mean_val = np.mean(recent_values)
            log_data[f"mean_{key}"] = float(mean_val)

            # Format based on metric type
            if "dist" in key.lower() or "offset" in key.lower() or "height" in key.lower():
                # Distance metrics in cm
                print(f"Mean {key:20s}: {mean_val*100:6.2f}cm")
            elif "success" in key.lower() or "close" in key.lower():
                # Count metrics
                print(f"Mean {key:20s}: {mean_val:6.1f}")
            else:
                # Generic metric
                print(f"Mean {key:20s}: {mean_val:6.2f}")

        print("=" * 70 + "\n")

        # Save to JSONL file (one JSON object per line - easy to parse)
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_data) + '\n')

        # Save to CSV file
        if not self.csv_initialized:
            # Write header
            with open(self.csv_file, 'w') as f:
                f.write(','.join(log_data.keys()) + '\n')
            self.csv_initialized = True

        # Write data row
        with open(self.csv_file, 'a') as f:
            f.write(','.join(str(v) for v in log_data.values()) + '\n')


class EarlyStoppingCallback(BaseCallback):
    """
    Stop training if agent gets stuck (hack detection).

    Checks if reward hasn't improved after N episodes.
    """

    def __init__(self, check_freq=500, patience=1000, min_reward_improvement=10.0, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.patience = patience  # Episodes without improvement before stopping
        self.min_reward_improvement = min_reward_improvement
        self.best_mean_reward = -np.inf
        self.episodes_without_improvement = 0
        self.episode_rewards = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        if self.locals.get("dones")[0]:
            self.episode_count += 1
            episode_reward = self.locals.get("rewards")[0]
            self.episode_rewards.append(episode_reward)

            # Check every check_freq episodes
            if self.episode_count % self.check_freq == 0:
                n = min(self.check_freq, len(self.episode_rewards))
                mean_reward = np.mean(self.episode_rewards[-n:])

                if mean_reward > self.best_mean_reward + self.min_reward_improvement:
                    self.best_mean_reward = mean_reward
                    self.episodes_without_improvement = 0
                    if self.verbose > 0:
                        print(f"\n✓ Improvement detected! New best mean reward: {mean_reward:.2f}\n")
                else:
                    self.episodes_without_improvement += self.check_freq
                    if self.verbose > 0:
                        print(f"\n⚠ No improvement for {self.episodes_without_improvement} episodes "
                              f"(best: {self.best_mean_reward:.2f}, current: {mean_reward:.2f})\n")

                # Stop if stuck
                if self.episodes_without_improvement >= self.patience:
                    if self.verbose > 0:
                        print("\n" + "=" * 70)
                        print("⛔ EARLY STOPPING - Agent appears stuck (possible exploit)")
                        print(f"No improvement for {self.episodes_without_improvement} episodes")
                        print(f"Best reward: {self.best_mean_reward:.2f}")
                        print("Consider: adjusting rewards, curriculum learning, or hyperparameters")
                        print("=" * 70 + "\n")
                    return False

        return True
