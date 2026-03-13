"""
Stage 7 with Scripted Initial Approach

Adds scripted movement phase before agent control.
Agent learns fine positioning after script brings gripper near cube.
"""

import numpy as np
from envs.stage_7_task import Stage7Task
import mujoco


class Stage7Scripted(Stage7Task):
    """Stage 7 with optional scripted initial approach."""

    # Scripted approach configuration
    USE_SCRIPTED_APPROACH = True       # Enable scripted initial movement
    SCRIPTED_APPROACH_STEPS = 50       # Steps for scripted approach
    SCRIPTED_HANDOFF_DISTANCE = 0.05   # Distance at handoff (5cm from cube)

    def __init__(self, render_mode=None, freeze_object=False):
        super().__init__(render_mode=render_mode, freeze_object=freeze_object)
        self.scripted_phase = False     # Currently in scripted phase?
        self.scripted_step_count = 0    # Steps in scripted phase

    def reset(self, seed=None, options=None):
        """Reset with optional scripted approach phase."""
        obs, info = super().reset(seed=seed, options=options)

        # Start in scripted phase if enabled
        self.scripted_phase = self.USE_SCRIPTED_APPROACH
        self.scripted_step_count = 0

        return obs, info

    def step(self, action):
        """Step with scripted approach phase."""

        # PHASE 1: SCRIPTED APPROACH (if enabled and in scripted phase)
        if self.scripted_phase:
            # Override agent action with scripted movement
            action = self._get_scripted_action()

            # Execute scripted action
            obs, reward, terminated, truncated, info = super().step(action)

            self.scripted_step_count += 1

            # Check if ready for handoff to agent
            if self._check_handoff_condition():
                print(f"[Scripted] Handoff to agent at step {self.scripted_step_count}")
                print(f"  Distance: {info['distance_to_target']:.4f}m")
                self.scripted_phase = False  # Agent takes over!

            # Check if scripted phase timeout
            if self.scripted_step_count >= self.SCRIPTED_APPROACH_STEPS:
                print(f"[Scripted] Timeout at step {self.scripted_step_count}, handoff to agent")
                self.scripted_phase = False

            # Mark in info that we're in scripted phase
            info['scripted_phase'] = True
            info['scripted_step'] = self.scripted_step_count

            return obs, reward, terminated, truncated, info

        # PHASE 2: AGENT CONTROL (normal RL)
        else:
            obs, reward, terminated, truncated, info = super().step(action)
            info['scripted_phase'] = False
            return obs, reward, terminated, truncated, info

    def _get_scripted_action(self):
        """
        Generate scripted action to move gripper toward cube.

        Strategy: Move end-effector toward cube position using inverse direction.
        """
        # Get current positions
        eef_pos = self.data.site_xpos[self.ee_site_id]
        cube_pos = self.data.site_xpos[self.object_site_id]

        # Calculate direction to cube
        direction = cube_pos - eef_pos
        distance = np.linalg.norm(direction)

        # Normalize direction
        if distance > 0.001:
            direction_normalized = direction / distance
        else:
            direction_normalized = np.zeros(3)

        # Simple scripted movement: Move joints based on direction
        # This is a simplified approach - you could use IK instead
        action = np.zeros(6, dtype=np.float32)

        # X direction (shoulder_pan)
        action[0] = np.clip(direction_normalized[0] * 2.0, -1.0, 1.0)

        # Y direction (shoulder_lift + elbow)
        action[1] = np.clip(direction_normalized[1] * 2.0, -1.0, 1.0)
        action[2] = np.clip(direction_normalized[1] * 1.5, -1.0, 1.0)

        # Z direction (wrist_flex)
        action[3] = np.clip(direction_normalized[2] * 2.0, -1.0, 1.0)

        # Keep wrist_roll and gripper at default (handled by parent step())
        action[4] = 0.0  # wrist_roll (locked anyway)
        action[5] = 0.0  # gripper (frozen open)

        return action

    def _check_handoff_condition(self):
        """
        Check if ready to hand off control to agent.

        Returns True when gripper is close enough to cube.
        """
        eef_pos = self.data.site_xpos[self.ee_site_id]
        cube_pos = self.data.site_xpos[self.object_site_id]
        distance = np.linalg.norm(eef_pos - cube_pos)

        # Handoff when within 5cm of cube
        return distance <= self.SCRIPTED_HANDOFF_DISTANCE


# Test script
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    print("="*80)
    print("STAGE 7 SCRIPTED: Testing Scripted Approach")
    print("="*80)

    env = Stage7Scripted()

    print(f"\nConfiguration:")
    print(f"  USE_SCRIPTED_APPROACH: {env.USE_SCRIPTED_APPROACH}")
    print(f"  SCRIPTED_APPROACH_STEPS: {env.SCRIPTED_APPROACH_STEPS}")
    print(f"  SCRIPTED_HANDOFF_DISTANCE: {env.SCRIPTED_HANDOFF_DISTANCE}m")

    obs, info = env.reset()

    print(f"\nRunning episode with scripted approach...")

    for step in range(100):
        # Agent action (will be overridden if in scripted phase)
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        if info.get('scripted_phase', False):
            if step % 10 == 0:
                print(f"  [SCRIPT] Step {step}: distance={info['distance_to_target']:.4f}m")
        else:
            if step == info.get('scripted_step', 0):
                print(f"  [AGENT] Taking over at step {step}!")
            if step % 10 == 0:
                print(f"  [AGENT] Step {step}: distance={info['distance_to_target']:.4f}m")

        if terminated or truncated:
            break

    print(f"\nEpisode complete: {step+1} steps")
    env.close()
