"""
STAGE 1: Base Rotation - Align with Cube (First Gate of 3-Stage Curriculum)

STATUS: 🔄 IN PROGRESS
Date Started: 2026-01-29
Best Checkpoint: None yet
Success Rate: 0%
Target: 25K steps, >75% success
Next Stage: Stage 2 (Claw Positioning)

================================================================================
CURRICULUM OVERVIEW (3 STAGES)
================================================================================
Stage 1 (THIS): Base Rotation - Align with cube
Stage 2 (NEXT): Claw Positioning - Get claw to cube (gated by Stage 1)
Stage 3 (FINAL): Grasp & Lift - Close gripper and lift (gated by Stage 1+2)

================================================================================
TASK OBJECTIVE
================================================================================
Learn the most basic control: rotate base joint (shoulder_pan) to face the cube.

PRIMARY GOALS:
    1. **Align base with cube**: Rotate shoulder_pan so robot "faces" the cube
    2. **Learn 6D control**: All joints controllable (rewards focus on base rotation)
    3. **Foundation for Stage 2**: Must face cube before positioning claw

SPECIFIC REQUIREMENTS:
    - Start with arm extended forward (home position)
    - Cube is placed randomly on table (±20cm, 24-37cm from base)
    - Success: Base rotation aligned within 15 degrees of cube direction
    - All joints can move, but rewards only consider shoulder_pan alignment
    - Hold alignment for 10 consecutive steps

Action Space: 6D (all joints for sim-to-real transfer)
Control Method: Actuator position control
Constraints: wrist_roll fixed at 90° (horizontal jaws for optimal gripping)

Success Criteria: Align base with cube and hold for 10 consecutive steps

================================================================================
REWARD STRUCTURE
================================================================================

OBJECT POSITION:
    Cube placed randomly on table (X: ±12cm, Y: 18-31cm from base)
    Cube is visible and stationary

HOME POSITION (starting configuration):
    shoulder_pan: 0.0 (facing forward)
    All other joints: folded/safe pose (fixed, don't move)

ALIGNMENT THRESHOLD:
    Angular error < 15 degrees (0.26 radians) = aligned

REWARDS (sparse, no penalties):
    +100.0  when aligned with cube (angle error < 15°)
    0.0     otherwise (pure exploration)

HOLD REQUIREMENT:
    Must maintain alignment for 10 consecutive steps
    If alignment lost (>15°), counter resets

Example episode:
    Start: shoulder_pan at 0.0, cube at (0.1, 0.25) → need to rotate ~22°
    Step 1-15: Exploring, rotating left/right (reward = 0)
    Step 16: Aligned! (reward = +100, hold_steps = 1/10)
    Step 17: Still aligned (reward = +100, hold_steps = 2/10)
    ...
    Step 25: Still aligned (reward = +100, hold_steps = 10/10) → SUCCESS!

Total possible reward: ~1000-1500 per episode (100 × 10-15 steps aligned)

Why "base rotation" as Stage 1?
    ✅ Simplest possible task - only 1 degree of freedom
    ✅ Fast learning - minimal action space
    ✅ Foundation for reaching - must face cube first
    ✅ Natural progression: Align → Reach → Touch → Grasp → Lift

================================================================================
TRAINING PLAN (CURRICULUM LEARNING)
================================================================================
Target: 25K steps per stage
Goal: >75% success rate to advance

Stage 1 (25K): Touch cube (contact within 3cm, any gripper state)
Stage 2C (25K): Touch + Gripper (contact + gripper at 90°)
Stage 3 (25K): Grasp object (close gripper around cube)
Stage 4 (25K): Lift object (raise 5cm above table)
Stage 5 (25K): Place object (move to target position)

Total to full pick-and-place: ~125K steps
Each stage builds on previous checkpoint

================================================================================
WHEN TO MARK SUCCESSFUL
================================================================================
Success criteria to advance to Stage 2C:
    ✅ >75% success rate over 100 test episodes
    ✅ Average distance to cube < 4cm
    ✅ Consistent contact (no exploits)
    ✅ Smooth approach (not erratic movement)

Expected at 25K steps checkpoint

Update this header to:
    STATUS: ✅ SUCCESSFUL
    Date Completed: 2026-01-XX
    Best Checkpoint: stage_1_task_25000_steps.zip
    Success Rate: XX%
    Average Distance: X.Xcm
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

# Handle imports for both direct execution and module import
try:
    from envs.so101_base_env import SO101BaseEnv
except ModuleNotFoundError:
    from so101_base_env import SO101BaseEnv


class Stage1Task(SO101BaseEnv):
    """Stage 1: Base Rotation - Align shoulder_pan with cube (1-DOF)"""

    # Success criteria
    ALIGNMENT_THRESHOLD = 0.26  # Within 15 degrees (radians) = aligned
    ALIGNMENT_REWARD = 100.0    # Reward for aligning with cube
    REQUIRED_HOLD_STEPS = 10    # Must hold alignment for 10 steps to succeed

    # Home position - arm extended straight forward
    HOME_POSITION = {
        "shoulder_pan": 0.0,         # Centered (no left/right rotation)
        "shoulder_lift": 0.0,        # 0° (horizontal, pointing forward)
        "elbow_flex": 0.0,           # 0° (straight, extended forward)
        "wrist_flex": 0.8,           # Points gripper downward toward table
        "wrist_roll": 1.5708,        # 90° = π/2 (jaws horizontal for grasping)
        "gripper": 0.9,              # Wide open, ready to descend
    }

    def __init__(self, render_mode=None, freeze_object=False, curriculum_learning=True):
        super().__init__(render_mode=render_mode)

        # Target tracking
        self.hold_steps = 0          # How many steps at target
        self._episode_count = 0      # Track total episodes

        # Legacy parameters (kept for compatibility)
        self.curriculum_learning = curriculum_learning
        self.freeze_object = freeze_object
        self._frozen_object_pos = None

        # Action space: 6D (all joints for sim-to-real transfer)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # Build joint scaling arrays for actuator control
        self._joint_low = np.array([self.JOINT_LIMITS[name][0] for name in
                                     ["shoulder_pan", "shoulder_lift", "elbow_flex",
                                      "wrist_flex", "wrist_roll", "gripper"]])
        self._joint_high = np.array([self.JOINT_LIMITS[name][1] for name in
                                      ["shoulder_pan", "shoulder_lift", "elbow_flex",
                                       "wrist_flex", "wrist_roll", "gripper"]])
        self._joint_mid = (self._joint_low + self._joint_high) / 2
        self._joint_range = (self._joint_high - self._joint_low) / 2

        # Gripper range for normalization
        self.gripper_min = self.JOINT_LIMITS["gripper"][0]  # -0.17453 (closed)
        self.gripper_max = self.JOINT_LIMITS["gripper"][1]  # 1.74533 (open)

    def step(self, action):
        """Execute one step with actuator position control (6D action space, but Stage 1 only rewards shoulder_pan)."""
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)

        # Convert normalized action to target joint positions
        target_qpos = self._joint_mid + action * self._joint_range

        # Apply to all actuators (6D for sim-to-real consistency)
        # But Stage 1 rewards only consider shoulder_pan alignment
        self.data.ctrl[:6] = target_qpos

        # Override wrist_roll to keep jaws horizontal (fixed at 90°)
        # Index 4 = wrist_roll
        self.data.ctrl[4] = self.HOME_POSITION["wrist_roll"]

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Freeze object position if enabled (curriculum learning)
        if self.freeze_object and self._frozen_object_pos is not None:
            self.data.qpos[self.object_qpos_start:self.object_qpos_start + 3] = self._frozen_object_pos
            self.data.qvel[self.object_qpos_start:self.object_qpos_start + 3] = 0  # Zero velocity
            mujoco.mj_forward(self.model, self.data)  # Update physics

        # Get observation
        obs = self._get_obs()

        # Increment step counter
        self._step_count += 1

        # Calculate reward
        reward = self._compute_reward()

        # Check if aligned with cube
        is_aligned = self._is_aligned()

        if is_aligned:
            self.hold_steps += 1
        else:
            self.hold_steps = 0

        # Check termination (Stage 1: held alignment for required steps)
        success = self.hold_steps >= self.REQUIRED_HOLD_STEPS
        terminated = success

        # Truncation: max steps
        truncated = self._step_count >= self.MAX_EPISODE_STEPS

        # Get alignment angle error
        angle_error = self._get_angle_error()

        # Info
        info = {
            "is_success": success,
            "is_aligned": is_aligned,
            "angle_error": angle_error,
            "hold_steps": self.hold_steps,
            "shoulder_pan_angle": self.data.qpos[0],  # Current base angle
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward(self):
        """STAGE 1: Reward for aligning shoulder_pan with cube direction.

        Returns +100 for aligning base with cube (angle error < 15°).
        No penalties for exploring.
        """
        # Check if aligned
        if self._is_aligned():
            return self.ALIGNMENT_REWARD
        else:
            return 0.0

    def _get_angle_error(self):
        """Calculate angle error between shoulder_pan and cube direction."""
        # Get cube position in world frame
        cube_pos = self.data.site_xpos[self.object_site_id]

        # Robot base is at origin (0, 0) in XY plane
        # Calculate desired angle to face cube
        desired_angle = np.arctan2(cube_pos[0], cube_pos[1])  # atan2(x, y) for shoulder_pan

        # Get current shoulder_pan angle
        current_angle = self.data.qpos[0]

        # Calculate angular error (shortest path)
        angle_error = desired_angle - current_angle
        # Normalize to [-pi, pi]
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))

        return abs(angle_error)

    def _is_aligned(self):
        """Check if shoulder_pan is aligned with cube direction."""
        angle_error = self._get_angle_error()
        return angle_error < self.ALIGNMENT_THRESHOLD

    def _is_lifted(self):
        """Check if object is ACTUALLY lifted with grasp (not just pushed)."""
        # 1. Check if object is lifted
        obj_pos = self.data.site_xpos[self.object_site_id]
        lift_height = obj_pos[2] - self._initial_obj_height
        is_high_enough = lift_height > self.LIFT_HEIGHT

        if not is_high_enough:
            return False

        # 2. Check if gripper is closed (actually grasping)
        gripper_state = self._get_gripper_state()
        gripper_is_closed = gripper_state < self.GRASP_THRESHOLD

        if not gripper_is_closed:
            return False

        # 3. Check if gripper is near/touching object (has contact)
        ee_pos = self.data.site_xpos[self.ee_site_id]
        distance = np.linalg.norm(ee_pos - obj_pos)
        has_contact = distance < self.CONTACT_DISTANCE

        if not has_contact:
            return False

        # All conditions met: lifted + closed gripper + contact
        return True

    def _get_gripper_state(self):
        """Get normalized gripper state [0=closed, 1=open]."""
        gripper_qpos = self.data.qpos[self.gripper_joint_idx]
        return (gripper_qpos - self.gripper_min) / (self.gripper_max - self.gripper_min)

    def reset(self, seed=None, options=None):
        """Reset to initial state with curriculum learning."""
        # Handle seeding (from Gym API)
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)

        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)

        # Increment episode count for curriculum
        self._episode_count += 1

        # === RANDOMIZE OBJECT POSITION on table ===
        # Randomize cube position on table (within easy reach, not too close)
        target_world_x = self.np_random.uniform(-0.20, 0.20)  # Left-right (wider arc, ±20cm)
        target_world_y = self.np_random.uniform(0.24, 0.37)   # Front-back (24-37cm from base, 6cm further forward)

        # Set object position (qpos = target_world - XML_default_pos)
        # XML default is now (0.0, 0.4, 0.015) - centered and far from camera
        self.data.qpos[self.object_qpos_start] = target_world_x - 0.0
        self.data.qpos[self.object_qpos_start + 1] = target_world_y - 0.4
        self.data.qpos[self.object_qpos_start + 2] = 0  # Keep at table height

        # Hide goal marker (not used in Stage 1 - just learning to touch cube)
        # Move it underground so it's not visible
        self.data.qpos[self.goal_qpos_start] = 0.0 - 0.2
        self.data.qpos[self.goal_qpos_start + 1] = 0.0 - 0.35
        self.data.qpos[self.goal_qpos_start + 2] = -1.0 - 0.05  # Underground

        # === SET ARM TO HOME POSITION (FOLDED) ===
        # Robot starts folded at neutral/middle position
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        for idx, joint_name in enumerate(joint_names):
            self.data.qpos[idx] = self.HOME_POSITION[joint_name]

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

        # Get observation
        obs = self._get_obs()

        # Store initial object height
        self._initial_obj_height = self.data.site_xpos[self.object_site_id][2]

        # Store frozen object position (if using freeze mode)
        if self.freeze_object:
            self._frozen_object_pos = self.data.qpos[self.object_qpos_start:self.object_qpos_start + 3].copy()

        # Reset counters (Stage 1: track hold steps at target)
        self.hold_steps = 0       # No steps at target yet
        self._step_count = 0

        info = {}
        return obs, info


# Test the environment
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    env = Stage1Task()
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial gripper state: {env._get_gripper_state():.3f}")

    # Test a few random steps
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {i+1}: reward={reward:.2f}, dist={info['distance_to_cube']:.3f}, touching={info['touching']}")
        if terminated or truncated:
            break

    print(f"\nTotal reward: {total_reward:.2f}")
    env.close()
