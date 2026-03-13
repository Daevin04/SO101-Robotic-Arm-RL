"""
STAGE 2: Claw Positioning - Get Stationary Claw to Cube (Second Gate of 3-Stage Curriculum)

STATUS: 🆕 NEW STAGE
Date Started: 2026-01-29
Best Checkpoint: None yet
Success Rate: 0%
Target: 50K steps, >75% success
Resume From: checkpoints/stage_1/stage_1_task_25000_steps.zip
Next Stage: Stage 3 (Grasp & Lift)

================================================================================
CURRICULUM OVERVIEW (3 STAGES)
================================================================================
Stage 1 (PREV): Base Rotation - Align with cube ✅
Stage 2 (THIS): Claw Positioning - Get claw to cube (gated by Stage 1)
Stage 3 (NEXT): Grasp & Lift - Close gripper and lift (gated by Stage 1+2)

================================================================================
TASK OBJECTIVE
================================================================================
Learn to position the STATIONARY CLAW TIP at the cube to prepare for gripping.

PRIMARY GOALS:
    1. **Complete Stage 1 first**: Base must be aligned with cube
    2. **Position stationary claw**: Move claw tip to cube location
    3. **Optimal grip positioning**: Cube should be INSIDE the claw (between jaws)
    4. **Prepare for Stage 3**: Set up perfect position for closing gripper

SPECIFIC REQUIREMENTS:
    - Stage 1 condition MUST be met first (base aligned within 15°)
    - Stationary claw tip (left_fingertip) must reach target position:
        * Y (depth): Same as cube (within 2cm)
        * X (side): Offset so cube is between jaws (within 3cm)
        * Z (height): Same as cube (within 2cm)
    - Hold position for 10 consecutive steps
    - Gripper stays open throughout

Action Space: 6D (all arm joints + gripper)
Control Method: Actuator position control
Constraints: wrist_roll fixed at 90° (horizontal jaws for optimal gripping)

Success Criteria: Stage 1 complete + claw positioned + hold for 10 steps

================================================================================
REWARD STRUCTURE (GATED)
================================================================================

STAGE 1 GATE (must complete first):
    Base must be aligned with cube (angle error < 15°)
    If not aligned: Only reward = 0, encourage base rotation

STAGE 2 REWARDS (only after Stage 1 complete):
    +10 * exp(-2 * dist)      Distance reward (stationary claw tip to target)
                               - 0cm: +10 reward
                               - 10cm: +2.7 reward
                               - 20cm: +0.7 reward

    +100                       Bonus for achieving optimal position:
                               - Y within 2cm of cube
                               - X within 3cm (cube between jaws)
                               - Z within 2cm of cube
                               - Hold for 10 consecutive steps

PENALTIES:
    -5 * cube_speed           Penalty for moving/pushing the cube
                              - Only applies if cube speed > 3 cm/s

    -10 * gripper_closing     Penalty for closing gripper
                              - Gripper should stay open (state > 0.7)
                              - Prevents premature gripping

TARGET POSITION:
    - Y: Same as cube (forward-back alignment)
    - X: Offset by 3cm from cube center (so cube is between jaws)
    - Z: Same height as cube

Total possible reward per episode: ~1500 (distance) + 1000+ (position bonuses)

Why gated rewards?
    ✅ Forces curriculum learning - must master base rotation first
    ✅ Clear progression: Align → Position → Grip
    ✅ Prevents shortcut behaviors
    ✅ Builds on Stage 1 checkpoint naturally

================================================================================
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


class Stage2Task(SO101BaseEnv):
    """Stage 2: Claw Positioning - Get stationary claw to cube (gated rewards)"""

    # Stage 1 gate criteria (must complete first)
    ALIGNMENT_THRESHOLD = 0.26  # Within 15 degrees = Stage 1 complete

    # Stage 2 success criteria
    POSITION_REWARD = 100.0      # Reward for achieving optimal position
    REQUIRED_HOLD_STEPS = 10     # Must hold position for 10 steps
    DISTANCE_REWARD_SCALE = 10.0 # Scale for distance-based reward

    # Positioning thresholds (for stationary claw tip to target)
    MAX_Y_DIFF = 0.02            # 2cm - depth must match cube
    MAX_X_DIFF = 0.03            # 3cm - side offset (cube between jaws)
    MAX_Z_DIFF = 0.02            # 2cm - height must match cube

    # Cube positioning offset (to get cube between jaws)
    CLAW_OFFSET_X = 0.03         # 3cm offset from cube center

    # Penalties
    CUBE_MOVEMENT_PENALTY_SCALE = -5.0
    GRIPPER_CLOSING_PENALTY_SCALE = -10.0
    GRIPPER_OPEN_THRESHOLD = 0.7  # Gripper should stay > 0.7 (open)

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
        self.hold_steps = 0          # How many steps at optimal position
        self._episode_count = 0      # Track total episodes
        self._stage1_complete = False # Track if Stage 1 gate is passed

        # Legacy parameters (kept for compatibility)
        self.curriculum_learning = curriculum_learning
        self.freeze_object = freeze_object
        self._frozen_object_pos = None

        # Action space: 6D (all arm joints + gripper)
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
        self.gripper_min = self.JOINT_LIMITS["gripper"][0]
        self.gripper_max = self.JOINT_LIMITS["gripper"][1]

    def step(self, action):
        """Execute one step with actuator position control (all joints)."""
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)

        # Convert normalized action to target joint positions
        target_qpos = self._joint_mid + action * self._joint_range

        # Apply to all actuators
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

        # Check Stage 1 condition (base aligned)
        self._stage1_complete = self._is_aligned()

        # Check Stage 2 condition (claw positioned)
        is_positioned = self._is_positioned()

        if is_positioned and self._stage1_complete:
            self.hold_steps += 1
        else:
            self.hold_steps = 0

        # Calculate reward
        reward = self._compute_reward()

        # Check termination (Stage 2: held position for required steps)
        success = self.hold_steps >= self.REQUIRED_HOLD_STEPS
        terminated = success

        # Truncation: max steps
        truncated = self._step_count >= self.MAX_EPISODE_STEPS

        # Get metrics
        angle_error = self._get_angle_error()
        distance_to_target = self._get_distance_to_target()

        # Info
        info = {
            "is_success": success,
            "stage1_complete": self._stage1_complete,
            "is_positioned": is_positioned,
            "angle_error": angle_error,
            "distance_to_target": distance_to_target,
            "hold_steps": self.hold_steps,
            "shoulder_pan_angle": self.data.qpos[0],
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward(self):
        """STAGE 2: Gated reward structure.

        First must complete Stage 1 (base aligned), then reward Stage 2 (claw positioning).
        """
        # Check Stage 1 gate
        if not self._stage1_complete:
            # Stage 1 not complete - no Stage 2 rewards yet
            # Could add small reward for base alignment here if desired
            return 0.0

        # Stage 1 complete! Now give Stage 2 rewards

        # Distance reward (stationary claw tip to target position)
        distance = self._get_distance_to_target()
        distance_reward = self.DISTANCE_REWARD_SCALE * np.exp(-2.0 * distance)

        # Position bonus (if optimally positioned)
        position_reward = self.POSITION_REWARD if self._is_positioned() else 0.0

        # Penalty for moving the cube (pushing instead of positioning)
        obj_vel = self.data.qvel[self.n_robot_joints:self.n_robot_joints+3]
        obj_speed = np.linalg.norm(obj_vel)
        movement_penalty = 0.0
        if obj_speed > 0.03:  # 3 cm/s
            movement_penalty = self.CUBE_MOVEMENT_PENALTY_SCALE * obj_speed

        # Penalty for closing gripper (should stay open in Stage 2)
        gripper_state = self._get_gripper_state()
        gripper_penalty = 0.0
        if gripper_state < self.GRIPPER_OPEN_THRESHOLD:
            # Penalize for closing gripper
            closing_amount = self.GRIPPER_OPEN_THRESHOLD - gripper_state
            gripper_penalty = self.GRIPPER_CLOSING_PENALTY_SCALE * closing_amount

        return distance_reward + position_reward + movement_penalty + gripper_penalty

    def _get_angle_error(self):
        """Calculate angle error between shoulder_pan and cube direction (Stage 1 check)."""
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
        """Check if shoulder_pan is aligned with cube direction (Stage 1 condition)."""
        angle_error = self._get_angle_error()
        return angle_error < self.ALIGNMENT_THRESHOLD

    def _get_distance_to_target(self):
        """Calculate distance from stationary claw tip to target position."""
        # Get stationary claw tip position (left fingertip)
        claw_tip_pos = self.data.site_xpos[self.left_fingertip_id]

        # Get target position (near cube, offset for optimal grip)
        cube_pos = self.data.site_xpos[self.object_site_id]
        target_pos = cube_pos.copy()
        target_pos[0] += self.CLAW_OFFSET_X  # Offset X so cube is between jaws

        # Calculate distance
        distance = np.linalg.norm(claw_tip_pos - target_pos)
        return distance

    def _is_positioned(self):
        """Check if stationary claw tip is at optimal position (ready for gripping)."""
        # Get positions
        claw_tip_pos = self.data.site_xpos[self.left_fingertip_id]
        cube_pos = self.data.site_xpos[self.object_site_id]

        # Target position (offset for optimal grip)
        target_pos = cube_pos.copy()
        target_pos[0] += self.CLAW_OFFSET_X

        # Check each axis
        y_diff = abs(claw_tip_pos[1] - target_pos[1])  # Depth
        x_diff = abs(claw_tip_pos[0] - target_pos[0])  # Side
        z_diff = abs(claw_tip_pos[2] - target_pos[2])  # Height

        # All must be within thresholds
        return (y_diff < self.MAX_Y_DIFF and
                x_diff < self.MAX_X_DIFF and
                z_diff < self.MAX_Z_DIFF)

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

        # Hide goal marker (not used in Stage 2 - just learning to position claw)
        # Move it underground so it's not visible
        self.data.qpos[self.goal_qpos_start] = 0.0 - 0.2
        self.data.qpos[self.goal_qpos_start + 1] = 0.0 - 0.35
        self.data.qpos[self.goal_qpos_start + 2] = -1.0 - 0.05  # Underground

        # === SET ARM TO HOME POSITION ===
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

        # Reset counters
        self.hold_steps = 0
        self._step_count = 0
        self._stage1_complete = False

        info = {}
        return obs, info


# Test the environment
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    env = Stage2Task()
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
        print(f"Step {i+1}: reward={reward:.2f}, stage1={info['stage1_complete']}, positioned={info['is_positioned']}")
        if terminated or truncated:
            break

    print(f"\nTotal reward: {total_reward:.2f}")
    env.close()
