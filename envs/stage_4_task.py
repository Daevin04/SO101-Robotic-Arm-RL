"""
STAGE 4: Penalty-Based Positioning (Urgency Through Negative Rewards)

STATUS: 🆕 NEW STAGE (Penalty-Driven Design)
Date Started: 2026-01-29
Best Checkpoint: None yet
Success Rate: TBD
Target: 100K+ steps
Resume From: checkpoints/stage_2/stage_2_task_50000_steps.zip
Next Stage: Stage 5 (Grasp & Lift - Future)

================================================================================
CURRICULUM OVERVIEW (4 STAGES)
================================================================================
Stage 1 (PREV): Base Rotation - Align with cube ✅
Stage 2 (PREV): Claw Positioning - Get claw to cube (±2-3cm tolerance) ✅
Stage 3 (PREV): Infinite Hold - Continuous rewards for positioning
Stage 4 (THIS): Penalty-Based - PUNISH out-of-position, SPARSE REWARD for position

================================================================================
TASK OBJECTIVE
================================================================================
Create URGENCY to achieve and maintain positioning through negative penalties.

PRIMARY GOALS:
    1. **Complete Stage 1 first**: Base must be aligned with cube
    2. **Achieve precise positioning**: Get claw tip within ±1.2-1.5cm
    3. **AVOID PENALTIES**: Lose points every step NOT positioned
    4. **Maximize sparse rewards**: Big reward only when positioned

SPECIFIC REQUIREMENTS:
    - Stage 1 gate: Base aligned within 15° (same as previous stages)
    - Positioning target (Slightly relaxed from Stage 3 for achievability):
        * Y (depth): Within ±1.2cm of cube (was ±1.0cm in Stage 3)
        * X (side): Within ±1.5cm offset (was ±1.2cm in Stage 3)
        * Z (height): Within ±1.2cm of cube (was ±1.0cm in Stage 3)
    - NO EARLY TERMINATION: Episode runs full 150 steps
    - Cube spawn: Narrower range (±15cm, 26-34cm) to ensure reachability

Action Space: 6D (all arm joints + gripper)
Control Method: Actuator position control
Constraints: wrist_roll fixed at 90° (horizontal jaws for optimal gripping)

Success Criteria: Maximize time spent in position (minimize penalties)

================================================================================
REWARD STRUCTURE (PENALTY-DRIVEN DESIGN)
================================================================================

STAGE 1 GATE (must complete first):
    Base must be aligned with cube (angle error < 15°)
    If not aligned: reward = 0

STAGE 2 REWARDS (only after Stage 1 complete):
    +10 * exp(-2 * dist)      Distance reward (guides toward target)
                               - 0cm: +10 reward
                               - 5cm: +3.7 reward
                               - 10cm: +1.4 reward

    +100 EVERY STEP           SPARSE REWARD for achieving precise position:
    (POSITIONED)              - Y within ±1.2cm of cube
                               - X within ±1.5cm (cube between jaws)
                               - Z within ±1.2cm of cube
                               - 150 steps positioned = 15,000 points!

    -20 EVERY STEP            PENALTY for NOT being positioned:
    (NOT POSITIONED)          - Creates urgency to get into position
                               - Wasting time = losing points
                               - 150 steps out of position = -3,000 points!

PENALTIES (MASSIVE - Prevents Exploitation):
    -50 * cube_speed          10x penalty for moving/pushing the cube
                              - Applies if cube speed > 0.5 cm/s (very sensitive!)
                              - Pushing = instant negative reward
                              - Makes exploit impossible

Total possible reward per episode:
    - Perfect positioning all 150 steps: ~16,500 points (+100/step × 150)
    - Never positioned: -3,000 points (-20/step × 150)
    - Brief positioning (10 steps): ~-1,800 points (10×100 - 140×20)

Why penalty-driven design?
    ✅ Creates URGENCY - can't just explore, must get to target
    ✅ Clear signal - positioned = good, not positioned = bad
    ✅ Sparse rewards work better with penalty baseline
    ✅ Prevents aimless wandering - every step matters
    ✅ Forces commitment to maintaining position once achieved

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


class Stage4Task(SO101BaseEnv):
    """Stage 4: Penalty-Based Positioning (Urgency Through Negative Rewards)"""

    # Stage 1 gate criteria (same as previous stages)
    ALIGNMENT_THRESHOLD = 0.26  # Within 15 degrees = Stage 1 complete

    # Stage 4 positioning criteria - Slightly relaxed from Stage 3 for achievability
    MAX_Y_DIFF = 0.012           # 1.2cm - depth (was 1.0cm in Stage 3)
    MAX_X_DIFF = 0.015           # 1.5cm - side offset (was 1.2cm in Stage 3)
    MAX_Z_DIFF = 0.012           # 1.2cm - height (was 1.0cm in Stage 3)
    CLAW_OFFSET_X = 0.025        # 2.5cm offset from cube center

    # Penalty-driven reward structure
    SPARSE_POSITION_REWARD = 100.0   # Large sparse reward when positioned
    OUT_OF_POSITION_PENALTY = -20.0  # Penalty every step NOT positioned
    DISTANCE_REWARD_SCALE = 10.0     # Scale for distance-based reward
    DISTANCE_REWARD_FALLOFF = 2.0    # Moderate falloff (for guidance)

    # Massive penalties to prevent exploitation
    CUBE_MOVEMENT_PENALTY_SCALE = -50.0  # 10x stronger (was -5.0)
    CUBE_MOVEMENT_THRESHOLD = 0.005      # 0.5cm/s (was 3cm/s - 6x more sensitive)

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
        self.hold_steps = 0          # How many steps at precise position
        self._episode_count = 0      # Track total episodes
        self._stage1_complete = False # Stage 1 gate (base aligned)

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

        # Check Stage 1 gate (base aligned)
        self._stage1_complete = self._is_aligned()

        # Check if precisely positioned (tight tolerances)
        is_positioned = self._is_positioned()

        # Track consecutive steps in position (for info only)
        if is_positioned and self._stage1_complete:
            self.hold_steps += 1
        else:
            self.hold_steps = 0

        # Calculate reward (penalty-driven)
        reward = self._compute_reward()

        # NO EARLY TERMINATION - penalty-driven design
        # Agent penalized every step not positioned, creating urgency
        terminated = False

        # Truncation: max steps (only way episode ends)
        truncated = self._step_count >= self.MAX_EPISODE_STEPS

        # Get metrics
        angle_error = self._get_angle_error()
        distance_to_target = self._get_distance_to_target()

        # Info
        info = {
            "is_success": False,  # No early termination, success = max hold_steps
            "stage1_complete": self._stage1_complete,
            "is_positioned": is_positioned,
            "angle_error": angle_error,
            "distance_to_target": distance_to_target,
            "hold_steps": self.hold_steps,
            "shoulder_pan_angle": self.data.qpos[0],
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward(self):
        """STAGE 4: Penalty-driven design - punish out-of-position, reward positioned.

        Stage 1 gate → Distance guidance + SPARSE positioned reward + OUT-OF-POSITION PENALTY
        """
        # Check Stage 1 gate
        if not self._stage1_complete:
            return 0.0

        # Stage 1 passed! Apply penalty-driven reward structure
        distance = self._get_distance_to_target()

        # Distance reward for guidance (moderate falloff)
        distance_reward = self.DISTANCE_REWARD_SCALE * np.exp(
            -self.DISTANCE_REWARD_FALLOFF * distance
        )

        # Check if positioned
        is_positioned = self._is_positioned()

        # MAIN REWARD STRUCTURE: Penalty-driven
        if is_positioned:
            # SPARSE REWARD: Big reward for being in position
            position_reward = self.SPARSE_POSITION_REWARD
        else:
            # PENALTY: Lose points every step NOT positioned
            # This creates urgency to get into position and stay there
            position_reward = self.OUT_OF_POSITION_PENALTY

        # MASSIVE penalty for moving the cube (prevents exploitation)
        # Pushing cube to make positioning easier = heavily penalized
        obj_vel = self.data.qvel[self.n_robot_joints:self.n_robot_joints+3]
        obj_speed = np.linalg.norm(obj_vel)
        movement_penalty = 0.0
        if obj_speed > self.CUBE_MOVEMENT_THRESHOLD:  # 0.5 cm/s
            movement_penalty = self.CUBE_MOVEMENT_PENALTY_SCALE * obj_speed

        return distance_reward + position_reward + movement_penalty

    def _get_angle_error(self):
        """Calculate angle error between shoulder_pan and cube direction (Stage 1)."""
        cube_pos = self.data.site_xpos[self.object_site_id]
        desired_angle = np.arctan2(cube_pos[0], cube_pos[1])
        current_angle = self.data.qpos[0]
        angle_error = desired_angle - current_angle
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
        return abs(angle_error)

    def _is_aligned(self):
        """Check if shoulder_pan is aligned with cube (Stage 1 gate)."""
        angle_error = self._get_angle_error()
        return angle_error < self.ALIGNMENT_THRESHOLD

    def _get_distance_to_target(self):
        """Calculate distance from claw tip to target position."""
        claw_tip_pos = self.data.site_xpos[self.left_fingertip_id]
        cube_pos = self.data.site_xpos[self.object_site_id]
        target_pos = cube_pos.copy()
        target_pos[0] += self.CLAW_OFFSET_X
        distance = np.linalg.norm(claw_tip_pos - target_pos)
        return distance

    def _is_positioned(self):
        """Check if claw tip is positioned correctly (tight tolerances ±1.2-1.5cm)."""
        claw_tip_pos = self.data.site_xpos[self.left_fingertip_id]
        cube_pos = self.data.site_xpos[self.object_site_id]
        target_pos = cube_pos.copy()
        target_pos[0] += self.CLAW_OFFSET_X

        y_diff = abs(claw_tip_pos[1] - target_pos[1])
        x_diff = abs(claw_tip_pos[0] - target_pos[0])
        z_diff = abs(claw_tip_pos[2] - target_pos[2])

        return (y_diff < self.MAX_Y_DIFF and
                x_diff < self.MAX_X_DIFF and
                z_diff < self.MAX_Z_DIFF)


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
        # Narrower + closer spawn to ensure all positions are reachable
        target_world_x = self.np_random.uniform(-0.15, 0.15)  # Left-right (±15cm, was ±20cm)
        target_world_y = self.np_random.uniform(0.26, 0.34)   # Front-back (26-34cm, was 24-37cm)

        # Set object position
        self.data.qpos[self.object_qpos_start] = target_world_x - 0.0
        self.data.qpos[self.object_qpos_start + 1] = target_world_y - 0.4
        self.data.qpos[self.object_qpos_start + 2] = 0  # Keep at table height

        # Hide goal marker (not used in Stage 4)
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

    env = Stage4Task()
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
        print(f"Step {i+1}: reward={reward:.2f}, stage1={info['stage1_complete']}, positioned={info['is_positioned']}, hold_steps={info['hold_steps']}")
        if terminated or truncated:
            break

    print(f"\nTotal reward: {total_reward:.2f}")
    env.close()
