"""
STAGE 5: Perfect Positioning (Binary Reward - No Exploitation)

STATUS: 🆕 NEW STAGE (Pure Binary Reward)
Date Started: 2026-01-29
Best Checkpoint: None yet
Success Rate: TBD
Target: 100K+ steps
Resume From: checkpoints/stage_4/stage_4_task_50000_steps.zip
Next Stage: TBD

================================================================================
CURRICULUM OVERVIEW (5 STAGES)
================================================================================
Stage 1 (PREV): Base Rotation - Align with cube ✅
Stage 2 (PREV): Claw Positioning - Get claw to cube (±2-3cm tolerance) ✅
Stage 3 (PREV): Infinite Hold - Continuous rewards for positioning
Stage 4 (PREV): Penalty-Based with distance guidance (±1.2-1.5cm tolerance)
Stage 5 (THIS): Binary Reward - Penalties until perfect, NO DENSE REWARDS

================================================================================
TASK OBJECTIVE
================================================================================
Pure binary reward structure: penalties until perfect position achieved, then
massive sparse rewards. NO dense distance rewards to exploit.

PRIMARY GOALS:
    1. **ESCAPE PENALTIES**: Stop losing points by achieving perfect position
    2. **HOLD PERFECT POSITION**: Get claw tip within ±1mm and HOLD IT
    3. **MAXIMIZE SPARSE REWARDS**: +1000 every step in perfect position

SPECIFIC REQUIREMENTS:
    - Positioning target (TIGHT):
        * Y (depth): Within ±1cm of cube
        * X (side): Within ±1cm offset
        * Z (height): Within ±1cm of cube
    - NO EARLY TERMINATION: Episode runs full 150 steps
    - Cube spawn: Narrower range (±15cm, 26-34cm) to ensure reachability

Action Space: 6D (5 active: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, gripper action ignored)
Control Method: Actuator position control
Constraints:
    - wrist_roll FROZEN at 90° (horizontal jaws)
    - gripper FROZEN at 100% open (fully open - doesn't need to close)

Success Criteria: Maximize time spent in perfect position

================================================================================
REWARD STRUCTURE (PURE BINARY - NO EXPLOITATION)
================================================================================

BINARY REWARD SYSTEM:
    NOT POSITIONED:
        -20 EVERY STEP        Constant penalty until positioned
                              - No guidance, just pain
                              - MUST find position to escape
                              - 150 steps out = -3,000 points

    POSITIONED (±1cm):
        +1000 EVERY STEP      MASSIVE sparse reward for correct position!
                              - PENALTIES STOP when positioned
                              - Pure positive reward (+1000, not +980)
                              - 150 steps positioned = +150,000 points!
                              - Clear signal: this is what we want

ANTI-EXPLOITATION:
    -50 * cube_speed          10x penalty for moving/pushing the cube
                              - Applies if cube speed > 0.5 cm/s
                              - Prevents any cube manipulation

NO DENSE REWARDS:
    ❌ NO distance guidance (removed +10*exp(-2*dist))
    ❌ NO partial credit
    ❌ NO smooth gradients to exploit
    ✅ ONLY binary signal: penalty or massive reward

Total possible reward per episode:
    - Perfect positioning all 150 steps: +150,000 points (MASSIVE!)
    - Never positioned: -3,000 points (painful)
    - 10 steps positioned: (10×1000) + (140×-20) = +7,200 points
    - 50 steps positioned: (50×1000) + (100×-20) = +48,000 points

Why this design works:
    ✅ No dense rewards to exploit
    ✅ Clear binary signal: wrong or RIGHT
    ✅ Massive reward creates strong signal
    ✅ Penalties create urgency to find solution
    ✅ Can't game the system - either positioned or not
    ✅ Forces agent to discover perfect positioning through exploration

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


class Stage5Task(SO101BaseEnv):
    """Stage 5: Perfect Positioning (Near-Zero Tolerance)"""

    # Stage 5 positioning criteria - BETWEEN JAWS (more intuitive)
    MAX_PERPENDICULAR_OFFSET = 0.010  # 1cm - how far cube can be from jaw centerline
    MAX_FORWARD_OFFSET = 0.010        # 1cm - how far cube can be forward/back from jaw plane
    MAX_HEIGHT_OFFSET = 0.010         # 1cm - how far cube can be above/below jaw height

    # BINARY reward structure - NO DENSE REWARDS
    PERFECT_POSITION_REWARD = 1000.0  # MASSIVE sparse reward when positioned
    OUT_OF_POSITION_PENALTY = -20.0   # Penalty every step NOT positioned

    # Anti-exploitation penalties
    CUBE_MOVEMENT_PENALTY_SCALE = -50.0  # 10x stronger
    CUBE_MOVEMENT_THRESHOLD = 0.005      # 0.5cm/s
    TABLE_CONTACT_PENALTY = -5.0         # Small penalty for touching table

    # Home position - arm extended straight forward
    HOME_POSITION = {
        "shoulder_pan": 0.0,         # Centered (no left/right rotation)
        "shoulder_lift": 0.0,        # 0° (horizontal, pointing forward)
        "elbow_flex": 0.0,           # 0° (straight, extended forward)
        "wrist_flex": 0.8,           # Points gripper downward toward table
        "wrist_roll": 1.5708,        # 90° = π/2 (jaws horizontal for grasping)
        "gripper": 1.0,              # FULLY OPEN - easier to fit cube
    }

    def __init__(self, render_mode=None, freeze_object=False, curriculum_learning=True):
        super().__init__(render_mode=render_mode)

        # Target tracking
        self.hold_steps = 0          # How many steps at precise position
        self._episode_count = 0      # Track total episodes

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

        # Freeze wrist_roll to keep jaws horizontal (fixed at 90°)
        # Index 4 = wrist_roll
        self.data.ctrl[4] = self.HOME_POSITION["wrist_roll"]

        # Freeze gripper to keep it fully open (doesn't need to close in Stage 5)
        # Index 5 = gripper
        self.data.ctrl[5] = self.HOME_POSITION["gripper"]

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

        # Check if precisely positioned (EXTREMELY tight tolerances)
        is_positioned = self._is_positioned()

        # Track consecutive steps in position (for info only)
        if is_positioned:
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
        distance_to_target = self._get_distance_to_target()

        # Info
        info = {
            "is_success": False,  # No early termination, success = max hold_steps
            "is_positioned": is_positioned,
            "distance_to_target": distance_to_target,
            "hold_steps": self.hold_steps,
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward(self):
        """STAGE 5: Pure binary reward - NO dense rewards, NO Stage 1 gate.

        Either losing points (-20/step) or winning BIG (+1000/step). Nothing in between.
        """
        # Check if positioned
        is_positioned = self._is_positioned()

        # BINARY REWARD STRUCTURE
        if is_positioned:
            # MASSIVE SPARSE REWARD: Perfect position = winning
            position_reward = self.PERFECT_POSITION_REWARD  # +1000
        else:
            # PENALTY: Not positioned = losing
            position_reward = self.OUT_OF_POSITION_PENALTY  # -20

        # MASSIVE penalty for moving the cube (prevents exploitation)
        obj_vel = self.data.qvel[self.n_robot_joints:self.n_robot_joints+3]
        obj_speed = np.linalg.norm(obj_vel)
        movement_penalty = 0.0
        if obj_speed > self.CUBE_MOVEMENT_THRESHOLD:  # 0.5 cm/s
            movement_penalty = self.CUBE_MOVEMENT_PENALTY_SCALE * obj_speed

        # Small penalty for touching table (discourages dragging on surface)
        table_penalty = 0.0
        if self._is_touching_table():
            table_penalty = self.TABLE_CONTACT_PENALTY  # -5

        return position_reward + movement_penalty + table_penalty

    def _is_touching_table(self):
        """Check if gripper is in contact with the table."""
        # Get table geom ID
        try:
            table_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        except:
            return False  # Table geom not found

        # Check all contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            # Check if contact involves table and gripper body
            if geom1_id == table_geom_id or geom2_id == table_geom_id:
                # Check if other geom belongs to gripper body
                other_geom = geom2_id if geom1_id == table_geom_id else geom1_id
                body_id = self.model.geom_bodyid[other_geom]

                # Check if body is part of gripper (gripper_body or jaw_body)
                if body_id in [self.gripper_body_id, self.jaw_body_id]:
                    return True

        return False

    def _get_distance_to_target(self):
        """Calculate distance from jaw centerline to cube.

        Returns the perpendicular distance from cube to the line between fingertips.
        """
        left_tip = self.data.site_xpos[self.left_fingertip_id]
        right_tip = self.data.site_xpos[self.right_fingertip_id]
        cube_pos = self.data.site_xpos[self.object_site_id]

        # Midpoint between fingertips (jaw centerline)
        jaw_center = (left_tip + right_tip) / 2

        # Distance from cube to jaw centerline
        distance = np.linalg.norm(cube_pos - jaw_center)
        return distance

    def _is_positioned(self):
        """Check if cube is positioned between the jaws (ready for grasping).

        The cube is considered positioned if:
        1. It's between the left and right fingertips (along jaw opening direction)
        2. It's close to the centerline between fingertips (perpendicular offset)
        3. It's aligned in height with the fingertips
        4. It's aligned forward/back with the fingertips

        This is much more intuitive than checking against a fixed target position!
        """
        left_tip = self.data.site_xpos[self.left_fingertip_id]
        right_tip = self.data.site_xpos[self.right_fingertip_id]
        cube_pos = self.data.site_xpos[self.object_site_id]

        # Calculate jaw centerline and jaw opening direction
        jaw_center = (left_tip + right_tip) / 2
        jaw_vector = right_tip - left_tip  # Vector from left to right fingertip
        jaw_opening_dist = np.linalg.norm(jaw_vector)

        # Skip check if jaws are too close (degenerate case)
        if jaw_opening_dist < 0.001:
            return False

        jaw_direction = jaw_vector / jaw_opening_dist  # Normalize

        # 1. Check if cube is between the jaws (project onto jaw opening direction)
        cube_to_left = cube_pos - left_tip
        projection_on_jaw = np.dot(cube_to_left, jaw_direction)

        # Cube should be between 0 (left tip) and jaw_opening_dist (right tip)
        if projection_on_jaw < -0.01 or projection_on_jaw > jaw_opening_dist + 0.01:
            return False  # Cube is outside the jaw span

        # 2. Check perpendicular distance from cube to jaw centerline
        cube_to_center = cube_pos - jaw_center
        # Project perpendicular to jaw direction
        perpendicular_component = cube_to_center - np.dot(cube_to_center, jaw_direction) * jaw_direction
        perpendicular_dist = np.linalg.norm(perpendicular_component)

        if perpendicular_dist > self.MAX_PERPENDICULAR_OFFSET:
            return False  # Cube too far from centerline

        # 3. Check height alignment (Z direction)
        height_diff = abs(cube_pos[2] - jaw_center[2])
        if height_diff > self.MAX_HEIGHT_OFFSET:
            return False  # Cube not aligned in height

        # 4. Check forward/back alignment (Y direction)
        forward_diff = abs(cube_pos[1] - jaw_center[1])
        if forward_diff > self.MAX_FORWARD_OFFSET:
            return False  # Cube not aligned forward/back

        return True


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

        # Hide goal marker (not used in Stage 5)
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

        info = {}
        return obs, info


# Test the environment
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    env = Stage5Task()
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
        print(f"Step {i+1}: reward={reward:.2f}, positioned={info['is_positioned']}, hold_steps={info['hold_steps']}")
        if terminated or truncated:
            break

    print(f"\nTotal reward: {total_reward:.2f}")
    env.close()
