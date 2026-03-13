"""
STAGE 6: Gated Curriculum - Grasp Then Navigate

STATUS: 🆕 GATED CURRICULUM (Two-Stage Learning)
Date Started: 2026-02-10
Best Checkpoint: None yet
Success Rate: TBD
Target: 150K+ steps
Resume From: checkpoints/stage_5/stage_5_task_XXXXX_steps.zip
Next Stage: TBD

================================================================================
GATED CURRICULUM: TWO-STAGE LEARNING
================================================================================
This stage uses a novel two-gate curriculum that:
1. DECOUPLES grasp learning from navigation learning
2. TRANSITIONS rewards once grasping achieved
3. FORCES sequential skill mastery (grasp → navigate while maintaining)

GATE 1: LEARN TO GRASP 🎯
    Goal: Achieve stable grasp of cube
    Method: Dense penalties guide toward grasp position
    Gripper: ACTIVE (agent controls closing)
    Episode: Continues after grasp achieved

GATE 2: LEARN TO NAVIGATE 🎯 (Unlocked after Gate 1)
    Goal: Move cube to target (green circle) while maintaining grasp
    Method: Dense rewards for reducing distance to target
    Gripper: ACTIVE (agent must maintain grasp while navigating!)
    Target: Randomized each episode (10cm height)
    Challenge: Must learn to navigate WITHOUT losing grasp

================================================================================
TASK OBJECTIVE
================================================================================
Learn to grasp the cube, then navigate it to a target position.

PHASE 1 (Gate 1): GRASP THE CUBE
    1. Position jaws around cube
    2. Close gripper to achieve grasp
    3. → Gate 2 unlocked (reward structure changes)

PHASE 2 (Gate 2): NAVIGATE TO TARGET WHILE GRASPING
    1. Maintain grasp (gripper still active - must hold!)
    2. Move arm to bring cube toward green circle
    3. Reach target within 2cm for success bonus
    4. Challenge: Navigate without losing grasp!

SPECIFIC REQUIREMENTS:
    - Grasping: Gripper closed (<0.25) + fingertips near cube (40mm)
    - Target: Green circle spawns randomly (±13cm X, ±8cm Y, 10cm Z)
    - EPISODE LENGTH: 100 steps (3.3 sec) - continues after freeze!
    - NO EARLY TERMINATION: Episodes always run full 100 steps

Action Space: 6D (5 joints + gripper, ALL active throughout episode)
Control Method: Actuator position control
Constraints: wrist_roll FROZEN at 90° (horizontal jaws only)

Success Criteria: Gate 1 complete + cube at target (<2cm distance)

================================================================================
REWARD STRUCTURE (GATED CURRICULUM)
================================================================================

GATE 1: PENALTIES FOR NOT GRASPING (Active until grasp achieved)

    XYZ Positioning Penalties (GENTLE - allows learning gradient):
        -10 per cm in X error   Penalty for misalignment perpendicular to jaws
        -10 per cm in Y error   Penalty for forward/back misalignment
        -10 per cm in Z error   Penalty for height misalignment

        Examples:
        - Perfect alignment (0cm error): 0 penalty
        - 1cm error all axes: -30 penalty
        - 2cm error all axes: -60 penalty
        - 3cm error all axes: -90 penalty

    Grasp Status Penalties (GENTLE):
        -20 EVERY STEP          No jaw contact at all (nowhere near cube)
        -10 EVERY STEP          Partial contact (one jaw touching)
        -5 EVERY STEP           Both jaws touching but not closed enough

    Total Gate 1 penalties: -20 to -100 per step (5× gentler than v1)
    Agent learns: "Minimize penalties → Achieve grasp → Unlock Gate 2!"

GATE 2: REWARDS FOR NAVIGATION (Active after gripper frozen)

    Distance-Based Reward (Dense, GENTLE):
        -20 per cm distance     Penalty for being far from target
                                - At target (0cm): 0 penalty
                                - 10cm away: -200 penalty
                                - 50cm away: -1000 penalty

    Movement Reward (Delta bonus):
        +50 per cm closer       Bonus for moving in right direction!
                                - Getting closer = reward boost
                                - Encourages exploration toward target

    Success Bonus:
        +1000 ONE-TIME          Reaching target (within 2cm threshold)

    Total Gate 2 rewards: Highly variable based on distance and progress
    Agent learns: "Move cube toward green circle → Maximize reward!"

GATING MECHANISM:
    Episode Start → Gate 1 active (penalties drive grasping)
    Grasp Achieved → Gate 2 unlocks (reward structure switches to navigation)
    Episode continues for full 100 steps regardless of gate status
    Agent must maintain grasp during Gate 2 (gripper NOT frozen!)

Total possible rewards per episode (100 steps) - v2 GENTLER PENALTIES:
    GATE 1 ONLY (Never grasped):
        - Poor positioning (3cm errors): -90/step × 100 = -9,000 pts
        - Better positioning (1cm errors): -30/step × 100 = -3,000 pts
        - Almost grasping (touching): -5/step × 100 = -500 pts

    GATE 1 + GATE 2 (Grasped at step 40):
        - Gate 1 phase (40 steps): ~-1,200 pts (learning to grasp)
        - Gate 2 phase (60 steps): Depends on navigation skill
          - Far from target (30cm avg): -600/step × 60 = -36,000 pts
          - Getting closer (momentum): +(50 × progress) bonus
          - Reach target: +1000 bonus
        - Realistic total: -10,000 to +5,000 pts depending on navigation

Learning progression (expected with GATED CURRICULUM v2 - GENTLER):
    Phase 1: GATE 1 - Learn Grasping (0-75K steps)
      → Minimize penalties by improving positioning
      → Learn to make jaw contact
      → Learn to close gripper properly
      → Expected: -9,000 → -3,000 → -500 points
      → UNLOCK Gate 2 when consistently grasping

    Phase 2: GATE 2 - Learn Navigation (75-150K steps)
      → Gate 1 mastered (grasp consistently in 30-40 steps)
      → Now focus on Gate 2: move cube to target
      → Learn arm control while maintaining grasp (gripper frozen!)
      → Expected: -10,000 → -5,000 → +2,000 points
      → Reach target for +1000 success bonus!

Why GATED CURRICULUM works better:
    ✅ DECOUPLES two hard skills: grasp learning vs navigation learning
    ✅ SEQUENTIAL mastery: Learn grasp first, then navigate
    ✅ DENSE PENALTIES in Gate 1 provide clear gradient to grasp
    ✅ DENSE REWARDS in Gate 2 provide clear gradient to target
    ✅ NO local optimum trap: Must complete both gates for success
    ✅ NATURAL curriculum: Master fundamentals before advanced skills
    ✅ REALISTIC challenge: Must maintain grasp while navigating (no freeze)
    ✅ Progression: Minimize penalties (Gate 1) → Maximize rewards (Gate 2)

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


class Stage6Task(SO101BaseEnv):
    """Stage 6: Grasping with XYZ Guidance"""

    # REDUCED episode length for faster training
    MAX_EPISODE_STEPS = 100  # 3.3 seconds (was 150 / 5 seconds)

    # Early termination threshold
    FAILURE_DISTANCE = 0.25  # 25cm - terminate if gripper moves this far from cube

    # Stage 6 positioning criteria - based on jaw geometry
    MAX_PERPENDICULAR_OFFSET = 0.010  # 1cm - perpendicular to jaw centerline (precise)
    MAX_FORWARD_OFFSET = 0.010        # 1cm - forward/back alignment (precise)
    MAX_HEIGHT_OFFSET = 0.010         # 1cm - height alignment (precise)

    # Graduated positioning thresholds (for intermediate rewards)
    LOOSE_THRESHOLD = 0.030           # 3cm - loose positioning
    TIGHT_THRESHOLD = 0.020           # 2cm - tight positioning
    PRECISE_THRESHOLD = 0.010         # 1cm - precise positioning (ready to grasp)

    # Grasp criteria
    GRASP_THRESHOLD = 0.25            # Gripper closed if state < 0.25
    PROXIMITY_THRESHOLD = 0.040       # 40mm - proximity for grasp detection (tuned for jaw geometry)

    # Dense XYZ guidance rewards
    XYZ_GUIDANCE_SCALE = 10.0         # Scale for each axis reward
    XYZ_GUIDANCE_FALLOFF = 15.0       # STEEPER falloff (was 10.0) - stronger gradient in 1-3cm range

    # Graduated positioning rewards (provides intermediate milestones)
    LOOSE_POSITIONING_REWARD = 20.0   # All errors <3cm - learning to approach
    TIGHT_POSITIONING_REWARD = 50.0   # All errors <2cm - getting close
    PRECISE_POSITIONING_REWARD = 100.0 # All errors <1cm - ready to grasp!

    # Contact rewards (guide toward touching before closing) - INCREASED to prevent exploit
    SINGLE_JAW_CONTACT_REWARD = 50.0  # One jaw touching cube (was 30)
    BOTH_JAWS_CONTACT_REWARD = 200.0  # Both jaws touching cube (was 100) - 2x positioning!

    # Grasp reward
    GRASP_REWARD = 500.0              # MASSIVE reward for successful grasp!

    # Lifting rewards (final milestone - cube must be grasped)
    LIFT_THRESHOLD_LOW = 0.03         # 3cm above table (small lift)
    LIFT_THRESHOLD_MED = 0.06         # 6cm above table (medium lift)
    LIFT_THRESHOLD_HIGH = 0.10        # 10cm above table (high lift)
    LIFT_REWARD_LOW = 100.0           # Small lift bonus
    LIFT_REWARD_MED = 200.0           # Medium lift bonus
    LIFT_REWARD_HIGH = 300.0          # High lift bonus

    # Penalties (conditional - only when failing to position)
    CUBE_MOVEMENT_PENALTY_SCALE = -50.0
    CUBE_MOVEMENT_THRESHOLD = 0.005
    TABLE_CONTACT_PENALTY = -5.0
    PREMATURE_CLOSING_PENALTY = -10.0  # Penalty for closing without contact

    # Home position - arm extended straight forward, gripper open
    HOME_POSITION = {
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": 0.8,
        "wrist_roll": 1.5708,  # 90° (horizontal jaws)
        "gripper": 1.0,        # FULLY OPEN (ready to grasp)
    }

    def __init__(self, render_mode=None, freeze_object=False, curriculum_learning=True):
        super().__init__(render_mode=render_mode)

        # Target tracking
        self.hold_steps = 0          # Steps at precise position
        self.grasp_steps = 0         # Steps grasping
        self.lift_steps = 0          # Steps lifting cube
        self._episode_count = 0
        self._prev_gripper_state = 1.0  # Track for detecting closing
        self._prev_grasping = False     # Track for one-time grasp bonus
        self._initial_obj_height = 0.015  # Initial cube height (updated each reset)

        # GATED CURRICULUM: Two-stage learning (no freeze - must maintain grasp!)
        self._gate2_unlocked = False     # True once grasping achieved
        self._target_pos = None          # Random target position (green circle)

        # Legacy parameters
        self.curriculum_learning = curriculum_learning
        self.freeze_object = freeze_object
        self._frozen_object_pos = None

        # Action space: 6D (all joints active - gripper can close!)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # Build joint scaling arrays
        self._joint_low = np.array([self.JOINT_LIMITS[name][0] for name in
                                     ["shoulder_pan", "shoulder_lift", "elbow_flex",
                                      "wrist_flex", "wrist_roll", "gripper"]])
        self._joint_high = np.array([self.JOINT_LIMITS[name][1] for name in
                                      ["shoulder_pan", "shoulder_lift", "elbow_flex",
                                       "wrist_flex", "wrist_roll", "gripper"]])
        self._joint_mid = (self._joint_low + self._joint_high) / 2
        self._joint_range = (self._joint_high - self._joint_low) / 2

        self.gripper_min = self.JOINT_LIMITS["gripper"][0]
        self.gripper_max = self.JOINT_LIMITS["gripper"][1]

    def step(self, action):
        """Execute one step - gripper can now close!"""
        assert self.action_space.contains(action), f"Invalid action: {action}"

        action = np.clip(action, -1.0, 1.0)
        target_qpos = self._joint_mid + action * self._joint_range

        # Apply to all actuators (gripper is NOW ACTIVE!)
        self.data.ctrl[:6] = target_qpos

        # Freeze wrist_roll to keep jaws horizontal
        self.data.ctrl[4] = self.HOME_POSITION["wrist_roll"]

        # Check if Gate 2 should unlock (grasping achieved)
        if not self._gate2_unlocked and self._is_grasping():
            self._gate2_unlocked = True
            print(f"[Gate 1→2] Grasp achieved! Gate 2 unlocked (no freeze - agent must maintain grasp!)")

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Freeze object if enabled
        if self.freeze_object and self._frozen_object_pos is not None:
            self.data.qpos[self.object_qpos_start:self.object_qpos_start + 3] = self._frozen_object_pos
            self.data.qvel[self.object_qpos_start:self.object_qpos_start + 3] = 0
            mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        self._step_count += 1

        # Check states
        is_positioned = self._is_positioned()
        is_grasping = self._is_grasping()

        # Check if lifting
        cube_pos = self.data.site_xpos[self.object_site_id]
        cube_height = cube_pos[2]
        lift_height = cube_height - self._initial_obj_height
        is_lifting = is_grasping and lift_height >= self.LIFT_THRESHOLD_LOW

        # Track consecutive steps
        if is_positioned:
            self.hold_steps += 1
        else:
            self.hold_steps = 0

        if is_grasping:
            self.grasp_steps += 1
        else:
            self.grasp_steps = 0

        if is_lifting:
            self.lift_steps += 1
        else:
            self.lift_steps = 0

        # Calculate reward
        reward = self._compute_reward()

        # NO EARLY TERMINATION for gated curriculum
        # Episodes must run full 100 steps to allow:
        # - Gate 1: Time to learn grasping
        # - Gate 2: Time to learn navigation (requires moving cube!)
        terminated = False  # Never terminate early
        truncated = self._step_count >= self.MAX_EPISODE_STEPS

        # Get metrics
        xyz_errors = self._get_xyz_errors()

        # Info
        info = {
            "is_success": is_lifting,  # Success = grasping AND lifting
            "is_positioned": is_positioned,
            "is_grasping": is_grasping,
            "is_lifting": is_lifting,
            "lift_height": lift_height,
            "hold_steps": self.hold_steps,
            "grasp_steps": self.grasp_steps,
            "lift_steps": self.lift_steps,
            "x_error": xyz_errors[0],
            "y_error": xyz_errors[1],
            "z_error": xyz_errors[2],
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward(self):
        """STAGE 6: GATED CURRICULUM - Two-stage learning with gripper freeze."""
        reward = 0.0

        is_grasping = self._is_grasping()
        gripper_state = self._get_gripper_state()

        # ===================================================================
        # GATE 1: LEARN TO GRASP (before gripper frozen)
        # ===================================================================
        if not self._gate2_unlocked:
            # Dense penalties for NOT being in grasp position
            xyz_errors = self._get_xyz_errors()

            # Penalty scales inversely with quality (REDUCED 5× for better gradient)
            # Perfect alignment (0cm error) = 0 penalty
            # Poor alignment (30cm error) = -30 penalty (was -150!)
            x_penalty = -10 * xyz_errors[0]  # -10 per cm (was -50)
            y_penalty = -10 * xyz_errors[1]
            z_penalty = -10 * xyz_errors[2]
            reward += x_penalty + y_penalty + z_penalty

            # Dense penalty for NOT grasping (REDUCED 5× for better gradient)
            if not is_grasping:
                # Penalty based on how far from grasping
                left_contact, right_contact = self._get_jaw_contacts()

                if not left_contact and not right_contact:
                    reward += -20  # No contact at all (was -100)
                elif not (left_contact and right_contact):
                    reward += -10  # Partial contact (was -50)
                else:
                    # Both touching but not grasping (gripper not closed enough)
                    reward += -5   # Small penalty - almost there! (was -20)

        # ===================================================================
        # GATE 2: LEARN TO NAVIGATE (after gripper frozen)
        # ===================================================================
        else:
            # Dense reward for moving cube closer to target (green circle)
            cube_pos = self.data.site_xpos[self.object_site_id]
            target_pos = self._target_pos

            # Distance from cube to target
            distance_to_target = np.linalg.norm(cube_pos - target_pos)

            # Dense reward: closer = better (REDUCED 5× for better gradient)
            # Scale: 20 pts per cm to target (was 100)
            # At target (0cm away) = 0 penalty
            # 50cm away = -100 penalty (was -500!)
            # Getting closer increases reward
            reward += -20 * distance_to_target  # Minimize distance

            # Track previous distance for delta reward
            if self._prev_distance_to_target is None:
                # First step in Gate 2 - initialize tracking
                self._prev_distance_to_target = distance_to_target
                distance_delta = 0  # No delta on first step
            else:
                # Calculate movement toward/away from target
                distance_delta = self._prev_distance_to_target - distance_to_target

            # Bonus for reducing distance (moving in right direction)
            if distance_delta > 0:
                reward += 50 * distance_delta  # Bonus for getting closer! (was 200)

            # Update for next step
            self._prev_distance_to_target = distance_to_target

            # Success bonus for reaching target (within 2cm)
            if distance_to_target < 0.02:  # 2cm threshold
                reward += 1000  # Large success bonus!

        # ===================================================================
        # SHARED: Update tracking variables
        # ===================================================================
        self._prev_gripper_state = gripper_state
        self._prev_grasping = is_grasping

        return reward

    def _get_xyz_errors(self):
        """Get alignment errors in X, Y, Z directions."""
        left_tip = self.data.site_xpos[self.left_fingertip_id]
        right_tip = self.data.site_xpos[self.right_fingertip_id]
        cube_pos = self.data.site_xpos[self.object_site_id]

        jaw_center = (left_tip + right_tip) / 2
        jaw_vector = right_tip - left_tip
        jaw_opening_dist = np.linalg.norm(jaw_vector)

        if jaw_opening_dist < 0.001:
            return np.array([999.0, 999.0, 999.0])  # Invalid state

        jaw_direction = jaw_vector / jaw_opening_dist

        # Perpendicular distance (X-axis error in jaw frame)
        cube_to_center = cube_pos - jaw_center
        perpendicular_component = cube_to_center - np.dot(cube_to_center, jaw_direction) * jaw_direction
        x_error = np.linalg.norm(perpendicular_component)

        # Forward/back alignment (Y-axis error)
        y_error = abs(cube_pos[1] - jaw_center[1])

        # Height alignment (Z-axis error)
        z_error = abs(cube_pos[2] - jaw_center[2])

        return np.array([x_error, y_error, z_error])

    def _is_positioned(self):
        """Check if cube is positioned between the jaws."""
        left_tip = self.data.site_xpos[self.left_fingertip_id]
        right_tip = self.data.site_xpos[self.right_fingertip_id]
        cube_pos = self.data.site_xpos[self.object_site_id]

        jaw_center = (left_tip + right_tip) / 2
        jaw_vector = right_tip - left_tip
        jaw_opening_dist = np.linalg.norm(jaw_vector)

        if jaw_opening_dist < 0.001:
            return False

        jaw_direction = jaw_vector / jaw_opening_dist

        # Check if cube is between jaws
        cube_to_left = cube_pos - left_tip
        projection = np.dot(cube_to_left, jaw_direction)
        if projection < -0.01 or projection > jaw_opening_dist + 0.01:
            return False

        # Check perpendicular distance
        cube_to_center = cube_pos - jaw_center
        perpendicular_component = cube_to_center - np.dot(cube_to_center, jaw_direction) * jaw_direction
        perp_dist = np.linalg.norm(perpendicular_component)
        if perp_dist > self.MAX_PERPENDICULAR_OFFSET:
            return False

        # Check height alignment
        if abs(cube_pos[2] - jaw_center[2]) > self.MAX_HEIGHT_OFFSET:
            return False

        # Check forward/back alignment
        if abs(cube_pos[1] - jaw_center[1]) > self.MAX_FORWARD_OFFSET:
            return False

        return True

    def _is_grasping(self):
        """
        Proximity-based grasp detection (industry standard).

        Checks if gripper is closed and both fingertips are near the cube.
        This avoids physics penetration issues while being functionally equivalent.
        """
        # Must be closed enough
        gripper_state = self._get_gripper_state()
        if gripper_state >= self.GRASP_THRESHOLD:
            return False

        # Get fingertip and cube positions
        left_pos = self.data.site_xpos[self.left_fingertip_id]
        right_pos = self.data.site_xpos[self.right_fingertip_id]
        cube_pos = self.data.site_xpos[self.object_site_id]

        # Check if both fingertips are close to cube
        left_dist = np.linalg.norm(left_pos - cube_pos)
        right_dist = np.linalg.norm(right_pos - cube_pos)

        return left_dist < self.PROXIMITY_THRESHOLD and right_dist < self.PROXIMITY_THRESHOLD

    def _get_jaw_contacts(self):
        """Check which jaws are touching the cube. Returns (left_contact, right_contact)."""
        cube_geom_name = "object_geom"
        left_contact = False
        right_contact = False

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)

            # Check if cube is involved
            cube_in_contact = False
            other_geom_id = None

            if geom1_name and cube_geom_name in geom1_name:
                cube_in_contact = True
                other_geom_id = geom2_id
            elif geom2_name and cube_geom_name in geom2_name:
                cube_in_contact = True
                other_geom_id = geom1_id

            if cube_in_contact and other_geom_id is not None:
                # Get body name of the other geometry
                body_id = self.model.geom_bodyid[other_geom_id]
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)

                if body_name:
                    # gripper_body is the fixed jaw (left)
                    # moving_jaw_so101_v1 is the moving jaw (right)
                    if "gripper_body" in body_name.lower():
                        left_contact = True
                    if "moving_jaw" in body_name.lower():
                        right_contact = True

        return left_contact, right_contact

    def _has_both_jaws_contact(self):
        """Check if both jaws are touching the cube."""
        # Use the fixed _get_jaw_contacts() which properly detects unnamed geometries
        left_contact, right_contact = self._get_jaw_contacts()
        return left_contact and right_contact

    def _is_touching_table(self):
        """Check if gripper is touching the table."""
        try:
            table_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        except:
            return False

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            if geom1_id == table_geom_id or geom2_id == table_geom_id:
                other_geom = geom2_id if geom1_id == table_geom_id else geom1_id
                body_id = self.model.geom_bodyid[other_geom]

                if body_id in [self.gripper_body_id, self.jaw_body_id]:
                    return True

        return False

    def reset(self, seed=None, options=None):
        """Reset to initial state."""
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)

        mujoco.mj_resetData(self.model, self.data)
        self._episode_count += 1

        # Randomize cube position (spawn closer - slightly narrower to reduce max distance)
        cube_world_x = self.np_random.uniform(-0.13, 0.13)  # ±13cm left/right (26cm width, was 30cm)
        cube_world_y = self.np_random.uniform(0.26, 0.34)  # 8cm depth (maintained)

        self.data.qpos[self.object_qpos_start] = cube_world_x - 0.0
        self.data.qpos[self.object_qpos_start + 1] = cube_world_y - 0.4
        self.data.qpos[self.object_qpos_start + 2] = 0

        # Randomize goal marker (green circle) - Gate 2 target
        # WIDER spawn range than cube to force real navigation!
        goal_world_x = self.np_random.uniform(-0.25, 0.25)  # ±25cm left/right (50cm width - 2× cube range)
        goal_world_y = self.np_random.uniform(0.20, 0.40)  # 20cm depth (2.5× cube range)
        goal_height = 0.10  # 10cm above table - target lift height

        self.data.qpos[self.goal_qpos_start] = goal_world_x - 0.2
        self.data.qpos[self.goal_qpos_start + 1] = goal_world_y - 0.35
        self.data.qpos[self.goal_qpos_start + 2] = goal_height - 0.05

        # Store target position for reward calculation
        self._target_pos = np.array([goal_world_x - 0.2, goal_world_y - 0.35, goal_height - 0.05])

        # Set arm to home position
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        for idx, joint_name in enumerate(joint_names):
            self.data.qpos[idx] = self.HOME_POSITION[joint_name]

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        self._initial_obj_height = self.data.site_xpos[self.object_site_id][2]

        if self.freeze_object:
            self._frozen_object_pos = self.data.qpos[self.object_qpos_start:self.object_qpos_start + 3].copy()

        # Reset counters
        self.hold_steps = 0
        self.grasp_steps = 0
        self.lift_steps = 0
        self._step_count = 0
        self._prev_gripper_state = 1.0  # Start fully open
        self._prev_grasping = False     # Not grasping at start

        # Reset gating variables
        self._gate2_unlocked = False
        self._prev_distance_to_target = None

        info = {}
        return obs, info


# Test the environment
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    env = Stage6Task()
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial gripper state: {env._get_gripper_state():.3f} (can now close!)")

    # Test a few steps
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {i+1}: reward={reward:.2f}, positioned={info['is_positioned']}, grasping={info['is_grasping']}")
        if terminated or truncated:
            break

    print(f"\nTotal reward: {total_reward:.2f}")
    env.close()
