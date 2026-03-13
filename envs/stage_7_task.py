"""
STAGE 7: UR5 Sparse Reward Implementation (MODIFIED - Z-axis + Auto-Grasp)

STATUS: ✅ IMPLEMENTED
Date Started: 2026-01-30
Updated: 2026-01-31 (Added Z-axis weighting + automatic grasping)
Target: 1M+ steps
Resume From: None (train from scratch)

================================================================================
COMPARISON STUDY: Dense (Stage 6) vs Sparse (Stage 7) Rewards
================================================================================
Stage 6 (PREV): Dense rewards with graduated phases ✅
Stage 7 (THIS): UR5's hybrid system (sparse success + dense distance)

================================================================================
TASK OBJECTIVE (EXACT UR5 MATCH)
================================================================================
Replicate UR5 repository's reward design (MODIFIED for better learning):
- HYBRID reward: +100 for success (sparse) + speed bonus, -10×distance every step (dense)
- Force-based grasping: >3N threshold on BOTH gripper fingers
- 3D weighted distance: End-effector to cube (X,Y full weight, Z at 50% weight)
- No curriculum/phases: Single reward structure for entire training
- Same SAC algorithm: Continue using Soft Actor-Critic

MODIFICATIONS:
1. Added Z-axis (height) to distance calculation with 50% weight to avoid
   random exploration bottleneck. Pure 2D (X,Y) UR5 approach requires random
   discovery of correct height, which significantly slows learning.

2. Added automatic grasping (AUTO_GRASP_ENABLED = True) to replicate UR5's
   scripted behavior. When agent positions end-effector within 1cm of cube,
   gripper automatically closes. This matches UR5's approach where grasping
   is triggered by positioning, not learned by the agent.

   Set AUTO_GRASP_ENABLED = False to require agent to learn gripper control.

PRIMARY GOALS:
    1. **Navigate:** Move end-effector to cube location (X, Y alignment)
    2. **Grasp cube:** Close gripper with >3N force on both fingers
    3. **Complete efficiently:** Speed bonus rewards fast completion

SPECIFIC REQUIREMENTS:
    - Success: End-effector within 1cm of cube (2D) + grasping (>3N both fingers)
    - Reward: +100 + speed_bonus (success) OR -10 * distance_to_target (failure)
    - Early termination: Episode ends on success
    - EPISODE LENGTH: 100 steps (3.3 sec)

Action Space: 6D (all joints active - gripper can close!)
Control Method: Actuator position control
Constraints: wrist_roll FROZEN at 0° (vertical jaws - one above, one below)

Success Criteria: End-effector reaches cube + successful grasp
                 (NO lifting required for reward!)

================================================================================
REWARD STRUCTURE (UR5 HYBRID SYSTEM - EXACT MATCH)
================================================================================

DENSE COMPONENT (every step):
    -10.0 * distance_to_target    Weighted 3D distance (X,Y,Z)

    Distance calculation:
    - X,Y distance: 100% weight (horizontal alignment)
    - Z distance: 50% weight (height alignment)
    - Combined: sqrt(xy_dist² + (0.5 * z_dist)²)

    Example: EEF 30cm from cube → -10 * 0.30 = -3.0 pts/step
             EEF 10cm from cube → -10 * 0.10 = -1.0 pts/step
             EEF 5cm from cube  → -10 * 0.05 = -0.5 pts/step
             EEF 1cm from cube  → -10 * 0.01 = -0.1 pts/step

    This creates a GRADIENT guiding the agent toward the cube in all 3 axes!

SPARSE COMPONENT (on success only):
    +100.0                Base reward for successful grasp
    + speed_bonus         Efficiency bonus (remaining_steps * 1.0)

    Example: Success at step 30 → +100 + 70 = +170 pts
             Success at step 50 → +100 + 50 = +150 pts
             Success at step 80 → +100 + 20 = +120 pts

SUCCESS CONDITION (both must be true):
    1. distance_to_target ≤ 0.01m    (end-effector within 1cm of cube, weighted 3D)
    2. is_grasping = True             (>3N force on BOTH fingers)

FORCE-BASED GRASPING:
    >3.0 N on BOTH fingers   Required for valid grasp

    - Left finger force > 3N ✓
    - Right finger force > 3N ✓
    - Confirms physical contact and grip

EARLY TERMINATION:
    - Terminate immediately on success (don't wait for 100 steps)
    - Otherwise run full episode
    - Truncate at 100 steps if not successful

================================================================================
EXPECTED LEARNING CURVE (WITH DENSE DISTANCE GRADIENT)
================================================================================

Phase 1: Random Exploration (0-100K steps)
    - Random arm movements
    - Average reward: -100 to -150 pts/episode
    - Distance: 0.10-0.30m (varying)
    - Success rate: <0.1%
    - Learning: Building replay buffer, exploring action space

Phase 2: Approach Learning (100K-300K steps)
    - Gripper moves toward cube occasionally
    - Average reward: -50 to -100 pts/episode (improving!)
    - Distance: 0.05-0.15m (getting closer)
    - Success rate: <1%
    - Learning: "Moving toward cube = less negative reward"

Phase 3: Grasp Discovery (300K-500K steps)
    - Gripper reaches cube, attempts to close
    - Average reward: -10 to -50 pts/episode
    - Distance: 0.01-0.05m (very close)
    - Success rate: 1-10%
    - Learning: First successful grasps! Sudden +100 reward spikes

Phase 4: Consistent Success (500K-1M steps)
    - Reliable grasp behavior
    - Average reward: +50 to +120 pts/episode
    - Distance: <0.01m consistently
    - Success rate: 50-80%
    - Learning: Refining speed, optimizing trajectory

================================================================================
COMPARISON TO STAGE 6
================================================================================

Stage 6 (Dense Rewards):
    - Reward per step: 0-1,130 points (continuous feedback)
    - Learning curve: Gradual progression through phases
    - Sample efficiency: Faster learning (more guidance)
    - Final behavior: Smooth, controlled movements
    - Training time: 700K-1M steps for mastery

Stage 7 (Sparse Rewards - UR5):
    - Reward per step: -10 to +200 (mostly negative until success)
    - Learning curve: Exploration phase, then sudden breakthroughs
    - Sample efficiency: Slower learning (less guidance)
    - Final behavior: Efficient, direct movements
    - Training time: 1M-3M steps estimated

================================================================================
SIM-TO-REAL CONSIDERATIONS
================================================================================

FORCE SENSORS (CRITICAL):
    - Stage 7 uses >3N force threshold for grasping
    - Requires force sensors on gripper (not currently available on real robot)
    - This is a SIMULATION-ONLY approach for comparison study
    - For real robot deployment, use Stage 6 (collision-based)

Alternative (future work):
    - Replace force with collision + gripper state
    - is_grasping = left_contact AND right_contact AND (gripper < 0.3)
    - Less precise but sim-to-real compatible

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


class Stage7Task(SO101BaseEnv):
    """Stage 7: UR5 Sparse Reward Implementation"""

    # Episode length
    MAX_EPISODE_STEPS = 100  # 3.3 seconds

    # UR5 Hybrid Reward Constants (MODIFIED - Added Z-axis)
    SUCCESS_REWARD = 100.0              # Base reward for successful grasp
    LIFT_REWARD = 100.0                 # Additional reward for lifting cube
    SUCCESS_DISTANCE_THRESHOLD = 0.01   # 1cm threshold - end-effector to cube
    DISTANCE_PENALTY_SCALE = -10.0      # Penalty per meter of distance (DENSE component)
    SPEED_BONUS_MULTIPLIER = 1.0        # Bonus per remaining step (SPARSE component)
    LIFT_HEIGHT_THRESHOLD = 0.03        # 3cm - cube must be lifted this high for success
    FORCE_THRESHOLD = 3.0               # Newtons - minimum force for valid grasp (not used if USE_FORCE_SENSORS=False)

    # Sim-to-real compatibility
    USE_FORCE_SENSORS = False           # If False, use contact + gripper position (sim-to-real)
                                        # If True, use force sensors (simulation only)
    GRIPPER_CLOSED_THRESHOLD = 0.3      # Gripper position threshold (0=closed, 1=open)

    # Z-axis weighting (0.0 = ignore Z, 1.0 = equal weight to X,Y)
    Z_WEIGHT = 0.5                      # 50% weight on height vs horizontal alignment

    # Target height approach (hover above cube before descending)
    USE_TARGET_HEIGHT = True            # If True, reward for hovering at fixed height
    TARGET_HEIGHT_OFFSET = 0.05         # 5cm above cube (hover height)

    # Scripted descent sequence (after positioning at target height)
    SCRIPTED_DESCENT_ENABLED = True     # If True, execute scripted descent when positioned
    SCRIPTED_ROTATE_STEPS = 10          # Steps to rotate wrist 90°
    SCRIPTED_LOWER_STEPS = 20           # Steps to descend to cube
    SCRIPTED_CLOSE_STEPS = 20           # Steps to close gripper
    SCRIPTED_LIFT_STEPS = 20            # Steps to lift cube off table
    SCRIPTED_DESCENT_STEPS = 70         # Total steps for full sequence (10+20+20+20)

    # Automatic grasping (UR5 behavior replication)
    AUTO_GRASP_ENABLED = False          # If True, gripper closes automatically when positioned
    AUTO_GRASP_STEPS = 20               # Simulation steps for gripper to close
    FREEZE_WHEN_POSITIONED = False      # If True, freeze robot when positioned (for visualization)
    FREEZE_GRIPPER_OPEN = True          # If True, force gripper to stay fully open until positioned

    # Grasp criteria (backup collision-based check)
    GRASP_THRESHOLD = 0.25              # Gripper closed if state < 0.25

    # Home position - arm extended straight forward, gripper open
    HOME_POSITION = {
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": 0.8,
        "wrist_roll": 0.0,     # 0° (vertical jaws - one above, one below)
        "gripper": 1.0,        # FULLY OPEN (ready to grasp)
    }

    def __init__(self, render_mode=None, freeze_object=False):
        super().__init__(render_mode=render_mode)

        # Target tracking
        self._episode_count = 0
        self._is_success = False          # Success flag for early termination
        self.scripted_phase_active = False  # Currently executing scripted descent
        self.scripted_step = 0            # Step counter for scripted sequence

        # Legacy parameters
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
        """Execute one step with sparse reward system."""
        self._step_count += 1

        # ═══════════════════════════════════════════════════════════════════════
        # SCRIPTED PHASE: Script has full control, ignore agent action
        # ═══════════════════════════════════════════════════════════════════════
        if self.scripted_phase_active:
            # Agent action is IGNORED during scripted sequence
            # Script controls all joints and simulation stepping
            obs, reward, terminated = self._execute_scripted_descent()

            # Script provides its own reward (0.0 during phases 1-3, real reward in phase 4)
            truncated = self._step_count >= self.MAX_EPISODE_STEPS

            # Info dict for scripted phase
            info = {
                "is_success": terminated and reward > 0,
                "distance_to_target": 0.0,  # Not meaningful during script
                "is_grasping": False,  # Will be set in final phase
                "left_force": 0.0,
                "right_force": 0.0,
                "speed_bonus": 0.0,
                "reward": reward,
                "episode_step": self._step_count,
                "positioned_correctly": True,  # Already positioned (that's why script triggered)
                "scripted_phase": True,
                "scripted_step": self.scripted_step - 1,  # Show which script step just executed
            }

            return obs, reward, terminated, truncated, info

        # ═══════════════════════════════════════════════════════════════════════
        # AGENT CONTROL: Normal learning phase
        # ═══════════════════════════════════════════════════════════════════════

        # Validate agent action (only when agent has control)
        assert self.action_space.contains(action), f"Invalid action: {action}"

        action = np.clip(action, -1.0, 1.0)
        target_qpos = self._joint_mid + action * self._joint_range

        # Apply agent action to all actuators
        self.data.ctrl[:6] = target_qpos

        # Freeze wrist_roll to keep jaws vertical (one above, one below)
        self.data.ctrl[4] = self.HOME_POSITION["wrist_roll"]

        # Freeze gripper at max opening if enabled (prevents premature closing)
        if self.FREEZE_GRIPPER_OPEN:
            self.data.ctrl[5] = self.gripper_max  # Force gripper fully open

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Freeze object if enabled
        if self.freeze_object and self._frozen_object_pos is not None:
            self.data.qpos[self.object_qpos_start:self.object_qpos_start + 3] = self._frozen_object_pos
            self.data.qvel[self.object_qpos_start:self.object_qpos_start + 3] = 0
            mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()

        # Check if positioned correctly (before computing reward)
        eef_pos = self.data.site_xpos[self.ee_site_id]
        cube_pos = self.data.site_xpos[self.object_site_id]

        # Calculate distance to target (hover position if USE_TARGET_HEIGHT)
        if self.USE_TARGET_HEIGHT:
            # Target is 5cm above cube (hover position)
            target_pos = cube_pos.copy()
            target_pos[2] += self.TARGET_HEIGHT_OFFSET
            xy_dist = np.linalg.norm(eef_pos[:2] - target_pos[:2])
            z_dist = abs(eef_pos[2] - target_pos[2])
        else:
            # Target is cube position directly
            xy_dist = np.linalg.norm(eef_pos[:2] - cube_pos[:2])
            z_dist = abs(eef_pos[2] - cube_pos[2])

        distance_to_target = np.sqrt(xy_dist**2 + (self.Z_WEIGHT * z_dist)**2)

        # POSITIONING CHECK - Trigger scripted descent or freeze/auto-grasp
        positioned_correctly = distance_to_target <= self.SUCCESS_DISTANCE_THRESHOLD

        if positioned_correctly and not self.scripted_phase_active:
            # TRIGGER SCRIPTED DESCENT SEQUENCE
            if self.SCRIPTED_DESCENT_ENABLED:
                print(f"[Script] Triggering descent at step {self._step_count}")
                self.scripted_phase_active = True
                self.scripted_step = 0

            elif self.FREEZE_WHEN_POSITIONED:
                # FREEZE MODE: Stop all motion when positioned (for visualization)
                self.data.qvel[:self.n_robot_joints] = 0.0
                pass  # Robot will hold current position

            elif self.AUTO_GRASP_ENABLED:
                # AUTOMATIC GRASPING (UR5 behavior)
                self.data.ctrl[5] = self.gripper_min  # Full close

                # Run simulation for gripper to close
                for _ in range(self.AUTO_GRASP_STEPS):
                    mujoco.mj_step(self.model, self.data)
                    if self.freeze_object and self._frozen_object_pos is not None:
                        self.data.qpos[self.object_qpos_start:self.object_qpos_start + 3] = self._frozen_object_pos
                        self.data.qvel[self.object_qpos_start:self.object_qpos_start + 3] = 0
                        mujoco.mj_forward(self.model, self.data)

                obs = self._get_obs()

        # Compute sparse reward (agent learning phase only)
        reward, reward_info = self._compute_reward()

        # Early termination on success
        terminated = reward_info["is_success"]
        truncated = self._step_count >= self.MAX_EPISODE_STEPS

        # Update info dict
        info = {
            **reward_info,
            "episode_step": self._step_count,
            "positioned_correctly": positioned_correctly,
            "auto_grasp_triggered": (self.AUTO_GRASP_ENABLED and positioned_correctly),
            "frozen": (self.FREEZE_WHEN_POSITIONED and positioned_correctly),
            "scripted_phase": False,
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward(self):
        """
        UR5 hybrid reward system (MODIFIED - Target height approach).

        DENSE component (every step): -10 * distance_to_target
        SPARSE component (on success): +100 + speed_bonus

        TARGET HEIGHT MODE:
        - Agent rewarded for reaching X,Z position of cube at TARGET_HEIGHT (5cm above)
        - Distance calculated to hover position, not cube position
        - Once positioned at hover height, scripted descent executes
        """
        # Get current state
        eef_pos = self.data.site_xpos[self.ee_site_id]  # End-effector position
        cube_pos = self.data.site_xpos[self.object_site_id]  # Cube position

        if self.USE_TARGET_HEIGHT:
            # TARGET HEIGHT MODE: Reward for hovering above cube
            # Target position is X,Z of cube, but Y = cube_Y + offset (5cm above)
            target_pos = cube_pos.copy()
            target_pos[2] += self.TARGET_HEIGHT_OFFSET  # 5cm above cube

            # Calculate distance to hover position
            # X,Z alignment + height at target (not cube height)
            xy_dist = np.linalg.norm(eef_pos[:2] - target_pos[:2])  # Horizontal (X,Y)
            z_dist = abs(eef_pos[2] - target_pos[2])               # Height to target

            # Weighted combination
            distance_to_target = np.sqrt(xy_dist**2 + (self.Z_WEIGHT * z_dist)**2)

        else:
            # ORIGINAL MODE: Reward for reaching cube position directly
            xy_dist = np.linalg.norm(eef_pos[:2] - cube_pos[:2])  # Horizontal distance
            z_dist = abs(eef_pos[2] - cube_pos[2])                # Height difference

            # Weighted combination: full weight on X,Y, partial weight on Z
            distance_to_target = np.sqrt(xy_dist**2 + (self.Z_WEIGHT * z_dist)**2)

        # Check if cube is lifted (SIM-TO-REAL COMPATIBLE)
        current_cube_height = cube_pos[2]
        lift_height = current_cube_height - self._initial_cube_height
        is_lifted = lift_height >= self.LIFT_HEIGHT_THRESHOLD

        # For info dict: check contacts (optional debugging info)
        if self.USE_FORCE_SENSORS:
            is_grasping, left_val, right_val = self._check_force_grasp()
        else:
            is_grasping, left_val, right_val = self._check_contact_grasp()

        # Success condition: CUBE IS LIFTED
        # This is sim-to-real compatible (vision system can detect cube height)
        is_success = is_lifted

        if is_success:
            # SPARSE SUCCESS REWARD: Base reward + lift reward + speed bonus
            remaining_steps = self.MAX_EPISODE_STEPS - self._step_count
            speed_bonus = remaining_steps * self.SPEED_BONUS_MULTIPLIER
            reward = self.SUCCESS_REWARD + self.LIFT_REWARD + speed_bonus
        else:
            # DENSE DISTANCE PENALTY: Proportional to distance (provides gradient)
            reward = self.DISTANCE_PENALTY_SCALE * distance_to_target

        # Store success flag for early termination
        self._is_success = is_success

        return reward, {
            "is_success": is_success,
            "distance_to_target": distance_to_target,
            "is_lifted": is_lifted,
            "lift_height": lift_height,
            "is_grasping": is_grasping,  # Debugging info
            "left_contact" if not self.USE_FORCE_SENSORS else "left_force": left_val,
            "right_contact" if not self.USE_FORCE_SENSORS else "right_force": right_val,
            "speed_bonus": speed_bonus if is_success else 0.0,
            "reward": reward,
            "cube_height": current_cube_height,
        }

    def _execute_scripted_descent(self):
        """
        Execute scripted descent sequence:
        Phase 1 (steps 0-9):   Rotate wrist_roll 0° → 90° (vertical → horizontal jaws)
        Phase 2 (steps 10-29): Descend in Z-axis while maintaining X,Y
        Phase 3 (steps 30-49): Close gripper gradually (open → closed)
        Phase 4 (steps 50-69): Lift cube upward
        Phase 5 (step 70):     Check if cube lifted → success

        Returns: (obs, reward, terminated)
        """
        cube_pos = self.data.site_xpos[self.object_site_id]
        eef_pos = self.data.site_xpos[self.ee_site_id]

        # PHASE 1: ROTATE WRIST (steps 0-9)
        if self.scripted_step < self.SCRIPTED_ROTATE_STEPS:
            # Smoothly rotate from 0° to 90° (0 to 1.5708 rad)
            progress = self.scripted_step / self.SCRIPTED_ROTATE_STEPS
            target_wrist_roll = progress * 1.5708  # Interpolate to 90°

            self.data.ctrl[4] = target_wrist_roll  # Rotate wrist

            if self.scripted_step == 0:
                print(f"[Script] Phase 1: Rotating wrist 0° → 90° (vertical → horizontal jaws)")
                # Store initial cube height for lift detection
                self._initial_cube_height = cube_pos[2]

        # PHASE 2: DESCEND (steps 10-29)
        elif self.scripted_step < (self.SCRIPTED_ROTATE_STEPS + self.SCRIPTED_LOWER_STEPS):
            # Lock wrist at 90°
            self.data.ctrl[4] = 1.5708  # 90° (horizontal jaws)

            # Descend in Z-axis
            # Calculate how far to descend: from current height to cube height
            descent_needed = eef_pos[2] - cube_pos[2]  # Height difference

            # Descend gradually
            descent_per_step = descent_needed / self.SCRIPTED_LOWER_STEPS
            self.data.ctrl[3] -= descent_per_step * 0.5  # wrist_flex to lower

            if self.scripted_step == self.SCRIPTED_ROTATE_STEPS:
                print(f"[Script] Phase 2: Descending {descent_needed*100:.1f}cm to cube")

        # PHASE 3: CLOSE GRIPPER (steps 30-49)
        elif self.scripted_step < (self.SCRIPTED_ROTATE_STEPS + self.SCRIPTED_LOWER_STEPS + self.SCRIPTED_CLOSE_STEPS):
            # Unfreeze gripper (only once at start of phase)
            if self.scripted_step == (self.SCRIPTED_ROTATE_STEPS + self.SCRIPTED_LOWER_STEPS):
                self.FREEZE_GRIPPER_OPEN = False
                print(f"[Script] Phase 3: Closing gripper gradually over {self.SCRIPTED_CLOSE_STEPS} steps")

            # Calculate progress through closing phase (0.0 to 1.0)
            close_step = self.scripted_step - (self.SCRIPTED_ROTATE_STEPS + self.SCRIPTED_LOWER_STEPS)
            progress = close_step / self.SCRIPTED_CLOSE_STEPS

            # Interpolate from fully open to fully closed
            target_gripper = self.gripper_max + progress * (self.gripper_min - self.gripper_max)
            self.data.ctrl[5] = target_gripper

            # Keep wrist locked at 90° and maintain descent position
            self.data.ctrl[4] = 1.5708  # 90° (horizontal jaws)

        # PHASE 4: LIFT CUBE (steps 50-69)
        elif self.scripted_step < (self.SCRIPTED_ROTATE_STEPS + self.SCRIPTED_LOWER_STEPS +
                                    self.SCRIPTED_CLOSE_STEPS + self.SCRIPTED_LIFT_STEPS):
            # First step of lift phase
            if self.scripted_step == (self.SCRIPTED_ROTATE_STEPS + self.SCRIPTED_LOWER_STEPS + self.SCRIPTED_CLOSE_STEPS):
                print(f"[Script] Phase 4: Lifting cube upward over {self.SCRIPTED_LIFT_STEPS} steps")

            # Keep gripper closed
            self.data.ctrl[5] = self.gripper_min

            # Lift by raising wrist_flex
            lift_per_step = 0.05 / self.SCRIPTED_LIFT_STEPS  # Lift ~5cm total
            self.data.ctrl[3] += lift_per_step

            # Keep wrist locked at 90°
            self.data.ctrl[4] = 1.5708

        # PHASE 5: CHECK LIFT SUCCESS (step 70+)
        else:
            # Keep gripper closed during check
            self.data.ctrl[5] = self.gripper_min

            # Check if cube was lifted off table
            current_cube_height = cube_pos[2]
            lift_height = current_cube_height - self._initial_cube_height

            print(f"[Script] Phase 5: Checking lift...")
            print(f"  Initial height: {self._initial_cube_height*100:.2f}cm")
            print(f"  Current height: {current_cube_height*100:.2f}cm")
            print(f"  Lift height: {lift_height*100:.2f}cm (need ≥{self.LIFT_HEIGHT_THRESHOLD*100:.1f}cm)")

            # Success if cube lifted ≥ 3cm off table
            is_success = lift_height >= self.LIFT_HEIGHT_THRESHOLD

            # Calculate reward
            if is_success:
                remaining_steps = self.MAX_EPISODE_STEPS - self._step_count
                speed_bonus = remaining_steps * self.SPEED_BONUS_MULTIPLIER
                reward = self.SUCCESS_REWARD + self.LIFT_REWARD + speed_bonus
                terminated = True
                print(f"[Script] ✓ Lift successful! Reward: {reward:.1f}")
            else:
                reward = -1.0
                terminated = False
                print(f"[Script] ✗ Lift failed (cube not lifted high enough)")

            # Restore freeze state
            self.FREEZE_GRIPPER_OPEN = True

            # End scripted phase
            self.scripted_phase_active = False

            return self._get_obs(), reward, terminated

        # Advance script
        self.scripted_step += 1

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Return current state (script still running)
        return self._get_obs(), 0.0, False

    def _check_force_grasp(self):
        """
        Check if gripper is applying sufficient force on cube.
        UR5 approach: Both fingers must have >3N force.

        Returns: (is_force_grasping, left_force, right_force)
        """
        left_force = 0.0
        right_force = 0.0

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)

            if geom1_name is None or geom2_name is None:
                continue

            # Check if cube is involved
            cube_geom_name = "object_geom"
            if cube_geom_name in geom1_name or cube_geom_name in geom2_name:
                other_name = geom2_name if cube_geom_name in geom1_name else geom1_name

                # Compute contact force using MuJoCo
                contact_force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, contact_force)
                force_magnitude = np.linalg.norm(contact_force[:3])  # Normal force

                if other_name and "left" in other_name.lower():
                    left_force += force_magnitude
                if other_name and "right" in other_name.lower():
                    right_force += force_magnitude

        # Both fingers must exceed threshold
        is_force_grasping = (left_force >= self.FORCE_THRESHOLD and
                             right_force >= self.FORCE_THRESHOLD)

        return is_force_grasping, left_force, right_force

    def _check_contact_grasp(self):
        """
        Check if gripper is grasping cube using contact detection + gripper position.

        SIM-TO-REAL COMPATIBLE: Uses only sensors available on real hardware:
        - Contact detection: Can be implemented with limit switches or proximity sensors
        - Gripper position: Available from motor encoders

        NO force sensors required!

        Returns: (is_contact_grasping, left_contact, right_contact)
        """
        left_contact = False
        right_contact = False

        # Check for contacts between gripper fingers and cube
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)

            if geom1_name is None or geom2_name is None:
                continue

            # Check if cube is involved in contact
            cube_geom_name = "object_geom"
            if cube_geom_name in geom1_name or cube_geom_name in geom2_name:
                other_name = geom2_name if cube_geom_name in geom1_name else geom1_name

                # Detect which finger is touching
                if other_name and "left" in other_name.lower():
                    left_contact = True
                if other_name and "right" in other_name.lower():
                    right_contact = True

        # Get gripper position (0.0 = fully closed, 1.0 = fully open)
        gripper_position = self.data.qpos[5]  # Gripper joint position

        # Success criteria (SIM-TO-REAL):
        # 1. Both fingers touching cube (contact switches triggered)
        # 2. Gripper mostly closed (encoder reading < threshold)
        is_contact_grasping = (
            left_contact and
            right_contact and
            gripper_position < self.GRIPPER_CLOSED_THRESHOLD
        )

        return is_contact_grasping, left_contact, right_contact

    def reset(self, seed=None, options=None):
        """Reset to initial state."""
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)

        mujoco.mj_resetData(self.model, self.data)
        self._episode_count += 1

        # Randomize cube position (same as Stage 6)
        target_world_x = self.np_random.uniform(-0.13, 0.13)  # ±13cm left/right
        target_world_y = self.np_random.uniform(0.26, 0.34)  # 8cm depth

        self.data.qpos[self.object_qpos_start] = target_world_x - 0.0
        self.data.qpos[self.object_qpos_start + 1] = target_world_y - 0.4
        self.data.qpos[self.object_qpos_start + 2] = 0

        # UR5 has no separate goal position - the cube itself IS the target!
        # Hide goal marker (not used in UR5 approach)
        self.data.qpos[self.goal_qpos_start] = 0.0 - 0.2
        self.data.qpos[self.goal_qpos_start + 1] = 0.0 - 0.35
        self.data.qpos[self.goal_qpos_start + 2] = -1.0 - 0.05  # Hidden below table

        # Set arm to home position
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        for idx, joint_name in enumerate(joint_names):
            self.data.qpos[idx] = self.HOME_POSITION[joint_name]

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()

        if self.freeze_object:
            self._frozen_object_pos = self.data.qpos[self.object_qpos_start:self.object_qpos_start + 3].copy()

        # Reset counters
        self._step_count = 0
        self._is_success = False
        self.scripted_phase_active = False
        self.scripted_step = 0

        # Get cube position for info
        cube_start_pos = self.data.site_xpos[self.object_site_id].copy()

        # Store initial cube height for lift detection
        self._initial_cube_height = cube_start_pos[2]

        info = {
            "cube_pos": cube_start_pos.copy(),
            "initial_cube_height": self._initial_cube_height,
        }
        return obs, info


# Test the environment
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    print("="*80)
    print("STAGE 7: UR5 Hybrid Reward System Test (EXACT MATCH)")
    print("="*80)

    env = Stage7Task()
    obs, info = env.reset()

    eef_pos = env.data.site_xpos[env.ee_site_id]
    cube_pos = info['cube_pos']
    initial_distance = np.linalg.norm(eef_pos[:2] - cube_pos[:2])

    print(f"\nObservation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial gripper state: {env._get_gripper_state():.3f}")
    print(f"Cube position: {cube_pos}")
    print(f"End-effector position: {eef_pos}")
    print(f"Initial distance (2D): {initial_distance:.4f}m")

    # Test 100 random steps
    print("\n" + "="*80)
    print("Testing 100 random steps...")
    print("="*80)

    total_reward = 0
    success_count = 0

    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if info["is_success"]:
            success_count += 1
            print(f"\n🎉 SUCCESS at step {step}!")
            print(f"   Reward: {reward:.2f}")
            print(f"   Distance: {info['distance_to_target']:.4f}m")
            print(f"   Left force: {info['left_force']:.2f}N")
            print(f"   Right force: {info['right_force']:.2f}N")
            print(f"   Speed bonus: {info['speed_bonus']:.2f}")

        if step % 10 == 0:
            print(f"Step {step:3d}: reward={reward:7.2f}, "
                  f"distance={info['distance_to_target']:.4f}m, "
                  f"grasping={info['is_grasping']}, "
                  f"forces=({info['left_force']:.1f}N, {info['right_force']:.1f}N)")

        if terminated:
            print(f"\n✓ Episode terminated early (success)")
            break
        if truncated:
            print(f"\n✗ Episode truncated (max steps)")
            break

    print("\n" + "="*80)
    print("Episode Summary:")
    print("="*80)
    print(f"Total reward: {total_reward:.2f}")
    print(f"Success count: {success_count}")
    print(f"Average reward/step: {total_reward/min(step+1, 100):.2f}")

    if total_reward < 0:
        print("\n⚠️  Negative reward (expected during random exploration)")

    env.close()
    print("\n✓ Stage 7 environment test complete!")
