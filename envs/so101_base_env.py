"""
Base environment for SO-101 robotic grasping.

This base class contains shared functionality for all task environments:
- MuJoCo model loading
- Observation generation (29D: joints, EE pose, object/goal state, gripper)
- Helper utilities (contact detection, grasp checking, success criteria)
- Reset logic (randomize object/goal positions)
- Rendering

Task-specific environments inherit from this class and define:
- Reward structure (_compute_reward method)
- Action execution (step method)
- Task-specific parameters
"""

import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces
from pathlib import Path


class SO101BaseEnv(gym.Env):
    """Base environment with shared functionality."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # Constants
    MAX_EPISODE_STEPS = 150
    SUCCESS_DIST_THRESHOLD = 0.05  # 5cm to goal
    LIFT_HEIGHT_THRESHOLD = 0.03   # 3cm above table

    # Joint limits (for action scaling)
    JOINT_LIMITS = {
        "shoulder_pan": (-1.57, 1.57),
        "shoulder_lift": (-1.57, 1.57),
        "elbow_flex": (-2.0, 2.0),
        "wrist_flex": (-1.57, 1.57),
        "wrist_roll": (-3.14, 3.14),
        "gripper": (0.0, 1.0),
    }

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # Load MuJoCo model
        model_path = Path(__file__).parent.parent / "assets" / "so101_pusher.xml"
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # Rendering setup (MUJOCO_GL should be set before importing mujoco)
        self.viewer = None
        self.renderer = None
        if self.render_mode in ["human", "rgb_array"]:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            # Set camera to use angled side view
            try:
                self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "side_angle")
            except:
                self.camera_id = -1  # Use default camera if named camera not found

        # Get important IDs
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")  # FIXED: was "ee_site"
        self.object_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "object_site")
        self.goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal_site")
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_body")

        # Fingertip sites for accurate grasping detection
        try:
            self.left_fingertip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_fingertip")
            self.right_fingertip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_fingertip")
        except:
            # Fallback if fingertip sites don't exist
            self.left_fingertip_id = self.ee_site_id
            self.right_fingertip_id = self.ee_site_id

        # Geom IDs for contact detection
        self.object_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "object")
        self.gripper_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_body")
        try:
            self.jaw_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "jaw_body")
        except:
            self.jaw_body_id = self.gripper_body_id

        # Joint indices
        self.n_robot_joints = 6
        self.gripper_joint_idx = 5
        self.object_qpos_start = 6  # Object slide joints
        self.goal_qpos_start = 9    # Goal slide joints

        # Observation space: 29D (sim-to-real compatible)
        # [joint_pos(6), joint_vel(6), ee_pos(3), ee_orientation(4), obj_pos(3), obj_vel(3), goal_pos(3), gripper_state(1)]
        # Note: Removed grasped flag (requires force sensor not available on real robot)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(29,), dtype=np.float64
        )

        # Action space defined by subclasses
        self.action_space = None

        # State tracking
        self._step_count = 0
        self._initial_obj_height = 0.0
        self._positioned_steps = 0  # Consecutive steps in positioned state
        self._grasped_steps = 0     # Consecutive steps in grasped state
        self._contact_lost_steps = 0  # Consecutive steps without contact

    def _get_obs(self):
        """Get current observation."""
        # Joint positions and velocities
        joint_pos = self.data.qpos[:self.n_robot_joints].copy()
        joint_vel = self.data.qvel[:self.n_robot_joints].copy()

        # End-effector position
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()

        # End-effector orientation (NEW!)
        ee_xmat = self.data.xmat[self.ee_body_id].reshape(3, 3)
        ee_quat = self._rotation_matrix_to_quaternion(ee_xmat)

        # Object position and velocity
        obj_pos = self.data.site_xpos[self.object_site_id].copy()
        obj_vel = self.data.qvel[self.n_robot_joints:self.n_robot_joints+3].copy()

        # Goal position
        goal_pos = self.data.site_xpos[self.goal_site_id].copy()

        # Gripper state
        gripper_state = np.array([self._get_gripper_state()])

        # Note: Grasped flag removed for sim-to-real compatibility
        # Agent must infer grasp success from proprioception and object motion

        obs = np.concatenate([
            joint_pos,      # 6
            joint_vel,      # 6
            ee_pos,         # 3
            ee_quat,        # 4
            obj_pos,        # 3 (from camera on real robot)
            obj_vel,        # 3 (from camera on real robot)
            goal_pos,       # 3 (from camera on real robot)
            gripper_state,  # 1
        ])  # Total: 29D (sim-to-real compatible)

        return obs.astype(np.float64)

    def _get_gripper_state(self):
        """Get normalized gripper state (0=closed, 1=open)."""
        gripper_pos = self.data.qpos[self.gripper_joint_idx]
        limits = self.JOINT_LIMITS["gripper"]
        normalized = (gripper_pos - limits[0]) / (limits[1] - limits[0])
        return float(np.clip(normalized, 0.0, 1.0))

    def _has_gripper_contact(self):
        """Check if gripper is in contact with object."""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            # Check if contact involves object and gripper
            if (geom1 == self.object_geom_id or geom2 == self.object_geom_id):
                if (self.model.geom_bodyid[geom1] in [self.gripper_body_id, self.jaw_body_id] or
                    self.model.geom_bodyid[geom2] in [self.gripper_body_id, self.jaw_body_id]):
                    return True
        return False

    def _is_grasped(self):
        """Check if object is grasped (closed gripper + contact)."""
        gripper_state = self._get_gripper_state()
        has_contact = self._has_gripper_contact()
        gripper_closed = gripper_state < 0.25
        return has_contact and gripper_closed

    def _rotation_matrix_to_quaternion(self, R):
        """
        Convert 3x3 rotation matrix to quaternion [w, x, y, z].
        Uses Shepperd's method for numerical stability.
        """
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return np.array([w, x, y, z], dtype=np.float64)

    def _is_positioned(self):
        """
        Check if gripper is correctly positioned for grasping.

        Returns True if ALL conditions met:
        - Close to object
        - Height matched
        - Jaws horizontal (left/right orientation)
        - Forward axis horizontal
        - Object centered between jaws
        - Gripper is open
        """
        ee_pos = self.data.site_xpos[self.ee_site_id]
        obj_pos = self.data.site_xpos[self.object_site_id]
        ee_xmat = self.data.xmat[self.ee_body_id].reshape(3, 3)
        gripper_state = self._get_gripper_state()

        # Distance check
        ee_to_obj = ee_pos - obj_pos
        ee_to_obj_dist = np.linalg.norm(ee_to_obj)
        if ee_to_obj_dist > 0.08:
            return False

        # Height check
        height_diff = abs(ee_pos[2] - obj_pos[2])
        if height_diff > 0.03:
            return False

        # Orientation check: jaws horizontal
        jaw_axis = ee_xmat[:, 1]  # Y-axis
        jaw_horizontal = 1.0 - abs(jaw_axis[2])
        if jaw_horizontal < 0.85:
            return False

        # Orientation check: forward horizontal
        forward_axis = ee_xmat[:, 0]  # X-axis
        forward_horizontal = 1.0 - abs(forward_axis[2])
        if forward_horizontal < 0.85:
            return False

        # Jaw centering check
        side_offset = abs(np.dot(ee_to_obj, jaw_axis))
        if side_offset > 0.02:
            return False

        # Gripper must be open
        if gripper_state < 0.7:
            return False

        return True

    def _is_position_lost(self):
        """Check if positioning has been lost (for recovery)."""
        ee_pos = self.data.site_xpos[self.ee_site_id]
        obj_pos = self.data.site_xpos[self.object_site_id]
        ee_xmat = self.data.xmat[self.ee_body_id].reshape(3, 3)

        # Distance check
        ee_to_obj_dist = np.linalg.norm(ee_pos - obj_pos)
        if ee_to_obj_dist > 0.12:
            return True

        # Orientation check
        jaw_axis = ee_xmat[:, 1]
        forward_axis = ee_xmat[:, 0]
        jaw_horizontal = 1.0 - abs(jaw_axis[2])
        forward_horizontal = 1.0 - abs(forward_axis[2])

        if jaw_horizontal < 0.6 or forward_horizontal < 0.6:
            return True

        return False

    def _is_grasp_lost(self):
        """Check if grasp has been lost (for recovery)."""
        has_contact = self._has_gripper_contact()

        # Track consecutive steps without contact
        if not has_contact:
            self._contact_lost_steps += 1
        else:
            self._contact_lost_steps = 0

        # Lost if no contact for 10 consecutive steps
        if self._contact_lost_steps >= 10:
            return True

        # Also check distance
        ee_pos = self.data.site_xpos[self.ee_site_id]
        obj_pos = self.data.site_xpos[self.object_site_id]
        if np.linalg.norm(ee_pos - obj_pos) > 0.10:
            return True

        return False

    def _is_success(self):
        """Check if task is complete (object at goal)."""
        obj_pos = self.data.site_xpos[self.object_site_id]
        goal_pos = self.data.site_xpos[self.goal_site_id]
        dist = np.linalg.norm(obj_pos - goal_pos)
        return dist < self.SUCCESS_DIST_THRESHOLD

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)

        # Randomize object position on table
        obj_x = self.np_random.uniform(0.12, 0.25)
        obj_y = self.np_random.uniform(0.12, 0.22)

        # Randomize goal position
        goal_x = self.np_random.uniform(0.15, 0.30)
        goal_y = self.np_random.uniform(0.25, 0.40)
        goal_z = self.np_random.uniform(0.03, 0.08)

        # Set positions
        self.data.qpos[self.object_qpos_start] = obj_x - 0.15
        self.data.qpos[self.object_qpos_start + 1] = obj_y - 0.2
        self.data.qpos[self.object_qpos_start + 2] = 0

        self.data.qpos[self.goal_qpos_start] = goal_x - 0.2
        self.data.qpos[self.goal_qpos_start + 1] = goal_y - 0.35
        self.data.qpos[self.goal_qpos_start + 2] = goal_z - 0.05

        # Start with gripper open
        self.data.qpos[self.gripper_joint_idx] = self.JOINT_LIMITS["gripper"][1] * 0.8

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

        # Store initial object height
        self._initial_obj_height = self.data.site_xpos[self.object_site_id][2]

        # Reset tracking
        self._step_count = 0
        self._positioned_steps = 0
        self._grasped_steps = 0
        self._contact_lost_steps = 0

        obs = self._get_obs()
        info = {"is_success": False, "is_positioned": False, "is_grasped": False}

        return obs, info

    def render(self):
        """Render the environment."""
        if self.render_mode in ["human", "rgb_array"]:
            if self.renderer is not None:
                self.renderer.update_scene(self.data, camera=self.camera_id)
                pixels = self.renderer.render()

                if self.render_mode == "human":
                    # Display using cv2 if available
                    try:
                        import cv2
                        cv2.imshow("SO-101 Environment", pixels[::-1, :, :])  # Flip vertically
                        cv2.waitKey(1)
                    except ImportError:
                        # Fallback: return pixels (user needs to display manually)
                        pass
                    return pixels
                else:  # rgb_array
                    return pixels
        return None

    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        # Close cv2 windows if using human rendering
        if self.render_mode == "human":
            try:
                import cv2
                cv2.destroyAllWindows()
            except ImportError:
                pass
