# Stage 7: Adding Scripted Movements

**Date:** 2026-01-31
**Purpose:** Guide for adding scripted movement sequences to Stage 7

---

## Overview

You can add scripted movements to Stage 7 in several ways while keeping the reward system intact. Here are the main approaches:

---

## Option 1: Pre-Scripted Initial Approach

**When to use:** You want to skip the initial approach phase and focus agent learning on fine positioning.

### How It Works

```python
Phase 1: SCRIPTED (Steps 1-50)
  - Script moves gripper toward cube
  - Gets within 5cm of cube
  - Hands off to agent

Phase 2: AGENT (Steps 51-100)
  - Agent takes over for fine positioning
  - Learns final 5cm → 1cm approach
  - Gets reward/learns as normal
```

### Implementation

**File created:** `envs/stage_7_scripted.py`

**Key features:**
```python
class Stage7Scripted(Stage7Task):
    USE_SCRIPTED_APPROACH = True
    SCRIPTED_APPROACH_STEPS = 50
    SCRIPTED_HANDOFF_DISTANCE = 0.05  # 5cm

    def step(self, action):
        if self.scripted_phase:
            # Override action with script
            action = self._get_scripted_action()

        # Normal step() continues
        return super().step(action)
```

### Benefits

✅ **Faster learning:** Agent only learns last 5cm
✅ **Reduced exploration:** Skip random wandering phase
✅ **Focused training:** Agent learns precision, not navigation
✅ **Same rewards:** Reward system unchanged

### Drawbacks

❌ **Limited learning:** Agent never learns full navigation
❌ **Script dependency:** Relies on scripted approach quality
❌ **Transfer issues:** Won't work if starting position changes

---

## Option 2: Triggered Scripted Grasp

**When to use:** You want agent to learn positioning, but automate the grasping motion.

### How It Works

```python
Phase 1: AGENT CONTROL (Steps 1-70)
  - Agent learns positioning
  - Gets gripper within 1cm of cube
  - Distance < 1cm → TRIGGER

Phase 2: SCRIPTED GRASP (Steps 71-90)
  - Script takes over
  - Executes grasp sequence:
    1. Fine-tune height
    2. Close gripper (20 steps)
    3. Check force
  - Returns control to agent

Phase 3: SUCCESS/FAILURE
  - Check if grasp succeeded
  - Give reward
  - Terminate episode
```

### Implementation Example

```python
def step(self, action):
    # Normal agent control
    obs, reward, terminated, truncated, info = super().step(action)

    # Check trigger condition
    if info['distance_to_target'] <= 0.01 and not self.grasp_sequence_running:
        # START SCRIPTED GRASP SEQUENCE
        self.grasp_sequence_running = True
        self.grasp_step = 0

    # Execute grasp sequence if active
    if self.grasp_sequence_running:
        reward, terminated = self._execute_grasp_sequence()

    return obs, reward, terminated, truncated, info

def _execute_grasp_sequence(self):
    """Scripted grasping sequence."""
    if self.grasp_step < 5:
        # Step 1-5: Fine-tune height
        self._adjust_height()
    elif self.grasp_step < 25:
        # Step 6-25: Close gripper
        self.data.ctrl[5] = self.gripper_min
    elif self.grasp_step == 25:
        # Step 25: Check success
        is_grasping, left_force, right_force = self._check_force_grasp()

        if is_grasping:
            reward = 100 + speed_bonus
            terminated = True

        self.grasp_sequence_running = False

    self.grasp_step += 1
    return reward, terminated
```

### Benefits

✅ **Reliable grasping:** Script ensures correct grasp execution
✅ **Agent learns positioning:** Full navigation learned by agent
✅ **Consistent behavior:** Same grasp every time
✅ **UR5-like:** Similar to UR5's automatic grasping

### Drawbacks

❌ **No grasp learning:** Agent doesn't learn gripper control
❌ **Fixed strategy:** Can't adapt grasp to object variation
❌ **Script quality:** Depends on scripted grasp quality

---

## Option 3: Hybrid Control (Script Some Joints, Agent Others)

**When to use:** You want to simplify learning by scripting difficult joints while agent learns others.

### How It Works

```python
Every step:
  - Agent action: [shoulder_pan, shoulder_lift, elbow_flex]
  - Script controls: [wrist_flex, wrist_roll, gripper]

Agent learns: Arm positioning (3 DOF)
Script handles: Wrist orientation + gripper (3 DOF)
```

### Implementation Example

```python
def step(self, action):
    # Agent controls first 3 joints
    agent_joints = action[:3]

    # Script controls last 3 joints
    scripted_joints = self._get_scripted_wrist_and_gripper()

    # Combine
    full_action = np.concatenate([agent_joints, scripted_joints])

    # Execute
    return super().step(full_action)

def _get_scripted_wrist_and_gripper(self):
    """Script wrist orientation and gripper opening."""
    eef_pos = self.data.site_xpos[self.ee_site_id]
    cube_pos = self.data.site_xpos[self.object_site_id]
    distance = np.linalg.norm(eef_pos - cube_pos)

    # Scripted wrist_flex (adjust based on distance)
    if distance < 0.05:
        wrist_flex = 0.5  # Lower wrist when close
    else:
        wrist_flex = 0.8  # Higher wrist when far

    # Wrist_roll locked (already handled)
    wrist_roll = 0.0

    # Gripper: Open until very close
    if distance < 0.01:
        gripper = -1.0  # Close
    else:
        gripper = 1.0   # Open

    return np.array([wrist_flex, wrist_roll, gripper])
```

### Benefits

✅ **Simplified learning:** 3 DOF vs 6 DOF (easier)
✅ **Focused training:** Agent learns arm positioning only
✅ **Script assists:** Wrist/gripper automated
✅ **Faster convergence:** Simpler action space

### Drawbacks

❌ **Limited control:** Agent can't learn full manipulation
❌ **Script constraints:** Wrist/gripper behavior fixed
❌ **Less flexible:** Can't adapt to different orientations

---

## Option 4: Waypoint-Based Scripted Path

**When to use:** You want agent to learn sub-goals, but follow scripted path between waypoints.

### How It Works

```python
Define waypoints:
  1. Start position
  2. Above cube (10cm)
  3. At cube height
  4. Positioned for grasp

Agent learns: How to reach each waypoint
Script handles: Path between waypoints
```

### Implementation Example

```python
WAYPOINTS = [
    "start",      # Home position
    "approach",   # 10cm above cube
    "descend",    # At cube height
    "position",   # Within 1cm of cube
    "grasp"       # Gripper closed
]

def step(self, action):
    current_waypoint = self._get_current_waypoint()
    next_waypoint = self._get_next_waypoint()

    # If between waypoints, use scripted motion
    if self.moving_to_waypoint:
        action = self._interpolate_to_waypoint(next_waypoint)

    # If at waypoint, agent decides direction
    else:
        # Agent's action determines next waypoint
        if self._reached_waypoint(next_waypoint):
            self.moving_to_waypoint = False
            self.current_waypoint_idx += 1

    return super().step(action)
```

### Benefits

✅ **Structured learning:** Clear sub-goals
✅ **Smooth motion:** Scripted interpolation
✅ **Curriculum-like:** Progress through stages
✅ **Easier debugging:** Can inspect waypoint progress

### Drawbacks

❌ **Fixed path:** Scripted trajectory between waypoints
❌ **Waypoint design:** Requires good waypoint selection
❌ **Less exploration:** Limited to waypoint graph

---

## Option 5: Force Control Override

**When to use:** You want to add specific force-controlled behaviors (e.g., "push down gently").

### Implementation Example

```python
def step(self, action):
    obs, reward, terminated, truncated, info = super().step(action)

    # If positioned, add scripted downward force
    if info['distance_to_target'] < 0.02:
        # Apply gentle downward force
        self.data.ctrl[3] -= 0.1  # wrist_flex down

    # If touching cube, add scripted squeeze force
    if info['left_force'] > 0 or info['right_force'] > 0:
        # Close gripper gently
        self.data.ctrl[5] = max(self.data.ctrl[5] - 0.05, -1.0)

    return obs, reward, terminated, truncated, info
```

---

## Recommended Approach for Stage 7

### For Your Use Case

Based on Stage 7's current configuration (vertical jaws, frozen gripper, freeze on positioning), I recommend:

**Option 2: Triggered Scripted Grasp**

```python
1. Agent learns positioning (0-70 steps)
   - Navigate to cube
   - Align vertically (descend from above)
   - Get distance < 1cm

2. Script triggers grasp (steps 71-90)
   - Unfreeze gripper
   - Close gripper (20 steps)
   - Check forces
   - Give reward

3. Episode ends
   - Success or failure determined
   - Agent gets reward
   - Reset for next episode
```

### Why This Works Best

✅ **Builds on current setup:** Uses freeze behavior you have
✅ **Clear separation:** Agent does positioning, script does grasping
✅ **UR5-like:** Matches UR5's automatic grasping
✅ **Easier to debug:** Can test positioning and grasping separately
✅ **Faster training:** Agent only learns positioning (simpler)

---

## Implementation Steps

### Step 1: Modify Stage7Task

Add triggered grasp sequence:

```python
def step(self, action):
    # ... existing step code ...

    # Check for grasp trigger
    if info.get('frozen', False) and not hasattr(self, 'grasp_triggered'):
        # TRIGGER GRASP SEQUENCE
        self.grasp_triggered = True
        self._execute_grasp_sequence()

    return obs, reward, terminated, truncated, info

def _execute_grasp_sequence(self):
    """Execute scripted grasp after positioning."""
    # Unfreeze gripper
    original_freeze_state = self.FREEZE_GRIPPER_OPEN
    self.FREEZE_GRIPPER_OPEN = False

    # Close gripper over 20 steps
    for i in range(20):
        self.data.ctrl[5] = self.gripper_min
        mujoco.mj_step(self.model, self.data)

    # Check if grasp succeeded
    is_grasping, left_force, right_force = self._check_force_grasp()

    # Restore freeze state
    self.FREEZE_GRIPPER_OPEN = original_freeze_state

    return is_grasping
```

### Step 2: Test Scripted Sequence

```bash
python envs/stage_7_scripted.py
```

### Step 3: Train with Script

```bash
python scripts/train.py --stage 7 --timesteps 800000
```

---

## Customization Examples

### Custom Script 1: Slow Descent

```python
def _get_scripted_action(self):
    """Slowly descend gripper toward cube."""
    action = np.zeros(6)
    action[3] = -0.3  # Slow downward wrist_flex
    return action
```

### Custom Script 2: Spiral Approach

```python
def _get_scripted_action(self):
    """Spiral approach to cube."""
    t = self.scripted_step_count / 50.0
    action = np.zeros(6)
    action[0] = 0.5 * np.cos(2 * np.pi * t)  # Circular X
    action[1] = 0.5 * np.sin(2 * np.pi * t)  # Circular Y
    action[3] = -0.3  # Descend Z
    return action
```

### Custom Script 3: Two-Phase Grasp

```python
def _execute_grasp_sequence(self):
    """Two-phase grasp: align, then close."""
    # Phase 1: Align height (10 steps)
    for i in range(10):
        self.data.ctrl[3] = -0.5  # Adjust wrist
        mujoco.mj_step(self.model, self.data)

    # Phase 2: Close gripper (10 steps)
    for i in range(10):
        self.data.ctrl[5] = self.gripper_min
        mujoco.mj_step(self.model, self.data)

    return self._check_force_grasp()[0]
```

---

## Testing Your Scripted Movements

### Test Script Alone

```bash
# Test scripted approach
python envs/stage_7_scripted.py
```

### Test with Trained Agent

```bash
# Test full behavior
python scripts/visualize_stage7_positioning.py \
    --checkpoint checkpoints/stage_7/stage_7_task_XXXXX_steps.zip
```

### Debug Script

```python
# Add debug prints
def _get_scripted_action(self):
    action = ...
    print(f"Script: {action}")  # See what script does
    return action
```

---

## Summary

**Multiple ways to add scripted movements:**

1. **Pre-scripted approach** - Script initial navigation
2. **Triggered grasp** - Script grasping after positioning ✓ RECOMMENDED
3. **Hybrid control** - Script some joints, agent others
4. **Waypoint-based** - Script paths between waypoints
5. **Force control** - Add force-based behaviors

**For Stage 7, I recommend Option 2 (Triggered Grasp):**
- Agent learns positioning
- Script handles grasping
- Matches current freeze-on-position behavior
- UR5-like approach
- Faster training

**Next step:** Tell me what specific scripted movements you want, and I'll implement them!
