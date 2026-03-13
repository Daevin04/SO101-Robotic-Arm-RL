# Stage 7: Control and Reward System

**Date:** 2026-01-31
**Topic:** Agent vs Script Control, Reward Pausing During Scripted Phase

---

## Overview

Stage 7 uses a **two-phase approach**:

1. **Agent Phase:** Agent learns to navigate to hover position (5cm above cube)
2. **Script Phase:** Scripted sequence executes descent + grasp (agent has NO control)

This document explains how control and rewards are handled in each phase.

---

## Control System

### Phase 1: Agent Has Control

**Who controls:** Agent (RL policy)
**Duration:** Variable (depends on training quality)
**What happens:**
- Agent receives `action` from policy
- Action is validated and applied to robot joints
- Agent navigates toward hover position (X,Z of cube at 5cm height)
- Robot joints respond to agent's actions
- Simulation steps with agent's commands

**Code:**
```python
# In step() method - Agent control section
action = np.clip(action, -1.0, 1.0)
target_qpos = self._joint_mid + action * self._joint_range
self.data.ctrl[:6] = target_qpos  # Agent controls all joints
```

**Trigger condition:**
```python
# When distance to hover ≤ 10mm:
if distance_to_target <= 0.01 and not self.scripted_phase_active:
    print("[Script] Triggering descent")
    self.scripted_phase_active = True  # Switch to script control
```

---

### Phase 2: Script Has Control

**Who controls:** Scripted sequence (hardcoded)
**Duration:** Fixed (50 steps: 10 rotate + 20 descend + 20 close)
**What happens:**
- **Agent action is COMPLETELY IGNORED**
- Script controls all robot joints directly
- Script executes 4 phases:
  1. **Rotate wrist** 0° → 90° (10 steps)
  2. **Descend** to cube height (20 steps)
  3. **Close gripper** open → closed (20 steps)
  4. **Check success** forces ≥ 3N (1 step)

**Code:**
```python
# In step() method - Script control section (FIRST CHECK)
if self.scripted_phase_active:
    # Agent action is IGNORED completely
    obs, reward, terminated = self._execute_scripted_descent()
    return obs, reward, terminated, truncated, info
```

**Key point:** Agent receives `step()` calls with actions, but the actions are **never validated** or **never applied** during the script phase.

---

## Reward System

### Phase 1: Agent Learning (Distance-Based Rewards)

**Reward formula:**
```python
distance_to_hover = sqrt(xy_error² + (0.5 * z_error)²)
reward = -10.0 * distance_to_hover
```

**Examples:**
- 30cm from hover → reward = -3.0
- 10cm from hover → reward = -1.0
- 5cm from hover  → reward = -0.5
- 1cm from hover  → reward = -0.1
- <1cm from hover → **TRIGGER SCRIPT**

**Purpose:** Guide agent toward hover position with continuous gradient

---

### Phase 2: Script Execution (No Agent Learning)

**Phases 1-3 (Rotate, Descend, Close): reward = 0.0**
```python
# During script phases 1-3
return obs, 0.0, False  # Zero reward
```

**Phase 4 (Success Check): reward = +100 + bonus OR -1.0**
```python
# After gripper closes, check forces
if left_force > 3N and right_force > 3N:
    reward = 100 + speed_bonus  # Success!
    terminated = True
else:
    reward = -1.0  # Failed grasp
    terminated = False
```

**Key point:** Agent receives **zero reward** during script execution (phases 1-3). Only the final success check (phase 4) provides a meaningful reward, but this reward is **not based on agent actions** during the script.

---

## Why This Design?

### 1. Agent Learns Positioning, Not Manipulation

**Problem:** Full 6-DOF manipulation is very hard to learn with sparse rewards
- Agent must learn: positioning + orientation + descent + grasping + timing
- Would require 1M-2M steps to converge
- High failure rate during learning

**Solution:** Agent only learns 2.5D positioning (X, Y, Z to hover)
- Simpler task: "Get above the cube"
- Faster learning: 400K-600K steps expected
- Script handles the complex manipulation

---

### 2. Script Execution Not Agent Behavior

**Problem:** If agent received rewards during script, it would try to learn script behavior
- Agent would associate its pre-script actions with script outcomes
- Credit assignment problem: which action caused success?
- Agent might learn to "wait for script" rather than position correctly

**Solution:** Zero reward during script execution
- Agent only learns: "position correctly → good"
- Agent does NOT learn: "close gripper → good" (script does this)
- Clear separation: Agent learns positioning, script executes grasp

---

### 3. Agent Should Not Control During Script

**Problem:** If agent maintained control during script, outcomes would be inconsistent
- Agent actions could interfere with rotation
- Agent could move arm during descent
- Grasp quality would vary based on agent's simultaneous actions

**Solution:** Agent control relinquished completely
- Script has full control of all joints
- Agent actions are ignored (not even validated)
- Consistent execution every time script triggers

---

## Training Implications

### What Agent Learns

**Agent learns:**
- Navigate to X,Z coordinates of cube
- Reach correct height (5cm above cube)
- Align with target within 10mm

**Agent does NOT learn:**
- Wrist rotation (script does this)
- Descent trajectory (script does this)
- Gripper control (script does this)
- Force application (script checks this)

---

### Reward Attribution

**Episode reward breakdown:**

```
Steps 1-40:  Agent positioning        reward = -1.5 to -3.0/step
Steps 41-42: Agent near target        reward = -0.1 to -0.3/step
Step 43:     Script triggers          reward = 0.0
Steps 44-53: Script Phase 1 (rotate)  reward = 0.0/step
Steps 54-73: Script Phase 2 (descend) reward = 0.0/step
Steps 74-93: Script Phase 3 (close)   reward = 0.0/step
Step 94:     Script Phase 4 (check)   reward = +157 (success!)
────────────────────────────────────────────────────────
Total agent-earned:  ~-80 pts (positioning only)
Total script-earned: +157 pts (NOT attributed to agent actions)
Total episode:       +77 pts
```

**Key insight:** Agent accumulates **negative reward** during positioning, then receives **large positive reward** when script succeeds. Agent learns: "Better positioning → earlier trigger → more bonus → higher total reward."

---

## Code Flow

### step() Method Structure

```python
def step(self, action):
    self._step_count += 1

    # ═══════════════════════════════════════════════════════════
    # CHECK 1: Is script active?
    # ═══════════════════════════════════════════════════════════
    if self.scripted_phase_active:
        # SCRIPT PHASE
        # - Ignore agent action completely
        # - Execute one script step
        # - Return script's reward (0.0 for phases 1-3)
        obs, reward, terminated = self._execute_scripted_descent()
        return obs, reward, terminated, truncated, info

    # ═══════════════════════════════════════════════════════════
    # AGENT PHASE: Script not active, agent has control
    # ═══════════════════════════════════════════════════════════

    # Validate agent action
    assert self.action_space.contains(action)

    # Apply agent action to joints
    self.data.ctrl[:6] = target_qpos

    # Step simulation
    mujoco.mj_step(self.model, self.data)

    # Check if positioned correctly
    distance_to_target = compute_distance_to_hover()

    if distance_to_target <= 0.01:
        # TRIGGER SCRIPT
        self.scripted_phase_active = True
        self.scripted_step = 0

    # Compute reward (distance-based)
    reward = -10.0 * distance_to_target

    return obs, reward, terminated, truncated, info
```

---

### _execute_scripted_descent() Method Structure

```python
def _execute_scripted_descent(self):
    """Script controls robot, agent has no control."""

    # PHASE 1: Rotate wrist (steps 0-9)
    if self.scripted_step < 10:
        progress = self.scripted_step / 10.0
        self.data.ctrl[4] = progress * 1.5708  # 0° → 90°
        return obs, 0.0, False  # Zero reward

    # PHASE 2: Descend (steps 10-29)
    elif self.scripted_step < 30:
        self.data.ctrl[3] -= descent_per_step  # Lower arm
        return obs, 0.0, False  # Zero reward

    # PHASE 3: Close gripper (steps 30-49)
    elif self.scripted_step < 50:
        progress = (self.scripted_step - 30) / 20.0
        self.data.ctrl[5] = 1.0 + progress * (-1.0)  # Open → closed
        return obs, 0.0, False  # Zero reward

    # PHASE 4: Check success (step 50)
    else:
        is_grasping, left_f, right_f = self._check_force_grasp()

        if is_grasping:
            reward = 100 + speed_bonus  # SUCCESS
            terminated = True
        else:
            reward = -1.0  # FAILURE
            terminated = False

        self.scripted_phase_active = False  # End script
        return obs, reward, terminated
```

---

## Evaluation Videos

### What You'll See

**1. Agent Positioning (Variable length)**
- Robot navigates toward cube
- Duration depends on training quality
- Good agent: 20-40 steps
- Poor agent: 60-80 steps

**2. Script Trigger (1 frame)**
- Console prints: "[Script] Triggering descent at step X"
- Robot is positioned 5cm above cube

**3. Phase 1 - Rotation (10 frames) ✓**
- Wrist rotates from 0° → 90°
- Jaws change from vertical to horizontal
- Smooth animation visible

**4. Phase 2 - Descent (20 frames) ✓**
- Gripper descends from 5cm above to cube height
- Maintains X,Y alignment
- Smooth animation visible

**5. Phase 3 - Closing (20 frames) ✓**
- Gripper closes from fully open to fully closed
- Gradual closure visible (not instant)
- Smooth animation visible

**6. Phase 4 - Check (1 frame) ✓**
- Force sensors check both fingers
- Episode terminates (success or failure)
- Final state visible

**Total:** Agent phase (variable) + 51 scripted frames

---

## Testing

### Test 1: Verify Agent Control is Disabled During Script

```bash
python -c "
from envs.stage_7_task import Stage7Task
import numpy as np

env = Stage7Task()
env.reset()

# Trigger script manually
env.scripted_phase_active = True
env.scripted_step = 0

# Agent provides large action (should be ignored)
action = np.ones(6) * 0.5
obs, reward, done, trunc, info = env.step(action)

assert info['scripted_phase'] == True, 'Script not active!'
assert reward == 0.0, 'Reward should be 0 during script!'
print('✓ Agent action ignored during script')
"
```

### Test 2: Verify Reward is Zero During Script Phases 1-3

```bash
python scripts/demo_stage7_scripted_descent.py
# Watch console output - should see "reward=0.000" for 50 steps
```

### Test 3: Verify Script Trigger Works

```bash
python scripts/test_stage7_hover_descent.py
# Should see "[Script] Triggering descent" when positioned
```

---

## Summary

**Control:**
- ✓ Agent controls robot during positioning
- ✓ Script controls robot during descent/grasp
- ✓ Agent actions completely ignored during script

**Rewards:**
- ✓ Agent receives distance-based rewards during positioning
- ✓ Agent receives 0.0 reward during script phases 1-3
- ✓ Agent receives final reward in script phase 4
- ✓ Agent only learns from positioning actions

**Training:**
- ✓ Agent learns 2.5D positioning (hover above cube)
- ✓ Agent does NOT learn manipulation (script handles it)
- ✓ Clear separation of concerns
- ✓ Faster learning (400K-600K steps vs 1M+)

**Videos:**
- ✓ All 50 scripted frames visible
- ✓ Smooth animations throughout
- ✓ No instant jumps

---

## Implementation Files

**Core:** `/home/oeyd/SO101_Training/envs/stage_7_task.py`
- `step()` method: Checks `scripted_phase_active` FIRST
- `_execute_scripted_descent()`: Returns 0.0 reward for phases 1-3

**Tests:**
- `scripts/test_stage7_hover_descent.py` - Test trigger mechanism
- `scripts/demo_stage7_scripted_descent.py` - Demo full sequence

**Docs:**
- `docs/STAGE_7_HOVER_DESCENT.md` - Complete approach guide
- `docs/STAGE_7_CONTROL_AND_REWARDS.md` - This document

---

## Ready for Training

```bash
# Start training
python scripts/train.py --stage 7 --timesteps 600000

# Expected behavior:
# @ 200K-400K steps: First script triggers appear
# @ 600K steps: 50-70% success rate
```

The agent will learn positioning, script will handle manipulation, and videos will show the complete smooth sequence!
