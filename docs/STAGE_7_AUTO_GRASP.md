# Stage 7: Automatic Grasping Feature

**Date:** 2026-01-31
**Purpose:** Replicate UR5's scripted grasping behavior

---

## What It Does

When **AUTO_GRASP_ENABLED = True** (default), the gripper **automatically closes** when the agent positions the end-effector within 1cm of the cube.

### Behavior

```python
if distance_to_target <= 0.01m:
    # 1. Force gripper to close
    gripper_actuator = CLOSED

    # 2. Run 20 simulation steps (~0.66 seconds)
    #    for gripper to physically close

    # 3. Check if grasp succeeded (force-based)
    if left_force > 3N and right_force > 3N:
        reward = +100 + speed_bonus
        success = True
    else:
        reward = -1.0  # Positioned but didn't grasp
        success = False
```

---

## What The Agent Learns

### With Auto-Grasp (Current - UR5 Approach)

**Agent learns:**
- ✅ 3D positioning (X, Y, Z navigation)
- ✅ Joint coordination to reach target
- ✅ When to stop (at target position)

**Agent does NOT learn:**
- ❌ When to close gripper (automatic!)
- ❌ How to close gripper (forced full close)
- ❌ Timing of grasp (triggered by position)

**Problem complexity:** 3D navigation (moderate difficulty)

### Without Auto-Grasp (Set to False)

**Agent learns:**
- ✅ 3D positioning (X, Y, Z navigation)
- ✅ Joint coordination
- ✅ **When to close gripper** (timing)
- ✅ **How much to close** (force control)
- ✅ **Gripper control strategy**

**Problem complexity:** Full 6-DOF manipulation (high difficulty)

---

## Expected Learning Times

### With Auto-Grasp (Easier)

```
Phase 1 (0-200K):   Learn to approach cube
Phase 2 (200K-400K): Get within 1cm consistently
Phase 3 (400K-600K): Refine positioning
Phase 4 (600K+):     Mastery (fast, efficient approach)

Expected first success: 300K-400K steps
Expected mastery (>70%): 600K-800K steps
```

**Why faster:** Agent only learns navigation, grasping is guaranteed

### Without Auto-Grasp (Harder)

```
Phase 1 (0-300K):   Learn to approach + position
Phase 2 (300K-600K): Discover grasping strategy
Phase 3 (600K-1M):   Refine grasp timing
Phase 4 (1M+):       Mastery (reliable grasping)

Expected first success: 500K-800K steps
Expected mastery (>70%): 1M-1.5M steps
```

**Why slower:** Agent must discover full manipulation pipeline

---

## Configuration

### Enable/Disable Auto-Grasp

**File:** `/home/oeyd/SO101_Training/envs/stage_7_task.py`

**Line ~148:**
```python
# Automatic grasping (UR5 behavior replication)
AUTO_GRASP_ENABLED = True   # Set to False to disable
AUTO_GRASP_STEPS = 20       # Simulation steps for closing (0.66 sec)
```

### When to Use Each Mode

**Use AUTO_GRASP_ENABLED = True when:**
- ✅ Replicating UR5 results
- ✅ Want faster training
- ✅ Testing navigation learning
- ✅ Comparing to UR5 benchmark
- ✅ Limited compute budget

**Use AUTO_GRASP_ENABLED = False when:**
- ✅ Learning full manipulation
- ✅ Deploying to real robot
- ✅ Testing agent capabilities
- ✅ Research on grasping strategies
- ✅ Have 1M+ steps compute budget

---

## Comparison to UR5

### UR5 Original

```
Action space: [x_position, y_position]  (2D only!)
Gripper control: Automatic when distance < 1cm
Z positioning: Automatic (forced to 0.8m)
Lifting: Automatic sequence after grasp
```

### Our Stage 7 (Auto-Grasp Enabled)

```
Action space: [6 joint angles]  (Full robot control)
Gripper control: Automatic when distance < 1cm ✓
Z positioning: Learned by agent (3D navigation)
Lifting: Not required for reward
```

**Key difference:** We still learn 3D navigation (harder than UR5's 2D), but gripper closing is automatic (like UR5).

---

## Training Example

### Starting Training

```bash
# With auto-grasp (default)
python scripts/train.py --stage 7 --timesteps 800000

# Without auto-grasp (full manipulation)
# First, set AUTO_GRASP_ENABLED = False in stage_7_task.py
python scripts/train.py --stage 7 --timesteps 1500000
```

### Monitoring Progress

**With auto-grasp enabled:**
- Watch for distance decreasing: 0.10m → 0.05m → 0.02m → 0.01m
- First success when distance consistently < 0.01m
- Success rate should ramp up quickly once positioning learned

**With auto-grasp disabled:**
- Watch for both distance AND force metrics
- First success requires discovering gripper closing
- Success rate ramps up slower (harder problem)

---

## Info Dictionary

The `info` returned from `step()` now includes:

```python
info = {
    "is_success": bool,              # Grasp succeeded
    "distance_to_target": float,     # 3D distance (weighted)
    "is_grasping": bool,             # Force check passed
    "left_force": float,             # Left finger force (N)
    "right_force": float,            # Right finger force (N)
    "speed_bonus": float,            # Speed bonus earned
    "reward": float,                 # Reward value
    "episode_step": int,             # Current step count
    "auto_grasp_triggered": bool,    # Was auto-grasp used this step? (NEW)
}
```

**Use `auto_grasp_triggered` to:**
- Track when agent gets positioned correctly
- Debug if auto-grasp is activating too early/late
- Analyze learning progression

---

## Implementation Details

### Code Flow

```python
# In step() method:

# 1. Agent takes action (controls 6 joints)
target_qpos = action_to_joint_positions(action)
apply_action(target_qpos)
step_simulation()

# 2. Check if positioned correctly
distance = calculate_distance(eef_pos, cube_pos)

# 3. Auto-grasp check
if AUTO_GRASP_ENABLED and distance <= 0.01:
    # Force gripper closed
    gripper_actuator = CLOSED

    # Run 20 steps for gripper to close
    for _ in range(20):
        step_simulation()

    # Check forces
    is_grasping = check_force_threshold()

# 4. Compute reward
if is_grasping:
    reward = +100 + speed_bonus
else:
    reward = -10 * distance

# 5. Return
return obs, reward, terminated, truncated, info
```

### Gripper Closing Duration

**AUTO_GRASP_STEPS = 20:**
- 20 simulation steps × 0.033 sec/step = 0.66 seconds
- Realistic gripper closing time
- Allows physics to settle

**To adjust closing speed:**
```python
AUTO_GRASP_STEPS = 10   # Faster (0.33 sec)
AUTO_GRASP_STEPS = 20   # Normal (0.66 sec) ← current
AUTO_GRASP_STEPS = 30   # Slower (1.0 sec)
```

---

## Advantages & Disadvantages

### Advantages (Auto-Grasp Enabled)

✅ **Faster learning:** 600K-800K vs 1M-1.5M steps (25-40% faster)
✅ **Simpler problem:** Navigation only vs full manipulation
✅ **UR5-comparable:** Similar to UR5's approach
✅ **Lower compute cost:** Less GPU time needed
✅ **Easier to debug:** Fewer failure modes

### Disadvantages (Auto-Grasp Enabled)

❌ **Less learned:** Agent doesn't discover grasping strategy
❌ **Not real manipulation:** Grasping is scripted, not adaptive
❌ **Limited transfer:** Can't adapt to different objects/grips
❌ **Research value:** Doesn't test full RL capabilities
❌ **Deployment:** May not work on real robot (needs learned control)

---

## Recommended Settings

### For This Project (Stage 7 Training)

**Use AUTO_GRASP_ENABLED = True:**
- Goal is to replicate and compare to UR5
- Faster iteration and testing
- Validates that sparse rewards work
- More reasonable training time

**Later (Stage 8+):**
- Disable auto-grasp for full manipulation learning
- Add curriculum (Stage 6 → Stage 8)
- Test on real hardware

---

## Testing

### Test Auto-Grasp Trigger

```bash
# Quick test
python envs/stage_7_task.py

# Or detailed test:
python -c "
from envs.stage_7_task import Stage7Task
env = Stage7Task()
print(f'Auto-grasp enabled: {env.AUTO_GRASP_ENABLED}')
print(f'Trigger distance: {env.SUCCESS_DISTANCE_THRESHOLD}m')
"
```

### Verify Behavior

```bash
# Run evaluation and check for auto_grasp_triggered in logs
python scripts/evaluate.py \
    --model checkpoints/stage_7/stage_7_task_XXXXX_steps.zip \
    --env stage_7 \
    --n-episodes 10
```

---

## Summary

**Feature:** Automatic gripper closing when positioned (distance < 1cm)

**Purpose:** Replicate UR5's scripted grasping behavior

**Default:** Enabled (AUTO_GRASP_ENABLED = True)

**Expected benefit:** 25-40% faster training (800K vs 1.2M steps)

**Trade-off:** Agent learns navigation only, not full manipulation

**Recommendation:** Keep enabled for Stage 7 UR5 comparison study

---

## Version History

- **v1 (2026-01-30):** Stage 7 created - exact UR5 match attempt
- **v2 (2026-01-31):** Added weighted Z-axis distance calculation
- **v3 (2026-01-31):** Added automatic grasping feature (UR5 scripted behavior)

**Current version:** Stage 7 v3 (Z-axis + auto-grasp)
