# Stage 7: Hover + Scripted Descent Approach

**Date:** 2026-01-31
**Strategy:** Agent learns hover positioning, script executes descent + grasp

---

## Overview

This approach splits the task into two clear phases:

**Phase 1 (Agent learns):** Navigate to X,Z position of cube + hover 5cm above
**Phase 2 (Script executes):** Rotate jaws + descend + grasp

---

## Why This Works Better

### Problem with Previous Approaches

**Full 3D positioning:** Agent must learn X, Y, Z simultaneously + grasp timing
- Hard to learn (6 DOF manipulation)
- Long training time (1M+ steps)
- Grasp timing discovery difficult

**2D positioning only:** Agent learns X, Z but Z height is random
- Random height discovery bottleneck
- Inconsistent approach angle
- Grasp reliability issues

### Hover + Descent Solution

**Agent task simplified:**
- Learn 2.5D navigation: X, Z position + fixed height (5cm above)
- Clear target: "Get above the cube"
- Simpler reward gradient

**Script handles complexity:**
- Orientation change (vertical → horizontal jaws)
- Precise descent
- Grasp execution

**Benefits:**
✅ **Easier learning:** 2.5D vs 6D problem
✅ **Faster training:** ~400K-600K steps (vs 1M+)
✅ **Reliable grasping:** Scripted descent ensures correct approach
✅ **Clear phases:** Positioning separate from grasping
✅ **Consistent behavior:** Same descent every time

---

## How It Works

### Phase 1: Agent Positioning (Learned)

**Goal:** Reach hover position
```
Target position:
  X: cube_X  (horizontal left/right)
  Y: cube_Y  (horizontal forward/back)
  Z: cube_Z + 0.05m  (5cm above cube)
```

**Reward:**
```python
distance_to_hover = sqrt(xy_error² + z_error²)
reward = -10 * distance_to_hover

# Gets closer = less negative reward
# Within 1cm → triggers script
```

**What agent learns:**
- Navigate arm to position above cube
- Maintain 5cm hover height
- Align X,Z with cube center
- Does NOT learn: Grasping, descent, orientation

---

### Phase 2: Scripted Descent (Automatic)

**Trigger condition:** `distance_to_hover ≤ 0.01m` (within 1cm)

**Sequence:**

**Step 1-10: Rotate wrist (0° → 90°)**
```python
# Smoothly interpolate wrist_roll
for step in range(10):
    progress = step / 10.0
    wrist_roll = 0.0 + progress * 1.5708  # 0° → 90°

# Result: Jaws rotate from vertical to horizontal
```

**Step 11-30: Descend to cube**
```python
# Lower gripper straight down
descent_needed = eef_Z - cube_Z  # ~5cm
descent_per_step = descent_needed / 20

for step in range(20):
    eef_Z -= descent_per_step  # Gradual descent

# Maintain X,Y position throughout
# Result: Gripper at cube height with horizontal jaws
```

**Step 31-50: Close gripper and check**
```python
# Close gripper over 20 steps
for step in range(20):
    gripper_actuator = CLOSED

# Check forces
left_force, right_force = get_contact_forces()

if left_force > 3N and right_force > 3N:
    success = True
    reward = 100 + speed_bonus
else:
    success = False
    reward = -1.0
```

---

## Visual Sequence

### Initial State
```
Side view:

      ═══  ← Gripper (vertical jaws)
       |
       |

  ───────────  ← Table
     □    ← Cube
```

### Phase 1: Agent Navigates to Hover
```
Side view:

      ═══  ← Gripper moves here
       |     (5cm above cube)
       ↓
  ───────────
     □    ← Cube

Gripper:
- Positioned at X,Z of cube
- Height: 5cm above cube
- Jaws: VERTICAL (0°)
- Trigger condition met!
```

### Phase 2A: Script Rotates Jaws (10 steps)
```
Side view:
   0° → 30° → 60° → 90°
   ═══   ╱═╱   ──   |  |
    |                | |
                     ↓ ↓
  ───────────
     □

Jaws rotate from vertical to horizontal
```

### Phase 2B: Script Descends (20 steps)
```
Side view:
     |  |  ← Horizontal jaws
     ↓  ↓  Descending...

  ─|──|──
     □    ← At cube height

Gripper descends while maintaining X,Y
```

### Phase 2C: Script Closes Gripper (20 steps)
```
Side view:
   | → ← |  Jaws closing

  ─|─■─|──

Gripper squeezes cube from sides
Check forces → Success!
```

---

## Configuration

### Current Settings

```python
# In stage_7_task.py:

# Target height approach
USE_TARGET_HEIGHT = True
TARGET_HEIGHT_OFFSET = 0.05  # 5cm above cube

# Scripted descent
SCRIPTED_DESCENT_ENABLED = True
SCRIPTED_ROTATE_STEPS = 10   # Rotate jaws
SCRIPTED_LOWER_STEPS = 20    # Descend
AUTO_GRASP_STEPS = 20        # Close gripper

# Gripper control
FREEZE_GRIPPER_OPEN = True   # Keep open during approach
FREEZE_WHEN_POSITIONED = False  # Don't freeze (trigger script instead)
```

---

## Reward Structure

### Distance Calculation

**Target position:**
```python
hover_pos = [cube_X, cube_Y, cube_Z + 0.05]
```

**Distance:**
```python
xy_dist = sqrt((eef_X - cube_X)² + (eef_Y - cube_Y)²)
z_dist = abs(eef_Z - (cube_Z + 0.05))  # Distance to hover height

distance = sqrt(xy_dist² + (0.5 * z_dist)²)  # Weighted
```

**Reward:**
```python
if scripted_phase:
    reward = 0.0  # No reward during script execution
elif success:
    reward = 100 + (100 - step_count)  # Success + speed bonus
else:
    reward = -10 * distance  # Distance penalty
```

---

## Expected Learning Progression

### @ 100K Steps
- **Behavior:** Random exploration
- **Distance:** 0.10-0.30m from hover
- **Reward:** -1.0 to -3.0 average
- **Script triggers:** Never

### @ 200K-300K Steps
- **Behavior:** Approaching hover position
- **Distance:** 0.05-0.15m from hover
- **Reward:** -0.5 to -1.5 average
- **Script triggers:** Occasionally (first triggers!)

### @ 400K Steps
- **Behavior:** Consistent hover positioning
- **Distance:** <0.02m from hover
- **Reward:** -0.2 to +50 average
- **Script triggers:** 20-40% of episodes
- **Success rate:** 10-30%

### @ 600K Steps (Mastery)
- **Behavior:** Fast, efficient hover
- **Distance:** <0.01m consistently
- **Reward:** +50 to +120 average
- **Script triggers:** 60-80% of episodes
- **Success rate:** 50-70%

---

## Advantages

### Compared to Full Manipulation

| Aspect | Full Manipulation | Hover + Descent |
|--------|------------------|-----------------|
| **Agent learns** | 6 DOF manipulation | 2.5D positioning |
| **Problem difficulty** | Very hard | Moderate |
| **Training time** | 1M-1.5M steps | 400K-600K steps |
| **Grasp reliability** | Variable | Consistent |
| **Debugging** | Complex | Clear phases |

### Compared to 2D Only

| Aspect | 2D Only (UR5) | Hover + Descent |
|--------|--------------|-----------------|
| **Height control** | Random discovery | Guided (5cm target) |
| **Approach angle** | Variable | Consistent (from above) |
| **Grasp setup** | Hope for good Z | Guaranteed good Z |
| **Sim-to-real** | Hard | Easier |

---

## Disadvantages

### Limitations

❌ **Not full manipulation:** Agent doesn't learn complete task
❌ **Fixed strategy:** Always descend from above
❌ **Script dependency:** Relies on scripted descent quality
❌ **Limited adaptability:** Can't adjust to different object heights
❌ **Orientation locked:** Always rotates to horizontal

### When This Won't Work

⚠️ **Tall objects:** If object > 5cm, hover position might be inside object
⚠️ **Different shapes:** Script assumes cube-like geometry
⚠️ **Moving objects:** Script assumes object stays stationary
⚠️ **Tight spaces:** Needs room for vertical approach + rotation

---

## Testing

### Test Reward Calculation

```bash
python -c "
from envs.stage_7_task import Stage7Task
import numpy as np

env = Stage7Task()
obs, info = env.reset()

cube_pos = info['cube_pos']
target_hover = cube_pos + [0, 0, 0.05]

print(f'Cube at: {cube_pos}')
print(f'Target hover: {target_hover}')
print(f'Target height offset: {env.TARGET_HEIGHT_OFFSET}m')
"
```

### Test Scripted Sequence

```bash
# Demo with manual positioning
python scripts/demo_stage7_scripted_descent.py
```

### Test with Random Actions

```bash
# See if random actions ever trigger
python scripts/test_stage7_hover_descent.py
```

---

## Training Commands

### Start Training

```bash
python scripts/train.py --stage 7 --timesteps 600000
```

### Monitor Progress

```bash
# Watch tensorboard
tensorboard --logdir tensorboard_logs/

# Key metrics:
# - distance_to_target: Should decrease (0.15m → 0.01m)
# - reward: Should increase (-1.5 → +100)
# - Script triggers will show in console output
```

### Evaluate Checkpoint

```bash
python scripts/evaluate.py \
    --model checkpoints/stage_7/stage_7_task_400000_steps.zip \
    --env stage_7 \
    --n-episodes 20
```

---

## Customization

### Adjust Hover Height

```python
# In stage_7_task.py:
TARGET_HEIGHT_OFFSET = 0.03  # 3cm above (lower)
TARGET_HEIGHT_OFFSET = 0.05  # 5cm above (current)
TARGET_HEIGHT_OFFSET = 0.10  # 10cm above (higher)
```

**Trade-offs:**
- Lower (3cm): Faster descent, harder to reach, more collisions
- Higher (10cm): Easier to reach, longer descent, slower

### Adjust Descent Speed

```python
# In stage_7_task.py:
SCRIPTED_LOWER_STEPS = 10   # Fast descent (risky)
SCRIPTED_LOWER_STEPS = 20   # Normal (current)
SCRIPTED_LOWER_STEPS = 40   # Slow descent (safer)
```

### Disable Rotation (Keep Vertical)

```python
# In stage_7_task.py:
SCRIPTED_ROTATE_STEPS = 0   # No rotation, stay vertical
# Or modify _execute_scripted_descent() to skip rotation phase
```

### Change Success Threshold

```python
# In stage_7_task.py:
SUCCESS_DISTANCE_THRESHOLD = 0.005  # 5mm (tighter)
SUCCESS_DISTANCE_THRESHOLD = 0.01   # 1cm (current)
SUCCESS_DISTANCE_THRESHOLD = 0.02   # 2cm (looser)
```

---

## Troubleshooting

### Script Never Triggers

**Problem:** Agent not reaching hover position

**Check:**
- What's minimum distance achieved? (logs)
- Is hover height too high/low?
- Is reward gradient working?

**Solution:**
- Lower success threshold temporarily (0.02m)
- Adjust TARGET_HEIGHT_OFFSET
- Continue training (may need 400K+ steps)

### Script Triggers But Grasp Fails

**Problem:** Descent or grasp sequence not working

**Check:**
- Is rotation smooth?
- Does gripper reach cube height?
- Are contact forces detected?

**Solution:**
- Increase SCRIPTED_LOWER_STEPS (slower descent)
- Check cube/gripper collision geometry
- Print debug info in _execute_scripted_descent()

### Agent Learns Wrong Height

**Problem:** Agent finds local minimum at wrong Z

**Check:**
- Is Z_WEIGHT correct? (0.5)
- Is reward gradient pointing to hover?
- Print distance components (X, Y, Z separate)

**Solution:**
- Adjust Z_WEIGHT (try 0.75)
- Check reward calculation
- Verify target_pos calculation

---

## Summary

**Hover + Descent approach:**

1. **Agent learns:** Navigate to hover position (X, Z of cube, 5cm above)
2. **Script executes:** Rotate jaws + descend + grasp
3. **Benefits:** Simpler learning, faster training, reliable grasping
4. **Expected:** 400K-600K steps to 50%+ success rate

**Configuration:**
```python
USE_TARGET_HEIGHT = True
TARGET_HEIGHT_OFFSET = 0.05  # 5cm hover
SCRIPTED_DESCENT_ENABLED = True
FREEZE_GRIPPER_OPEN = True
```

**Next step:** Train and monitor for script triggers around 200K-400K steps!

---

## Quick Reference

```bash
# Test configuration
python -c "from envs.stage_7_task import Stage7Task; e = Stage7Task(); print(f'Hover offset: {e.TARGET_HEIGHT_OFFSET}m')"

# Demo scripted sequence
python scripts/demo_stage7_scripted_descent.py

# Start training
python scripts/train.py --stage 7 --timesteps 600000

# Monitor (watch for "[Script]" messages in console)
tensorboard --logdir tensorboard_logs/
```

**Success indicator:** Console prints "[Script] Triggering descent" when agent reaches hover position!
