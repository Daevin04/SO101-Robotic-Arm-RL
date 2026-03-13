# Stage 7: Freeze Mode for Position Verification

**Date:** 2026-01-31
**Purpose:** Visually verify positioning quality before grasping

---

## What It Does

When **FREEZE_WHEN_POSITIONED = True**, the robot **freezes in place** when the end-effector gets within 1cm of the cube.

### Behavior

```python
if distance_to_target <= 0.01m:
    # Set all joint velocities to zero
    robot.qvel[:] = 0.0

    # Robot holds current position
    # No gripper closing
    # Episode continues (allows inspection)
```

**Purpose:** Allows you to visually verify that the agent is positioning correctly before any grasping happens.

---

## Configuration

### Current Settings (in stage_7_task.py)

```python
AUTO_GRASP_ENABLED = False          # No automatic grasping
FREEZE_WHEN_POSITIONED = True       # Freeze when positioned ✓
```

### What Each Mode Does

| Mode | AUTO_GRASP | FREEZE | Behavior |
|------|-----------|--------|----------|
| **Freeze (Current)** | False | True | Robot freezes when positioned, holds position for inspection |
| **Auto-Grasp** | True | False | Gripper automatically closes when positioned |
| **Full Learning** | False | False | Agent must learn everything (positioning + grasping) |

---

## What You'll See Visually

### Successful Positioning Episode

**Sequence:**

1. **Approach Phase (Steps 1-50)**
   - Robot arm moves toward cube
   - Gripper stays fully open
   - Distance decreasing: 20cm → 10cm → 5cm

2. **Fine Positioning (Steps 50-70)**
   - Small adjustments
   - Gripper hovering near cube
   - Distance: 5cm → 2cm → 1cm

3. **🎯 FREEZE MOMENT (Step 71)**
   - Distance reaches < 1cm
   - **Robot suddenly stops moving**
   - All joints freeze in place
   - Gripper remains open
   - Cube remains on table

4. **Frozen State (Steps 71-100)**
   - Robot holds position
   - No more movement
   - Episode continues (or can terminate)
   - Allows visual inspection

### What the Freeze Looks Like

```
Before freeze (Step 70):
  [==]  ← Gripper moving
   ↓
  [ ]   ← Cube

At freeze (Step 71+):
  [==]  ← Gripper STOPS
   ||   ← Frozen!
  [ ]   ← Cube (Gripper hovering ~1cm away)
```

**Visual indicators:**
- ✓ Gripper stops moving instantly
- ✓ Gripper stays open (no closing)
- ✓ Cube centered between jaws (good positioning)
- ✓ Height aligned with cube
- ✓ Robot holds pose until episode ends

---

## Using the Visualization Script

### Run with Trained Checkpoint

```bash
python scripts/visualize_stage7_positioning.py \
    --checkpoint checkpoints/stage_7/stage_7_task_400000_steps.zip \
    --episodes 5
```

### What You'll See

**Terminal output:**
```
================================================================================
EPISODE 1/5
================================================================================
Initial distance: 0.1234m
Cube at: [0.050, 0.300, 0.015]
Starting episode...

  Step   0: distance=0.1234m, reward=  -1.23
  Step  20: distance=0.0876m, reward=  -0.88
  Step  40: distance=0.0432m, reward=  -0.43
  Step  60: distance=0.0156m, reward=  -0.16

********************************************************************************
🎯 ROBOT FROZEN AT STEP 68!
********************************************************************************
Positioning achieved:
  Distance: 0.0098m (threshold: 0.01m)
  EEF position: [0.051, 0.299, 0.016]
  Cube position: [0.050, 0.300, 0.015]
  Difference:
    X: 0.0010m
    Y: 0.0010m
    Z: 0.0010m

✓ Robot will hold this position for inspection
********************************************************************************
```

**Visual (if rendering enabled):**
- MuJoCo viewer window opens
- Robot arm frozen in position
- Gripper hovering at cube
- You can rotate camera to inspect alignment

---

## What to Check During Freeze

### 1. X-Axis Alignment (Left/Right)

**Good positioning:**
- Cube centered between gripper jaws (left-right)
- Equal spacing on both sides

**Bad positioning:**
- Cube offset to one side
- Would miss during grasp

### 2. Y-Axis Alignment (Forward/Back)

**Good positioning:**
- Cube between jaw tips (front-back)
- Jaws not too far forward/back

**Bad positioning:**
- Cube in front of/behind jaws
- Would miss during grasp

### 3. Z-Axis Alignment (Height)

**Good positioning:**
- Gripper at same height as cube center
- Jaws aligned horizontally with cube

**Bad positioning:**
- Gripper too high (above cube)
- Gripper too low (below cube)
- Would miss during grasp

### 4. Jaw Opening

**Good:**
- Jaws wide open
- Cube fits between jaws
- Space for closing

**Bad:**
- Gripper partially closed
- Not enough space for cube

---

## Visual Quality Checks

### ✅ Perfect Positioning

**What you see:**
```
Side view:          Top view:
  [==]               |     |
   ||                |  □  |  ← Cube centered
  [■]  ← Cube        |     |
```
- Cube exactly centered between jaws
- Height aligned
- Ready for grasp

### ⚠️ Good but Imperfect

**What you see:**
```
Side view:          Top view:
  [==]               |      |
   ||                |  □   |  ← Slightly off-center
  [■]                |      |
```
- Close but not perfect
- Might still grasp (if <1cm)
- Could improve with more training

### ❌ Bad Positioning

**What you see:**
```
Side view:          Top view:
  [==]               |      |
   ||                □      |  ← Way off
  [■]                |      |
```
- Cube not between jaws
- Would definitely miss
- Should NOT be frozen (distance > 1cm)
- Indicates bug or evaluation error

---

## Interpreting Results

### If Robot Freezes Often (>50% of episodes)

**Good sign:**
- Agent learned positioning successfully
- Ready to add grasping behavior
- Can enable AUTO_GRASP or train full control

**Next step:** Switch to auto-grasp or full learning mode

### If Robot Rarely Freezes (<10% of episodes)

**Need more training:**
- Agent still learning positioning
- Distance not consistently reaching < 1cm
- Keep training 100K-200K more steps

**Next step:** Continue training, monitor distance metric

### If Robot Freezes But Looks Misaligned

**Problem:**
- Distance metric says < 1cm
- But visual inspection shows bad alignment
- Might be weighted distance issue (XY good, Z bad)

**Next step:** Check Z-weight, verify distance calculation

---

## Transitioning from Freeze Mode

### Option 1: Enable Auto-Grasp (UR5-like)

```python
# In stage_7_task.py:
AUTO_GRASP_ENABLED = True
FREEZE_WHEN_POSITIONED = False
```

**Result:** Gripper closes automatically when positioned

### Option 2: Full Learning Mode

```python
# In stage_7_task.py:
AUTO_GRASP_ENABLED = False
FREEZE_WHEN_POSITIONED = False
```

**Result:** Agent must learn when/how to close gripper

### Option 3: Keep Freeze for Debugging

```python
# Keep current settings for ongoing inspection
AUTO_GRASP_ENABLED = False
FREEZE_WHEN_POSITIONED = True
```

**Result:** Continue visual verification during training

---

## Common Issues

### Robot Never Freezes

**Cause:** Distance never reaches < 1cm

**Check:**
- What's the minimum distance achieved? (in logs)
- Is agent still learning? (early training)
- Is distance threshold too tight? (try 0.02m)

**Solution:**
- Continue training
- Check tensorboard for distance metrics
- Verify positioning reward is working

### Robot Freezes but Gripper Empty

**Cause:** Distance < 1cm but cube not between jaws

**Check:**
- Visual alignment (cube should be centered)
- Z-axis alignment (height)
- Distance calculation (might be weighted wrong)

**Solution:**
- Adjust Z_WEIGHT if height is issue
- More training for better positioning
- Verify XYZ errors separately

### Freeze Happens Too Early

**Cause:** Distance threshold too loose or calculation wrong

**Check:**
- Print actual distance when frozen
- Verify XY and Z components separately
- Check if Z-weight is too low

**Solution:**
- Tighten threshold (0.01m → 0.008m)
- Adjust Z_WEIGHT
- Verify distance formula

---

## Recording for Analysis

### Save Video

```bash
# With rendering + recording
python scripts/visualize_stage7_positioning.py \
    --checkpoint checkpoints/stage_7/stage_7_task_400000_steps.zip \
    --episodes 10 \
    > positioning_log.txt
```

### Analyze Metrics

```bash
# Without rendering (just metrics)
python scripts/visualize_stage7_positioning.py \
    --checkpoint checkpoints/stage_7/stage_7_task_400000_steps.zip \
    --episodes 50 \
    --no-render \
    > positioning_metrics.txt
```

**Look for:**
- Freeze success rate
- Distance when frozen
- XYZ component breakdown
- Step count to freeze

---

## Summary

**FREEZE_WHEN_POSITIONED = True:**
- ✅ Robot stops when positioned (distance < 1cm)
- ✅ Allows visual inspection of positioning quality
- ✅ Verifies agent learned positioning correctly
- ✅ Useful for debugging and analysis

**When to use:**
- During training validation
- Before enabling auto-grasp
- When verifying positioning quality
- For visual debugging

**Next steps:**
1. Run visualization script
2. Verify positioning looks good
3. Switch to auto-grasp or full learning
4. Continue training with chosen mode

---

## Quick Reference

```bash
# Visualize positioning (current mode)
python scripts/visualize_stage7_positioning.py \
    --checkpoint checkpoints/stage_7/stage_7_task_XXXXX_steps.zip

# Then based on results:

# Option A: Positioning looks good → Enable auto-grasp
# Set AUTO_GRASP_ENABLED = True, FREEZE = False

# Option B: Positioning needs work → Keep training
# Keep current settings, train 100K-200K more

# Option C: Ready for full learning → Disable both
# Set both AUTO_GRASP and FREEZE to False
```
