# Stage 7: Vertical Jaw Orientation (Top/Bottom Grasp)

**Date:** 2026-01-31
**Change:** Rotated gripper 90° for vertical jaw orientation + frozen gripper

---

## What Changed

### 1. Jaw Orientation: Horizontal → Vertical

**Before (Horizontal):**
```
Top view:
  |     |  ← Jaws side-by-side
    □    ← Cube between jaws (left/right pinch)
  |     |
```

**After (Vertical):**
```
Side view:
    ═══  ← Upper jaw (above)
     □   ← Cube between jaws (top/bottom pinch)
    ═══  ← Lower jaw (below)
```

**Configuration change:**
```python
# In HOME_POSITION:
"wrist_roll": 0.0,     # Was 1.5708 (90°)
                       # Now 0° → vertical jaws
```

### 2. Gripper Frozen Open

**New feature:**
```python
FREEZE_GRIPPER_OPEN = True  # Force gripper to stay fully open
```

**Result:** Gripper cannot close during approach, stays at maximum opening (1.0)

---

## Why This Matters

### Grasp Strategy Difference

**Horizontal (old):**
- Pinch from left/right sides
- Requires precise lateral alignment
- Both jaws move horizontally to squeeze

**Vertical (new):**
- Pinch from top/bottom
- Upper jaw descends, lower jaw ascends
- More natural "grab from above" motion

### Advantages of Vertical Orientation

✅ **More intuitive:** Mimics human "reach down and grab" motion
✅ **Gravity assist:** Upper jaw presses down naturally
✅ **Better for flat objects:** Cube can rest on lower jaw
✅ **Simpler Z-axis:** Descend from above, no lateral precision needed
✅ **Stable base:** Lower jaw provides support

### Potential Challenges

⚠️ **Height precision:** Must align with cube height exactly
⚠️ **Cube thickness:** 3cm cube vs 8cm jaw separation (large gap)
⚠️ **Force distribution:** Top-down forces vs lateral squeeze
⚠️ **Contact area:** May contact only edges vs full sides

---

## Visual Comparison

### Initial Configuration

**Home position:**
```
Side view:              Front view:
   ═══  ← Upper jaw        ═══
    |                       |
    |                       |
   ═══  ← Lower jaw        ═══

Z separation: ~8cm
Cube height: ~3cm (fits in gap)
```

### During Approach

**Agent positioning:**
```
1. Arm moves toward cube
2. Gripper descends toward cube height
3. Upper jaw approaches from above
4. Lower jaw stays below cube
5. When distance < 1cm → FREEZE
```

### Expected Grasp (if enabled)

**Auto-grasp would:**
```
Before:                 After:
  ═══ ← Open              ═══ ← Closed
   □                       ■   ← Squeezed
  ═══ ← Open              ═══ ← Closed
```

---

## Configuration Summary

### Current Settings

```python
# Gripper orientation
HOME_POSITION["wrist_roll"] = 0.0      # 0° = vertical jaws

# Gripper control
FREEZE_GRIPPER_OPEN = True             # Keep gripper fully open
FREEZE_WHEN_POSITIONED = True          # Freeze robot when aligned
AUTO_GRASP_ENABLED = False             # No automatic closing

# Distance threshold
SUCCESS_DISTANCE_THRESHOLD = 0.01      # 1cm for freeze/success
```

### What Agent Learns

**With this configuration:**
- ✅ 3D positioning (X, Y, Z navigation)
- ✅ Aligning gripper over cube (top-down approach)
- ✅ Descending to correct height
- ❌ Gripper closing (frozen at max open)

**Agent cannot:**
- ❌ Close gripper during training
- ❌ Rotate wrist roll (locked at 0°)
- ❌ Learn gripper control strategies

---

## Testing Results

```
Configuration verified:
  ✓ Wrist roll: 0.0000 rad (0.0°)
  ✓ Gripper: 1.000 (fully open)
  ✓ Jaw separation: 0.082m (8.2cm vertical)
  ✓ Upper jaw above lower jaw
  ✓ Gripper stays open during random actions

Summary:
  - Jaws: VERTICAL ✓
  - Gripper: FROZEN OPEN ✓
  - Wrist roll: LOCKED at 0° ✓
```

---

## Visual Indicators During Training

### What Success Should Look Like

**Side view (freeze moment):**
```
  ═══  ← Upper jaw at cube top
   ■   ← Cube aligned
  ═══  ← Lower jaw at cube bottom

Distance < 1cm → FREEZE!
```

**Expected positioning:**
- Upper jaw ~1cm above cube top
- Lower jaw ~1cm below cube bottom
- Cube centered between jaws (front/back, left/right)
- Robot frozen in this pose

### What to Check

**1. Height alignment:**
- Upper jaw should be just above cube top
- Lower jaw should be just below cube bottom
- Cube should be "sandwiched" in gap

**2. Horizontal centering:**
- Cube centered front/back between jaws
- Cube centered left/right between jaws
- No offset to any side

**3. Jaw opening:**
- Gripper stays fully open (8cm gap)
- No closing during approach
- Maximum opening maintained

---

## Expected Learning Impact

### Easier Aspects

✅ **Simpler approach:** Descend from above (more natural)
✅ **No gripper distraction:** Agent can't close prematurely
✅ **Clear goal:** Get cube between vertical jaws
✅ **Gravity intuition:** Natural top-down motion

### Harder Aspects

⚠️ **Height precision:** Must align Z-axis exactly
⚠️ **Large gap:** 8cm jaw separation vs 3cm cube (2.5x)
⚠️ **Contact detection:** Forces may be different (vertical vs horizontal)
⚠️ **Stability:** Top-down forces vs lateral squeeze

### Expected Training Time

**Prediction:**
- Similar to horizontal approach (~600K-800K steps)
- May be slightly faster (more natural motion)
- Or slightly slower (height precision)
- Need to train and compare!

---

## Transitioning to Grasping

### Option 1: Enable Auto-Grasp

```python
FREEZE_GRIPPER_OPEN = False
AUTO_GRASP_ENABLED = True
```

**Result:** When positioned, gripper closes automatically (vertical squeeze)

### Option 2: Learn Full Control

```python
FREEZE_GRIPPER_OPEN = False
AUTO_GRASP_ENABLED = False
FREEZE_WHEN_POSITIONED = False
```

**Result:** Agent must learn when/how to close gripper (full manipulation)

### Option 3: Keep Current (Positioning Only)

```python
FREEZE_GRIPPER_OPEN = True
FREEZE_WHEN_POSITIONED = True
```

**Result:** Continue learning positioning, verify quality visually

---

## Comparison: Horizontal vs Vertical

| Aspect | Horizontal (Old) | Vertical (New) |
|--------|------------------|----------------|
| **Wrist roll** | 90° (1.5708 rad) | 0° (0.0 rad) |
| **Jaw orientation** | Side-by-side | One above, one below |
| **Grasp motion** | Lateral squeeze | Top-down pinch |
| **Natural motion** | Side approach | Descend from above |
| **Gravity** | No assist | Assists upper jaw |
| **Cube support** | No base | Lower jaw provides base |
| **Precision needed** | Lateral (X-axis) | Height (Z-axis) |
| **Contact area** | Side faces | Top/bottom faces |

---

## Visualization Commands

### Test Vertical Orientation

```bash
# Basic test
python -c "
from envs.stage_7_task import Stage7Task
env = Stage7Task()
obs, info = env.reset()
import numpy as np
print(f'Wrist roll: {np.degrees(env.data.qpos[4]):.1f}°')
print(f'Gripper: {env.data.qpos[5]:.2f} (1.0 = fully open)')
"
```

### Visualize with Checkpoint

```bash
python scripts/visualize_stage7_positioning.py \
    --checkpoint checkpoints/stage_7/stage_7_task_XXXXX_steps.zip \
    --episodes 5
```

**Look for:**
- Vertical jaw orientation (one above, one below)
- Gripper stays fully open
- Descending approach from above
- Freeze when cube is between jaws vertically

---

## Troubleshooting

### Gripper Still Closing

**Check:**
```python
# In stage_7_task.py:
FREEZE_GRIPPER_OPEN = True  # Should be True
```

**Verify:** Gripper control should stay at 1.0

### Jaws Not Vertical

**Check:**
```python
# In stage_7_task.py HOME_POSITION:
"wrist_roll": 0.0,  # Should be 0.0, not 1.5708
```

**Verify:** Z separation > X separation

### Freeze Not Working

**Check:**
```python
FREEZE_WHEN_POSITIONED = True  # Should be True
SUCCESS_DISTANCE_THRESHOLD = 0.01  # 1cm threshold
```

**Verify:** Robot stops when distance < 1cm

---

## Summary

**Two major changes:**

1. **Vertical jaws (wrist_roll = 0°)**
   - Jaws oriented top/bottom instead of left/right
   - More natural "grab from above" motion
   - Upper jaw descends, lower jaw provides base

2. **Frozen gripper (FREEZE_GRIPPER_OPEN = True)**
   - Gripper locked at maximum opening
   - Agent cannot close during training
   - Focuses learning on positioning only

**Current mode:** Positioning-only with vertical approach

**Next step:** Train and verify positioning quality, then enable grasping

---

## Quick Reference

```python
# Current configuration (stage_7_task.py)

HOME_POSITION = {
    "wrist_roll": 0.0,              # Vertical jaws ✓
}

FREEZE_GRIPPER_OPEN = True          # Gripper locked open ✓
FREEZE_WHEN_POSITIONED = True       # Freeze when aligned ✓
AUTO_GRASP_ENABLED = False          # No auto-grasp
```

**Agent learns:** Vertical positioning (top-down approach)
**Agent doesn't learn:** Gripper control (frozen)
**Visual result:** Robot descends from above, freezes with cube sandwiched between vertical jaws
