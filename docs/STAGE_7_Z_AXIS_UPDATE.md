# Stage 7: Z-Axis Update

**Date:** 2026-01-31
**Change:** Added Z-axis (height) to distance calculation with 50% weighting

---

## Problem Identified

### Original UR5 Implementation (2D Distance)

```python
# Only X and Y
distance_to_target = np.linalg.norm(eef_pos[:2] - cube_pos[:2])
```

**Issue:** No reward signal for Z-axis (height) alignment
- ✅ Agent learns X, Y positioning (horizontal)
- ❌ Agent must randomly discover correct height
- ❌ No gradient telling agent "too high" or "too low"
- **Result:** Significant learning bottleneck

### Example Scenario

```
Agent state after 300K steps:
- X alignment: Perfect (0.001m error)
- Y alignment: Perfect (0.001m error)
- Z alignment: Wrong (0.10m too high)

Distance calculation (2D only): 0.0014m → reward = -0.014
Agent tries to grasp → No contact (too high) → reward = -0.014 (unchanged!)

Problem: No signal that Z is wrong!
```

---

## Solution: Weighted 3D Distance

### New Implementation

```python
# Calculate horizontal and vertical distances separately
xy_dist = np.linalg.norm(eef_pos[:2] - cube_pos[:2])  # X, Y
z_dist = abs(eef_pos[2] - cube_pos[2])                 # Z

# Combine with weighting
Z_WEIGHT = 0.5  # 50% importance on height vs horizontal
distance_to_target = np.sqrt(xy_dist**2 + (Z_WEIGHT * z_dist)**2)
```

### Why Weighted (Not Full 3D)?

**Full 3D (equal weight):**
```python
distance = np.linalg.norm(eef_pos - cube_pos)
# Z errors count as much as X,Y errors
```

**Problem with full 3D:**
- Gripper approaches from above (height difference expected)
- Don't want to penalize height difference equally
- Horizontal alignment more critical initially

**Weighted 3D (50% on Z):**
```python
distance = sqrt(xy² + (0.5 * z)²)
# Z errors count half as much as X,Y errors
```

**Benefits:**
- ✅ Still emphasizes horizontal alignment (primary)
- ✅ Provides feedback for height (secondary)
- ✅ Balances learning priorities

---

## Comparison Examples

### Example 1: Good XY, Wrong Z (Original Problem)

```
Positions:
- End-effector: [0.00, 0.30, 0.15]  (15cm high)
- Cube:         [0.00, 0.30, 0.02]  (2cm high)

Distances:
- XY distance: 0.00m (perfect!)
- Z distance:  0.13m (13cm too high)

Rewards:
- OLD (2D only):     -10 * 0.00 = 0.00    ← No signal!
- NEW (weighted 3D): -10 * 0.065 = -0.65  ← Height penalty!
- Full 3D:           -10 * 0.13 = -1.30   ← Too harsh
```

**Agent now learns:** "I'm close but something is still wrong (Z-axis)"

### Example 2: Approaching Horizontally

```
Positions:
- End-effector: [0.10, 0.30, 0.15]  (10cm left, 13cm high)
- Cube:         [0.00, 0.30, 0.02]  (on table)

Distances:
- XY distance: 0.10m
- Z distance:  0.13m

Rewards:
- OLD (2D only):     -10 * 0.10 = -1.00
- NEW (weighted 3D): -10 * 0.119 = -1.19  ← Slight increase
- Full 3D:           -10 * 0.164 = -1.64  ← Much harsher
```

**Agent learns:** "Move horizontally (main priority) and adjust height (secondary)"

### Example 3: Perfect Alignment

```
Positions:
- End-effector: [0.00, 0.30, 0.02]
- Cube:         [0.00, 0.30, 0.02]

Distances:
- XY distance: 0.00m
- Z distance:  0.00m

Rewards:
- OLD (2D only):     -10 * 0.00 = 0.00
- NEW (weighted 3D): -10 * 0.00 = 0.00  ← Same at goal!
- Full 3D:           -10 * 0.00 = 0.00
```

**All methods agree when perfectly aligned**

---

## Expected Impact on Learning

### Before (2D Only)

```
Phase 1 (0-300K): Learn X,Y alignment
Phase 2 (300K-500K): Random Z exploration ← BOTTLENECK
Phase 3 (500K-800K): Discover correct height
Phase 4 (800K-1M): Learn to grasp
```

**Total: ~1M steps to consistent success**

### After (Weighted 3D)

```
Phase 1 (0-200K): Learn X,Y,Z alignment simultaneously
Phase 2 (200K-400K): Refine positioning (all 3 axes)
Phase 3 (400K-600K): Discover grasp timing
Phase 4 (600K-800K): Consistent success
```

**Total: ~800K steps to consistent success** (20% faster!)

---

## Configuration Options

You can adjust Z-axis importance by changing `Z_WEIGHT`:

```python
# In stage_7_task.py class definition:

Z_WEIGHT = 0.0   # Ignore Z completely (original UR5)
Z_WEIGHT = 0.25  # Light Z penalty (25% vs XY)
Z_WEIGHT = 0.5   # Moderate Z penalty (50% vs XY) ← CURRENT
Z_WEIGHT = 0.75  # Strong Z penalty (75% vs XY)
Z_WEIGHT = 1.0   # Full 3D distance (equal weight)
```

**Recommendation:** Keep at 0.5 (current setting)

---

## Testing Results

**Test with random actions:**

```
Initial state:
- XY distance: 0.095m (9.5cm horizontally away)
- Z distance: 0.115m (11.5cm too high)

Reward comparison:
- OLD (2D only):     -0.95 pts
- NEW (weighted 3D): -1.11 pts
- Full 3D:           -1.49 pts

After 10 random steps:
- Rewards vary from -1.11 to -1.45
- Both XY and Z distances affect reward
- Agent gets feedback for all 3 axes
```

**✓ Verified:** Z-axis now influences reward appropriately

---

## Training Recommendations

### Option 1: Continue from Current Checkpoint (If XY Aligned)

If your agent already learned good X,Y positioning:

```bash
# Resume training with Z-axis feedback added
python scripts/train.py \
    --stage 7 \
    --resume checkpoints/stage_7/stage_7_task_XXXXX_steps.zip \
    --timesteps 500000
```

**Expected:** Agent quickly learns Z positioning (already knows XY)

### Option 2: Start Fresh (Recommended)

For cleanest learning progression:

```bash
# Train from scratch with full 3D feedback
python scripts/train.py \
    --stage 7 \
    --timesteps 1000000
```

**Expected:** Learns X,Y,Z simultaneously from the start

---

## Monitoring Progress

### Key Metrics to Watch

**Distance components (new metrics to track):**
```python
# In evaluation, track separately:
xy_dist = np.linalg.norm(eef_pos[:2] - cube_pos[:2])
z_dist = abs(eef_pos[2] - cube_pos[2])

# Should both decrease over training:
# - XY: 0.10m → 0.05m → 0.02m → 0.01m
# - Z:  0.12m → 0.08m → 0.04m → 0.02m
```

**Check if Z was the bottleneck:**

If agent now succeeds much faster (300K-500K steps vs 800K-1M), Z was the limiting factor!

---

## Technical Details

### Distance Formula Derivation

**Weighted Euclidean distance:**

```
distance = √(Δx² + Δy² + w²·Δz²)

where:
- Δx = eef_x - cube_x
- Δy = eef_y - cube_y
- Δz = eef_z - cube_z
- w = Z_WEIGHT = 0.5
```

**Example calculation:**

```python
eef_pos = [0.008, 0.343, 0.130]
cube_pos = [0.065, 0.267, 0.015]

dx = 0.008 - 0.065 = -0.057
dy = 0.343 - 0.267 = 0.076
dz = 0.130 - 0.015 = 0.115

xy_dist = sqrt(dx² + dy²) = sqrt(0.003249 + 0.005776) = 0.0953
z_dist = |dz| = 0.1151

distance = sqrt(0.0953² + (0.5 * 0.1151)²)
         = sqrt(0.009082 + 0.003316)
         = sqrt(0.012398)
         = 0.1113m
```

### Code Location

**Modified file:** `/home/oeyd/SO101_Training/envs/stage_7_task.py`

**Modified method:** `_compute_reward()` (lines ~180-220)

**Key changes:**
1. Added `Z_WEIGHT = 0.5` class constant
2. Calculate `xy_dist` and `z_dist` separately
3. Combine: `distance_to_target = sqrt(xy_dist² + (Z_WEIGHT * z_dist)²)`

---

## Rollback Instructions

If you want to revert to pure 2D (original UR5):

```python
# In stage_7_task.py, _compute_reward():

# Change from:
xy_dist = np.linalg.norm(eef_pos[:2] - cube_pos[:2])
z_dist = abs(eef_pos[2] - cube_pos[2])
distance_to_target = np.sqrt(xy_dist**2 + (self.Z_WEIGHT * z_dist)**2)

# Back to:
distance_to_target = np.linalg.norm(eef_pos[:2] - cube_pos[:2])
```

---

## Summary

**Change:** Added Z-axis to distance with 50% weighting

**Reason:** Avoid random exploration bottleneck for height discovery

**Expected benefit:** 20-30% faster learning (800K vs 1M steps)

**Trade-off:** No longer exact UR5 match, but more practical for learning

**Next step:** Restart training and monitor for faster grasp discovery!

---

## Version History

- **v1 (2026-01-30):** Stage 7 created - exact UR5 match (2D distance)
- **v2 (2026-01-31):** Added weighted Z-axis (50%) to distance calculation

**Current version:** Stage 7 v2 (with Z-axis feedback)
