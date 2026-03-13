# Stage 7: Option A Implementation - Exact UR5 Match

**Date:** 2026-01-31
**Status:** ✅ IMPLEMENTED AND TESTED

---

## What Changed

Stage 7 has been updated to **exactly match** the UR5 repository's reward system.

### Critical Fix: Distance Calculation

**BEFORE (WRONG):**
```python
# Measured: Cube position vs Lifted goal (3D)
cube_pos = self.data.site_xpos[self.object_site_id]
goal_pos = self.goal_pos  # 10cm above cube start
distance = np.linalg.norm(cube_pos - goal_pos)

# Result: Always ~0.10m (cube never moves)
# Reward: Always -1.0 (no learning signal!)
```

**AFTER (CORRECT - UR5 EXACT):**
```python
# Measured: End-effector position vs Cube position (2D only)
eef_pos = self.data.site_xpos[self.ee_site_id]
cube_pos = self.data.site_xpos[self.object_site_id]
distance = np.linalg.norm(eef_pos[:2] - cube_pos[:2])  # X,Y only

# Result: Varies as gripper moves (0.03m to 0.30m)
# Reward: Varies (-0.3 to -3.0), providing learning gradient!
```

---

## Reward System Breakdown

### UR5 is HYBRID (Not Purely Sparse!)

| Component | Type | Formula | Purpose |
|-----------|------|---------|---------|
| **Distance penalty** | DENSE | `-10 × distance_to_target` | Guides exploration every step |
| **Success reward** | SPARSE | `+100` | Marks goal achievement |
| **Speed bonus** | SPARSE | `+remaining_steps` | Encourages efficiency |

### Reward Examples

**During exploration (no grasp):**
```
EEF 30cm from cube → reward = -3.0 pts/step
EEF 10cm from cube → reward = -1.0 pts/step
EEF 5cm from cube  → reward = -0.5 pts/step
EEF 1cm from cube  → reward = -0.1 pts/step
```

**On success:**
```
Success at step 30 → reward = 100 + (70) = +170 pts
Success at step 50 → reward = 100 + (50) = +150 pts
Success at step 80 → reward = 100 + (20) = +120 pts
```

---

## Task Definition

### What the Agent Must Learn

1. **Navigate:** Move end-effector to cube location (2D: X, Y)
2. **Approach:** Get within 1cm of cube horizontally
3. **Grasp:** Close gripper with >3N force on both fingers
4. **Optimize:** Complete task as quickly as possible (speed bonus)

### What is NOT Required

- ❌ Lifting the cube
- ❌ Placing the cube
- ❌ 3D alignment (height difference ignored)
- ❌ Reaching a lifted goal position

### Success Condition

```python
is_success = (distance_to_target <= 0.01) AND (is_grasping)
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^
#             Position: EEF within 1cm          Force: >3N both fingers
```

---

## Expected Learning Progression

### Phase 1: Random Exploration (0-100K steps)
- **Reward:** -100 to -150 pts/episode
- **Distance:** 0.10-0.30m (varying)
- **Behavior:** Random arm movements
- **What's happening:** Building replay buffer, exploring action space

### Phase 2: Approach Learning (100K-300K steps)
- **Reward:** -50 to -100 pts/episode (improving!)
- **Distance:** 0.05-0.15m (getting closer)
- **Behavior:** Gripper moves toward cube occasionally
- **What's happening:** Learning "closer to cube = less negative reward"

### Phase 3: Grasp Discovery (300K-500K steps)
- **Reward:** -10 to -50 pts/episode
- **Distance:** 0.01-0.05m (very close)
- **Behavior:** Gripper reaches cube, attempts to close
- **What's happening:** First successful grasps! Sudden +100 reward spikes

### Phase 4: Consistent Success (500K-1M steps)
- **Reward:** +50 to +120 pts/episode
- **Distance:** <0.01m consistently
- **Behavior:** Reliable, efficient grasping
- **Success rate:** 50-80%

---

## How to Restart Training

### Step 1: Stop Current Training
```bash
# Press Ctrl+C to stop current training process
```

### Step 2: Start Fresh Training (Recommended)
```bash
# Train from scratch with fixed reward system
python scripts/train.py --stage 7 --timesteps 1000000
```

**Why from scratch:**
- Old checkpoints learned on broken reward system (stuck at -1.0)
- Agent learned bad behaviors (not approaching cube)
- Fresh start will learn correct behaviors with proper gradient

### Step 3: Monitor Progress
```bash
# Watch tensorboard
tensorboard --logdir tensorboard_logs/

# Key metrics to watch:
# - rollout/ep_rew_mean: Should improve from -100 → 0 → +100
# - Distance to target: Should decrease over time
# - Success rate: Should increase after 300K steps
```

---

## Validation Test Results

**Test:** 100 random steps
**Results:**
```
Step 0:  reward = -0.54, distance = 0.0538m
Step 50: reward = -0.34, distance = 0.0335m (closer!)
Step 90: reward = -1.40, distance = 0.1403m (farther)
```

**✅ Verified:**
- Reward varies dynamically with end-effector position
- Provides learning gradient (-0.34 to -1.40)
- No longer stuck at constant -1.0

---

## Comparison to Stage 6

| Aspect | Stage 6 (Dense) | Stage 7 (Hybrid) |
|--------|----------------|------------------|
| **Reward system** | Fully dense (continuous feedback) | Hybrid (dense penalty + sparse success) |
| **Learning signal** | Very strong gradient | Moderate gradient |
| **Sample efficiency** | High (700K-1M steps) | Medium (500K-1M steps) |
| **Final behavior** | Smooth, controlled | Direct, efficient |
| **Grasp detection** | Collision-based | Force-based (>3N) |
| **Sim-to-real** | ✅ Compatible | ❌ Requires force sensors |

---

## Files Modified

1. **`/home/oeyd/SO101_Training/envs/stage_7_task.py`**
   - Changed distance calculation: 3D cube-to-goal → 2D end-effector-to-cube
   - Updated success condition: cube lifted → end-effector positioned + grasping
   - Removed goal position (cube itself is the target)
   - Updated reward calculation to use distance_to_target

2. **`/home/oeyd/SO101_Training/scripts/evaluate_stage7.py`**
   - Updated metric names: distance_to_goal → distance_to_target
   - Updated plot labels to reflect 2D distance measurement

3. **`/home/oeyd/SO101_Training/scripts/evaluate.py`**
   - Added Stage 7 to environment map
   - Added Stage 7 import and registration

4. **`/home/oeyd/SO101_Training/scripts/train.py`**
   - Increased early stopping patience for Stage 7 (100K episodes)
   - Sparse rewards need longer exploration phase

---

## Success Criteria

### @ 300K steps:
- ✅ First successful grasps appearing
- ✅ Success rate: >5%
- ✅ Mean reward: >-50 pts

### @ 500K steps:
- ✅ Consistent grasping behavior
- ✅ Success rate: >40%
- ✅ Mean reward: >0 pts

### @ 1M steps:
- ✅ Reliable, efficient grasping
- ✅ Success rate: >70%
- ✅ Mean reward: >+80 pts

---

## Troubleshooting

### If reward stays at -3.0 after 200K steps:
- **Problem:** Agent not learning to approach cube
- **Solution:** Check observation includes end-effector and cube positions
- **Check:** Run `python envs/stage_7_task.py` to verify observations

### If distance improves but no grasps:
- **Problem:** Agent approaches but doesn't close gripper
- **Solution:** This is normal up to ~300K steps, be patient
- **Check:** Verify forces are being detected (print contact forces)

### If early stopping triggered:
- **Problem:** Early stopping too aggressive for sparse rewards
- **Solution:** Already fixed in train.py (patience=100K for Stage 7)
- **Check:** Restart training from scratch

---

## Next Steps

1. **Start fresh training:**
   ```bash
   python scripts/train.py --stage 7 --timesteps 1000000
   ```

2. **Monitor progress every 100K steps:**
   ```bash
   python scripts/evaluate.py \
       --model checkpoints/stage_7/stage_7_task_100000_steps.zip \
       --env stage_7 \
       --n-episodes 20
   ```

3. **Compare to Stage 6 at 500K steps:**
   - Which achieves first success faster?
   - Which has better final performance?
   - Which behavior is more efficient?

---

## References

- **UR5 Repository:** https://github.com/leesweqq/ur5_reinforcement_learning_grasp_object
- **UR5 Environment File:** `ur5_env.py` (reward system in `step()` method)
- **Stage 6 Documentation:** `STAGE_6_GRASPING.md`

---

## Summary

Stage 7 now **exactly matches** the UR5 repository's hybrid reward approach:

✅ **Dense component:** Distance penalty provides learning gradient
✅ **Sparse component:** Success reward marks achievement
✅ **2D distance:** End-effector to cube (X, Y only)
✅ **Force-based grasp:** >3N threshold on both fingers
✅ **Early termination:** On success only
✅ **Speed bonus:** Encourages efficient behavior

The broken reward system (stuck at -1.0) has been fixed. Training can now proceed with proper learning signal!
