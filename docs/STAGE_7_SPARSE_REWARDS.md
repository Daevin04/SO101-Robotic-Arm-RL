# Stage 7: UR5 Sparse Reward Implementation

**Status:** 🆕 NEW STAGE
**Date Started:** 2026-01-30
**Purpose:** Compare sparse (UR5) vs dense (Stage 6) reward approaches
**Training Target:** 2M+ steps

---

## Overview

Stage 7 implements the UR5 repository's exact sparse reward system to provide a direct comparison with Stage 6's dense reward approach. This is a **comparison study** to determine which reward structure is more effective for the SO-101 grasping task.

### Key Differences from Stage 6

| Aspect | Stage 6 (Dense) | Stage 7 (Sparse - UR5) |
|--------|----------------|----------------------|
| **Reward per step** | 0-1,130 points (continuous feedback) | -10 to +200 (mostly negative until success) |
| **Learning curve** | Gradual progression through phases | Exploration phase, then sudden breakthroughs |
| **Sample efficiency** | Faster learning (more guidance) | Slower learning (less guidance) |
| **Final behavior** | Smooth, controlled movements | Efficient, direct movements |
| **Training time** | 700K-1M steps for mastery | 1M-3M steps estimated |
| **Curriculum** | Position → Contact → Grasp → Lift | None (learns everything simultaneously) |
| **Grasp detection** | Collision-based (sim-to-real compatible) | Force-based (>3N threshold, simulation-only) |

---

## Reward Structure

### Success Reward
```python
if is_success:
    reward = 100.0 + (remaining_steps * 1.0)
```

**Examples:**
- Success at step 30: `+100 + (70 * 1.0) = +170 pts`
- Success at step 50: `+100 + (50 * 1.0) = +150 pts`
- Success at step 90: `+100 + (10 * 1.0) = +110 pts`

**Speed Bonus:** Encourages efficient completion. Earlier success = higher reward.

### Failure Penalty
```python
else:
    reward = -10.0 * distance_to_goal  # distance in meters
```

**Examples:**
- 5cm from goal: `-10 * 0.05 = -0.5 pts`
- 10cm from goal: `-10 * 0.10 = -1.0 pts`
- 20cm from goal: `-10 * 0.20 = -2.0 pts`

**Proportional Penalty:** Closer to goal = less negative reward. Provides weak gradient toward goal.

### Success Condition
```python
is_success = is_grasping and (distance_to_goal <= 0.01)
```

**Requirements:**
1. **Grasping:** Both fingers applying >3N force on cube
2. **At goal:** Cube within 1cm of goal position
3. **Goal position:** 10cm above starting position (lifted cube)

### Force-Based Grasping
```python
# Both fingers must exceed 3N force threshold
left_force = sum(contact_forces with left finger)
right_force = sum(contact_forces with right finger)

is_grasping = (left_force >= 3.0) and (right_force >= 3.0)
```

**Note:** This requires force sensors, which are NOT available on the real SO-101 robot. This is a **simulation-only** approach for comparison purposes.

### Early Termination
```python
if is_success:
    terminated = True  # End episode immediately
```

Episodes terminate early on success, preventing wasted steps and encouraging efficient behavior.

---

## Expected Learning Curve

### Phase 1: Exploration (0-500K steps)
- **Reward:** -100 to -50 pts average
- **Success rate:** <1%
- **Behavior:** Random exploration, discovering action space
- **Challenge:** Very sparse signal, mostly negative feedback

### Phase 2: Breakthrough (500K-1M steps)
- **Reward:** -50 to 0 pts average
- **Success rate:** 1-10%
- **Behavior:** First successful grasps and partial lifts
- **Challenge:** Inconsistent success, high variance

### Phase 3: Refinement (1M-2M steps)
- **Reward:** 0 to +50 pts average
- **Success rate:** 10-50%
- **Behavior:** More consistent grasping, improving efficiency
- **Challenge:** Optimizing speed, reducing wasted movements

### Phase 4: Mastery (2M+ steps)
- **Reward:** +50 to +100 pts average
- **Success rate:** >50%
- **Behavior:** Efficient, direct movements to goal
- **Achievement:** Comparable to Stage 6 performance

---

## Training Guide

### Starting Training

```bash
# Start Stage 7 training (2M steps recommended)
python scripts/train.py --stage 7 --timesteps 2000000

# Monitor with tensorboard
tensorboard --logdir logs/stage_7/
```

### Monitoring Progress

**Key metrics to watch:**
1. **Average reward:** Should increase from ~-100 to >0
2. **Success rate:** Should increase from 0% to >50%
3. **Distance to goal:** Should decrease over time
4. **Force magnitudes:** Should increase (learning to apply force)
5. **Episode length:** Should decrease on success (speed bonus)

### Checkpoints

Checkpoints saved every 25K steps:
```
checkpoints/stage_7/
├── stage_7_task_25000_steps.zip
├── stage_7_task_50000_steps.zip
├── ...
└── stage_7_task_2000000_steps.zip
```

### Evaluation

```bash
# Evaluate Stage 7 checkpoint
python scripts/evaluate_stage7.py \
    --checkpoint checkpoints/stage_7/stage_7_task_1000000_steps.zip \
    --episodes 100 \
    --render
```

---

## Comparison to Stage 6

### Sample Efficiency

**Stage 6 (Dense):**
- First success: ~100K-300K steps
- 50% success rate: ~700K steps
- 80% success rate: ~1M steps

**Stage 7 (Sparse) - Expected:**
- First success: ~300K-500K steps
- 50% success rate: ~1.5M steps
- 80% success rate: ~2M-3M steps

**Conclusion:** Dense rewards (Stage 6) expected to be ~2-3x more sample efficient.

### Final Behavior

**Stage 6 (Dense):**
- Smooth, controlled movements
- Gradual approach to cube
- Careful positioning before closing
- Slower but safer

**Stage 7 (Sparse) - Expected:**
- Direct, efficient movements
- Faster approach to cube
- More aggressive grasping
- Faster but potentially less robust

### Reward Comparison

**Stage 6 successful episode (100 steps):**
```
Phase 1 (0-20):   Approach        ~30-50/step   =     800 pts
Phase 2 (21-40):  Position       ~100-130/step  =   2,600 pts
Phase 3 (41-60):  Contact        ~200-330/step  =   6,600 pts
Phase 4 (61-80):  Grasp           ~830/step     =  16,600 pts
Phase 5 (81-100): Lift          ~1,130/step     =  22,600 pts
───────────────────────────────────────────────────────────────
Total:                                            49,200 pts
```

**Stage 7 successful episode (71 steps):**
```
Step 0-70:   Explore + approach    -10*dist    =   -50 pts
Step 71:     SUCCESS!              +100+29     =  +129 pts
───────────────────────────────────────────────────────────────
Total:                                              79 pts
```

**Key Insight:** Stage 6 provides much denser feedback (always positive), while Stage 7 is mostly negative until sudden success.

---

## Sim-to-Real Considerations

### Force Sensors (CRITICAL LIMITATION)

**Problem:** Stage 7 requires force sensors (>3N threshold) for grasp detection, which are **NOT available** on the real SO-101 robot.

**Impact:**
- Stage 7 is **SIMULATION-ONLY** for comparison purposes
- Cannot deploy Stage 7 policy to real robot without hardware modifications
- Stage 6 remains the canonical approach for real robot deployment

### Alternative Approach (Future Work)

To make Stage 7 sim-to-real compatible, replace force-based grasping:

```python
# Current (simulation-only):
is_grasping = (left_force > 3.0) and (right_force > 3.0)

# Alternative (sim-to-real compatible):
is_grasping = (
    left_contact and right_contact and  # Both jaws touching
    gripper_state < 0.3                  # Gripper mostly closed
)
```

**Trade-offs:**
- ✅ No force sensors required
- ✅ Sim-to-real compatible
- ❌ Less precise grasp detection
- ❌ May not match UR5 behavior exactly

---

## Implementation Details

### Environment Class

```python
class Stage7Task(SO101BaseEnv):
    """Stage 7: UR5 Sparse Reward Implementation"""

    # Constants
    SUCCESS_REWARD = 100.0
    SUCCESS_DISTANCE_THRESHOLD = 0.01  # 1cm
    FAILURE_PENALTY_SCALE = -10.0
    SPEED_BONUS_MULTIPLIER = 1.0
    FORCE_THRESHOLD = 3.0  # Newtons
    GOAL_LIFT_HEIGHT = 0.10  # 10cm
```

### Force Computation

```python
def _check_force_grasp(self):
    """Check if gripper is applying sufficient force on cube."""
    left_force = 0.0
    right_force = 0.0

    for i in range(self.data.ncon):
        contact = self.data.contact[i]
        # ... check if cube-finger contact ...

        # Compute contact force
        contact_force = np.zeros(6)
        mujoco.mj_contactForce(self.model, self.data, i, contact_force)
        force_magnitude = np.linalg.norm(contact_force[:3])

        # Accumulate forces
        if "left" in gripper_geom_name:
            left_force += force_magnitude
        if "right" in gripper_geom_name:
            right_force += force_magnitude

    # Both fingers must exceed threshold
    is_grasping = (left_force >= 3.0) and (right_force >= 3.0)
    return is_grasping, left_force, right_force
```

### Goal Position

```python
def reset(self, seed=None, options=None):
    # Randomize cube start position
    cube_start_pos = self.data.site_xpos[self.object_site_id].copy()

    # Set goal 10cm above start (lifted position)
    self.goal_pos = cube_start_pos.copy()
    self.goal_pos[2] += 0.10  # 10cm lift

    # Visualize goal marker
    self.data.qpos[self.goal_qpos_start:goal_qpos_start+3] = self.goal_pos
```

---

## Comparison Analysis Tools

### Compare Stage 6 vs Stage 7

```bash
# Generate comparison report
python scripts/compare_stage6_vs_stage7.py \
    --stage6-checkpoint checkpoints/stage_6/stage_6_task_1000000_steps.zip \
    --stage7-checkpoint checkpoints/stage_7/stage_7_task_2000000_steps.zip \
    --episodes 100 \
    --output comparison_report.md
```

**Metrics:**
- Success rate
- Average reward
- Episode length
- Sample efficiency (steps to first success)
- Final performance
- Behavior comparison (videos)

---

## Success Criteria

### Stage 7 @ 500K Steps
- ✅ First successful episode achieved
- ✅ Success rate: >5%
- ✅ Average reward: >-50 pts (was ~-100 early)
- ✅ Force-based grasp detected reliably

### Stage 7 @ 1M Steps
- ✅ Success rate: >40%
- ✅ Average reward on success: >+100 pts
- ✅ Efficient completion: <60 steps average
- ✅ Force profiles: Both jaws >3N consistently

### Stage 7 @ 2M Steps
- ✅ Success rate: >80%
- ✅ Average reward on success: >+150 pts
- ✅ Efficient completion: <40 steps average
- ✅ Comparable to Stage 6 final performance

---

## Troubleshooting

### No successful episodes by 500K steps

**Possible causes:**
1. Sparse reward too difficult (not enough gradient)
2. Force threshold too high (>3N unrealistic)
3. Goal position too difficult (10cm lift too far)

**Solutions:**
1. Add distance shaping (weak gradient toward goal)
2. Lower force threshold to 1-2N
3. Reduce goal lift to 5cm

### Reward stuck around -100

**Diagnosis:** Agent not learning to approach goal at all.

**Solutions:**
1. Check observation includes goal position
2. Verify goal position is visible (not hidden)
3. Add weak distance-based reward component
4. Increase training time (may need >500K for breakthrough)

### Force sensors not detecting grasp

**Diagnosis:** Contact forces < 3N even when visually grasping.

**Solutions:**
1. Print force magnitudes during episodes
2. Check MuJoCo contact force computation
3. Lower force threshold
4. Increase gripper stiffness in MJCF model

### Success but low speed bonus

**Diagnosis:** Taking full 100 steps even on success.

**Solutions:**
1. Verify early termination is working
2. Check that agent learns to terminate quickly
3. May need more training to optimize speed

---

## Future Work

### Hybrid Approach

Combine best of both worlds:
- Use Stage 6's dense curriculum for initial learning
- Switch to Stage 7's sparse rewards for final refinement
- Expected: Faster convergence + more efficient final behavior

### Multi-Object Extension

Extend Stage 7 to multiple objects:
- Sparse reward: +100 per object successfully placed
- Same force-based grasping
- Test generalization capabilities

### Real Robot Deployment

Create Stage 7B with collision-based grasping:
- Replace force threshold with collision + gripper state
- Test sim-to-real transfer
- Compare to Stage 6 real robot performance

---

## References

- **UR5 Repository:** [Original sparse reward implementation]
- **Stage 6:** Dense reward approach (baseline comparison)
- **MuJoCo Force Computation:** `mj_contactForce` documentation

---

## Conclusion

Stage 7 provides a clean comparison between dense (Stage 6) and sparse (Stage 7) reward approaches. Expected outcome:

- **Stage 6 (Dense):** Faster learning, more sample efficient, sim-to-real compatible
- **Stage 7 (Sparse):** Simpler reward engineering, potentially more efficient final behavior, simulation-only

Both approaches are valid, and the comparison will inform future curriculum design decisions.

**Recommendation:** Use Stage 6 for real robot deployment, Stage 7 for comparison studies and simulation-only tasks.
