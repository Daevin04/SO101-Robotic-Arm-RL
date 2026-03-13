# Fast Iteration Strategy: 25K Steps Per Stage

**Date:** 2026-01-25
**Philosophy:** Small, achievable stages with >75% success

---

## Strategy

**Each stage = 25K steps, >75% success to advance**

### Why 25K Steps?

1. **Fast feedback** - 30-45 minutes per stage
2. **Quick iteration** - Try different approaches same day
3. **Clear milestones** - Know quickly if stage works
4. **Reduced waste** - Don't overtrain if approach is wrong
5. **Motivation** - Frequent success milestones

---

## Stage Progression Plan

### Stage 1: Reach/Point (25K steps) ★☆☆☆☆
**Objective:** Get within 5cm of object, hold 10 steps
**Success:** >75% success rate
**Checkpoint:** `stage_1_task_25000_steps.zip`

### Stage 2: Close Approach (25K steps) ★★☆☆☆
**Objective:** Get within 2cm, center object between jaws
**Success:** >75% within 2cm, centered
**Checkpoint:** `stage_2_task_25000_steps.zip`
**Resume from:** Stage 1 checkpoint

### Stage 3: Grasp (25K steps) ★★★☆☆
**Objective:** Close gripper on object with contact
**Success:** >75% grasp success
**Checkpoint:** `stage_3_task_25000_steps.zip`
**Resume from:** Stage 2 checkpoint

### Stage 4: Lift (25K steps) ★★★★☆
**Objective:** Lift object 3cm above table
**Success:** >75% lift success
**Checkpoint:** `stage_4_task_25000_steps.zip`
**Resume from:** Stage 3 checkpoint

### Stage 5: Hold & Place (25K steps) ★★★★★
**Objective:** Hold for 5 steps, place at goal
**Success:** >75% full task success
**Checkpoint:** `stage_5_task_25000_steps.zip`
**Resume from:** Stage 4 checkpoint

---

## Total Training Time

```
Stage 1: 25K steps (~45 min)
Stage 2: 25K steps (~45 min)
Stage 3: 25K steps (~45 min)
Stage 4: 25K steps (~45 min)
Stage 5: 25K steps (~45 min)
━━━━━━━━━━━━━━━━━━━━━━━━━━
Total:   125K steps (~3.5 hours)
```

**vs traditional approach:** 500K+ steps (~10+ hours) with likely failures

---

## Success Criteria (>75%)

**Why 75% and not lower?**
- High enough to confirm learning
- Low enough to be achievable at 25K steps
- Clear signal to move forward
- Prevents overtraining on simple tasks

**If <75% at 25K:**
- Train 25K more (up to 50K max)
- If still <75%, revise reward structure
- Don't waste time on failing approach

---

## Training Commands

```bash
# Stage 1: Start from scratch
python scripts/train.py --total-timesteps 25000

# Evaluate at 25K
python scripts/evaluate.py \
  --model checkpoints/stage_1/stage_1_task_25000_steps.zip \
  --env stage_1 \
  --n-episodes 100

# If >75% success: Create Stage 2
cp envs/stage_1_task.py envs/stage_2_task.py
# (update rewards, class name, etc.)

# Stage 2: Resume from Stage 1
python scripts/train.py \
  --total-timesteps 25000 \
  --resume checkpoints/stage_1/stage_1_task_25000_steps.zip
```

---

## Checkpoint Naming Convention

```
checkpoints/stage_1/stage_1_task_25000_steps.zip   ✅ SUCCESSFUL (78%)
checkpoints/stage_2/stage_2_task_25000_steps.zip   ✅ SUCCESSFUL (81%)
checkpoints/stage_3/stage_3_task_25000_steps.zip   🔄 IN PROGRESS
```

---

## Stage Transition Checklist

**Before advancing from Stage N to Stage N+1:**

1. ✅ Evaluate Stage N checkpoint (100 episodes)
2. ✅ Confirm >75% success rate
3. ✅ Update Stage N header: `STATUS: ✅ SUCCESSFUL`
4. ✅ Record final success rate and metrics
5. ✅ Copy Stage N as Stage N+1 template
6. ✅ Update Stage N+1 objective and rewards
7. ✅ Start Stage N+1 training from Stage N checkpoint

---

## Example: Stage 1 → Stage 2 Transition

### 1. Evaluate Stage 1
```bash
python scripts/evaluate.py \
  --model checkpoints/stage_1/stage_1_task_25000_steps.zip \
  --env stage_1 \
  --n-episodes 100

# Output: 78% success, avg distance 4.2cm ✅
```

### 2. Mark Stage 1 Successful
```python
# In envs/stage_1_task.py header:
STATUS: ✅ SUCCESSFUL
Date Completed: 2026-01-25
Best Checkpoint: stage_1_task_25000_steps.zip
Success Rate: 78%
Average Distance: 4.2cm
```

### 3. Create Stage 2
```bash
cp envs/stage_1_task.py envs/stage_2_task.py
```

Edit `stage_2_task.py`:
- Change "STAGE 1" → "STAGE 2"
- Reset status to 🔄 IN PROGRESS
- Update objective: "Get within 2cm + center between jaws"
- Add centering reward: `-2.0 × lateral_offset`
- Update success: `distance < 0.02 and centered`

### 4. Train Stage 2
```bash
python scripts/train.py \
  --total-timesteps 25000 \
  --resume checkpoints/stage_1/stage_1_task_25000_steps.zip
```

---

## Reward Complexity by Stage

### Stage 1: Reaching (3 reward terms)
```python
-10.0 × distance
-0.5 × height_diff
-0.01 × action²
```

### Stage 2: Close Approach (5 terms, +2 from Stage 1)
```python
# Stage 1 rewards +
-2.0 × lateral_offset    # NEW: centering
+10.0 if < 2cm           # NEW: closer threshold
```

### Stage 3: Grasping (7 terms, +2 from Stage 2)
```python
# Stage 2 rewards +
+0.5 gripper open        # NEW: prep for grasp
+30.0 grasp success      # NEW: actual grasp
```

### Stage 4: Lifting (9 terms, +2 from Stage 3)
```python
# Stage 3 rewards +
+50.0 lift 3cm           # NEW: lift
+100.0 hold 5 steps      # NEW: stable hold
```

**Progressive complexity!** Each stage adds 1-2 new reward terms.

---

## Benefits of 25K Fast Iteration

| Aspect | Traditional (500K) | Fast Iteration (5×25K) |
|--------|-------------------|------------------------|
| Time to first result | 10+ hours | 45 minutes |
| Failure detection | Late (after 200K+) | Early (after 25K) |
| Pivot speed | Slow (restart from 0) | Fast (just redo 1 stage) |
| Success visibility | Unclear | Clear milestones |
| Motivation | Low (long wait) | High (frequent wins) |
| Total time | 10-20+ hours | 3-5 hours |

---

## Decision Tree at 25K Checkpoint

```
Evaluate at 25K steps
    ↓
Success rate?
    ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓                  ↓
  >75%               <75%
    ↓                  ↓
✅ SUCCESS!        Continue training?
    ↓                  ↓
Mark stage      ━━━━━━━━━━━━━━━━━━━
successful           ↓              ↓
    ↓            Try 25K more   Revise rewards
Create next         ↓              ↓
stage            Evaluate at 50K  Restart stage
    ↓                  ↓
Resume from      >75%?  <75%?
this checkpoint      ↓      ↓
                  ✅    Pivot to new
                      approach
```

---

## Monitoring Progress

```bash
# Check success rate during training
tail -f logs/stage_1_training.log | grep success_rate

# At 25K, evaluate
python scripts/evaluate.py \
  --model checkpoints/stage_1/stage_1_task_25000_steps.zip \
  --env stage_1 \
  --n-episodes 100

# Record result in stage header
```

---

## Files to Update Per Stage

**When creating new stage:**
1. ✅ Copy `envs/stage_N_task.py` → `envs/stage_N+1_task.py`
2. ✅ Update class name: `StageNTask` → `StageN1Task`
3. ✅ Update header: Reset status, change objective
4. ✅ Update `__init__.py`: Add new import
5. ✅ Create training script or update existing
6. ✅ Update checkpoint directory in training script

**When marking stage successful:**
1. ✅ Update header in `envs/stage_N_task.py`
2. ✅ Add final success rate and metrics
3. ✅ Document any issues/learnings

---

## Example Timeline (Same Day)

```
9:00 AM  - Start Stage 1 training (25K)
9:45 AM  - Stage 1 complete, evaluate (78% ✅)
10:00 AM - Create Stage 2, start training
10:45 AM - Stage 2 complete, evaluate (82% ✅)
11:00 AM - Create Stage 3, start training
11:45 AM - Stage 3 complete, evaluate (71% ❌)
12:00 PM - Revise Stage 3 rewards
12:15 PM - Restart Stage 3 training
1:00 PM  - Stage 3 complete, evaluate (79% ✅)
1:15 PM  - Create Stage 4, start training
2:00 PM  - Stage 4 complete, evaluate (76% ✅)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Result: 4 stages completed in 5 hours
```

**vs traditional:** Would still be training Stage 1 at 200K steps, no results yet.

---

## Summary

**Strategy:** 25K steps per stage, >75% success to advance

**Benefits:**
- ✅ Fast iteration (45 min per stage)
- ✅ Early failure detection
- ✅ Clear progress milestones
- ✅ Lower risk of wasted training time
- ✅ Higher motivation (frequent wins)

**Result:** Full task in ~125K steps (3-5 hours) vs 500K+ (10-20 hours)

---

**Last Updated:** 2026-01-25
