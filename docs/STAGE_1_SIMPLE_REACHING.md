# Stage 1: Simple Reaching Task

**Date Updated:** 2026-01-25
**Status:** 🔄 IN PROGRESS
**Complexity:** ★☆☆☆☆ (Simplest)

---

## Task Objective

**Simplest possible task:** Straighten arm and point towards object.

**Success:** Get end-effector within 5cm of object and maintain for 10 consecutive steps.

**No grasping, no lifting, no complex positioning - just pure reaching!**

---

## Why Start This Simple?

### Curriculum Learning Principle
```
Stage 1: Reach/point at object       ★☆☆☆☆
    ↓ (builds foundation)
Stage 2: Approach & position closely  ★★☆☆☆
    ↓ (adds precision)
Stage 3: Grasp & lift                ★★★☆☆
    ↓ (full task)
Stage 4: Multiple objects, sorting   ★★★★☆
```

**Starting simple:**
- ✅ Faster learning (50-100K steps vs 300K+)
- ✅ Builds solid foundation
- ✅ Prevents early frustration
- ✅ Each stage builds on previous success

---

## Reward Structure (SIMPLE!)

### Dense Rewards (Always Active)
```python
-10.0 × distance         # Main signal: minimize distance to object
-0.5 × height_diff       # Match object height
-0.01 × action²          # Smooth motion
```

### Sparse Rewards (Conditional)
```python
+5.0  if distance < 10cm   # Close approach bonus
+10.0 if distance < 5cm    # Very close bonus
+50.0 if close for 10 steps # Success!
```

**Total possible reward range:** -15 (far away) to +50 (success)

---

## What's Removed from Complex Version

**Removed (will add in Stage 2+):**
- ❌ Gripper open/close rewards
- ❌ Object centering between jaws
- ❌ Orientation matching
- ❌ Grasp detection
- ❌ Lift detection
- ❌ Contact rewards

**Why removed:** Not needed for simple reaching. Agent just needs to learn:
1. Move arm towards object location
2. Get close and stay there

---

## Expected Results

### Learning Curve (25K Target)
```
0-10K steps:   Random exploration, distance slowly decreasing
10-20K steps:  Agent discovers reaching behavior
20-25K steps:  Consistent reaching, close approaches

Target at 25K: >75% success rate
```

### Success Criteria to Move to Stage 2
- ✅ **>75% success rate** over 100 episodes
- ✅ **Average distance < 6cm** over 100 episodes
- ✅ Consistent reaching behavior (no exploits)

---

## Training Command

```bash
cd SO101-Robotic-Arm-RL

# Start training (25K steps = ~45 minutes)
python scripts/train.py --total-timesteps 25000

# Checkpoints saved to: checkpoints/stage_1/
# Naming: stage_1_task_25000_steps.zip
```

**Expected training time:** 30-45 minutes for 25K steps

---

## Evaluation

```bash
# Evaluate 25K checkpoint
python scripts/evaluate.py \
  --model checkpoints/stage_1/stage_1_task_25000_steps.zip \
  --env stage_1 \
  --n-episodes 100

# Record video
MUJOCO_GL=egl python scripts/evaluate.py \
  --model checkpoints/stage_1/stage_1_task_25000_steps.zip \
  --env stage_1 \
  --record \
  --video-dir videos/stage_1_25k \
  --n-episodes 5
```

**Target:** >75% success rate over 100 episodes

---

## Monitoring Training

```bash
# Watch training log
tail -f logs/stage_1_training.log

# Check progress
grep "success_rate" logs/stage_1_training.log | tail -10
```

**Look for:**
- `ee_to_obj_dist` decreasing over time
- `close_steps` increasing (agent staying close longer)
- `success_rate` climbing towards 70%+

---

## When to Mark Successful

After evaluating 25K checkpoint with >75% success, update header in `envs/stage_1_task.py`:

```python
"""
STAGE 1: Reach Towards Object (Pointing Task)

STATUS: ✅ SUCCESSFUL           # Change from 🔄 IN PROGRESS
Date Started: 2026-01-25
Date Completed: 2026-01-25      # Same day completion!
Best Checkpoint: stage_1_task_25000_steps.zip  # 25K checkpoint
Success Rate: 78%                # >75% achieved
Average Distance: 4.8cm          # <6cm achieved
Target: 25K steps, >75% success  # Keep this line
"""
```

**Then immediately create Stage 2!**

---

## Next Steps After Stage 1 Success

### Create Stage 2: Close Approach & Positioning

1. **Copy Stage 1 as template:**
   ```bash
   cp envs/stage_1_task.py envs/stage_2_task.py
   ```

2. **Update Stage 2 to add:**
   - Tighter distance threshold (< 2cm for success)
   - Gripper open reward (prepare for grasping)
   - Object centering between jaws
   - More precise positioning

3. **Train Stage 2 from Stage 1 checkpoint:**
   ```bash
   python scripts/train.py \
     --total-timesteps 200000 \
     --resume checkpoints/stage_1/stage_1_task_75k_steps.zip
   ```

---

## Comparison: Traditional vs Fast Iteration

| Aspect | Traditional (Direct) | Fast Iteration (Stage 1) |
|--------|---------------------|-------------------------|
| Task | Full grasp & lift | Just reach/point |
| Success | Lift 3cm for 5 steps | Within 5cm for 10 steps |
| Reward terms | 10 terms | 3 terms |
| Training steps | 500K+ | 25K |
| Training time | 8-10 hours | 30-45 minutes |
| Success rate goal | 10%+ | >75% |
| Difficulty | ★★★★★ | ★☆☆☆☆ |
| Feedback speed | Slow (days) | Fast (same hour) |

---

## Benefits of Fast Iteration Approach

**Speed:**
- Stage 1: 25K steps (~45 min) →  >75% success
- Stage 2: 25K steps (~45 min) → >75% success
- Stage 3: 25K steps (~45 min) → >75% success
- Stage 4: 25K steps (~45 min) → >75% success
- **Total: ~125K steps (~3 hours) to full task**

**vs traditional:** 500K+ steps (10+ hours) with likely <15% success

**Better Learning:**
- Each stage builds on previous success
- Clear milestones every 45 minutes
- Early failure detection (pivot after 25K, not 200K)
- Higher final success rate (75%+ at each stage)

**Motivation:**
- ✅ See progress within 1 hour
- ✅ Achieve stage success same day
- ✅ Complete full task in single session
- ✅ Clear "wins" to build on

**Industry Standard:**
- OpenAI Dactyl: 5 curriculum stages
- DeepMind: Progressive difficulty
- All modern RL: Fast iteration wins

---

## File Changes Made

1. ✅ `envs/stage_1_task.py` header - Updated to "Simple Reaching Task"
2. ✅ `_compute_reward()` - Simplified to 5 reward terms (was 10)
3. ✅ `step()` - Check for "close for 10 steps" instead of lifting
4. ✅ `reset()` - Initialize `_close_steps` counter
5. ✅ Success criteria - Within 5cm for 10 steps (not lift + hold)

---

**Status:** Ready to train! 🚀

**Expected outcome:** 70%+ success in 50-100K steps, then move to Stage 2.

**Last Updated:** 2026-01-25
