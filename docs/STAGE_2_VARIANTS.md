# Stage 2 Reward Strategy Comparison

## Overview

Four different reward strategies for Stage 2, each testing a different approach to training the robot to interact with randomly positioned cubes. Train all in parallel and compare results using `python evaluate.py` to see which works best.

## Variants

### Stage 2A: Simple Touch (Sparse Contact Only)
**Command:** `python scripts/train.py --stage 2a --timesteps 25000 --resume none`

**Reward Structure:**
- +100 when touching cube (distance < 3cm)
- 0 otherwise
- Must hold contact for 10 consecutive steps

**Why test this:**
- Simplest possible reward signal
- Pure sparse reward approach
- Tests if agent can learn from rare success signals
- Baseline for comparison

**Expected behavior:**
- Slow initial learning (sparse rewards difficult)
- May get stuck in local minima
- Once it finds cube, should learn quickly

---

### Stage 2B: Distance Shaping (Dense Reward)
**Command:** `python scripts/train.py --stage 2b --timesteps 25000 --resume none`

**Reward Structure:**
- Dense: -distance × 100 (continuous feedback)
- Bonus: +100 when touching (distance < 3cm)
- Must hold contact for 10 consecutive steps

**Why test this:**
- Dense rewards easier to learn
- Continuous guidance toward goal
- Standard reward shaping approach
- Should converge faster than sparse

**Expected behavior:**
- Faster initial learning
- Smooth progression toward target
- May be most reliable learner

---

### Stage 2C: Gripper Positioning (Touch + Gripper Control)
**Command:** `python scripts/train.py --stage 2c --timesteps 25000 --resume none`

**Reward Structure:**
- +60 when touching cube (distance < 3cm)
- +40 when gripper at 90° (normalized 0.5 ± 0.1)
- +50 bonus when both held for 10 steps

**Why test this:**
- Multi-objective learning (reach + gripper control)
- More realistic pre-grasp behavior
- Tests if agent can learn two skills simultaneously
- Direct prep for actual grasping

**Expected behavior:**
- More complex learning task
- May take longer to master
- Better prepares for Stage 3 (grasping)

---

### Stage 2 (Original): Touch Cube
**Command:** `python scripts/train.py --stage 2 --timesteps 25000 --resume none`

**Reward Structure:**
- Same as Stage 2A (sparse contact reward)

---

## Training Strategy

### Option 1: Sequential Training (One Command)
Train all variants one after another automatically:
```bash
# Interactive mode - select multiple stages
python scripts/train.py
# Then enter: 2a, 2b, 2c

# Or CLI mode
python scripts/train.py --stage 2a,2b,2c --timesteps 25000 --resume none
```

### Option 2: Parallel Training (Multiple Terminals)
Start all variants simultaneously for faster completion:
```bash
# Terminal 1
python scripts/train.py --stage 2a --timesteps 25000 --resume none

# Terminal 2
python scripts/train.py --stage 2b --timesteps 25000 --resume none

# Terminal 3
python scripts/train.py --stage 2c --timesteps 25000 --resume none
```

### 2. Monitor training
Each training run saves to:
- Checkpoints: `checkpoints/stage_2a/`, `checkpoints/stage_2b/`, `checkpoints/stage_2c/`
- Logs: `logs/stage_2a_training_*.jsonl`, etc.

### 3. Evaluate all variants
After training completes, evaluate each:
```bash
python scripts/evaluate.py --checkpoint checkpoints/stage_2a/stage_2a_25000_steps.zip --episodes 100 --render
python scripts/evaluate.py --checkpoint checkpoints/stage_2b/stage_2b_25000_steps.zip --episodes 100 --render
python scripts/evaluate.py --checkpoint checkpoints/stage_2c/stage_2c_25000_steps.zip --episodes 100 --render
```

### 4. Compare results
Look at:
- Success rate (% of episodes that touch cube)
- Average reward per episode
- Behavior quality (smooth vs jerky movements)
- Generalization (works for all cube positions?)

### 5. Select best approach
Choose the variant that:
- Has highest success rate
- Shows smooth, controlled movements
- Generalizes well to all positions
- Use this checkpoint for Stage 3

---

## Checkpoints Structure

```
checkpoints/
├── stage_1/
│   └── stage_1_task_25000_steps.zip
├── stage_2a/
│   └── stage_2a_25000_steps.zip
├── stage_2b/
│   └── stage_2b_25000_steps.zip
└── stage_2c/
    └── stage_2c_25000_steps.zip
```

---

## Evaluation Metrics to Compare

| Metric | 2A (Sparse) | 2B (Dense) | 2C (Gripper) |
|--------|-------------|------------|--------------|
| Success Rate | ?% | ?% | ?% |
| Avg Reward | ? | ? | ? |
| Learning Speed | Slow | Fast | Medium |
| Movement Quality | ? | ? | ? |
| Gripper Control | N/A | N/A | Yes |

Fill in after training and evaluation.

---

## Next Steps After Comparison

1. Identify winning strategy
2. Use best checkpoint as base for Stage 3
3. Document what worked and why
4. Apply lessons learned to future stages
