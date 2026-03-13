# Multi-Stage Training Guide

## Quick Start

### Parallel Training (All at Once - FASTEST)

```bash
python scripts/train.py --stage 2a,2b,2c --timesteps 25000 --resume none --parallel
```

### Sequential Training (One After Another)

```bash
python scripts/train.py --stage 2a,2b,2c --timesteps 25000 --resume none
```

### Interactive Mode

```bash
python scripts/train.py
```

When prompted:
1. Enter: `2a, 2b, 2c`
2. Choose mode: `1` (Sequential) or `2` (Parallel)

## How It Works

### Interactive Mode Example

```
======================================================================
SELECT TRAINING STAGE
======================================================================
  1. Stage 1: Reach Out from Folded Position
     Train agent to extend from folded home position to target
     Default: 25,000 timesteps

  2a. Stage 2A: Touch Cube (Sparse Contact Only)
     Sparse reward for touching cube (distance < 3cm)
     Default: 25,000 timesteps

  2b. Stage 2B: Distance Shaping (Dense Reward)
     Dense distance-based reward + contact bonus
     Default: 25,000 timesteps

  2c. Stage 2C: Gripper Positioning (Touch + Gripper)
     Touch cube + open gripper to 90 degrees simultaneously
     Default: 25,000 timesteps

======================================================================
TIP: Enter multiple stages separated by commas to train sequentially
     Example: '2a, 2b, 2c' will train all three variants one after another

Select stage(s): 2a, 2b, 2c

======================================================================
SEQUENTIAL TRAINING PLAN: 3 stages
======================================================================
  1. Stage 2A: Touch Cube (Sparse Contact Only)
  2. Stage 2B: Distance Shaping (Dense Reward)
  3. Stage 2C: Gripper Positioning (Touch + Gripper)
======================================================================

[Training begins for each stage in sequence...]
```

### CLI Mode Examples

**Single stage:**
```bash
python scripts/train.py --stage 2a --timesteps 25000 --resume none
```

**Multiple stages (no spaces):**
```bash
python scripts/train.py --stage 2a,2b,2c --timesteps 25000 --resume none
```

**Multiple stages (with spaces - use quotes):**
```bash
python scripts/train.py --stage "2a, 2b, 2c" --timesteps 25000 --resume none
```

**Train all Stage 2 variants:**
```bash
python scripts/train.py --stage 2,2a,2b,2c --timesteps 25000 --resume none
```

## Training Behavior

### Sequential Training
- Trains each stage one after another
- Uses the same hyperparameters for all stages
- Each stage gets its own checkpoint directory
- Progress shown for each stage

### Checkpoint Handling
Each stage saves to its own directory:
```
checkpoints/
├── stage_2a/
│   └── stage_2a_25000_steps.zip
├── stage_2b/
│   └── stage_2b_25000_steps.zip
└── stage_2c/
    └── stage_2c_25000_steps.zip
```

### Resume Options
- `--resume none`: Train from scratch (recommended for new variants)
- `--resume auto`: Resume from latest checkpoint for each stage
- `--resume <path>`: Resume from specific checkpoint (only for single stage)

## Use Cases

### Compare Reward Strategies
Train all Stage 2 variants to see which learns best:
```bash
python scripts/train.py --stage 2a,2b,2c --timesteps 25000 --resume none
```

### Continue Training Multiple Stages
Add more training to all variants:
```bash
python scripts/train.py --stage 2a,2b,2c --timesteps 25000 --resume auto
```

### Train Full Pipeline
Train Stage 1 then all Stage 2 variants:
```bash
python scripts/train.py --stage 1,2a,2b,2c --timesteps 25000 --resume none
```

## Tips

1. **Use quotes with spaces:** `--stage "2a, 2b, 2c"` or no spaces: `--stage 2a,2b,2c`
2. **Monitor progress:** Each stage shows its own progress bar
3. **Stop anytime:** Ctrl+C stops current stage, saves checkpoint
4. **Review after:** Use `evaluate.py` to compare all variants

## After Training

Evaluate each variant:
```bash
python scripts/evaluate.py --checkpoint checkpoints/stage_2a/stage_2a_25000_steps.zip --episodes 100 --render
python scripts/evaluate.py --checkpoint checkpoints/stage_2b/stage_2b_25000_steps.zip --episodes 100 --render
python scripts/evaluate.py --checkpoint checkpoints/stage_2c/stage_2c_25000_steps.zip --episodes 100 --render
```

Compare success rates and select the best approach for Stage 3.
