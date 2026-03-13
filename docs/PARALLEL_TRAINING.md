# Parallel Training Guide

Train multiple Stage 2 variants simultaneously for faster results!

## Quick Start

### Interactive Mode (Recommended)

```bash
python scripts/train.py
```

1. Enter stages: `2, 2a, 2b, 2c`
2. Choose mode: `2` (In Parallel)
3. Enter timesteps: `25000`
4. **Select checkpoint for each stage individually**
   - Stage 2: Select checkpoint or train from scratch
   - Stage 2a: Select checkpoint or train from scratch
   - Stage 2b: Select checkpoint or train from scratch
   - Stage 2c: Select checkpoint or train from scratch

### CLI Mode

**All stages from scratch:**
```bash
python scripts/train.py --stage 2,2a,2b,2c --timesteps 25000 --resume none --parallel
```

**All stages resume from latest checkpoint:**
```bash
python scripts/train.py --stage 2,2a,2b,2c --timesteps 25000 --resume auto --parallel
```

**Interactive checkpoint selection (no --resume flag):**
```bash
python scripts/train.py --stage 2,2a,2b,2c --timesteps 25000 --parallel
# Will prompt for checkpoint selection for each stage
```

## How It Works

### Parallel Mode
- Launches separate Python processes for each stage
- All stages train simultaneously
- Each stage gets its own log file
- Progress monitored in main terminal
- Ctrl+C stops all processes

### Process Monitoring
The main terminal shows:
- Status updates every 30 seconds
- Which stages are RUNNING or COMPLETED
- Final success/failure report

### Log Files
Each stage writes to its own log:
```
logs/
├── parallel_stage_2_20260125_103500.log
├── parallel_stage_2a_20260125_103501.log
├── parallel_stage_2b_20260125_103502.log
└── parallel_stage_2c_20260125_103503.log
```

Check individual logs for detailed progress:
```bash
tail -f logs/parallel_stage_2a_*.log
```

## Example Session

```
================================================================================
SO-101 ROBOTIC ARM TRAINING
================================================================================

======================================================================
SELECT TRAINING STAGE
======================================================================
  1. Stage 1: Reach Out from Folded Position
  2. Stage 2: Touch Cube (Original - Sparse)
  2a. Stage 2A: Touch Cube (Sparse Contact Only)
  2b. Stage 2B: Distance Shaping (Dense Reward)
  2c. Stage 2C: Gripper Positioning (Touch + Gripper)
======================================================================

Select stage(s): 2a, 2b, 2c

======================================================================
Train 3 stages:
  1. Sequentially (one after another)
  2. In Parallel (all at the same time)
======================================================================
Select mode (1 or 2): 2

======================================================================
PARALLEL TRAINING - TIMESTEPS
======================================================================
Timesteps for each stage [default: 25,000]: 25000

======================================================================
STAGE 2a - SELECT CHECKPOINT
======================================================================
  0. Train from scratch (new model)
======================================================================
Select checkpoint (0-0): 0

======================================================================
STAGE 2b - SELECT CHECKPOINT
======================================================================
  0. Train from scratch (new model)
======================================================================
Select checkpoint (0-0): 0

======================================================================
STAGE 2c - SELECT CHECKPOINT
======================================================================
  0. Train from scratch (new model)
======================================================================
Select checkpoint (0-0): 0

================================================================================
PARALLEL TRAINING: 3 stages simultaneously
================================================================================
  1. Stage 2A: Touch Cube (Sparse Contact Only) (from scratch)
  2. Stage 2B: Distance Shaping (Dense Reward) (from scratch)
  3. Stage 2C: Gripper Positioning (Touch + Gripper) (from scratch)
================================================================================

Launching 3 training processes...
Each process will run in the background.
Check individual log files for progress.

Press Ctrl+C to stop all training processes.

Starting Stage 2a: Stage 2A: Touch Cube (Sparse Contact Only)
  Log: logs/parallel_stage_2a_20260125_103500.log
Starting Stage 2b: Stage 2B: Distance Shaping (Dense Reward)
  Log: logs/parallel_stage_2b_20260125_103501.log
Starting Stage 2c: Stage 2C: Gripper Positioning (Touch + Gripper)
  Log: logs/parallel_stage_2c_20260125_103502.log

✓ All 3 training processes launched!

================================================================================
MONITORING (Ctrl+C to stop all)
================================================================================

[10:35:30] Status update:
  Stage 2a: RUNNING
  Stage 2b: RUNNING
  Stage 2c: RUNNING

[10:36:00] Status update:
  Stage 2a: RUNNING
  Stage 2b: RUNNING
  Stage 2c: RUNNING

... (continues until all complete)

================================================================================
ALL TRAINING COMPLETE
================================================================================
✓ Stage 2a: SUCCESS
✓ Stage 2b: SUCCESS
✓ Stage 2c: SUCCESS
```

## Stopping Training

Press **Ctrl+C** to gracefully stop all training processes:
- Saves checkpoints for each stage
- Terminates all processes cleanly
- Logs show where each stage stopped

## After Training

Checkpoints saved to:
```
checkpoints/
├── stage_2a/stage_2a_25000_steps.zip
├── stage_2b/stage_2b_25000_steps.zip
└── stage_2c/stage_2c_25000_steps.zip
```

Evaluate each:
```bash
python scripts/evaluate.py --checkpoint checkpoints/stage_2a/stage_2a_25000_steps.zip --episodes 100 --render
python scripts/evaluate.py --checkpoint checkpoints/stage_2b/stage_2b_25000_steps.zip --episodes 100 --render
python scripts/evaluate.py --checkpoint checkpoints/stage_2c/stage_2c_25000_steps.zip --episodes 100 --render
```

## Advantages

**Parallel Training:**
- ✓ All variants finish at same time (~25min for 25K steps)
- ✓ Compare results immediately after
- ✓ Better use of multi-core CPU
- ✗ More resource intensive

**Sequential Training:**
- ✓ Lower resource usage
- ✓ Can monitor each stage individually
- ✗ Takes 3-4× longer total time
- ✗ Wait for all to finish before comparing

## Tips

1. **Use parallel for experiments** - When testing multiple reward strategies
2. **Monitor logs** - Each stage has its own detailed log file
3. **Check resources** - Make sure you have enough CPU/memory for parallel training
4. **Stop anytime** - Ctrl+C saves progress for all stages

## Command Reference

```bash
# Parallel training (interactive)
python scripts/train.py
# Select multiple stages, choose "In Parallel"

# Parallel training (CLI)
python scripts/train.py --stage 2a,2b,2c --timesteps 25000 --resume none --parallel

# Sequential training (CLI)
python scripts/train.py --stage 2a,2b,2c --timesteps 25000 --resume none

# Check log for specific stage
tail -f logs/parallel_stage_2a_*.log
```
