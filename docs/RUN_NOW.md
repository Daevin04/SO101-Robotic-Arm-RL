# 🚀 RUN TRAINING NOW (You're already on VCU!)

You're on: `egr-s-26-604-1.rams.adp.vcu.edu` with **NVIDIA L4 GPU**
All dependencies: ✅ Installed

---

## Quick Start (Choose One)

### Option 1: Dense Rewards Training (RECOMMENDED - 2-4 hours)

```bash
./run_dense_training.sh
```

**What this does:**
- Trains with dense distance-based rewards
- Uses 2026 best practices (LR=1e-4, Buffer=500k)
- Expected success rate: 40-60% @ 50k steps
- Time: ~2-4 hours on your L4 GPU

### Option 2: Full Hyperparameter Sweep (12-24 hours)

```bash
# Run in background
nohup ./run_hyperparam_sweep_direct.sh > sweep.log 2>&1 &

# Check progress
tail -f sweep.log
```

**What this does:**
- Tests 12 different configurations
- Compares sparse vs dense rewards
- Finds optimal hyperparameters
- Time: ~12-24 hours total

---

## Monitoring Training

### While training runs:

```bash
# Watch logs in real-time
tail -f logs/dense_training.log

# Check GPU usage
watch -n 1 nvidia-smi

# Kill if needed
pkill -f train_stage_2c_dense.py
```

### After training completes:

```bash
# Evaluate performance
python scripts/evaluate.py --checkpoint checkpoints/dense_training/final.zip

# Watch trained policy
python scripts/watch.py --checkpoint checkpoints/dense_training/final.zip

# Analyze sweep results (if you ran sweep)
python cluster/analyze_sweep.py
```

---

## Why Dense Rewards?

Your current sparse rewards give **0 reward** until the arm touches the cube (within 3cm).

**Problem:** In 6-DOF space, this could take thousands of random steps to discover!

**Solution:** Dense rewards provide gradient at ANY distance:

```
Distance | Sparse | Dense | Total
---------|--------|-------|------
50cm     | 0      | 4.4   | 4.4   ← Agent knows it's far
30cm     | 0      | 8.2   | 8.2   ← Getting closer!
10cm     | 0      | 12.9  | 12.9  ← Almost there!
2cm      | 60     | 13.9  | 73.9  ← Contact! Big reward!
```

The dense reward **guides exploration**, while sparse rewards **signal success**.

---

## Expected Results

| Reward Type | Success @ 50k | Training Time |
|-------------|---------------|---------------|
| Sparse (old) | 0-10% | ~2 hours |
| Dense (new) | 40-60% | ~2-4 hours |

---

## Your Current Environment

- **GPU:** NVIDIA L4 (23GB VRAM) ✅
- **Python:** 3.13.5 ✅
- **Libraries:** stable-baselines3, mujoco, gymnasium ✅
- **Rendering:** Set to `MUJOCO_GL=egl` (headless) ✅

---

## What Happens Next?

### After ~2 hours (dense training):

1. **Check success rate** in final logs
2. **If >40% success:** Great! Use this model
3. **If <40% success:** Try sweep to find better hyperparameters

### After ~24 hours (full sweep):

1. Run: `python cluster/analyze_sweep.py`
2. Find best configuration
3. Use that config for longer training

---

## Quick Commands

```bash
# START dense training
./run_dense_training.sh

# START sweep (in background)
nohup ./run_hyperparam_sweep_direct.sh > sweep.log 2>&1 &

# MONITOR progress
tail -f logs/dense_training.log
tail -f sweep.log

# CHECK GPU
nvidia-smi

# ANALYZE results
python cluster/analyze_sweep.py

# EVALUATE model
python scripts/evaluate.py --checkpoint checkpoints/dense_training/final.zip

# WATCH policy
python scripts/watch.py --checkpoint checkpoints/dense_training/final.zip
```

---

## Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size in the script
# Edit: run_dense_training.sh
# Change: --batch-size 128  # Instead of 256
```

### "Process killed"
```bash
# Check if another process is using GPU
nvidia-smi
pkill -f train_stage
```

### Want to stop training?
```bash
# Find process
ps aux | grep train_stage

# Kill it
pkill -f train_stage_2c_dense.py
```

---

## 💡 My Recommendation

**START NOW with dense training:**

```bash
./run_dense_training.sh
```

Then grab coffee for 2-4 hours.

If results are good (>40% success), you're done!
If not, run the full sweep overnight.

---

**Ready? Run:**
```bash
./run_dense_training.sh
```

Good luck! 🚀
