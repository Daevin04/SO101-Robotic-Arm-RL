# GPU Training Guide

This guide covers running training on a GPU-accelerated machine or remote cluster.

---

## Prerequisites

- NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- MuJoCo headless rendering configured (`MUJOCO_GL=egl` for Linux servers)
- Dependencies installed: `pip install -r requirements.txt`

---

## Recommended: Dense Reward Training

Dense rewards provide a gradient at every distance from the target, dramatically speeding up early training compared to sparse rewards.

```bash
python scripts/train.py --stage 2 --timesteps 50000 --reward-type dense
```

**Why dense rewards?**

Sparse rewards give zero signal until the arm is within 3 cm of the target. In a 6-DOF space, random exploration may take thousands of steps to discover this region.

Dense rewards signal progress at any distance:

```
Distance | Sparse | Dense | Total
---------|--------|-------|------
50 cm    | 0      | 4.4   | 4.4
30 cm    | 0      | 8.2   | 8.2
10 cm    | 0      | 12.9  | 12.9
2 cm     | 60     | 13.9  | 73.9   ← Contact reward
```

---

## Full Hyperparameter Sweep (Optional)

To compare reward structures and find optimal hyperparameters:

```bash
# Run in background on a remote server
nohup python scripts/train.py --stage 2 --timesteps 150000 > logs/sweep.log 2>&1 &

# Monitor progress
tail -f logs/sweep.log
```

---

## Expected Results

| Reward Type | Success @ 50k steps | Approx. Training Time (L4/A100) |
|-------------|---------------------|--------------------------------|
| Sparse      | 0–10%               | ~2 hours                       |
| Dense       | 40–60%              | ~2–4 hours                     |

---

## Monitoring

```bash
# Live log output
tail -f logs/training.log

# GPU utilization
watch -n 1 nvidia-smi

# TensorBoard
tensorboard --logdir logs/
```

---

## Headless Rendering (Linux Servers)

Set MuJoCo to use EGL for headless GPU rendering:

```bash
export MUJOCO_GL=egl
python scripts/train.py --stage 1 --timesteps 25000
```

---

## After Training

```bash
# Evaluate a saved checkpoint
python scripts/evaluate.py \
  --model checkpoints/stage_2/stage_2_task_50000_steps.zip \
  --env stage_2 \
  --n-episodes 100

# Visualize the trained policy
python scripts/watch.py \
  --model checkpoints/stage_2/stage_2_task_50000_steps.zip \
  --env stage_2
```
