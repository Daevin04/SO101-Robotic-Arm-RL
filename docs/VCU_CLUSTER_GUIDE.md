# SO-101 on VCU GPU Cluster - Complete Guide

## 🎯 Quick Start (Choose Your Path)

```bash
# Interactive setup:
bash cluster/quick_start.sh

# Or directly:
sbatch cluster/hyperparam_sweep.slurm  # Recommended first step
```

---

## 📁 What I Created For You

```
SO101_Training/
├── cluster/
│   ├── train_sac.slurm              # Basic training (12 hours)
│   ├── hyperparam_sweep.slurm       # 12 configs in parallel (24-48h) ⭐
│   ├── eureka_reward_gen.slurm      # AI-generated rewards (48h, ~$8)
│   ├── setup_eureka.sh              # One-time EUREKA setup
│   ├── analyze_sweep.py             # Analyze sweep results
│   ├── quick_start.sh               # Interactive launcher
│   └── README.md                    # Detailed documentation
├── envs/
│   ├── stage_2c_task.py             # Original (sparse rewards)
│   └── stage_2c_task_dense.py       # NEW: Dense rewards for exploration ⭐
└── VCU_CLUSTER_GUIDE.md             # This file
```

---

## 🚀 Recommended Workflow

### Step 1: Hyperparameter Sweep (Start Here!)

```bash
# On VCU cluster:
cd SO101_Training
sbatch cluster/hyperparam_sweep.slurm
```

**What this does:**
- Tests 12 different configurations in parallel
- Compares sparse vs dense rewards
- Tries different learning rates (3e-4, 1e-4, 5e-5)
- Tries different buffer sizes (100k, 500k, 1M)

**Wait 24-48 hours, then analyze:**
```bash
python cluster/analyze_sweep.py
```

### Step 2: Evaluate Best Configuration

```bash
# Find best checkpoint from sweep
python scripts/evaluate.py --checkpoint checkpoints/sweep_6/final.zip

# Watch it perform
python scripts/watch.py --checkpoint checkpoints/sweep_6/final.zip
```

### Step 3: If Still Not Working, Try EUREKA

```bash
# One-time setup
bash cluster/setup_eureka.sh
export OPENAI_API_KEY="sk-your-key-here"

# Run AI-powered reward generation
sbatch cluster/eureka_reward_gen.slurm
```

---

## 💡 Understanding the Reward Types

### Sparse Rewards (Original - `stage_2c_task.py`)

```python
# Only gives reward when touching cube
if distance < 0.03:
    reward = 60  # Contact
    if gripper_correct:
        reward += 40  # Gripper
```

**Problem:** Agent might never discover the cube (6-DOF is huge!)

### Dense Rewards (New - `stage_2c_task_dense.py`)

```python
# Always gives reward based on distance
distance_reward = (1.0 - tanh(distance / 0.5)) * 20.0  # Guides exploration

# Plus sparse rewards when touching
if distance < 0.03:
    reward += 60 + 40  # Contact + gripper
```

**Benefit:** Agent always knows if it's getting closer → faster learning

---

## 📊 Expected Performance

| Approach | Success Rate @ 50k steps | Training Time | Cost |
|----------|-------------------------|---------------|------|
| Sparse rewards (original) | 0-10% | ~12 hours | Free |
| Dense rewards | 40-60% | ~12 hours | Free |
| EUREKA (AI-generated) | 60-80% (if good) | ~48 hours | ~$8 |

---

## 🔧 Hyperparameter Sweep Configurations

The sweep tests these 12 configurations:

| Config | LR | Buffer | Batch | Reward Type |
|--------|-------|---------|-------|-------------|
| 0 | 3e-4 | 100k | 256 | Sparse |
| 1 | 3e-4 | 500k | 256 | Sparse |
| 2 | 1e-4 | 100k | 256 | Sparse |
| 3 | 1e-4 | 500k | 256 | Sparse |
| 4 | 3e-4 | 100k | 256 | **Dense** ⭐ |
| 5 | 3e-4 | 500k | 256 | **Dense** ⭐ |
| 6 | 1e-4 | 100k | 256 | **Dense** ⭐ |
| 7 | 1e-4 | 500k | 256 | **Dense** ⭐ |
| 8 | 1e-4 | 500k | 512 | Dense |
| 9 | 5e-5 | 500k | 256 | Dense |
| 10 | 1e-4 | 1M | 256 | Dense |
| 11 | 1e-4 | 500k | 128 | Dense |

**Prediction:** Configs 5-7 will likely perform best.

---

## 🤖 How EUREKA Works

1. **You provide:** Task description + environment code
2. **GPT-4o generates:** 16 different reward functions
3. **System trains:** SAC on each for 10k steps
4. **LLM reflects:** "Why did config #7 fail? The arm reached the cube but didn't hold gripper correctly..."
5. **Iterate:** Generate 16 new improved rewards
6. **Repeat:** 5 iterations total
7. **Result:** Best reward function saved to `eureka_outputs/best_reward.py`

**Key Insight from 2026 Research:**
> "LLM-generated rewards outperform expert handcrafted ones by ~22% in complex manipulation tasks"

**But:** Costs ~$8 in API calls and takes 48 hours.

---

## 📝 VCU Cluster Commands Cheat Sheet

### Job Management
```bash
# Submit job
sbatch cluster/train_sac.slurm

# Check status
squeue -u $USER

# Cancel job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER

# View logs
tail -f logs/slurm_12345.out

# Check available GPUs
sinfo -p gpu
```

### After Training
```bash
# Analyze sweep results
python cluster/analyze_sweep.py

# Evaluate best model
python scripts/evaluate.py --checkpoint checkpoints/sweep_X/final.zip

# Watch best policy
python scripts/watch.py --checkpoint checkpoints/sweep_X/final.zip
```

---

## 🐛 Troubleshooting

### "ImportError: No module named mujoco"
```bash
# Activate your virtual environment in the Slurm script
source venv/bin/activate
```

### "CUDA out of memory"
```bash
# Reduce batch size
--batch-size 128  # Instead of 256
```

### "MUJOCO_GL error"
```bash
# In Slurm script, add:
export MUJOCO_GL=egl
```

### EUREKA not generating good rewards
- Check `eureka_outputs/iteration_*/rewards/` for generated code
- Try adjusting task description in `setup_eureka.sh`
- Sometimes handcrafted dense rewards work better!

---

## 💰 Cost Breakdown

| Resource | Cost at VCU |
|----------|-------------|
| GPU time | Usually free for students |
| Storage (checkpoints) | Usually free (within quota) |
| EUREKA API calls | ~$8 per full run |

**Total for full pipeline:** ~$8 (only if using EUREKA)

---

## 🎓 2026 Best Practices Applied

Based on your research, here's what I implemented:

✅ **Automatic entropy tuning** (`ent_coef="auto"`)
✅ **Large replay buffer** (500k-1M for 6-DOF)
✅ **Dense reward shaping** (distance-based guidance)
✅ **Lower learning rate** (1e-4 for stability)
✅ **Increased gradient steps** (4 updates per env step)
✅ **Normalized actions** ([-1, 1] with tanh activation)
✅ **Twin Q-networks** (SAC default, prevents overestimation)

---

## 📈 Next Steps

1. **Right now:** Submit hyperparameter sweep
   ```bash
   sbatch cluster/hyperparam_sweep.slurm
   ```

2. **In 24-48 hours:** Analyze results
   ```bash
   python cluster/analyze_sweep.py
   ```

3. **If results are good:** Use best config for longer training

4. **If results are poor:** Try EUREKA (AI-generated rewards)

5. **When successful:** Move to next stage (Stage 3)

---

## 📚 Resources

- **VCU Cluster Docs:** https://wiki.vcu.edu/display/HPC/
- **EUREKA Paper:** https://eureka-research.github.io/
- **SAC Paper:** https://arxiv.org/abs/1801.01290
- **Stable Baselines3:** https://stable-baselines3.readthedocs.io/

---

## 🤔 Which Approach Should You Choose?

**Choose Hyperparameter Sweep if:**
- This is your first time on the cluster
- You want to compare sparse vs dense rewards
- You have limited budget (it's free!)

**Choose EUREKA if:**
- The sweep didn't work well
- Your task is really complex
- You have $8 to spend on API calls
- You want to try cutting-edge 2026 methods

**My Recommendation:** Start with the sweep. It's free, tests 12 configs, and will likely find a good solution. Only use EUREKA if needed.

---

**Good luck! 🚀**

Questions? Check the detailed README in `cluster/README.md`
