# SO-101 Grasp Training

Fast iteration training using 25K-step stages with >75% success milestones.

**Goal:** Train full grasp & lift in ~3 hours (vs 10+ hours traditional)

---

## Quick Start

**See [`QUICK_START.md`](QUICK_START.md) for complete guide!**

### 1. Setup Environment

```bash
cd SO101-Robotic-Arm-RL
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train Stage 1 (Simple Reaching - 25K steps)

```bash
python scripts/train.py --total-timesteps 25000
```

**Training time:** 30-45 minutes
**Target:** >75% success (reach within 5cm of object)

**With frozen object (curriculum learning):**
```bash
python scripts/train.py \
  --total-timesteps 25000 \
  --freeze-object
```

### 3. Resume from Previous Stage

```bash
python scripts/train.py \
  --total-timesteps 25000 \
  --resume checkpoints/stage_1/stage_1_task_25000_steps.zip
```

### 4. Evaluate Trained Model

```bash
python scripts/evaluate.py \
  --model checkpoints/grasp_prep/grasp_prep_final.zip \
  --env grasp_prep \
  --n-episodes 20
```

### 5. Record Videos

```bash
MUJOCO_GL=egl python scripts/evaluate.py \
  --model checkpoints/grasp_prep/grasp_prep_100000_steps.zip \
  --env grasp_prep \
  --record \
  --video-dir videos/my_evaluation \
  --n-episodes 5
```

### 6. Watch Live

```bash
python scripts/watch.py \
  --model checkpoints/grasp_prep/grasp_prep_100000_steps.zip \
  --env grasp_prep
```

---

## Project Structure

```
SO101_Training/
├── envs/                           # Environment definitions & rewards
│   ├── so101_base_env.py          # Base class with shared functionality
│   └── stage_1_task.py            # Stage 1: Reach towards object
├── assets/                         # Robot model (MuJoCo XML)
│   ├── so101_pusher.xml           # Robot, object, table physics
│   └── meshes/                    # 3D visualization meshes
├── scripts/                        # Training & evaluation
│   ├── train.py                   # Training script (all stages)
│   ├── evaluate.py                # Evaluation script
│   └── watch.py                   # Live visualization
├── checkpoints/                    # Saved models (created during training)
├── logs/                           # Training logs (created during training)
├── videos/                         # Evaluation videos (created by evaluate.py)
└── tensorboard_logs/              # TensorBoard logs (created during training)
```

---

## Environment: Grasp Prep

**Task:** Position gripper correctly and grasp object

**Action Space:** 6D continuous
- 5 arm joints (position control)
- 1 gripper joint (open/close)

**Observation Space:** 30D
- Joint positions (6)
- Joint velocities (6)
- End-effector position (3)
- End-effector orientation (4)
- Object position (3)
- Object velocity (3)
- Goal position (3)
- Gripper state (1)
- Grasped flag (1)

**Reward Structure:**

Dense rewards (always active):
- Distance to object: -5.0 × distance
- Gripper open: +0.5 (when distance < 15cm)
- Lateral centering: -2.0 × offset
- Orientation: -1.0 × error
- Height matching: -0.5 × difference
- Action smoothness: -0.01 × action²

Sparse rewards (conditional):
- Contact: +20.0 (distance < 2cm)
- Grasp: +30.0 (contact + gripper closed)
- Lift: +50.0 (object lifted 3cm)
- Success: +100.0 (held for 5 steps)

**Success Criteria:** Lift object 3cm and hold for 5 steps with gripper closed

**Episode Length:** 150 steps

---

## Training Parameters

**Algorithm:** SAC (Soft Actor-Critic)

**Default Hyperparameters:**
- Learning rate: 3e-4
- Batch size: 256
- Buffer size: 100,000
- Tau: 0.005
- Gamma: 0.99
- Target update interval: 1

**Checkpoints:** Saved every 25,000 steps

---

## Monitor Training

**View logs:**
```bash
tail -f logs/grasp_prep_training.log
```

**TensorBoard:**
```bash
tensorboard --logdir tensorboard_logs/
```

---

## Modify Rewards

Edit `envs/so101_positioning_grasp_prep_env.py`, line 138 in `_compute_reward()` method.

---

## Documentation

Additional documentation available in [`docs/`](docs/) folder:
- Sim-to-real changes
- Environment explanation
- Privileged info analysis
- Curriculum learning strategy

---

## Requirements

- Python 3.8+
- MuJoCo
- gymnasium
- stable-baselines3
- numpy

See `requirements.txt` for full list.
