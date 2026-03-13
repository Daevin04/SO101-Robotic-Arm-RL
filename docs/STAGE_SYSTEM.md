# Stage-Based Training System

**Date Implemented:** 2026-01-25
**Status:** ✅ Active

---

## Overview

Training is now organized into **stages** for better trial-and-error iteration:
- Clear status tracking (In Progress → Successful)
- Consistent naming convention
- Each stage is self-contained
- Build on previous successful stages

---

## Changes Made

### File Renaming

| Before | After |
|--------|-------|
| `so101_positioning_grasp_prep_env.py` | `stage_1_task.py` |
| `SO101PositioningGraspPrepEnv` class | `Stage1Task` class |
| `checkpoints/grasp_prep/` | `checkpoints/stage_1/` |
| `grasp_prep_25000_steps.zip` | `stage_1_task_25000_steps.zip` |

### Updated Files

1. ✅ `envs/stage_1_task.py` - Renamed and updated header
2. ✅ `envs/__init__.py` - Updated imports
3. ✅ `scripts/train.py` - Updated imports and checkpoint naming
4. ✅ `scripts/evaluate.py` - Simplified to use Stage1Task
5. ✅ `envs/README.md` - Created stage system documentation

---

## Stage 1 Status Header

```python
"""
STAGE 1: Grasp & Lift Task

STATUS: 🔄 IN PROGRESS
Date Started: 2026-01-25
Best Checkpoint: None yet
Success Rate: 0%

================================================================================
TASK OBJECTIVE
================================================================================
Position gripper correctly and grasp object:
    1. Approach object (gripper open)
    2. Center object between jaws
    3. Close gripper on object
    4. Lift object 3cm
    5. Hold for 5 steps

... (reward structure, training notes)

================================================================================
WHEN TO MARK SUCCESSFUL
================================================================================
Success criteria to advance to Stage 2:
    - 10%+ success rate over 100 episodes
    - Consistent grasping behavior
    - No reward exploits

Update this header to:
    STATUS: ✅ SUCCESSFUL
    Best Checkpoint: stage_1_task_XXXk_steps.zip
    Success Rate: XX%
"""
```

---

## Training Commands

### Start Training
```bash
cd /home/oeyd/SO101_Training
python scripts/train.py --total-timesteps 300000
```

**Checkpoint naming:** `checkpoints/stage_1/stage_1_task_#_steps.zip`

### Evaluate
```bash
MUJOCO_GL=egl python scripts/evaluate.py \
  --model checkpoints/stage_1/stage_1_task_100000_steps.zip \
  --env stage_1 \
  --record \
  --video-dir videos/stage_1_100k \
  --n-episodes 5
```

---

## When Stage 1 Succeeds

### 1. Update Stage File Header
```python
STATUS: ✅ SUCCESSFUL
Date Completed: 2026-01-XX
Best Checkpoint: stage_1_task_300k_steps.zip
Success Rate: 15%
```

### 2. Create Stage 2
```bash
# Copy stage 1 as template
cp envs/stage_1_task.py envs/stage_2_task.py

# Update:
# - Change "STAGE 1" → "STAGE 2"
# - Reset STATUS to 🔄 IN PROGRESS
# - Update task objective
# - Modify reward structure
# - Rename class: Stage1Task → Stage2Task
```

### 3. Update Training Script
```bash
cp scripts/train.py scripts/train.py

# Update:
# - Import Stage2Task
# - Change checkpoint dir to checkpoints/stage_2
# - Change name_prefix to stage_2_task
```

### 4. Train Stage 2 from Stage 1 Checkpoint
```bash
python scripts/train.py \
  --total-timesteps 300000 \
  --resume checkpoints/stage_1/stage_1_task_300k_steps.zip
```

---

## Naming Convention

### Files (lowercase, underscore)
- `stage_1_task.py`
- `stage_2_task.py`
- `train.py`

### Classes (PascalCase)
- `Stage1Task`
- `Stage2Task`

### Checkpoints
- `stage_1_task_25000_steps.zip`
- `stage_1_task_50000_steps.zip`
- `stage_1_task_final.zip`

### Directories
- `checkpoints/stage_1/`
- `checkpoints/stage_2/`
- `videos/stage_1_eval/`

---

## Status Indicators

| Icon | Meaning |
|------|---------|
| 🔄 | IN PROGRESS - Currently training |
| ✅ | SUCCESSFUL - Met success criteria |
| ❌ | FAILED - Abandoned (fundamental issues) |
| ⏸️ | NOT STARTED - Planned future stage |

---

## Benefits

**Organization:**
- Clear which stage you're working on
- Status visible in file header
- Consistent naming across checkpoints/videos

**Iteration:**
- Easy to start new stage from successful one
- Failed attempts clearly marked
- History preserved in stage files

**Collaboration:**
- Anyone can see current stage status
- Success criteria clearly defined
- Easy to pick up where you left off

---

## Example Stage Progression

```
Stage 1: Basic Grasp & Lift
├── Status: ✅ SUCCESSFUL (15% success)
├── Checkpoint: stage_1_task_300k_steps.zip
└── Issues Fixed: Object pushing, reward exploits

Stage 2: Grasp with Distractors (3 objects)
├── Status: 🔄 IN PROGRESS
├── Started from: Stage 1 checkpoint
└── Goal: 10% success with correct object

Stage 3: Visual Servoing (Camera-based)
├── Status: ⏸️ NOT STARTED
└── Will use: Vision observations instead of privileged state
```

---

## Testing

```bash
cd /home/oeyd/SO101_Training

# Test import
python -c "from envs.stage_1_task import Stage1Task; print('✅ Import successful')"

# Test environment
python -c "
from envs.stage_1_task import Stage1Task
env = Stage1Task()
obs, _ = env.reset()
print(f'✅ Observation shape: {obs.shape}')
env.close()
"
```

---

**Last Updated:** 2026-01-25
