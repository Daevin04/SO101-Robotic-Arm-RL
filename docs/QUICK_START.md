# Quick Start: 25K Fast Iteration Training

**Strategy:** Train in small 25K-step stages, >75% success to advance

**Total time to full grasp & lift:** ~3 hours (vs 10+ hours traditional)

---

## Stage 1: Reach (30-45 min) ★☆☆☆☆

**Task:** Point arm at object, get within 5cm

```bash
# Train
python scripts/train.py --total-timesteps 25000

# Evaluate
python scripts/evaluate.py \
  --model checkpoints/stage_1/stage_1_task_25000_steps.zip \
  --env stage_1 \
  --n-episodes 100

# Target: >75% success
```

**If >75% success:** Mark successful, create Stage 2

---

## Stage 2: Close Approach (30-45 min) ★★☆☆☆

**Task:** Get within 2cm, center object between jaws

```bash
# Copy Stage 1 as template
cp envs/stage_1_task.py envs/stage_2_task.py
# (Update: objective, rewards, class name)

# Train from Stage 1 checkpoint
python scripts/train.py \
  --total-timesteps 25000 \
  --resume checkpoints/stage_1/stage_1_task_25000_steps.zip

# Target: >75% within 2cm + centered
```

---

## Stage 3: Grasp (30-45 min) ★★★☆☆

**Task:** Close gripper on object with contact

```bash
# Resume from Stage 2
python scripts/train.py \
  --total-timesteps 25000 \
  --resume checkpoints/stage_2/stage_2_task_25000_steps.zip

# Target: >75% grasp success
```

---

## Stage 4: Lift (30-45 min) ★★★★☆

**Task:** Lift object 3cm above table

```bash
# Resume from Stage 3
python scripts/train.py \
  --total-timesteps 25000 \
  --resume checkpoints/stage_3/stage_3_task_25000_steps.zip

# Target: >75% lift success
```

---

## Full Pipeline

```
Stage 1 (25K) → Evaluate → >75%? ✅
       ↓
Stage 2 (25K) → Evaluate → >75%? ✅
       ↓
Stage 3 (25K) → Evaluate → >75%? ✅
       ↓
Stage 4 (25K) → Evaluate → >75%? ✅
       ↓
DONE! Full grasp & lift in ~3 hours
```

---

## Key Rules

1. **25K steps per stage** - Fast feedback
2. **>75% success to advance** - Clear milestone
3. **Resume from previous stage** - Build on success
4. **If <75% at 25K** - Train 25K more OR revise rewards
5. **Mark successful immediately** - Don't overtrain

---

## File Structure

```
envs/
├── stage_1_task.py   # 🔄 IN PROGRESS → ✅ SUCCESSFUL
├── stage_2_task.py   # Copy from stage_1, update
├── stage_3_task.py   # Copy from stage_2, update
└── stage_4_task.py   # Copy from stage_3, update

checkpoints/
├── stage_1/stage_1_task_25000_steps.zip  ✅ 78% success
├── stage_2/stage_2_task_25000_steps.zip  ✅ 81% success
├── stage_3/stage_3_task_25000_steps.zip  🔄 IN PROGRESS
└── stage_4/...
```

---

## Current Status

**Stage 1:** 🔄 Ready to train (updated to simple reaching task)

**Next:** Run training, evaluate, advance!

---

## Documentation

- **Full strategy:** `docs/FAST_ITERATION_STRATEGY.md`
- **Stage 1 details:** `docs/STAGE_1_SIMPLE_REACHING.md`
- **Stage system:** `docs/STAGE_SYSTEM.md`
- **Environment guide:** `envs/README.md`

---

**Ready to start!** 🚀

```bash
cd SO101-Robotic-Arm-RL
python scripts/train.py --total-timesteps 25000
```
