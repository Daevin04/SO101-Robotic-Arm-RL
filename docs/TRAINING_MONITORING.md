# Training Monitoring & Analysis

**Date:** 2026-01-25

Monitor training in real-time and catch hacks/exploits early.

---

## During Training

When you run training:
```bash
python scripts/train_stage_1.py --total-timesteps 25000
```

You'll see updates every 100 episodes:

```
======================================================================
TRAINING MONITOR - Episode 100
======================================================================
Mean reward (last 100):  -305.23
Mean ee_to_obj_dist     :  20.15cm     ← Agent's distance to object
Mean close_steps        :    0.0       ← How many steps within 5cm
Mean gripper_state      :    0.85      ← Gripper position [0=closed, 1=open]
======================================================================
```

**Logs saved to:**
- `logs/stage_1_training_TIMESTAMP.jsonl` - JSON format (for analysis)
- `logs/stage_1_training_TIMESTAMP.csv` - CSV format (for Excel/pandas)

---

## Detecting Hacks During Training

### "Do Nothing" Exploit
```
Episode 100: Mean ee_to_obj_dist: 20.15cm
Episode 200: Mean ee_to_obj_dist: 20.18cm  ← Distance not improving!
Episode 300: Mean ee_to_obj_dist: 20.22cm
```
**Agent learned to stay still instead of reaching.**

### No Close Approaches
```
Mean close_steps: 0.0   ← Agent NEVER gets within 5cm
```
**Agent hasn't discovered the object yet.**

### Reward Not Improving
```
⚠ No improvement for 1000 episodes (best: -305.2, current: -310.3)
```
**Agent stuck in local minimum.**

### Early Stopping
```
⛔ EARLY STOPPING - Agent appears stuck (possible exploit)
No improvement for 2000 episodes
Consider: adjusting rewards, curriculum learning, or hyperparameters
```
**Training auto-stopped to save compute.**

---

## After Training - Analyze Logs

### View latest log
```bash
python scripts/view_training_logs.py
```

Output:
```
======================================================================
TRAINING LOG SUMMARY
======================================================================
Episodes logged:     167
Start episode:       100
End episode:         16700
Total timesteps:     25,000

Reward progression:
  Initial:           -305.23
  Final:             -185.45
  Change:            +119.78  ← Improved!
  Best:              -175.12
  Worst:             -320.55

Latest metrics (episode 16700):
  ee_to_obj_dist      :   8.45cm    ← Getting closer!
  close_steps         :  23.0       ← Spending time near object
======================================================================
```

### View specific stage log
```bash
python scripts/view_training_logs.py --stage stage_1
```

### Plot metrics
```bash
python scripts/view_training_logs.py --plot
```

Creates `training_metrics.png` with graphs of:
- Reward over time
- Distance over time
- Close steps over time
- Any other stage-specific metrics

---

## What to Look For

### Good Training (Learning)
```
Episode 100:  reward=-305.2  dist=20.1cm  close=0.0
Episode 500:  reward=-245.8  dist=15.3cm  close=5.0   ← Getting closer!
Episode 1000: reward=-180.5  dist=8.2cm   close=35.0  ← Much better!
Episode 1500: reward=-95.3   dist=4.5cm   close=85.0  ← Close to success!
```
✅ Reward improving
✅ Distance decreasing
✅ Close steps increasing

### Bad Training (Hacking)
```
Episode 100:  reward=-305.2  dist=20.1cm  close=0.0
Episode 500:  reward=-308.5  dist=20.5cm  close=0.0   ← No change
Episode 1000: reward=-302.1  dist=19.8cm  close=0.0   ← Still stuck
Episode 1500: reward=-310.9  dist=21.2cm  close=0.0   ← Getting worse!
```
❌ Reward not improving
❌ Distance staying high
❌ Never getting close

### Mixed Results (Partial Learning)
```
Episode 1000: reward=-150.5  dist=6.2cm   close=40.0  ← Good!
Episode 1100: reward=-290.3  dist=18.5cm  close=2.0   ← Regressed!
Episode 1200: reward=-165.2  dist=7.1cm   close=38.0  ← Recovered
```
⚠️ Unstable learning - may need:
- Lower learning rate
- More training steps
- Better reward shaping

---

## Log File Formats

### JSONL (logs/stage_1_training_TIMESTAMP.jsonl)
One JSON object per checkpoint:
```json
{"episode": 100, "timestep": 15000, "mean_reward": -305.23, "mean_ee_to_obj_dist": 0.2015, "mean_close_steps": 0.0}
{"episode": 200, "timestep": 30000, "mean_reward": -245.81, "mean_ee_to_obj_dist": 0.1530, "mean_close_steps": 5.2}
```

**Use for:** Python analysis with json/pandas

### CSV (logs/stage_1_training_TIMESTAMP.csv)
```csv
episode,timestep,mean_reward,mean_ee_to_obj_dist,mean_close_steps
100,15000,-305.23,0.2015,0.0
200,30000,-245.81,0.1530,5.2
```

**Use for:** Excel, Google Sheets, quick plots

---

## Python Analysis Example

```python
import json
import pandas as pd

# Load JSONL
with open('logs/stage_1_training_20260125_123456.jsonl') as f:
    data = [json.loads(line) for line in f]

# Or load CSV
df = pd.read_csv('logs/stage_1_training_20260125_123456.csv')

# Check if distance improved
print(f"Start distance: {df['mean_ee_to_obj_dist'].iloc[0] * 100:.2f}cm")
print(f"End distance:   {df['mean_ee_to_obj_dist'].iloc[-1] * 100:.2f}cm")
print(f"Improvement:    {(df['mean_ee_to_obj_dist'].iloc[0] - df['mean_ee_to_obj_dist'].iloc[-1]) * 100:.2f}cm")
```

---

## Comparing Multiple Runs

```bash
# List all logs
ls -lh logs/

# View each one
python scripts/view_training_logs.py logs/stage_1_training_20260125_120000.jsonl
python scripts/view_training_logs.py logs/stage_1_training_20260125_140000.jsonl

# Compare final results
```

Look for:
- Which run had best final reward?
- Which run reached closest distance?
- Which run had most close_steps?

---

## Integration with Tensorboard

Stable-Baselines3 also logs to Tensorboard:
```bash
tensorboard --logdir tensorboard_logs/
```

**Comparison:**
- **Tensorboard**: Lower-level (gradient norms, policy loss, value loss)
- **Our logs**: Higher-level (distance, close_steps, stage metrics)

Use both:
- Our logs → See if agent is reaching object
- Tensorboard → Debug why it's not learning (if needed)

---

## Files Created

```
logs/
├── stage_1_training_20260125_120000.jsonl   ← JSON format
├── stage_1_training_20260125_120000.csv     ← CSV format
├── stage_2_training_20260125_130000.jsonl
└── stage_2_training_20260125_130000.csv

training_metrics.png   ← Generated by --plot
```

---

**Last Updated:** 2026-01-25
