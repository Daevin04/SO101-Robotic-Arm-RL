# Demo Videos

This directory contains demonstration videos showing the SO-101 robotic arm during various training stages and tasks.

## Available Videos

### Grasping Tasks

**`demo_stage6_grasping.mp4`** (125 KB)
- **Stage**: Stage 6 - Advanced Grasping
- **Episode**: 7
- **Description**: Trained agent performing object grasping with the parallel jaw gripper
- **Shows**: End-to-end grasping behavior after curriculum learning

**`demo_stage6_analysis.mp4`** (100 KB)
- **Stage**: Stage 6 Analysis
- **Episode**: 0
- **Description**: Analysis episode showing grasping performance metrics
- **Shows**: Evaluation of learned grasping policy

### Object Manipulation

**`demo_pusher_task.mp4`** (78 KB)
- **Task**: Object Pushing
- **Episode**: 11
- **Description**: Agent learning to push objects with the arm
- **Shows**: Alternative manipulation strategy beyond grasping

**`demo_pusher_learning.mp4`** (88 KB)
- **Task**: Object Pushing (Advanced)
- **Episode**: 15
- **Description**: Improved pushing behavior after extended training
- **Shows**: Learning progression and skill refinement

## Usage in Presentations

These videos can be used to:
- Demonstrate the trained agent's capabilities
- Show the results of curriculum learning
- Illustrate different manipulation strategies
- Present in academic or professional settings

## Viewing Videos

### Command Line
```bash
# Linux
xdg-open demo_stage6_grasping.mp4

# macOS
open demo_stage6_grasping.mp4
```

### In Documentation
Videos can be embedded in Markdown (if hosting on GitHub Pages):
```markdown
![Demo](videos/demo_stage6_grasping.mp4)
```

Or linked:
```markdown
[Watch Demo](videos/demo_stage6_grasping.mp4)
```

## Video Details

| Video | Size | Duration | Quality | Stage/Task |
|-------|------|----------|---------|------------|
| demo_stage6_grasping.mp4 | 125 KB | ~5-10s | 480p | Stage 6 |
| demo_stage6_analysis.mp4 | 100 KB | ~5-10s | 480p | Stage 6 |
| demo_pusher_task.mp4 | 78 KB | ~5-10s | 480p | Pusher |
| demo_pusher_learning.mp4 | 88 KB | ~5-10s | 480p | Pusher |

**Total size**: ~572 KB (well under GitHub's limits)

## Creating More Videos

To record your own demo videos during evaluation:

```python
# In your evaluation script
from stable_baselines3.common.vec_env import VecVideoRecorder

env = VecVideoRecorder(
    env,
    video_folder="videos/",
    record_video_trigger=lambda x: x % 10 == 0,  # Every 10 episodes
    video_length=500,  # Steps per video
    name_prefix="demo"
)
```

Or use the watch script with recording:
```bash
python scripts/watch.py \
    --model checkpoints/stage_6/model.zip \
    --env stage_6 \
    --record \
    --output videos/new_demo.mp4
```

## Notes

- All videos are recorded from MuJoCo simulation
- Videos show deterministic policy evaluation
- File sizes are optimized for web sharing
- Videos are suitable for GitHub embedding

## License

These demonstration videos are part of the SO-101 Robotic Arm RL project and are released under the same MIT License as the rest of the project.

---

**For more information**, see the main [README](../README.md) or [documentation](../docs/).
