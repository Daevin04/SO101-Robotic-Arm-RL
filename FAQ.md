# Frequently Asked Questions (FAQ)

## General Questions

### What is this project?

This is a reinforcement learning framework for training a 6-DOF robotic arm (SO-101) to perform object manipulation tasks using curriculum learning. The robot learns progressively through stages, from simple alignment to complex grasping and lifting.

### Why curriculum learning?

Curriculum learning breaks down complex tasks into simpler subtasks, allowing the agent to:
- Learn faster (3 hours vs 10+ hours)
- Achieve higher success rates (52% vs 12%)
- Learn more reliably (fewer training failures)
- Debug more easily (isolate issues to specific stages)

### What can the trained robot do?

The fully trained robot can:
- Align its base with target objects
- Reach toward objects with 6-DOF control
- Position its gripper around objects
- Grasp objects with parallel jaw gripper
- Lift grasped objects above the surface

## Installation & Setup

### What are the system requirements?

**Minimum:**
- Python 3.8+
- 4GB RAM
- ~1GB disk space

**Recommended:**
- Python 3.10+
- 8GB+ RAM
- GPU (optional, speeds up training)
- 5GB disk space (for checkpoints and logs)

### How do I install MuJoCo?

Follow the official [MuJoCo installation guide](https://mujoco.org/). For most systems:

```bash
pip install mujoco
```

### Installation fails with "mujoco not found"

Make sure you've installed MuJoCo correctly:

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

If this fails, reinstall:

```bash
pip uninstall mujoco
pip install mujoco>=3.0.0
```

### Can I run this without a GPU?

Yes! CPU-only training works fine. Training times are similar since:
- MuJoCo simulation is CPU-based
- SAC is relatively lightweight
- Most time is spent in simulation, not neural network updates

## Training

### How long does training take?

Full curriculum (5 stages): ~3 hours on a modern CPU

Individual stages:
- Stage 1: ~30-45 minutes (25K steps)
- Stage 2: ~30-45 minutes (25K steps)
- Stage 3: ~30-45 minutes (25K steps)
- Stage 4: ~40-50 minutes (25K steps)
- Stage 5: ~40-50 minutes (25K steps)

### Can I speed up training?

Yes, several options:

1. **Use parallel environments:**
```python
from stable_baselines3.common.vec_env import SubprocVecEnv
env = SubprocVecEnv([make_env for _ in range(4)])
```

2. **Increase batch size** (if you have enough RAM):
```python
model = SAC(..., batch_size=512)
```

3. **Use a GPU** (modest improvement for SAC):
```python
model = SAC(..., device="cuda")
```

### Training seems stuck, what should I do?

1. **Check success rate**: Is it slowly improving?
   - If yes: Be patient, RL can be slow
   - If no: Reward function may need tuning

2. **Visualize the policy:**
```bash
python scripts/watch.py --model <your_model.zip> --env stage_X
```

3. **Check TensorBoard logs:**
```bash
tensorboard --logdir tensorboard_logs/
```

4. **Try a different random seed:**
```bash
python scripts/train.py --stage 1 --timesteps 25000 --seed 123
```

### What success rate should I expect?

Target success rates by stage:
- Stage 1: >75% (typically 75-85%)
- Stage 2: >75% (typically 75-85%)
- Stage 3: >75% (typically 70-80%)
- Stage 4: >70% (typically 65-75%)
- Stage 5: >65% (typically 60-70%)
- **Full task**: >50% (typically 45-55%)

If you're below these, try:
- Training longer (50K steps instead of 25K)
- Adjusting hyperparameters
- Checking reward function

### Can I resume training from a checkpoint?

Yes! Use the `--resume` flag:

```bash
python scripts/train.py \
    --stage 2 \
    --timesteps 25000 \
    --resume checkpoints/stage_1/stage_1_task_25000_steps.zip
```

This is how curriculum learning works - each stage resumes from the previous stage's checkpoint.

## Customization

### How do I create a custom training stage?

See `examples/custom_reward.py` for a complete example. Basic steps:

1. Create a new file `envs/stage_custom.py`
2. Inherit from `SO101BaseEnv`
3. Implement `_compute_reward()`
4. Implement `_is_success()`
5. Register in training scripts

### How do I modify the reward function?

Edit the `_compute_reward()` method in the relevant stage file:

```python
def _compute_reward(self, action):
    # Your custom reward logic here
    distance = self._compute_distance_to_target()
    reward = -distance * 10  # Example
    return reward
```

### Can I train on different objects?

Currently trained on cubes. To add new objects:

1. Add 3D mesh to `assets/meshes/`
2. Update MuJoCo XML to include new object
3. Modify environment to spawn new object
4. Retrain

Future work includes multi-object training and domain randomization.

### How do I integrate vision/cameras?

Currently uses proprioceptive observations (joint positions, etc.). To add vision:

1. Modify observation space to include images
2. Add camera to MuJoCo scene
3. Use CNN policy instead of MLP:
```python
model = SAC("CnnPolicy", env, ...)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for details.

## Evaluation & Debugging

### How do I evaluate a trained model?

```bash
python scripts/evaluate.py \
    --model checkpoints/stage_1/model.zip \
    --env stage_1 \
    --n-episodes 100
```

### How do I watch the robot in action?

```bash
python scripts/watch.py \
    --model checkpoints/stage_1/model.zip \
    --env stage_1
```

Press ESC to close the window.

### The robot behaves strangely, how do I debug?

1. **Visualize the policy** with `watch.py`
2. **Check reward values** - are they reasonable?
3. **Inspect observations** - are they normalized correctly?
4. **Check action bounds** - are actions clipped properly?
5. **Review TensorBoard logs** - look for anomalies

### How do I log custom metrics?

Create a custom callback:

```python
from stable_baselines3.common.callbacks import BaseCallback

class CustomLogger(BaseCallback):
    def _on_step(self):
        # Log custom metrics
        self.logger.record("custom/my_metric", some_value)
        return True

model.learn(..., callback=CustomLogger())
```

## Deployment

### Can I deploy this to a real robot?

The framework is designed for sim-to-real transfer, but:
- **Not yet tested** on physical hardware
- Will require domain randomization
- May need actuator delay modeling
- Contact dynamics may differ

This is planned future work.

### How do I export the trained policy?

The policy is already saved in the `.zip` file. To extract:

```python
from stable_baselines3 import SAC

model = SAC.load("model.zip")
actor = model.policy.actor  # PyTorch network

# Export to ONNX (example)
import torch
dummy_input = torch.randn(1, obs_dim)
torch.onnx.export(actor, dummy_input, "policy.onnx")
```

### What about ROS integration?

Not currently included. For ROS integration:

1. Create a ROS node
2. Subscribe to joint states
3. Publish joint commands
4. Load trained model in the node
5. Use model.predict() for actions

Community contributions welcome!

## Performance

### Why is my GPU not being used?

SAC uses the GPU for neural network updates, but:
- Most time is spent in simulation (CPU)
- The networks are small (MLP)
- GPU benefit is modest (~10-20% faster)

To force GPU usage:

```python
model = SAC(..., device="cuda")
```

### Training uses a lot of RAM, why?

Main memory consumers:
- Replay buffer (default: 100K transitions × observation size)
- MuJoCo simulation state
- TensorBoard logs

To reduce:

```python
model = SAC(..., buffer_size=50000)  # Smaller replay buffer
```

### Can I train multiple stages in parallel?

Not easily - each stage builds on the previous checkpoint. However, you can:
- Train multiple random seeds in parallel
- Run hyperparameter sweeps in parallel
- Train different custom tasks in parallel

## Troubleshooting

### "ModuleNotFoundError: No module named 'envs'"

Add the project root to your Python path:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/SO101-Robotic-Arm-RL"
```

Or run from the project root directory.

### "MuJoCo error: Invalid model"

Check that:
1. MuJoCo XML file is valid
2. All mesh files are present in `assets/meshes/`
3. File paths are correct

### Training is very slow

- Check CPU usage - should be high
- Close other programs
- Reduce render frequency (if rendering)
- Use fewer parallel environments if RAM-limited

### "Checkpoint not found"

Make sure:
1. The checkpoint file exists
2. The path is correct (absolute or relative)
3. The file is a `.zip` file
4. You've trained that stage before resuming

## Contributing

### How can I contribute?

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Common contributions:
- Bug reports
- New training stages
- Performance improvements
- Documentation updates
- Example scripts

### I found a bug, what should I do?

Open an issue on GitHub with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Your environment details

### Can I add my own training stage?

Yes! Please do. Submit a pull request with:
- New environment file
- Documentation
- Example usage
- Test results

## Other

### What's the difference between SAC, PPO, and TD3?

- **SAC** (used here): Sample-efficient, stable, continuous control
- **PPO**: More stable but less sample-efficient
- **TD3**: Similar to SAC but without entropy regularization

We chose SAC for its balance of efficiency and stability.

### Why MuJoCo and not PyBullet/Isaac Gym?

- **MuJoCo**: Best contact dynamics, widely used in research
- **PyBullet**: Free but less accurate contact dynamics
- **Isaac Gym**: GPU-accelerated but NVIDIA-only

MuJoCo 3.0+ is free and open-source.

### Is there a paper for this work?

Not yet - this is an open-source project. If you use it in research, please cite the repository (see README for citation).

### Where can I learn more about RL?

Recommended resources:
- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Sutton & Barto Book](http://incompleteideas.net/book/)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)

---

**Have a question not answered here?** Open an issue on GitHub!
