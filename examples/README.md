# Examples

This directory contains example scripts demonstrating various aspects of the SO-101 RL framework.

## Available Examples

### 1. Quick Training Demo (`quick_train.py`)

A minimal example showing the basic training workflow.

```bash
python examples/quick_train.py
```

**What it demonstrates:**
- Creating an environment
- Initializing a SAC agent
- Training for a small number of steps
- Saving and evaluating a model

**Note**: Uses only 5000 steps for demo purposes. Real training requires 25K+ steps.

### 2. Custom Reward Function (`custom_reward.py`)

Shows how to create a custom training task with a custom reward function.

```bash
python examples/custom_reward.py
```

**What it demonstrates:**
- Inheriting from `SO101BaseEnv`
- Defining custom reward functions
- Setting custom success criteria
- Training the custom task

**Use this as a template** when creating new training stages.

## Running Examples

### Prerequisites

Make sure you've installed the package:

```bash
pip install -r requirements.txt
```

### Running an Example

```bash
# From the project root directory
python examples/quick_train.py
```

### Expected Output

Each example will:
1. Print what it's demonstrating
2. Run the training/evaluation
3. Report results
4. Save any generated models

## Creating Your Own Examples

To add a new example:

1. Create a new Python file in this directory
2. Add appropriate docstrings explaining what it demonstrates
3. Include usage instructions
4. Update this README

## Common Patterns

### Basic Training Loop

```python
from stable_baselines3 import SAC
from envs.stage_1_task import Stage1Task

# Create environment
env = Stage1Task()

# Create agent
model = SAC("MlpPolicy", env, verbose=1)

# Train
model.learn(total_timesteps=25000)

# Save
model.save("my_model")

# Evaluate
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
```

### Loading and Evaluating a Trained Model

```python
from stable_baselines3 import SAC
from envs.stage_1_task import Stage1Task

# Load model
model = SAC.load("checkpoints/stage_1/model.zip")

# Create environment
env = Stage1Task()

# Evaluate
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"Episode finished. Success: {info.get('is_success', False)}")
        obs, _ = env.reset()
```

### Custom Callback

```python
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def _on_step(self) -> bool:
        # This is called after every step
        if self.n_calls % 1000 == 0:
            print(f"Step: {self.n_calls}")
        return True  # Continue training

# Use in training
model.learn(total_timesteps=10000, callback=CustomCallback())
```

## Tips

1. **Start small**: Use fewer timesteps when experimenting
2. **Monitor progress**: Use `verbose=1` to see training progress
3. **Save checkpoints**: Use `CheckpointCallback` to save periodically
4. **Visualize**: Use TensorBoard for detailed metrics
5. **Test changes**: Always test on a few episodes before long training

## Additional Resources

- [Main README](../README.md) - Project overview
- [Quick Start Guide](../docs/QUICK_START.md) - Get started quickly
- [Architecture](../ARCHITECTURE.md) - System design details
- [API Documentation](../docs/) - Detailed API docs

## Questions?

Open an issue on GitHub or check the [FAQ](../FAQ.md).
