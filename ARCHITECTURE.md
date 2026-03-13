# Architecture Overview

This document provides a technical overview of the SO-101 Robotic Arm RL system architecture.

## System Components

### 1. Simulation Environment (MuJoCo)

The physical simulation is built on MuJoCo (Multi-Joint dynamics with Contact), providing:
- High-fidelity physics simulation
- Accurate contact dynamics
- Efficient computation for parallel training
- Realistic actuator dynamics

**Robot Specifications:**
- **Degrees of Freedom**: 6 (shoulder_pan, shoulder_lift, elbow, wrist_pitch, wrist_roll, gripper)
- **Control Method**: Position control for joints, effort control for gripper
- **Workspace**: ~30cm radius hemisphere
- **Gripper**: Parallel jaw gripper with force feedback

### 2. Environment Design (Gymnasium Interface)

#### Base Environment (`SO101BaseEnv`)

Core functionality shared across all training stages:

```python
class SO101BaseEnv(gymnasium.Env):
    - MuJoCo model loading and initialization
    - Observation space generation
    - Action space mapping
    - Contact detection
    - Rendering utilities
    - Episode management
```

**Observation Space** (45-dimensional):
- Joint positions (6D)
- Joint velocities (6D)
- End-effector position (3D)
- End-effector velocity (3D)
- Object position (3D)
- Object orientation (4D - quaternion)
- Object velocity (6D - linear + angular)
- Gripper state (1D)
- Distance to object (1D)
- Contact indicators (6D)
- Previous action (6D)

**Action Space** (6-dimensional):
- Continuous control: [-1, 1] normalized
- Maps to joint position targets
- Clipped to joint limits

#### Stage Environments

Each training stage inherits from `SO101BaseEnv` and implements:
- Custom reward function
- Stage-specific success criteria
- Progressive difficulty scaling
- Episode termination conditions

### 3. Reinforcement Learning

#### Algorithm: Soft Actor-Critic (SAC)

SAC is chosen for its:
- **Sample efficiency**: Learns from fewer interactions
- **Stability**: Entropy regularization prevents premature convergence
- **Continuous control**: Natural fit for robotic manipulation
- **Off-policy learning**: Can reuse past experience

**Hyperparameters:**
```python
{
    "learning_rate": 3e-4,
    "buffer_size": 100000,
    "learning_starts": 1000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
}
```

### 4. Curriculum Learning Framework

#### Stage Progression

```
Stage 1: Base Rotation (1-DOF effective)
   ↓ (resume checkpoint)
Stage 2: Claw Positioning (6-DOF)
   ↓ (resume checkpoint)
Stage 3: Grasp Preparation
   ↓ (resume checkpoint)
Stage 4: Object Grasping
   ↓ (resume checkpoint)
Stage 5: Lifting
```

Each stage:
1. **Builds on previous checkpoint**: Warm-starts from learned policy
2. **Incremental difficulty**: Adds complexity gradually
3. **Clear success criteria**: >75% success rate to advance
4. **Fast iteration**: 25K steps per stage (~30-45 minutes)

#### Reward Shaping Strategy

**Sparse vs. Dense Rewards:**
- Early stages: Sparse rewards with dense guidance
- Later stages: Mostly sparse rewards for robust behavior

**Reward Components:**
```python
reward = (
    alignment_reward +      # Stage 1
    distance_reward +       # Stage 2
    positioning_reward +    # Stage 3
    grasp_reward +          # Stage 4
    lift_reward             # Stage 5
)
```

### 5. Training Pipeline

#### Workflow

```
1. Initialize environment
2. Load previous checkpoint (if resuming)
3. Train for N timesteps
4. Save checkpoint every M steps
5. Evaluate performance
6. Advance to next stage if criteria met
```

#### Callbacks

**CheckpointCallback**: Saves model periodically
**StageMonitorCallback**: Tracks stage-specific metrics
**EarlyStoppingCallback**: Stops if success criteria met early
**TensorBoardCallback**: Logs metrics for visualization

### 6. Monitoring & Evaluation

#### TensorBoard Integration

Real-time metrics:
- Episode reward (mean, std, min, max)
- Success rate (rolling average)
- Episode length
- Q-value estimates
- Actor/Critic losses
- Entropy coefficient

#### Evaluation Protocol

```python
# Run 100 episodes with trained model
python scripts/evaluate.py \
    --model checkpoints/stage_X/model.zip \
    --n-episodes 100 \
    --env stage_X
```

Reports:
- Mean/std episode reward
- Success rate
- Average episode length
- Confidence intervals

## Data Flow

```
User Action
    ↓
Environment.step(action)
    ↓
MuJoCo Simulation (physics)
    ↓
Observation Generation
    ↓
Reward Computation
    ↓
SAC Agent (policy update)
    ↓
Next Action
```

## File Organization

```
SO101-Robotic-Arm-RL/
│
├── envs/                    # Environment definitions
│   ├── so101_base_env.py   # Base class (shared utilities)
│   ├── stage_1_task.py     # Stage 1 environment
│   └── ...                  # Additional stages
│
├── scripts/                 # Executable scripts
│   ├── train.py            # Main training entry point
│   ├── evaluate.py         # Model evaluation
│   ├── watch.py            # Visualize trained policy
│   └── training_monitor.py # Custom callbacks
│
├── assets/                  # 3D models and resources
│   └── meshes/             # STL files for robot parts
│
├── docs/                    # Documentation
│   ├── QUICK_START.md
│   ├── CURRICULUM_STRATEGY.md
│   └── ...
│
├── checkpoints/            # Saved models (git-ignored)
├── logs/                   # Training logs (git-ignored)
└── videos/                 # Episode recordings (git-ignored)
```

## Key Design Decisions

### 1. Curriculum Learning over End-to-End Training

**Rationale**: Breaking down the complex grasping task into progressive stages:
- Faster convergence (3 hours vs 10+ hours)
- More reliable learning
- Easier debugging (isolate failures to specific stages)
- Better sim-to-real transfer potential

### 2. Position Control over Velocity Control

**Rationale**:
- More stable for manipulation tasks
- Easier to specify desired configurations
- Better matches physical robot capabilities

### 3. SAC over PPO/TD3

**Rationale**:
- More sample-efficient than PPO
- More stable than TD3 (entropy regularization)
- Excellent for continuous control

### 4. Sparse Rewards with Dense Guidance

**Rationale**:
- Sparse rewards: Prevent reward hacking
- Dense guidance: Provide learning signal in early stages
- Balance: Achieve both efficiency and robustness

## Extensibility

### Adding New Stages

1. Create new environment file inheriting from `SO101BaseEnv`
2. Implement `_compute_reward()` method
3. Define success criteria in `_is_success()`
4. Register in `ENV_MAP` in training scripts
5. Add documentation

### Customizing Observations

Modify `_get_obs()` in `SO101BaseEnv`:
```python
def _get_obs(self):
    # Add custom observations
    custom_obs = self._compute_custom_feature()
    return np.concatenate([base_obs, custom_obs])
```

### Integrating Vision

Replace proprioceptive observations with image observations:
```python
observation_space = gymnasium.spaces.Box(
    low=0, high=255,
    shape=(84, 84, 3),
    dtype=np.uint8
)
```

## Performance Considerations

### Computational Requirements

- **Training**: 1 GPU (optional, CPU sufficient)
- **Memory**: ~4GB RAM
- **Storage**: ~1GB for checkpoints
- **Time**: ~3 hours for full curriculum (5 stages)

### Optimization Opportunities

1. **Parallel environments**: Use `SubprocVecEnv` for faster data collection
2. **Batch size tuning**: Larger batches for stable gradients
3. **Replay buffer**: Adjust size based on memory availability
4. **Checkpoint frequency**: Balance storage vs. granularity

## Future Enhancements

- **Domain Randomization**: Vary object properties, lighting, friction
- **Vision Integration**: Camera-based observations
- **Sim-to-Real Transfer**: Deploy to physical robot
- **Multi-Task Learning**: Train single policy for multiple tasks
- **Hierarchical RL**: Learn high-level and low-level policies

---

For implementation details, see the code documentation and inline comments.
