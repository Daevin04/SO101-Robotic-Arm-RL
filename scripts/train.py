#!/usr/bin/env python3
"""
Unified training interface for SO-101 robotic arm.

Interactive menu to:
  - Select training stage (Stage 1, Stage 2, etc.)
  - Resume from checkpoint or start fresh
  - Specify training duration

Usage:
    python train.py                    # Interactive mode
    python train.py --stage 1 --timesteps 25000 --resume auto  # CLI mode
"""

import argparse
import sys
import subprocess
import time
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.stage_1_task import Stage1Task

# Import each stage individually with separate try-except
try:
    from envs.stage_2_task import Stage2Task
except ImportError:
    Stage2Task = None

try:
    from envs.stage_2a_task import Stage2ATask
except ImportError:
    Stage2ATask = None

try:
    from envs.stage_2b_task import Stage2BTask
except ImportError:
    Stage2BTask = None

try:
    from envs.stage_2c_task import Stage2CTask
except ImportError:
    Stage2CTask = None

try:
    from envs.stage_3_task import Stage3Task
except ImportError:
    Stage3Task = None

try:
    from envs.stage_4_task import Stage4Task
except ImportError:
    Stage4Task = None

try:
    from envs.stage_5_task import Stage5Task
except ImportError:
    Stage5Task = None

try:
    from envs.stage_6_task import Stage6Task
except ImportError:
    Stage6Task = None

try:
    from envs.stage_7_task import Stage7Task
except ImportError:
    Stage7Task = None

from scripts.training_monitor import StageMonitorCallback, EarlyStoppingCallback


STAGE_INFO = {
    1: {
        "name": "Stage 1: Base Rotation (Align with Cube)",
        "env_class": Stage1Task,
        "description": "Train shoulder_pan to rotate and face cube (1-DOF, simplest task)",
        "default_timesteps": 25000,
    },
    2: {
        "name": "Stage 2: Claw Positioning (Gated)",
        "env_class": Stage2Task,
        "description": "Position stationary claw tip to cube (gated: requires Stage 1 complete first)",
        "default_timesteps": 50000,
    },
    "2a": {
        "name": "Stage 2A: Touch Cube (Sparse Contact Only)",
        "env_class": Stage2ATask,
        "description": "Sparse reward for touching cube (distance < 3cm)",
        "default_timesteps": 25000,
    },
    "2b": {
        "name": "Stage 2B: Distance Shaping (Dense Reward)",
        "env_class": Stage2BTask,
        "description": "Dense distance-based reward + contact bonus",
        "default_timesteps": 25000,
    },
    "2c": {
        "name": "Stage 2C: Gripper Positioning (Touch + Gripper)",
        "env_class": Stage2CTask,
        "description": "Touch cube + open gripper to 90 degrees simultaneously",
        "default_timesteps": 25000,
    },
    3: {
        "name": "Stage 3: Ultra-Precise Positioning (Infinite Hold)",
        "env_class": Stage3Task,
        "description": "Infinite hold design with ultra-tight tolerances (±1.0-1.2cm), continuous positioning rewards",
        "default_timesteps": 100000,
    },
    4: {
        "name": "Stage 4: Penalty-Based Positioning (Urgency)",
        "env_class": Stage4Task,
        "description": "Penalty-driven: -20/step NOT positioned, +100/step positioned. Creates urgency.",
        "default_timesteps": 100000,
    },
    5: {
        "name": "Stage 5: Binary Positioning (Cube Between Jaws)",
        "env_class": Stage5Task,
        "description": "Binary reward: -20/step NOT positioned, +1000/step when cube between jaws. Gripper frozen open.",
        "default_timesteps": 100000,
    },
    6: {
        "name": "Stage 6: Gated Curriculum (Grasp → Navigate)",
        "env_class": Stage6Task,
        "description": "GATE 1: Penalties drive grasping, gripper freezes when achieved. GATE 2: Rewards drive navigation to target (green circle). Sequential skill mastery!",
        "default_timesteps": 150000,
    },
    7: {
        "name": "Stage 7: UR5 Sparse Reward System",
        "env_class": Stage7Task,
        "description": "Sparse rewards: +100 for success + speed bonus, -10×distance for failure. Force-based grasping (>3N).",
        "default_timesteps": 200000,
    },
}


def find_checkpoints(stage_num):
    """Find all checkpoints from all stages.

    Shows all available checkpoints in the checkpoints folder,
    allowing you to resume training from any stage.
    """
    checkpoints_root = Path(__file__).parent.parent / "checkpoints"

    if not checkpoints_root.exists():
        return []

    # Find all stage directories
    all_checkpoints = []
    for stage_dir in checkpoints_root.iterdir():
        if stage_dir.is_dir() and stage_dir.name.startswith('stage_'):
            # Find all checkpoint files in this directory
            checkpoints = list(stage_dir.glob("*.zip"))
            all_checkpoints.extend(checkpoints)

    # Sort by stage number and then timesteps
    def sort_key(path):
        try:
            name = path.stem
            parts = name.split('_')

            # Extract stage (e.g., "stage_1" or "stage_2a")
            stage_str = parts[1] if len(parts) > 1 else "0"

            # Extract timesteps
            for i, part in enumerate(parts):
                if part == "steps" and i > 0:
                    timesteps = int(parts[i-1])
                    # Sort by stage as string (handles "2a" < "2b"), then timesteps
                    return (stage_str, timesteps)
            return (stage_str, 0)
        except:
            return ("0", 0)

    all_checkpoints.sort(key=sort_key)
    return all_checkpoints


def select_stage():
    """Interactive stage selection. Returns (stages_list, use_parallel)."""
    print("\n" + "=" * 70)
    print("SELECT TRAINING STAGE")
    print("=" * 70)

    available_stages = []
    for stage_num, info in STAGE_INFO.items():
        if info["env_class"] is not None:
            available_stages.append(stage_num)
            print(f"  {stage_num}. {info['name']}")
            print(f"     {info['description']}")
            print(f"     Default: {info['default_timesteps']:,} timesteps")
            print()

    print("=" * 70)
    print("TIP: Enter multiple stages separated by commas")
    print("     Example: '2a, 2b, 2c' will train three variants")

    while True:
        try:
            choice = input(f"\nSelect stage(s): ").strip().lower()

            # Check if comma-separated (multiple stages)
            if ',' in choice:
                stage_strings = [s.strip() for s in choice.split(',')]
                stages = []
                invalid = []

                for s in stage_strings:
                    try:
                        stage_num = int(s)
                    except ValueError:
                        stage_num = s

                    if stage_num in available_stages:
                        stages.append(stage_num)
                    else:
                        invalid.append(s)

                if invalid:
                    print(f"Invalid stages: {', '.join(invalid)}")
                    print(f"Available: {', '.join(str(s) for s in available_stages)}")
                elif stages:
                    # Ask about parallel vs sequential
                    if len(stages) > 1:
                        print("\n" + "=" * 70)
                        print(f"Train {len(stages)} stages:")
                        print("  1. Sequentially (one after another)")
                        print("  2. In Parallel (all at the same time)")
                        print("=" * 70)
                        while True:
                            mode = input("Select mode (1 or 2): ").strip()
                            if mode == "1":
                                return stages, False
                            elif mode == "2":
                                return stages, True
                            else:
                                print("Please enter 1 or 2")
                    else:
                        return stages, False
            else:
                # Single stage
                try:
                    stage_num = int(choice)
                except ValueError:
                    stage_num = choice

                if stage_num in available_stages:
                    return [stage_num], False
                else:
                    print(f"Invalid stage. Available: {', '.join(str(s) for s in available_stages)}")
        except KeyboardInterrupt:
            print("\nCancelled")
            sys.exit(0)


def select_checkpoint(stage_num):
    """Interactive checkpoint selection."""
    checkpoints = find_checkpoints(stage_num)

    print("\n" + "=" * 70)
    print(f"STAGE {stage_num} - SELECT CHECKPOINT")
    print("=" * 70)

    if not checkpoints:
        print("  No existing checkpoints found")
        print("  Will train from scratch")
        print("=" * 70)
        return None

    print("  0. Train from scratch (new model)")
    for i, ckpt in enumerate(checkpoints, 1):
        # Extract timesteps from filename
        name = ckpt.stem
        parts = name.split('_')
        for j, part in enumerate(parts):
            if part == "steps" and j > 0:
                timesteps = int(parts[j-1])
                print(f"  {i}. Resume from {timesteps:,} timesteps ({ckpt.name})")
                break

    print("=" * 70)

    while True:
        try:
            choice = input(f"\nSelect checkpoint (0-{len(checkpoints)}): ").strip()
            idx = int(choice)
            if idx == 0:
                return None  # Train from scratch
            elif 1 <= idx <= len(checkpoints):
                return checkpoints[idx - 1]
            else:
                print(f"Please enter a number between 0 and {len(checkpoints)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nCancelled")
            sys.exit(0)


def select_timesteps(default=25000, checkpoint_path=None):
    """Interactive timestep selection."""
    if checkpoint_path:
        print("\n" + "=" * 70)
        print("TRAINING DURATION")
        print("=" * 70)
        print(f"How many ADDITIONAL timesteps to train?")
        print(f"(Default: {default:,} timesteps)")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("TRAINING DURATION")
        print("=" * 70)
        print(f"How many timesteps to train?")
        print(f"(Default: {default:,} timesteps)")
        print("=" * 70)

    while True:
        try:
            timesteps_input = input(f"\nTimesteps [default: {default:,}]: ").strip()
            if timesteps_input == "":
                return default
            timesteps = int(timesteps_input)
            if timesteps > 0:
                return timesteps
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nCancelled")
            sys.exit(0)


def make_env(env_class, freeze_object=False):
    """Create environment."""
    if env_class == Stage1Task:
        return Stage1Task(freeze_object=freeze_object)
    elif env_class == Stage2Task and Stage2Task is not None:
        return Stage2Task(freeze_object=freeze_object)
    elif env_class == Stage2ATask and Stage2ATask is not None:
        return Stage2ATask(freeze_object=freeze_object)
    elif env_class == Stage2BTask and Stage2BTask is not None:
        return Stage2BTask(freeze_object=freeze_object)
    elif env_class == Stage2CTask and Stage2CTask is not None:
        return Stage2CTask(freeze_object=freeze_object)
    elif env_class == Stage3Task and Stage3Task is not None:
        return Stage3Task(freeze_object=freeze_object)
    elif env_class == Stage4Task and Stage4Task is not None:
        return Stage4Task(freeze_object=freeze_object)
    elif env_class == Stage5Task and Stage5Task is not None:
        return Stage5Task(freeze_object=freeze_object)
    elif env_class == Stage6Task and Stage6Task is not None:
        return Stage6Task(freeze_object=freeze_object)
    elif env_class == Stage7Task and Stage7Task is not None:
        return Stage7Task(freeze_object=freeze_object)
    else:
        raise ValueError(f"Unknown environment class: {env_class}")


def train(stage_num, checkpoint_path, timesteps, learning_rate=3e-4, batch_size=256,
          buffer_size=100000, device="auto", freeze_object=False):
    """Run training with specified parameters."""

    stage_info = STAGE_INFO[stage_num]
    env_class = stage_info["env_class"]

    print("\n" + "=" * 80)
    print(f"TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Stage: {stage_info['name']}")
    print(f"Checkpoint: {checkpoint_path.name if checkpoint_path else 'None (train from scratch)'}")
    print(f"Timesteps: {timesteps:,} {'additional' if checkpoint_path else 'total'}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"Freeze object: {'YES' if freeze_object else 'NO'}")
    print("=" * 80)

    # Create environment
    env = DummyVecEnv([lambda: make_env(env_class, freeze_object=freeze_object)])

    # Create or load model
    if checkpoint_path:
        print(f"\n✓ Loading model from: {checkpoint_path}")
        model = SAC.load(str(checkpoint_path), env=env, device=device)

        resume_timesteps = model.num_timesteps
        print(f"✓ Model loaded successfully")
        print(f"✓ Previously trained: {resume_timesteps:,} timesteps")
        print(f"✓ Will train for: {timesteps:,} additional timesteps")
        print(f"✓ Total after training: {resume_timesteps + timesteps:,} timesteps")
    else:
        print("\n✓ Creating new model from scratch")
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=0.005,
            gamma=0.99,
            ent_coef="auto",
            device=device,
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
        )

    # Setup callbacks
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints" / f"stage_{stage_num}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=str(checkpoint_dir),
        name_prefix=f"stage_{stage_num}_task",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # Monitor stage metrics during training
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    monitor_callback = StageMonitorCallback(
        check_freq=100,
        log_dir="logs",
        stage_name=f"stage_{stage_num}",
        verbose=1
    )

    # Early stopping if agent gets stuck
    # Stage 7 (sparse rewards) needs much higher patience
    if stage_num == 7:
        # Sparse rewards: disable early stopping (or very high patience)
        early_stop_callback = EarlyStoppingCallback(
            check_freq=500,
            patience=100000,  # Effectively disabled - sparse rewards need long exploration
            min_reward_improvement=1.0,
            verbose=1
        )
    else:
        # Dense rewards: normal early stopping
        early_stop_callback = EarlyStoppingCallback(
            check_freq=500,
            patience=2000,
            min_reward_improvement=10.0,
            verbose=1
        )

    print(f"\n📊 Logging training metrics to: logs/stage_{stage_num}_training_{timestamp}.jsonl")
    print(f"📊 CSV metrics saved to: logs/stage_{stage_num}_training_{timestamp}.csv")

    print("\n" + "=" * 80)
    print(f"STAGE {stage_num} TRAINING: {stage_info['name']}")
    print("=" * 80)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    print(f"Checkpoint naming: stage_{stage_num}_task_#_steps.zip")
    print(f"Checkpoints saved every 25,000 steps")
    print(f"Monitoring: Stage metrics logged every 100 episodes")
    print(f"Early stopping: Enabled (stops if stuck for 2000 episodes)")
    print("=" * 80 + "\n")

    # Train
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_callback, monitor_callback, early_stop_callback],
        log_interval=10,
        progress_bar=True,
    )

    # Save final model with total timesteps in name
    total_trained_timesteps = model.num_timesteps
    final_path = checkpoint_dir / f"stage_{stage_num}_task_{total_trained_timesteps}_steps.zip"
    model.save(str(final_path))
    print(f"\n✓ Final model saved to: {final_path}")
    print(f"✓ Total timesteps trained: {total_trained_timesteps:,}")

    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION (20 episodes)")
    print("=" * 80)

    obs = env.reset()
    total_reward = 0
    total_success = 0
    episodes = 0

    while episodes < 20:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]

        if done[0]:
            episodes += 1
            if info[0].get("is_success", False):
                total_success += 1
            obs = env.reset()

    mean_reward = total_reward / episodes
    success_rate = (total_success / episodes) * 100

    print(f"\nMean Reward: {mean_reward:.2f}")
    print(f"Success Rate: {success_rate:.1f}%")
    print("=" * 80)

    env.close()


def train_parallel(stages, checkpoint_paths, timesteps, learning_rate, batch_size, buffer_size, device, freeze_object):
    """Launch parallel training processes for multiple stages.

    Args:
        stages: List of stage identifiers
        checkpoint_paths: Dict mapping stage_num -> checkpoint path (or None)
        timesteps: Number of timesteps to train
        learning_rate, batch_size, buffer_size, device, freeze_object: Training params
    """

    print("\n" + "=" * 80)
    print(f"PARALLEL TRAINING: {len(stages)} stages simultaneously")
    print("=" * 80)
    for i, stage_num in enumerate(stages, 1):
        ckpt = checkpoint_paths.get(stage_num)
        ckpt_str = ckpt.name if ckpt else "from scratch"
        print(f"  {i}. {STAGE_INFO[stage_num]['name']} ({ckpt_str})")
    print("=" * 80)
    print(f"\nLaunching {len(stages)} training processes...")
    print("Each process will run in the background.")
    print("Check individual log files for progress.")
    print("\nPress Ctrl+C to stop all training processes.\n")

    # Build command for each stage
    processes = []
    script_path = Path(__file__).resolve()

    for stage_num in stages:
        checkpoint = checkpoint_paths.get(stage_num)

        # Build command arguments
        cmd = [
            sys.executable,
            str(script_path),
            "--stage", str(stage_num),
            "--timesteps", str(timesteps),
            "--learning-rate", str(learning_rate),
            "--batch-size", str(batch_size),
            "--buffer-size", str(buffer_size),
            "--device", device,
        ]

        # Add resume argument
        if checkpoint:
            cmd.extend(["--resume", str(checkpoint)])
        else:
            cmd.extend(["--resume", "none"])

        if freeze_object:
            cmd.append("--freeze-object")

        # Create log file for this process
        log_file = Path(f"logs/parallel_stage_{stage_num}_{time.strftime('%Y%m%d_%H%M%S')}.log")
        log_file.parent.mkdir(exist_ok=True)

        print(f"Starting Stage {stage_num}: {STAGE_INFO[stage_num]['name']}")
        print(f"  Log: {log_file}")

        # Launch process
        with open(log_file, 'w') as f:
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
            processes.append((stage_num, proc, log_file))

        time.sleep(1)  # Stagger launches slightly

    print(f"\n✓ All {len(processes)} training processes launched!\n")
    print("=" * 80)
    print("MONITORING (Ctrl+C to stop all)")
    print("=" * 80)

    try:
        # Wait for all processes to complete
        while True:
            all_done = True
            for stage_num, proc, log_file in processes:
                if proc.poll() is None:
                    all_done = False

            if all_done:
                break

            # Show status update every 30 seconds
            time.sleep(30)
            print(f"\n[{time.strftime('%H:%M:%S')}] Status update:")
            for stage_num, proc, log_file in processes:
                status = "RUNNING" if proc.poll() is None else "COMPLETED"
                print(f"  Stage {stage_num}: {status}")

        print("\n" + "=" * 80)
        print("ALL TRAINING COMPLETE")
        print("=" * 80)

        # Report final status
        for stage_num, proc, log_file in processes:
            return_code = proc.returncode
            if return_code == 0:
                print(f"✓ Stage {stage_num}: SUCCESS")
            else:
                print(f"✗ Stage {stage_num}: FAILED (code {return_code})")
                print(f"  Check log: {log_file}")

    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("STOPPING ALL TRAINING PROCESSES")
        print("=" * 80)

        for stage_num, proc, log_file in processes:
            if proc.poll() is None:
                print(f"Terminating Stage {stage_num}...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"Force killing Stage {stage_num}...")
                    proc.kill()

        print("\nAll processes stopped.")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Train SO-101 Robotic Arm")
    parser.add_argument("--stage", type=str, default=None,
                        help="Stage identifier(s) - single (e.g., '2a') or comma-separated (e.g., '2a,2b,2c'). If not specified, shows interactive menu.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path to resume from, or 'auto' for latest, or 'none' for scratch")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total timesteps to train")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--buffer-size", type=int, default=100000,
                        help="Replay buffer size")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cpu/cuda/auto)")
    parser.add_argument("--freeze-object", action="store_true",
                        help="Freeze object position during training")
    parser.add_argument("--parallel", action="store_true",
                        help="Train multiple stages in parallel (simultaneously) instead of sequentially")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("SO-101 ROBOTIC ARM TRAINING")
    print("=" * 80)

    # Parse stages (can be multiple)
    if args.stage is None:
        stages, use_parallel_interactive = select_stage()  # Returns (list, bool)
        # Override args.parallel if interactive mode selected it
        if use_parallel_interactive:
            args.parallel = True
    else:
        # Parse comma-separated stages
        if ',' in args.stage:
            stage_strings = [s.strip() for s in args.stage.split(',')]
        else:
            stage_strings = [args.stage.strip()]

        stages = []
        for s in stage_strings:
            try:
                stage_num = int(s)
            except ValueError:
                stage_num = s.lower()

            if stage_num not in STAGE_INFO:
                print(f"Error: Invalid stage '{stage_num}'. Available: {', '.join(str(s) for s in STAGE_INFO.keys())}")
                sys.exit(1)
            stages.append(stage_num)

    # Handle parallel training if requested and multiple stages
    if len(stages) > 1 and args.parallel:
        # For interactive mode, prompt for timesteps and resume
        if args.timesteps is None:
            print("\n" + "=" * 70)
            print("PARALLEL TRAINING - TIMESTEPS")
            print("=" * 70)
            default_timesteps = STAGE_INFO[stages[0]]["default_timesteps"]
            timesteps_input = input(f"Timesteps for each stage [default: {default_timesteps:,}]: ").strip()
            if timesteps_input == "":
                args.timesteps = default_timesteps
            else:
                args.timesteps = int(timesteps_input)

        # Collect checkpoint selection for each stage
        checkpoint_paths = {}

        # Handle CLI resume argument
        if args.resume == 'none':
            # All stages train from scratch
            for stage_num in stages:
                checkpoint_paths[stage_num] = None
        elif args.resume == 'auto':
            # Each stage uses latest checkpoint
            for stage_num in stages:
                checkpoints = find_checkpoints(stage_num)
                checkpoint_paths[stage_num] = checkpoints[-1] if checkpoints else None
                if checkpoint_paths[stage_num]:
                    print(f"\n✓ Stage {stage_num}: Auto-selected {checkpoint_paths[stage_num].name}")
        elif args.resume:
            # Specific checkpoint path only works for single stage
            print("Error: Specific checkpoint path (--resume <path>) only works for single stage training.")
            print("       For parallel training, use --resume none or --resume auto")
            sys.exit(1)
        else:
            # Interactive mode: prompt for each stage
            for stage_num in stages:
                checkpoint_paths[stage_num] = select_checkpoint(stage_num)

        # Launch parallel training
        train_parallel(
            stages=stages,
            checkpoint_paths=checkpoint_paths,
            timesteps=args.timesteps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            device=args.device,
            freeze_object=args.freeze_object,
        )
        return  # Exit after parallel training completes

    # Show training plan if multiple stages (sequential mode)
    if len(stages) > 1:
        print("\n" + "=" * 80)
        print(f"SEQUENTIAL TRAINING PLAN: {len(stages)} stages")
        print("=" * 80)
        for i, stage_num in enumerate(stages, 1):
            print(f"  {i}. {STAGE_INFO[stage_num]['name']}")
        print("=" * 80)
        print()

    # Train each stage sequentially
    for idx, stage_num in enumerate(stages, 1):
        if len(stages) > 1:
            print("\n" + "=" * 80)
            print(f"TRAINING STAGE {idx}/{len(stages)}: {STAGE_INFO[stage_num]['name']}")
            print("=" * 80)

        stage_info = STAGE_INFO[stage_num]

        # Handle checkpoint selection
        if args.resume == 'none':
            checkpoint_path = None
        elif args.resume == 'auto':
            checkpoints = find_checkpoints(stage_num)
            checkpoint_path = checkpoints[-1] if checkpoints else None
            if checkpoint_path:
                print(f"\n✓ Auto-selected latest checkpoint: {checkpoint_path.name}")
        elif args.resume:
            checkpoint_path = Path(args.resume)
            if not checkpoint_path.exists():
                print(f"Error: Checkpoint not found: {checkpoint_path}")
                sys.exit(1)
        else:
            # Interactive checkpoint selection
            checkpoint_path = select_checkpoint(stage_num)

        # Handle timestep selection
        if args.timesteps is None:
            timesteps = select_timesteps(
                default=stage_info["default_timesteps"],
                checkpoint_path=checkpoint_path
            )
        else:
            timesteps = args.timesteps

        # Run training
        train(
            stage_num=stage_num,
            checkpoint_path=checkpoint_path,
            timesteps=timesteps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            device=args.device,
            freeze_object=args.freeze_object,
        )

        if len(stages) > 1 and idx < len(stages):
            print(f"\n✓ Stage {idx}/{len(stages)} complete. Moving to next stage...\n")


if __name__ == "__main__":
    main()
