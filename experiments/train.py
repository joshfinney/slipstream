#!/usr/bin/env python3
"""
Train RL agent for trade execution.

Usage:
    uv run python experiments/train.py --timesteps 50000 --seed 42
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from env.execution_env import ExecutionEnv


def make_env(seed: int = 0, **kwargs) -> ExecutionEnv:
    """Factory for creating environments."""
    return ExecutionEnv(seed=seed, **kwargs)


def train(
    timesteps: int = 50_000,
    seed: int = 42,
    log_dir: str = "logs",
    model_dir: str = "models",
) -> None:
    """Train PPO agent and save model."""
    print(f"Training PPO for {timesteps:,} timesteps (seed={seed})")

    # Create dirs
    Path(log_dir).mkdir(exist_ok=True)
    Path(model_dir).mkdir(exist_ok=True)

    # Environment
    env = Monitor(make_env(seed=seed))
    eval_env = Monitor(make_env(seed=seed + 1000))

    # PPO with sensible defaults for this problem
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        seed=seed,
        tensorboard_log=log_dir,
    )

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=2000,
        n_eval_episodes=10,
        deterministic=True,
    )

    # Train
    model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=True)

    # Save final model
    model_path = Path(model_dir) / "ppo_execution_final"
    model.save(model_path)
    print(f"Model saved to {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train RL execution agent")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-dir", type=str, default="logs", help="TensorBoard log dir")
    parser.add_argument("--model-dir", type=str, default="models", help="Model save dir")
    args = parser.parse_args()

    train(
        timesteps=args.timesteps,
        seed=args.seed,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    main()
