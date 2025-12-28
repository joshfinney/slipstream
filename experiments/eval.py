#!/usr/bin/env python3
"""
Evaluate trained agent vs baselines.

Usage:
    uv run python experiments/eval.py --model models/ppo_execution_final.zip --episodes 500
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO

from agents.baselines import PanicPolicy, RandomPolicy, TWAPPolicy, evaluate_policy
from env.execution_env import ExecutionEnv, MarketParams


class SB3PolicyWrapper:
    """Wrap SB3 model to match our policy interface."""

    def __init__(self, model, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    def act(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return action

    def reset(self) -> None:
        pass


def evaluate_all(
    model_path: str | None = None,
    n_episodes: int = 500,
    seed: int = 42,
    output_dir: str = "reports",
) -> dict:
    """Evaluate all policies and generate comparison."""
    print(f"Evaluating over {n_episodes} episodes...")

    # Create environments for different regimes
    regimes = {
        "normal": MarketParams(volatility=0.02, temp_impact=0.1),
        "high_vol": MarketParams(volatility=0.05, temp_impact=0.1),
        "high_impact": MarketParams(volatility=0.02, temp_impact=0.3),
    }

    results = {}

    for regime_name, params in regimes.items():
        print(f"\n--- Regime: {regime_name} ---")
        env = ExecutionEnv(market_params=params, seed=seed)

        # Baselines
        policies = {
            "TWAP": TWAPPolicy(),
            "Random": RandomPolicy(seed=seed),
            "Panic": PanicPolicy(),
        }

        # Add trained model if available
        if model_path and Path(model_path).exists():
            model = PPO.load(model_path)
            policies["PPO"] = SB3PolicyWrapper(model)

        regime_results = {}
        for name, policy in policies.items():
            stats = evaluate_policy(policy, env, n_episodes=n_episodes, seed=seed)
            regime_results[name] = stats
            print(
                f"{name:10} | mean={stats['mean']:.4f} | std={stats['std']:.4f} | VaR95={stats['var_95']:.4f}"
            )

        results[regime_name] = regime_results

    # Save results
    Path(output_dir).mkdir(exist_ok=True)
    output_path = Path(output_dir) / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Generate markdown report
    generate_report(results, output_dir)

    return results


def generate_report(results: dict, output_dir: str) -> None:
    """Generate markdown evaluation report."""
    lines = [
        "# Slipstream Evaluation Results\n",
        "## Performance by Regime\n",
    ]

    for regime, policies in results.items():
        lines.append(f"### {regime.replace('_', ' ').title()}\n")
        lines.append("| Policy | Mean Cost | Std | Median | VaR 95% |")
        lines.append("|--------|-----------|-----|--------|---------|")

        # Sort by mean cost (lower is better)
        sorted_policies = sorted(policies.items(), key=lambda x: x[1]["mean"])
        for name, stats in sorted_policies:
            lines.append(
                f"| {name} | {stats['mean']:.4f} | {stats['std']:.4f} | "
                f"{stats['median']:.4f} | {stats['var_95']:.4f} |"
            )
        lines.append("")

    # Key observations
    lines.append("## Key Observations\n")
    lines.append("- **TWAP** provides a consistent baseline with moderate variance")
    lines.append("- **Panic** policy shows worst tail risk (high VaR) as expected")
    lines.append("- **RL agent** should outperform in mean cost while managing risk")
    lines.append("")

    report_path = Path(output_dir) / "results.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate execution policies")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=500, help="Number of eval episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="reports", help="Output directory")
    args = parser.parse_args()

    evaluate_all(
        model_path=args.model,
        n_episodes=args.episodes,
        seed=args.seed,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
