"""
Baseline execution policies.

These are simple heuristics for benchmarking RL agents.
"""

from abc import ABC, abstractmethod

import numpy as np


class BasePolicy(ABC):
    """Abstract base for execution policies."""

    @abstractmethod
    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return action given observation."""
        ...


class TWAPPolicy(BasePolicy):
    """
    Time-Weighted Average Price: execute uniformly over time.

    Classic benchmark - slices order equally across remaining steps.
    """

    def act(self, obs: np.ndarray) -> np.ndarray:
        remaining_qty, time_left, *_ = obs
        if time_left <= 0 or remaining_qty <= 0:
            return np.array([1.0], dtype=np.float32)

        # Estimate remaining steps (time_left is normalised)
        # Execute 1/remaining_steps of what's left
        steps_estimate = max(time_left * 20, 1)  # assuming 20 total steps
        action = 1.0 / steps_estimate
        return np.array([min(action, 1.0)], dtype=np.float32)


class RandomPolicy(BasePolicy):
    """Random execution - useful as lower bound."""

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.array([self._rng.uniform(0, 1)], dtype=np.float32)


class PanicPolicy(BasePolicy):
    """
    Conservative then panic: hold back, then rush at deadline.

    Common bad behavior that appears "safe" but causes slippage.
    """

    def __init__(self, panic_threshold: float = 0.2):
        self.panic_threshold = panic_threshold

    def act(self, obs: np.ndarray) -> np.ndarray:
        remaining_qty, time_left, *_ = obs

        if time_left <= self.panic_threshold:
            # Panic! Execute everything
            return np.array([1.0], dtype=np.float32)
        else:
            # Be conservative
            return np.array([0.05], dtype=np.float32)


def evaluate_policy(
    policy: BasePolicy,
    env,
    n_episodes: int = 100,
    seed: int = 42,
) -> dict:
    """
    Evaluate a policy over multiple episodes.

    Returns stats: mean/std/min/max/VaR of total costs.
    """
    costs = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = policy.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        costs.append(info["total_cost"])

    costs = np.array(costs)
    return {
        "mean": float(costs.mean()),
        "std": float(costs.std()),
        "min": float(costs.min()),
        "max": float(costs.max()),
        "median": float(np.median(costs)),
        "var_95": float(np.percentile(costs, 95)),  # 95% VaR
    }
