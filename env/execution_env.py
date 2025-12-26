"""
ExecutionEnv: Gymnasium environment for optimal trade execution.

Models slicing a large order over time under market impact.
State captures inventory, time pressure, and market conditions.
Agent decides how aggressively to execute at each step.
"""

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class MarketParams:
    """Market microstructure parameters (Almgren-Chriss style)."""

    volatility: float = 0.02  # daily vol ~2%
    temp_impact: float = 0.1  # temporary impact coefficient η
    perm_impact: float = 0.01  # permanent impact coefficient γ
    impact_power: float = 0.5  # impact exponent β (square root law)


class ExecutionEnv(gym.Env):
    """
    Trade execution environment.

    State: [remaining_qty, time_left, last_return, realised_vol, impact_coeff]
    Action: fraction of remaining to execute ∈ [0, 1]
    Reward: -execution_cost - λ * risk_penalty
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        total_steps: int = 20,
        risk_aversion: float = 0.5,
        market_params: MarketParams | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.total_steps = total_steps
        self.risk_aversion = risk_aversion  # λ in reward
        self.params = market_params or MarketParams()

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # Episode state
        self._rng = np.random.default_rng(seed)
        self._reset_state()

    def _reset_state(self) -> None:
        """Initialize episode state."""
        self.remaining_qty = 1.0
        self.steps_left = self.total_steps
        self.price = 100.0  # arrival price
        self.arrival_price = self.price
        self.return_history: list[float] = []
        self.total_cost = 0.0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_state()
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Clip action to valid range
        exec_frac = float(np.clip(action[0], 0.0, 1.0))
        qty_to_exec = exec_frac * self.remaining_qty

        # Market dynamics: price moves before execution
        vol = self._current_vol()
        price_return = self._rng.normal(0, vol)
        self.price *= 1 + price_return
        self.return_history.append(price_return)

        # Market impact
        participation = qty_to_exec / max(self.steps_left, 1)
        temp_impact = self.params.temp_impact * (participation**self.params.impact_power)
        perm_impact = self.params.perm_impact * qty_to_exec

        # Execution cost (vs arrival price)
        exec_price = self.price * (1 + temp_impact)
        slippage = (exec_price - self.arrival_price) / self.arrival_price
        execution_cost = qty_to_exec * (slippage + temp_impact)

        # Permanent impact affects future prices
        self.price *= 1 + perm_impact

        # Update state
        self.remaining_qty -= qty_to_exec
        self.remaining_qty = max(0.0, self.remaining_qty)
        self.steps_left -= 1
        self.total_cost += execution_cost

        # Risk penalty: holding inventory in volatile market
        risk_penalty = (self.remaining_qty**2) * vol

        # Reward
        reward = -execution_cost - self.risk_aversion * risk_penalty

        # Termination
        terminated = self.steps_left <= 0 or self.remaining_qty < 1e-6
        truncated = False

        # Force liquidation at deadline
        if terminated and self.remaining_qty > 1e-6:
            forced_cost = self.remaining_qty * 0.05  # 5% penalty for leftover
            self.total_cost += forced_cost
            reward -= forced_cost

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        return np.array(
            [
                self.remaining_qty,
                self.steps_left / self.total_steps,
                self.return_history[-1] if self.return_history else 0.0,
                self._realised_vol(),
                self.params.temp_impact,  # observable impact regime
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        """Auxiliary info for logging."""
        return {
            "total_cost": self.total_cost,
            "remaining_qty": self.remaining_qty,
            "steps_left": self.steps_left,
            "price": self.price,
        }

    def _current_vol(self) -> float:
        """Current volatility (could add regime switching here)."""
        return self.params.volatility

    def _realised_vol(self) -> float:
        """Rolling realised volatility from recent returns."""
        if len(self.return_history) < 2:
            return self.params.volatility
        recent = self.return_history[-5:]  # last 5 periods
        return float(np.std(recent))

    def render(self) -> None:
        """Simple text render for debugging."""
        print(
            f"Step {self.total_steps - self.steps_left}/{self.total_steps} | "
            f"Remaining: {self.remaining_qty:.2%} | "
            f"Price: {self.price:.2f} | "
            f"Cost: {self.total_cost:.4f}"
        )
