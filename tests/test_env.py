"""Tests for ExecutionEnv."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.execution_env import ExecutionEnv, MarketParams


class TestExecutionEnv:
    """Test environment invariants."""

    def test_reset_returns_valid_state(self):
        """Initial state should be valid."""
        env = ExecutionEnv(seed=42)
        obs, info = env.reset()

        assert obs.shape == (5,)
        assert env.observation_space.contains(obs)
        assert info["remaining_qty"] == 1.0
        assert info["steps_left"] == env.total_steps

    def test_action_clipping(self):
        """Actions outside [0,1] should be clipped."""
        env = ExecutionEnv(seed=42)
        env.reset()

        # Action > 1 should be clipped
        obs, _, _, _, info = env.step(np.array([1.5]))
        assert info["remaining_qty"] < 1.0  # executed something

        env.reset()
        # Action < 0 should be clipped to 0
        obs, _, _, _, info = env.step(np.array([-0.5]))
        assert info["remaining_qty"] == 1.0  # executed nothing

    def test_episode_terminates(self):
        """Episode should terminate when time runs out."""
        env = ExecutionEnv(total_steps=5, seed=42)
        env.reset()

        for _ in range(10):  # more than total_steps
            _, _, terminated, truncated, _ = env.step(np.array([0.1]))
            if terminated or truncated:
                break

        assert terminated or truncated

    def test_full_execution_terminates_early(self):
        """Executing everything should terminate episode."""
        env = ExecutionEnv(seed=42)
        env.reset()

        # Execute 100% on first step
        _, _, terminated, _, info = env.step(np.array([1.0]))
        assert terminated
        assert info["remaining_qty"] < 1e-6

    def test_reward_is_negative(self):
        """Rewards should be negative (cost minimisation)."""
        env = ExecutionEnv(seed=42)
        env.reset()

        total_reward = 0
        done = False
        while not done:
            _, reward, terminated, truncated, _ = env.step(np.array([0.1]))
            total_reward += reward
            done = terminated or truncated

        # Total reward should be negative (we're minimising cost)
        assert total_reward < 0

    def test_observation_space_bounds(self):
        """All observations should be within bounds."""
        env = ExecutionEnv(seed=42)
        env.reset()

        for _ in range(20):
            action = np.array([np.random.uniform(0, 1)])
            obs, _, terminated, truncated, _ = env.step(action)

            # Check observation is valid
            assert env.observation_space.contains(obs), f"Invalid obs: {obs}"

            if terminated or truncated:
                break

    def test_reproducibility(self):
        """Same seed should give same trajectory."""
        env1 = ExecutionEnv(seed=42)
        env2 = ExecutionEnv(seed=42)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

        actions = [np.array([0.1]), np.array([0.2]), np.array([0.3])]
        for action in actions:
            obs1, r1, _, _, _ = env1.step(action)
            obs2, r2, _, _, _ = env2.step(action)
            np.testing.assert_array_almost_equal(obs1, obs2)
            assert abs(r1 - r2) < 1e-10

    def test_market_params_affect_dynamics(self):
        """Different market params should affect outcomes."""
        low_impact = MarketParams(temp_impact=0.01)
        high_impact = MarketParams(temp_impact=0.5)

        env_low = ExecutionEnv(market_params=low_impact, seed=42)
        env_high = ExecutionEnv(market_params=high_impact, seed=42)

        env_low.reset(seed=42)
        env_high.reset(seed=42)

        # Execute same trajectory
        action = np.array([0.5])
        _, r_low, _, _, info_low = env_low.step(action)
        _, r_high, _, _, info_high = env_high.step(action)

        # High impact should cost more (more negative reward)
        assert r_low > r_high
