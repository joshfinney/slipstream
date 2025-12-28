# Slipstream: RL for Optimal Trade Execution

> A minimal implementation exploring reinforcement learning for algorithmic execution.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## The Problem

When executing a large order, you face a tradeoff:
- **Execute fast** → High market impact, you move the price against yourself
- **Execute slow** → Price risk, the market may move against you while waiting

This is the classic **optimal execution problem** from market microstructure. This project frames it as an RL problem and trains agents to learn execution strategies.

## MDP Formulation

| Component | Definition |
|-----------|------------|
| **State** | `(remaining_qty, time_left, last_return, realised_vol, impact_coeff)` |
| **Action** | Fraction of remaining order to execute now ∈ [0, 1] |
| **Reward** | `-execution_cost - λ·risk_penalty` |
| **Transition** | Stochastic price (random walk) + Almgren-Chriss impact model |
| **Horizon** | Fixed T steps, or early termination when fully executed |

### Market Impact Model

Uses the [Almgren-Chriss](https://www.risk.net/journal-risk/2161150/optimal-execution-portfolio-transactions) framework:
- **Temporary impact**: `η · (participation_rate)^β` — price bounce from execution
- **Permanent impact**: `γ · quantity` — lasting price shift from information leakage

## Quick Start

```bash
# Clone and setup (using uv for speed)
git clone https://github.com/joshfinney/slipstream.git
cd slipstream
uv sync

# Run tests
uv run pytest

# Train an agent (~5 min on laptop)
uv run python experiments/train.py --timesteps 50000

# Evaluate against baselines
uv run python experiments/eval.py --model models/ppo_execution_final.zip

# View training curves
tensorboard --logdir logs/
```

## Project Structure

```
├── env/
│   └── execution_env.py    # Custom Gymnasium environment
├── agents/
│   └── baselines.py        # TWAP, Random, Panic policies
├── experiments/
│   ├── train.py            # PPO training script
│   └── eval.py             # Evaluation harness
├── tests/
│   └── test_env.py         # Environment invariants
└── reports/
    └── results.md          # Generated evaluation report
```

## Baselines

| Policy | Strategy |
|--------|----------|
| **TWAP** | Execute uniformly over time (industry standard) |
| **Random** | Random execution rate (lower bound) |
| **Panic** | Hold back, then rush at deadline (common mistake) |

## Evaluation

The eval harness tests across regimes:
- **Normal**: σ=2%, η=0.1
- **High Volatility**: σ=5%, η=0.1
- **High Impact**: σ=2%, η=0.3

Metrics: Mean cost, Std, Median, VaR 95%

## Stack

- **[Gymnasium](https://gymnasium.farama.org/)** — Environment API standard
- **[Stable-Baselines3](https://stable-baselines3.readthedocs.io/)** — PPO implementation
- **[uv](https://github.com/astral-sh/uv)** — Fast Python package manager
- **[Ruff](https://github.com/astral-sh/ruff)** — Linting & formatting

## References

- Almgren, R., & Chriss, N. (2001). *Optimal execution of portfolio transactions*. **The Journal of Risk**, 3(2), 5–39. **DOI:** 10.21314/JOR.2001.041. ([Risk.net](https://www.risk.net/journal-risk/2161150/optimal-execution-portfolio-transactions))
- Cartea, Á., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press. **ISBN:** 9781107091146. ([Cambridge Assets](https://assets.cambridge.org/97811070/91146/frontmatter/9781107091146_frontmatter.pdf))
