# Slipstream Evaluation Results

## Performance by Regime

### Normal

| Policy | Mean Cost | Std | Median | VaR 95% |
|--------|-----------|-----|--------|---------|
| TWAP | 0.0249 | 0.0553 | 0.0252 | 0.1249 |
| PPO | 0.0293 | 0.0239 | 0.0285 | 0.0666 |
| Random | 0.0348 | 0.0262 | 0.0351 | 0.0785 |
| Panic | 0.0421 | 0.0602 | 0.0418 | 0.1462 |

### High Vol

| Policy | Mean Cost | Std | Median | VaR 95% |
|--------|-----------|-----|--------|---------|
| TWAP | 0.0294 | 0.1379 | 0.0258 | 0.2924 |
| PPO | 0.0311 | 0.0596 | 0.0285 | 0.1256 |
| Random | 0.0375 | 0.0652 | 0.0363 | 0.1412 |
| Panic | 0.0462 | 0.1505 | 0.0373 | 0.3169 |

### High Impact

| Policy | Mean Cost | Std | Median | VaR 95% |
|--------|-----------|-----|--------|---------|
| TWAP | 0.0590 | 0.0564 | 0.0594 | 0.1611 |
| PPO | 0.0790 | 0.0243 | 0.0784 | 0.1171 |
| Random | 0.0958 | 0.0305 | 0.0958 | 0.1433 |
| Panic | 0.1128 | 0.0628 | 0.1129 | 0.2208 |

## Key Observations

- **TWAP** provides a consistent baseline with moderate variance
- **Panic** policy shows worst tail risk (high VaR) as expected
- **RL agent** should outperform in mean cost while managing risk
