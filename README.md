# Donchian Pullback Strategy

This project implements and backtests a trend-following mean reversion strategy on SPY.

## Project Navigation

### 1. Viewing Results
The comprehensive analysis is located in:
`results/v1_classic/Report.md`

**What Report.md contains:**
*   **Strategy Logic:** detailed breakdown of entry/exit rules and the underlying hypothesis.
*   **Performance Metrics:** Sharpe Ratio, Win Rate, and Drawdown figures.
*   **Monte Carlo Analysis:** Statistical validation of the strategy's robustness (10,000 simulations).
*   **Conclusion:** Critical assessment of strategy weaknesses and future improvements.

### 2. Viewing Code
The source code is located in the `src/` directory:
*   `src/strategies/`: Strategy logic implementation.
*   `src/backtester.py`: Backtesting engine.
*   `src/monte_carlo.py`: Statistical simulation module.
