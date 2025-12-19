"""
Monte Carlo simulation module.
Provides trade bootstrapping, confidence intervals, and probability of ruin analysis.
"""

import os
from dataclasses import dataclass, field
from typing import Type, Dict, Optional, Tuple, List
import pandas as pd
import numpy as np
import vectorbt as vbt
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .backtester import BacktestConfig
from .strategies.base_strategy import BaseStrategy
import config


@dataclass
class MonteCarloConfig:
    """
    Configuration for Monte Carlo simulations.
    """
    n_simulations: int = 10000  # number of bootstrap iterations
    confidence_levels: Tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)   # percentiles to calculate
    random_seed: int = 42   # seed for reproducibility
    ruin_thresholds: Tuple[float, ...] = (0.15, 0.20, 0.25, 0.30)   # drawdown thresholds to calculate probability of ruin


@dataclass
class MonteCarloResults:
    """
    Results from Monte Carlo simulation.
    """
    n_simulations: int                      # number of simulations run
    n_trades: int                           # number of trades in original sample
    trade_returns: np.ndarray               # original trade returns
    simulated_final_returns: np.ndarray     # final return for each simulation
    simulated_max_drawdowns: np.ndarray     # max drawdown for each simulation
    simulated_sharpes: np.ndarray           # Sharpe ratio for each simulation
    equity_curves: np.ndarray               # matrix of equity curves (n_simulations x n_trades)
    confidence_intervals: Dict[str, Dict[float, float]] = field(default_factory=dict)  # Dict of [metric -> percentile values]
    probability_of_ruin: Dict[float, float] = field(default_factory=dict)              # Dict of [ruin threshold -> probability]


def extract_trade_returns(portfolio: vbt.Portfolio) -> np.ndarray:
    """
    Extract per-trade returns from a vectorbt Portfolio.
    
    Args:
        portfolio: vectorbt Portfolio object with completed trades

    Returns:
        Array of per-trade percentage returns
    """
    trades = portfolio.trades.records_readable      
    if len(trades) == 0:
        raise ValueError("No trades found in portfolio")
    
    trade_returns = trades['Return'].values     # Return = (Exit - Entry) / Entry
    return trade_returns


def run_trade_bootstrap(
    trade_returns: np.ndarray,
    mc_config: MonteCarloConfig,
    trading_days: int = 504  # Default: approx. 2 years of trading days for the backtest period
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap trades to generate synthetic equity curves.
    
    Args:
        trade_returns: Array of per-trade returns
        mc_config: Monte Carlo configuration
        trading_days: Number of trading days in the original backtest period
        
    Returns:
        Tuple of (equity_curves, final_returns, max_drawdowns, sharpes)
    """ 
    # random seed for reproducibility
    np.random.seed(mc_config.random_seed)   

    # Pre-allocate arrays
    n_trades = len(trade_returns)
    n_sims = mc_config.n_simulations
    equity_curves = np.zeros((n_sims, n_trades + 1))    # rows: simulation, column: acc value (starts at 1.0)
    final_returns = np.zeros(n_sims)
    max_drawdowns = np.zeros(n_sims)
    sharpes = np.zeros(n_sims)
    
    # Annualisation factor = trades per day scaled by full year trading days
    trades_per_year = (n_trades / trading_days) * 252 if trading_days > 0 else n_trades

    # for each simulation: 
    for i in range(n_sims):
        # Resample sample trades with replacement   
        resampled_returns = np.random.choice(trade_returns, size=n_trades, replace=True)
        
        # Build equity curve
        equity = np.empty(n_trades + 1)     # equity: array of (n_trades+1) equity values including 1.0
        equity[0] = 1.0
        equity[1:] = np.cumprod(1 + resampled_returns)
        equity_curves[i] = equity   # store equity curve for ith simulation
        
        # Final return
        final_returns[i] = equity[-1] - 1
        
        # Maximum drawdown
        running_max = np.maximum.accumulate(equity)         # array of running max equity values 
        drawdowns = (running_max - equity) / running_max    # array of drawdowns after each trade
        max_drawdowns[i] = np.max(drawdowns)                # store max drawdown for this simulation
        
        # Sharpe ratio: (mean return / std of returns) * sqrt(trades per year)
        std_returns = np.std(resampled_returns, ddof=1)     # unbiased sample std of returns
        if std_returns > 0:
            sharpes[i] = (np.mean(resampled_returns) / std_returns) * np.sqrt(trades_per_year)
        else:
            sharpes[i] = 0
    
    return equity_curves, final_returns, max_drawdowns, sharpes


def calculate_confidence_intervals(
    final_returns: np.ndarray,
    max_drawdowns: np.ndarray,
    sharpes: np.ndarray,
    confidence_levels: Tuple[float, ...]
) -> Dict[str, Dict[float, float]]:
    """
    Calculate confidence intervals for key metrics.
    
    Args:
        final_returns: Array of simulated final returns
        max_drawdowns: Array of simulated max drawdowns
        sharpes: Array of simulated Sharpe ratios
        confidence_levels: Percentiles to calculate
        
    Returns:
        Dict mapping metric names to percentile values
    """
    intervals = {}
    
    for name, values in [
        ('final_return', final_returns),
        ('max_drawdown', max_drawdowns),
        ('sharpe_ratio', sharpes)
    ]:
        intervals[name] = {
            level: np.percentile(values, level * 100)   
            for level in confidence_levels  
        }   
    
    return intervals


def calculate_probability_of_ruin(
    max_drawdowns: np.ndarray,
    thresholds: Tuple[float, ...]
) -> Dict[float, float]:
    """
    Calculate probability of hitting each drawdown threshold.
    
    Args:
        max_drawdowns: Array of simulated max drawdowns
        thresholds: Drawdown thresholds to check
        
    Returns:
        Dict mapping threshold to probability
    """
    return {
        threshold: np.mean(max_drawdowns >= threshold)
        for threshold in thresholds
    }


def run_monte_carlo_analysis(
    strategy_class: Type[BaseStrategy],
    ticker: str,
    strategy_params: Dict,
    backtest_config: Optional[BacktestConfig] = None,
    mc_config: Optional[MonteCarloConfig] = None
) -> MonteCarloResults:
    """
    Run full Monte Carlo analysis on a strategy.
    
    Args:
        strategy_class: Strategy class implementing BaseStrategy
        ticker: Stock symbol
        strategy_params: Dict of strategy parameters (e.g., {'trend_period': 200, 'donchian_window': 20})
        backtest_config: BacktestConfig instance, uses defaults if None
        mc_config: MonteCarloConfig instance, uses defaults if None
        
    Returns:
        MonteCarloResults with all simulation data
    """
    bt_config = backtest_config or BacktestConfig()
    mc_config = mc_config or MonteCarloConfig()
    
    print(f"Running Monte Carlo Analysis")
    print(f"Strategy: {strategy_class.__name__}")
    print(f"Parameters: {strategy_params}")
    print(f"Simulations: {mc_config.n_simulations}")
    
    # Load data and generate signals
    df = strategy_class.load_data(ticker)
    entries, exits = strategy_class.get_signals_for_params(df, **strategy_params)
    
    # Run backtest to get trade returns
    pf = vbt.Portfolio.from_signals(
        close=df['Close'],
        entries=entries,
        exits=exits,
        price=df['Open'].shift(-1),
        fees=bt_config.fees,
        sl_stop=bt_config.sl_stop,
        init_cash=bt_config.init_cash,
        freq=bt_config.freq
    )
    
    # Extract trade returns
    trade_returns = extract_trade_returns(pf)
    n_trades = len(trade_returns)
    print(f"Extracted {n_trades} trade returns")
    
    # Calculate trading days from the data for proper Sharpe annualisation
    trading_days = len(df['Close'].resample('D').last().dropna())
    
    # Run bootstrap
    print(f"Running {mc_config.n_simulations} bootstrap simulations...")
    equity_curves, final_returns, max_drawdowns, sharpes = run_trade_bootstrap(
        trade_returns, mc_config, trading_days
    )
    
    # Calculate confidence intervals
    confidence_intervals = calculate_confidence_intervals(
        final_returns, max_drawdowns, sharpes, mc_config.confidence_levels
    )
    
    # Calculate probability of ruin
    probability_of_ruin = calculate_probability_of_ruin(
        max_drawdowns, mc_config.ruin_thresholds
    )
    
    return MonteCarloResults(
        n_simulations=mc_config.n_simulations,
        n_trades=n_trades,
        trade_returns=trade_returns,
        simulated_final_returns=final_returns,
        simulated_max_drawdowns=max_drawdowns,
        simulated_sharpes=sharpes,
        equity_curves=equity_curves,
        confidence_intervals=confidence_intervals,
        probability_of_ruin=probability_of_ruin
    )

def plot_equity_fan_chart(
    results: MonteCarloResults,
    output_path: Path,
    percentiles: Tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)
) -> None:
    """
    Plot equity curve fan chart showing percentile bands.
    
    Args:
        results: MonteCarloResults object
        output_path: Path to save the plot
        percentiles: Percentiles to plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    n_trades = results.n_trades
    x = np.arange(1, n_trades + 1)
    
    # Calculate percentiles at each trade step
    percentile_curves = {}
    for p in percentiles:
        percentile_curves[p] = np.percentile(results.equity_curves, p * 100, axis=0)
    
    # Plot shaded regions
    colors = ['#e8f4ea', '#a8d5ba', '#52b788', '#a8d5ba', '#e8f4ea']
    
    # 5-95 percentile band
    ax.fill_between(x, percentile_curves[0.05], percentile_curves[0.95],
                   alpha=0.3, color='#2d6a4f', label='5th-95th percentile')
    
    # 25-75 percentile band
    ax.fill_between(x, percentile_curves[0.25], percentile_curves[0.75],
                   alpha=0.5, color='#40916c', label='25th-75th percentile')
    
    # Median line
    ax.plot(x, percentile_curves[0.50], color='#1b4332', linewidth=2, 
            label='Median (50th percentile)')
    
    # Original equity curve (if we start at 1.0 and compound)
    original_equity = np.cumprod(1 + results.trade_returns)
    ax.plot(x, original_equity, color='#d00000', linewidth=2, linestyle='--',
            label='Original sequence', alpha=0.8)
    
    ax.set_xlabel('Trade Number', fontsize=12)
    ax.set_ylabel('Equity (Starting at 1.0)', fontsize=12)
    ax.set_title(f'Monte Carlo Equity Fan Chart ({results.n_simulations:,} Simulations)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Equity fan chart saved to: {output_path}")

# Following 3 functions to generate visualisations
def plot_drawdown_distribution(
    results: MonteCarloResults,
    output_path: Path
) -> None:
    """
    Plot histogram of maximum drawdowns with ruin probability annotations.
    
    Args:
        results: MonteCarloResults object
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    ax.hist(results.simulated_max_drawdowns * 100, bins=50, color='#457b9d', 
            edgecolor='white', alpha=0.7)
    
    # Add vertical lines for original and ruin thresholds
    # Calculate original MDD correctly: (running_max - equity) / running_max
    original_equity = np.cumprod(1 + results.trade_returns)
    original_running_max = np.maximum.accumulate(original_equity)
    original_mdd = np.max((original_running_max - original_equity) / original_running_max)
    ax.axvline(x=original_mdd * 100, color='#d00000', linestyle='--', linewidth=2,
               label=f'Original MDD: {original_mdd:.1%}')
    
    # Add ruin thresholds
    for threshold, prob in results.probability_of_ruin.items():
        ax.axvline(x=threshold * 100, color='#e63946', linestyle=':', alpha=0.7)
        ax.annotate(f'{threshold:.0%} DD\nP={prob:.1%}', 
                   xy=(threshold * 100, ax.get_ylim()[1] * 0.9),
                   fontsize=9, ha='center', color='#e63946')
    
    ax.set_xlabel('Maximum Drawdown (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Maximum Drawdown Distribution ({results.n_simulations:,} Simulations)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Drawdown distribution saved to: {output_path}")


def plot_returns_distribution(
    results: MonteCarloResults,
    output_path: Path
) -> None:
    """
    Plot distribution of final returns with percentile markers.
    
    Args:
        results: MonteCarloResults object
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    ax.hist(results.simulated_final_returns * 100, bins=50, color='#52b788',
            edgecolor='white', alpha=0.7)
    
    # Add percentile lines
    ci = results.confidence_intervals['final_return']
    colors = {'5th': '#e63946', '25th': '#f4a261', '50th': '#2a9d8f', 
              '75th': '#f4a261', '95th': '#e63946'}
    
    for level, label in [(0.05, '5th'), (0.25, '25th'), (0.50, '50th'), 
                         (0.75, '75th'), (0.95, '95th')]:
        ax.axvline(x=ci[level] * 100, color=colors[label], linestyle='--', 
                   linewidth=1.5, alpha=0.8)
        ax.annotate(f'{label}: {ci[level]:.1%}', 
                   xy=(ci[level] * 100, ax.get_ylim()[1] * 0.95),
                   fontsize=9, rotation=90, va='top', ha='right')
    
    # Original return
    original_return = np.prod(1 + results.trade_returns) - 1
    ax.axvline(x=original_return * 100, color='#d00000', linestyle='-', linewidth=2,
               label=f'Original Return: {original_return:.1%}')
    
    ax.set_xlabel('Final Return (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Final Return Distribution ({results.n_simulations:,} Simulations)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Returns distribution saved to: {output_path}")


def plot_sharpe_distribution(
    results: MonteCarloResults,
    output_path: Path
) -> None:
    """
    Plot distribution of Sharpe ratios.
    
    Args:
        results: MonteCarloResults object
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    ax.hist(results.simulated_sharpes, bins=50, color='#8338ec',
            edgecolor='white', alpha=0.7)
    
    # Add percentile lines
    ci = results.confidence_intervals['sharpe_ratio']
    
    for level, label in [(0.05, '5th'), (0.50, 'Median'), (0.95, '95th')]:
        ax.axvline(x=ci[level], color='#e63946' if level != 0.50 else '#2a9d8f', 
                   linestyle='--', linewidth=1.5)
        ax.annotate(f'{label}: {ci[level]:.2f}', 
                   xy=(ci[level], ax.get_ylim()[1] * 0.95),
                   fontsize=9, rotation=90, va='top', ha='right')
    
    ax.set_xlabel('Sharpe Ratio', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Sharpe Ratio Distribution ({results.n_simulations:,} Simulations)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Sharpe distribution saved to: {output_path}")

# Report Generation
def generate_monte_carlo_report(
    results: MonteCarloResults,
    strategy_name: str,
    strategy_params: Dict,
    output_path: Path
) -> None:
    """
    Generate text report summarising Monte Carlo results.
    
    Args:
        results: MonteCarloResults object
        strategy_name: Name of the strategy
        strategy_params: Strategy parameters used
        output_path: Path to save the report
    """
    original_return = np.prod(1 + results.trade_returns) - 1
    original_equity = np.cumprod(1 + results.trade_returns)
    original_mdd = np.max((np.maximum.accumulate(original_equity) - original_equity) / 
                          np.maximum.accumulate(original_equity))
    
    ci_return = results.confidence_intervals['final_return']
    ci_mdd = results.confidence_intervals['max_drawdown']
    ci_sharpe = results.confidence_intervals['sharpe_ratio']
    
    report = f"""
================================================================================
                    MONTE CARLO SIMULATION REPORT
================================================================================

Strategy: {strategy_name}
Parameters: {strategy_params}
Simulations: {results.n_simulations:,}
Original Trades: {results.n_trades}

================================================================================
                         ORIGINAL RESULTS
================================================================================
Total Return:       {original_return:>10.2%}
Max Drawdown:       {original_mdd:>10.2%}

================================================================================
                    CONFIDENCE INTERVALS (Bootstrapped)
================================================================================

FINAL RETURN:
    5th Percentile:     {ci_return[0.05]:>10.2%}  (worst case)
    25th Percentile:    {ci_return[0.25]:>10.2%}
    50th Percentile:    {ci_return[0.50]:>10.2%}  (median)
    75th Percentile:    {ci_return[0.75]:>10.2%}
    95th Percentile:    {ci_return[0.95]:>10.2%}  (best case)

MAX DRAWDOWN:
    5th Percentile:     {ci_mdd[0.05]:>10.2%}  (best case)
    25th Percentile:    {ci_mdd[0.25]:>10.2%}
    50th Percentile:    {ci_mdd[0.50]:>10.2%}  (median)
    75th Percentile:    {ci_mdd[0.75]:>10.2%}
    95th Percentile:    {ci_mdd[0.95]:>10.2%}  (worst case)

SHARPE RATIO:
    5th Percentile:     {ci_sharpe[0.05]:>10.2f}  (worst case)
    25th Percentile:    {ci_sharpe[0.25]:>10.2f}
    50th Percentile:    {ci_sharpe[0.50]:>10.2f}  (median)
    75th Percentile:    {ci_sharpe[0.75]:>10.2f}
    95th Percentile:    {ci_sharpe[0.95]:>10.2f}  (best case)

================================================================================
                       PROBABILITY OF RUIN
================================================================================
"""
    for threshold, prob in sorted(results.probability_of_ruin.items()):
        report += f"    P(Max Drawdown >= {threshold:.0%}):  {prob:>10.2%}\n"
    
    report += f"""
================================================================================
                         INTERPRETATION
================================================================================

Based on {results.n_simulations:,} bootstrap simulations of {results.n_trades} trades:

1. RETURN STABILITY:
   - Your original return of {original_return:.2%} ranks at the 
     {np.mean(results.simulated_final_returns <= original_return) * 100:.0f}th percentile
   - 90% confidence interval: [{ci_return[0.05]:.2%}, {ci_return[0.95]:.2%}]

2. DRAWDOWN RISK:
   - Your original MDD of {original_mdd:.2%} ranks at the 
     {np.mean(results.simulated_max_drawdowns <= original_mdd) * 100:.0f}th percentile
   - There is a {results.probability_of_ruin.get(0.20, 0):.1%} probability of experiencing 
     a 20% drawdown with this strategy

3. SHARPE STABILITY:
   - 90% confidence interval: [{ci_sharpe[0.05]:.2f}, {ci_sharpe[0.95]:.2f}]
   - Median Sharpe: {ci_sharpe[0.50]:.2f}

================================================================================
"""
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Monte Carlo report saved to: {output_path}")


def save_monte_carlo_results(
    results: MonteCarloResults,
    strategy_name: str,
    strategy_params: Dict
) -> None:
    """
    Save all Monte Carlo results and visualisations.
    
    Args:
        results: MonteCarloResults object
        strategy_name: Name of the strategy
        strategy_params: Strategy parameters used
    """
    # Create output directory
    output_dir = config.BASE_DIR / "results" / strategy_name / "monte_carlo"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all outputs
    generate_monte_carlo_report(results, strategy_name, strategy_params, 
                               output_dir / "monte_carlo_summary.txt")
    plot_equity_fan_chart(results, output_dir / "equity_fan_chart.png")
    plot_drawdown_distribution(results, output_dir / "drawdown_distribution.png")
    plot_returns_distribution(results, output_dir / "returns_distribution.png")
    plot_sharpe_distribution(results, output_dir / "sharpe_distribution.png")
    
    print(f"\nAll Monte Carlo results saved to: {output_dir}")

if __name__ == "__main__":
    from .strategies.v1_classic import DonchianClassicStrategy
    
    # Run Monte Carlo analysis with optimal parameters
    results = run_monte_carlo_analysis(
        strategy_class=DonchianClassicStrategy,
        ticker="SPY",
        strategy_params={"trend_period": 200, "donchian_window": 20},
        mc_config=MonteCarloConfig(n_simulations=10000)
    )
    
    # Save all results and visualisations
    save_monte_carlo_results(
        results,
        strategy_name="v1_classic",
        strategy_params={"trend_period": 200, "donchian_window": 20}
    )
