"""
Strategy optimiser module.
Runs grid search over parameter combinations and outputs results.
"""

import os
from itertools import product
from typing import Type, Dict, Optional
import pandas as pd
import numpy as np
import vectorbt as vbt
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .backtest_config import BacktestConfig
from .strategies.base_strategy import BaseStrategy
import config


def run_optimisation(
    strategy_class: Type[BaseStrategy],
    ticker: str = "SPY",
    param_overrides: Optional[Dict[str, range]] = None,
    backtest_config: Optional[BacktestConfig] = None
) -> pd.DataFrame:
    """
    Run grid search optimisation for a strategy.
    
    Args:
        strategy_class: Strategy class implementing BaseStrategy
        ticker: Stock symbol to optimise on
        param_overrides: Optional custom parameter ranges (overrides strategy defaults)
        backtest_config: BacktestConfig instance, uses defaults if None
        
    Returns:
        DataFrame with results for all parameter combinations, ranked by Sharpe Ratio
    """
    bt_config = backtest_config or BacktestConfig()
    params = param_overrides or strategy_class.get_optimisable_params()
    
    # Load data once
    df = strategy_class.load_data(ticker)
    
    # Get parameter names and values
    param_names = list(params.keys())
    param_values = [list(params[name]) for name in param_names]
    
    print(f"Optimising {strategy_class.__name__} on {ticker}")
    print(f"Parameters: {param_names}")
    print(f"Total combinations: {np.prod([len(v) for v in param_values])}")
    
    results = []
    
    # Grid search
    for combo in product(*param_values):
        param_dict = dict(zip(param_names, combo))
        
        # Get signals for this parameter combination
        entries, exits = strategy_class.get_signals_for_params(df, **param_dict)
        
        # Run backtest
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
        
        # Collect metrics
        result = {
            **param_dict,
            'total_return': pf.total_return(),
            'sharpe_ratio': pf.sharpe_ratio(),
            'max_drawdown': pf.max_drawdown(),
            'win_rate': pf.trades.win_rate() if len(pf.trades.records) > 0 else 0,
            'num_trades': len(pf.trades.records)
        }
        results.append(result)
    
    # Create DataFrame and rank by Sharpe Ratio
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('sharpe_ratio', ascending=False).reset_index(drop=True)
    
    return results_df


def save_optimisation_results(
    results_df: pd.DataFrame,
    strategy_name: str,
    param_names: list
) -> None:
    """
    Save optimisation results to CSV and generate heatmap.
    
    Args:
        results_df: DataFrame with optimisation results
        strategy_name: Strategy name for output folder
        param_names: List of parameter names for heatmap axes
    """
    # Results saved to 'results/strategy_name' directory
    output_dir = config.BASE_DIR / "results" / strategy_name
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV
    csv_path = output_dir / "optimisation_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    # Generate heatmap (assumes 2 parameters)
    if len(param_names) == 2:
        heatmap_path = output_dir / "optimisation_heatmap.png"
        
        pivot = results_df.pivot(
            index=param_names[0],
            columns=param_names[1],
            values='sharpe_ratio'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'Sharpe Ratio'}
        )
        plt.title(f'{strategy_name} Optimisation: Sharpe Ratio')
        plt.xlabel(param_names[1])
        plt.ylabel(param_names[0])
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
        print(f"Heatmap saved to: {heatmap_path}")


def print_top_results(results_df: pd.DataFrame, n: int = 5) -> None:
    """Print top N results to console."""
    print(f"\n{'='*60}")
    print(f"TOP {n} RESULTS (by Sharpe Ratio)")
    print(f"{'='*60}")
    
    for i, row in results_df.head(n).iterrows():
        print(f"\nRank {i+1}:")
        for col, val in row.items():
            if isinstance(val, float):
                if col in ['total_return', 'max_drawdown', 'win_rate']:
                    print(f"  {col}: {val:.2%}")
                else:
                    print(f"  {col}: {val:.2f}")
            else:
                print(f"  {col}: {val}")


if __name__ == "__main__":
    from .strategies.v1_classic import DonchianClassicStrategy
    
    # Run optimisation
    results = run_optimisation(
        strategy_class=DonchianClassicStrategy,
        ticker="SPY"
    )
    
    # Save results and heatmap
    save_optimisation_results(
        results,
        strategy_name="v1_classic",
        param_names=["trend_period", "donchian_window"]
    )
    
    # Print top results
    print_top_results(results, n=5)
