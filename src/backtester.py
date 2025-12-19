"""
Backtester module.
Runs simulations using strategy objects and saves results.
"""

import os
import vectorbt as vbt
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

from .strategies.base_strategy import BaseStrategy
import config


@dataclass
class BacktestConfig:
    """
    Configuration for backtesting parameters.
    
    Attributes:
        fees: Transaction costs as decimal (0.001 = 0.1%)
        sl_stop: Stop loss as decimal (0.02 = 2%)
        init_cash: Initial portfolio cash amount
        freq: Data frequency string (e.g., '1h', '1d')
    """
    fees: float = 0.001        # 0.1% Comms + Slippage
    sl_stop: float = 0.02      # 2% Hard Stop
    init_cash: float = 10000
    freq: str = '1h'


def ensure_dir(directory: Path) -> None:
    """
    Creates a directory if it doesn't exist.
    
    Args:
        directory: Path to directory to create
    """
    if not directory.exists():
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def run_backtest(
    strategy: BaseStrategy,
    ticker: str = "SPY",
    backtest_config: BacktestConfig = None
) -> None:
    """
    Run backtest simulation with given strategy and save results.
    
    Args:
        strategy: BaseStrategy instance with get_signals() and get_name() methods
        ticker: Stock symbol to backtest (default: 'SPY')
        backtest_config: BacktestConfig instance, uses defaults if None
    """
    # Use default config if not provided
    bt_config = backtest_config or BacktestConfig()
    strategy_name = strategy.get_name()
    
    print(f"Starting Backtest: {strategy_name}")
    print(f"Config: fees={bt_config.fees}, sl_stop={bt_config.sl_stop}, "
          f"init_cash={bt_config.init_cash}, freq={bt_config.freq}")

    # Load data
    df = strategy.load_data(ticker)
    
    # Get signals from strategy
    entries, exits = strategy.get_signals(df)

    # Run simulation
    print(f"Running simulation for {strategy_name}...")
    pf = vbt.Portfolio.from_signals(
        close=df['Close'],
        entries=entries,
        exits=exits,
        price=df['Open'].shift(-1),    # Execution at next open
        fees=bt_config.fees,
        sl_stop=bt_config.sl_stop,
        init_cash=bt_config.init_cash,
        freq=bt_config.freq
    )

    # Save results to results directory
    output_dir = config.BASE_DIR / "results" / strategy_name / "base_results"
    ensure_dir(output_dir)

    # Save full stats to text file
    stats_path = output_dir / "stats.txt"
    with open(stats_path, "w") as f:
        f.write(f"STRATEGY REPORT: {strategy_name}\n")
        f.write(f"TICKER: {ticker}\n")
        f.write(f"CONFIG: fees={bt_config.fees}, sl_stop={bt_config.sl_stop}, "
                f"init_cash={bt_config.init_cash}, freq={bt_config.freq}\n")
        f.write("=" * 50 + "\n")
        f.write(pf.stats().to_string())
    print(f"Stats saved to: {stats_path}")

    # Save equity plot as interactive HTML
    plot_path = output_dir / "equity.html"
    pf.plot().write_html(str(plot_path))
    print(f"Equity plot saved to: {plot_path}")

    # Console summary
    print("\n" + "=" * 40)
    print(f"Strategy: {strategy_name}")
    print(f"Total Return: {pf.total_return():.2%}")
    print(f"Sharpe Ratio: {pf.sharpe_ratio():.2f}")
    print(f"Max Drawdown: {pf.max_drawdown():.2%}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    # Import and instantiate strategy with custom or default config
    from .strategies.v1_classic import DonchianClassicStrategy, DonchianClassicConfig
    
    # To run with custom parameters:
    # custom_strategy_config = DonchianClassicConfig(trend_period=150, donchian_window=15)
    # strategy = DonchianClassicStrategy(custom_strategy_config)
    
    # To run with default parameters:
    strategy = DonchianClassicStrategy()
    
    # To run using custom backtest config:
    # custom_bt_config = BacktestConfig(fees=0.0005, sl_stop=0.02, init_cash=50000)
    # run_backtest(strategy, ticker="SPY", backtest_config=custom_bt_config)
    
    # To run using default backtest config:
    run_backtest(strategy, ticker="SPY")