import sys
import os
import vectorbt as vbt
import pandas as pd

# Add the project root directory to sys.path so we can import 'config'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# import strategy module here
from strategies import v1_classic

def ensure_dir(directory):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def run_backtest(strategy_module, strategy_name, ticker="SPY"):
    """
    Accepts a strategy module (file), runs the simulation, and saves results.
    """
    print(f"Starting Backtest: {strategy_name}")

    # Delegate and run logic to strategy module
    # load data
    df = strategy_module.load_data(ticker)
    # Calculate indicators & signals
    sma, lower_band, upper_band = strategy_module.calculate_indicators(df)
    entries, exits = strategy_module.generate_signals(df['Close'], sma, lower_band, upper_band)

    # Run simulation
    print(f"Running simulation for {strategy_name}...")
    pf = vbt.Portfolio.from_signals(
        close=df['Close'],
        entries=entries,
        exits=exits,
        price=df['Open'].shift(-1),  # Execution at Next Open
        fees=0.001,                  # 0.1% Comms + Slippage
        sl_stop=0.02,                # 2% Hard Stop
        init_cash=10000,
        freq='1h'
    )

    # Save results to results directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "results", strategy_name)   # strategy name is file name
    ensure_dir(output_dir)

    # 1. Save full stats to text file
    stats_path = os.path.join(output_dir, "stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"STRATEGY REPORT: {strategy_name}\n")
        f.write(f"TICKER: {ticker}\n")
        f.write("="*50 + "\n")
        f.write(pf.stats().to_string()) # convert DataFrame to string
    print(f"Stats saved to: {stats_path}")

    # 2. Save equity plot as interactive HTML
    plot_path = os.path.join(output_dir, "equity.html")
    pf.plot().write_html(plot_path)
    print(f"Equity plot saved to: {plot_path}")

    # 3. Consule summary
    print("\n" + "="*40)
    print(f"QUICK RESULT: {strategy_name}")
    print(f"Total Return: {pf.total_return():.2%}")
    print(f"Sharpe Ratio: {pf.sharpe_ratio():.2f}")
    print(f"Max Drawdown: {pf.max_drawdown():.2%}")
    print("="*40 + "\n")

if __name__ == "__main__":
    # To run diff strategies, import strategy module and change function call here
    run_backtest(v1_classic, "v1_classic")