"""
Backtest configuration dataclass.
Defines parameters for portfolio simulation.
"""
from dataclasses import dataclass

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
