import pandas as pd
import numpy as np
import vectorbt as vbt
from dataclasses import dataclass
from typing import Tuple, Dict
from pathlib import Path

from .base_strategy import BaseStrategy, StrategyConfig
import config

"""
Modification
This strategy extends the v1_classic Donchian Pullback strategy by implementing 
Asymmetric Donchian Windows. 
- 'entry_window': Lookback period for the Lower Channel (Entry condition)
- 'exit_window': Lookback period for the Upper Channel (Exit condition)

This allows decoupling the entry logic (pullback verification) from the exit logic 
(trend strength verification), allowing for faster exits (shorter window) to capture 
profits earlier or wider stops (longer window) to ride trends longer.
"""

@dataclass
class DonchianAsymmetricConfig(StrategyConfig):
    """
    Configuration for Donchian Pullback Asymmetric strategy.
    
    Attributes:
        trend_period: Lookback period for SMA trend filter
        entry_window: Lookback period for Donchian Lower Band (Entry)
        exit_window: Lookback period for Donchian Upper Band (Exit)
    """
    trend_period: int = 200
    entry_window: int = 20
    exit_window: int = 10


class DonchianAsymmetricStrategy(BaseStrategy):
    """
    Donchian Pullback strategy with asymmetric entry/exit windows.
    
    Entry conditions:
        - Close > Previous SMA (bullish trend)
        - Close < Previous Donchian Lower Band (entry_window)
    
    Exit conditions:
        - Close > Previous Donchian Upper Band (exit_window)
    """
    
    def __init__(self, strategy_config: DonchianAsymmetricConfig = None):
        """
        Initialize strategy with configuration.
        """
        self.config = strategy_config or DonchianAsymmetricConfig()
        self.name = "v2_asymmetric"
    
    def get_name(self) -> str:
        return self.name
    
    @classmethod
    def load_data(cls, ticker: str) -> pd.DataFrame:
        file_path = config.DATA_DIR / f"{ticker}_{config.INTERVAL}_RTH.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Data file {file_path} not found.")
        return pd.read_parquet(file_path)
    
    @staticmethod
    def _calculate_indicators_static(
        df: pd.DataFrame, 
        trend_period: int, 
        entry_window: int,
        exit_window: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate strategy indicators with given parameters.
        """
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # Major trend filter
        sma = vbt.MA.run(close, window=trend_period).ma.shift(1)
        
        # Donchian channel bands (Asymmetric)
        # Entry band uses entry_window
        lower_band = low.rolling(window=entry_window).min().shift(1)
        
        # Exit band uses exit_window
        upper_band = high.rolling(window=exit_window).max().shift(1)
        
        return sma, lower_band, upper_band
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        return self._calculate_indicators_static(
            df, 
            self.config.trend_period, 
            self.config.entry_window,
            self.config.exit_window
        )
    
    def get_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        sma, lower_band, upper_band = self._calculate_indicators(df)
        close = df['Close']
        
        # Entry: Uptrend + Pullback to lower band (entry_window)
        entries = (close > sma) & (close < lower_band)
        
        # Exit: Price breaks above upper band (exit_window)
        exits = (close > upper_band)
        
        return entries, exits
    
    @classmethod
    def get_optimisable_params(cls) -> Dict[str, range]:
        """
        Return parameter ranges for optimisation.
        """
        return {
            "trend_period": range(50, 201, 25),     # 100, 150, 200
            "entry_window": range(5, 31, 5),        # 15, 20, 25, 30
            "exit_window": range(5, 31, 5)           # 5, 10, 15, 20
        }
    
    @classmethod
    def get_signals_for_params(
        cls, 
        df: pd.DataFrame, 
        trend_period: int, 
        entry_window: int,
        exit_window: int
    ) -> Tuple[pd.Series, pd.Series]:
        
        sma, lower_band, upper_band = cls._calculate_indicators_static(
            df, trend_period, entry_window, exit_window
        )
        close = df['Close']
        
        entries = (close > sma) & (close < lower_band)
        exits = (close > upper_band)
        
        return entries, exits

"""
Findings 
- average win does not improve from v1_classic, still below 1 for most 
- optimised results show no difference between using v1_classics and v2_asymmetric 
"""