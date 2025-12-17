import pandas as pd
import numpy as np
import vectorbt as vbt
from dataclasses import dataclass
from typing import Tuple, Dict
from pathlib import Path

from .base_strategy import BaseStrategy, StrategyConfig
import config

@dataclass
class DonchianClassicConfig(StrategyConfig):
    """
    Configuration for Donchian Pullback Classic strategy.
    
    Attributes:
        trend_period: Lookback period for SMA trend filter
        donchian_window: Lookback period for Donchian channel bands
    """
    trend_period: int = 200
    donchian_window: int = 20


class DonchianClassicStrategy(BaseStrategy):
    """
    Donchian Pullback entry with SMA trend filter.
    
    Entry conditions:
        - Close > Previous SMA (bullish trend)
        - Close < Previous Donchian Lower Band (pullback)
    
    Exit conditions:
        - Close > Previous Donchian Upper Band
    """
    
    def __init__(self, strategy_config: DonchianClassicConfig = None):
        """
        Initialize strategy with configuration.
        
        Args:
            strategy_config: DonchianClassicConfig instance, uses defaults if None
        """
        self.config = strategy_config or DonchianClassicConfig()
        self.name = "v1_classic"
    
    def get_name(self) -> str:
        """Return strategy name for results folder."""
        return self.name
    
    @classmethod
    def load_data(cls, ticker: str) -> pd.DataFrame:
        """
        Load price data from parquet file.
        
        Args:
            ticker: Stock symbol (e.g., 'SPY')
            
        Returns:
            DataFrame with OHLCV data
        """
        file_path = config.DATA_DIR / f"{ticker}_{config.INTERVAL}_RTH.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Data file {file_path} not found.")
        return pd.read_parquet(file_path)
    
    @staticmethod
    def _calculate_indicators_static(
        df: pd.DataFrame, 
        trend_period: int, 
        donchian_window: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate strategy indicators with given parameters.
        
        Args:
            df: DataFrame with OHLCV data
            trend_period: SMA lookback period
            donchian_window: Donchian channel lookback period
            
        Returns:
            Tuple of (sma, lower_band, upper_band) Series
        """
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # Major trend filter - shift(1) to avoid lookahead bias
        sma = vbt.MA.run(close, window=trend_period).ma.shift(1)
        
        # Donchian channel bands - shift(1) to avoid lookahead bias
        lower_band = low.rolling(window=donchian_window).min().shift(1)
        upper_band = high.rolling(window=donchian_window).max().shift(1)
        
        return sma, lower_band, upper_band
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate strategy indicators using instance config."""
        return self._calculate_indicators_static(
            df, self.config.trend_period, self.config.donchian_window
        )
    
    def get_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate entry and exit signals.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Tuple of (entries, exits) boolean Series
        """
        sma, lower_band, upper_band = self._calculate_indicators(df)
        close = df['Close']
        
        # Entry: Uptrend + Pullback to lower band
        entries = (close > sma) & (close < lower_band)
        
        # Exit: Price breaks above upper band
        exits = (close > upper_band)
        
        return entries, exits
    
    @classmethod
    def get_optimisable_params(cls) -> Dict[str, range]:
        """
        Return parameter ranges for optimisation.
        
        Returns:
            Dict with trend_period and donchian_window ranges
        """
        return {
            "trend_period": range(50, 201, 25),      # 50, 75, 100, 125, 150, 175, 200
            "donchian_window": range(10, 31, 5)      # 10, 15, 20, 25, 30
        }
    
    @classmethod
    def get_signals_for_params(
        cls, 
        df: pd.DataFrame, 
        trend_period: int, 
        donchian_window: int
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate signals for a specific parameter combination.
        
        Args:
            df: DataFrame with OHLCV data
            trend_period: SMA lookback period
            donchian_window: Donchian channel lookback period
            
        Returns:
            Tuple of (entries, exits) boolean Series
        """
        sma, lower_band, upper_band = cls._calculate_indicators_static(
            df, trend_period, donchian_window
        )
        close = df['Close']
        
        entries = (close > sma) & (close < lower_band)
        exits = (close > upper_band)
        
        return entries, exits
