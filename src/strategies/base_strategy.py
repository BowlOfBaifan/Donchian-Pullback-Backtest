"""
Base classes for strategy implementation.
All strategies inherit from BaseStrategy.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class StrategyConfig:
    """
    Base configuration class for strategy parameters.
    Strategies create their own config classes inheriting from this.
    """
    pass


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Strategies implement:
    - get_signals(): Returns entry and exit signals
    - get_name(): Returns strategy identifier for results folder
    - get_optimizable_params(): Returns parameter ranges for optimization
    - get_signals_for_params(): Generates signals for given parameters
    """
    
    @abstractmethod
    def get_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate trading signals from price data.
        
        Args:
            df: DataFrame with OHLCV data (Open, High, Low, Close, Volume)
            
        Returns:
            Tuple of (entries, exits) boolean Series aligned with df index
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Return the strategy name for results folder naming.
        
        Returns:
            Strategy identifier string (e.g., 'v1_classic')
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_optimisable_params(cls) -> Dict[str, range]:
        """
        Return parameter names and their ranges for optimisation.
        
        Returns:
            Dict mapping parameter names to range objects
            Example: {"trend_period": range(50, 201, 25), "donchian_window": range(10, 31, 5)}
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_signals_for_params(cls, df: pd.DataFrame, **params) -> Tuple[pd.Series, pd.Series]:
        """
        Generate signals for a specific parameter combination.
        
        Args:
            df: DataFrame with OHLCV data
            **params: Parameter values (e.g., trend_period=100, donchian_window=20)
            
        Returns:
            Tuple of (entries, exits) boolean Series
        """
        pass
