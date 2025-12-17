"""
Base classes for strategy implementation.
All strategies inherit from BaseStrategy.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple
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
