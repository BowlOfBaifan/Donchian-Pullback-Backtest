import os 
import sys
import pandas as pd
import numpy as np
import vectorbt as vbt 

# Add project root to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config

# TODO: INTERVAL should be dynamic based on data fetched
def load_data(ticker: str): 
    """Load specific data file from data directory."""
    file_path = config.DATA_DIR / f"{ticker}_{config.INTERVAL}_RTH.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Data file {file_path} not found.")
    df = pd.read_parquet(file_path)
    return df 

def calculate_indicators(df: pd.DataFrame):
    """
    Strategy: Donchian Pullback
    Future Changes: TP and SL levels based on ATR
    Technical indicators calculated using vectorisation
    """
    close = df['Close']
    high = df['High']
    low = df['Low']

    # TODO: Parameters Settings - for optimisation later 
    trend_period = 200    
    donchian_window = 20   

    # Major trend filter 
    # current bar SMA calculated using prev 200 bar data - avoid current bar from influencing decision
    sma = vbt.MA.run(close, window=trend_period).ma.shift(1)    

    # support and resistance using donchian channel 
    # shift(1) shifts current bar data forward in time by 1 row, prevent lookahead bias
    lower_band = low.rolling(window=donchian_window).min().shift(1)     
    upper_band = high.rolling(window=donchian_window).max().shift(1)

    return sma, lower_band, upper_band      # return each as pd.Series

def generate_signals(close, sma, lower_band, upper_band):
    """
    Entries:
      - Condition A: current Close > prev 200 SMA
      - Condition B: current Close < prev 20 Lower Band 
    Exits:
      - Condition: Close > Upper Band (Previous 20)
    """
    entries = (close > sma) & (close < lower_band)  # enter signal generated only after current bar close, to buy at next bar open
    exits = (close > upper_band)
    
    return entries, exits