import os 
from pathlib import Path

# define paths
BASE_DIR = Path(__file__).parent 
DATA_DIR = BASE_DIR / 'data'

# Settings 
TICKERS = ["SPY"]
INTERVAL = "1h"
PERIOD = "720d" # max allowed for hourly data from yfinance


