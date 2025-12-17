import sys
import os
# add root dir to sys.path to see config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yfinance as yf 
import pandas as pd 
import config

def download_ticker(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """download data from yfinance for specified ticker, period, and interval"""
    print(f"Downloading {ticker}...")
    df = yf.download(
        tickers=ticker, 
        period=period, 
        interval=interval, 
        auto_adjust=True,
        prepost=False
    )
    return df

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    # flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # localise to NY tz and filter RTH
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
    else:
        df.index = df.index.tz_convert('America/New_York')
    df = df.between_time('09:30', '16:00').copy()
    
    # remove zero volume rows
    df = df[df['Volume'] > 0]

    # remove tz info for clean storage
    df.index = df.index.tz_localize(None)

    return df

def verify_data(df: pd.DataFrame): 
    if df.empty:
        raise ValueError("Downloaded data is empty.")
    
    # check 7 rows in a day
    first_date = df.index[0].date()
    count = len(df[df.index.date == first_date])
    if count < 7:
        raise ValueError(f"Insufficient data for {first_date}: expected 7, got {count}.")
    
def save_to_parquet(df: pd.DataFrame, ticker: str, interval: str):
    # create data dir if not exists 
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    # save df to filepath
    filename = f"{ticker}_{interval}_RTH.parquet"
    path = config.DATA_DIR / filename
    df.to_parquet(path)
    print(f"Saved to {path}")

def run_pipeline():
    for ticker in config.TICKERS:
        raw_df = download_ticker(ticker, config.PERIOD, config.INTERVAL)
        if raw_df.empty:
            print(f"No data for {ticker}")
            continue
            
        clean_df = process_data(raw_df)
        verify_data(clean_df)
        save_to_parquet(clean_df, ticker, config.INTERVAL)

if __name__ == "__main__":
    run_pipeline()