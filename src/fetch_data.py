"""
Data fetcher module.
Downloads, processes, and saves market data from yfinance.
"""

import yfinance as yf
import pandas as pd
import os

import config


def download_ticker(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Download data from yfinance for specified ticker, period, and interval.
    
    Args:
        ticker: Stock symbol (e.g., 'SPY')
        period: Time period string (e.g., '720d')
        interval: Data frequency (e.g., '1h', '1d')
        
    Returns:
        DataFrame with OHLCV data
    """
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
    """
    Process raw data: flatten columns, filter RTH, remove zero volume.
    
    Args:
        df: Raw DataFrame from yfinance
        
    Returns:
        Cleaned DataFrame with RTH data only
    """
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Localize to NY timezone and filter RTH
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
    else:
        df.index = df.index.tz_convert('America/New_York')
    df = df.between_time('09:30', '16:00').copy()

    # Remove zero volume rows
    df = df[df['Volume'] > 0]

    # Remove tz info for clean storage
    df.index = df.index.tz_localize(None)

    return df


def verify_data(df: pd.DataFrame) -> None:
    """
    Verify data quality.
    
    Args:
        df: DataFrame to verify
        
    Raises:
        ValueError: If data is empty or insufficient
    """
    if df.empty:
        raise ValueError("Downloaded data is empty.")

    # Check minimum 7 rows in first day
    first_date = df.index[0].date()
    count = len(df[df.index.date == first_date])
    if count < 7:
        raise ValueError(f"Insufficient data for {first_date}: expected 7, got {count}.")


def save_to_parquet(df: pd.DataFrame, ticker: str, interval: str) -> None:
    """
    Save DataFrame to parquet file in data directory.
    
    Args:
        df: DataFrame to save
        ticker: Stock symbol for filename
        interval: Data interval for filename
    """
    # Create data dir if not exists
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # Save df to filepath
    filename = f"{ticker}_{interval}_RTH.parquet"
    path = config.DATA_DIR / filename
    df.to_parquet(path)
    print(f"Saved to {path}")


def run_pipeline() -> None:
    """Run the complete data fetch pipeline for all configured tickers."""
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