"""
Stock Data Fetcher
Downloads OHLCV data via yfinance and applies technical indicators.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from utils.indicators import compute_all_indicators

logger = logging.getLogger(__name__)


FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "EMA_9", "EMA_21", "EMA_50", "SMA_20", "SMA_50",
    "MACD", "MACD_Signal", "MACD_Hist",
    "RSI", "Stoch_K", "Stoch_D", "Williams_R",
    "BB_Upper", "BB_Lower", "BB_Width", "BB_Percent",
    "ATR", "OBV", "MFI",
    "Log_Return", "Return_5d",
    "Intraday_Range_Pct", "Volatility_20",
    "Close_vs_SMA20", "Close_vs_SMA50",
    "ADX",
]


def fetch_stock_data(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download stock OHLCV data and compute technical indicators.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'MSFT')
        period: Data period ('1y', '2y', '5y', 'max')
        interval: Candle interval ('1d', '1wk', '1mo')
    
    Returns:
        DataFrame with OHLCV + all technical indicator columns
    """
    logger.info(f"Fetching {ticker} data (period={period}, interval={interval})")

    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol.")

    # Clean column names
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)

    logger.info(f"Downloaded {len(df)} rows for {ticker}")

    # Compute all technical indicators
    df = compute_all_indicators(df)

    return df


def get_stock_info(ticker: str) -> dict:
    """Fetch company metadata from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", None),
            "52w_high": info.get("fiftyTwoWeekHigh", None),
            "52w_low": info.get("fiftyTwoWeekLow", None),
            "avg_volume": info.get("averageVolume", 0),
            "description": info.get("longBusinessSummary", "")[:300],
        }
    except Exception as e:
        logger.warning(f"Could not fetch info for {ticker}: {e}")
        return {"name": ticker}


def get_available_features(df: pd.DataFrame) -> list:
    """Return only the feature columns that exist in the DataFrame."""
    return [col for col in FEATURE_COLUMNS if col in df.columns]


def train_test_split_temporal(
    df: pd.DataFrame, test_ratio: float = 0.15
) -> tuple:
    """
    Split data temporally (no shuffling) for time-series evaluation.
    Returns (train_df, test_df)
    """
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def generate_price_history(df: pd.DataFrame, days: int = 90) -> list:
    """Convert recent price history to JSON-serializable format."""
    recent = df.tail(days)
    records = []
    for date, row in recent.iterrows():
        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
            "rsi": round(float(row["RSI"]), 1) if "RSI" in row else None,
            "macd": round(float(row["MACD"]), 4) if "MACD" in row else None,
            "bb_upper": round(float(row["BB_Upper"]), 2) if "BB_Upper" in row else None,
            "bb_lower": round(float(row["BB_Lower"]), 2) if "BB_Lower" in row else None,
        })
    return records
