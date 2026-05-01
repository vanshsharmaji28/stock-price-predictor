"""
Technical Indicators Calculator
Computes RSI, MACD, Bollinger Bands, ATR, OBV, and more using the `ta` library.
"""

import pandas as pd
import numpy as np
import ta
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
import logging

logger = logging.getLogger(__name__)


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a comprehensive set of technical indicators to a OHLCV DataFrame.

    Input columns required: Open, High, Low, Close, Volume
    
    Groups of indicators added:
        1. Trend:     EMA(9,21,50,200), SMA(20,50), MACD, ADX
        2. Momentum:  RSI(14), Stochastic %K/%D, Williams %R
        3. Volatility:Bollinger Bands, ATR, Keltner Channel
        4. Volume:    OBV, MFI
        5. Price:     Log Returns, Intraday Range %, Gap %
    """
    df = df.copy()

    # ── Trend Indicators ──────────────────────────────────────────────────────
    for period in [9, 21, 50, 200]:
        df[f"EMA_{period}"] = EMAIndicator(df["Close"], window=period).ema_indicator()
    for period in [20, 50]:
        df[f"SMA_{period}"] = SMAIndicator(df["Close"], window=period).sma_indicator()

    macd = MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    adx = ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
    df["ADX"] = adx.adx()
    df["ADX_Pos"] = adx.adx_pos()
    df["ADX_Neg"] = adx.adx_neg()

    # ── Momentum Indicators ───────────────────────────────────────────────────
    df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()

    stoch = StochasticOscillator(df["High"], df["Low"], df["Close"], window=14, smooth_window=3)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    df["Williams_R"] = WilliamsRIndicator(df["High"], df["Low"], df["Close"], lbp=14).williams_r()

    # ── Volatility Indicators ─────────────────────────────────────────────────
    bb = BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Width"] = bb.bollinger_wband()
    df["BB_Percent"] = bb.bollinger_pband()

    df["ATR"] = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()

    kc = KeltnerChannel(df["High"], df["Low"], df["Close"], window=20)
    df["KC_Upper"] = kc.keltner_channel_hband()
    df["KC_Lower"] = kc.keltner_channel_lband()
    df["KC_Middle"] = kc.keltner_channel_mband()

    # ── Volume Indicators ─────────────────────────────────────────────────────
    df["OBV"] = OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
    df["MFI"] = MFIIndicator(df["High"], df["Low"], df["Close"], df["Volume"], window=14).money_flow_index()

    # ── Price-Derived Features ────────────────────────────────────────────────
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Return_2d"] = df["Close"].pct_change(2)
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Intraday_Range_Pct"] = (df["High"] - df["Low"]) / df["Close"] * 100
    df["Gap_Pct"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1) * 100
    df["Close_vs_SMA20"] = (df["Close"] - df["SMA_20"]) / df["SMA_20"] * 100
    df["Close_vs_SMA50"] = (df["Close"] - df["SMA_50"]) / df["SMA_50"] * 100

    # ── Volatility Regimes ────────────────────────────────────────────────────
    df["Volatility_20"] = df["Log_Return"].rolling(20).std() * np.sqrt(252)
    df["Volatility_60"] = df["Log_Return"].rolling(60).std() * np.sqrt(252)

    # Drop NaN rows from indicator warmup periods
    initial_len = len(df)
    df.dropna(inplace=True)
    logger.info(f"Indicators computed. Dropped {initial_len - len(df)} warmup rows. Remaining: {len(df)}")

    return df


def get_trend_signals(df: pd.DataFrame) -> dict:
    """
    Generate human-readable trend signals from the latest indicator values.
    
    Returns a dict of signal names → (signal_value, strength, description)
    """
    latest = df.iloc[-1]
    signals = {}

    # RSI
    rsi = latest["RSI"]
    if rsi > 70:
        signals["RSI"] = ("OVERBOUGHT", "BEARISH", f"RSI={rsi:.1f} — overbought, potential reversal")
    elif rsi < 30:
        signals["RSI"] = ("OVERSOLD", "BULLISH", f"RSI={rsi:.1f} — oversold, potential bounce")
    else:
        signals["RSI"] = ("NEUTRAL", "NEUTRAL", f"RSI={rsi:.1f} — neutral momentum")

    # MACD
    macd_hist = latest["MACD_Hist"]
    macd_prev = df["MACD_Hist"].iloc[-2]
    if macd_hist > 0 and macd_prev < 0:
        signals["MACD"] = ("BULLISH_CROSS", "STRONG_BULLISH", "MACD bullish crossover")
    elif macd_hist < 0 and macd_prev > 0:
        signals["MACD"] = ("BEARISH_CROSS", "STRONG_BEARISH", "MACD bearish crossover")
    elif macd_hist > 0:
        signals["MACD"] = ("BULLISH", "BULLISH", f"MACD histogram positive ({macd_hist:.3f})")
    else:
        signals["MACD"] = ("BEARISH", "BEARISH", f"MACD histogram negative ({macd_hist:.3f})")

    # Bollinger Bands
    bb_pct = latest["BB_Percent"]
    if bb_pct > 1.0:
        signals["BB"] = ("ABOVE_UPPER", "BEARISH", "Price above upper Bollinger Band")
    elif bb_pct < 0.0:
        signals["BB"] = ("BELOW_LOWER", "BULLISH", "Price below lower Bollinger Band")
    else:
        signals["BB"] = ("INSIDE", "NEUTRAL", f"Price at {bb_pct*100:.0f}% of Bollinger Band width")

    # EMA Trend
    close = latest["Close"]
    ema9 = latest["EMA_9"]
    ema21 = latest["EMA_21"]
    ema50 = latest["EMA_50"]
    if close > ema9 > ema21 > ema50:
        signals["EMA_Trend"] = ("STRONG_UPTREND", "STRONG_BULLISH", "Price > EMA9 > EMA21 > EMA50")
    elif close < ema9 < ema21 < ema50:
        signals["EMA_Trend"] = ("STRONG_DOWNTREND", "STRONG_BEARISH", "Price < EMA9 < EMA21 < EMA50")
    else:
        signals["EMA_Trend"] = ("MIXED", "NEUTRAL", "Mixed EMA alignment")

    # ADX Trend Strength
    adx = latest["ADX"]
    if adx > 40:
        signals["ADX"] = ("VERY_STRONG", "INFO", f"ADX={adx:.1f} — very strong trend")
    elif adx > 25:
        signals["ADX"] = ("TRENDING", "INFO", f"ADX={adx:.1f} — trending market")
    else:
        signals["ADX"] = ("RANGING", "INFO", f"ADX={adx:.1f} — ranging/choppy market")

    # Volume MFI
    mfi = latest["MFI"]
    if mfi > 80:
        signals["MFI"] = ("OVERBOUGHT", "BEARISH", f"MFI={mfi:.1f} — volume overbought")
    elif mfi < 20:
        signals["MFI"] = ("OVERSOLD", "BULLISH", f"MFI={mfi:.1f} — volume oversold")
    else:
        signals["MFI"] = ("NEUTRAL", "NEUTRAL", f"MFI={mfi:.1f} — neutral volume flow")

    # Volatility Regime
    vol_20 = latest["Volatility_20"]
    vol_60 = latest["Volatility_60"]
    if vol_20 > vol_60 * 1.3:
        signals["Volatility"] = ("HIGH", "WARNING", f"20d vol ({vol_20:.1%}) > 60d vol — elevated risk")
    else:
        signals["Volatility"] = ("NORMAL", "INFO", f"Annualized vol: {vol_20:.1%}")

    return signals


def get_support_resistance(df: pd.DataFrame, lookback: int = 60) -> dict:
    """Calculate key support and resistance levels."""
    recent = df.tail(lookback)
    close = df["Close"].iloc[-1]

    pivot = (recent["High"].max() + recent["Low"].min() + recent["Close"].iloc[-1]) / 3
    r1 = 2 * pivot - recent["Low"].min()
    s1 = 2 * pivot - recent["High"].max()

    return {
        "pivot": float(pivot),
        "resistance_1": float(r1),
        "support_1": float(s1),
        "52w_high": float(df["High"].tail(252).max()),
        "52w_low": float(df["Low"].tail(252).min()),
        "current_price": float(close),
        "bb_upper": float(df["BB_Upper"].iloc[-1]),
        "bb_lower": float(df["BB_Lower"].iloc[-1]),
    }
