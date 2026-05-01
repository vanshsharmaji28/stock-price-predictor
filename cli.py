"""
CLI — Stock Price Predictor
Usage:
    python cli.py train AAPL --epochs 100
    python cli.py predict AAPL
    python cli.py analyze AAPL
    python cli.py backtest AAPL
"""

import argparse
import json
import sys
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from models.lstm_model import StockLSTMModel
from utils.data_fetcher import (
    fetch_stock_data,
    get_stock_info,
    get_available_features,
    train_test_split_temporal,
)
from utils.indicators import get_trend_signals, get_support_resistance
from utils.sentiment import get_sentiment_score_for_prediction


def cmd_train(args):
    ticker = args.ticker.upper()
    print(f"\n{'='*60}")
    print(f"  Training LSTM for: {ticker}")
    print(f"{'='*60}")

    df = fetch_stock_data(ticker, period="3y")
    feature_cols = get_available_features(df)
    train_df, test_df = train_test_split_temporal(df, test_ratio=0.15)

    print(f"  Training rows : {len(train_df)}")
    print(f"  Test rows     : {len(test_df)}")
    print(f"  Features      : {len(feature_cols)}")
    print(f"  Epochs        : {args.epochs}")
    print()

    model = StockLSTMModel(
        sequence_length=args.seq_len,
        units=[128, 64],
        dropout_rate=0.2,
    )

    history = model.fit(
        train_df,
        feature_cols=feature_cols,
        target_col="Close",
        epochs=args.epochs,
        batch_size=32,
    )

    metrics = model.evaluate(test_df)

    model_path = os.path.join("saved_models", ticker)
    model.save(model_path)

    print(f"\n{'─'*60}")
    print(f"  TEST SET METRICS")
    print(f"{'─'*60}")
    print(f"  RMSE                : ${metrics['rmse']:.2f}")
    print(f"  MAE                 : ${metrics['mae']:.2f}")
    print(f"  R²                  : {metrics['r2']:.4f}")
    print(f"  MAPE                : {metrics['mape']:.2f}%")
    print(f"  Directional Accuracy: {metrics['directional_accuracy']:.1f}%")
    print(f"{'─'*60}")
    print(f"  Model saved to: {model_path}")
    print()


def cmd_predict(args):
    ticker = args.ticker.upper()
    model_path = os.path.join("saved_models", ticker)

    if not os.path.exists(model_path):
        print(f"  No trained model found. Run: python cli.py train {ticker}")
        sys.exit(1)

    model = StockLSTMModel.load(model_path)
    df = fetch_stock_data(ticker, period="1y")
    result = model.predict_next_day(df)

    direction_emoji = "📈" if result["direction"] == "UP" else "📉"

    print(f"\n{'='*60}")
    print(f"  NEXT-DAY PREDICTION: {ticker} {direction_emoji}")
    print(f"{'='*60}")
    print(f"  Last Close      : ${result['last_close']:.2f}")
    print(f"  Predicted Price : ${result['predicted_price']:.2f}")
    print(f"  Change          : {result['change_pct']:+.2f}%")
    print(f"  Confidence Band : ${result['confidence_lower']:.2f} – ${result['confidence_upper']:.2f}")
    print(f"  Direction       : {result['direction']}")
    print()


def cmd_analyze(args):
    ticker = args.ticker.upper()

    print(f"\n{'='*60}")
    print(f"  FULL ANALYSIS: {ticker}")
    print(f"{'='*60}")

    info = get_stock_info(ticker)
    print(f"  {info.get('name', ticker)} | {info.get('sector', 'N/A')}")
    print()

    df = fetch_stock_data(ticker, period="2y")
    signals = get_trend_signals(df)
    levels = get_support_resistance(df)
    sentiment = get_sentiment_score_for_prediction(ticker, info.get("name", ticker))

    print(f"  TECHNICAL SIGNALS")
    print(f"{'─'*60}")
    for name, (value, strength, desc) in signals.items():
        emoji = {"STRONG_BULLISH": "🟢🟢", "BULLISH": "🟢", "NEUTRAL": "⚪", "BEARISH": "🔴", "STRONG_BEARISH": "🔴🔴", "WARNING": "🟡", "INFO": "ℹ️"}.get(strength, "⚪")
        print(f"  {emoji} {name:<18} {value:<20} {desc}")

    print(f"\n  PRICE LEVELS")
    print(f"{'─'*60}")
    print(f"  Current Price   : ${levels['current_price']:.2f}")
    print(f"  Resistance 1    : ${levels['resistance_1']:.2f}")
    print(f"  Pivot           : ${levels['pivot']:.2f}")
    print(f"  Support 1       : ${levels['support_1']:.2f}")
    print(f"  52W High / Low  : ${levels['52w_high']:.2f} / ${levels['52w_low']:.2f}")

    print(f"\n  SENTIMENT ANALYSIS ({sentiment['total_headlines']} headlines)")
    print(f"{'─'*60}")
    signal_emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}[sentiment["signal"]]
    print(f"  {signal_emoji} Aggregate Score : {sentiment['aggregate_score']:+.3f}")
    print(f"  Signal           : {sentiment['signal']}")
    print(f"  Bullish / Bearish / Neutral : {sentiment['bullish_count']} / {sentiment['bearish_count']} / {sentiment['neutral_count']}")

    # Prediction if model available
    model_path = os.path.join("saved_models", ticker)
    if os.path.exists(model_path):
        model = StockLSTMModel.load(model_path)
        pred = model.predict_next_day(df)
        direction_emoji = "📈" if pred["direction"] == "UP" else "📉"
        print(f"\n  LSTM PREDICTION {direction_emoji}")
        print(f"{'─'*60}")
        print(f"  Predicted Next Close : ${pred['predicted_price']:.2f} ({pred['change_pct']:+.2f}%)")
        print(f"  Confidence Band      : ${pred['confidence_lower']:.2f} – ${pred['confidence_upper']:.2f}")
    else:
        print(f"\n  💡 No trained model. Run: python cli.py train {ticker}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Stock Price Predictor CLI")
    subparsers = parser.add_subparsers(dest="command")

    # train
    train_p = subparsers.add_parser("train", help="Train LSTM model")
    train_p.add_argument("ticker", help="Stock ticker symbol")
    train_p.add_argument("--epochs", type=int, default=80)
    train_p.add_argument("--seq-len", type=int, default=60)
    train_p.set_defaults(func=cmd_train)

    # predict
    pred_p = subparsers.add_parser("predict", help="Predict next-day price")
    pred_p.add_argument("ticker")
    pred_p.set_defaults(func=cmd_predict)

    # analyze
    analyze_p = subparsers.add_parser("analyze", help="Full technical + sentiment analysis")
    analyze_p.add_argument("ticker")
    analyze_p.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
