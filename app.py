"""
Stock Price Predictor — Flask REST API
Endpoints for prediction, training, trend analysis, and sentiment.
"""

import os
import json
import logging
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from models.lstm_model import StockLSTMModel
from utils.data_fetcher import (
    fetch_stock_data,
    get_stock_info,
    get_available_features,
    train_test_split_temporal,
    generate_price_history,
)
from utils.indicators import get_trend_signals, get_support_resistance
from utils.sentiment import get_sentiment_score_for_prediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

MODEL_DIR = "saved_models"
_model_cache: dict[str, StockLSTMModel] = {}


# ── Helper ─────────────────────────────────────────────────────────────────────

def get_or_load_model(ticker: str) -> StockLSTMModel | None:
    if ticker in _model_cache:
        return _model_cache[ticker]
    path = os.path.join(MODEL_DIR, ticker)
    if os.path.exists(path):
        try:
            model = StockLSTMModel.load(path)
            _model_cache[ticker] = model
            return model
        except Exception as e:
            logger.warning(f"Could not load model for {ticker}: {e}")
    return None


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


@app.route("/api/stock/<ticker>/info")
def stock_info(ticker: str):
    """Return company metadata and latest price."""
    ticker = ticker.upper()
    try:
        info = get_stock_info(ticker)
        df = fetch_stock_data(ticker, period="5d")
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        change = float(latest["Close"] - prev["Close"])
        change_pct = float(change / prev["Close"] * 100)
        return jsonify({
            **info,
            "ticker": ticker,
            "current_price": round(float(latest["Close"]), 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "volume": int(latest["Volume"]),
            "last_updated": df.index[-1].strftime("%Y-%m-%d"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/stock/<ticker>/history")
def price_history(ticker: str):
    """Return OHLCV + indicator data for charting."""
    ticker = ticker.upper()
    days = int(request.args.get("days", 90))
    try:
        df = fetch_stock_data(ticker, period="1y")
        history = generate_price_history(df, days=days)
        return jsonify({"ticker": ticker, "history": history})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/stock/<ticker>/train", methods=["POST"])
def train_model(ticker: str):
    """Train (or retrain) the LSTM model for a ticker."""
    ticker = ticker.upper()
    body = request.get_json(silent=True) or {}
    epochs = body.get("epochs", 80)
    sequence_length = body.get("sequence_length", 60)

    try:
        logger.info(f"Starting training for {ticker}")

        df = fetch_stock_data(ticker, period="3y")
        feature_cols = get_available_features(df)
        train_df, test_df = train_test_split_temporal(
            df,
            test_ratio=0.15
        )

        model = StockLSTMModel(
            sequence_length=sequence_length,
            units=[128, 64],
            dropout_rate=0.2,
        )

        # Force build before training
        model.build_model()

        history = model.fit(
            train_df,
            feature_cols=feature_cols,
            target_col="Close",
            epochs=epochs,
            batch_size=32,
        )

        metrics = model.evaluate(test_df)

        model.save(
            os.path.join(MODEL_DIR, ticker)
        )

        _model_cache[ticker] = model

        return jsonify({
            "ticker": ticker,
            "status": "trained",
            "epochs_run": len(history.get("loss", [])),
            "final_train_loss": round(
                history["loss"][-1], 6
            ),
            "final_val_loss": round(
                history["val_loss"][-1], 6
            ),
            "test_metrics": metrics,
            "features_used": len(feature_cols),
            "model_params": model.model.count_params()
        })

    except Exception as e:
        logger.exception(
            f"Training failed for {ticker}"
        )
        return jsonify({
            "error": str(e)
        }), 500
@app.route("/api/stock/<ticker>/predict")
def predict(ticker: str):
    """Return next-day price prediction."""
    ticker = ticker.upper()
    model = get_or_load_model(ticker)
    if model is None:
        return jsonify({
            "error": f"No trained model for {ticker}. POST /api/stock/{ticker}/train first."
        }), 404

    try:
        df = fetch_stock_data(ticker, period="1y")
        prediction = model.predict_next_day(df)
        return jsonify({"ticker": ticker, "prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stock/<ticker>/analysis")
def full_analysis(ticker: str):
    """
    Full analysis: prediction + trend signals + support/resistance + sentiment.
    """
    ticker = ticker.upper()
    try:
        df = fetch_stock_data(ticker, period="2y")
        info = get_stock_info(ticker)
        trend_signals = get_trend_signals(df)
        levels = get_support_resistance(df)
        sentiment = get_sentiment_score_for_prediction(ticker, info.get("name", ticker))
        history = generate_price_history(df, days=60)

        # Prediction (if model exists)
        prediction = None
        model = get_or_load_model(ticker)
        if model:
            try:
                prediction = model.predict_next_day(df)
            except Exception as e:
                logger.warning(f"Prediction failed: {e}")

        # Serialize trend signals
        signals_out = {
            k: {"value": v[0], "strength": v[1], "description": v[2]}
            for k, v in trend_signals.items()
        }

        return jsonify({
            "ticker": ticker,
            "info": info,
            "prediction": prediction,
            "trend_signals": signals_out,
            "support_resistance": levels,
            "sentiment": sentiment,
            "price_history": history,
            "timestamp": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        logger.exception("Full analysis failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
