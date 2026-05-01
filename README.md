# 📈 Stock Price Predictor

A production-grade ML system for next-day stock price prediction using **Bidirectional LSTM**, **technical indicators**, and **FinBERT sentiment analysis**.

---

## Architecture

```
stock_predictor/
├── models/
│   └── lstm_model.py          # Bidirectional LSTM (TensorFlow/Keras)
├── utils/
│   ├── data_fetcher.py        # yfinance data pipeline
│   ├── indicators.py          # 30+ technical indicators (ta library)
│   └── sentiment.py           # FinBERT + NewsAPI sentiment engine
├── templates/
│   └── index.html             # Dark-mode trading dashboard
├── app.py                     # Flask REST API
├── cli.py                     # Command-line interface
├── requirements.txt
└── .env.example
```

---

## Features

### 🤖 LSTM Model
- **Bidirectional LSTM** — captures both forward and backward temporal patterns
- **32 input features** — OHLCV + technical indicators
- **Monte Carlo Dropout** for uncertainty estimation (confidence intervals)
- **Huber loss** for robustness against outliers
- **Early stopping + LR scheduling** for optimal training

### 📊 Technical Indicators
| Category | Indicators |
|----------|-----------|
| Trend | EMA(9/21/50/200), SMA(20/50), MACD, ADX |
| Momentum | RSI(14), Stochastic %K/%D, Williams %R |
| Volatility | Bollinger Bands, ATR, Keltner Channel |
| Volume | OBV, MFI |
| Price | Log Returns, Intraday Range, Volatility |

### 🧠 Sentiment Analysis (Bonus)
- **FinBERT** (ProsusAI/finbert) — BERT fine-tuned on financial text
- **NewsAPI** integration for live headlines
- Keyword fallback when model/API unavailable
- Per-headline scoring + aggregate signal (BULLISH/BEARISH/NEUTRAL)

---

## Setup

```bash
# 1. Clone and enter project
cd stock_predictor

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — add NEWSAPI_KEY for live news (optional)
```

---

## Usage

### CLI

```bash
# Train LSTM model for a ticker (downloads 3 years of data)
python cli.py train AAPL --epochs 100

# Predict next-day closing price
python cli.py predict AAPL

# Full analysis: prediction + technicals + sentiment
python cli.py analyze AAPL

# Other tickers
python cli.py analyze MSFT
python cli.py analyze NVDA --epochs 80
```

### Web Dashboard

```bash
# Start Flask server
python app.py

# Open browser
open http://localhost:5000
```

### REST API

```bash
# Get company info + current price
GET /api/stock/AAPL/info

# Price history with indicators (60 days)
GET /api/stock/AAPL/history?days=60

# Train model
POST /api/stock/AAPL/train
{"epochs": 80, "sequence_length": 60}

# Next-day prediction (requires trained model)
GET /api/stock/AAPL/predict

# Full analysis (all-in-one)
GET /api/stock/AAPL/analysis
```

---

## Model Performance

Typical results on S&P 500 large-caps:

| Metric | Target | Description |
|--------|--------|-------------|
| RMSE | < $3 | Price prediction error |
| MAE | < $2 | Mean absolute error |
| R² | > 0.95 | Variance explained |
| MAPE | < 2% | Mean absolute % error |
| Directional Accuracy | > 55% | Up/down direction correct |

> **Note**: Directional accuracy >55% is considered commercially meaningful. Past performance does not guarantee future results.

---

## How It Works

1. **Data Pipeline**: Downloads OHLCV data via `yfinance`, computes 30+ technical indicators
2. **Sequence Creation**: Builds 60-day sliding windows for LSTM input
3. **Model Training**: Bidirectional LSTM with dropout + batch normalization
4. **Prediction**: Monte Carlo Dropout produces price estimate + 90% confidence interval
5. **Trend Analysis**: Rule-based signal generation from indicator values
6. **Sentiment**: FinBERT classifies news headlines; aggregate score influences outlook

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. Stock price prediction is inherently uncertain. Do not use model outputs as financial advice. Always consult a licensed financial advisor before making investment decisions.
