"""
Sentiment Analysis Engine
Uses FinBERT (ProsusAI/finbert) — a BERT model fine-tuned on financial text.
Falls back to keyword-based scoring when the model is unavailable.
Optional: NewsAPI integration for live headlines.
"""

import os
import re
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# ── FinBERT Loader ────────────────────────────────────────────────────────────

_finbert_pipeline = None


def _load_finbert():
    """Lazy-load FinBERT to avoid startup cost."""
    global _finbert_pipeline
    if _finbert_pipeline is not None:
        return _finbert_pipeline
    try:
        from transformers import pipeline
        _finbert_pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,
        )
        logger.info("FinBERT loaded successfully")
    except Exception as e:
        logger.warning(f"FinBERT unavailable ({e}). Using keyword fallback.")
        _finbert_pipeline = None
    return _finbert_pipeline


# ── Keyword Fallback ──────────────────────────────────────────────────────────

BULLISH_KEYWORDS = [
    "surge", "soar", "rally", "gain", "rise", "profit", "beat", "record",
    "growth", "bullish", "upgrade", "outperform", "buy", "strong", "exceed",
    "upside", "recovery", "rebound", "positive", "revenue", "earnings beat",
]
BEARISH_KEYWORDS = [
    "plunge", "crash", "fall", "drop", "loss", "miss", "decline", "weak",
    "bearish", "downgrade", "underperform", "sell", "debt", "warning",
    "risk", "concern", "recession", "layoff", "cut", "below", "shortfall",
]


def _keyword_sentiment(text: str) -> dict:
    """Simple keyword-based sentiment when FinBERT is unavailable."""
    text_lower = text.lower()
    bullish_count = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
    bearish_count = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)

    total = bullish_count + bearish_count
    if total == 0:
        return {"label": "neutral", "score": 0.0, "positive": 0.33, "negative": 0.33, "neutral": 0.34}

    pos = bullish_count / total
    neg = bearish_count / total
    neu = 0.1

    label = "positive" if pos > neg else ("negative" if neg > pos else "neutral")
    score = (pos - neg)  # Range -1 to 1

    return {
        "label": label,
        "score": float(score),
        "positive": float(pos),
        "negative": float(neg),
        "neutral": float(neu),
    }


# ── Main Sentiment Functions ──────────────────────────────────────────────────

def analyze_text_sentiment(text: str) -> dict:
    """
    Analyze sentiment of a single text using FinBERT or keyword fallback.
    
    Returns:
        dict with label, score (-1 to 1), and class probabilities
    """
    if not text or len(text.strip()) < 5:
        return {"label": "neutral", "score": 0.0, "positive": 0.33, "negative": 0.33, "neutral": 0.34}

    model = _load_finbert()

    if model:
        try:
            results = model(text[:512])[0]  # FinBERT max 512 tokens
            probs = {r["label"]: r["score"] for r in results}
            positive = probs.get("positive", 0)
            negative = probs.get("negative", 0)
            neutral = probs.get("neutral", 0)
            score = positive - negative

            if positive > negative and positive > neutral:
                label = "positive"
            elif negative > positive and negative > neutral:
                label = "negative"
            else:
                label = "neutral"

            return {
                "label": label,
                "score": float(score),
                "positive": float(positive),
                "negative": float(negative),
                "neutral": float(neutral),
            }
        except Exception as e:
            logger.warning(f"FinBERT inference failed: {e}")

    return _keyword_sentiment(text)


def analyze_headlines(headlines: list[dict]) -> dict:
    """
    Analyze a list of news headlines and compute aggregate sentiment.
    
    Args:
        headlines: List of dicts with 'title' and optional 'description' keys
    
    Returns:
        Aggregate sentiment dict with per-headline details
    """
    if not headlines:
        return {
            "aggregate_score": 0.0,
            "aggregate_label": "neutral",
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "headlines": [],
        }

    results = []
    for h in headlines:
        text = h.get("title", "") + " " + h.get("description", "")
        sentiment = analyze_text_sentiment(text.strip())
        results.append({
            "title": h.get("title", ""),
            "source": h.get("source", ""),
            "published": h.get("published", ""),
            "url": h.get("url", ""),
            **sentiment,
        })

    scores = np.array([r["score"] for r in results])
    avg_score = float(np.mean(scores))

    bullish = sum(1 for r in results if r["label"] == "positive")
    bearish = sum(1 for r in results if r["label"] == "negative")
    neutral = sum(1 for r in results if r["label"] == "neutral")

    if avg_score > 0.1:
        aggregate_label = "positive"
    elif avg_score < -0.1:
        aggregate_label = "negative"
    else:
        aggregate_label = "neutral"

    return {
        "aggregate_score": avg_score,
        "aggregate_label": aggregate_label,
        "bullish_count": bullish,
        "bearish_count": bearish,
        "neutral_count": neutral,
        "total_headlines": len(results),
        "headlines": results,
    }


def fetch_news_headlines(ticker: str, company_name: str = "") -> list[dict]:
    """
    Fetch recent news headlines using NewsAPI.
    Requires NEWSAPI_KEY environment variable.
    Falls back to mock headlines if unavailable.
    """
    api_key = os.getenv("NEWSAPI_KEY")

    if api_key:
        try:
            import requests
            query = company_name or ticker
            url = (
                f"https://newsapi.org/v2/everything?"
                f"q={query}&language=en&sortBy=publishedAt"
                f"&pageSize=20&apiKey={api_key}"
            )
            resp = requests.get(url, timeout=10)
            data = resp.json()
            articles = data.get("articles", [])
            return [
                {
                    "title": a.get("title", ""),
                    "description": a.get("description", ""),
                    "source": a.get("source", {}).get("name", ""),
                    "published": a.get("publishedAt", ""),
                    "url": a.get("url", ""),
                }
                for a in articles
                if a.get("title")
            ]
        except Exception as e:
            logger.warning(f"NewsAPI fetch failed: {e}")

    # Demo headlines when no API key
    logger.info("Using demo headlines (set NEWSAPI_KEY for live news)")
    return [
        {"title": f"{ticker} reports strong quarterly earnings, beats analyst estimates", "source": "Demo", "published": "2024-01-15", "url": ""},
        {"title": f"Analysts upgrade {ticker} stock to 'Buy' citing growth potential", "source": "Demo", "published": "2024-01-14", "url": ""},
        {"title": f"Market volatility affects {ticker} amid global economic uncertainty", "source": "Demo", "published": "2024-01-13", "url": ""},
        {"title": f"{ticker} announces strategic partnership to expand market share", "source": "Demo", "published": "2024-01-12", "url": ""},
        {"title": f"Investors cautious as {ticker} faces supply chain challenges", "source": "Demo", "published": "2024-01-11", "url": ""},
    ]


def get_sentiment_score_for_prediction(
    ticker: str, company_name: str = ""
) -> dict:
    """
    Full pipeline: fetch news → analyze → return aggregate score for model input.
    """
    headlines = fetch_news_headlines(ticker, company_name)
    sentiment = analyze_headlines(headlines)

    # Normalize score to 0-1 for model feature input
    normalized = (sentiment["aggregate_score"] + 1) / 2

    return {
        **sentiment,
        "normalized_score": float(normalized),
        "signal": (
            "BULLISH" if sentiment["aggregate_score"] > 0.15
            else ("BEARISH" if sentiment["aggregate_score"] < -0.15 else "NEUTRAL")
        ),
    }
