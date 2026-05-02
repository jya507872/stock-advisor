"""Fetches stock market data from multiple sources."""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


def fetch_ohlcv(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV data for a ticker via Yahoo Finance."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol.")
    df.index = df.index.tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def fetch_info(ticker: str) -> dict:
    """Return basic company/instrument metadata."""
    stock = yf.Ticker(ticker)
    info = stock.info or {}
    return {
        "name": info.get("longName") or info.get("shortName", ticker),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "avg_volume": info.get("averageVolume"),
        "currency": info.get("currency", "USD"),
    }


def fetch_news_sentiment(ticker: str) -> list[dict]:
    """Fetch recent news headlines for a ticker."""
    stock = yf.Ticker(ticker)
    news = stock.news or []
    results = []
    for item in news[:10]:
        content = item.get("content", {})
        title = content.get("title", "") if isinstance(content, dict) else ""
        pub = content.get("pubDate", "") if isinstance(content, dict) else ""
        results.append({"title": title, "published": pub})
    return results
