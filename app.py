#!/usr/bin/env python3
"""FastAPI backend — Stock Advisor Dashboard."""
import math, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import uvicorn
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(__file__))
import analyzer, data_fetcher, patterns as pat
from profit_guide import generate_profit_guide

app = FastAPI(title="Stock Advisor API", docs_url="/api/docs")


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe(val):
    if isinstance(val, (np.floating, np.integer)):
        val = val.item()
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val

def _series(index, values) -> list[dict]:
    out = []
    for t, v in zip(index, values):
        sv = _safe(v)
        if sv is not None:
            out.append({"time": str(t.date()), "value": round(sv, 6)})
    return out

def _guide_dict(g) -> dict:
    return {k: (_safe(v) if isinstance(v, float) else v) for k, v in vars(g).items()}


# ── main analysis endpoint ────────────────────────────────────────────────────

@app.get("/api/analyze/{ticker}")
async def analyze_ticker(
    ticker:   str,
    period:   str = Query("6mo"),
    interval: str = Query("1d"),
):
    ticker = ticker.upper().strip()
    try:
        df   = data_fetcher.fetch_ohlcv(ticker, period=period, interval=interval)
        info = data_fetcher.fetch_info(ticker)
        news = data_fetcher.fetch_news_sentiment(ticker)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Data fetch error: {e}")

    ind          = analyzer.compute_indicators(df)
    signals      = analyzer.identify_signals(ind, df)
    cs_patterns  = pat.detect_candlestick_patterns(df)
    ch_patterns  = pat.detect_chart_patterns(df)
    all_patterns = cs_patterns + ch_patterns
    advice       = analyzer.generate_advice(ind, signals, all_patterns)
    guide        = generate_profit_guide(ind, all_patterns, df, advice.score)

    # OHLCV
    ohlcv = [
        {"time": str(ts.date()),
         "open":  round(float(row["Open"]),  4),
         "high":  round(float(row["High"]),  4),
         "low":   round(float(row["Low"]),   4),
         "close": round(float(row["Close"]), 4),
         "volume": int(row["Volume"])}
        for ts, row in df.iterrows()
    ]

    close = df["Close"]

    # SMA series
    sma_series: dict = {}
    for w, key in [(20,"sma_20"),(50,"sma_50"),(200,"sma_200")]:
        if len(close) >= w:
            r = close.rolling(w).mean().dropna()
            sma_series[key] = _series(r.index, r.values)

    # Bollinger Bands
    bb_series: dict = {"upper":[],"middle":[],"lower":[]}
    if len(close) >= 20:
        mid, std = close.rolling(20).mean(), close.rolling(20).std()
        bb_series = {
            "upper":  _series((mid+2*std).dropna().index, (mid+2*std).dropna().values),
            "middle": _series(mid.dropna().index,          mid.dropna().values),
            "lower":  _series((mid-2*std).dropna().index, (mid-2*std).dropna().values),
        }

    # MACD
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    ml, sl = ema12 - ema26, (ema12-ema26).ewm(span=9, adjust=False).mean()
    macd_series = {
        "macd":      _series(ml.index,       ml.values),
        "signal":    _series(sl.index,        sl.values),
        "histogram": _series((ml-sl).index,  (ml-sl).values),
    }

    # RSI
    rsi_series: list = []
    if len(close) >= 14:
        d = close.diff()
        g = d.clip(lower=0).ewm(com=13, min_periods=14).mean()
        l = (-d.clip(upper=0)).ewm(com=13, min_periods=14).mean()
        rsi = (100 - 100/(1+g/l.replace(0,np.nan))).dropna()
        rsi_series = _series(rsi.index, rsi.values)

    return {
        "ticker":     ticker,
        "info":       {k: _safe(v) for k, v in info.items()},
        "indicators": {k: _safe(v) for k, v in vars(ind).items()},
        "signals":    [{"name":s.name,"direction":s.direction,"strength":s.strength,"detail":s.detail} for s in signals],
        "patterns":   [{"name":p.name,"pattern_type":p.pattern_type,"direction":p.direction,
                        "confidence":p.confidence,"description":p.description,"detected_at":p.detected_at} for p in all_patterns],
        "advice":     {"action":advice.action,"confidence":advice.confidence,
                       "score":advice.score,"rationale":advice.rationale,"risk_notes":advice.risk_notes},
        "profit_guide": _guide_dict(guide),
        "news":       news,
        "ohlcv":      ohlcv,
        "sma_series": sma_series,
        "bb_series":  bb_series,
        "macd_series":macd_series,
        "rsi_series": rsi_series,
    }


# ── market overview ───────────────────────────────────────────────────────────

_MARKET_SYMBOLS = {
    "SPY":    "S&P 500",
    "QQQ":    "NASDAQ 100",
    "DIA":    "Dow Jones",
    "^VIX":   "VIX",
    "BTC-USD":"Bitcoin",
    "GLD":    "Gold",
}

def _fetch_overview_one(sym: str) -> dict | None:
    try:
        hist = yf.Ticker(sym).history(period="2d", interval="1d")
        if len(hist) < 1:
            return None
        curr = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else curr
        return {"ticker": sym, "name": _MARKET_SYMBOLS.get(sym, sym),
                "price": round(curr, 2),
                "change_pct": round((curr/prev - 1)*100, 2)}
    except:
        return None

@app.get("/api/market-overview")
async def market_overview():
    results = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(_fetch_overview_one, s): s for s in _MARKET_SYMBOLS}
        for f in as_completed(futures):
            r = f.result()
            if r:
                results.append(r)
    order = list(_MARKET_SYMBOLS.keys())
    results.sort(key=lambda x: order.index(x["ticker"]) if x["ticker"] in order else 99)
    return results


# ── lightweight quote (watchlist refresh) ────────────────────────────────────

@app.get("/api/quote/{ticker}")
async def quick_quote(ticker: str):
    ticker = ticker.upper().strip()
    try:
        hist = yf.Ticker(ticker).history(period="2d", interval="1d")
        if len(hist) < 1:
            raise HTTPException(404, f"No data for {ticker}")
        curr = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else curr
        return {"ticker": ticker, "price": round(curr, 2),
                "change_pct": round((curr/prev-1)*100, 2)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ── professional trends ───────────────────────────────────────────────────────

_TREND_SYMBOLS = [
    "NVDA","AAPL","MSFT","GOOGL","AMZN","META","TSLA","AMD","PLTR","MSTR",
    "SPY","QQQ","BTC-USD","ETH-USD",
]

_SECTOR_ETFS = {
    "XLK":"Technology","XLF":"Financials","XLE":"Energy","XLV":"Healthcare",
    "XLC":"Comm.","XLY":"Cons. Discr.","XLP":"Cons. Staples","XLI":"Industrials",
    "XLRE":"Real Estate","XLU":"Utilities","XLB":"Materials",
}

def _fetch_trend_one(sym: str) -> dict | None:
    try:
        df  = data_fetcher.fetch_ohlcv(sym, period="1mo", interval="1d")
        ind = analyzer.compute_indicators(df)
        sigs = analyzer.identify_signals(ind, df)
        pats = pat.detect_candlestick_patterns(df, lookback=3)

        score = sum(
            ({"strong":3,"moderate":2,"weak":1}.get(s.strength,1)) *
            (1 if s.direction=="bullish" else -1)
            for s in sigs
        )
        vol_ratio = (float(df["Volume"].iloc[-1]) / float(ind.volume_sma_20)
                     if ind.volume_sma_20 and not math.isnan(ind.volume_sma_20) else 1.0)
        action = "BUY" if score>=2 else ("SELL" if score<=-2 else "HOLD")
        return {
            "ticker": sym,
            "price":      round(float(ind.price), 2),
            "change_pct": round(float(ind.price_change_pct), 2) if ind.price_change_pct and not math.isnan(ind.price_change_pct) else 0,
            "score":      score,
            "action":     action,
            "vol_ratio":  round(vol_ratio, 1),
            "rsi":        round(float(ind.rsi_14), 1) if ind.rsi_14 and not math.isnan(ind.rsi_14) else None,
            "patterns":   [p.name for p in pats[:2]],
            "interest":   abs(score) * max(vol_ratio, 0.5),
        }
    except:
        return None

def _fetch_sector_one(sym: str) -> dict | None:
    try:
        hist = yf.Ticker(sym).history(period="2d", interval="1d")
        if len(hist) < 1: return None
        curr = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist)>=2 else curr
        return {"ticker":sym,"name":_SECTOR_ETFS.get(sym,sym),
                "change_pct":round((curr/prev-1)*100, 2)}
    except:
        return None

@app.get("/api/trends")
async def get_trends():
    stocks, sectors = [], []
    with ThreadPoolExecutor(max_workers=8) as ex:
        sf = {ex.submit(_fetch_trend_one, s): s  for s in _TREND_SYMBOLS}
        se = {ex.submit(_fetch_sector_one, s): s for s in _SECTOR_ETFS}
        for f in as_completed(sf):
            r = f.result()
            if r: stocks.append(r)
        for f in as_completed(se):
            r = f.result()
            if r: sectors.append(r)

    stocks.sort(key=lambda x: x["interest"], reverse=True)
    sectors.sort(key=lambda x: x["change_pct"], reverse=True)
    return {"stocks": stocks[:10], "sectors": sectors}


# ── static SPA ────────────────────────────────────────────────────────────────

_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
