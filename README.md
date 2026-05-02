# 📈 Stock Advisor — Live Dashboard

A real-time stock market analysis dashboard with technical indicators, candlestick & chart pattern recognition, profit guides, and professional trader trend tracking.

## Features
- **Live Dashboard** — auto-refreshing market overview, sector performance, trending setups
- **Technical Analysis** — SMA, EMA, MACD, RSI, Bollinger Bands, ATR, OBV
- **Pattern Recognition** — 15+ candlestick patterns + Double Top/Bottom, H&S, triangles
- **Profit Guide** — entry, stop-loss, T1/T2/T3 targets, R:R ratios, trade plan
- **Market Intelligence** — high-activity setups across top equities & crypto
- **Interactive Charts** — TradingView Lightweight Charts with toggleable overlays

## Quick Start (local)

```bash
pip install -r requirements.txt
python3 app.py
# Open http://localhost:8000
```

Or double-click **Launch Stock Advisor.command** in Finder.

## Stack
| Layer | Tech |
|---|---|
| Backend | Python · FastAPI · uvicorn |
| Data | yfinance (Yahoo Finance) |
| Charts | TradingView Lightweight Charts v4 |
| Frontend | Vanilla JS · CSS Grid |

## Deployment
- **Frontend**: deploy `static/` to Netlify (connected to this repo)
- **Backend**: deploy to [Render.com](https://render.com) (free tier) or Railway
  - Build command: `pip install -r requirements.txt`
  - Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
