"""Profit maximisation guide: entry, stop-loss, targets, R/R, trade plan."""
from dataclasses import dataclass, field
import math
import numpy as np
import pandas as pd


@dataclass
class ProfitGuide:
    direction: str          # "long" | "short" | "neutral"
    strategy: str
    entry: float
    entry_note: str
    stop: float
    stop_pct: float         # % from entry (negative = below for long)
    t1: float
    t2: float
    t3: float
    rr1: float
    rr2: float
    rr3: float
    risk_per_share: float
    trade_plan: list[str] = field(default_factory=list)
    tips: list[str] = field(default_factory=list)


def _safe(v) -> float | None:
    return None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)


def _fv(v: float) -> str:
    if v >= 1e6: return f"{v/1e6:.1f}M"
    return f"{int(v):,}"


def generate_profit_guide(ind, patterns: list, df: pd.DataFrame, score: int) -> ProfitGuide:
    price  = float(ind.price)
    atr    = _safe(ind.atr_14)    or (price * 0.02)
    rsi    = _safe(ind.rsi_14)    or 50.0
    sma20  = _safe(ind.sma_20)
    sma200 = _safe(ind.sma_200)
    bb_low = _safe(ind.bb_lower)
    bb_up  = _safe(ind.bb_upper)
    vol20  = _safe(ind.volume_sma_20)

    direction = "long" if score > 0 else ("short" if score < 0 else "neutral")

    # ── Entry ──────────────────────────────────────────────────────────────
    entry = price
    entry_note = "Current market price"
    if direction == "long":
        if sma20 and price > sma20 * 1.06:
            entry_note = f"Price extended — ideally wait for pullback to SMA20 ~${sma20:.2f}"
        elif bb_low and price <= bb_low * 1.015:
            entry_note = "Near lower Bollinger Band — high-value entry zone"
        elif rsi < 35:
            entry_note = "Oversold territory — favourable risk/reward entry"
    elif direction == "short":
        if sma20 and price < sma20 * 0.94:
            entry_note = f"Price extended — ideally wait for bounce to SMA20 ~${sma20:.2f}"
        elif rsi > 65:
            entry_note = "Overbought — confirm entry with a bearish close"

    # ── Stop loss ──────────────────────────────────────────────────────────
    recent_low  = float(df["Low"].tail(20).min())
    recent_high = float(df["High"].tail(20).max())

    if direction == "long":
        stop = max(entry - 1.5 * atr, recent_low * 0.995)
        if entry - stop < 0.5 * atr:
            stop = entry - 0.5 * atr
        stop_pct = (stop - entry) / entry * 100          # negative
    elif direction == "short":
        stop = min(entry + 1.5 * atr, recent_high * 1.005)
        if stop - entry < 0.5 * atr:
            stop = entry + 0.5 * atr
        stop_pct = (stop - entry) / entry * 100          # positive
    else:
        stop = entry * 0.97
        stop_pct = -3.0

    risk = abs(entry - stop)

    # ── Price targets ──────────────────────────────────────────────────────
    if direction == "long":
        t1, t2, t3 = entry + 1.5*risk, entry + 2.5*risk, entry + 4.0*risk
        for p in patterns:
            if "Double Bottom" in p.name:
                bottom = float(df["Low"].tail(60).min())
                neck   = sma20 or entry * 1.03
                pt     = neck + (neck - bottom)
                if pt > t2: t3 = pt; break
            elif "Inv. Head" in p.name:
                t3 = max(t3, entry + 3.5 * risk); break
            elif "Ascending Triangle" in p.name:
                if bb_up: t3 = max(t3, bb_up + risk); break
        if sma200 and sma200 > entry:
            t3 = max(t3, sma200)

    elif direction == "short":
        t1, t2, t3 = entry - 1.5*risk, entry - 2.5*risk, entry - 4.0*risk
        for p in patterns:
            if "Double Top" in p.name:
                top  = float(df["High"].tail(60).max())
                neck = sma20 or entry * 0.97
                pt   = neck - (top - neck)
                if pt < t2: t3 = pt; break
            elif "Head & Shoulders" in p.name:
                t3 = min(t3, entry - 3.5 * risk); break
        if sma200 and sma200 < entry:
            t3 = min(t3, sma200)
    else:
        t1, t2, t3 = entry + risk, entry + 2*risk, entry + 3*risk

    rr1 = round(abs(t1 - entry) / risk, 1) if risk > 0 else 0
    rr2 = round(abs(t2 - entry) / risk, 1) if risk > 0 else 0
    rr3 = round(abs(t3 - entry) / risk, 1) if risk > 0 else 0

    # ── Strategy label ─────────────────────────────────────────────────────
    if   score >= 6:  strategy = "Strong Bull Momentum"
    elif score >= 3:  strategy = "Bullish Trend Follow"
    elif score >= 1:  strategy = "Cautious Long Setup"
    elif score <= -6: strategy = "Strong Bear Momentum"
    elif score <= -3: strategy = "Bearish Trend Follow"
    elif score <= -1: strategy = "Cautious Short Setup"
    else:             strategy = "Neutral — No Edge"

    for p in patterns:
        nm = p.name
        if   "Morning Star"    in nm or "Three White" in nm: strategy = "Bullish Reversal ✦"
        elif "Evening Star"    in nm or "Three Black"  in nm: strategy = "Bearish Reversal ✦"
        elif "Double Bottom"   in nm:                          strategy = "Double Bottom Breakout ✦"
        elif "Double Top"      in nm:                          strategy = "Double Top Breakdown ✦"
        elif "Head & Shoulders" in nm and direction == "short":strategy = "H&S Breakdown ✦"
        elif "Inv. Head"       in nm and direction == "long":  strategy = "Inv. H&S Breakout ✦"
        elif "Ascending Triangle" in nm:                       strategy = "Ascending Triangle Breakout ✦"
        elif "Descending Triangle" in nm:                      strategy = "Descending Triangle Breakdown ✦"

    # ── Trade plan ─────────────────────────────────────────────────────────
    plan: list[str] = []
    if direction == "long":
        plan += [
            f"Enter at ${entry:.2f} — {entry_note}",
            f"Set stop at ${stop:.2f} ({stop_pct:.1f}%) — below recent swing low",
            f"Take 40% off at T1 ${t1:.2f}  →  move stop to breakeven",
            f"Take 35% off at T2 ${t2:.2f}  →  trail stop by 1× ATR",
            f"Hold final 25% to T3 ${t3:.2f} with trailing stop",
        ]
        if rsi > 65:
            plan.append(f"⚠  RSI {rsi:.0f} elevated — consider partial entry; add on pullback")
        if vol20:
            plan.append(f"✅ Confirm: volume should exceed {_fv(vol20)}/day before entry")
    elif direction == "short":
        plan += [
            f"Short at ${entry:.2f} — {entry_note}",
            f"Set stop at ${stop:.2f} (+{abs(stop_pct):.1f}%) — above recent swing high",
            f"Cover 40% at T1 ${t1:.2f}  →  move stop to breakeven",
            f"Cover 35% at T2 ${t2:.2f}  →  trail stop by 1× ATR",
            f"Cover final 25% at T3 ${t3:.2f} with trailing stop",
        ]
    else:
        plan += [
            "No directional edge — stay flat",
            "Watch for RSI extremes (<30 or >70) for entry signal",
            "Wait for decisive break above resistance or below support",
        ]

    # ── Max-profit tips ─────────────────────────────────────────────────────
    tips: list[str] = []
    if direction != "neutral":
        tips.append("Scale in 3 tranches (50% / 30% / 20%) to average entry & reduce risk")
        tips.append("Use limit orders at support (long) or resistance (short) for better fills")
        has_cs = any(
            kw in p.name for p in patterns
            for kw in ("Engulfing","Morning Star","Three White","Evening Star","Three Black","Hammer")
        )
        if has_cs:
            tips.append("Strong candlestick pattern present — high-probability setup, consider full position")
        tips.append("Activate 1× ATR trailing stop after price reaches T1 to lock gains")
        safe_risk = max(risk, 0.01)
        tips.append(f"Risk 1-2% of account per trade — at $10K that's ~{int(200/safe_risk)} shares max")
        if direction == "long" and rsi < 40:
            tips.append(f"RSI {rsi:.0f} is oversold — adds conviction; consider sizing up slightly")
        elif direction == "short" and rsi > 60:
            tips.append(f"RSI {rsi:.0f} is overbought — adds conviction to short side")
        tips.append("Close half before earnings announcements to avoid binary event risk")
    else:
        tips.append("Patience is a position — wait for the setup to mature")
        tips.append("Watchlist this ticker and re-evaluate in 3-5 sessions")

    return ProfitGuide(
        direction=direction, strategy=strategy,
        entry=round(entry, 2), entry_note=entry_note,
        stop=round(stop, 2),   stop_pct=round(stop_pct, 1),
        t1=round(t1, 2), t2=round(t2, 2), t3=round(t3, 2),
        rr1=rr1, rr2=rr2, rr3=rr3,
        risk_per_share=round(risk, 2),
        trade_plan=plan, tips=tips,
    )
