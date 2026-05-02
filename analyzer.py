"""Technical analysis: compute indicators and identify trends."""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Indicators:
    # Moving averages
    sma_20: float = float("nan")
    sma_50: float = float("nan")
    sma_200: float = float("nan")
    ema_12: float = float("nan")
    ema_26: float = float("nan")

    # Momentum
    rsi_14: float = float("nan")
    macd: float = float("nan")
    macd_signal: float = float("nan")
    macd_hist: float = float("nan")

    # Volatility
    bb_upper: float = float("nan")
    bb_middle: float = float("nan")
    bb_lower: float = float("nan")
    atr_14: float = float("nan")

    # Volume
    obv: float = float("nan")
    volume_sma_20: float = float("nan")

    # Current price
    price: float = float("nan")
    price_change_pct: float = float("nan")  # vs previous close


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, prev_close = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def _obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def compute_indicators(df: pd.DataFrame) -> Indicators:
    close = df["Close"]
    ind = Indicators()
    ind.price = close.iloc[-1]
    ind.price_change_pct = (close.iloc[-1] / close.iloc[-2] - 1) * 100 if len(close) >= 2 else float("nan")

    # SMAs
    if len(close) >= 20:
        ind.sma_20 = close.rolling(20).mean().iloc[-1]
    if len(close) >= 50:
        ind.sma_50 = close.rolling(50).mean().iloc[-1]
    if len(close) >= 200:
        ind.sma_200 = close.rolling(200).mean().iloc[-1]

    # EMAs
    ind.ema_12 = _ema(close, 12).iloc[-1]
    ind.ema_26 = _ema(close, 26).iloc[-1]

    # MACD
    macd_line = _ema(close, 12) - _ema(close, 26)
    signal_line = _ema(macd_line, 9)
    ind.macd = macd_line.iloc[-1]
    ind.macd_signal = signal_line.iloc[-1]
    ind.macd_hist = ind.macd - ind.macd_signal

    # RSI
    if len(close) >= 14:
        ind.rsi_14 = _rsi(close).iloc[-1]

    # Bollinger Bands (20-period, 2 std)
    if len(close) >= 20:
        mid = close.rolling(20).mean()
        std = close.rolling(20).std()
        ind.bb_upper = (mid + 2 * std).iloc[-1]
        ind.bb_middle = mid.iloc[-1]
        ind.bb_lower = (mid - 2 * std).iloc[-1]

    # ATR
    if len(df) >= 14:
        ind.atr_14 = _atr(df).iloc[-1]

    # OBV
    ind.obv = _obv(df).iloc[-1]
    if len(df) >= 20:
        ind.volume_sma_20 = df["Volume"].rolling(20).mean().iloc[-1]

    return ind


@dataclass
class TrendSignal:
    name: str
    direction: str   # "bullish", "bearish", "neutral"
    strength: str    # "strong", "moderate", "weak"
    detail: str


def identify_signals(ind: Indicators, df: pd.DataFrame) -> list[TrendSignal]:
    signals: list[TrendSignal] = []
    price = ind.price

    # --- Golden / Death cross (50 vs 200 SMA) ---
    if not (np.isnan(ind.sma_50) or np.isnan(ind.sma_200)):
        if ind.sma_50 > ind.sma_200 * 1.001:
            signals.append(TrendSignal("Golden Cross", "bullish", "strong",
                f"SMA50 ({ind.sma_50:.2f}) above SMA200 ({ind.sma_200:.2f})"))
        elif ind.sma_50 < ind.sma_200 * 0.999:
            signals.append(TrendSignal("Death Cross", "bearish", "strong",
                f"SMA50 ({ind.sma_50:.2f}) below SMA200 ({ind.sma_200:.2f})"))

    # --- Price vs SMA20 ---
    if not np.isnan(ind.sma_20):
        if price > ind.sma_20 * 1.02:
            signals.append(TrendSignal("Above SMA20", "bullish", "moderate",
                f"Price {price:.2f} > SMA20 {ind.sma_20:.2f} (+{(price/ind.sma_20-1)*100:.1f}%)"))
        elif price < ind.sma_20 * 0.98:
            signals.append(TrendSignal("Below SMA20", "bearish", "moderate",
                f"Price {price:.2f} < SMA20 {ind.sma_20:.2f} ({(price/ind.sma_20-1)*100:.1f}%)"))

    # --- RSI ---
    if not np.isnan(ind.rsi_14):
        if ind.rsi_14 >= 70:
            signals.append(TrendSignal("RSI Overbought", "bearish", "moderate",
                f"RSI14 = {ind.rsi_14:.1f} (≥70 = overbought)"))
        elif ind.rsi_14 <= 30:
            signals.append(TrendSignal("RSI Oversold", "bullish", "moderate",
                f"RSI14 = {ind.rsi_14:.1f} (≤30 = oversold, potential reversal)"))
        elif ind.rsi_14 > 55:
            signals.append(TrendSignal("RSI Bullish Zone", "bullish", "weak",
                f"RSI14 = {ind.rsi_14:.1f} (momentum favors buyers)"))
        elif ind.rsi_14 < 45:
            signals.append(TrendSignal("RSI Bearish Zone", "bearish", "weak",
                f"RSI14 = {ind.rsi_14:.1f} (momentum favors sellers)"))

    # --- MACD ---
    if not (np.isnan(ind.macd) or np.isnan(ind.macd_signal)):
        if ind.macd > ind.macd_signal and ind.macd_hist > 0:
            signals.append(TrendSignal("MACD Bullish", "bullish", "moderate",
                f"MACD {ind.macd:.4f} above signal {ind.macd_signal:.4f}"))
        elif ind.macd < ind.macd_signal and ind.macd_hist < 0:
            signals.append(TrendSignal("MACD Bearish", "bearish", "moderate",
                f"MACD {ind.macd:.4f} below signal {ind.macd_signal:.4f}"))

    # --- Bollinger Bands ---
    if not (np.isnan(ind.bb_upper) or np.isnan(ind.bb_lower)):
        bb_width = ind.bb_upper - ind.bb_lower
        bb_pos = (price - ind.bb_lower) / bb_width if bb_width > 0 else 0.5
        if price >= ind.bb_upper:
            signals.append(TrendSignal("BB Upper Touch", "bearish", "moderate",
                f"Price touching upper band ({ind.bb_upper:.2f}) — potential mean reversion"))
        elif price <= ind.bb_lower:
            signals.append(TrendSignal("BB Lower Touch", "bullish", "moderate",
                f"Price touching lower band ({ind.bb_lower:.2f}) — potential bounce"))
        elif bb_pos > 0.8:
            signals.append(TrendSignal("BB Upper Zone", "bullish", "weak",
                f"Price in upper Bollinger zone ({bb_pos*100:.0f}th percentile of band)"))

    # --- Volume spike ---
    if not np.isnan(ind.volume_sma_20):
        last_vol = df["Volume"].iloc[-1]
        ratio = last_vol / ind.volume_sma_20
        if ratio >= 2.0:
            direction = "bullish" if ind.price_change_pct >= 0 else "bearish"
            signals.append(TrendSignal("Volume Spike", direction, "moderate",
                f"Volume {ratio:.1f}x above 20-day average — confirms price move"))

    return signals


@dataclass
class Advice:
    action: str          # "BUY", "SELL", "HOLD", "WATCH"
    confidence: str      # "High", "Medium", "Low"
    score: int           # -10 to +10
    rationale: list[str] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)


_STRENGTH_WEIGHT = {"strong": 3, "moderate": 2, "weak": 1}
_CONF_WEIGHT = {"high": 3, "medium": 2, "low": 1}


def generate_advice(ind: Indicators, signals: list[TrendSignal], patterns: list | None = None) -> Advice:
    score = 0
    rationale: list[str] = []
    risk_notes: list[str] = []

    for sig in signals:
        w = _STRENGTH_WEIGHT.get(sig.strength, 1)
        if sig.direction == "bullish":
            score += w
        elif sig.direction == "bearish":
            score -= w
        rationale.append(f"[{sig.direction.upper()}] {sig.name}: {sig.detail}")

    for pat in (patterns or []):
        w = _CONF_WEIGHT.get(pat.confidence, 1)
        if pat.direction == "bullish":
            score += w
            rationale.append(f"[BULLISH PATTERN] {pat.name}: {pat.description}")
        elif pat.direction == "bearish":
            score -= w
            rationale.append(f"[BEARISH PATTERN] {pat.name}: {pat.description}")

    # Clip score
    score = max(-10, min(10, score))

    # Risk notes
    if not np.isnan(ind.atr_14):
        atr_pct = ind.atr_14 / ind.price * 100
        if atr_pct > 3:
            risk_notes.append(f"High volatility: ATR is {atr_pct:.1f}% of price — size positions carefully")
    if not np.isnan(ind.rsi_14):
        if ind.rsi_14 > 75:
            risk_notes.append("Extreme overbought conditions — buying here carries reversal risk")
        elif ind.rsi_14 < 25:
            risk_notes.append("Extreme oversold — falling knife risk, wait for confirmation")

    # Action thresholds
    if score >= 5:
        action, confidence = "BUY", "High"
    elif score >= 2:
        action, confidence = "BUY", "Medium"
    elif score <= -5:
        action, confidence = "SELL", "High"
    elif score <= -2:
        action, confidence = "SELL", "Medium"
    elif abs(score) <= 1:
        action, confidence = "HOLD", "Medium"
    else:
        action, confidence = "WATCH", "Low"

    return Advice(action=action, confidence=confidence, score=score,
                  rationale=rationale, risk_notes=risk_notes)
