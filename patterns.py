"""Candlestick and chart-level pattern detection."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Pattern:
    name: str
    pattern_type: str  # "candlestick" | "chart"
    direction: str     # "bullish" | "bearish" | "neutral"
    confidence: str    # "high" | "medium" | "low"
    description: str
    detected_at: str = ""


# ── helpers ──────────────────────────────────────────────────────────────────

def _body(o, c): return abs(c - o)
def _upper(o, h, c): return h - max(o, c)
def _lower(o, l, c): return min(o, c) - l
def _bull(o, c): return c > o
def _bear(o, c): return c < o

def _recent_trend(closes: np.ndarray, n: int = 5) -> str:
    if len(closes) < n + 1:
        return "flat"
    ups = sum(1 for i in range(-n, 0) if closes[i] > closes[i - 1])
    return "up" if ups >= n * 0.6 else ("down" if ups <= n * 0.4 else "flat")


# ── candlestick patterns ──────────────────────────────────────────────────────

def detect_candlestick_patterns(df: pd.DataFrame, lookback: int = 5) -> list[Pattern]:
    """Detect single-, two-, and three-candle patterns in the most recent bars."""
    if len(df) < 3:
        return []

    work = df.tail(lookback + 5).copy()
    o = work["Open"].values
    h = work["High"].values
    l = work["Low"].values
    c = work["Close"].values
    n = len(work)
    dates = [str(d.date()) for d in work.index]
    patterns: list[Pattern] = []

    # — Single-candle (last `lookback` bars) —
    for i in range(max(1, n - lookback), n):
        O, H, L, C = o[i], h[i], l[i], c[i]
        rng = H - L
        if rng < 1e-9:
            continue
        body = _body(O, C)
        upper = _upper(O, H, C)
        lower = _lower(O, L, C)
        br = body / rng
        date = dates[i]
        trend = _recent_trend(c[: i + 1])

        if br < 0.10:
            patterns.append(Pattern("Doji", "candlestick", "neutral", "medium",
                "Open ≈ Close — indecision; watch for a strong directional follow-through candle", date))

        elif br < 0.35 and lower >= 2 * body and upper <= body:
            if trend == "up":
                patterns.append(Pattern("Hanging Man", "candlestick", "bearish", "medium",
                    "Long lower shadow at top of uptrend — bears testing highs; potential reversal", date))
            else:
                patterns.append(Pattern("Hammer", "candlestick", "bullish", "high",
                    "Long lower shadow at bottom — buyers rejected lower prices; strong reversal signal", date))

        elif br < 0.35 and upper >= 2 * body and lower <= body:
            if trend == "up":
                patterns.append(Pattern("Shooting Star", "candlestick", "bearish", "high",
                    "Long upper shadow at top of uptrend — sellers pushed price down from session highs", date))
            else:
                patterns.append(Pattern("Inverted Hammer", "candlestick", "bullish", "low",
                    "Long upper shadow at bottom — needs confirmation; possible reversal attempt", date))

        elif br > 0.85 and _bull(O, C):
            patterns.append(Pattern("Bullish Marubozu", "candlestick", "bullish", "high",
                "Full-body bullish candle, near-zero shadows — buyers controlled the entire session", date))

        elif br > 0.85 and _bear(O, C):
            patterns.append(Pattern("Bearish Marubozu", "candlestick", "bearish", "high",
                "Full-body bearish candle, near-zero shadows — sellers controlled the entire session", date))

    # — Two-candle (last 3 bars) —
    for i in range(max(1, n - 3), n):
        O1, H1, L1, C1 = o[i-1], h[i-1], l[i-1], c[i-1]
        O2, H2, L2, C2 = o[i],   h[i],   l[i],   c[i]
        b1, b2 = _body(O1, C1), _body(O2, C2)
        date = dates[i]
        if b1 < 1e-9 or b2 < 1e-9:
            continue

        if _bear(O1, C1) and _bull(O2, C2) and O2 <= C1 and C2 >= O1 and b2 > b1:
            patterns.append(Pattern("Bullish Engulfing", "candlestick", "bullish", "high",
                "Bullish candle fully engulfs prior bearish — strong demand reversal signal", date))

        elif _bull(O1, C1) and _bear(O2, C2) and O2 >= C1 and C2 <= O1 and b2 > b1:
            patterns.append(Pattern("Bearish Engulfing", "candlestick", "bearish", "high",
                "Bearish candle fully engulfs prior bullish — strong supply reversal signal", date))

        elif _bear(O1, C1) and _bull(O2, C2) and O2 >= C1 and C2 <= O1 and b2 < b1 * 0.5:
            patterns.append(Pattern("Bullish Harami", "candlestick", "bullish", "medium",
                "Small bullish body inside prior bearish — selling momentum stalling", date))

        elif _bull(O1, C1) and _bear(O2, C2) and O2 <= C1 and C2 >= O1 and b2 < b1 * 0.5:
            patterns.append(Pattern("Bearish Harami", "candlestick", "bearish", "medium",
                "Small bearish body inside prior bullish — buying momentum stalling", date))

        elif _bear(O1, C1) and _bull(O2, C2) and O2 < L1 and C2 > (O1 + C1) / 2:
            patterns.append(Pattern("Piercing Line", "candlestick", "bullish", "medium",
                "Bulls closed above midpoint of prior bearish candle — buying conviction", date))

        elif _bull(O1, C1) and _bear(O2, C2) and O2 > H1 and C2 < (O1 + C1) / 2:
            patterns.append(Pattern("Dark Cloud Cover", "candlestick", "bearish", "medium",
                "Bears closed below midpoint of prior bullish candle — selling pressure emerging", date))

    # — Three-candle (last 3 bars only) —
    if n >= 3:
        i = n - 1
        O1, H1, L1, C1 = o[i-2], h[i-2], l[i-2], c[i-2]
        O2, H2, L2, C2 = o[i-1], h[i-1], l[i-1], c[i-1]
        O3, H3, L3, C3 = o[i],   h[i],   l[i],   c[i]
        b1, b2, b3 = _body(O1, C1), _body(O2, C2), _body(O3, C3)
        date = dates[i]

        if (_bear(O1, C1) and b1 > 0.005 * C1 and b2 < 0.35 * b1
                and _bull(O3, C3) and C3 > (O1 + C1) / 2):
            patterns.append(Pattern("Morning Star", "candlestick", "bullish", "high",
                "Bearish → small body → strong bullish — classic 3-candle bottom reversal", date))

        elif (_bull(O1, C1) and b1 > 0.005 * C1 and b2 < 0.35 * b1
                and _bear(O3, C3) and C3 < (O1 + C1) / 2):
            patterns.append(Pattern("Evening Star", "candlestick", "bearish", "high",
                "Bullish → small body → strong bearish — classic 3-candle top reversal", date))

        elif (all(_bull(o[i-2+j], c[i-2+j]) for j in range(3))
                and c[i-1] > c[i-2] and c[i] > c[i-1]
                and o[i-1] > o[i-2] and o[i] > o[i-1]
                and all(_body(o[i-2+j], c[i-2+j]) / max(h[i-2+j]-l[i-2+j], 1e-9) > 0.5 for j in range(3))):
            patterns.append(Pattern("Three White Soldiers", "candlestick", "bullish", "high",
                "Three strong consecutive bullish candles — sustained and broadening buying momentum", date))

        elif (all(_bear(o[i-2+j], c[i-2+j]) for j in range(3))
                and c[i-1] < c[i-2] and c[i] < c[i-1]
                and o[i-1] < o[i-2] and o[i] < o[i-1]
                and all(_body(o[i-2+j], c[i-2+j]) / max(h[i-2+j]-l[i-2+j], 1e-9) > 0.5 for j in range(3))):
            patterns.append(Pattern("Three Black Crows", "candlestick", "bearish", "high",
                "Three strong consecutive bearish candles — sustained and broadening selling momentum", date))

    # Deduplicate (keep most recent occurrence per name)
    seen: set[str] = set()
    unique: list[Pattern] = []
    for p in reversed(patterns):
        if p.name not in seen:
            seen.add(p.name)
            unique.append(p)
    return list(reversed(unique))


# ── chart patterns ────────────────────────────────────────────────────────────

def _find_extremes(arr: np.ndarray, order: int) -> tuple[list[int], list[int]]:
    peaks, troughs = [], []
    n = len(arr)
    for i in range(order, n - order):
        window = arr[i - order: i + order + 1]
        if arr[i] >= window.max() - 1e-9:
            if not peaks or i - peaks[-1] >= order:
                peaks.append(i)
        if arr[i] <= window.min() + 1e-9:
            if not troughs or i - troughs[-1] >= order:
                troughs.append(i)
    return peaks, troughs


def detect_chart_patterns(df: pd.DataFrame) -> list[Pattern]:
    """Detect chart-level patterns: double top/bottom, H&S, trend, support/resistance."""
    if len(df) < 40:
        return []

    patterns: list[Pattern] = []
    closes = df["Close"].values
    n = len(closes)
    current = closes[-1]
    last_date = str(df.index[-1].date())

    order = max(3, n // 15)
    peaks, troughs = _find_extremes(closes, order)

    # — Double Bottom —
    if len(troughs) >= 2:
        t1i, t2i = troughs[-2], troughs[-1]
        t1, t2 = closes[t1i], closes[t2i]
        diff = abs(t1 - t2) / max(t1, t2)
        if diff < 0.04 and t2i > t1i + order:
            neck = closes[t1i:t2i].max()
            conf = "high" if current >= neck * 0.97 else "medium"
            status = f"confirmed above neckline ${neck:.2f}" if conf == "high" else f"watch for break above ${neck:.2f}"
            patterns.append(Pattern("Double Bottom (W)", "chart", "bullish", conf,
                f"Two lows ~${t1:.2f} / ${t2:.2f} ({diff*100:.1f}% apart) — {status}", last_date))

    # — Double Top —
    if len(peaks) >= 2:
        p1i, p2i = peaks[-2], peaks[-1]
        p1, p2 = closes[p1i], closes[p2i]
        diff = abs(p1 - p2) / max(p1, p2)
        if diff < 0.04 and p2i > p1i + order:
            neck = closes[p1i:p2i].min()
            conf = "high" if current <= neck * 1.03 else "medium"
            status = f"confirmed below neckline ${neck:.2f}" if conf == "high" else f"watch for breakdown below ${neck:.2f}"
            patterns.append(Pattern("Double Top (M)", "chart", "bearish", conf,
                f"Two peaks ~${p1:.2f} / ${p2:.2f} ({diff*100:.1f}% apart) — {status}", last_date))

    # — Head & Shoulders —
    if len(peaks) >= 3:
        s1i, hi, s2i = peaks[-3], peaks[-2], peaks[-1]
        s1, head, s2 = closes[s1i], closes[hi], closes[s2i]
        if head > s1 * 1.01 and head > s2 * 1.01 and abs(s1 - s2) / max(s1, s2) < 0.06:
            patterns.append(Pattern("Head & Shoulders", "chart", "bearish", "high",
                f"L-shoulder ${s1:.2f} | head ${head:.2f} | R-shoulder ${s2:.2f} — classic bearish topping pattern",
                last_date))

    # — Inverse Head & Shoulders —
    if len(troughs) >= 3:
        s1i, hi, s2i = troughs[-3], troughs[-2], troughs[-1]
        s1, head, s2 = closes[s1i], closes[hi], closes[s2i]
        if head < s1 * 0.99 and head < s2 * 0.99 and abs(s1 - s2) / max(s1, s2) < 0.06:
            patterns.append(Pattern("Inv. Head & Shoulders", "chart", "bullish", "high",
                f"L-shoulder ${s1:.2f} | head ${head:.2f} | R-shoulder ${s2:.2f} — classic bullish bottoming pattern",
                last_date))

    # — At Support / Resistance —
    if len(troughs) >= 3:
        support = float(np.median([closes[i] for i in troughs[-5:]]))
        if abs(current - support) / support < 0.025:
            patterns.append(Pattern("At Support", "chart", "bullish", "medium",
                f"Price ${current:.2f} at key support ${support:.2f} — historically attracted buyers here",
                last_date))

    if len(peaks) >= 3:
        resistance = float(np.median([closes[i] for i in peaks[-5:]]))
        if abs(current - resistance) / resistance < 0.025:
            patterns.append(Pattern("At Resistance", "chart", "bearish", "medium",
                f"Price ${current:.2f} at key resistance ${resistance:.2f} — historically attracted sellers here",
                last_date))

    # — Trend channel (linear regression over last ~20 bars) —
    window = min(20, n // 3)
    if window >= 10:
        recent = closes[-window:]
        x = np.arange(window, dtype=float)
        m, _ = np.polyfit(x, recent, 1)
        slope_pct = m / recent.mean() * 100
        if slope_pct > 0.4:
            patterns.append(Pattern("Uptrend", "chart", "bullish", "medium",
                f"Linear regression slope +{slope_pct:.2f}%/bar over {window} periods — bullish price momentum",
                last_date))
        elif slope_pct < -0.4:
            patterns.append(Pattern("Downtrend", "chart", "bearish", "medium",
                f"Linear regression slope {slope_pct:.2f}%/bar over {window} periods — bearish price momentum",
                last_date))

    # — Ascending / Descending Triangle —
    if len(peaks) >= 2 and len(troughs) >= 2:
        flat_top = abs(closes[peaks[-1]] - closes[peaks[-2]]) / max(closes[peaks[-1]], closes[peaks[-2]]) < 0.015
        rising_lows = closes[troughs[-1]] > closes[troughs[-2]] * 1.01
        flat_bottom = abs(closes[troughs[-1]] - closes[troughs[-2]]) / max(closes[troughs[-1]], closes[troughs[-2]]) < 0.015
        falling_highs = closes[peaks[-1]] < closes[peaks[-2]] * 0.99

        if flat_top and rising_lows:
            patterns.append(Pattern("Ascending Triangle", "chart", "bullish", "medium",
                f"Flat resistance ~${closes[peaks[-1]]:.2f} + rising support — coiling for bullish breakout",
                last_date))
        elif flat_bottom and falling_highs:
            patterns.append(Pattern("Descending Triangle", "chart", "bearish", "medium",
                f"Flat support ~${closes[troughs[-1]]:.2f} + falling resistance — coiling for bearish breakdown",
                last_date))

    return patterns
