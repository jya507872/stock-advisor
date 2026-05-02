"""Rich terminal output for stock analysis results."""
import math
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box
from rich.text import Text
from rich.rule import Rule

from analyzer import Indicators, TrendSignal, Advice

console = Console()

_ACTION_STYLE = {
    "BUY":   ("green", "▲"),
    "SELL":  ("red",   "▼"),
    "HOLD":  ("yellow","◆"),
    "WATCH": ("cyan",  "◉"),
}

_DIR_COLOR = {"bullish": "green", "bearish": "red", "neutral": "dim"}
_STR_STYLE = {"strong": "bold", "moderate": "", "weak": "dim"}


def _fmt(val: float, decimals: int = 2, prefix: str = "") -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "[dim]N/A[/dim]"
    return f"{prefix}{val:,.{decimals}f}"


def _fmt_large(val: float) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "[dim]N/A[/dim]"
    if val >= 1e12:
        return f"${val/1e12:.2f}T"
    if val >= 1e9:
        return f"${val/1e9:.2f}B"
    if val >= 1e6:
        return f"${val/1e6:.2f}M"
    return f"${val:,.0f}"


def print_header(ticker: str, info: dict) -> None:
    console.print()
    console.print(Rule(f"[bold cyan]  STOCK ADVISOR — {ticker.upper()}  [/bold cyan]", style="cyan"))
    console.print(f"  [bold]{info['name']}[/bold]  |  "
                  f"Sector: [cyan]{info['sector']}[/cyan]  |  "
                  f"Industry: [cyan]{info['industry']}[/cyan]")
    console.print(f"  Market Cap: [yellow]{_fmt_large(info['market_cap'])}[/yellow]  |  "
                  f"P/E: [yellow]{_fmt(info['pe_ratio'])}[/yellow]  |  "
                  f"52W High: [green]{_fmt(info['52w_high'], prefix=info['currency']+' ')}[/green]  |  "
                  f"52W Low: [red]{_fmt(info['52w_low'], prefix=info['currency']+' ')}[/red]")
    console.print()


def print_indicators(ind: Indicators, currency: str = "USD") -> None:
    t = Table(title="Technical Indicators", box=box.ROUNDED, show_header=True,
              header_style="bold cyan", border_style="dim")
    t.add_column("Indicator", style="bold", min_width=18)
    t.add_column("Value", justify="right")
    t.add_column("Indicator", style="bold", min_width=18)
    t.add_column("Value", justify="right")

    pct_color = "green" if ind.price_change_pct >= 0 else "red"
    pct_sign  = "+" if ind.price_change_pct >= 0 else ""

    rows = [
        ("Price", f"{currency} {_fmt(ind.price)}",
         "RSI (14)", _rsi_colored(ind.rsi_14)),
        ("Day Change", f"[{pct_color}]{pct_sign}{_fmt(ind.price_change_pct)}%[/{pct_color}]",
         "MACD", _fmt(ind.macd, 4)),
        ("SMA 20", _fmt(ind.sma_20),
         "MACD Signal", _fmt(ind.macd_signal, 4)),
        ("SMA 50", _fmt(ind.sma_50),
         "MACD Hist", _macd_hist_colored(ind.macd_hist)),
        ("SMA 200", _fmt(ind.sma_200),
         "BB Upper", _fmt(ind.bb_upper)),
        ("EMA 12", _fmt(ind.ema_12),
         "BB Middle", _fmt(ind.bb_middle)),
        ("EMA 26", _fmt(ind.ema_26),
         "BB Lower", _fmt(ind.bb_lower)),
        ("ATR (14)", _fmt(ind.atr_14),
         "Volume / Avg", _vol_ratio(ind)),
    ]
    for a, b, c, d in rows:
        t.add_row(a, b, c, d)
    console.print(t)
    console.print()


def _rsi_colored(rsi: float) -> str:
    v = _fmt(rsi, 1)
    if math.isnan(rsi):
        return v
    if rsi >= 70:
        return f"[bold red]{v}[/bold red]"
    if rsi <= 30:
        return f"[bold green]{v}[/bold green]"
    return v


def _macd_hist_colored(h: float) -> str:
    v = _fmt(h, 4)
    if math.isnan(h):
        return v
    return f"[green]{v}[/green]" if h >= 0 else f"[red]{v}[/red]"


def _vol_ratio(ind: Indicators) -> str:
    if math.isnan(ind.volume_sma_20) or ind.volume_sma_20 == 0:
        return "[dim]N/A[/dim]"
    ratio = ind.obv  # placeholder — show volume vs avg instead
    # We don't have last volume stored; just show avg
    return f"{_fmt(ind.volume_sma_20, 0)} avg"


def print_signals(signals: list[TrendSignal]) -> None:
    t = Table(title="Trend Signals", box=box.ROUNDED, show_header=True,
              header_style="bold cyan", border_style="dim")
    t.add_column("Signal", min_width=20)
    t.add_column("Direction", justify="center")
    t.add_column("Strength", justify="center")
    t.add_column("Details")

    for s in signals:
        color = _DIR_COLOR.get(s.direction, "white")
        style = _STR_STYLE.get(s.strength, "")
        t.add_row(
            f"[{style}]{s.name}[/{style}]" if style else s.name,
            f"[{color}]{s.direction.upper()}[/{color}]",
            f"[{style}]{s.strength.upper()}[/{style}]" if style else s.strength.upper(),
            s.detail,
        )

    if not signals:
        t.add_row("[dim]No signals detected[/dim]", "", "", "")

    console.print(t)
    console.print()


def print_advice(advice: Advice, ticker: str) -> None:
    color, icon = _ACTION_STYLE.get(advice.action, ("white", "?"))
    conf_color = {"High": "green", "Medium": "yellow", "Low": "red"}.get(advice.confidence, "white")

    bar = _score_bar(advice.score)
    header = (
        f"[bold {color}]{icon}  RECOMMENDATION: {advice.action}[/bold {color}]   "
        f"Confidence: [{conf_color}]{advice.confidence}[/{conf_color}]   "
        f"Score: {bar} [bold]{advice.score:+d}/10[/bold]"
    )

    body_lines = ["[bold]Rationale:[/bold]"]
    for r in advice.rationale:
        body_lines.append(f"  • {r}")

    if advice.risk_notes:
        body_lines.append("")
        body_lines.append("[bold yellow]Risk Notes:[/bold yellow]")
        for rn in advice.risk_notes:
            body_lines.append(f"  ⚠  [yellow]{rn}[/yellow]")

    body_lines.append("")
    body_lines.append("[dim italic]⚠  This is algorithmic analysis only — not financial advice. "
                      "Always do your own research and consult a professional.[/dim italic]")

    console.print(Panel("\n".join(body_lines), title=header, border_style=color,
                        box=box.DOUBLE_EDGE, padding=(1, 2)))
    console.print()


def _score_bar(score: int, width: int = 20) -> str:
    """Visual score bar from -10 to +10."""
    center = width // 2
    filled = int(abs(score) / 10 * center)
    if score >= 0:
        bar = "─" * center + "█" * filled + " " * (center - filled)
        return f"[dim]{'─'*center}[/dim][green]{'█'*filled}[/green][dim]{' '*(center-filled)}[/dim]"
    else:
        bar = " " * (center - filled) + "█" * filled + "─" * center
        return f"[dim]{' '*(center-filled)}[/dim][red]{'█'*filled}[/red][dim]{'─'*center}[/dim]"


def print_news(news: list[dict]) -> None:
    if not news:
        return
    t = Table(title="Recent News", box=box.ROUNDED, header_style="bold cyan", border_style="dim")
    t.add_column("Published", min_width=12, style="dim")
    t.add_column("Headline")
    for item in news[:5]:
        pub = item.get("published", "")[:10] if item.get("published") else "—"
        title = item.get("title", "—")
        t.add_row(pub, title)
    console.print(t)
    console.print()
