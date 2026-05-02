#!/usr/bin/env python3
"""
Stock Advisor — CLI entry point.

Usage:
  python main.py AAPL
  python main.py AAPL MSFT TSLA
  python main.py AAPL --period 1y --interval 1d
"""
import argparse
import sys
from rich.console import Console

import data_fetcher
import analyzer
import display

console = Console()


def analyze_ticker(ticker: str, period: str, interval: str) -> None:
    ticker = ticker.upper()
    try:
        with console.status(f"[cyan]Fetching data for {ticker}…[/cyan]"):
            df   = data_fetcher.fetch_ohlcv(ticker, period=period, interval=interval)
            info = data_fetcher.fetch_info(ticker)
            news = data_fetcher.fetch_news_sentiment(ticker)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        return
    except Exception as e:
        console.print(f"[red]Failed to fetch {ticker}:[/red] {e}")
        return

    ind     = analyzer.compute_indicators(df)
    signals = analyzer.identify_signals(ind, df)
    advice  = analyzer.generate_advice(ind, signals)

    display.print_header(ticker, info)
    display.print_indicators(ind, currency=info.get("currency", "USD"))
    display.print_signals(signals)
    display.print_advice(advice, ticker)
    display.print_news(news)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stock market trend analyser and advisor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py AAPL
  python main.py AAPL MSFT GOOGL
  python main.py BTC-USD --period 3mo --interval 1h
  python main.py SPY --period 2y --interval 1wk
        """,
    )
    parser.add_argument("tickers", nargs="+", help="Stock/ETF/crypto ticker symbols (e.g. AAPL MSFT BTC-USD)")
    parser.add_argument("--period",   default="6mo",
                        choices=["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"],
                        help="Historical data period (default: 6mo)")
    parser.add_argument("--interval", default="1d",
                        choices=["1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"],
                        help="Bar interval (default: 1d)")
    args = parser.parse_args()

    for ticker in args.tickers:
        analyze_ticker(ticker, period=args.period, interval=args.interval)


if __name__ == "__main__":
    main()
