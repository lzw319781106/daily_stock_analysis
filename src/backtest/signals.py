"""Trailing averages only; no AI/realtime/fundamental inputs enter the strategy."""

import pandas as pd

from src.stock_analyzer import StockTrendAnalyzer


def ma_signals(bars: pd.DataFrame, fast: int = 5, slow: int = 20) -> pd.DataFrame:
    if not isinstance(fast, int) or not isinstance(slow, int) or not 0 < fast < slow:
        raise ValueError('Require integer windows: 0 < fast < slow')
    prices = bars['signal_close'] if 'signal_close' in bars else bars['close']
    # Reuse the existing full-window, unrounded calculations for standard periods.
    # Do not use BaseFetcher ma5/ma20: they have partial warmup and round to 2 decimals.
    existing = StockTrendAnalyzer()._calculate_mas(pd.DataFrame({'close': prices}))
    def average(window):
        if window in (5, 10, 20):
            return existing[f'MA{window}']
        return prices.rolling(window=window, min_periods=window).mean()

    result = pd.DataFrame({'date': bars['date'], 'fast_ma': average(fast), 'slow_ma': average(slow)})
    ready = result[['fast_ma', 'slow_ma']].notna().all(axis=1)
    # Long when fast > slow, flat otherwise; equality exits. Warmup stays in cash.
    result['target'] = (ready & (result['fast_ma'] > result['slow_ma'])).astype(int)
    return result
