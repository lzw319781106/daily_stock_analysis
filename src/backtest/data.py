"""Adapters for the existing StockDaily schema and standardized fetcher columns."""

import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


def validate_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Fail closed: never fill missing prices or silently remove bad history."""
    required = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing = set(required) - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f'Empty history or missing columns: {sorted(missing)}')
    df = frame.copy()
    if 'code' in df and (df['code'].isna().any() or df['code'].nunique() != 1):
        raise ValueError('A single, non-null stock code is required')
    df['date'] = pd.to_datetime(df['date'], errors='raise')
    if df['date'].isna().any() or df['date'].dt.tz is not None:
        raise ValueError('Dates must be non-null, timezone-naive trading dates')
    if not df['date'].equals(df['date'].dt.normalize()):
        raise ValueError('Daily history requires dates without intraday timestamps')
    if df['date'].duplicated().any():
        raise ValueError('Duplicate trading dates')
    numeric = required[1:] + (['signal_close'] if 'signal_close' in df else [])
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors='raise')
    if not np.isfinite(df[numeric].to_numpy(dtype=float)).all():
        raise ValueError('Prices and volume must be finite')
    prices = ['open', 'high', 'low', 'close'] + (['signal_close'] if 'signal_close' in df else [])
    if (df[prices] <= 0).any().any() or (df['volume'] < 0).any():
        raise ValueError('Prices must be positive; volume must be nonnegative')
    if ((df['low'] > df[['open', 'close']].min(axis=1)) |
            (df['high'] < df[['open', 'close']].max(axis=1))).any():
        raise ValueError('Inconsistent OHLC range')
    # Optional execution permissions must be known at the opening auction.
    for column in ['can_buy', 'can_sell']:
        if column in df and not df[column].isin([True, False, 0, 1]).all():
            raise ValueError(f'{column} must contain only boolean or 0/1 values')
    return df.sort_values('date').reset_index(drop=True)


def load_csv(path: str) -> pd.DataFrame:
    return validate_bars(pd.read_csv(path, dtype={'code': str}))


def load_sqlite(path: str, code: str, start: str, end: str) -> pd.DataFrame:
    """Read only, no config, migrations, singleton initialization or production writes."""
    if date.fromisoformat(start) > date.fromisoformat(end):
        raise ValueError('start must not follow end')
    uri = Path(path).resolve().as_uri() + '?mode=ro'
    with sqlite3.connect(uri, uri=True) as conn:
        df = pd.read_sql_query(
            'SELECT code, date, open, high, low, close, volume, amount, data_source '
            'FROM stock_daily WHERE code = ? AND date >= ? AND date <= ? ORDER BY date',
            conn, params=(code, start, end),
        )
    # Source switching can silently mix raw and adjusted prices.
    if df['data_source'].isna().any() or df['data_source'].nunique() != 1:
        raise ValueError('SQLite history must have one known data_source; audit adjustment basis first')
    return validate_bars(df)


def load_from_storage(manager, code: str, start: date, end: date) -> pd.DataFrame:
    """Reuse DatabaseManager.get_data_range with an already initialized manager."""
    if start > end:
        raise ValueError('start must not follow end')
    records = manager.get_data_range(code, start, end)
    df = pd.DataFrame([record.to_dict() for record in records])
    if not df.empty and (df['data_source'].isna().any() or df['data_source'].nunique() != 1):
        raise ValueError('Mixed or unknown data_source; audit adjustment basis first')
    return validate_bars(df)


def demo_bars() -> pd.DataFrame:
    """Deterministic synthetic data, NOT actual A-share returns or a market calendar."""
    rng = np.random.default_rng(20261008)
    t = np.arange(180)
    close = 20 + 2.5 * np.sin(t / 11) + 0.018 * t + rng.normal(0, 0.15, len(t))
    open_price = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, len(t)))
    return pd.DataFrame({
        'date': pd.bdate_range('2025-01-02', periods=len(t)),
        'open': open_price, 'high': np.maximum(open_price, close) * 1.01,
        'low': np.minimum(open_price, close) * 0.99, 'close': close,
        'volume': np.full(len(t), 1_000_000),
    })
