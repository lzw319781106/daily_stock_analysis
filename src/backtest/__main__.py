"""Run with python -m src.backtest --demo --output reports/backtest-demo."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .data import demo_bars, load_csv, load_sqlite
from .engine import BacktestConfig, run_backtest


def main():
    parser = argparse.ArgumentParser(description='Offline A-share single-stock MA research prototype')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--demo', action='store_true', help='Deterministic synthetic prices; no network')
    inputs.add_argument('--csv', help='Standard daily OHLCV CSV')
    inputs.add_argument('--sqlite', help='Read-only existing StockDaily database')
    parser.add_argument('--code')
    parser.add_argument('--start', help='SQLite inclusive YYYY-MM-DD')
    parser.add_argument('--end', help='SQLite inclusive YYYY-MM-DD')
    parser.add_argument('--benchmark-csv', help='Optional date,close price index; exact date coverage required')
    parser.add_argument('--price-basis', choices=['unadjusted_no_actions', 'adjusted_research'],
                        help='Required on real inputs: audit no corporate actions or accept adjusted-price illustration')
    parser.add_argument('--output', default='reports/backtest-demo')
    for name, default in asdict(BacktestConfig()).items():
        parser.add_argument('--' + name.replace('_', '-'), type=int if isinstance(default, int) else float,
                            default=default)
    args = parser.parse_args()
    if not args.demo and args.price_basis is None:
        parser.error('--price-basis is required; existing storage does not record adjustment basis')
    if args.sqlite and not all([args.code, args.start, args.end]):
        parser.error('--sqlite requires --code, --start and --end')
    if args.demo:
        bars, source, basis = demo_bars(), 'synthetic_seed_20261008', 'synthetic_no_actions'
    elif args.csv:
        bars, source, basis = load_csv(args.csv), 'csv', args.price_basis
    else:
        bars = load_sqlite(args.sqlite, args.code, args.start, args.end)
        source, basis = 'stock_daily_read_only', args.price_basis
    config = BacktestConfig(**{key: getattr(args, key) for key in asdict(BacktestConfig())})
    benchmark = pd.read_csv(args.benchmark_csv) if args.benchmark_csv else None
    result = run_backtest(bars, config, benchmark)
    warnings = [
        'Research prototype: no corporate-action cash/share ledger or historical fee schedules.',
        'No auction liquidity, partial fills, limit-price rules or delisting settlement simulation.',
        'Daily zero volume is only a suspension proxy; supply opening-known can_buy/can_sell for real execution.',
        'Benchmark is a frictionless close-price index, not a cost-matched tradable portfolio.',
        'Final holdings are marked to last close without liquidation fees.',
    ]
    if args.demo:
        warnings.append('SYNTHETIC DATA: returns have no investment interpretation; dates are weekdays, not exchange calendar.')
    if basis == 'adjusted_research':
        warnings.append('ADJUSTED PRICE ILLUSTRATION: lot sizing, cash and fees are not realistic raw-price accounting.')
    metadata = {
        'source': source, 'price_basis': basis, 'code': args.code,
        'start': str(bars['date'].iloc[0]), 'end': str(bars['date'].iloc[-1]),
        'input_sha256': hashlib.sha256(bars.to_csv(index=False).encode()).hexdigest(),
        'benchmark_sha256': hashlib.sha256(benchmark.to_csv(index=False).encode()).hexdigest() if benchmark is not None else None,
        'pandas_version': pd.__version__, 'execution': 'prior_close_target_at_next_open',
        'warnings': warnings,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    bars.to_csv(output / 'input.csv', index=False)
    result.signals.to_csv(output / 'signals.csv', index=False)
    result.trades.to_csv(output / 'trades.csv', index=False)
    result.equity.to_csv(output / 'equity.csv', index=False)
    summary = {'metrics': result.metrics, 'config': result.config, 'metadata': metadata}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
