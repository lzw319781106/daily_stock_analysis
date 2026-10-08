"""Offline research only; importing this package never initializes API clients."""

from .engine import BacktestConfig, BacktestResult, run_backtest

__all__ = ['BacktestConfig', 'BacktestResult', 'run_backtest']
