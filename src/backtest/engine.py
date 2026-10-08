"""Single-stock long/cash daily simulator with next-open execution."""

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .data import validate_bars
from .signals import ma_signals


@dataclass(frozen=True)
class BacktestConfig:
    fast: int = 5
    slow: int = 20
    initial_cash: float = 100_000.0
    lot_size: int = 100
    commission_rate: float = 0.0003
    minimum_commission: float = 5.0
    sell_tax_rate: float = 0.0005
    transfer_rate: float = 0.00001
    slippage_bps: float = 5.0

    def __post_init__(self):
        if not isinstance(self.fast, int) or not isinstance(self.slow, int) or not 0 < self.fast < self.slow:
            raise ValueError('Require integer windows: 0 < fast < slow')
        if not isinstance(self.lot_size, int) or self.lot_size <= 0:
            raise ValueError('lot_size must be a positive integer')
        for key, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        if self.initial_cash <= 0 or self.slippage_bps >= 10_000:
            raise ValueError('Require positive cash and slippage_bps < 10000')


@dataclass
class BacktestResult:
    signals: pd.DataFrame
    trades: pd.DataFrame
    equity: pd.DataFrame
    metrics: dict
    config: dict


def _fees(notional: float, side: str, config: BacktestConfig) -> tuple:
    commission = max(config.minimum_commission, notional * config.commission_rate)
    transfer = notional * config.transfer_rate
    tax = notional * config.sell_tax_rate if side == 'SELL' else 0.0
    return commission, transfer, tax


def _max_drawdown(values, initial: float) -> float:
    values = np.r_[initial, np.asarray(values, dtype=float)]
    return float((values / np.maximum.accumulate(values) - 1).min())


def _benchmark(bars: pd.DataFrame, benchmark: Optional[pd.DataFrame]) -> pd.Series:
    if benchmark is None:
        prices = bars['close']
    else:
        if not {'date', 'close'}.issubset(benchmark.columns):
            raise ValueError('Benchmark needs date and close')
        frame = benchmark[['date', 'close']].copy()
        frame['date'] = pd.to_datetime(frame['date'], errors='raise')
        if frame['date'].duplicated().any():
            raise ValueError('Duplicate benchmark dates')
        prices = frame.set_index('date')['close'].reindex(bars['date']).reset_index(drop=True)
        prices = pd.to_numeric(prices, errors='raise')
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError('Benchmark needs positive finite closes on every strategy date; no filling')
    return prices / float(prices.iloc[0])


def run_backtest(bars: pd.DataFrame, config: Optional[BacktestConfig] = None,
                 benchmark: Optional[pd.DataFrame] = None) -> BacktestResult:
    """Caller must audit price basis/company actions. No corporate-action ledger yet.

    Volume zero prevents execution. Positive daily volume alone cannot establish
    auction fillability: supply can_buy/can_sell based on opening information.
    Pending target is reevaluated from the latest prior close after a blocked day.
    """
    config = config or BacktestConfig()
    bars = validate_bars(bars)
    signals = ma_signals(bars, config.fast, config.slow)
    cash, shares = config.initial_cash, 0
    trades, curve = [], []
    for i, bar in bars.iterrows():
        # Deliberate one-row lag. Today's close cannot affect today's execution.
        target = int(signals.iloc[i - 1]['target']) if i else 0
        side = 'BUY' if target and shares == 0 else 'SELL' if not target and shares else None
        allowed = bar.get('can_buy' if side == 'BUY' else 'can_sell', True)
        if side and bar['volume'] > 0 and allowed:
            price = float(bar['open']) * (1 + (1 if side == 'BUY' else -1) * config.slippage_bps / 10_000)
            if side == 'BUY':
                quantity = int(cash / price / config.lot_size) * config.lot_size
                while quantity > 0:
                    notional = quantity * price
                    commission, transfer, tax = _fees(notional, side, config)
                    if notional + commission + transfer + tax <= cash:
                        break
                    quantity -= config.lot_size
            else:
                quantity = shares  # All holdings were acquired on a prior session (T+1).
            if quantity > 0:
                notional = quantity * price
                commission, transfer, tax = _fees(notional, side, config)
                fees = commission + transfer + tax
                cash += -(notional + fees) if side == 'BUY' else notional - fees
                shares += quantity if side == 'BUY' else -quantity
                trades.append({
                    'signal_date': bars.iloc[i - 1]['date'], 'date': bar['date'], 'side': side,
                    'quantity': quantity, 'open': float(bar['open']), 'price': price,
                    'notional': notional, 'commission': commission, 'transfer_fee': transfer,
                    'sell_tax': tax, 'fees': fees,
                    'slippage_cost': quantity * abs(price - float(bar['open'])),
                    'cash_after': cash, 'shares_after': shares,
                })
        curve.append({'date': bar['date'], 'cash': cash, 'shares': shares,
                      'equity': cash + shares * float(bar['close'])})
    equity = pd.DataFrame(curve)
    equity['benchmark_equity'] = _benchmark(bars, benchmark) * config.initial_cash
    trade_columns = ['signal_date', 'date', 'side', 'quantity', 'open', 'price', 'notional',
                     'commission', 'transfer_fee', 'sell_tax', 'fees', 'slippage_cost',
                     'cash_after', 'shares_after']
    trades = pd.DataFrame(trades, columns=trade_columns)
    strategy_return = float(equity['equity'].iloc[-1] / config.initial_cash - 1)
    benchmark_return = float(equity['benchmark_equity'].iloc[-1] / config.initial_cash - 1)
    metrics = {
        'total_return': strategy_return,
        'max_drawdown': _max_drawdown(equity['equity'], config.initial_cash),
        'benchmark_return': benchmark_return,
        'benchmark_max_drawdown': _max_drawdown(equity['benchmark_equity'], config.initial_cash),
        'excess_return': strategy_return - benchmark_return,
        'benchmark_kind': 'external_close_price_no_costs' if benchmark is not None else 'same_stock_close_price_no_costs',
        'trade_count': len(trades), 'total_fees': float(trades['fees'].sum()),
        'total_slippage_cost': float(trades['slippage_cost'].sum()),
        'final_equity': float(equity['equity'].iloc[-1]), 'final_shares': shares,
        'rows': len(bars), 'warmup_rows': min(config.slow - 1, len(bars)),
    }
    # Terminal holdings are marked to last close, not artificially liquidated.
    return BacktestResult(signals, trades, equity, metrics, asdict(config))
