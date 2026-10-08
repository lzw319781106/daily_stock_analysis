"""Network-free tests, including adversarial causality and independent accounting."""

import sqlite3
import contextlib
import io
import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.backtest.data import demo_bars, load_csv, load_from_storage, load_sqlite, validate_bars
from src.backtest.engine import BacktestConfig, run_backtest
from src.backtest.signals import ma_signals


def simple_bars(closes, opens=None):
    opens = np.array(opens if opens is not None else closes, dtype=float)
    closes = np.array(closes, dtype=float)
    return pd.DataFrame({'date': pd.bdate_range('2025-01-02', periods=len(closes)),
                         'open': opens, 'high': np.maximum(opens, closes) + 1,
                         'low': np.minimum(opens, closes) - 1, 'close': closes,
                         'volume': 10_000})


ZERO_COST = BacktestConfig(fast=1, slow=2, initial_cash=10_000, commission_rate=0,
                           minimum_commission=0, sell_tax_rate=0, transfer_rate=0, slippage_bps=0)


class BacktestTests(unittest.TestCase):
    def test_signal_executes_at_next_open_not_signal_close(self):
        bars = simple_bars([10, 12, 13, 11, 10], [10, 10, 20, 13, 15])
        result = run_backtest(bars, ZERO_COST)
        self.assertEqual(result.trades['side'].tolist(), ['BUY', 'SELL'])
        self.assertEqual(result.trades['price'].tolist(), [20, 15])
        self.assertEqual(result.trades['quantity'].tolist(), [500, 500])
        self.assertEqual(result.trades['signal_date'].tolist(), bars['date'].iloc[[1, 3]].tolist())
        self.assertEqual(result.metrics['total_return'], -0.25)
        self.assertAlmostEqual(result.metrics['max_drawdown'], -0.45)
        self.assertEqual(result.metrics['benchmark_return'], 0)

    def test_costs_lot_rounding_cash_and_slippage_independent(self):
        bars = simple_bars([10, 12, 13, 11, 10], [10, 10, 20, 13, 15])
        cfg = replace(ZERO_COST, commission_rate=0.001, minimum_commission=5,
                      transfer_rate=0.0001, sell_tax_rate=0.002, slippage_bps=100)
        result = run_backtest(bars, cfg)
        buy, sell = result.trades.iloc[0], result.trades.iloc[1]
        # 500 shares cost > 10000 after slippage, so only 400 shares are affordable.
        self.assertEqual(buy['quantity'], 400)
        self.assertAlmostEqual(buy['notional'], 8080)
        self.assertAlmostEqual(buy['fees'], 8.08 + 0.808)
        self.assertAlmostEqual(sell['notional'], 5940)
        self.assertAlmostEqual(sell['fees'], 5.94 + 0.594 + 11.88)
        self.assertAlmostEqual(result.metrics['final_equity'], 10_000 - 8080 - 8.888 + 5940 - 18.414)
        self.assertAlmostEqual(result.metrics['total_slippage_cost'], 80 + 60)
        self.assertTrue((result.equity['cash'] >= 0).all())
        self.assertTrue((result.equity['shares'] % 100 == 0).all())

    def test_minimum_commission_and_unaffordable_order(self):
        bars = simple_bars([10, 12, 13])
        cfg = replace(ZERO_COST, initial_cash=1400, minimum_commission=5, commission_rate=0.0003)
        result = run_backtest(bars, cfg)
        self.assertEqual(result.trades.iloc[0]['commission'], 5)
        self.assertEqual(result.equity.iloc[-1]['cash'], 95)
        self.assertTrue(run_backtest(bars, replace(cfg, initial_cash=1304)).trades.empty)

    def test_every_prefix_matches_full_run_including_trades_and_nav(self):
        bars = demo_bars().iloc[:90]
        full = run_backtest(bars)
        for size in range(1, len(bars) + 1):
            prefix = run_backtest(bars.iloc[:size])
            assert_frame_equal(prefix.signals, full.signals.iloc[:size].reset_index(drop=True))
            assert_frame_equal(prefix.equity, full.equity.iloc[:size].reset_index(drop=True))
            expected = full.trades[full.trades['date'] <= bars.iloc[size - 1]['date']].reset_index(drop=True)
            assert_frame_equal(prefix.trades, expected, check_dtype=False)

    def test_poison_future_prices_and_cached_indicators(self):
        bars = demo_bars()
        mutated = bars.copy()
        mutated.loc[80:, ['open', 'high', 'low', 'close']] *= 100
        mutated['ma5'], mutated['ma20'], mutated['MA5'], mutated['MA20'] = 999, -999, 999, -999
        first, second = run_backtest(bars), run_backtest(mutated)
        assert_frame_equal(first.signals.iloc[:80], second.signals.iloc[:80])
        assert_frame_equal(first.equity.iloc[:80], second.equity.iloc[:80])
        assert_frame_equal(first.trades[first.trades['date'] < bars.iloc[80]['date']].reset_index(drop=True),
                           second.trades[second.trades['date'] < bars.iloc[80]['date']].reset_index(drop=True))

    def test_current_close_high_low_cannot_change_current_open_trade(self):
        bars = simple_bars([10, 12, 13, 11, 10], [10, 10, 20, 13, 15])
        changed = bars.copy()
        changed.loc[2, ['close', 'high', 'low']] = [1, 1000, 0.5]
        first = run_backtest(bars, ZERO_COST).trades.iloc[:1]
        second = run_backtest(changed, ZERO_COST).trades.iloc[:1]
        assert_frame_equal(first, second)

    def test_warmup_full_windows_and_no_last_close_forced_trade(self):
        bars = simple_bars([10, 11, 12, 13])
        result = run_backtest(bars, replace(ZERO_COST, fast=2, slow=4))
        self.assertEqual(result.signals['target'].tolist(), [0, 0, 0, 1])
        self.assertTrue(result.trades.empty)
        self.assertEqual(result.metrics['total_return'], 0)

    def test_zero_volume_and_opening_permissions_block_trades(self):
        bars = simple_bars([10, 12, 13, 14, 11, 10, 9])
        bars.loc[2, 'volume'] = 0
        bars['can_sell'] = True
        bars.loc[5, 'can_sell'] = False
        result = run_backtest(bars, ZERO_COST)
        self.assertEqual(result.trades['date'].tolist(), bars['date'].iloc[[3, 6]].tolist())
        self.assertTrue((result.trades['signal_date'] < result.trades['date']).all())

    def test_no_terminal_liquidation_and_initial_peak_drawdown(self):
        bars = simple_bars([10, 12, 13])
        result = run_backtest(bars, replace(ZERO_COST, minimum_commission=5))
        self.assertEqual(result.trades['side'].tolist(), ['BUY'])
        self.assertEqual(result.metrics['final_shares'], 700)
        self.assertAlmostEqual(result.metrics['max_drawdown'], -5 / 10_000)

    def test_benchmark_does_not_influence_strategy_and_requires_coverage(self):
        bars = demo_bars()
        benchmark = pd.DataFrame({'date': bars['date'], 'close': np.linspace(100, 150, len(bars))})
        first, second = run_backtest(bars), run_backtest(bars, benchmark=benchmark)
        assert_frame_equal(first.trades, second.trades)
        self.assertEqual(second.metrics['benchmark_return'], 0.5)
        self.assertEqual(second.metrics['benchmark_max_drawdown'], 0)
        for bad in [benchmark.iloc[1:], pd.concat([benchmark, benchmark.iloc[:1]])]:
            with self.assertRaises(ValueError):
                run_backtest(bars, benchmark=bad)

    def test_invalid_data_and_configs_fail_closed(self):
        bars = simple_bars([10, 12, 13])
        for bad in [bars.iloc[:0], pd.concat([bars, bars.iloc[:1]]),
                    bars.assign(close=np.nan), bars.assign(volume=-1),
                    bars.assign(code=['a', 'b', 'a']), bars.assign(high=1),
                    bars.assign(can_buy='False'), bars.assign(date=pd.NaT)]:
            with self.assertRaises(ValueError):
                validate_bars(bad)
        for kwargs in [{'fast': 20}, {'slippage_bps': 10_000}, {'initial_cash': np.inf},
                       {'commission_rate': -1}, {'lot_size': 0}, {'fast': 1.5}]:
            with self.assertRaises(ValueError):
                BacktestConfig(**kwargs)

    def test_existing_indicator_reuse_and_custom_windows_are_causal(self):
        from src.stock_analyzer import StockTrendAnalyzer
        bars = demo_bars()
        existing = StockTrendAnalyzer()._calculate_mas(bars)
        signals = ma_signals(bars)
        self.assertTrue(signals['fast_ma'].equals(existing['MA5']))
        self.assertTrue(signals['slow_ma'].equals(existing['MA20']))
        for size in [8, 15, 30]:
            assert_frame_equal(ma_signals(bars.iloc[:size], 3, 7), ma_signals(bars, 3, 7).iloc[:size])

    def test_real_stockdaily_roundtrip_and_read_only_adapter(self):
        from src.storage import DatabaseManager
        DatabaseManager.reset_instance()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.db'
            manager = DatabaseManager('sqlite:///' + str(path))
            try:
                bars = demo_bars().iloc[:45]
                self.assertEqual(manager.save_daily_data(bars, '000001', 'OfflineFixture'), 45)
                original_bytes = path.read_bytes()
                loaded = load_sqlite(str(path), '000001', '2025-01-02', '2025-12-31')
                reused = load_from_storage(manager, '000001', date(2025, 1, 2), date(2025, 12, 31))
                assert_frame_equal(run_backtest(loaded).equity, run_backtest(bars).equity)
                assert_frame_equal(run_backtest(reused).trades, run_backtest(bars).trades)
                self.assertEqual(path.read_bytes(), original_bytes)
                with sqlite3.connect(path) as conn:
                    conn.execute("UPDATE stock_daily SET data_source='Another' WHERE id=1")
                with self.assertRaises(ValueError):
                    load_sqlite(str(path), '000001', '2025-01-02', '2025-12-31')
                with self.assertRaises(sqlite3.OperationalError):
                    load_sqlite(str(Path(tmp) / 'missing.db'), '000001', '2025-01-02', '2025-12-31')
                self.assertFalse((Path(tmp) / 'missing.db').exists())
            finally:
                DatabaseManager.reset_instance()

    def test_csv_roundtrip_and_no_network_in_demo(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'prices.csv'
            bars = demo_bars()
            bars.to_csv(path, index=False)
            with patch('socket.socket', side_effect=AssertionError('Network forbidden')):
                loaded = load_csv(str(path))
                result = run_backtest(loaded)
                again = run_backtest(loaded)
            assert_frame_equal(result.equity, again.equity)
            self.assertEqual(result.metrics, again.metrics)

    def test_cli_outputs_are_byte_reproducible_without_network(self):
        from src.backtest.__main__ import main
        with tempfile.TemporaryDirectory() as tmp:
            directories = [Path(tmp) / 'first', Path(tmp) / 'second']
            for directory in directories:
                with patch('sys.argv', ['backtest', '--demo', '--output', str(directory)]), \
                        patch('socket.socket', side_effect=AssertionError('Network forbidden')), \
                        contextlib.redirect_stdout(io.StringIO()):
                    main()
            for filename in ['input.csv', 'signals.csv', 'trades.csv', 'equity.csv', 'summary.json']:
                self.assertEqual((directories[0] / filename).read_bytes(), (directories[1] / filename).read_bytes())

    def test_cli_refuses_unknown_real_price_basis(self):
        from src.backtest.__main__ import main
        with patch('sys.argv', ['backtest', '--csv', 'not-opened.csv']), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exc:
                main()
        self.assertEqual(exc.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
