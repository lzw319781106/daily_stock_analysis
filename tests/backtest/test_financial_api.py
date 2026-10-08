import unittest

from src.backtest.financial_api import normalize_financial_api_export


class FinancialAPIExportTests(unittest.TestCase):
    def test_explicit_mapping_dates_units_and_price_basis(self):
        # Synthetic provider shape, NOT a claimed live Financial-API response.
        rows = [{'day': 20250102, 'o': 10, 'h': 12, 'l': 9, 'c': 11, 'v': 25}]
        mapping = {'day': 'date', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
        result = normalize_financial_api_export(rows, mapping, volume_unit='lots',
                                                price_basis='unadjusted_no_actions')
        self.assertEqual(str(result.iloc[0]['date'].date()), '2025-01-02')
        self.assertEqual(result.iloc[0]['volume'], 2500)
        self.assertEqual(result.attrs['price_basis'], 'unadjusted_no_actions')
        for unit, basis in [('unknown', 'unadjusted_no_actions'), ('shares', 'unknown')]:
            with self.assertRaises(ValueError):
                normalize_financial_api_export(rows, mapping, volume_unit=unit, price_basis=basis)

    def test_schema_drift_fails_instead_of_guessing(self):
        with self.assertRaises(ValueError):
            normalize_financial_api_export([{'new_field': 10}], {'old_field': 'close'},
                                           volume_unit='shares', price_basis='adjusted_research')


if __name__ == '__main__':
    unittest.main()
