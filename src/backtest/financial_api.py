"""Offline Financial-API export boundary; no invented live endpoint contract."""

from typing import Mapping

import pandas as pd

from .data import validate_bars


def normalize_financial_api_export(records, field_map: Mapping[str, str], *,
                                   volume_unit: str, price_basis: str) -> pd.DataFrame:
    """Map a captured provider export to existing BaseFetcher/StockDaily columns.

    field_map maps ACTUAL provider field names -> date/open/high/low/close/volume.
    The caller extracts the rows after verifying the provider's response envelope.
    Explicit units/basis prevent silently treating lots as shares or adjusted bars
    as raw execution prices. This function does not authenticate or contact a server.
    """
    if volume_unit not in ('shares', 'lots'):
        raise ValueError('volume_unit must be shares or lots (100 shares per lot)')
    if price_basis not in ('unadjusted_no_actions', 'adjusted_research'):
        raise ValueError('Audit corporate actions or explicitly opt into adjusted research')
    if len(set(field_map.values())) != len(field_map):
        raise ValueError('field_map destinations must be unique')
    frame = pd.DataFrame(records)
    if not set(field_map).issubset(frame.columns):
        raise ValueError('Export does not match field_map')
    frame = frame.rename(columns=field_map)
    if frame.columns.duplicated().any():
        raise ValueError('field_map creates duplicate output columns')
    # String conversion avoids interpreting an integer YYYYMMDD as epoch nanoseconds.
    if 'date' in frame:
        frame['date'] = pd.to_datetime(frame['date'].astype(str), errors='raise')
    if 'volume' in frame:
        frame['volume'] = pd.to_numeric(frame['volume'], errors='raise') * (100 if volume_unit == 'lots' else 1)
    result = validate_bars(frame)
    result['data_source'] = 'FinancialAPIExport'
    result.attrs.update({'price_basis': price_basis, 'volume_unit': 'shares'})
    return result
