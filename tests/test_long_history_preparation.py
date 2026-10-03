import json

import numpy as np
import pandas as pd

from scripts.prepare_long_history import prepare


def test_membership_effective_date_and_missing_history_denominators(tmp_path):
    dates = pd.bdate_range('1990-01-01', '1994-01-31')
    (tmp_path / 'prices').mkdir()
    manifest = []
    for symbol, start in [('A', '1990-01-01'), ('NEW', '1993-01-04')]:
        index = dates[dates >= start]
        frame = pd.DataFrame({'date': index, 'adjClose': np.arange(len(index)) + 100.0})
        frame.to_csv(tmp_path / 'prices' / f'{symbol}.csv.gz', index=False, compression='gzip')
        manifest.append({'symbol': symbol, 'usable': True})
    spy_dates = dates[dates >= '1993-01-04']
    spy = [{'date': str(date.date()), 'adjClose': 100.0} for date in spy_dates]
    for filename, payload in [
        ('acquisition_manifest.json', manifest),
        ('current_constituents.json', [{'symbol': 'A'}, {'symbol': 'NEW'}]),
        ('historical_constituents.json', [{'date': '1993-01-05', 'symbol': 'NEW', 'removedTicker': 'OLD'}]),
        ('spy_legacy.json', {'historical': spy}),
        ('universe.json', ['A', 'NEW', 'OLD']),
    ]:
        (tmp_path / filename).write_text(json.dumps(payload))
    prepare(tmp_path)
    fixed = pd.read_csv(tmp_path / 'current_fixed.csv').set_index('Date')
    eligible = pd.read_csv(tmp_path / 'current_eligible.csv').set_index('Date')
    historical = pd.read_csv(tmp_path / 'historical_reconstructed.csv').set_index('Date')
    assert fixed.loc['1993-01-04', 'Breadth_Index_Raw'] == 0.5
    assert eligible.loc['1993-01-04', 'Breadth_Index_Raw'] == 1
    assert historical.loc['1993-01-04', 'Observed_Count'] == 1
    assert historical.loc['1993-01-05', 'Observed_Count'] == 2
    assert historical.loc['1993-01-05', 'Eligible_Count'] == 1
    assert historical.loc['1993-01-05', 'Constituent_Count'] == 2
    assert historical.loc['1993-01-05', 'Eligibility_Coverage'] == 0.5
    assert historical.iloc[-1]['Eligibility_Coverage'] == 1
