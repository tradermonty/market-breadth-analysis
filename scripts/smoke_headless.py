"""Exercise saved-data analysis in a clean core-only environment, without network."""

import contextlib
import importlib.metadata
import io
import os
import socket
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def no_network(*_args, **_kwargs):
    raise AssertionError('Network attempted during headless smoke test')


def main():
    for package in ('kaleido', 'PyQt5', 'alpaca-trade-api'):
        try:
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
        raise AssertionError(f'{package} must not be installed in the core profile')

    import numpy as np
    import pandas as pd

    import market_breadth as mb

    rng = np.random.default_rng(42)
    dates = pd.bdate_range('2022-01-03', '2024-06-03')
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        data = root / 'data'
        reports = root / 'reports'
        data.mkdir()
        reports.mkdir()
        pd.DataFrame(
            {ticker: 200 + rng.normal(0, 1, len(dates)).cumsum() for ticker in ('AAPL', 'MSFT', 'GOOGL')},
            index=dates,
        ).to_csv(data / 'sp500_all_stocks.csv')
        pd.Series(400 + np.arange(len(dates)) * 0.1, index=dates, name='adjusted_close').to_csv(
            data / 'sp500_price_data.csv'
        )
        argv = ['market_breadth.py', '--use_saved_data', '--start_date', '2023-02-01', '--end_date', '2024-06-03']
        output = io.StringIO()
        with (
            mock.patch.object(mb, 'data_dir', data),
            mock.patch.object(mb, 'reports_dir', reports),
            mock.patch.object(sys, 'argv', argv),
            mock.patch.object(socket.socket, 'connect', no_network),
            mock.patch.object(socket, 'create_connection', no_network),
            mock.patch.dict(os.environ, {'FMP_API_KEY': ''}),
            contextlib.redirect_stdout(output),
            contextlib.chdir(root),
        ):
            mb.main()
        html = reports / 'market_breadth.html'
        assert html.exists() and 'Plotly.newPlot' in html.read_text()
        csvs = list(reports.glob('market_breadth_data_*.csv'))
        assert csvs and not pd.read_csv(csvs[0]).empty
        assert list(reports.glob('market_breadth_summary_*.csv'))
        assert not (reports / 'market_breadth.png').exists()
        assert 'PNG export skipped' in output.getvalue()
    print('PASS: core-only saved-data analysis emitted HTML/CSV; optional PNG failure was nonfatal; network blocked.')


if __name__ == '__main__':
    main()
