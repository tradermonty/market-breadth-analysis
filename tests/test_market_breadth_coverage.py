import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import market_breadth as mb
from market_breadth import (
    calculate_above_ma,
    compute_breadth_coverage,
    enforce_coverage_threshold,
    extract_chart_data,
)


def _make_dates(n):
    return pd.bdate_range('2023-01-02', periods=n)


def _make_stock_data(n=260, seed=0):
    dates = _make_dates(n)
    rng = np.random.default_rng(seed)
    base = np.linspace(100, 300, n)
    df = pd.DataFrame(
        {
            'A': base,
            'B': base + np.cumsum(rng.normal(0, 1, n)),
        },
        index=dates,
    )
    return df


class TestCalculateAboveMaDenominator(unittest.TestCase):
    def test_01_one_eligible_one_missing_denominator(self):
        data = _make_stock_data()
        # Stock B has a missing price on the final day -> ineligible, not below-MA.
        data.iloc[-1, data.columns.get_loc('B')] = np.nan
        above_ma = calculate_above_ma(data, window=200)

        self.assertTrue(pd.api.types.is_float_dtype(above_ma.dtypes.iloc[0]))
        last_date = above_ma.index[-1]
        # A is eligible and above its 200-day MA on the last day.
        self.assertEqual(above_ma.loc[last_date, 'A'], 1.0)
        self.assertTrue(np.isnan(above_ma.loc[last_date, 'B']))
        self.assertEqual(above_ma.mean(axis=1).loc[last_date], 1.0)

        coverage = compute_breadth_coverage(above_ma)
        row = coverage.loc[last_date]
        self.assertEqual(int(row['eligible_count']), 1)
        self.assertEqual(int(row['missing_count']), 1)
        self.assertEqual(int(row['above_count']), 1)
        self.assertEqual(row['coverage'], 0.5)

    def test_02_ma_warmup_early_days_nan(self):
        data = _make_stock_data(n=260)
        above_ma = calculate_above_ma(data, window=200)
        # Before the window is full, no stock is MA-eligible -> NaN.
        self.assertTrue(above_ma.iloc[:199].isna().all().all())
        coverage = compute_breadth_coverage(above_ma)
        self.assertEqual(coverage.iloc[0]['coverage'], 0.0)

    def test_03_ipo_history_partial(self):
        data = _make_stock_data(n=300, seed=1)
        # Stock C only exists for the final 200 days (IPO history).
        data['C'] = np.nan
        data.iloc[-200:, data.columns.get_loc('C')] = np.linspace(50, 200, 200)
        above_ma = calculate_above_ma(data, window=200)
        # Before C's window fills, C is ineligible (NaN).
        self.assertTrue(np.isnan(above_ma.iloc[-201, above_ma.columns.get_loc('C')]))
        # Once 200 consecutive values exist, C is eligible on the last day.
        self.assertFalse(np.isnan(above_ma.loc[above_ma.index[-1], 'C']))

    def test_04_gap_missing_price_not_below(self):
        data = _make_stock_data(n=260)
        data.iloc[-100, data.columns.get_loc('A')] = np.nan
        above_ma = calculate_above_ma(data, window=200)
        gap_date = above_ma.index[-100]
        # Missing price on a gap day is ineligible (NaN), not counted as below-MA (0.0).
        self.assertTrue(np.isnan(above_ma.loc[gap_date, 'A']))

    def test_05_all_missing_day_coverage_zero(self):
        data = _make_stock_data(n=260)
        data.iloc[-1] = np.nan
        above_ma = calculate_above_ma(data, window=200)
        last_date = above_ma.index[-1]
        coverage = compute_breadth_coverage(above_ma)
        self.assertEqual(coverage.loc[last_date, 'coverage'], 0.0)
        with self.assertRaises(ValueError):
            enforce_coverage_threshold(coverage, 0.9, '200-day breadth', last_date)

    def test_08_round_trip_dtype_and_nan(self):
        data = _make_stock_data(n=260)
        data.iloc[-1, data.columns.get_loc('B')] = np.nan
        above_ma = calculate_above_ma(data, window=200)
        # Returned values are float64 with 1.0/0.0/NaN, never bool/object.
        self.assertTrue(pd.api.types.is_float_dtype(above_ma.dtypes.iloc[0]))
        last_date = above_ma.index[-1]
        self.assertEqual(above_ma.loc[last_date, 'A'], 1.0)
        self.assertTrue(np.isnan(above_ma.loc[last_date, 'B']))
        # Downstream mean is the eligible ratio.
        self.assertEqual(above_ma.mean(axis=1).loc[last_date], 1.0)

    def test_09_eligibility_independent_per_window(self):
        data = _make_stock_data(n=260)
        data.iloc[-1, data.columns.get_loc('B')] = np.nan
        above_200 = calculate_above_ma(data, window=200)
        above_50 = calculate_above_ma(data, window=50)
        last_date = above_200.index[-1]
        cov_200 = compute_breadth_coverage(above_200)
        cov_50 = compute_breadth_coverage(above_50)
        # 200-day and 50-day eligibility are recorded independently.
        self.assertEqual(cov_200.loc[last_date, 'coverage'], 0.5)
        self.assertEqual(cov_50.loc[last_date, 'coverage'], 0.5)
        self.assertEqual(
            compute_breadth_coverage(above_200).columns.tolist(),
            ['constituent_count', 'eligible_count', 'missing_count', 'above_count', 'coverage'],
        )


class TestComputeBreadthCoverage(unittest.TestCase):
    def test_06_counts(self):
        dates = _make_dates(5)
        above_ma = pd.DataFrame(
            {
                'A': [1.0, 0.0, np.nan, 1.0, np.nan],
                'B': [0.0, 1.0, 1.0, np.nan, np.nan],
            },
            index=dates,
        )
        coverage = compute_breadth_coverage(above_ma)
        self.assertEqual(coverage['constituent_count'].iloc[0], 2)
        self.assertEqual(int(coverage['eligible_count'].iloc[0]), 2)
        self.assertEqual(int(coverage['missing_count'].iloc[0]), 0)
        self.assertEqual(int(coverage['above_count'].iloc[0]), 1)
        self.assertEqual(coverage['coverage'].iloc[0], 1.0)
        # Middle day: A missing, B eligible -> coverage 0.5.
        self.assertEqual(coverage['coverage'].iloc[2], 0.5)
        self.assertEqual(int(coverage['missing_count'].iloc[2]), 1)

    def test_12_fetch_failure_counts_against_expected_constituents(self):
        # P1 regression: a constituent that failed to fetch is dropped from above_ma entirely.
        # Coverage must be measured against the EXPECTED constituent list (len(ticker_list)),
        # otherwise the failed stock vanishes from the denominator and coverage inflates to 100%.
        dates = _make_dates(210)
        # Only stock A present; stock B failed to fetch (its column is absent).
        above_ma = pd.DataFrame({'A': np.linspace(100, 300, 210)}, index=dates)
        coverage = compute_breadth_coverage(above_ma, constituent_count=2)
        last_date = dates[-1]
        self.assertEqual(int(coverage['eligible_count'].loc[last_date]), 1)
        self.assertEqual(int(coverage['missing_count'].loc[last_date]), 1)
        self.assertEqual(coverage['coverage'].loc[last_date], 0.5)
        with self.assertRaises(mb.CoverageThresholdError):
            enforce_coverage_threshold(coverage, 0.9, '200-day breadth', last_date)


class TestEnforceCoverageThreshold(unittest.TestCase):
    def test_07_accepted_when_at_or_above_threshold(self):
        dates = _make_dates(2)
        coverage = pd.DataFrame(
            {'eligible_count': [9, 10], 'missing_count': [1, 0], 'coverage': [0.9, 1.0]},
            index=dates,
        )
        # At threshold -> no raise.
        enforce_coverage_threshold(coverage, 0.9, '200-day breadth', dates[0])

    def test_07b_rejected_below_threshold(self):
        dates = _make_dates(2)
        coverage = pd.DataFrame(
            {'eligible_count': [5, 10], 'missing_count': [5, 0], 'coverage': [0.5, 1.0]},
            index=dates,
        )
        with self.assertRaises(ValueError):
            enforce_coverage_threshold(coverage, 0.9, '200-day breadth', dates[0])

    def test_07c_raises_when_market_date_missing_from_index(self):
        dates = _make_dates(2)
        coverage = pd.DataFrame(
            {'eligible_count': [10, 10], 'missing_count': [0, 0], 'coverage': [1.0, 1.0]},
            index=dates,
        )
        with self.assertRaises(ValueError) as ctx:
            enforce_coverage_threshold(coverage, 0.9, '200-day breadth', dates[0] + pd.Timedelta(days=5))
        self.assertIn('not computable', str(ctx.exception))

    def test_07d_coverage_failure_is_coverage_threshold_error(self):
        dates = _make_dates(2)
        coverage = pd.DataFrame(
            {'eligible_count': [5, 10], 'missing_count': [5, 0], 'coverage': [0.5, 1.0]},
            index=dates,
        )
        with self.assertRaises(mb.CoverageThresholdError):
            enforce_coverage_threshold(coverage, 0.9, '200-day breadth', dates[0])


class TestMainPublicationGate(unittest.TestCase):
    def test_11_below_threshold_propagates_and_never_plots(self):
        dates = _make_dates(210)
        start = dates[0].strftime('%Y-%m-%d')
        end = dates[-1].strftime('%Y-%m-%d')
        stock = pd.DataFrame({'A': np.linspace(100, 300, 210), 'B': np.linspace(90, 280, 210)}, index=dates)
        # Stock B is missing on the final day -> coverage 0.5 below the 0.9 threshold.
        stock.iloc[-1, stock.columns.get_loc('B')] = np.nan
        sp500 = pd.Series(np.linspace(4000, 5000, 210), index=dates)

        with (
            mock.patch.object(mb, 'get_sp500_tickers_from_fmp', return_value=['A', 'B']),
            mock.patch.object(mb, 'get_sp500_price_data', return_value=sp500),
            mock.patch.object(mb, 'get_multiple_stock_data', return_value=stock),
            mock.patch.object(mb, 'plot_breadth_and_sp500_with_peaks') as plot_mock,
            mock.patch.object(mb, 'export_chart_data_to_csv') as export_mock,
            mock.patch('sys.argv', ['market_breadth.py', '--use_saved_data', '--start_date', start, '--end_date', end]),
        ):
            with self.assertRaises(mb.CoverageThresholdError):
                mb.main()
            # Below-threshold run must not plot or export anything.
            plot_mock.assert_not_called()
            export_mock.assert_not_called()

    def test_14_dropped_constituent_inflates_coverage(self):
        # P1 regress: a constituent that fails to fetch is dropped from stock_data entirely
        # (get_multiple_stock_data only returns successful tickers). The coverage gate must still
        # count that stock as missing against len(ticker_list), so a 1-of-2 fetch does NOT pass.
        dates = _make_dates(210)
        start = dates[0].strftime('%Y-%m-%d')
        end = dates[-1].strftime('%Y-%m-%d')
        # Only stock A returned; B failed to fetch (column absent).
        stock = pd.DataFrame({'A': np.linspace(100, 300, 210)}, index=dates)
        sp500 = pd.Series(np.linspace(4000, 5000, 210), index=dates)

        with (
            mock.patch.object(mb, 'get_sp500_tickers_from_fmp', return_value=['A', 'B']),
            mock.patch.object(mb, 'get_sp500_price_data', return_value=sp500),
            mock.patch.object(mb, 'get_multiple_stock_data', return_value=stock),
            mock.patch.object(mb, 'plot_breadth_and_sp500_with_peaks') as plot_mock,
            mock.patch.object(mb, 'export_chart_data_to_csv') as export_mock,
            mock.patch('sys.argv', ['market_breadth.py', '--use_saved_data', '--start_date', start, '--end_date', end]),
        ):
            with self.assertRaises(mb.CoverageThresholdError):
                mb.main()
            # Even when the coverage gate fails on a dropped constituent, nothing is published.
            plot_mock.assert_not_called()
            export_mock.assert_not_called()


class TestShortIpoRetention(unittest.TestCase):
    def test_15_short_ipo_history_is_retained_for_50day_breadth(self):
        # R1 regress: get_multiple_stock_data must not discard a valid short IPO history
        # (shorter than the 200-day warmup). Such a ticker can still contribute to the 50-day
        # breadth, so it has to remain in the frame; only then does the coverage gate count it
        # as eligible (50-day) / missing (200-day) rather than silently dropping it.
        def fake_fetch(ticker, start, end):
            n = 60 if ticker == 'IPO' else 260
            idx = pd.bdate_range('2023-01-02', periods=n)
            return pd.Series(np.linspace(100, 200, n), index=idx)

        with mock.patch.object(mb, 'fetch_price_data_fmp', side_effect=fake_fetch):
            combined = mb.get_multiple_stock_data(['A', 'IPO'], '2024-01-01', '2024-12-31', use_saved_data=False)

        self.assertIn('A', combined.columns)
        self.assertIn('IPO', combined.columns)
        # 200-day: the short history never reaches the warmup -> all ineligible (missing).
        above_200 = calculate_above_ma(combined, window=200)
        self.assertTrue(above_200['IPO'].isna().all())
        # 50-day: the short history is long enough -> contributes eligible observations.
        above_50 = calculate_above_ma(combined, window=50)
        self.assertTrue(above_50['IPO'].notna().any())


class TestCachedUniverseReconciliation(unittest.TestCase):
    def test_16_replacement_constituent_in_cache_blocks_publication(self):
        # R1-P1 regression (reviewer case 1): the returned/cached frame holds a DEPARTED
        # constituent C in place of the missing current constituent B. main() must reconcile
        # the frame against the requested ticker identities so B is counted as missing, giving
        # coverage 0.5 (not 1.0) and blocking publication.
        dates = _make_dates(210)
        start = dates[0].strftime('%Y-%m-%d')
        end = dates[-1].strftime('%Y-%m-%d')
        # Cache returns A and C; expected universe is [A, B] -> B is missing.
        cache = pd.DataFrame(
            {'A': np.linspace(100, 300, 210), 'C': np.linspace(80, 270, 210)},
            index=dates,
        )
        sp500 = pd.Series(np.linspace(4000, 5000, 210), index=dates)

        with (
            mock.patch.object(mb, 'get_sp500_tickers_from_fmp', return_value=['A', 'B']),
            mock.patch.object(mb, 'get_sp500_price_data', return_value=sp500),
            mock.patch.object(mb, 'get_multiple_stock_data', return_value=cache),
            mock.patch.object(mb, 'plot_breadth_and_sp500_with_peaks') as plot_mock,
            mock.patch.object(mb, 'export_chart_data_to_csv') as export_mock,
            mock.patch('sys.argv', ['market_breadth.py', '--use_saved_data', '--start_date', start, '--end_date', end]),
        ):
            with self.assertRaises(mb.CoverageThresholdError):
                mb.main()
            plot_mock.assert_not_called()
            export_mock.assert_not_called()

    def test_17_extra_cached_column_excluded_and_coverage_valid(self):
        # R1-P1 regression (reviewer case 2): the cache contains an EXTRA obsolete column C
        # that is not in the current universe (expected [A, B]). main() must exclude C from
        # the breadth frame so coverage = eligible/2 stays in [0, 1] and missing >= 0.
        dates = _make_dates(210)
        start = dates[0].strftime('%Y-%m-%d')
        end = dates[-1].strftime('%Y-%m-%d')
        cache = pd.DataFrame(
            {
                'A': np.linspace(100, 300, 210),
                'B': np.linspace(90, 280, 210),
                'C': np.linspace(80, 270, 210),
            },
            index=dates,
        )
        sp500 = pd.Series(np.linspace(4000, 5000, 210), index=dates)
        captured = {}

        def fake_plot(above_ma_200, *args, **kwargs):
            captured['cols'] = list(above_ma_200.columns)
            chart = {'breadth_index_200': above_ma_200.mean(axis=1)}
            captured['chart'] = chart
            return None, chart

        with (
            mock.patch.object(mb, 'get_sp500_tickers_from_fmp', return_value=['A', 'B']),
            mock.patch.object(mb, 'get_sp500_price_data', return_value=sp500),
            mock.patch.object(mb, 'get_multiple_stock_data', return_value=cache),
            mock.patch.object(mb, 'plot_breadth_and_sp500_with_peaks', side_effect=fake_plot) as plot_mock,
            mock.patch.object(mb, 'export_chart_data_to_csv') as export_mock,
            mock.patch(
                'sys.argv',
                ['market_breadth.py', '--use_saved_data', '--start_date', start, '--end_date', end, '--no_export_csv'],
            ),
        ):
            # Coverage 1.0 (A and B eligible) passes the gate; main() must complete normally.
            mb.main()
        # Obsolete cached column C is excluded from the breadth frame.
        self.assertEqual(captured['cols'], ['A', 'B'])
        self.assertNotIn('C', captured['cols'])
        self.assertTrue(plot_mock.called)
        # With --no_export_csv nothing is exported.
        export_mock.assert_not_called()
        # Coverage stays in [0, 1] and missing count is non-negative over the current universe.
        cov = captured['chart']['coverage_200']
        self.assertTrue(cov['coverage'].between(0, 1).all())
        self.assertTrue((cov['missing_count'] >= 0).all())


class TestExportCoverageColumns(unittest.TestCase):
    def test_10_preserves_existing_column_order(self):
        dates = _make_dates(210)
        above_ma = pd.DataFrame(
            {'A': np.linspace(100, 300, 210), 'B': np.linspace(90, 280, 210)},
            index=dates,
        )
        sp500 = pd.Series(np.linspace(4000, 5000, 210), index=dates)
        chart_data = extract_chart_data(above_ma, sp500, short_ma_period=10)
        coverage_200 = compute_breadth_coverage(above_ma)
        chart_data['coverage_200'] = coverage_200

        with tempfile.TemporaryDirectory() as tmp:
            original = mb.reports_dir
            mb.reports_dir = pathlib.Path(tmp)
            try:
                mb.export_chart_data_to_csv(chart_data, 10, filename='out.csv')
                with open(pathlib.Path(tmp) / 'out.csv') as f:
                    header = f.readline().strip().split(',')
            finally:
                mb.reports_dir = original

        base_cols = [
            'Date',
            'S&P500_Price',
            'Breadth_Index_Raw',
            'Breadth_Index_200MA',
            'Breadth_Index_10MA',
            'Breadth_200MA_Trend',
            'Bearish_Signal',
            'Is_Peak',
            'Is_Trough',
            'Is_Trough_10MA_Below_04',
        ]
        # Base columns appear first in order; coverage columns appended after.
        self.assertEqual(header[: len(base_cols)], base_cols)
        self.assertEqual(header[-4:], ['Eligible_Count_200', 'Missing_Count_200', 'Above_Count_200', 'Coverage_200'])

    def test_13_50day_coverage_appended_after_all_existing_columns(self):
        # P2 regression: with 50-day columns present, the new coverage columns must be appended
        # at the very END (after the base AND the 50-day columns), not inserted mid-stream.
        dates = _make_dates(210)
        stock = pd.DataFrame(
            {'A': np.linspace(100, 300, 210), 'B': np.linspace(90, 280, 210)},
            index=dates,
        )
        sp500 = pd.Series(np.linspace(4000, 5000, 210), index=dates)
        above_200 = calculate_above_ma(stock, window=200)
        above_50 = calculate_above_ma(stock, window=50)
        breadth_50 = above_50.mean(axis=1)
        chart_data = extract_chart_data(above_200, sp500, short_ma_period=10)
        chart_data['chart_data_50'] = {
            'breadth_index_50': breadth_50,
            'breadth_ma_50_long': breadth_50.rolling(50).mean(),
            'breadth_ma_50_short': breadth_50.rolling(10).mean(),
            'breadth_ma_50_trend': breadth_50.rolling(50).mean().diff(),
            'peaks_50': [],
            'troughs_50': [],
            'peaks_avg_50': 0.0,
            'troughs_avg_50': 0.0,
        }
        chart_data['coverage_200'] = compute_breadth_coverage(above_200)
        chart_data['coverage_50'] = compute_breadth_coverage(above_50)

        with tempfile.TemporaryDirectory() as tmp:
            original = mb.reports_dir
            mb.reports_dir = pathlib.Path(tmp)
            try:
                mb.export_chart_data_to_csv(chart_data, 10, filename='out.csv')
                with open(pathlib.Path(tmp) / 'out.csv') as f:
                    header = f.readline().strip().split(',')
            finally:
                mb.reports_dir = original

        base_cols = [
            'Date',
            'S&P500_Price',
            'Breadth_Index_Raw',
            'Breadth_Index_200MA',
            'Breadth_Index_10MA',
            'Breadth_200MA_Trend',
            'Bearish_Signal',
            'Is_Peak',
            'Is_Trough',
            'Is_Trough_10MA_Below_04',
        ]
        existing_50 = [
            'Breadth_50_Index_Raw',
            'Breadth_50_Index_50MA',
            'Breadth_50_Index_10MA',
            'Breadth_50_MA_Trend',
            'Bearish_Signal_50',
            'Is_Peak_50',
            'Is_Trough_50',
        ]
        tail_coverage = [
            'Eligible_Count_200',
            'Missing_Count_200',
            'Above_Count_200',
            'Coverage_200',
            'Eligible_Count_50',
            'Missing_Count_50',
            'Above_Count_50',
            'Coverage_50',
        ]
        # Existing base + 50-day columns keep their order; coverage columns appended at the end.
        self.assertEqual(header[: len(base_cols) + len(existing_50)], base_cols + existing_50)
        self.assertEqual(header[-len(tail_coverage) :], tail_coverage)


if __name__ == '__main__':
    unittest.main()
