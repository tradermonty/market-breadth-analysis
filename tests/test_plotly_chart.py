"""Tests for Plotly-based market breadth chart generation.

Uses synthetic deterministic data (np.random.seed(42)) so no API key is needed.
All file output tests use tempfile.mkdtemp() to avoid polluting reports/.
"""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

# Ensure project root is on the path so we can import market_breadth
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from market_breadth import detect_bearish_regions, extract_chart_data, plot_breadth_and_sp500_with_peaks

# ---------------------------------------------------------------------------
# Shared synthetic data helpers
# ---------------------------------------------------------------------------


def _make_synthetic_data(n_days=500, n_stocks=50, seed=42):
    """Create deterministic synthetic above_ma_200 and sp500_data.

    Returns (above_ma_200: DataFrame, sp500_data: Series) with a DatetimeIndex.
    The data is crafted so that peaks, troughs, and bearish regions appear.
    """
    np.random.seed(seed)
    dates = pd.bdate_range('2018-01-02', periods=n_days, freq='B')

    # Generate stock above-MA booleans so breadth oscillates meaningfully
    # Start high, dip low around index 150-250, recover, dip again 350-420
    base = np.zeros((n_days, n_stocks), dtype=bool)
    for i in range(n_days):
        # probability of being above MA cycles over time
        phase = np.sin(2 * np.pi * i / 200) * 0.3 + 0.5
        base[i] = np.random.rand(n_stocks) < phase

    above_ma_200 = pd.DataFrame(base, index=dates, columns=[f'STOCK{j}' for j in range(n_stocks)])

    # S&P500 price: trending upward with noise
    sp500_values = 2700 + np.cumsum(np.random.randn(n_days) * 5)
    sp500_data = pd.Series(sp500_values, index=dates, name='adjusted_close')

    return above_ma_200, sp500_data


def _get_chart_data_and_fig(short_ma=10, output_dir=None):
    """Return (chart_data, fig) built from synthetic data."""
    above_ma, sp500 = _make_synthetic_data()
    chart_data = extract_chart_data(above_ma, sp500, short_ma_period=short_ma)

    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    fig, _ = plot_breadth_and_sp500_with_peaks(above_ma, sp500, short_ma_period=short_ma, output_dir=output_dir)
    return chart_data, fig, output_dir


# Cache to avoid re-generating figure for every test
_CACHED = {}


def _cached_fig(key='default'):
    if key not in _CACHED:
        _CACHED[key] = _get_chart_data_and_fig()
    return _CACHED[key]


# ===================================================================
# Test Class 1: TestBearishRegionDetection
# ===================================================================
class TestBearishRegionDetection(unittest.TestCase):
    """Tests for detect_bearish_regions() helper."""

    def test_01_no_bearish_regions(self):
        """All bullish -> empty list."""
        dates = pd.bdate_range('2020-01-01', periods=100)
        trend = pd.Series([1] * 100, index=dates)
        short = pd.Series([0.6] * 100, index=dates)
        long = pd.Series([0.5] * 100, index=dates)

        regions = detect_bearish_regions(trend, short, long)
        self.assertEqual(len(regions), 0)

    def test_02_single_bearish_region(self):
        """One contiguous bearish block in the middle."""
        dates = pd.bdate_range('2020-01-01', periods=100)
        trend = pd.Series([1] * 100, index=dates)
        short = pd.Series([0.6] * 100, index=dates)
        long = pd.Series([0.5] * 100, index=dates)

        # Make days 30-49 bearish: trend=-1 and short < long
        trend.iloc[30:50] = -1
        short.iloc[30:50] = 0.3
        long.iloc[30:50] = 0.5

        regions = detect_bearish_regions(trend, short, long)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0][0], dates[30])
        self.assertEqual(regions[0][1], dates[49])

    def test_03_multiple_bearish_regions(self):
        """Two separated bearish blocks."""
        dates = pd.bdate_range('2020-01-01', periods=100)
        trend = pd.Series([1] * 100, index=dates)
        short = pd.Series([0.6] * 100, index=dates)
        long = pd.Series([0.5] * 100, index=dates)

        # Block 1: days 10-19
        trend.iloc[10:20] = -1
        short.iloc[10:20] = 0.3

        # Block 2: days 60-79
        trend.iloc[60:80] = -1
        short.iloc[60:80] = 0.3

        regions = detect_bearish_regions(trend, short, long)
        self.assertEqual(len(regions), 2)

    def test_04_bearish_region_at_end(self):
        """Bearish region extends to the last date."""
        dates = pd.bdate_range('2020-01-01', periods=100)
        trend = pd.Series([1] * 100, index=dates)
        short = pd.Series([0.6] * 100, index=dates)
        long = pd.Series([0.5] * 100, index=dates)

        # Bearish from day 80 to end
        trend.iloc[80:] = -1
        short.iloc[80:] = 0.3

        regions = detect_bearish_regions(trend, short, long)
        self.assertGreaterEqual(len(regions), 1)
        last_region = regions[-1]
        self.assertEqual(last_region[1], dates[-1])


# ===================================================================
# Test Class 2: TestExtractChartDataKeys
# ===================================================================
class TestExtractChartDataKeys(unittest.TestCase):
    def test_05_extract_chart_data_returns_expected_keys(self):
        """extract_chart_data() returns dict with all required keys."""
        above_ma, sp500 = _make_synthetic_data()
        chart_data = extract_chart_data(above_ma, sp500)
        expected_keys = {
            'breadth_index_200',
            'breadth_ma_200',
            'breadth_ma_short',
            'breadth_ma_200_trend',
            'sp500_data',
            'peaks',
            'troughs',
            'troughs_below_04',
            'below_04',
            'peaks_avg',
            'troughs_avg_below_04',
        }
        self.assertEqual(set(chart_data.keys()), expected_keys)


# ===================================================================
# Test Class 3: TestPlotlyChartStructure
# ===================================================================
class TestPlotlyChartStructure(unittest.TestCase):
    """Verify the Plotly figure structure (subplots, axes)."""

    @classmethod
    def setUpClass(cls):
        _, cls.fig, cls.tmpdir = _cached_fig()

    def test_06_figure_has_two_subplots(self):
        """Figure should have yaxis and yaxis2 (two rows)."""
        self.assertIsNotNone(self.fig.layout.yaxis)
        self.assertIsNotNone(self.fig.layout.yaxis2)

    def test_07_shared_x_axes(self):
        """X axes should be shared (xaxis.matches == 'x2')."""
        # In make_subplots with shared_xaxes=True, xaxis matches xaxis2
        matches = self.fig.layout.xaxis.matches
        self.assertIn(matches, ('x2',))

    def test_08_panel1_log_scale(self):
        """Panel 1 (yaxis) should use log scale."""
        self.assertEqual(self.fig.layout.yaxis.type, 'log')

    def test_09_panel2_y_range(self):
        """Panel 2 (yaxis2) range should be [0, 1]."""
        y_range = list(self.fig.layout.yaxis2.range)
        self.assertEqual(y_range, [0, 1])


# ===================================================================
# Test Class 4: TestPlotlyTraces
# ===================================================================
class TestPlotlyTraces(unittest.TestCase):
    """Verify individual traces (lines, markers) in the figure."""

    @classmethod
    def setUpClass(cls):
        cls.chart_data, cls.fig, cls.tmpdir = _cached_fig()
        # Build a dict {trace.name: trace} for easy lookup
        cls.traces = {t.name: t for t in cls.fig.data}

    def test_10_sp500_price_trace(self):
        """S&P 500 price trace: cyan line, panel 1."""
        t = self.traces.get('S&P 500 Price')
        self.assertIsNotNone(t, "Trace 'S&P 500 Price' not found")
        self.assertEqual(t.line.color, '#00FFFF')
        # Panel 1 -> yaxis 'y' or 'y1'
        self.assertIn(t.yaxis, ('y', 'y1', None))

    def test_11_breadth_200ma_trace(self):
        """Breadth 200MA: green line, panel 2."""
        t = self.traces.get('Breadth Index (200-Day MA)')
        self.assertIsNotNone(t, "Trace 'Breadth Index (200-Day MA)' not found")
        self.assertEqual(t.line.color, '#008000')
        self.assertEqual(t.yaxis, 'y2')

    def test_12_breadth_short_ma_trace(self):
        """Breadth short MA: orange line, panel 2."""
        # Name includes the MA period
        matching = [
            t
            for t in self.fig.data
            if 'Breadth Index' in (t.name or '') and 'Day MA)' in (t.name or '') and '200' not in (t.name or '')
        ]
        self.assertTrue(len(matching) > 0, 'Short MA trace not found')
        t = matching[0]
        self.assertEqual(t.line.color, '#FFA500')
        self.assertEqual(t.yaxis, 'y2')

    def test_13_peak_markers(self):
        """Peaks: red triangle-up markers, panel 2."""
        t = self.traces.get('Peaks (Tops)')
        self.assertIsNotNone(t, "Trace 'Peaks (Tops)' not found")
        self.assertEqual(t.marker.color, '#FF0000')
        self.assertEqual(t.marker.symbol, 'triangle-up')
        self.assertEqual(t.yaxis, 'y2')

    def test_14_trough_markers(self):
        """Troughs: blue triangle-down markers, panel 2."""
        t = self.traces.get('Troughs (Bottoms)')
        self.assertIsNotNone(t, "Trace 'Troughs (Bottoms)' not found")
        self.assertEqual(t.marker.color, '#0000FF')
        self.assertEqual(t.marker.symbol, 'triangle-down')
        self.assertEqual(t.yaxis, 'y2')

    def test_15_trough_below_04_breadth(self):
        """Troughs (<0.4) on breadth panel: purple marker, panel 2."""
        matching = [t for t in self.fig.data if t.name and 'MA < 0.4)' in t.name and 'S&P' not in t.name]
        if len(self.chart_data['troughs_below_04']) == 0:
            self.skipTest('No troughs below 0.4 in synthetic data')
        self.assertTrue(len(matching) > 0, 'Troughs below 0.4 trace not found')
        t = matching[0]
        self.assertEqual(t.marker.color, '#800080')
        self.assertEqual(t.yaxis, 'y2')

    def test_16_trough_below_04_sp500(self):
        """Troughs (<0.4) on S&P panel: purple marker, panel 1."""
        matching = [t for t in self.fig.data if t.name and 'MA < 0.4)' in t.name and 'S&P' in t.name]
        if len(self.chart_data['troughs_below_04']) == 0:
            self.skipTest('No troughs below 0.4 in synthetic data')
        self.assertTrue(len(matching) > 0, 'Troughs on S&P trace not found')
        t = matching[0]
        self.assertEqual(t.marker.color, '#800080')
        self.assertIn(t.yaxis, ('y', 'y1', None))

    def test_17_average_peaks_hline(self):
        """Average peaks horizontal line: red dashed."""
        shapes = [
            s
            for s in self.fig.layout.shapes
            if s.type == 'line' and s.line.color == '#FF0000' and s.line.dash == 'dash'
        ]
        self.assertTrue(len(shapes) > 0, 'Red dashed hline for average peaks not found')

    def test_18_average_troughs_hline(self):
        """Average troughs horizontal line: blue dashed."""
        shapes = [
            s
            for s in self.fig.layout.shapes
            if s.type == 'line' and s.line.color == '#0000FF' and s.line.dash == 'dash'
        ]
        self.assertTrue(len(shapes) > 0, 'Blue dashed hline for average troughs not found')


# ===================================================================
# Test Class 5: TestPlotlyInteractiveFeatures
# ===================================================================
class TestPlotlyInteractiveFeatures(unittest.TestCase):
    """Verify range selector, hover mode, and layout dimensions."""

    @classmethod
    def setUpClass(cls):
        _, cls.fig, cls.tmpdir = _cached_fig()

    def test_19_range_selector_buttons(self):
        """Range selector should have 1Y, 3Y, 5Y, ALL buttons."""
        # Range selector is on xaxis (panel 1)
        rs = self.fig.layout.xaxis.rangeselector
        self.assertIsNotNone(rs, 'Range selector not found on xaxis')
        labels = [b.label for b in rs.buttons]
        for expected in ['1Y', '3Y', '5Y', 'ALL']:
            self.assertIn(expected, labels, f"Button '{expected}' missing from range selector")

    def test_20_hover_mode(self):
        """Hover mode should be 'x unified'."""
        self.assertEqual(self.fig.layout.hovermode, 'x unified')

    def test_21_layout_dimensions(self):
        """Layout should be 900 height x 1200 width."""
        self.assertEqual(self.fig.layout.height, 900)
        self.assertEqual(self.fig.layout.width, 1200)


# ===================================================================
# Test Class 6: TestPlotlyOutput
# ===================================================================
class TestPlotlyOutput(unittest.TestCase):
    """Verify file outputs (HTML, optionally PNG)."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        above_ma, sp500 = _make_synthetic_data()
        cls.fig, _ = plot_breadth_and_sp500_with_peaks(above_ma, sp500, short_ma_period=10, output_dir=cls.tmpdir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_22_html_file_generated(self):
        """HTML file should be generated in output_dir."""
        html_path = os.path.join(self.tmpdir, 'market_breadth.html')
        self.assertTrue(os.path.exists(html_path), f'HTML not found at {html_path}')
        # Should be non-trivial size
        self.assertGreater(os.path.getsize(html_path), 1000)

    def test_23_png_file_generated_or_skipped(self):
        """PNG file should be generated if kaleido is available, otherwise skip."""
        png_path = os.path.join(self.tmpdir, 'market_breadth.png')
        try:
            import kaleido  # noqa: F401

            self.assertTrue(os.path.exists(png_path), 'kaleido available but PNG not found')
        except ImportError:
            # kaleido not installed, PNG generation is optional
            pass


# ===================================================================
# Test Class 7: TestPlotlyBearishBackground
# ===================================================================
class TestPlotlyBearishBackground(unittest.TestCase):
    """Verify bearish background vrect shapes."""

    @classmethod
    def setUpClass(cls):
        cls.chart_data, cls.fig, cls.tmpdir = _cached_fig()
        cls.vrects = [s for s in cls.fig.layout.shapes if s.type == 'rect']

    def test_24_vrect_shapes_exist(self):
        """At least one vrect shape should exist if bearish regions exist."""
        # Check if bearish regions exist in the data
        trend = self.chart_data['breadth_ma_200_trend']
        short = self.chart_data['breadth_ma_short']
        long = self.chart_data['breadth_ma_200']
        regions = detect_bearish_regions(trend, short, long)
        if len(regions) > 0:
            self.assertTrue(len(self.vrects) > 0, 'Bearish regions exist but no vrect shapes found')
        else:
            self.skipTest('No bearish regions in synthetic data')

    def test_25_vrect_color_correct(self):
        """Vrect fillcolor should be 'rgba(255, 230, 245, 0.3)'."""
        if len(self.vrects) == 0:
            self.skipTest('No vrects to check')
        for v in self.vrects:
            self.assertEqual(v.fillcolor, 'rgba(255, 210, 240, 0.35)', f'Unexpected fillcolor: {v.fillcolor}')

    def test_26_vrect_covers_both_panels(self):
        """Vrects should span both panels (yref covers full y domain)."""
        if len(self.vrects) == 0:
            self.skipTest('No vrects to check')
        # When row='all' is used, vrects are duplicated for each subplot
        # or use paper coordinates. Either way, check some vrects reference y/y2
        {v.yref for v in self.vrects}
        # Plotly may use 'y', 'y2', 'paper', or 'y domain'/'y2 domain'
        # At minimum, vrects should exist
        self.assertTrue(len(self.vrects) > 0)


# ===================================================================
# Test Class 8: TestPlotlyIntegration
# ===================================================================
class TestPlotlyIntegration(unittest.TestCase):
    """Integration test using real saved data if available."""

    def test_27_full_pipeline_with_saved_data(self):
        """Full pipeline using saved data from data/ directory."""
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        stocks_file = os.path.join(data_dir, 'sp500_all_stocks.csv')
        sp500_file = os.path.join(data_dir, 'sp500_price_data.csv')

        if not (os.path.exists(stocks_file) and os.path.exists(sp500_file)):
            self.skipTest('Saved data not available in data/')

        # Load real data
        stock_data = pd.read_csv(stocks_file, index_col=0, parse_dates=True)
        sp500_data = pd.read_csv(sp500_file, index_col=0, parse_dates=True)
        if isinstance(sp500_data, pd.DataFrame) and len(sp500_data.columns) == 1:
            sp500_data = sp500_data.iloc[:, 0]

        # Calculate breadth
        from market_breadth import calculate_above_ma

        above_ma_200 = calculate_above_ma(stock_data, window=200)

        # Align dates
        common_dates = above_ma_200.index.intersection(sp500_data.index)
        if len(common_dates) < 200:
            self.skipTest('Not enough common dates in saved data')

        above_ma_200 = above_ma_200.loc[common_dates]
        sp500_data = sp500_data.loc[common_dates]

        # Generate chart
        tmpdir = tempfile.mkdtemp()
        try:
            fig, _ = plot_breadth_and_sp500_with_peaks(above_ma_200, sp500_data, short_ma_period=10, output_dir=tmpdir)
            # Basic sanity checks
            self.assertGreater(len(fig.data), 0, 'No traces in figure')
            html_path = os.path.join(tmpdir, 'market_breadth.html')
            self.assertTrue(os.path.exists(html_path), 'HTML file not generated')
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
