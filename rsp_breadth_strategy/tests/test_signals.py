import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from rsp_breadth_strategy.signals import build_signal_frame, classify_regime, classify_regime_series


class TestSignals(unittest.TestCase):
    def test_classify_regime_matrix(self):
        self.assertEqual(classify_regime(raw=0.40, ma8=0.45, ma200=0.50), 'Recovery')
        self.assertEqual(classify_regime(raw=0.65, ma8=0.66, ma200=0.60), 'RiskOn')
        self.assertEqual(classify_regime(raw=0.80, ma8=0.83, ma200=0.70), 'Overheat')
        self.assertEqual(classify_regime(raw=0.80, ma8=0.70, ma200=0.70), 'Deterioration')

    def test_build_signal_frame_adds_ratios_and_trends(self):
        dates = pd.date_range('2024-01-31', periods=12, freq='ME')
        spy = [100.0] * 12
        rsp = [100.0] * 10 + [110.0, 120.0]
        xle = [100.0] * 10 + [90.0, 80.0]

        frame = pd.DataFrame(
            {
                'Date': dates,
                'SPY': spy,
                'RSP': rsp,
                'XLE': xle,
                'Breadth_Index_Raw': [0.55] * 11 + [0.80],
                'Breadth_Index_8MA': [0.56] * 11 + [0.82],
                'Breadth_Index_200MA': [0.50] * 12,
            }
        )

        out = build_signal_frame(frame, ratio_ma=10)

        self.assertIn('r_ratio', out.columns)
        self.assertIn('x_ratio', out.columns)
        self.assertIn('r_trend', out.columns)
        self.assertIn('x_trend', out.columns)
        self.assertIn('regime', out.columns)

        last = out.iloc[-1]
        self.assertEqual(int(last['r_trend']), 1)
        self.assertEqual(int(last['x_trend']), 0)
        self.assertEqual(last['regime'], 'Overheat')


class TestClassifyRegimeSeries(unittest.TestCase):
    """Tests for classify_regime_series with hysteresis and overrides."""

    def test_overheat_hysteresis_entry_and_exit(self):
        """Overheat enters at >0.75, exits only when <=0.70."""
        raw = pd.Series([0.60, 0.76, 0.73, 0.71, 0.70, 0.60])
        ma8 = pd.Series([0.55, 0.80, 0.80, 0.80, 0.80, 0.55])
        ma200 = pd.Series([0.50, 0.50, 0.50, 0.50, 0.50, 0.50])

        result = classify_regime_series(raw, ma8, ma200)

        self.assertEqual(result.iloc[0], 'RiskOn')
        self.assertEqual(result.iloc[1], 'Overheat')
        # 0.73 is between exit(0.70) and entry(0.75) -> stays Overheat
        self.assertEqual(result.iloc[2], 'Overheat')
        # 0.71 still above exit threshold -> stays Overheat
        self.assertEqual(result.iloc[3], 'Overheat')
        # 0.70 <= exit threshold -> exits Overheat
        self.assertNotEqual(result.iloc[4], 'Overheat')
        # 0.60 clearly not Overheat
        self.assertNotEqual(result.iloc[5], 'Overheat')

    def test_overheat_no_reentry_below_threshold(self):
        """After exiting Overheat at <=0.70, does not re-enter below 0.75."""
        raw = pd.Series([0.76, 0.70, 0.73, 0.74])
        ma8 = pd.Series([0.80, 0.80, 0.80, 0.80])
        ma200 = pd.Series([0.50, 0.50, 0.50, 0.50])

        result = classify_regime_series(raw, ma8, ma200)

        self.assertEqual(result.iloc[0], 'Overheat')
        # Exits at 0.70
        self.assertNotEqual(result.iloc[1], 'Overheat')
        # 0.73 < 0.75 entry -> should NOT be Overheat
        self.assertNotEqual(result.iloc[2], 'Overheat')
        # 0.74 < 0.75 entry -> should NOT be Overheat
        self.assertNotEqual(result.iloc[3], 'Overheat')

    def test_bearish_signal_overrides_recovery(self):
        """Bearish_Signal=True forces any regime to Deterioration."""
        raw = pd.Series([0.40, 0.65, 0.76])
        ma8 = pd.Series([0.45, 0.66, 0.80])
        ma200 = pd.Series([0.50, 0.60, 0.50])
        bearish = pd.Series([True, True, True])

        result = classify_regime_series(raw, ma8, ma200, bearish_signal=bearish)

        # All should be Deterioration regardless of base classification
        self.assertEqual(result.iloc[0], 'Deterioration')  # would be Recovery
        self.assertEqual(result.iloc[1], 'Deterioration')  # would be RiskOn
        self.assertEqual(result.iloc[2], 'Deterioration')  # would be Overheat

    def test_breadth_trend_overrides_recovery(self):
        """Breadth_200MA_Trend=-1 forces Recovery -> Deterioration."""
        raw = pd.Series([0.40, 0.40, 0.65])
        ma8 = pd.Series([0.45, 0.45, 0.66])
        ma200 = pd.Series([0.50, 0.50, 0.60])
        trend = pd.Series([-1, 1, -1])

        result = classify_regime_series(raw, ma8, ma200, breadth_trend=trend)

        # trend=-1 + Recovery -> Deterioration
        self.assertEqual(result.iloc[0], 'Deterioration')
        # trend=1 + Recovery -> stays Recovery
        self.assertEqual(result.iloc[1], 'Recovery')
        # trend=-1 but RiskOn (not Recovery) -> stays RiskOn
        self.assertEqual(result.iloc[2], 'RiskOn')

    def test_backward_compat_no_optional_columns(self):
        """Without optional columns, behaves like point-wise classify_regime."""
        raw = pd.Series([0.40, 0.65, 0.80])
        ma8 = pd.Series([0.45, 0.66, 0.83])
        ma200 = pd.Series([0.50, 0.60, 0.70])

        result = classify_regime_series(raw, ma8, ma200)

        self.assertEqual(result.iloc[0], 'Recovery')
        self.assertEqual(result.iloc[1], 'RiskOn')
        self.assertEqual(result.iloc[2], 'Overheat')

    def test_build_signal_frame_uses_bearish_signal_column(self):
        """build_signal_frame picks up Bearish_Signal column when present."""
        dates = pd.date_range('2024-01-31', periods=12, freq='ME')
        frame = pd.DataFrame(
            {
                'Date': dates,
                'SPY': [100.0] * 12,
                'RSP': [100.0] * 12,
                'XLE': [100.0] * 12,
                'Breadth_Index_Raw': [0.40] * 12,
                'Breadth_Index_8MA': [0.45] * 12,
                'Breadth_Index_200MA': [0.50] * 12,
                'Bearish_Signal': [False] * 6 + [True] * 6,
            }
        )

        out = build_signal_frame(frame, ratio_ma=10)

        # First 6 rows: no bearish signal -> Recovery
        self.assertEqual(out.iloc[0]['regime'], 'Recovery')
        # Last 6 rows: bearish signal -> Deterioration
        self.assertEqual(out.iloc[-1]['regime'], 'Deterioration')


class TestBuildSignalFrameRegimeMode(unittest.TestCase):
    """Tests for regime_mode parameter in build_signal_frame."""

    def _make_frame(self, n=12):
        dates = pd.date_range('2024-01-31', periods=n, freq='ME')
        return pd.DataFrame(
            {
                'Date': dates,
                'SPY': [100.0] * n,
                'RSP': [100.0] * n,
                'XLE': [100.0] * n,
                'Breadth_Index_Raw': [0.40] * n,
                'Breadth_Index_8MA': [0.45] * n,
                'Breadth_Index_200MA': [0.50] * n,
                'Bearish_Signal': [False] * 6 + [True] * 6,
            }
        )

    def test_pointwise_ignores_bearish_signal(self):
        """regime_mode='pointwise' uses classify_regime which ignores Bearish_Signal."""
        frame = self._make_frame()
        out = build_signal_frame(frame, ratio_ma=10, regime_mode='pointwise')

        # With pointwise: raw=0.40, ma8=0.45, ma200=0.50 -> Recovery
        # Bearish_Signal column is ignored
        self.assertEqual(out.iloc[-1]['regime'], 'Recovery')

    def test_invalid_regime_mode_raises(self):
        """regime_mode='invalid' should raise ValueError."""
        frame = self._make_frame()
        with self.assertRaises(ValueError) as ctx:
            build_signal_frame(frame, ratio_ma=10, regime_mode='invalid')
        self.assertIn('invalid', str(ctx.exception))

    def test_overheat_params_passthrough(self):
        """Custom overheat_entry/exit are passed through to classify_regime_series."""
        dates = pd.date_range('2024-01-31', periods=12, freq='ME')
        frame = pd.DataFrame(
            {
                'Date': dates,
                'SPY': [100.0] * 12,
                'RSP': [100.0] * 12,
                'XLE': [100.0] * 12,
                'Breadth_Index_Raw': [0.55] * 10 + [0.78, 0.78],
                'Breadth_Index_8MA': [0.56] * 10 + [0.82, 0.82],
                'Breadth_Index_200MA': [0.50] * 12,
            }
        )
        # With default overheat_entry=0.75, raw=0.78 should enter Overheat
        out_default = build_signal_frame(frame.copy(), ratio_ma=10)
        self.assertEqual(out_default.iloc[-1]['regime'], 'Overheat')

        # With overheat_entry=0.80, raw=0.78 should NOT be Overheat
        out_high = build_signal_frame(
            frame.copy(), ratio_ma=10, overheat_entry=0.80, overheat_exit=0.75,
        )
        self.assertNotEqual(out_high.iloc[-1]['regime'], 'Overheat')


if __name__ == '__main__':
    unittest.main()
