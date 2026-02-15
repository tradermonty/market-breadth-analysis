import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from rsp_breadth_strategy.weights import compute_target_weights


class TestWeights(unittest.TestCase):
    def test_riskon_base_weights(self):
        w = compute_target_weights(regime='RiskOn', r_trend=1, x_trend=1)
        self.assertAlmostEqual(w['SPY'], 0.45)
        self.assertAlmostEqual(w['RSP'], 0.40)
        self.assertAlmostEqual(w['XLE'], 0.10)
        self.assertAlmostEqual(w['SGOV'], 0.05)
        self.assertAlmostEqual(sum(w.values()), 1.0)

    def test_r_trend_off_reduces_rsp(self):
        w = compute_target_weights(regime='RiskOn', r_trend=0, x_trend=1)
        self.assertAlmostEqual(w['SPY'], 0.55)
        self.assertAlmostEqual(w['RSP'], 0.25)
        self.assertAlmostEqual(w['XLE'], 0.10)
        self.assertAlmostEqual(w['SGOV'], 0.10)

    def test_x_trend_off_moves_xle_to_sgov(self):
        w = compute_target_weights(regime='RiskOn', r_trend=1, x_trend=0)
        self.assertAlmostEqual(w['SPY'], 0.45)
        self.assertAlmostEqual(w['RSP'], 0.40)
        self.assertAlmostEqual(w['XLE'], 0.00)
        self.assertAlmostEqual(w['SGOV'], 0.15)

    def test_recovery_bonus_with_both_trends(self):
        w = compute_target_weights(regime='Recovery', r_trend=1, x_trend=1)
        self.assertAlmostEqual(w['SPY'], 0.20)
        self.assertAlmostEqual(w['RSP'], 0.48)
        self.assertAlmostEqual(w['XLE'], 0.12)
        self.assertAlmostEqual(w['SGOV'], 0.20)

    def test_drawdown_overlay(self):
        w = compute_target_weights(regime='RiskOn', r_trend=1, x_trend=1, portfolio_drawdown=-0.13)
        self.assertAlmostEqual(w['SPY'], 0.45)
        self.assertAlmostEqual(w['RSP'], 0.30)
        self.assertAlmostEqual(w['XLE'], 0.05)
        self.assertAlmostEqual(w['SGOV'], 0.20)

    def test_xle_stop_overlay(self):
        w = compute_target_weights(regime='RiskOn', r_trend=1, x_trend=1, xle_monthly_return=-0.13)
        self.assertAlmostEqual(w['XLE'], 0.0)
        self.assertAlmostEqual(w['SGOV'], 0.15)


if __name__ == '__main__':
    unittest.main()
