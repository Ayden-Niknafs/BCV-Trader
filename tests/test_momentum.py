import json
import os
import tempfile
import unittest

from bcv_trader import momentum as m
from bcv_trader.valuation import ValuationFlag


class ComplianceGateTests(unittest.TestCase):
    def test_broad_funds_allowed(self):
        for sym in ("VTI", "VXUS", "BND"):
            st = m.compliance_status(sym)
            self.assertTrue(st.trade_call_allowed, sym)
            self.assertTrue(st.market_order_compatible)

    def test_sector_and_industry_consult_compliance(self):
        for sym in ("VGT", "XLK", "XBI"):
            self.assertFalse(m.compliance_status(sym).trade_call_allowed, sym)

    def test_uit_structure_consult_compliance(self):
        st = m.compliance_status("QQQ")
        self.assertFalse(st.trade_call_allowed)
        self.assertFalse(st.open_end)

    def test_style_factor_allowed_but_flagged(self):
        for sym in ("VUG", "MTUM"):
            st = m.compliance_status(sym)
            self.assertTrue(st.trade_call_allowed, sym)
            self.assertTrue(any("concentrated" in r or "tilt" in r for r in st.reasons), sym)

    def test_unknown_symbol_consults_compliance(self):
        self.assertFalse(m.compliance_status("ZZZZ").trade_call_allowed)


class IndicatorTests(unittest.TestCase):
    def test_sma_basic_and_offset(self):
        self.assertEqual(m.sma([1, 2, 3, 4], 2), 3.5)
        self.assertEqual(m.sma([1, 2, 3, 4], 2, offset=1), 2.5)

    def test_sma_insufficient_raises(self):
        with self.assertRaises(ValueError):
            m.sma([1, 2], 5)

    def test_rsi_all_gains_is_100(self):
        self.assertEqual(m.rsi(list(range(1, 40)), period=14), 100.0)

    def test_rsi_uptrend_high_downtrend_low(self):
        up = [100 + i for i in range(60)]
        down = [100 - i for i in range(60)]
        self.assertGreater(m.rsi(up), 70)
        self.assertLess(m.rsi(down), 30)

    def test_rsi_bounds(self):
        prices = [100, 101, 100, 102, 99, 103, 101, 104, 100, 105, 102, 106, 101, 107, 103]
        self.assertTrue(0.0 <= m.rsi(prices) <= 100.0)

    def test_trend_golden_and_death(self):
        up = [100 + i for i in range(260)]
        down = [100 + (260 - i) for i in range(260)]
        self.assertEqual(m.trend_signal(up).signal, m.TrendSignal.GOLDEN_CROSS)
        self.assertEqual(m.trend_signal(down).signal, m.TrendSignal.DEATH_CROSS)

    def test_trend_insufficient_data(self):
        self.assertEqual(m.trend_signal([1, 2, 3]).signal, m.TrendSignal.INSUFFICIENT_DATA)

    def test_rsi_bands(self):
        self.assertEqual(m.rsi_band(80), m.RSIBand.OVERBOUGHT)
        self.assertEqual(m.rsi_band(50), m.RSIBand.NEUTRAL)
        self.assertEqual(m.rsi_band(20), m.RSIBand.OVERSOLD)


class RatingTests(unittest.TestCase):
    def test_rating_thresholds(self):
        self.assertEqual(m._rating_from_score(2), m.EntryRating.STRONG)
        self.assertEqual(m._rating_from_score(1), m.EntryRating.FAVORABLE)
        self.assertEqual(m._rating_from_score(0), m.EntryRating.NEUTRAL)
        self.assertEqual(m._rating_from_score(-1), m.EntryRating.UNFAVORABLE)
        self.assertEqual(m._rating_from_score(-2), m.EntryRating.AVOID)

    def test_score_is_sum_of_components(self):
        up = [100 + i for i in range(260)]  # golden cross + overbought (monotonic)
        res = m.momentum_signals("VTI", up, valuation_flag=ValuationFlag.STRETCHED)
        self.assertEqual(res.score, sum(res.components.values()))
        # trend +1, rsi -1 (overbought), valuation -1 -> -1 -> UNFAVORABLE
        self.assertEqual(res.rating, m.EntryRating.UNFAVORABLE)

    def test_valuation_optional_changes_rating(self):
        up = [100 + i for i in range(260)]
        neutral = m.momentum_signals("VTI", up)  # +1 trend, -1 rsi -> 0 -> NEUTRAL
        self.assertEqual(neutral.rating, m.EntryRating.NEUTRAL)
        favorable = m.momentum_signals("VTI", up, valuation_flag=ValuationFlag.ATTRACTIVE)
        self.assertEqual(favorable.rating, m.EntryRating.FAVORABLE)


class LoadPricesTests(unittest.TestCase):
    def test_json_list_and_dict_forms(self):
        data = {
            "_README": "note",
            "VTI": [100, 101, 102],
            "BND": {"closes": [70, 71], "as_of": "2026-01-01", "source": "test"},
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(data, fh)
            path = fh.name
        try:
            loaded = m.load_prices(path)
            self.assertEqual(loaded["VTI"].closes, [100.0, 101.0, 102.0])
            self.assertEqual(loaded["BND"].source, "test")
            self.assertNotIn("_README", loaded)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
