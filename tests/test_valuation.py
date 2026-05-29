import os
import unittest

from bcv_trader.valuation import ValuationFlag, ValuationSnapshot, load_snapshots

_SAMPLE = os.path.join(os.path.dirname(__file__), "..", "examples", "valuation_sample.json")


def _snap(**kw):
    base = dict(symbol="X", metric="Trailing P/E", current=20.0,
                ten_year_mean=20.0, as_of="test")
    base.update(kw)
    return ValuationSnapshot(**base)


class FlagTests(unittest.TestCase):
    def test_pe_above_mean_is_stretched(self):
        self.assertEqual(_snap(current=25, ten_year_mean=20).flag(), ValuationFlag.STRETCHED)

    def test_pe_below_mean_is_attractive(self):
        self.assertEqual(_snap(current=15, ten_year_mean=20).flag(), ValuationFlag.ATTRACTIVE)

    def test_pe_near_mean_is_neutral(self):
        self.assertEqual(_snap(current=20.5, ten_year_mean=20).flag(), ValuationFlag.NEUTRAL)

    def test_yield_above_mean_is_attractive(self):
        # Higher yield = cheaper/more attractive entry.
        s = _snap(metric="SEC yield", current=4.3, ten_year_mean=2.6, higher_is_cheaper=True)
        self.assertEqual(s.flag(), ValuationFlag.ATTRACTIVE)

    def test_zscore_used_when_std_present(self):
        s = _snap(current=22, ten_year_mean=20, ten_year_std=1.0)  # z = +2 -> stretched
        self.assertEqual(s.zscore, 2.0)
        self.assertEqual(s.flag(), ValuationFlag.STRETCHED)

    def test_deviation_pct(self):
        self.assertAlmostEqual(_snap(current=24, ten_year_mean=20).deviation_pct, 0.20)


class LoadTests(unittest.TestCase):
    def test_sample_loads_and_ignores_readme(self):
        snaps = load_snapshots(_SAMPLE)
        by_sym = {s.symbol: s for s in snaps}
        self.assertEqual(set(by_sym), {"VTI", "SPY", "QQQ", "BND", "VXUS"})

    def test_sample_flags_directionally_correct(self):
        by_sym = {s.symbol: s for s in load_snapshots(_SAMPLE)}
        self.assertEqual(by_sym["VTI"].flag(), ValuationFlag.STRETCHED)
        self.assertEqual(by_sym["VXUS"].flag(), ValuationFlag.NEUTRAL)
        self.assertEqual(by_sym["BND"].flag(), ValuationFlag.ATTRACTIVE)


if __name__ == "__main__":
    unittest.main()
