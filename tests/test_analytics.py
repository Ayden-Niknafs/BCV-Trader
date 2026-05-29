import unittest

from bcv_trader import analytics
from bcv_trader.analytics import CMA
from bcv_trader.catalog import MODEL_PORTFOLIOS, UNIVERSE, AssetClass


class CorrelationTests(unittest.TestCase):
    def test_diagonal_and_symmetry(self):
        self.assertEqual(analytics.correlation(AssetClass.US_EQUITY, AssetClass.US_EQUITY), 1.0)
        self.assertEqual(
            analytics.correlation(AssetClass.US_EQUITY, AssetClass.US_BOND),
            analytics.correlation(AssetClass.US_BOND, AssetClass.US_EQUITY),
        )

    def test_stock_bond_low_correlation(self):
        self.assertEqual(
            analytics.correlation(AssetClass.US_EQUITY, AssetClass.US_BOND), 0.10
        )


class PortfolioStatsTests(unittest.TestCase):
    def test_expected_return_is_weighted_mean(self):
        weights = {AssetClass.US_EQUITY: 0.5, AssetClass.US_BOND: 0.5}
        expected = 0.5 * CMA[AssetClass.US_EQUITY][0] + 0.5 * CMA[AssetClass.US_BOND][0]
        self.assertAlmostEqual(analytics.portfolio_expected_return(weights), expected)

    def test_diversification_reduces_volatility(self):
        weights = MODEL_PORTFOLIOS["balanced"]
        total = sum(weights.values())
        weighted_avg_vol = sum(w / total * CMA[ac][1] for ac, w in weights.items())
        self.assertLess(analytics.portfolio_volatility(weights), weighted_avg_vol)

    def test_single_asset_matches_cma(self):
        weights = {AssetClass.CASH: 1.0}
        self.assertAlmostEqual(analytics.portfolio_volatility(weights), CMA[AssetClass.CASH][1])
        self.assertAlmostEqual(
            analytics.portfolio_expected_return(weights), CMA[AssetClass.CASH][0]
        )

    def test_stats_real_return_consistent(self):
        stats = analytics.portfolio_stats(MODEL_PORTFOLIOS["aggressive"])
        self.assertAlmostEqual(
            stats.expected_return_real,
            analytics.real_return(stats.expected_return_nominal),
        )
        self.assertGreater(stats.expected_return_nominal, stats.expected_return_real)


class ProbabilityModelTests(unittest.TestCase):
    def test_probabilities_in_unit_interval(self):
        w = MODEL_PORTFOLIOS["balanced"]
        for t in (1, 5, 10, 30):
            p = analytics.probability_beat_inflation(w, t)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_longer_horizon_more_likely_to_beat_inflation(self):
        w = MODEL_PORTFOLIOS["aggressive"]
        self.assertGreater(
            analytics.probability_beat_inflation(w, 30),
            analytics.probability_beat_inflation(w, 5),
        )

    def test_nonpositive_target_is_certain(self):
        self.assertEqual(analytics.probability_at_least(MODEL_PORTFOLIOS["balanced"], 10, 0.0), 1.0)

    def test_real_cagr_percentiles_ordered(self):
        pct = analytics.terminal_real_cagr_percentiles(MODEL_PORTFOLIOS["balanced"], 10)
        self.assertLess(pct[0.05], pct[0.50])
        self.assertLess(pct[0.50], pct[0.95])

    def test_ideal_holding_period_meets_threshold(self):
        w = MODEL_PORTFOLIOS["balanced"]
        t = analytics.ideal_holding_period(w, success_threshold=0.90, real_cagr_target=0.0)
        self.assertIsNotNone(t)
        self.assertGreaterEqual(
            analytics.probability_beat_inflation(w, t, 0.0), 0.90
        )

    def test_ideal_holding_period_unreachable_returns_none(self):
        # No equity portfolio beats inflation by 50%/yr with 99.9% odds soon.
        w = MODEL_PORTFOLIOS["conservative"]
        self.assertIsNone(
            analytics.ideal_holding_period(w, success_threshold=0.999,
                                           real_cagr_target=0.50, max_years=5)
        )


class BuildingBlockTests(unittest.TestCase):
    def test_net_return_is_gross_minus_fee(self):
        for b in analytics.building_blocks(MODEL_PORTFOLIOS["balanced"]):
            self.assertAlmostEqual(b.net_return, b.gross_return - b.expense_ratio)
            self.assertIn(b.symbol, UNIVERSE)


if __name__ == "__main__":
    unittest.main()
