import unittest

from bcv_trader.catalog import (
    DEFAULT_FUND,
    MODEL_PORTFOLIOS,
    UNIVERSE,
)
from bcv_trader.compliance import (
    BainAffiliation,
    InstrumentType,
    Side,
)
from bcv_trader.compliance import POOLED_FUND_TYPES
from bcv_trader.strategy import build_portfolio, rebalance


class CatalogInvarariantTests(unittest.TestCase):
    def test_universe_is_entirely_compliant(self):
        # The whole point of the catalog: every building block is a pooled,
        # non-leveraged, non-firm-affiliated fund.
        for sym, inst in UNIVERSE.items():
            self.assertIn(inst.instrument_type, POOLED_FUND_TYPES, sym)
            self.assertFalse(inst.leveraged_or_inverse, sym)
            self.assertEqual(inst.bain_affiliation, BainAffiliation.NONE, sym)

    def test_models_reference_known_funds_and_sum_to_one(self):
        for name, weights in MODEL_PORTFOLIOS.items():
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=6, msg=name)
            for asset_class in weights:
                self.assertIn(DEFAULT_FUND[asset_class], UNIVERSE)


class BuildPortfolioTests(unittest.TestCase):
    def test_balanced_portfolio_is_compliant(self):
        plan = build_portfolio(100_000, risk="balanced", broker="Vanguard")
        self.assertTrue(plan.is_compliant)
        self.assertFalse(plan.is_blocked)
        self.assertGreater(len(plan.holdings), 0)
        # All proposed orders are market BUYs of pooled funds.
        for t in plan.trades:
            self.assertEqual(t.side, Side.BUY)
            self.assertIn(t.instrument_type, POOLED_FUND_TYPES)

    def test_allocations_sum_to_amount(self):
        amount = 50_000.0
        plan = build_portfolio(amount, risk="conservative")
        self.assertAlmostEqual(sum(h.amount for h in plan.holdings), amount, places=2)
        self.assertAlmostEqual(sum(h.weight for h in plan.holdings), 1.0, places=6)

    def test_all_risk_models_build(self):
        for risk in MODEL_PORTFOLIOS:
            plan = build_portfolio(10_000, risk=risk)
            self.assertTrue(plan.is_compliant, risk)

    def test_prices_produce_whole_shares_and_cash(self):
        prices = {sym: 100.0 for sym in UNIVERSE}
        plan = build_portfolio(100_000, risk="balanced", prices=prices)
        for h in plan.holdings:
            self.assertIsNotNone(h.shares)
            self.assertEqual(h.shares, float(int(h.shares)))  # whole shares
        self.assertGreaterEqual(plan.cash_remaining, 0.0)

    def test_unrecognized_broker_needs_preclearance(self):
        plan = build_portfolio(10_000, broker="Acme Bucket Shop")
        self.assertFalse(plan.is_blocked)
        self.assertTrue(plan.needs_preclearance)
        self.assertFalse(plan.is_compliant)

    def test_flagged_affiliate_broker_blocks_every_trade(self):
        plan = build_portfolio(10_000, broker="TD Institutional")
        self.assertTrue(plan.is_blocked)
        self.assertFalse(plan.is_compliant)
        self.assertEqual(len(plan.blocking_findings()), len(plan.trades))

    def test_weighted_expense_ratio_reasonable(self):
        plan = build_portfolio(10_000, risk="aggressive")
        # Broad-market funds -> a low blended fee.
        self.assertLess(plan.weighted_expense_ratio, 0.01)
        self.assertGreater(plan.weighted_expense_ratio, 0.0)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            build_portfolio(0)
        with self.assertRaises(ValueError):
            build_portfolio(1000, risk="yolo")


class RebalanceTests(unittest.TestCase):
    def test_rebalance_moves_toward_target(self):
        # Start 100% in US equity; rebalance toward the balanced model.
        current = {"VTI": 100_000.0}
        orders = rebalance(current, target_model="balanced")
        by_symbol = {o.symbol: o for o in orders}
        # Must sell down the overweight US-equity position...
        self.assertEqual(by_symbol["VTI"].side, Side.SELL)
        # ...and buy into bonds/international to diversify.
        self.assertIn("BND", by_symbol)
        self.assertEqual(by_symbol["BND"].side, Side.BUY)

    def test_rebalance_skips_tiny_trades(self):
        current = {"VTI": 100.0}
        orders = rebalance(current, target_model="balanced", min_trade=1000.0)
        self.assertEqual(orders, [])

    def test_rebalance_requires_positive_value(self):
        with self.assertRaises(ValueError):
            rebalance({"VTI": 0.0})


if __name__ == "__main__":
    unittest.main()
