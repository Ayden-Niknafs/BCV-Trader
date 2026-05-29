import datetime as dt
import unittest

from bcv_trader.compliance import (
    BainAffiliation,
    ComplianceContext,
    EmployeeGroup,
    InstrumentType,
    OrderType,
    ProposedTrade,
    Severity,
    Side,
    is_broker_approved,
    onboarding_deadlines,
    preclearance_execution_deadline,
    screen_trade,
)


def _trade(**kw):
    base = dict(
        symbol="VTI",
        instrument_type=InstrumentType.ETF,
        broker="Vanguard",
    )
    base.update(kw)
    return ProposedTrade(**base)


class BrokerTests(unittest.TestCase):
    def test_approved_brokers(self):
        self.assertTrue(is_broker_approved("Vanguard"))
        self.assertTrue(is_broker_approved("  charles   SCHWAB "))  # normalized
        self.assertTrue(is_broker_approved("Interactive Brokers"))

    def test_unambiguous_aliases(self):
        self.assertTrue(is_broker_approved("Schwab"))
        self.assertTrue(is_broker_approved("IBKR"))
        self.assertTrue(is_broker_approved("T Rowe Price"))

    def test_unapproved_broker(self):
        self.assertFalse(is_broker_approved("Acme Bucket Shop"))
        self.assertFalse(is_broker_approved(""))

    def test_unapproved_affiliate(self):
        # Parent brand may be fine, but flagged divisions are not.
        self.assertFalse(is_broker_approved("TD Institutional"))
        self.assertFalse(is_broker_approved("E*Trade UK"))

    def test_flagged_affiliate_is_hard_blocked(self):
        r = screen_trade(_trade(broker="TD Institutional"))
        self.assertTrue(r.is_blocked)

    def test_unrecognized_broker_needs_preclearance(self):
        # Unknown != banned: flag it for verification rather than hard-blocking.
        r = screen_trade(_trade(broker="Acme Bucket Shop"))
        self.assertTrue(r.requires_preclearance)
        self.assertFalse(r.is_blocked)


class ProhibitedInstrumentTests(unittest.TestCase):
    def test_individual_equity_buy_blocked(self):
        r = screen_trade(_trade(symbol="AAPL", instrument_type=InstrumentType.INDIVIDUAL_EQUITY))
        self.assertTrue(r.is_blocked)
        self.assertIn("NO_INDIVIDUAL_EQUITY", {f.rule_id for f in r.blocking_findings()})

    def test_ipo_blocked(self):
        r = screen_trade(_trade(symbol="NEWCO", is_ipo=True))
        self.assertTrue(r.is_blocked)

    def test_corporate_bond_blocked(self):
        r = screen_trade(_trade(symbol="XYZ4.5", instrument_type=InstrumentType.CORPORATE_BOND))
        self.assertTrue(r.is_blocked)

    def test_loan_blocked(self):
        r = screen_trade(_trade(instrument_type=InstrumentType.LOAN))
        self.assertTrue(r.is_blocked)

    def test_derivatives_blocked(self):
        for t in (InstrumentType.OPTION, InstrumentType.FUTURE, InstrumentType.OTHER_DERIVATIVE):
            self.assertTrue(screen_trade(_trade(instrument_type=t)).is_blocked)


class ProhibitedPracticeTests(unittest.TestCase):
    def test_short_sale_blocked(self):
        r = screen_trade(_trade(order_type=OrderType.SHORT_SALE))
        self.assertTrue(r.is_blocked)

    def test_restricted_order_type_on_preclear_instrument_blocked(self):
        r = screen_trade(_trade(
            symbol="AAPL",
            instrument_type=InstrumentType.INDIVIDUAL_EQUITY,
            order_type=OrderType.LIMIT,
        ))
        self.assertTrue(r.is_blocked)
        self.assertIn("NO_RESTRICTED_ORDER_TYPE", {f.rule_id for f in r.findings})

    def test_limit_order_on_fund_not_blocked_by_order_rule(self):
        # Pooled funds don't require pre-clearance, so the order-type rule
        # doesn't fire (the trade is otherwise clear).
        r = screen_trade(_trade(order_type=OrderType.LIMIT))
        self.assertNotIn("NO_RESTRICTED_ORDER_TYPE", {f.rule_id for f in r.findings})
        self.assertFalse(r.is_blocked)

    def test_mnpi_blocked(self):
        self.assertTrue(screen_trade(_trade(based_on_mnpi=True)).is_blocked)

    def test_front_running_flag_blocked(self):
        self.assertTrue(screen_trade(_trade(overlaps_client_activity=True)).is_blocked)

    def test_restricted_symbol_context_blocked(self):
        ctx = ComplianceContext(restricted_symbols=frozenset({"VTI"}))
        self.assertTrue(screen_trade(_trade(symbol="vti"), ctx).is_blocked)


class PreclearanceTests(unittest.TestCase):
    def test_private_investment_buy_needs_preclearance(self):
        r = screen_trade(_trade(
            symbol="FUND-LP", instrument_type=InstrumentType.PRIVATE_INVESTMENT, side=Side.BUY))
        self.assertTrue(r.requires_preclearance)
        self.assertFalse(r.is_blocked)

    def test_private_investment_sale_is_reporting_only(self):
        r = screen_trade(_trade(
            symbol="FUND-LP", instrument_type=InstrumentType.PRIVATE_INVESTMENT, side=Side.SELL))
        self.assertTrue(r.is_clear)
        self.assertFalse(r.requires_preclearance)

    def test_unspecified_broker_needs_preclearance(self):
        r = screen_trade(_trade(broker=""))
        self.assertTrue(r.requires_preclearance)

    def test_leveraged_fund_needs_preclearance(self):
        r = screen_trade(_trade(is_leveraged_or_inverse=True))
        self.assertTrue(r.requires_preclearance)


class BainAffiliatedFundTests(unittest.TestCase):
    def test_credit_40act_blocked_for_credit_group(self):
        ctx = ComplianceContext(employee_group=EmployeeGroup.BC_CREDIT_SPECIAL_SITS)
        r = screen_trade(_trade(bain_affiliation=BainAffiliation.BC_CREDIT_40ACT), ctx)
        self.assertTrue(r.is_blocked)

    def test_credit_40act_allowed_for_general_group(self):
        ctx = ComplianceContext(employee_group=EmployeeGroup.GENERAL)
        r = screen_trade(_trade(bain_affiliation=BainAffiliation.BC_CREDIT_40ACT), ctx)
        self.assertFalse(r.is_blocked)

    def test_public_equity_ucits_blocked_for_public_equity_group(self):
        ctx = ComplianceContext(employee_group=EmployeeGroup.PUBLIC_EQUITY)
        r = screen_trade(_trade(bain_affiliation=BainAffiliation.PUBLIC_EQUITY_UCITS), ctx)
        self.assertTrue(r.is_blocked)


class CompliantTradeTests(unittest.TestCase):
    def test_plain_etf_buy_is_clear_but_reportable(self):
        r = screen_trade(_trade())
        self.assertTrue(r.is_clear)
        self.assertFalse(r.is_blocked)
        self.assertFalse(r.requires_preclearance)
        # Reporting reminder is always present.
        self.assertEqual(r.status, Severity.REPORTING)
        self.assertIn("REPORTING_REQUIRED", {f.rule_id for f in r.findings})


class DeadlineTests(unittest.TestCase):
    def test_onboarding_deadline_offsets(self):
        start = dt.date(2026, 5, 29)
        ds = {d.name: d.due for d in onboarding_deadlines(start)}
        self.assertEqual(ds["Account & activity disclosure"], start + dt.timedelta(days=10))
        self.assertEqual(ds["Holdings as-of date"], start + dt.timedelta(days=45))
        # Q2 ends 2026-06-30; report due within 30 days.
        self.assertEqual(ds["First quarterly report"], dt.date(2026, 6, 30) + dt.timedelta(days=30))

    def test_year_end_quarter(self):
        start = dt.date(2026, 11, 15)  # Q4
        ds = {d.name: d.due for d in onboarding_deadlines(start)}
        self.assertEqual(ds["First quarterly report"], dt.date(2026, 12, 31) + dt.timedelta(days=30))

    def test_preclearance_execution_window(self):
        cleared = dt.datetime(2026, 5, 29, 9, 0, 0)
        self.assertEqual(
            preclearance_execution_deadline(cleared),
            cleared + dt.timedelta(hours=48),
        )


if __name__ == "__main__":
    unittest.main()
