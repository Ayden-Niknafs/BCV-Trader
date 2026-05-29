"""Personal-trading compliance engine.

This module encodes the personal-trading rules from the firm's New Hire
Onboarding / Personal Compliance policy into a small, testable rules engine.

The policy is built around a handful of bright lines for *personal* trading:

* Do not buy individual equities or bonds -- including IPOs and limited
  secondary offerings -- or corporate loans.
* No derivatives or short sales, and no limit / stop-loss / good-till-canceled
  orders on anything that requires pre-clearance.
* Never trade on improperly obtained / material non-public information, and
  never front-run, trade alongside, or tailgate a client account.
* Use an approved broker for every account.
* Pre-clear trades that require it (e.g. new private-investment commitments)
  and execute pre-cleared trades within 48 hours.
* Disclose all accounts/holdings on joining and report quarterly thereafter.

Each rule carries a ``policy_reference`` back to the slide it comes from so the
output is auditable rather than a black box.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Callable, Iterable, Optional

# Slide references map back to the New Hire Onboarding / Personal Compliance deck.
_DECK = "Personal Compliance (New Hire Onboarding)"


class Severity(IntEnum):
    """How a finding affects whether a trade may proceed.

    Ordered from least to most restrictive (an ``IntEnum`` so the overall
    status is simply ``max(...)`` of a result's findings, and the ``<=`` / ``>=``
    threshold checks read naturally).
    """

    ALLOWED = 0       # Permitted with no further action.
    REPORTING = 1     # Permitted, but must be disclosed/reported to Compliance.
    PRECLEARANCE = 2  # Permitted only after Compliance pre-clears it.
    PROHIBITED = 3    # Hard block -- the policy forbids it.


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


class InstrumentType(Enum):
    """Instrument categories the policy treats differently."""

    # Pooled / diversified vehicles -- the compliant building blocks.
    ETF = "etf"
    INDEX_FUND = "index_fund"
    MUTUAL_FUND = "mutual_fund"
    MONEY_MARKET_FUND = "money_market_fund"

    # Outright prohibited to purchase.
    INDIVIDUAL_EQUITY = "individual_equity"
    CORPORATE_BOND = "corporate_bond"
    LOAN = "loan"
    IPO = "ipo"

    # Derivatives / leverage.
    OPTION = "option"
    FUTURE = "future"
    OTHER_DERIVATIVE = "other_derivative"

    # Generally permitted but worth distinguishing.
    GOVERNMENT_BOND = "government_bond"

    # Requires pre-clearance for new commitments.
    PRIVATE_INVESTMENT = "private_investment"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"
    GOOD_TILL_CANCELED = "good_till_canceled"
    SHORT_SALE = "short_sale"


class EmployeeGroup(Enum):
    """Business unit -- some units have extra fund restrictions."""

    GENERAL = "general"
    BC_CREDIT_SPECIAL_SITS = "credit_special_situations"
    PUBLIC_EQUITY = "public_equity"
    PLATFORM_SERVICES = "platform_services"


class BainAffiliation(Enum):
    """Tags a fund that is advised/subadvised by the firm and therefore
    off-limits to certain groups."""

    NONE = "none"
    BC_CREDIT_40ACT = "bc_credit_40act"          # 40 Act funds advised by BC Credit.
    PUBLIC_EQUITY_UCITS = "public_equity_ucits"  # U Access (IRL) BC Global Equity LS UCITS.


# Diversified pooled vehicles -- the only instruments a compliant strategy buys.
POOLED_FUND_TYPES = frozenset(
    {
        InstrumentType.ETF,
        InstrumentType.INDEX_FUND,
        InstrumentType.MUTUAL_FUND,
        InstrumentType.MONEY_MARKET_FUND,
    }
)

# Order types the policy bars for pre-clearance instruments.
RESTRICTED_ORDER_TYPES = frozenset(
    {
        OrderType.LIMIT,
        OrderType.STOP_LOSS,
        OrderType.STOP_LIMIT,
        OrderType.GOOD_TILL_CANCELED,
    }
)


# --------------------------------------------------------------------------- #
# Approved brokers (deck slide 11).                                           #
# --------------------------------------------------------------------------- #
APPROVED_BROKERS: tuple[str, ...] = (
    "Ameriprise",
    "Barclays Stockbrokers",
    "Betterment",
    "Charles Schwab",
    "Chase Investment Services Corp.",
    "Fidelity",
    "Fidelity International",
    "Goldman Sachs (PWM)",
    "Hargreaves Lansdown",
    "Interactive Brokers",
    "Janney Montgomery Scott",
    "JP Morgan Private Bank",
    "JP Morgan Securities",
    "Merrill Lynch",
    "Morgan Stanley",
    "Pacific Premier Trust",
    "Raymond James",
    "Robinhood",
    "Stifel Financial",
    "TD Ameritrade",
    "T. Rowe Price",
    "UBS",
    "Vanguard",
    "Wealthfront",
)

# Divisions / foreign affiliates that are NOT approved even though the parent
# brand is (deck slide 11 note). Matched case-insensitively as substrings.
UNAPPROVED_AFFILIATE_HINTS: tuple[str, ...] = (
    "td institutional",
    "e*trade uk",
    "etrade uk",
)

# Unambiguous shorthand -> canonical approved-broker name. Deliberately small:
# only brands with a single approved entity (so we never alias an ambiguous
# parent like "Goldman Sachs" or "JP Morgan", where only some divisions qualify).
BROKER_ALIASES: dict[str, str] = {
    "schwab": "charles schwab",
    "ibkr": "interactive brokers",
    "t rowe price": "t. rowe price",
}


def _normalize(name: str) -> str:
    return " ".join(name.strip().lower().split())


_APPROVED_NORMALIZED = {_normalize(b) for b in APPROVED_BROKERS}


def _is_flagged_affiliate(normalized: str) -> bool:
    return any(hint in normalized for hint in UNAPPROVED_AFFILIATE_HINTS)


def is_broker_approved(broker: str) -> bool:
    """Return ``True`` if *broker* is a recognized approved broker.

    A flagged division/affiliate (e.g. "TD Institutional") is rejected even if
    its parent brand is approved. Unrecognized names return ``False`` (the rule
    layer decides whether that is a hard block or a verify-with-Compliance flag).
    """

    n = _normalize(broker)
    if not n or _is_flagged_affiliate(n):
        return False
    n = BROKER_ALIASES.get(n, n)
    return n in _APPROVED_NORMALIZED


# --------------------------------------------------------------------------- #
# Core data structures.                                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Finding:
    """A single compliance observation about a proposed trade."""

    rule_id: str
    severity: Severity
    message: str
    policy_reference: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.severity.name}] {self.rule_id}: {self.message} ({self.policy_reference})"


@dataclass
class ProposedTrade:
    """A trade a person wants to place, to be checked against policy."""

    symbol: str
    instrument_type: InstrumentType
    side: Side = Side.BUY
    order_type: OrderType = OrderType.MARKET
    broker: str = ""
    notional: float = 0.0
    # Extra context that the policy keys off of.
    is_ipo: bool = False
    is_limited_secondary: bool = False
    is_leveraged_or_inverse: bool = False
    based_on_mnpi: bool = False           # material non-public / improper info
    overlaps_client_activity: bool = False  # front-running / tailgating risk
    bain_affiliation: BainAffiliation = BainAffiliation.NONE


@dataclass
class ComplianceContext:
    """Who is trading and what the firm currently restricts."""

    employee_group: EmployeeGroup = EmployeeGroup.GENERAL
    approved_brokers_only: bool = True
    # Symbols with current client/firm activity or on the restricted list.
    restricted_symbols: frozenset[str] = frozenset()


@dataclass
class ComplianceResult:
    """The outcome of screening a single trade."""

    trade: ProposedTrade
    findings: list[Finding] = field(default_factory=list)

    @property
    def status(self) -> Severity:
        if not self.findings:
            return Severity.ALLOWED
        return max(f.severity for f in self.findings)

    @property
    def is_blocked(self) -> bool:
        """True if the policy forbids the trade outright."""
        return self.status == Severity.PROHIBITED

    @property
    def requires_preclearance(self) -> bool:
        """True if the trade is permitted only after pre-clearance."""
        return self.status == Severity.PRECLEARANCE

    @property
    def is_clear(self) -> bool:
        """True if the trade may proceed now (reporting may still apply)."""
        return self.status <= Severity.REPORTING

    def blocking_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.PROHIBITED]


# --------------------------------------------------------------------------- #
# Rules. Each takes (trade, context) and returns a Finding or None.            #
# --------------------------------------------------------------------------- #
Rule = Callable[[ProposedTrade, ComplianceContext], Optional[Finding]]


def _requires_preclearance_instrument(t: ProposedTrade) -> bool:
    """Instruments for which restricted order types are barred."""
    return t.instrument_type not in POOLED_FUND_TYPES


def rule_mnpi(t: ProposedTrade, ctx: ComplianceContext) -> Optional[Finding]:
    if t.based_on_mnpi:
        return Finding(
            "MNPI",
            Severity.PROHIBITED,
            "Trading on improperly obtained or material non-public information "
            "is forbidden, as is helping anyone else do so.",
            f"{_DECK}, slides 3-4",
        )
    return None


def rule_front_running(t: ProposedTrade, ctx: ComplianceContext) -> Optional[Finding]:
    if t.overlaps_client_activity or t.symbol.upper() in {
        s.upper() for s in ctx.restricted_symbols
    }:
        return Finding(
            "FRONT_RUNNING",
            Severity.PROHIBITED,
            "Front-running, trading alongside, or tailgating a client account "
            "(trading just before, with, or just after a similar client trade) "
            "is forbidden.",
            f"{_DECK}, slide 4",
        )
    return None


def rule_no_individual_equity(t: ProposedTrade, ctx: ComplianceContext) -> Optional[Finding]:
    if t.side == Side.BUY and t.instrument_type == InstrumentType.INDIVIDUAL_EQUITY:
        return Finding(
            "NO_INDIVIDUAL_EQUITY",
            Severity.PROHIBITED,
            "Purchasing individual equities is prohibited; use diversified "
            "funds instead.",
            f"{_DECK}, slides 3-4",
        )
    return None


def rule_no_ipo(t: ProposedTrade, ctx: ComplianceContext) -> Optional[Finding]:
    if t.side == Side.BUY and (t.is_ipo or t.instrument_type == InstrumentType.IPO):
        return Finding(
            "NO_IPO",
            Severity.PROHIBITED,
            "Participation in IPOs is prohibited.",
            f"{_DECK}, slides 3-4",
        )
    if t.side == Side.BUY and t.is_limited_secondary:
        return Finding(
            "NO_LIMITED_SECONDARY",
            Severity.PROHIBITED,
            "Limited secondary offerings are prohibited.",
            f"{_DECK}, slide 4",
        )
    return None


def rule_no_individual_bond_or_loan(t: ProposedTrade, ctx: ComplianceContext) -> Optional[Finding]:
    if t.side == Side.BUY and t.instrument_type in (
        InstrumentType.CORPORATE_BOND,
        InstrumentType.LOAN,
    ):
        return Finding(
            "NO_INDIVIDUAL_BOND_OR_LOAN",
            Severity.PROHIBITED,
            "Purchasing individual corporate bonds or loans is prohibited.",
            f"{_DECK}, slide 3",
        )
    return None


def rule_no_derivatives(t: ProposedTrade, ctx: ComplianceContext) -> Optional[Finding]:
    if t.instrument_type in (
        InstrumentType.OPTION,
        InstrumentType.FUTURE,
        InstrumentType.OTHER_DERIVATIVE,
    ):
        return Finding(
            "NO_DERIVATIVES",
            Severity.PROHIBITED,
            "Derivatives are prohibited for personal trading.",
            f"{_DECK}, slide 4",
        )
    return None


def rule_no_short_sale(t: ProposedTrade, ctx: ComplianceContext) -> Optional[Finding]:
    if t.order_type == OrderType.SHORT_SALE:
        return Finding(
            "NO_SHORT_SALE",
            Severity.PROHIBITED,
            "Short sales are prohibited for personal trading.",
            f"{_DECK}, slide 4",
        )
    return None


def rule_leveraged_inverse(t: ProposedTrade, ctx: ComplianceContext) -> Optional[Finding]:
    # Leveraged / inverse ETFs are pooled vehicles but use derivatives and
    # behave like leveraged/short exposure, so treat them as needing review.
    if t.side == Side.BUY and t.is_leveraged_or_inverse:
        return Finding(
            "LEVERAGED_INVERSE_REVIEW",
            Severity.PRECLEARANCE,
            "Leveraged or inverse funds use embedded derivatives/short exposure; "
            "pre-clear with Compliance before purchasing.",
            f"{_DECK}, slide 4",
        )
    return None


def rule_restricted_order_type(t: ProposedTrade, ctx: ComplianceContext) -> Optional[Finding]:
    if t.order_type in RESTRICTED_ORDER_TYPES and _requires_preclearance_instrument(t):
        return Finding(
            "NO_RESTRICTED_ORDER_TYPE",
            Severity.PROHIBITED,
            f"{t.order_type.value} orders are prohibited for instruments that "
            "require pre-clearance; use a market order.",
            f"{_DECK}, slide 4",
        )
    return None


def rule_approved_broker(t: ProposedTrade, ctx: ComplianceContext) -> Optional[Finding]:
    if not ctx.approved_brokers_only:
        return None
    n = _normalize(t.broker)
    if not n:
        return Finding(
            "BROKER_UNSPECIFIED",
            Severity.PRECLEARANCE,
            "No broker specified; trades must be placed at an approved broker.",
            f"{_DECK}, slides 3 & 11",
        )
    # A flagged division/affiliate is a known exclusion -> hard block.
    if _is_flagged_affiliate(n):
        return Finding(
            "BROKER_NOT_APPROVED",
            Severity.PROHIBITED,
            f"'{t.broker}' is a division/affiliate that is not approved, even "
            "though its parent brand may be.",
            f"{_DECK}, slides 3 & 11",
        )
    if is_broker_approved(t.broker):
        return None
    # Unrecognized name: don't presume it's banned, but it must be verified.
    return Finding(
        "BROKER_UNRECOGNIZED",
        Severity.PRECLEARANCE,
        f"'{t.broker}' is not on the approved-broker list as written; confirm "
        "the exact entity/division with Compliance before trading.",
        f"{_DECK}, slides 3 & 11",
    )


def rule_private_investment(t: ProposedTrade, ctx: ComplianceContext) -> Optional[Finding]:
    if t.instrument_type != InstrumentType.PRIVATE_INVESTMENT:
        return None
    if t.side == Side.BUY:
        return Finding(
            "PRIVATE_INVESTMENT_PRECLEAR",
            Severity.PRECLEARANCE,
            "New private investments or additional commitments require "
            "pre-clearance with Compliance.",
            f"{_DECK}, slide 3",
        )
    # Distributions / sales generally do not require pre-clearance absent a conflict.
    return Finding(
        "PRIVATE_INVESTMENT_SALE",
        Severity.REPORTING,
        "Distributions/sales of private investments generally do not require "
        "pre-clearance absent a firm conflict, but must be reported.",
        f"{_DECK}, slide 3",
    )


def rule_bain_affiliated_fund(t: ProposedTrade, ctx: ComplianceContext) -> Optional[Finding]:
    aff = t.bain_affiliation
    if aff == BainAffiliation.BC_CREDIT_40ACT and ctx.employee_group in (
        EmployeeGroup.BC_CREDIT_SPECIAL_SITS,
        EmployeeGroup.PLATFORM_SERVICES,
    ):
        return Finding(
            "BC_CREDIT_40ACT_RESTRICTED",
            Severity.PROHIBITED,
            "Credit & Special Situations / Platform Services personnel may not "
            "invest in 40 Act funds advised or subadvised by Bain Capital Credit.",
            f"{_DECK}, slide 4",
        )
    if aff == BainAffiliation.PUBLIC_EQUITY_UCITS and ctx.employee_group in (
        EmployeeGroup.PUBLIC_EQUITY,
        EmployeeGroup.PLATFORM_SERVICES,
    ):
        return Finding(
            "PUBLIC_EQUITY_UCITS_RESTRICTED",
            Severity.PROHIBITED,
            "Public Equity / certain Platform Services personnel may not invest "
            "in the U Access (IRL) Bain Capital Global Equity LS Sustainable UCITS.",
            f"{_DECK}, slide 4",
        )
    return None


def rule_reporting_reminder(t: ProposedTrade, ctx: ComplianceContext) -> Optional[Finding]:
    # Every covered holding/trade must ultimately be reported.
    return Finding(
        "REPORTING_REQUIRED",
        Severity.REPORTING,
        "All covered trades and holdings must be reported via BCCS within 30 "
        "days of quarter-end (or duplicate statements provided).",
        f"{_DECK}, slide 2",
    )


# Order matters only for readability; status is the max severity regardless.
DEFAULT_RULES: tuple[Rule, ...] = (
    rule_mnpi,
    rule_front_running,
    rule_no_individual_equity,
    rule_no_ipo,
    rule_no_individual_bond_or_loan,
    rule_no_derivatives,
    rule_no_short_sale,
    rule_leveraged_inverse,
    rule_restricted_order_type,
    rule_approved_broker,
    rule_private_investment,
    rule_bain_affiliated_fund,
    rule_reporting_reminder,
)


def screen_trade(
    trade: ProposedTrade,
    context: Optional[ComplianceContext] = None,
    rules: Iterable[Rule] = DEFAULT_RULES,
) -> ComplianceResult:
    """Run *trade* through the rule set and return a :class:`ComplianceResult`."""

    ctx = context or ComplianceContext()
    findings = [f for rule in rules if (f := rule(trade, ctx)) is not None]
    return ComplianceResult(trade=trade, findings=findings)


def screen_trades(
    trades: Iterable[ProposedTrade],
    context: Optional[ComplianceContext] = None,
    rules: Iterable[Rule] = DEFAULT_RULES,
) -> list[ComplianceResult]:
    rules = tuple(rules)  # allow re-iteration
    return [screen_trade(t, context, rules) for t in trades]


# --------------------------------------------------------------------------- #
# Disclosure / pre-clearance deadlines.                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Deadline:
    name: str
    due: _dt.date
    description: str

    @property
    def days_remaining(self) -> int:
        return (self.due - _dt.date.today()).days


def _quarter_end(d: _dt.date) -> _dt.date:
    q_end_month = ((d.month - 1) // 3) * 3 + 3
    if q_end_month == 12:
        return _dt.date(d.year, 12, 31)
    # First day of the month after quarter end, minus one day.
    return _dt.date(d.year, q_end_month + 1, 1) - _dt.timedelta(days=1)


def onboarding_deadlines(start_date: _dt.date) -> list[Deadline]:
    """Key personal-compliance deadlines derived from a start date."""

    qe = _quarter_end(start_date)
    return [
        Deadline(
            "Account & activity disclosure",
            start_date + _dt.timedelta(days=10),
            "Disclose all investment accounts and outside activities within 10 "
            "days of joining (BCCS).",
        ),
        Deadline(
            "Holdings as-of date",
            start_date + _dt.timedelta(days=45),
            "Disclosed personal holdings must have an as-of date within 45 days "
            "of the start date.",
        ),
        Deadline(
            "First quarterly report",
            qe + _dt.timedelta(days=30),
            "Report trades/holdings (or confirm no changes) within 30 days of "
            "quarter-end.",
        ),
    ]


def preclearance_execution_deadline(cleared_at: _dt.datetime) -> _dt.datetime:
    """Pre-cleared trades must be executed within 48 hours of clearance."""
    return cleared_at + _dt.timedelta(hours=48)
