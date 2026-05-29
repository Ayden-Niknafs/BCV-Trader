"""BCV-Trader: a compliant personal-trading strategy tool.

This package encodes the firm's Personal Compliance policy (from New Hire
Onboarding) into a rules engine, and builds diversified portfolios that are
screened against it. The guiding principle: a compliant edge comes from
disciplined, low-cost, diversified allocation -- never from individual-name
selection, IPOs, derivatives, shorting, or any information/timing advantage
over client accounts.

Public API:

* :func:`~bcv_trader.compliance.screen_trade` / ``screen_trades``
* :func:`~bcv_trader.strategy.build_portfolio` / ``rebalance``
* :func:`~bcv_trader.compliance.onboarding_deadlines`
"""

from .compliance import (
    APPROVED_BROKERS,
    BainAffiliation,
    ComplianceContext,
    ComplianceResult,
    EmployeeGroup,
    Finding,
    InstrumentType,
    OrderType,
    ProposedTrade,
    Severity,
    Side,
    is_broker_approved,
    onboarding_deadlines,
    preclearance_execution_deadline,
    screen_trade,
    screen_trades,
)
from .strategy import PortfolioPlan, build_portfolio, rebalance, validate_holdings

__version__ = "0.1.0"

__all__ = [
    "APPROVED_BROKERS",
    "BainAffiliation",
    "ComplianceContext",
    "ComplianceResult",
    "EmployeeGroup",
    "Finding",
    "InstrumentType",
    "OrderType",
    "ProposedTrade",
    "Severity",
    "Side",
    "is_broker_approved",
    "onboarding_deadlines",
    "preclearance_execution_deadline",
    "screen_trade",
    "screen_trades",
    "PortfolioPlan",
    "build_portfolio",
    "rebalance",
    "validate_holdings",
]
