"""Command-line interface for the compliant strategy tool.

Examples::

    python -m bcv_trader brokers
    python -m bcv_trader models
    python -m bcv_trader build --amount 100000 --risk balanced --broker Vanguard
    python -m bcv_trader screen --symbol AAPL --type individual_equity --broker Robinhood
    python -m bcv_trader disclosures --start-date 2026-05-29
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from typing import Optional

from . import catalog
from .compliance import (
    APPROVED_BROKERS,
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
    screen_trade,
)
from .strategy import build_portfolio


def _enum_from_name(enum_cls, value: str):
    """Resolve an enum member from a case-insensitive name or value."""
    key = value.strip().lower()
    for member in enum_cls:
        if member.name.lower() == key or str(member.value).lower() == key:
            return member
    choices = ", ".join(m.name.lower() for m in enum_cls)
    raise argparse.ArgumentTypeError(f"'{value}' is not valid; choose from: {choices}")


# --------------------------------------------------------------------------- #
# Subcommands.                                                                 #
# --------------------------------------------------------------------------- #
def cmd_brokers(args: argparse.Namespace) -> int:
    print("Approved brokers for personal trading (deck slide 11):\n")
    for b in APPROVED_BROKERS:
        print(f"  - {b}")
    print(
        "\nNote: some divisions / foreign affiliates (e.g. TD Institutional, "
        "E*Trade UK)\nand certain accounts (e.g. some 401(k)s) may not be "
        "approved even if the\nparent brand is. Consult Compliance for edge cases."
    )
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    for name, weights in catalog.MODEL_PORTFOLIOS.items():
        print(f"\n{name.upper()}")
        for asset_class, w in weights.items():
            fund = catalog.DEFAULT_FUND[asset_class]
            print(f"  {w * 100:>5.1f}%  {asset_class.value:<32} via {fund}")
    print()
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    prices: Optional[dict[str, float]] = None
    if args.prices:
        with open(args.prices, "r", encoding="utf-8") as fh:
            prices = {k.upper(): float(v) for k, v in json.load(fh).items()}

    plan = build_portfolio(
        amount=args.amount,
        risk=args.risk,
        broker=args.broker,
        employee_group=args.group,
        prices=prices,
    )
    print(plan.summary())
    # 0 = clear, 1 = needs pre-clearance, 2 = blocked.
    if plan.is_blocked:
        return 2
    return 0 if plan.is_compliant else 1


def cmd_screen(args: argparse.Namespace) -> int:
    trade = ProposedTrade(
        symbol=args.symbol,
        instrument_type=args.type,
        side=args.side,
        order_type=args.order,
        broker=args.broker,
        is_ipo=args.ipo,
        is_limited_secondary=args.limited_secondary,
        is_leveraged_or_inverse=args.leveraged,
        based_on_mnpi=args.mnpi,
        overlaps_client_activity=args.front_running,
        bain_affiliation=args.bain,
    )
    ctx = ComplianceContext(employee_group=args.group)
    result = screen_trade(trade, ctx)

    status = (
        "BLOCKED" if result.is_blocked
        else "NEEDS PRE-CLEARANCE" if result.requires_preclearance
        else "CLEAR"
    )
    print(f"{trade.side.value.upper()} {trade.symbol} "
          f"({trade.instrument_type.value}) via {trade.broker or '(no broker)'}")
    print(f"Status: {status}\n")
    if not result.findings:
        print("  (no findings)")
    for f in sorted(result.findings, key=lambda x: x.severity.value, reverse=True):
        print(f"  - {f}")
    # 0 = clear, 1 = needs pre-clearance, 2 = blocked.
    if result.is_blocked:
        return 2
    return 1 if result.requires_preclearance else 0


def cmd_disclosures(args: argparse.Namespace) -> int:
    start = dt.date.fromisoformat(args.start_date) if args.start_date else dt.date.today()
    print(f"Personal-compliance deadlines for start date {start.isoformat()}:\n")
    for d in onboarding_deadlines(start):
        print(f"  {d.due.isoformat()}  {d.name}")
        print(f"               {d.description}")
    print(
        "\nPre-cleared trades must be executed within 48 hours of clearance.\n"
        "Thereafter, report trades/holdings within 30 days of each quarter-end."
    )
    return 0


# --------------------------------------------------------------------------- #
# Parser.                                                                      #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bcv_trader",
        description="Build and screen personal-trading strategies that comply "
        "with the firm's Personal Compliance policy.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("brokers", help="List approved brokers.").set_defaults(func=cmd_brokers)
    sub.add_parser("models", help="List model portfolios.").set_defaults(func=cmd_models)

    b = sub.add_parser("build", help="Build a compliant portfolio.")
    b.add_argument("--amount", type=float, required=True, help="Dollars to invest.")
    b.add_argument("--risk", default="balanced", help="conservative | balanced | aggressive")
    b.add_argument("--broker", default="Vanguard", help="Broker to place trades at.")
    b.add_argument("--group", type=lambda v: _enum_from_name(EmployeeGroup, v),
                   default=EmployeeGroup.GENERAL, help="Business unit.")
    b.add_argument("--prices", help="Path to JSON {symbol: price} for whole-share sizing.")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("screen", help="Screen a single proposed trade.")
    s.add_argument("--symbol", required=True)
    s.add_argument("--type", type=lambda v: _enum_from_name(InstrumentType, v),
                   default=InstrumentType.ETF, help="Instrument type.")
    s.add_argument("--side", type=lambda v: _enum_from_name(Side, v), default=Side.BUY)
    s.add_argument("--order", type=lambda v: _enum_from_name(OrderType, v),
                   default=OrderType.MARKET)
    s.add_argument("--broker", default="")
    s.add_argument("--group", type=lambda v: _enum_from_name(EmployeeGroup, v),
                   default=EmployeeGroup.GENERAL)
    s.add_argument("--bain", type=lambda v: _enum_from_name(BainAffiliation, v),
                   default=BainAffiliation.NONE, help="Firm-fund affiliation tag.")
    s.add_argument("--ipo", action="store_true")
    s.add_argument("--limited-secondary", action="store_true")
    s.add_argument("--leveraged", action="store_true", help="Leveraged/inverse fund.")
    s.add_argument("--mnpi", action="store_true", help="Based on material non-public info.")
    s.add_argument("--front-running", action="store_true",
                   help="Overlaps current client activity.")
    s.set_defaults(func=cmd_screen)

    d = sub.add_parser("disclosures", help="Show personal-compliance deadlines.")
    d.add_argument("--start-date", help="Start date (YYYY-MM-DD); defaults to today.")
    d.set_defaults(func=cmd_disclosures)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
