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

from . import analytics, catalog, momentum, valuation
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


def cmd_analyze(args: argparse.Namespace) -> int:
    risk = args.risk.lower()
    if risk not in catalog.MODEL_PORTFOLIOS:
        print(f"Unknown risk model '{risk}'; choose from "
              f"{', '.join(catalog.model_names())}.", file=sys.stderr)
        return 2
    weights = catalog.MODEL_PORTFOLIOS[risk]
    stats = analytics.portfolio_stats(weights)

    print(f"ANALYSIS — {risk} model (illustrative long-run assumptions)\n")
    print(f"Expected return: {stats.expected_return_nominal * 100:.1f}% nominal / "
          f"{stats.expected_return_real * 100:.1f}% real    "
          f"Volatility: {stats.volatility * 100:.1f}%    "
          f"Sharpe (vs cash): {stats.sharpe:.2f}\n")

    # Building blocks, with a live compliance check so the link is explicit.
    blocks = analytics.building_blocks(weights)
    all_clear = all(
        screen_trade(ProposedTrade(symbol=b.symbol, instrument_type=InstrumentType.ETF,
                                   broker="Vanguard")).is_clear
        for b in blocks
    )
    banner = "all pre-cleared (pooled funds; screened CLEAR)" if all_clear else "REVIEW NEEDED"
    print(f"Pre-cleared building blocks — {banner}:")
    print(f"  {'Sym':<6}{'Asset class':<32}{'Gross':>7}{'Fee':>7}{'Net':>7}{'Vol':>7}  Role")
    for b in blocks:
        print(f"  {b.symbol:<6}{b.asset_class.value:<32}"
              f"{b.gross_return * 100:>6.2f}%{b.expense_ratio * 100:>6.2f}%"
              f"{b.net_return * 100:>6.2f}%{b.volatility * 100:>6.1f}%  {b.role}")

    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    print(f"\nProbability of success by holding horizon "
          f"(entry = invest now; no market timing):")
    print(f"  {'Horizon':<9}{'P(no nominal loss)':>20}{'P(beat inflation)':>20}"
          f"{'P(>= target real)':>20}   Real CAGR p5 / p50 / p95")
    for t in horizons:
        p_nom = analytics.probability_at_least(weights, t, 1.0)
        p_inf = analytics.probability_beat_inflation(weights, t, 0.0)
        p_tgt = analytics.probability_beat_inflation(weights, t, args.target_real)
        pct = analytics.terminal_real_cagr_percentiles(weights, t)
        print(f"  {str(t) + ' yr':<9}{p_nom * 100:>19.0f}%{p_inf * 100:>19.0f}%"
              f"{p_tgt * 100:>19.0f}%   "
              f"{pct[0.05] * 100:>5.1f}% / {pct[0.50] * 100:>4.1f}% / {pct[0.95] * 100:>4.1f}%")

    ideal = analytics.ideal_holding_period(weights, args.success, 0.0)
    ideal_txt = f"{ideal} years" if ideal is not None else f">{40} years"
    print(f"\nIdeal holding horizon (P(beat inflation) >= {args.success * 100:.0f}%): {ideal_txt}")
    print("  -> 'Exit' = rebalance / withdraw at your goal date, not at a price target.")

    print("\nModels & assumptions: mean-variance / MPT; CAPM-style risk premia; "
          "lognormal\nterminal-wealth model. Inflation assumed "
          f"{analytics.INFLATION * 100:.1f}%. Capital-market assumptions are\n"
          "illustrative long-run estimates (edit in analytics.py). This is "
          "educational,\nnot a guarantee or personalized investment advice; "
          "pre-clear and report per policy.")
    return 0


def cmd_valuation(args: argparse.Namespace) -> int:
    snaps = valuation.load_snapshots(args.data)
    print(f"Valuation context (backward-looking; check 'as of' for staleness):\n")
    print(f"  {'Sym':<6}{'Metric':<22}{'Current':>9}{'10y mean':>10}{'Dev':>7}  Flag")
    for s in snaps:
        print(f"  {s.symbol:<6}{s.metric:<22}{s.current:>9g}{s.ten_year_mean:>10g}"
              f"{s.deviation_pct * 100:>6.0f}%  {s.flag().value.upper()}")
    if snaps:
        print(f"\n  as of: {snaps[0].as_of}  |  source: {snaps[0].source}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    horizon = args.horizon

    snaps: dict[str, valuation.ValuationSnapshot] = {}
    if args.valuation:
        snaps = {s.symbol.upper(): s for s in valuation.load_snapshots(args.valuation)}

    print("COMPLIANCE-AWARE INVESTMENT ANALYTICS REPORT")
    print(f"Horizon: {horizon}y   Broker context: {args.broker}\n")

    for sym in symbols:
        assumption = analytics.INSTRUMENT_ASSUMPTIONS.get(sym)
        print("=" * 72)
        if assumption is None:
            print(f"{sym}: no assumption set on file; add it to "
                  "INSTRUMENT_ASSUMPTIONS to analyze.\n")
            continue
        print(f"{sym} — {assumption.label}")
        print("=" * 72)

        # 1) Compliance.
        result = screen_trade(ProposedTrade(symbol=sym, instrument_type=InstrumentType.ETF,
                                            broker=args.broker))
        verdict = "PRE-CLEARED" if result.is_clear else result.status.name
        print(f"COMPLIANCE: {verdict} — diversified pooled fund (ETF); no "
              "single-name selection; market order, so it bypasses individual "
              "pre-clearance.")
        if "QQQ" in sym:
            print("  ! Caveat: concentrated/sector-tilted (top-10 ~50%); passes the "
                  "single-name screen but is NOT 'broadly diversified'.")
        print("  ! Verify against the current restricted/watch list and any "
              "blackout window before transacting.")

        # 2) Entry (backward-looking valuation context).
        snap = snaps.get(sym)
        if snap is not None:
            print(f"ENTRY (backward-looking): {snap.describe()}")
        else:
            print("ENTRY: no valuation data supplied — pass --valuation FILE "
                  "(e.g. examples/valuation_sample.json) for a fair-value flag.")

        # 3) Return decomposition + fee drag.
        a = assumption
        print(f"RETURN DECOMPOSITION (illustrative): income {a.income_yield * 100:.1f}% "
              f"+ growth {a.earnings_growth * 100:.1f}% + valuation {a.valuation_change * 100:+.1f}% "
              f"= {a.gross_return * 100:.2f}% gross")
        print(f"  fee drag {a.fee * 100:.3f}% -> NET {a.net_return * 100:.2f}%   "
              f"(vol {a.volatility * 100:.1f}%)")
        if a.note:
            print(f"  note: {a.note}")

        # 4) Ideal exit = horizon, not a price.
        print("IDEAL EXIT: holding horizon, not a price target — suited to "
              f"{max(7, horizon - 3)}-{horizon + 5}y compounding; reassess at a "
              "rebalance trigger of ±5% from target weight.")

        # 5) Probability of success (lognormal terminal wealth, your formula).
        tw = analytics.terminal_wealth_lognormal(a.net_return, a.volatility, horizon)
        b1 = analytics.probability_band(a.net_return, a.volatility, horizon, 1.0)
        b2 = analytics.probability_band(a.net_return, a.volatility, horizon, 2.0)
        print(f"PROBABILITY OF SUCCESS ({horizon}y lognormal; mu=ln(1+r)-σ²/2="
              f"{tw.log_drift:.4f}):")
        print(f"  P(W_T > 1x) = {tw.p_above_1x * 100:.0f}%  "
              f"[band {b1[0] * 100:.0f}-{b1[1] * 100:.0f}% under r±1.5%]")
        print(f"  P(W_T > 2x) = {tw.p_above_2x * 100:.0f}%  "
              f"[band {b2[0] * 100:.0f}-{b2[1] * 100:.0f}%]")
        print(f"  terminal multiple p5/p50/p95 = {tw.multiple_p05:.2f}x / "
              f"{tw.multiple_p50:.2f}x / {tw.multiple_p95:.2f}x")
        print()

    print("=" * 72)
    print("MODELS & ASSUMPTIONS")
    print(f"  CMAs: {analytics.CMA_SOURCES}")
    print("  Return decomposition: income yield + earnings growth + valuation change "
          "(Grinold-Kroner building blocks).")
    print("  Probability: W_T = W_0·exp(N(mu·T, σ²·T)), mu = ln(1+r) − σ²/2, "
          "r = expected return NET of fees, σ = annualized volatility.")
    print("  Fee drag is subtracted explicitly (see each NET figure).")
    print("HONEST CAVEATS")
    print("  • Probabilities are model-implied under stated assumptions — not "
          "guaranteed; real outcomes will deviate.")
    print("  • ENTRY valuation context is backward-looking and (here) illustrative/"
          "stale — replace with live data before acting.")
    print("  • This is analytical output, not personalized investment advice. "
          "Pre-clear and report per policy (BCCS).")
    return 0


def cmd_momentum(args: argparse.Namespace) -> int:
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    prices = momentum.load_prices(args.prices) if args.prices else {}
    val_flags: dict[str, valuation.ValuationFlag] = {}
    if args.valuation:
        val_flags = {s.symbol.upper(): s.flag()
                     for s in valuation.load_snapshots(args.valuation)}

    print("MOMENTUM / TREND ENTRY SIGNALS (rules-driven, backward-looking)\n")
    for sym in symbols:
        st = momentum.compliance_status(sym)
        print("-" * 72)
        print(f"{sym} — {momentum.FUND_PROFILE.get(sym).name if sym in momentum.FUND_PROFILE else 'unknown'}")
        print(f"  1) COMPLIANCE: {st.summary()}")
        print(f"     structure={st.structure.value}; scope={st.scope.value}; "
              f"~{st.holdings} holdings; market-order compatible: "
              f"{'yes' if st.market_order_compatible else 'no'}")

        if not st.trade_call_allowed:
            print("  2) ENTRY SIGNAL: consult Compliance — narrow-based or "
                  "non-open-end; NO trade call.")
            continue

        series = prices.get(sym)
        if series is None or len(series.closes) < args.long:
            have = 0 if series is None else len(series.closes)
            print(f"  2) ENTRY SIGNAL: no/insufficient price data (have {have}, "
                  f"need >= {args.long}). Supply daily closes via --prices to "
                  "compute MA/RSI.")
            continue

        res = momentum.momentum_signals(
            sym, series.closes, valuation_flag=val_flags.get(sym),
            short=args.short, long=args.long, rsi_period=args.rsi_period)
        print(f"  2) ENTRY SIGNAL: {res.summary()}")
        if series.as_of or series.source:
            print(f"     price data: as of {series.as_of or '?'} "
                  f"(source: {series.source or '?'})")

    print("-" * 72)
    print("Scoring: trend(golden +1/death -1) + RSI(oversold +1/overbought -1) + "
          "valuation(attractive +1/stretched -1)")
    print("  => STRONG>=2, FAVORABLE 1, NEUTRAL 0, UNFAVORABLE -1, AVOID<=-2.")
    print("CAVEATS: signals are mechanical & backward-looking, NOT predictions; "
          "momentum reverses.\n  Report all executed trades in BCCS within 30 days "
          "of quarter-end. Not personalized advice.")
    return 0


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

    a = sub.add_parser("analyze", help="Expected return, risk, and goal "
                       "probability for a model portfolio.")
    a.add_argument("--risk", default="balanced", help="conservative | balanced | aggressive")
    a.add_argument("--horizons", default="5,10,15,20,30",
                   help="Comma-separated holding horizons in years.")
    a.add_argument("--target-real", type=float, default=0.04, dest="target_real",
                   help="Real CAGR goal to beat (e.g. 0.04 = inflation + 4%%/yr).")
    a.add_argument("--success", type=float, default=0.90,
                   help="Success-probability threshold for the ideal horizon.")
    a.set_defaults(func=cmd_analyze)

    r = sub.add_parser("report", help="Per-instrument compliance + valuation + "
                       "probability report.")
    r.add_argument("--symbols", default="VTI,SPY,QQQ,BND,VXUS",
                   help="Comma-separated instrument symbols.")
    r.add_argument("--horizon", type=int, default=10, help="Holding horizon in years.")
    r.add_argument("--broker", default="Vanguard")
    r.add_argument("--valuation", help="Path to a valuation JSON for ENTRY flags "
                   "(e.g. examples/valuation_sample.json).")
    r.set_defaults(func=cmd_report)

    v = sub.add_parser("valuation", help="Show valuation context from a data file.")
    v.add_argument("--data", required=True, help="Path to a valuation JSON file.")
    v.set_defaults(func=cmd_valuation)

    mo = sub.add_parser("momentum", help="Rules-driven momentum/trend entry signals.")
    mo.add_argument("--symbols", default="QQQ,VGT,XLK,XBI,VUG,MTUM,VTI,VXUS,BND",
                    help="Comma-separated symbols.")
    mo.add_argument("--prices", help="Path to daily-closes JSON/CSV (oldest first).")
    mo.add_argument("--valuation", help="Optional valuation JSON to fold into the rating.")
    mo.add_argument("--short", type=int, default=50, help="Short MA window.")
    mo.add_argument("--long", type=int, default=200, help="Long MA window.")
    mo.add_argument("--rsi-period", type=int, default=14, dest="rsi_period")
    mo.set_defaults(func=cmd_momentum)

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
