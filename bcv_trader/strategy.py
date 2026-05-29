"""Build diversified, fully-compliant portfolios.

The strategy here is deliberately boring -- and that is the point. Since the
policy forbids individual securities, IPOs, derivatives, shorting, and exotic
order types, a compliant "edge" comes from disciplined, low-cost, diversified
asset allocation rather than security selection or timing.

:func:`build_portfolio` turns a risk-based model into concrete trades and runs
every one of them through the compliance engine, so the plan it returns is
guaranteed to be screened, not just assumed clean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from . import catalog
from .catalog import AssetClass, Instrument
from .compliance import (
    ComplianceContext,
    ComplianceResult,
    EmployeeGroup,
    OrderType,
    ProposedTrade,
    Severity,
    Side,
    screen_trades,
)

# Warn if any single fund would exceed this share of the portfolio.
CONCENTRATION_LIMIT = 0.60


@dataclass
class Holding:
    instrument: Instrument
    weight: float
    amount: float
    shares: Optional[float] = None  # set only when prices are provided

    @property
    def symbol(self) -> str:
        return self.instrument.symbol


@dataclass
class PortfolioPlan:
    model_name: str
    broker: str
    amount: float
    holdings: list[Holding]
    trades: list[ProposedTrade]
    results: list[ComplianceResult]
    warnings: list[str] = field(default_factory=list)
    cash_remaining: float = 0.0

    @property
    def is_compliant(self) -> bool:
        """True only if every trade may proceed now (nothing blocked or pending)."""
        return all(r.is_clear for r in self.results)

    @property
    def is_blocked(self) -> bool:
        return any(r.is_blocked for r in self.results)

    @property
    def needs_preclearance(self) -> bool:
        return any(r.requires_preclearance for r in self.results)

    @property
    def weighted_expense_ratio(self) -> float:
        return sum(h.weight * h.instrument.expense_ratio for h in self.holdings)

    def blocking_findings(self):
        return [f for r in self.results for f in r.blocking_findings()]

    def summary(self) -> str:
        lines = [
            f"Model: {self.model_name}    Broker: {self.broker}    "
            f"Investable: ${self.amount:,.2f}",
            f"Weighted expense ratio: {self.weighted_expense_ratio * 100:.3f}% / yr",
            "",
            f"{'Symbol':<7}{'Asset class':<32}{'Weight':>8}{'Amount':>14}{'Shares':>10}",
            "-" * 71,
        ]
        for h in self.holdings:
            shares = f"{h.shares:.0f}" if h.shares is not None else "-"
            lines.append(
                f"{h.symbol:<7}{h.instrument.asset_class.value:<32}"
                f"{h.weight * 100:>7.1f}%{h.amount:>14,.2f}{shares:>10}"
            )
        lines.append("-" * 71)
        if self.cash_remaining:
            lines.append(f"Uninvested cash (whole-share rounding): ${self.cash_remaining:,.2f}")

        status = (
            "BLOCKED" if self.is_blocked
            else "NEEDS PRE-CLEARANCE" if self.needs_preclearance
            else "CLEAR"
        )
        lines += ["", f"Compliance status: {status}"]
        for r in self.results:
            for f in r.findings:
                if f.severity >= Severity.PRECLEARANCE:
                    lines.append(f"  - {r.trade.symbol}: {f}")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


def _resolve_fund(asset_class: AssetClass, overrides: Optional[dict[AssetClass, str]]) -> Instrument:
    symbol = (overrides or {}).get(asset_class) or catalog.DEFAULT_FUND[asset_class]
    return catalog.get_instrument(symbol)


def build_portfolio(
    amount: float,
    risk: str = "balanced",
    broker: str = "Vanguard",
    employee_group: EmployeeGroup = EmployeeGroup.GENERAL,
    prices: Optional[dict[str, float]] = None,
    fund_overrides: Optional[dict[AssetClass, str]] = None,
    context: Optional[ComplianceContext] = None,
) -> PortfolioPlan:
    """Construct a compliant portfolio for *amount* using a risk model.

    Args:
        amount: Total dollars to invest.
        risk: One of :data:`catalog.MODEL_PORTFOLIOS` (e.g. "balanced").
        broker: Broker the trades would be placed at (validated against policy).
        employee_group: Business unit, for unit-specific fund restrictions.
        prices: Optional ``{symbol: price}`` map; enables whole-share sizing.
        fund_overrides: Optional ``{AssetClass: symbol}`` to swap default funds.
        context: Optional pre-built :class:`ComplianceContext`; if omitted one
            is created from *employee_group*.

    Returns:
        A :class:`PortfolioPlan` whose trades have all been compliance-screened.
    """

    if amount <= 0:
        raise ValueError("amount must be positive")
    risk = risk.lower()
    if risk not in catalog.MODEL_PORTFOLIOS:
        raise ValueError(
            f"unknown risk model '{risk}'; choose from {', '.join(catalog.model_names())}"
        )

    weights = catalog.MODEL_PORTFOLIOS[risk]
    ctx = context or ComplianceContext(employee_group=employee_group)

    holdings: list[Holding] = []
    trades: list[ProposedTrade] = []
    warnings: list[str] = []
    cash_remaining = 0.0

    for asset_class, weight in weights.items():
        inst = _resolve_fund(asset_class, fund_overrides)
        alloc = round(amount * weight, 2)

        shares: Optional[float] = None
        if prices is not None:
            price = prices.get(inst.symbol)
            if price is None or price <= 0:
                warnings.append(f"No valid price for {inst.symbol}; share count omitted.")
            else:
                shares = math.floor(alloc / price)
                cash_remaining += alloc - shares * price

        holdings.append(Holding(instrument=inst, weight=weight, amount=alloc, shares=shares))
        trades.append(
            ProposedTrade(
                symbol=inst.symbol,
                instrument_type=inst.instrument_type,
                side=Side.BUY,
                order_type=OrderType.MARKET,  # policy-safe default
                broker=broker,
                notional=alloc,
                is_leveraged_or_inverse=inst.leveraged_or_inverse,
                bain_affiliation=inst.bain_affiliation,
            )
        )

    _add_structural_warnings(weights, holdings, warnings)
    results = screen_trades(trades, ctx)

    return PortfolioPlan(
        model_name=risk,
        broker=broker,
        amount=amount,
        holdings=holdings,
        trades=trades,
        results=results,
        warnings=warnings,
        cash_remaining=round(cash_remaining, 2),
    )


def _add_structural_warnings(
    weights: dict[AssetClass, float],
    holdings: list[Holding],
    warnings: list[str],
) -> None:
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        warnings.append(f"Model weights sum to {total:.3f}, not 1.0.")
    for h in holdings:
        if h.weight > CONCENTRATION_LIMIT:
            warnings.append(
                f"{h.symbol} is {h.weight * 100:.0f}% of the portfolio "
                f"(> {CONCENTRATION_LIMIT * 100:.0f}% concentration guideline)."
            )


def validate_holdings(
    trades: list[ProposedTrade],
    context: Optional[ComplianceContext] = None,
) -> list[ComplianceResult]:
    """Screen an arbitrary set of existing/proposed trades or holdings."""
    return screen_trades(trades, context)


@dataclass
class RebalanceOrder:
    symbol: str
    side: Side
    amount: float


def rebalance(
    current_values: dict[str, float],
    target_model: str = "balanced",
    fund_overrides: Optional[dict[AssetClass, str]] = None,
    min_trade: float = 50.0,
) -> list[RebalanceOrder]:
    """Compute buy/sell orders to move *current_values* toward a target model.

    ``current_values`` maps fund symbol -> current market value. Trades smaller
    than *min_trade* are skipped to avoid churn. Returns market-order intents
    (sells of diversified funds are permitted; they only require reporting).
    """

    if target_model.lower() not in catalog.MODEL_PORTFOLIOS:
        raise ValueError(f"unknown risk model '{target_model}'")

    total = sum(current_values.values())
    if total <= 0:
        raise ValueError("current portfolio value must be positive")

    weights = catalog.MODEL_PORTFOLIOS[target_model.lower()]
    target_value: dict[str, float] = {}
    for asset_class, weight in weights.items():
        sym = _resolve_fund(asset_class, fund_overrides).symbol
        target_value[sym] = target_value.get(sym, 0.0) + weight * total

    orders: list[RebalanceOrder] = []
    for sym in sorted(set(current_values) | set(target_value)):
        delta = target_value.get(sym, 0.0) - current_values.get(sym, 0.0)
        if abs(delta) < min_trade:
            continue
        side = Side.BUY if delta > 0 else Side.SELL
        orders.append(RebalanceOrder(symbol=sym, side=side, amount=round(abs(delta), 2)))
    return orders
