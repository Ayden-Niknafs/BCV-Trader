"""Rules-driven momentum & trend signals for policy-compliant funds.

Honest framing
--------------
Moving-average crosses and RSI require real price history. This module computes
them *correctly from a supplied price series* -- it never invents prices. Feed
it daily closes (oldest first, newest last) via :func:`load_prices`. Signals are
mechanical and backward-looking; they are not predictions, and momentum can
reverse without warning.

Compliance gate
---------------
Under the BCV personal-trading policy only broad, open-end funds get a trade
call. Sector/industry (narrow-based) funds, and non-open-end structures (e.g.
unit investment trusts), return "consult Compliance" and NO entry signal -- the
gate is applied before any momentum math runs.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .valuation import ValuationFlag


# --------------------------------------------------------------------------- #
# Fund structure / breadth registry (structural facts, not market data).       #
# --------------------------------------------------------------------------- #
class Scope(Enum):
    BROAD = "broad market"
    STYLE = "broad style tilt"
    FACTOR = "broad single-factor"
    SECTOR = "single sector"
    INDUSTRY = "single industry"


class Structure(Enum):
    OPEN_END = "open-end fund"
    UIT = "unit investment trust"


@dataclass(frozen=True)
class FundProfile:
    symbol: str
    name: str
    structure: Structure
    scope: Scope
    holdings: int
    note: str = ""


# Approximate, structural metadata for the requested universe. Holdings counts
# are round figures; the scope/structure classifications are the policy-relevant
# part and are deliberately conservative.
FUND_PROFILE: dict[str, FundProfile] = {
    "VTI": FundProfile("VTI", "Vanguard Total US Stock Market",
                       Structure.OPEN_END, Scope.BROAD, 3500),
    "VXUS": FundProfile("VXUS", "Vanguard Total International Stock",
                        Structure.OPEN_END, Scope.BROAD, 8500),
    "BND": FundProfile("BND", "Vanguard Total US Bond Market",
                       Structure.OPEN_END, Scope.BROAD, 11000),
    "VUG": FundProfile("VUG", "Vanguard Growth (large-cap growth style)",
                       Structure.OPEN_END, Scope.STYLE, 200,
                       "Multi-sector style tilt; broad-based but growth-concentrated."),
    "MTUM": FundProfile("MTUM", "iShares MSCI USA Momentum Factor",
                        Structure.OPEN_END, Scope.FACTOR, 125,
                        "Single-factor; multi-sector but rebalances can concentrate."),
    "QQQ": FundProfile("QQQ", "Invesco QQQ Trust (Nasdaq-100)",
                       Structure.UIT, Scope.STYLE, 100,
                       "UIT structure (not open-end); ~50% top-10 tech. QQQM is the "
                       "open-end equivalent."),
    "VGT": FundProfile("VGT", "Vanguard Information Technology",
                       Structure.OPEN_END, Scope.SECTOR, 320),
    "XLK": FundProfile("XLK", "Technology Select Sector SPDR",
                       Structure.OPEN_END, Scope.SECTOR, 65),
    "XBI": FundProfile("XBI", "SPDR S&P Biotech (equal weight)",
                       Structure.OPEN_END, Scope.INDUSTRY, 140),
}


@dataclass(frozen=True)
class ComplianceStatus:
    symbol: str
    structure: Structure
    scope: Scope
    holdings: int
    broad: bool
    open_end: bool
    market_order_compatible: bool
    trade_call_allowed: bool
    reasons: tuple[str, ...]

    def summary(self) -> str:
        verdict = "TRADE CALL ALLOWED" if self.trade_call_allowed else "CONSULT COMPLIANCE"
        return f"{verdict} — {'; '.join(self.reasons)}"


def compliance_status(symbol: str) -> ComplianceStatus:
    """Apply the policy gate to a fund's structure/breadth (no market data)."""
    p = FUND_PROFILE.get(symbol.upper())
    if p is None:
        return ComplianceStatus(
            symbol.upper(), Structure.OPEN_END, Scope.BROAD, 0, False, True, True, False,
            ("Unknown fund; not on the confirmed universe — consult Compliance.",),
        )

    open_end = p.structure == Structure.OPEN_END
    broad = p.scope in (Scope.BROAD, Scope.STYLE, Scope.FACTOR)
    reasons: list[str] = [f"{p.structure.value}", f"{p.scope.value}", f"~{p.holdings} holdings"]
    allowed = True

    if not open_end:
        allowed = False
        reasons.append("policy requires open-end structure")
    if p.scope in (Scope.SECTOR, Scope.INDUSTRY):
        allowed = False
        reasons.append("narrow-based (sector/industry) — not a broad fund")
    if p.scope in (Scope.STYLE, Scope.FACTOR) and allowed:
        reasons.append("broad but concentrated/tilted — size position with care")
    if p.note:
        reasons.append(p.note)

    return ComplianceStatus(
        symbol=p.symbol,
        structure=p.structure,
        scope=p.scope,
        holdings=p.holdings,
        broad=broad,
        open_end=open_end,
        # ETFs and open-end funds both transact fine with a market order / at NAV.
        market_order_compatible=True,
        trade_call_allowed=allowed,
        reasons=tuple(reasons),
    )


# --------------------------------------------------------------------------- #
# Indicators (computed from a supplied price series; newest last).             #
# --------------------------------------------------------------------------- #
class TrendSignal(Enum):
    GOLDEN_CROSS = "golden cross (50d > 200d)"
    DEATH_CROSS = "death cross (50d < 200d)"
    NEUTRAL = "neutral"
    INSUFFICIENT_DATA = "insufficient data"


class RSIBand(Enum):
    OVERBOUGHT = "overbought (>70)"
    NEUTRAL = "neutral"
    OVERSOLD = "oversold (<30)"


class EntryRating(Enum):
    STRONG = "STRONG"
    FAVORABLE = "FAVORABLE"
    NEUTRAL = "NEUTRAL"
    UNFAVORABLE = "UNFAVORABLE"
    AVOID = "AVOID"


def sma(prices: list[float], window: int, offset: int = 0) -> float:
    """Simple moving average of ``window`` closes ending ``offset`` bars back."""
    end = len(prices) - offset
    if window <= 0 or end < window:
        raise ValueError("not enough data for the requested SMA window")
    segment = prices[end - window:end]
    return sum(segment) / window


def rsi(prices: list[float], period: int = 14) -> float:
    """Wilder's RSI on the supplied closes (needs > ``period`` points; more is
    better for a stable value)."""
    if len(prices) < period + 1:
        raise ValueError("not enough data for RSI")
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(deltas)):  # Wilder smoothing
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


@dataclass(frozen=True)
class TrendResult:
    signal: TrendSignal
    sma_short: Optional[float] = None
    sma_long: Optional[float] = None
    fresh_cross: bool = False


def trend_signal(
    prices: list[float],
    short: int = 50,
    long: int = 200,
    fresh_lookback: int = 10,
) -> TrendResult:
    if len(prices) < long:
        return TrendResult(TrendSignal.INSUFFICIENT_DATA)
    s, l = sma(prices, short), sma(prices, long)
    diff = s - l
    if diff > 0:
        sig = TrendSignal.GOLDEN_CROSS
    elif diff < 0:
        sig = TrendSignal.DEATH_CROSS
    else:
        sig = TrendSignal.NEUTRAL

    fresh = False
    if len(prices) >= long + fresh_lookback:
        prev = sma(prices, short, fresh_lookback) - sma(prices, long, fresh_lookback)
        fresh = (prev <= 0 < diff) or (prev >= 0 > diff)
    return TrendResult(sig, s, l, fresh)


def rsi_band(value: float) -> RSIBand:
    if value > 70:
        return RSIBand.OVERBOUGHT
    if value < 30:
        return RSIBand.OVERSOLD
    return RSIBand.NEUTRAL


# --------------------------------------------------------------------------- #
# Combined entry rating.                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MomentumResult:
    symbol: str
    trend: TrendResult
    rsi_value: float
    rsi_zone: RSIBand
    valuation_flag: Optional[ValuationFlag]
    score: int
    rating: EntryRating
    components: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        t = self.trend
        ma = (f"{t.signal.value}"
              + (" [FRESH]" if t.fresh_cross else "")
              + (f" (50d {t.sma_short:.2f} / 200d {t.sma_long:.2f})"
                 if t.sma_short is not None else ""))
        val = self.valuation_flag.value.upper() if self.valuation_flag else "n/a"
        return (f"MA: {ma} | RSI(14): {self.rsi_value:.0f} "
                f"[{self.rsi_zone.value}] | Valuation: {val} "
                f"=> ENTRY RATING: {self.rating.value} (score {self.score:+d})")


def _rating_from_score(score: int) -> EntryRating:
    if score >= 2:
        return EntryRating.STRONG
    if score == 1:
        return EntryRating.FAVORABLE
    if score == 0:
        return EntryRating.NEUTRAL
    if score == -1:
        return EntryRating.UNFAVORABLE
    return EntryRating.AVOID


def momentum_signals(
    symbol: str,
    prices: list[float],
    valuation_flag: Optional[ValuationFlag] = None,
    short: int = 50,
    long: int = 200,
    rsi_period: int = 14,
) -> MomentumResult:
    """Compute trend + RSI + (optional) valuation into an ENTRY RATING.

    Transparent scoring (each in {-1, 0, +1}, summed):
      * Trend: golden cross +1, neutral 0, death cross -1
      * RSI:   oversold +1 (better entry), neutral 0, overbought -1
      * Valuation (if provided): attractive +1, neutral 0, stretched -1
    Score -> STRONG (>=2) / FAVORABLE (1) / NEUTRAL (0) / UNFAVORABLE (-1) /
    AVOID (<=-2). A "fresh" cross is reported but does not change the score.
    """

    trend = trend_signal(prices, short, long)
    rsi_value = rsi(prices, rsi_period)
    zone = rsi_band(rsi_value)

    trend_score = {TrendSignal.GOLDEN_CROSS: 1, TrendSignal.DEATH_CROSS: -1}.get(
        trend.signal, 0)
    rsi_score = {RSIBand.OVERSOLD: 1, RSIBand.OVERBOUGHT: -1}.get(zone, 0)
    components = {"trend": trend_score, "rsi": rsi_score}
    if valuation_flag is not None:
        components["valuation"] = {
            ValuationFlag.ATTRACTIVE: 1, ValuationFlag.STRETCHED: -1
        }.get(valuation_flag, 0)

    score = sum(components.values())
    return MomentumResult(
        symbol=symbol.upper(),
        trend=trend,
        rsi_value=rsi_value,
        rsi_zone=zone,
        valuation_flag=valuation_flag,
        score=score,
        rating=_rating_from_score(score),
        components=components,
    )


# --------------------------------------------------------------------------- #
# Price loading.                                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PriceSeries:
    closes: list[float]
    as_of: str = ""
    source: str = ""


def load_prices(path: str) -> dict[str, PriceSeries]:
    """Load daily closes from JSON or CSV.

    JSON: ``{"QQQ": [c1, c2, ...]}`` (oldest first) or
          ``{"QQQ": {"closes": [...], "as_of": "...", "source": "..."}}``.
    CSV:  columns including ``symbol`` and ``close`` (ascending by date), or a
          single-symbol file with ``date,close`` (symbol taken from filename).
    """

    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        out: dict[str, PriceSeries] = {}
        for sym, val in raw.items():
            if sym.startswith("_"):
                continue
            if isinstance(val, dict):
                out[sym.upper()] = PriceSeries(
                    [float(x) for x in val["closes"]],
                    val.get("as_of", ""), val.get("source", ""))
            else:
                out[sym.upper()] = PriceSeries([float(x) for x in val])
        return out

    # CSV
    series: dict[str, list[float]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        for row in reader:
            sym = (row[cols["symbol"]].upper() if "symbol" in cols else "SERIES")
            series.setdefault(sym, []).append(float(row[cols["close"]]))
    return {sym: PriceSeries(closes) for sym, closes in series.items()}
