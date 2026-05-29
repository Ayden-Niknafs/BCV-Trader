"""A universe of *compliant* building blocks and model portfolios.

Because the policy prohibits buying individual equities, bonds, IPOs, and
derivatives, a compliant personal strategy is built from broad-based, low-cost
pooled funds (ETFs / index funds). This catalog lists a handful of widely
available examples and groups them by asset class, then defines risk-based
model allocations on top of those asset classes.

Prices are intentionally not hard-coded -- pass a live price map into the
strategy builder when you want share quantities. Everything here is data, so it
is trivial to extend or swap for a firm-approved fund list.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .compliance import BainAffiliation, InstrumentType


class AssetClass(Enum):
    US_EQUITY = "US equity"
    INTL_DEVELOPED_EQUITY = "International developed equity"
    EM_EQUITY = "Emerging-market equity"
    US_BOND = "US investment-grade bonds"
    INTL_BOND = "International bonds"
    TIPS = "Inflation-protected bonds"
    REIT = "Real estate (REIT)"
    CASH = "Cash / money market"


@dataclass(frozen=True)
class Instrument:
    """A pooled fund usable as a portfolio building block."""

    symbol: str
    name: str
    instrument_type: InstrumentType
    asset_class: AssetClass
    expense_ratio: float  # annual, as a decimal (e.g. 0.0003 == 3 bps)
    leveraged_or_inverse: bool = False
    bain_affiliation: BainAffiliation = BainAffiliation.NONE


# A small, diversified, broad-market universe. All entries are pooled funds
# (ETF / index / money-market), which is what keeps them compliant.
UNIVERSE: dict[str, Instrument] = {
    inst.symbol: inst
    for inst in (
        Instrument("VTI", "Vanguard Total US Stock Market ETF",
                   InstrumentType.ETF, AssetClass.US_EQUITY, 0.0003),
        Instrument("VOO", "Vanguard S&P 500 ETF",
                   InstrumentType.ETF, AssetClass.US_EQUITY, 0.0003),
        Instrument("SCHB", "Schwab US Broad Market ETF",
                   InstrumentType.ETF, AssetClass.US_EQUITY, 0.0003),
        Instrument("VXUS", "Vanguard Total International Stock ETF",
                   InstrumentType.ETF, AssetClass.INTL_DEVELOPED_EQUITY, 0.0005),
        Instrument("IEFA", "iShares Core MSCI EAFE ETF",
                   InstrumentType.ETF, AssetClass.INTL_DEVELOPED_EQUITY, 0.0007),
        Instrument("VWO", "Vanguard FTSE Emerging Markets ETF",
                   InstrumentType.ETF, AssetClass.EM_EQUITY, 0.0008),
        Instrument("BND", "Vanguard Total US Bond Market ETF",
                   InstrumentType.ETF, AssetClass.US_BOND, 0.0003),
        Instrument("AGG", "iShares Core US Aggregate Bond ETF",
                   InstrumentType.ETF, AssetClass.US_BOND, 0.0003),
        Instrument("BNDX", "Vanguard Total International Bond ETF",
                   InstrumentType.ETF, AssetClass.INTL_BOND, 0.0007),
        Instrument("VTIP", "Vanguard Short-Term Inflation-Protected Securities ETF",
                   InstrumentType.ETF, AssetClass.TIPS, 0.0004),
        Instrument("VNQ", "Vanguard Real Estate ETF",
                   InstrumentType.ETF, AssetClass.REIT, 0.0013),
        Instrument("VMFXX", "Vanguard Federal Money Market Fund",
                   InstrumentType.MONEY_MARKET_FUND, AssetClass.CASH, 0.0011),
    )
}

# Default fund used to implement each asset class in a model portfolio.
DEFAULT_FUND: dict[AssetClass, str] = {
    AssetClass.US_EQUITY: "VTI",
    AssetClass.INTL_DEVELOPED_EQUITY: "VXUS",
    AssetClass.EM_EQUITY: "VWO",
    AssetClass.US_BOND: "BND",
    AssetClass.INTL_BOND: "BNDX",
    AssetClass.TIPS: "VTIP",
    AssetClass.REIT: "VNQ",
    AssetClass.CASH: "VMFXX",
}


# Risk-based model portfolios expressed as asset-class weights (each sums to 1).
MODEL_PORTFOLIOS: dict[str, dict[AssetClass, float]] = {
    "conservative": {
        AssetClass.US_EQUITY: 0.20,
        AssetClass.INTL_DEVELOPED_EQUITY: 0.10,
        AssetClass.US_BOND: 0.45,
        AssetClass.TIPS: 0.15,
        AssetClass.CASH: 0.10,
    },
    "balanced": {
        AssetClass.US_EQUITY: 0.36,
        AssetClass.INTL_DEVELOPED_EQUITY: 0.18,
        AssetClass.EM_EQUITY: 0.06,
        AssetClass.US_BOND: 0.28,
        AssetClass.INTL_BOND: 0.06,
        AssetClass.REIT: 0.06,
    },
    "aggressive": {
        AssetClass.US_EQUITY: 0.50,
        AssetClass.INTL_DEVELOPED_EQUITY: 0.25,
        AssetClass.EM_EQUITY: 0.13,
        AssetClass.US_BOND: 0.05,
        AssetClass.REIT: 0.07,
    },
}


def get_instrument(symbol: str) -> Instrument:
    try:
        return UNIVERSE[symbol.upper()]
    except KeyError:
        raise KeyError(f"Unknown instrument '{symbol}'. Known: {', '.join(sorted(UNIVERSE))}")


def model_names() -> list[str]:
    return list(MODEL_PORTFOLIOS)
