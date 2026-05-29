"""Transparent, assumption-driven analytics for compliant portfolios.

Honest framing
--------------
We cannot predict short-term prices, hand out "entry/exit" levels, or quote a
point "probability of success" for a timed trade -- and we will not fabricate
that precision. Instead this module uses standard, fully-disclosed long-run
*capital-market assumptions* (CMAs) and well-established models to estimate:

* expected return (gross and net of fund fees),
* portfolio risk (mean-variance / Modern Portfolio Theory),
* the model-implied probability of meeting a goal over a holding *horizon*
  (a lognormal terminal-wealth model), reported as a distribution.

Every assumption below is visible and editable in one place. None of this is a
guarantee or a prediction of any specific outcome; markets deviate from any
model, and past/assumed returns do not ensure future results.

Models used
-----------
* Mean-variance portfolio math (Markowitz / MPT): portfolio return is the
  weighted mean; portfolio variance uses asset volatilities and a correlation
  matrix, so diversification lowers risk below the weighted-average volatility.
* CAPM-style building blocks: expected returns are risk-free + an asset risk
  premium, consistent with long-run historical equity/credit premia.
* Lognormal terminal-wealth model: annual gross returns are treated as
  i.i.d. lognormal, so T-year log-growth is Normal(m*T, s^2*T). This yields
  closed-form probabilities and percentiles (via ``statistics.NormalDist``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

from . import catalog
from .catalog import AssetClass

# --------------------------------------------------------------------------- #
# Capital-market assumptions (ILLUSTRATIVE long-run, nominal). Edit here.      #
# Order-of-magnitude consistent with widely published 10-yr CMAs and long-run #
# historical premia. These are assumptions, not forecasts.                    #
# --------------------------------------------------------------------------- #
INFLATION = 0.025  # assumed long-run inflation

# asset class -> (expected nominal annual return, annual volatility)
CMA: dict[AssetClass, tuple[float, float]] = {
    AssetClass.US_EQUITY: (0.068, 0.155),
    AssetClass.INTL_DEVELOPED_EQUITY: (0.072, 0.170),
    AssetClass.EM_EQUITY: (0.080, 0.205),
    AssetClass.US_BOND: (0.040, 0.050),
    AssetClass.INTL_BOND: (0.035, 0.050),
    AssetClass.TIPS: (0.037, 0.055),
    AssetClass.REIT: (0.066, 0.180),
    AssetClass.CASH: (0.030, 0.010),
}

# Short, transparent role descriptions for the building-block "research card".
ROLE: dict[AssetClass, str] = {
    AssetClass.US_EQUITY: "Core growth engine; equity risk premium",
    AssetClass.INTL_DEVELOPED_EQUITY: "Geographic diversification of equity risk",
    AssetClass.EM_EQUITY: "Higher expected return / higher volatility",
    AssetClass.US_BOND: "Ballast; income; dampens drawdowns",
    AssetClass.INTL_BOND: "Rate/currency diversification of the bond sleeve",
    AssetClass.TIPS: "Explicit inflation hedge",
    AssetClass.REIT: "Real-asset diversifier; inflation sensitivity",
    AssetClass.CASH: "Liquidity; near-zero volatility",
}

# Correlation by broad category -- intentionally coarse and disclosed.
_EQUITY = {AssetClass.US_EQUITY, AssetClass.INTL_DEVELOPED_EQUITY, AssetClass.EM_EQUITY}
_BONDS = {AssetClass.US_BOND, AssetClass.INTL_BOND, AssetClass.TIPS}

_CATEGORY_CORR: dict[tuple[str, str], float] = {
    ("bond", "bond"): 0.60,
    ("bond", "cash"): 0.00,
    ("bond", "equity"): 0.10,
    ("bond", "reit"): 0.20,
    ("cash", "cash"): 0.00,
    ("cash", "equity"): 0.00,
    ("cash", "reit"): 0.00,
    ("equity", "equity"): 0.80,
    ("equity", "reit"): 0.70,
    ("reit", "reit"): 1.00,
}


def _category(ac: AssetClass) -> str:
    if ac in _EQUITY:
        return "equity"
    if ac in _BONDS:
        return "bond"
    if ac == AssetClass.REIT:
        return "reit"
    return "cash"


def correlation(a: AssetClass, b: AssetClass) -> float:
    if a == b:
        return 1.0
    key = tuple(sorted((_category(a), _category(b))))
    return _CATEGORY_CORR.get(key, 0.30)  # documented fallback


def real_return(nominal: float, inflation: float = INFLATION) -> float:
    return (1.0 + nominal) / (1.0 + inflation) - 1.0


# --------------------------------------------------------------------------- #
# Portfolio statistics.                                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PortfolioStats:
    expected_return_nominal: float
    expected_return_real: float
    volatility: float
    sharpe: float  # excess over cash, per unit of volatility

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"E[return] {self.expected_return_nominal * 100:.2f}% nominal / "
            f"{self.expected_return_real * 100:.2f}% real, "
            f"vol {self.volatility * 100:.2f}%, Sharpe {self.sharpe:.2f}"
        )


def _normalize_weights(weights: dict[AssetClass, float]) -> dict[AssetClass, float]:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    return {ac: w / total for ac, w in weights.items()}


def portfolio_expected_return(weights: dict[AssetClass, float]) -> float:
    w = _normalize_weights(weights)
    return sum(wt * CMA[ac][0] for ac, wt in w.items())


def portfolio_volatility(weights: dict[AssetClass, float]) -> float:
    w = _normalize_weights(weights)
    items = list(w.items())
    var = 0.0
    for ac_i, wi in items:
        for ac_j, wj in items:
            si, sj = CMA[ac_i][1], CMA[ac_j][1]
            var += wi * wj * si * sj * correlation(ac_i, ac_j)
    return math.sqrt(max(var, 0.0))


def portfolio_stats(weights: dict[AssetClass, float]) -> PortfolioStats:
    mu = portfolio_expected_return(weights)
    vol = portfolio_volatility(weights)
    cash = CMA[AssetClass.CASH][0]
    sharpe = (mu - cash) / vol if vol > 0 else float("inf")
    return PortfolioStats(
        expected_return_nominal=mu,
        expected_return_real=real_return(mu),
        volatility=vol,
        sharpe=sharpe,
    )


# --------------------------------------------------------------------------- #
# Lognormal terminal-wealth model.                                             #
# --------------------------------------------------------------------------- #
def _lognormal_params(mu: float, sigma: float) -> tuple[float, float]:
    """Per-year log-return mean/std for a lognormal gross return with arithmetic
    mean ``mu`` and volatility ``sigma``."""
    s2 = math.log(1.0 + (sigma ** 2) / ((1.0 + mu) ** 2))
    m = math.log(1.0 + mu) - s2 / 2.0
    return m, math.sqrt(s2)


def _growth_dist(weights: dict[AssetClass, float], years: float) -> NormalDist:
    mu = portfolio_expected_return(weights)
    sigma = portfolio_volatility(weights)
    m, s = _lognormal_params(mu, sigma)
    return NormalDist(m * years, s * math.sqrt(years))


def probability_at_least(
    weights: dict[AssetClass, float],
    years: float,
    target_multiple: float,
) -> float:
    """Model-implied P(terminal wealth >= ``target_multiple`` x initial)."""
    if target_multiple <= 0:
        return 1.0
    dist = _growth_dist(weights, years)
    return 1.0 - dist.cdf(math.log(target_multiple))


def probability_beat_inflation(
    weights: dict[AssetClass, float],
    years: float,
    real_cagr_target: float = 0.0,
) -> float:
    """P(real CAGR over the horizon >= ``real_cagr_target``).

    ``real_cagr_target=0`` means "preserve purchasing power"; e.g. ``0.04``
    means "beat inflation by 4%/yr in real terms".
    """
    nominal_multiple = ((1.0 + INFLATION) * (1.0 + real_cagr_target)) ** years
    return probability_at_least(weights, years, nominal_multiple)


def terminal_real_cagr_percentiles(
    weights: dict[AssetClass, float],
    years: float,
    percentiles: tuple[float, ...] = (0.05, 0.50, 0.95),
) -> dict[float, float]:
    """Percentiles of the *real* annualized return over the horizon."""
    dist = _growth_dist(weights, years)
    out: dict[float, float] = {}
    for p in percentiles:
        log_growth = dist.inv_cdf(p)
        nominal_multiple = math.exp(log_growth)
        real_multiple = nominal_multiple / ((1.0 + INFLATION) ** years)
        out[p] = real_multiple ** (1.0 / years) - 1.0
    return out


def ideal_holding_period(
    weights: dict[AssetClass, float],
    success_threshold: float = 0.90,
    real_cagr_target: float = 0.0,
    max_years: int = 40,
) -> int | None:
    """Smallest whole-year horizon where P(beat inflation target) >= threshold.

    This reframes "ideal exit": rather than a price level, it is the holding
    period after which the goal is met with high model-implied confidence.
    Returns ``None`` if the threshold is not reached within ``max_years``.
    """
    for t in range(1, max_years + 1):
        if probability_beat_inflation(weights, t, real_cagr_target) >= success_threshold:
            return t
    return None


# --------------------------------------------------------------------------- #
# Building-block "research cards".                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BuildingBlock:
    asset_class: AssetClass
    symbol: str
    name: str
    gross_return: float
    expense_ratio: float
    volatility: float
    role: str

    @property
    def net_return(self) -> float:
        return self.gross_return - self.expense_ratio


def building_blocks(weights: dict[AssetClass, float]) -> list[BuildingBlock]:
    """Per-asset research cards for the funds implementing a model."""
    blocks: list[BuildingBlock] = []
    for ac in weights:
        symbol = catalog.DEFAULT_FUND[ac]
        inst = catalog.get_instrument(symbol)
        ret, vol = CMA[ac]
        blocks.append(
            BuildingBlock(
                asset_class=ac,
                symbol=symbol,
                name=inst.name,
                gross_return=ret,
                expense_ratio=inst.expense_ratio,
                volatility=vol,
                role=ROLE.get(ac, ""),
            )
        )
    return blocks
