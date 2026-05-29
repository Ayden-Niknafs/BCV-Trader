"""Backward-looking valuation context for the 'ENTRY' assessment.

This module does NOT fetch or invent prices. It takes a *supplied* valuation
snapshot (a current metric and its 10-year mean/standard deviation) and computes
how rich or cheap the metric is versus history -- flagging attractive / neutral
/ stretched. Provenance (``as_of``) is mandatory so stale or illustrative data
is never mistaken for live data.

Load snapshots from JSON with :func:`load_snapshots`; see
``examples/valuation_sample.json`` for the schema (whose values are explicitly
illustrative and must be replaced with a live feed before acting).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ValuationFlag(Enum):
    ATTRACTIVE = "attractive"
    NEUTRAL = "neutral"
    STRETCHED = "stretched"


@dataclass(frozen=True)
class ValuationSnapshot:
    """A single valuation metric versus its 10-year history.

    ``higher_is_cheaper`` should be True for yield-type metrics (a higher yield
    is more attractive) and False for multiples like P/E or P/B (higher is
    richer / more stretched).
    """

    symbol: str
    metric: str
    current: float
    ten_year_mean: float
    as_of: str
    ten_year_std: Optional[float] = None
    higher_is_cheaper: bool = False
    source: str = ""

    @property
    def deviation_pct(self) -> float:
        """Signed % difference of the current metric vs its 10-year mean."""
        if self.ten_year_mean == 0:
            return 0.0
        return self.current / self.ten_year_mean - 1.0

    @property
    def zscore(self) -> Optional[float]:
        if not self.ten_year_std:
            return None
        return (self.current - self.ten_year_mean) / self.ten_year_std

    def flag(self, band: float = 0.10) -> ValuationFlag:
        """Attractive / neutral / stretched vs the 10-year mean.

        Uses the z-score when a 10-year standard deviation is available (|z| <
        0.5 == neutral), otherwise a +/-``band`` deviation around the mean.
        """

        z = self.zscore
        if z is not None:
            richness = z if not self.higher_is_cheaper else -z
            if richness > 0.5:
                return ValuationFlag.STRETCHED
            if richness < -0.5:
                return ValuationFlag.ATTRACTIVE
            return ValuationFlag.NEUTRAL

        dev = self.deviation_pct if not self.higher_is_cheaper else -self.deviation_pct
        if dev > band:
            return ValuationFlag.STRETCHED
        if dev < -band:
            return ValuationFlag.ATTRACTIVE
        return ValuationFlag.NEUTRAL

    def describe(self) -> str:
        return (
            f"{self.metric} {self.current:g} vs 10y mean {self.ten_year_mean:g} "
            f"({self.deviation_pct * 100:+.0f}%) -> {self.flag().value.upper()} "
            f"[as of {self.as_of}]"
        )


def load_snapshots(path: str) -> list[ValuationSnapshot]:
    """Load valuation snapshots from a JSON file.

    Keys beginning with ``_`` (e.g. ``_README``) are ignored, so the file can
    carry provenance/warning notes alongside the data.
    """

    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    entries = raw.get("instruments", raw) if isinstance(raw, dict) else raw
    snapshots: list[ValuationSnapshot] = []
    for item in entries:
        if isinstance(item, str) and item.startswith("_"):
            continue
        snapshots.append(ValuationSnapshot(**item))
    return snapshots
