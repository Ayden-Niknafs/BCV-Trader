#!/usr/bin/env python3
"""Fetch daily closing prices into the JSON format `bcv_trader momentum` expects.

Pulls from the public Yahoo Finance chart endpoint (no API key). Output:

    {"VTI": {"closes": [...oldest..newest...], "as_of": "YYYY-MM-DD",
             "source": "Yahoo Finance chart API"}, ...}

Usage:
    python scripts/fetch_prices.py --symbols VTI,VXUS,BND,VUG,MTUM --out prices.json
    python -m bcv_trader momentum --prices prices.json

Standard library only. Prices are *not* committed to the repo: regenerate them
so signals are computed on current data, never stale numbers.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import urllib.request

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range={rng}&interval=1d"


def fetch_one(symbol: str, rng: str = "2y") -> dict:
    url = CHART_URL.format(sym=symbol, rng=rng)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=20).read().decode()
    result = json.loads(raw)["chart"]["result"][0]
    timestamps = result["timestamp"]
    closes_raw = result["indicators"]["quote"][0]["close"]
    closes, last_ts = [], None
    for ts, c in zip(timestamps, closes_raw):
        if c is not None:
            closes.append(round(float(c), 4))
            last_ts = ts
    as_of = dt.datetime.utcfromtimestamp(last_ts).date().isoformat() if last_ts else ""
    return {"closes": closes, "as_of": as_of, "source": "Yahoo Finance chart API"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", required=True, help="Comma-separated tickers.")
    p.add_argument("--range", default="2y", dest="rng", help="History range (e.g. 2y, 5y).")
    p.add_argument("--out", required=True, help="Output JSON path.")
    args = p.parse_args(argv)

    out: dict[str, dict] = {}
    for sym in (s.strip().upper() for s in args.symbols.split(",") if s.strip()):
        try:
            out[sym] = fetch_one(sym, args.rng)
            print(f"{sym}: {len(out[sym]['closes'])} closes through {out[sym]['as_of']}",
                  file=sys.stderr)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"{sym}: FETCH FAILED ({type(e).__name__}: {e})", file=sys.stderr)
        time.sleep(0.25)  # be polite to the endpoint

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"wrote {args.out} ({len(out)} symbols)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
