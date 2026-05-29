# BCV-Trader

A **compliant personal-trading strategy tool**. It encodes the firm's
*Personal Compliance* policy (from New Hire Onboarding) into a rules engine and
builds diversified portfolios that are screened against it.

The guiding idea: under this policy, a trader's edge can't come from
individual-name selection, IPOs, derivatives, shorting, or any information or
timing advantage over client accounts — all of that is prohibited. So the tool
leans into the only edge left: **disciplined, low-cost, diversified asset
allocation** using broad-based pooled funds at an approved broker, with market
orders, full disclosure, and pre-clearance where required.

> This is a compliance/educational tool, not investment advice, and not a
> substitute for your Compliance team. Always pre-clear and report through the
> official system (BCCS).

## What it enforces

Encoded directly from the policy deck (rules carry a slide reference):

| Bright line | How the tool handles it |
|---|---|
| No individual equities / bonds / loans / IPOs / limited secondaries | `screen_trade` **blocks** the buy |
| No derivatives or short sales | **blocked** |
| No limit / stop-loss / GTC orders on pre-clearance instruments | **blocked**; strategy uses market orders only |
| No trading on MNPI; no front-running/tailgating client trades | **blocked** (flags + firm restricted list) |
| Approved broker required | unapproved brokers/affiliates **blocked** |
| New private-investment commitments | flagged **pre-clearance required** |
| Unit funds (BC Credit 40-Act, Public Equity UCITS) | **blocked** for the restricted groups |
| Disclose accounts (10d) / holdings as-of (45d) / quarterly (30d) | `onboarding_deadlines()` computes the dates |

## Install / run

Pure standard library — no dependencies. From the repo root:

```bash
python -m bcv_trader build --amount 100000 --risk balanced --broker Vanguard
python -m bcv_trader screen --symbol AAPL --type individual_equity --broker Robinhood
python -m bcv_trader brokers
python -m bcv_trader models
python -m bcv_trader analyze --risk balanced --target-real 0.04
python -m bcv_trader report --valuation examples/valuation_sample.json
python -m bcv_trader disclosures --start-date 2026-05-29

# Momentum/trend entry signals on real prices (fetch fresh data first):
python scripts/fetch_prices.py --symbols VTI,VXUS,BND,VUG,MTUM --out prices.json
python -m bcv_trader momentum --prices prices.json --valuation examples/valuation_sample.json
```

The momentum command applies the compliance gate first: sector/industry
(narrow-based) and non-open-end funds return "consult Compliance" with **no**
trade call; only broad, open-end funds get a signal.

Optionally install it so the `bcv-trader` command is on your PATH:

```bash
pip install -e .
bcv-trader build --amount 100000 --risk aggressive
```

## Use as a library

```python
from bcv_trader import build_portfolio, screen_trade, ProposedTrade, InstrumentType

# Build and screen a compliant portfolio in one call.
plan = build_portfolio(100_000, risk="balanced", broker="Vanguard")
print(plan.summary())
assert plan.is_compliant

# Screen an arbitrary trade.
result = screen_trade(ProposedTrade(
    symbol="AAPL",
    instrument_type=InstrumentType.INDIVIDUAL_EQUITY,
    broker="Robinhood",
))
print(result.is_blocked, [f.rule_id for f in result.blocking_findings()])
```

## Project layout

```
bcv_trader/
  compliance.py   # the rules engine: instruments, order types, brokers,
                  # pre-clearance, prohibited practices, disclosure deadlines
  catalog.py      # compliant building blocks (broad-based ETFs/funds) + models
  strategy.py     # builds & screens diversified portfolios; rebalancing
  analytics.py    # CMAs, MPT stats, lognormal goal-probability, research cards
  valuation.py    # backward-looking attractive/neutral/stretched flags (supplied data)
  momentum.py     # compliance gate + 50/200 MA cross, RSI(14), entry rating
  cli.py          # `python -m bcv_trader ...`
scripts/
  fetch_prices.py # pull daily closes (Yahoo) into the momentum --prices format
examples/         # illustrative sample data (clearly marked non-live)
tests/            # unittest suite (no external deps)
```

> Data honesty: the engine never fabricates prices. Valuation/momentum inputs
> are either supplied by you or fetched live (with `as_of`/source recorded);
> the bundled `examples/` data is explicitly illustrative and stale.

## Tests

```bash
python -m unittest discover -s tests -v
```
