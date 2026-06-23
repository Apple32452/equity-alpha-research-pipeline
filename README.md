# Cross-Sectional Equity Alpha Research Pipeline

A reproducible daily equity-research backtest for evaluating simple cross-sectional signals under next-day-return, transaction-cost-aware, liquidity-aware, and regime-diagnostic settings.

> **Research prototype, not a live trading system.**
> This repository is designed for transparent historical simulation and portfolio-research workflows. It does not represent live PnL, production execution, or institutional-grade market-impact modeling.

---

## Overview

The pipeline downloads or generates daily OHLCV panels, constructs cross-sectional alpha features, forms long/short portfolios, applies transaction-cost assumptions, and exports performance, turnover, drawdown, rank-IC, slippage-sensitivity, and regime-diagnostic outputs.

The project supports two data modes:

* `synthetic`: deterministic simulated data for validating the code path.
* `yfinance`: daily historical OHLCV data for a supplied U.S. equity universe.

Synthetic results are **only** for testing the pipeline. Do not use synthetic results in a resume, portfolio, or professional discussion of strategy performance.

---

## Research question

The project compares:

1. A simple cross-sectional momentum baseline.
2. A composite signal and portfolio-construction workflow with liquidity filtering, inverse-volatility sizing, turnover controls, and drawdown-triggered risk reduction.

The comparison is intentionally a **combined signal-and-portfolio-construction experiment**. A performance difference should not be interpreted as evidence that one individual alpha feature alone caused the improvement.

---

## Pipeline design

### 1. Data

The expected long-panel format is:

```text
date, ticker, open, high, low, close, adj_close, volume
```

For `yfinance` runs, the pipeline downloads daily OHLCV data and retains tickers meeting the requested history requirement.

The project uses:

* adjusted-close returns for feature and return calculations;
* close price times volume for average dollar-volume estimates;
* a user-supplied ticker universe.

### 2. Features

All signal features are computed from information available at close (t). Portfolio weights formed at close (t) are evaluated using the return from close (t) to close (t+1).

Implemented features:

* 21-day momentum;
* 63-day momentum;
* 5-day short-term reversal;
* 21-day realized volatility;
* 63-day realized volatility;
* 20-day average dollar volume;
* cross-sectional liquidity rank;
* cross-sectional rank-normalized feature inputs.

### 3. Signals

#### Baseline signal

```text
signal = 0.70 * rank(momentum_63)
       + 0.30 * rank(momentum_21)
```

#### Composite signal

```text
signal = 0.66 * rank(momentum_63)
       + 0.27 * rank(momentum_21)
       + 0.05 * rank(reversal_5)
       + 0.02 * rank(low_volatility_21)
```

The composite signal emphasizes medium-term momentum while adding a modest short-term reversal and low-volatility adjustment.

### 4. Portfolio construction

The baseline portfolio:

* forms an equal-gross long/short portfolio;
* buys the top decile and shorts the bottom decile of the baseline signal;
* uses no explicit liquidity filter, volatility scaling, or turnover controls.

The improved workflow:

* removes stocks below the 30th percentile of cross-sectional liquidity;
* forms long/short positions from the top and bottom approximately 18% of the remaining universe;
* applies inverse-volatility scaling;
* suppresses small position changes with a trade threshold;
* limits daily turnover;
* reduces gross exposure after sufficiently large trailing-market drawdowns.

The strategy is a research portfolio with long/short targets. Before interpreting results as market-neutral, inspect realized net exposure after volatility scaling and trade-control transformations.

### 5. Backtest timing

For each trading date (t):

1. Compute features from data available at close (t).
2. Construct the cross-sectional signal at close (t).
3. Form target portfolio weights at close (t).
4. Apply target weights to close-to-close returns from (t) to (t+1).
5. Deduct estimated transaction costs based on daily changes in portfolio weights.

This setup avoids same-day signal-return look-ahead. It is a forward-return historical backtest, not a rolling train/validation/test or walk-forward model-selection framework.

---

## Transaction-cost model

Daily costs include:

* commission in basis points;
* half-spread in basis points;
* slippage in basis points;
* a relative liquidity multiplier based on 20-day average dollar volume.

The model is intended for sensitivity analysis. It does **not** include:

* portfolio-NAV-based participation rates;
* calibrated square-root market impact;
* intraday order-book depth;
* borrow costs;
* exchange fees or rebates;
* production execution logic.

Therefore, all results should be described as **simulated historical, transaction-cost-aware research results**.

---

## Metrics and diagnostics

The pipeline exports:

* annualized return;
* annualized volatility;
* a risk-adjusted return statistic currently reported in code as `sharpe`;
* gross risk-adjusted return before estimated costs;
* maximum drawdown;
* daily turnover;
* average daily transaction cost;
* rank information coefficient;
* rank-IC t-statistic;
* calm, volatile, and stress regime summaries;
* slippage-sensitivity results;
* equity-curve, drawdown, and turnover figures.

### Important metric conventions

* The current `sharpe` field is calculated as annualized geometric return divided by annualized volatility. It is best interpreted as an internal risk-adjusted return statistic, not a conventional excess-return Sharpe ratio.
* The rank-IC t-statistic assumes independent daily IC observations and is not HAC or Newey--West adjusted.
* Regime labels are retrospective diagnostics based on full-sample volatility quantiles and drawdown conditions. They are not a real-time regime-forecasting model.

---

## Repository structure

```text
equity-alpha-research-pipeline/
├── data/
│   └── universe/
│       ├── example_universe_50.csv
│       └── example_universe_300.csv
├── outputs/
│   └── demo/
├── src/
│   └── equity_alpha/
│       ├── backtest.py
│       ├── data.py
│       ├── features.py
│       ├── metrics.py
│       ├── portfolio.py
│       ├── reporting.py
│       ├── run_pipeline.py
│       └── signals.py
├── tests/
│   └── test_metrics.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run the unit tests:

```bash
pytest -q
```

---

## Run the synthetic validation demo

Use the synthetic mode first to confirm that the code, dependencies, figures, and output files work correctly.

```bash
python -m equity_alpha.run_pipeline \
  --data-source synthetic \
  --max-tickers 300 \
  --synthetic-days 756 \
  --seed 7 \
  --output-dir outputs/demo
```

Expected outputs include:

```text
outputs/demo/baseline_daily_backtest.csv
outputs/demo/improved_daily_backtest.csv
outputs/demo/baseline_summary.csv
outputs/demo/improved_summary.csv
outputs/demo/strategy_comparison.csv
outputs/demo/rank_ic_summary.csv
outputs/demo/baseline_regime_summary.csv
outputs/demo/improved_regime_summary.csv
outputs/demo/slippage_sensitivity.csv
outputs/demo/resume_metrics.json
outputs/demo/equity_curve.png
outputs/demo/drawdown.png
outputs/demo/turnover.png
```

Do not use synthetic performance outputs as resume or portfolio claims.

---

## Run on historical U.S. equities

The included universe files are examples only. For professional analysis, use a documented and liquidity-screened universe whose composition you understand.

```bash
python -m equity_alpha.run_pipeline \
  --data-source yfinance \
  --universe-file data/universe/example_universe_300.csv \
  --max-tickers 300 \
  --start 2023-01-01 \
  --end 2026-01-01 \
  --min-history 500 \
  --commission-bps 1.0 \
  --half-spread-bps 2.0 \
  --slippage-bps 4.0 \
  --output-dir outputs/real_run
```

After the run, inspect:

```text
outputs/real_run/strategy_comparison.csv
outputs/real_run/rank_ic_summary.csv
outputs/real_run/improved_regime_summary.csv
outputs/real_run/slippage_sensitivity.csv
outputs/real_run/resume_metrics.json
```

Always verify every number in `resume_metrics.json` against the corresponding CSV outputs before using it externally.

---

## Reproducibility checklist

For every run that may support a resume bullet, interview discussion, or project result, save:

* exact universe CSV used;
* requested and retained ticker counts;
* data-download timestamp;
* start and end dates;
* commission, spread, and slippage assumptions;
* repository commit hash;
* Python and package versions;
* all exported CSV files and figures.

Because yfinance data can change through corrections, delistings, ticker changes, or vendor adjustments, preserve the downloaded raw panel or a versioned processed panel for any reported result.

---

## Appropriate resume language

Use wording such as:

> Built a cross-sectional equity research pipeline across a documented U.S. equity universe, evaluating momentum, reversal, volatility, and liquidity features with forward next-day returns, transaction-cost assumptions, and regime diagnostics.

> Compared a baseline momentum portfolio with a composite signal-and-portfolio-construction workflow incorporating liquidity filters, volatility scaling, turnover controls, and drawdown-triggered risk reduction.

> Reported simulated historical performance using transaction-cost-aware returns, turnover, drawdown, rank IC, and slippage-sensitivity diagnostics.

Avoid describing this project as:

* live trading;
* production execution;
* institutional-grade backtesting;
* intraday market making;
* a survivorship-bias-free performance study;
* validated live PnL.

---

## Limitations

This project intentionally does not include:

* survivorship-bias-free point-in-time universe construction;
* delisting-return treatment;
* point-in-time fundamentals;
* institutional corporate-action validation;
* borrow availability or borrow fees;
* short-sale constraints;
* tax effects;
* calibrated market impact;
* portfolio-NAV-based participation rates;
* intraday execution simulation;
* order-book data;
* production trading infrastructure;
* walk-forward hyperparameter tuning;
* statistical multiple-testing corrections.

It is therefore best used as a transparent, reproducible equity-alpha research project and a foundation for further model-validation work.
