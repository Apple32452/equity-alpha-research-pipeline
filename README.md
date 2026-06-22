# Cross-Sectional Equity Alpha Research Pipeline

This project is the real, defensible version of the resume bullets below:

> Built a cross-sectional equity research pipeline across 300+ U.S. equities over 3 years of historical data, testing statistical signals under out-of-sample, regime-split, and transaction-cost-aware evaluation.
>
> Designed alpha-evaluation logic using rank correlation, volatility normalization, turnover controls, drawdown diagnostics, and liquidity-aware filters; improved simulated transaction-cost-adjusted Sharpe from X.XX to Y.YY.
>
> Reduced simulated turnover by X% and slippage sensitivity by Y% by adding volatility-scaled sizing, liquidity screens, and trade/no-trade thresholds.
>
> Cut maximum drawdown from X% to Y% in regime-stress tests by applying downside-risk filters and execution-aware position sizing.

The project intentionally separates **real research outputs** from **resume wording**. Run it on real historical data first, then copy the values from `outputs/real_run/resume_metrics.json` into the resume. Do not use the synthetic demo results as resume claims.

---

## What this project does

### 1. Data pipeline

The pipeline can run in two modes:

- `synthetic`: creates a realistic test panel so you can verify the code without internet access.
- `yfinance`: downloads daily OHLCV data for a user-provided U.S. equity universe.

Input panel format:

```text
date, ticker, open, high, low, close, adj_close, volume
```

### 2. Feature generation

Features are computed at close `t` and evaluated against close-to-close return `t+1` to avoid look-ahead bias.

Implemented features:

- 21-day momentum
- 63-day momentum
- 5-day short-term reversal
- 21-day and 63-day realized volatility
- 20-day average dollar volume
- cross-sectional liquidity rank
- cross-sectional rank-normalized alpha inputs

### 3. Signal generation

The project compares two strategy versions:

#### Baseline strategy

Simple medium-term cross-sectional momentum:

```text
signal = 0.70 * rank(momentum_63) + 0.30 * rank(momentum_21)
```

#### Improved strategy

Composite alpha signal:

```text
signal = 0.42 * rank(momentum_63)
       + 0.23 * rank(momentum_21)
       + 0.20 * rank(reversal_5)
       + 0.15 * rank(low_volatility_21)
```

The improved portfolio then adds:

- liquidity-aware stock filtering
- volatility-scaled sizing
- trade/no-trade threshold to reduce small rebalances
- turnover cap
- downside-risk filter during stress regimes
- transaction-cost and liquidity-sensitive slippage model

### 4. Backtest design

For each date:

1. Compute features using data available at close `t`.
2. Build signal at close `t`.
3. Form long/short portfolio weights at close `t`.
4. Apply weights to next-day returns from close `t` to close `t+1`.
5. Subtract transaction costs and slippage.

This is a simple daily research backtest, not an intraday execution simulator.

### 5. Evaluation metrics

The output includes:

- transaction-cost-adjusted Sharpe
- gross Sharpe before costs
- annualized return and volatility
- maximum drawdown
- daily turnover
- average daily transaction cost in basis points
- rank information coefficient (Spearman rank correlation)
- calm / volatile / stress regime summaries
- equity curve plot
- drawdown plot
- turnover plot

---

## Folder structure

```text
equity_alpha_research_pipeline/
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

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run tests:

```bash
pytest -q
```

---

## Run the synthetic demo

Use this first to check that everything works:

```bash
python -m equity_alpha.run_pipeline \
  --data-source synthetic \
  --max-tickers 300 \
  --synthetic-days 756 \
  --seed 7 \
  --output-dir outputs/demo
```

Expected files:

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

Again: synthetic results are only for verifying the code.

---

## Run on real U.S. equities

The included `example_universe_300.csv` is only a starter universe. Before using results professionally, verify that the universe matches what you want: S&P 500 names, Russell 1000 names, NASDAQ liquid universe, or your own liquidity-screened list.

Example command:

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

After the run, open:

```text
outputs/real_run/resume_metrics.json
outputs/real_run/strategy_comparison.csv
outputs/real_run/improved_regime_summary.csv
outputs/real_run/slippage_sensitivity.csv
```

---

## How to make the resume numbers honest

After running on real data, replace the resume numbers using the JSON file.

Example format:

```json
{
  "resume_sharpe_before": 0.84,
  "resume_sharpe_after": 1.07,
  "resume_turnover_reduction_pct": 12.0,
  "resume_max_drawdown_before_pct": 18.6,
  "resume_max_drawdown_after_pct": 14.9,
  "resume_slippage_sensitivity_reduction_pct": 9.0,
  "resume_mean_rank_ic": 0.0184,
  "resume_ic_tstat": 2.11
}
```

Then your resume bullet should become:

```text
Designed alpha-evaluation logic using rank correlation, volatility normalization,
turnover controls, drawdown diagnostics, and liquidity-aware filters; improved
simulated transaction-cost-adjusted Sharpe from 0.84 to 1.07.
```

Only use the numbers if they come from your real run.

---

## Defensible interview explanation

A concise explanation:

> I built a daily cross-sectional equity alpha research pipeline. It downloads OHLCV data, creates momentum, reversal, volatility, and liquidity features, ranks stocks cross-sectionally each day, and tests a long/short portfolio using next-day returns. I compared a baseline momentum signal against an improved version with liquidity filters, volatility-scaled sizing, trade thresholds, turnover caps, and a downside-risk filter. I evaluated the strategy using transaction-cost-adjusted Sharpe, rank IC, max drawdown, turnover, and regime-split performance.

If asked why this is not overclaiming:

> It is a research backtest, not a live production trading system. The transaction-cost and slippage model is simulated, so I report the results as simulated and transaction-cost-adjusted, not live PnL.

---

## Important limitations

This project is intentionally realistic but still simple. It does not include:

- survivorship-bias-free universe construction
- point-in-time fundamentals
- corporate-action validation beyond adjusted close
- intraday order book simulation
- borrow costs for shorting
- market impact model calibrated to proprietary execution data
- production execution stack

That is why the resume should say **simulated**, **historical**, and **research pipeline** rather than live trading or production execution.

---

## Suggested final resume version after real run

Replace all `X` values after running the real data pipeline:

```text
Quantitative Researcher | Remote / Part-Time
• Built a cross-sectional equity research pipeline across 300+ U.S. equities over 3 years of historical data, testing statistical signals under out-of-sample, regime-split, and transaction-cost-aware evaluation.
• Designed alpha-evaluation logic using rank correlation, volatility normalization, turnover controls, drawdown diagnostics, and liquidity-aware filters; improved simulated transaction-cost-adjusted Sharpe from X.XX to Y.YY.
• Reduced simulated turnover by X% and slippage sensitivity by Y% by adding volatility-scaled sizing, liquidity screens, and trade/no-trade thresholds.
• Cut maximum drawdown from X.X% to Y.Y% in regime-stress tests by applying downside-risk filters and execution-aware position sizing.
```

The slippage sensitivity number comes from `slippage_sensitivity.csv`, where both strategies are re-run across several slippage assumptions and compared by the slope of Sharpe versus slippage bps.
