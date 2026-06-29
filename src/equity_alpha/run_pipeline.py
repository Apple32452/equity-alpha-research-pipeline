```python
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from equity_alpha.backtest import CostModel, run_backtest
from equity_alpha.data import (
    download_yfinance_panel,
    generate_synthetic_panel,
    load_universe,
)
from equity_alpha.features import build_features, feature_frame
from equity_alpha.metrics import (
    compare_summaries,
    daily_rank_ic,
    regime_labels,
    summarize_backtest,
    summarize_by_regime,
    summarize_ic,
)
from equity_alpha.portfolio import build_baseline_weights, build_improved_weights
from equity_alpha.reporting import (
    ensure_dir,
    plot_drawdowns,
    plot_equity_curves,
    plot_turnover,
    save_backtest_outputs,
)
from equity_alpha.signals import build_baseline_signal, build_improved_signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-sectional equity alpha research pipeline"
    )
    parser.add_argument(
        "--data-source",
        choices=["synthetic", "yfinance"],
        default="synthetic",
    )
    parser.add_argument(
        "--universe-file",
        default="data/universe/example_universe_300.csv",
    )
    parser.add_argument("--max-tickers", type=int, default=300)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--min-history", type=int, default=252)
    parser.add_argument("--synthetic-days", type=int, default=756)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gross", type=float, default=1.0)
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--half-spread-bps", type=float, default=2.0)
    parser.add_argument("--slippage-bps", type=float, default=4.0)
    parser.add_argument("--output-dir", default="outputs/demo")
    parser.add_argument("--save-feature-frame", action="store_true")
    return parser.parse_args()


def get_git_commit_hash() -> str | None:
    """Return the current Git commit hash when available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def count_requested_tickers(universe_file: str, max_tickers: int) -> int | None:
    """Count requested tickers for historical runs."""
    try:
        return len(load_universe(universe_file, max_tickers=max_tickers))
    except (FileNotFoundError, KeyError, ValueError):
        return None


def save_run_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    panel: pd.DataFrame,
    requested_tickers: int | None,
) -> None:
    """Save reproducibility metadata for each experiment."""
    dates = pd.to_datetime(panel["date"])

    metadata = {
        "data_source": args.data_source,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": get_git_commit_hash(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "universe_file": args.universe_file if args.data_source == "yfinance" else None,
        "requested_ticker_count": requested_tickers,
        "retained_ticker_count": int(panel["ticker"].nunique()),
        "panel_start_date": str(dates.min().date()),
        "panel_end_date": str(dates.max().date()),
        "panel_trading_days": int(dates.nunique()),
        "panel_rows": int(len(panel)),
        "requested_start_date": args.start,
        "requested_end_date": args.end,
        "minimum_history_days": args.min_history,
        "synthetic_days": args.synthetic_days if args.data_source == "synthetic" else None,
        "random_seed": args.seed if args.data_source == "synthetic" else None,
        "gross_exposure_target": args.gross,
        "commission_bps": args.commission_bps,
        "half_spread_bps": args.half_spread_bps,
        "slippage_bps": args.slippage_bps,
        "notes": (
            "Historical daily OHLCV data downloaded through yfinance."
            if args.data_source == "yfinance"
            else "Synthetic validation data; do not interpret performance as market results."
        ),
    }

    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    requested_tickers = None

    if args.data_source == "synthetic":
        panel = generate_synthetic_panel(
            n_tickers=args.max_tickers,
            n_days=args.synthetic_days,
            seed=args.seed,
        )
    else:
        tickers = load_universe(
            args.universe_file,
            max_tickers=args.max_tickers,
        )
        requested_tickers = len(tickers)

        panel = download_yfinance_panel(
            tickers=tickers,
            start=args.start,
            end=args.end,
            min_history=args.min_history,
        )

    if panel.empty:
        raise ValueError(
            "No data were returned. Check the universe file, date range, "
            "internet connection, and minimum-history requirement."
        )

    save_run_metadata(
        output_dir=output_dir,
        args=args,
        panel=panel,
        requested_tickers=requested_tickers,
    )

    features = build_features(panel)

    if args.save_feature_frame:
        feature_frame(features).to_csv(
            output_dir / "feature_frame.csv",
            index=True,
        )

    baseline_signal = build_baseline_signal(features)
    improved_signal = build_improved_signal(features)

    # Signals use information at close t; performance uses close-to-close t+1 returns.
    next_returns = features["next_ret_1d"]
    market_returns = features["ret_1d"].mean(axis=1)

    baseline_weights = build_baseline_weights(
        baseline_signal,
        gross=args.gross,
    )

    improved_weights = build_improved_weights(
        improved_signal,
        liquidity_rank=features["liquidity_rank"],
        volatility=features["volatility_21"],
        market_returns=market_returns,
        gross=args.gross,
    )

    cost_model = CostModel(
        commission_bps=args.commission_bps,
        half_spread_bps=args.half_spread_bps,
        slippage_bps=args.slippage_bps,
    )

    baseline_bt = run_backtest(
        baseline_weights,
        next_returns=next_returns,
        adv20=features["adv20"],
        cost_model=cost_model,
    )

    improved_bt = run_backtest(
        improved_weights,
        next_returns=next_returns,
        adv20=features["adv20"],
        cost_model=cost_model,
    )

    baseline_summary = summarize_backtest(baseline_bt)
    improved_summary = summarize_backtest(improved_bt)
    comparison = compare_summaries(baseline_summary, improved_summary)

    # Re-run the same portfolios across transaction-cost slippage assumptions.
    slippage_grid = [0.0, 2.0, 4.0, 8.0, 12.0]
    slippage_rows = []

    for slip_bps in slippage_grid:
        stress_cost_model = CostModel(
            commission_bps=args.commission_bps,
            half_spread_bps=args.half_spread_bps,
            slippage_bps=slip_bps,
        )

        baseline_stress_bt = run_backtest(
            baseline_weights,
            next_returns=next_returns,
            adv20=features["adv20"],
            cost_model=stress_cost_model,
        )

        improved_stress_bt = run_backtest(
            improved_weights,
            next_returns=next_returns,
            adv20=features["adv20"],
            cost_model=stress_cost_model,
        )

        baseline_stress_summary = summarize_backtest(baseline_stress_bt)
        improved_stress_summary = summarize_backtest(improved_stress_bt)

        slippage_rows.append(
            {
                "slippage_bps": slip_bps,
                "baseline_risk_adjusted_return": baseline_stress_summary["sharpe"],
                "improved_risk_adjusted_return": improved_stress_summary["sharpe"],
                "baseline_avg_daily_cost_bps": baseline_stress_summary[
                    "avg_daily_cost_bps"
                ],
                "improved_avg_daily_cost_bps": improved_stress_summary[
                    "avg_daily_cost_bps"
                ],
            }
        )

    slippage_stress = pd.DataFrame(slippage_rows)

    if len(slippage_stress) >= 2:
        baseline_slope = float(
            np.polyfit(
                slippage_stress["slippage_bps"],
                slippage_stress["baseline_risk_adjusted_return"],
                1,
            )[0]
        )

        improved_slope = float(
            np.polyfit(
                slippage_stress["slippage_bps"],
                slippage_stress["improved_risk_adjusted_return"],
                1,
            )[0]
        )

        comparison["baseline_slippage_sensitivity_per_bp"] = baseline_slope
        comparison["improved_slippage_sensitivity_per_bp"] = improved_slope

        if abs(baseline_slope) > 1e-12:
            comparison["slippage_sensitivity_reduction_pct"] = (
                100
                * (abs(baseline_slope) - abs(improved_slope))
                / abs(baseline_slope)
            )

    rank_ic = daily_rank_ic(improved_signal, next_returns)
    ic_summary = summarize_ic(rank_ic)

    # These are retrospective diagnostics, not real-time predicted regimes.
    regimes = regime_labels(market_returns)
    regime_baseline = summarize_by_regime(baseline_bt, regimes)
    regime_improved = summarize_by_regime(improved_bt, regimes)

    save_backtest_outputs(
        output_dir=output_dir,
        baseline_bt=baseline_bt,
        improved_bt=improved_bt,
        baseline_summary=baseline_summary,
        improved_summary=improved_summary,
        comparison=comparison,
        ic_summary=ic_summary,
        regime_baseline=regime_baseline,
        regime_improved=regime_improved,
    )

    slippage_stress.to_csv(
        output_dir / "slippage_sensitivity.csv",
        index=False,
    )

    plot_equity_curves(output_dir, baseline_bt, improved_bt)
    plot_drawdowns(output_dir, baseline_bt, improved_bt)
    plot_turnover(output_dir, baseline_bt, improved_bt)

    print("Saved outputs to:", output_dir.resolve())
    print("Data source:", args.data_source)
    print("Retained tickers:", panel["ticker"].nunique())
    print(
        "Observed date range:",
        pd.to_datetime(panel["date"]).min().date(),
        "to",
        pd.to_datetime(panel["date"]).max().date(),
    )
    print("\nBaseline summary:")
    print(pd.Series(baseline_summary).round(4).to_string())
    print("\nImproved summary:")
    print(pd.Series(improved_summary).round(4).to_string())
    print("\nComparison:")
    print(pd.Series(comparison).round(4).to_string())
    print("\nMetadata saved to:", output_dir / "run_metadata.json")
    print("Resume metrics saved to:", output_dir / "resume_metrics.json")


if __name__ == "__main__":
    main()
```
