from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from equity_alpha.backtest import CostModel, run_backtest
from equity_alpha.data import download_yfinance_panel, generate_synthetic_panel, load_universe
from equity_alpha.features import build_features, feature_frame
from equity_alpha.metrics import (
    compare_summaries,
    daily_rank_ic,
    regime_labels,
    summarize_backtest,
    summarize_by_regime,
    summarize_ic,
)
import numpy as np
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
    parser = argparse.ArgumentParser(description="Cross-sectional equity alpha research pipeline")
    parser.add_argument("--data-source", choices=["synthetic", "yfinance"], default="synthetic")
    parser.add_argument("--universe-file", default="data/universe/example_universe_300.csv")
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


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    if args.data_source == "synthetic":
        panel = generate_synthetic_panel(n_tickers=args.max_tickers, n_days=args.synthetic_days, seed=args.seed)
    else:
        tickers = load_universe(args.universe_file, max_tickers=args.max_tickers)
        panel = download_yfinance_panel(
            tickers=tickers,
            start=args.start,
            end=args.end,
            min_history=args.min_history,
        )

    features = build_features(panel)
    if args.save_feature_frame:
        ff = feature_frame(features)
        ff.to_csv(output_dir / "feature_frame.csv")

    baseline_signal = build_baseline_signal(features)
    improved_signal = build_improved_signal(features)
    next_returns = features["next_ret_1d"]
    market_returns = features["ret_1d"].mean(axis=1)

    baseline_weights = build_baseline_weights(baseline_signal, gross=args.gross)
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

    # Slippage sensitivity: rerun both portfolios under multiple slippage assumptions.
    # Slope is change in Sharpe per one additional slippage bp. A less negative slope
    # means the strategy is less sensitive to execution assumptions.
    slippage_grid = [0.0, 2.0, 4.0, 8.0, 12.0]
    slippage_rows = []
    for slip_bps in slippage_grid:
        cm = CostModel(
            commission_bps=args.commission_bps,
            half_spread_bps=args.half_spread_bps,
            slippage_bps=slip_bps,
        )
        b_bt = run_backtest(baseline_weights, next_returns=next_returns, adv20=features["adv20"], cost_model=cm)
        i_bt = run_backtest(improved_weights, next_returns=next_returns, adv20=features["adv20"], cost_model=cm)
        b_sum = summarize_backtest(b_bt)
        i_sum = summarize_backtest(i_bt)
        slippage_rows.append({
            "slippage_bps": slip_bps,
            "baseline_sharpe": b_sum["sharpe"],
            "improved_sharpe": i_sum["sharpe"],
            "baseline_avg_daily_cost_bps": b_sum["avg_daily_cost_bps"],
            "improved_avg_daily_cost_bps": i_sum["avg_daily_cost_bps"],
        })
    slippage_stress = pd.DataFrame(slippage_rows)
    if len(slippage_stress) >= 2:
        base_slope = float(np.polyfit(slippage_stress["slippage_bps"], slippage_stress["baseline_sharpe"], 1)[0])
        imp_slope = float(np.polyfit(slippage_stress["slippage_bps"], slippage_stress["improved_sharpe"], 1)[0])
        comparison["baseline_slippage_sensitivity_sharpe_per_bp"] = base_slope
        comparison["improved_slippage_sensitivity_sharpe_per_bp"] = imp_slope
        if abs(base_slope) > 1e-12:
            comparison["slippage_sensitivity_reduction_pct"] = 100 * (abs(base_slope) - abs(imp_slope)) / abs(base_slope)

    ic = daily_rank_ic(improved_signal, next_returns)
    ic_summary = summarize_ic(ic)
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
    slippage_stress.to_csv(output_dir / "slippage_sensitivity.csv", index=False)
    plot_equity_curves(output_dir, baseline_bt, improved_bt)
    plot_drawdowns(output_dir, baseline_bt, improved_bt)
    plot_turnover(output_dir, baseline_bt, improved_bt)

    print("Saved outputs to", Path(output_dir).resolve())
    print("Baseline summary:")
    print(pd.Series(baseline_summary).round(4).to_string())
    print("\nImproved summary:")
    print(pd.Series(improved_summary).round(4).to_string())
    print("\nComparison:")
    print(pd.Series(comparison).round(4).to_string())
    print("\nResume metrics saved to", Path(output_dir, "resume_metrics.json"))


if __name__ == "__main__":
    main()
