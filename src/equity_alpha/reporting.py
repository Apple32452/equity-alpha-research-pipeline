from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Mapping, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def save_backtest_outputs(
    output_dir: str | Path,
    baseline_bt: pd.DataFrame,
    improved_bt: pd.DataFrame,
    baseline_summary: Mapping,
    improved_summary: Mapping,
    comparison: Mapping,
    ic_summary: Mapping,
    regime_baseline: pd.DataFrame,
    regime_improved: pd.DataFrame,
) -> None:
    out = ensure_dir(output_dir)
    baseline_bt.to_csv(out / "baseline_daily_backtest.csv")
    improved_bt.to_csv(out / "improved_daily_backtest.csv")
    pd.DataFrame([baseline_summary]).to_csv(out / "baseline_summary.csv", index=False)
    pd.DataFrame([improved_summary]).to_csv(out / "improved_summary.csv", index=False)
    pd.DataFrame([comparison]).to_csv(out / "strategy_comparison.csv", index=False)
    pd.DataFrame([ic_summary]).to_csv(out / "rank_ic_summary.csv", index=False)
    if not regime_baseline.empty:
        regime_baseline.to_csv(out / "baseline_regime_summary.csv")
    if not regime_improved.empty:
        regime_improved.to_csv(out / "improved_regime_summary.csv")

    resume_metrics = {
        "resume_sharpe_before": round(float(comparison.get("sharpe_before", 0.0)), 2),
        "resume_sharpe_after": round(float(comparison.get("sharpe_after", 0.0)), 2),
        "resume_turnover_reduction_pct": round(float(comparison.get("turnover_reduction_pct", 0.0)), 1),
        "resume_max_drawdown_before_pct": round(float(comparison.get("drawdown_before_pct", 0.0)), 1),
        "resume_max_drawdown_after_pct": round(float(comparison.get("drawdown_after_pct", 0.0)), 1),
        "resume_slippage_sensitivity_reduction_pct": round(float(comparison.get("slippage_sensitivity_reduction_pct", 0.0)), 1),
        "resume_mean_rank_ic": round(float(ic_summary.get("mean_ic", 0.0)), 4),
        "resume_ic_tstat": round(float(ic_summary.get("ic_tstat", 0.0)), 2),
    }
    save_json(resume_metrics, out / "resume_metrics.json")


def plot_equity_curves(output_dir: str | Path, baseline_bt: pd.DataFrame, improved_bt: pd.DataFrame) -> None:
    out = ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(9, 5))
    baseline_bt["equity_curve"].plot(ax=ax, label="Baseline")
    improved_bt["equity_curve"].plot(ax=ax, label="Improved")
    ax.set_title("Net Equity Curve")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "equity_curve.png", dpi=180)
    plt.close(fig)


def plot_drawdowns(output_dir: str | Path, baseline_bt: pd.DataFrame, improved_bt: pd.DataFrame) -> None:
    out = ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, bt in [("Baseline", baseline_bt), ("Improved", improved_bt)]:
        wealth = (1 + bt["net_return"]).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        drawdown.plot(ax=ax, label=label)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "drawdown.png", dpi=180)
    plt.close(fig)


def plot_turnover(output_dir: str | Path, baseline_bt: pd.DataFrame, improved_bt: pd.DataFrame) -> None:
    out = ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(9, 5))
    baseline_bt["turnover"].rolling(21).mean().plot(ax=ax, label="Baseline")
    improved_bt["turnover"].rolling(21).mean().plot(ax=ax, label="Improved")
    ax.set_title("21-Day Average Turnover")
    ax.set_ylabel("Turnover")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "turnover.png", dpi=180)
    plt.close(fig)
