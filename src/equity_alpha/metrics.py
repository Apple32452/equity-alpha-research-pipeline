from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TRADING_DAYS = 252


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())


def annualized_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return float("nan")
    total = float((1 + returns).prod())
    years = len(returns) / TRADING_DAYS
    if years <= 0 or total <= 0:
        return float("nan")
    return total ** (1 / years) - 1


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series) -> float:
    vol = annualized_volatility(returns)
    if not np.isfinite(vol) or vol == 0:
        return float("nan")
    return annualized_return(returns) / vol


def hit_rate(returns: pd.Series) -> float:
    x = returns.dropna()
    if x.empty:
        return float("nan")
    return float((x > 0).mean())


def summarize_backtest(bt: pd.DataFrame) -> dict[str, float]:
    r = bt["net_return"]
    gross_r = bt["gross_return"]
    return {
        "ann_return": annualized_return(r),
        "ann_vol": annualized_volatility(r),
        "sharpe": sharpe_ratio(r),
        "max_drawdown": max_drawdown(r),
        "hit_rate": hit_rate(r),
        "avg_daily_turnover": float(bt["turnover"].mean()),
        "avg_daily_cost_bps": float(bt["transaction_cost"].mean() * 10_000),
        "gross_sharpe_before_costs": sharpe_ratio(gross_r),
        "n_days": float(len(bt)),
    }


def daily_rank_ic(signal: pd.DataFrame, next_returns: pd.DataFrame) -> pd.Series:
    """Daily Spearman rank correlation between signal and next-day return."""
    sig = signal.reindex_like(next_returns)
    out = {}
    for dt in sig.index.intersection(next_returns.index):
        x = sig.loc[dt]
        y = next_returns.loc[dt]
        mask = x.notna() & y.notna()
        if mask.sum() < 10:
            out[dt] = np.nan
            continue
        val = spearmanr(x[mask], y[mask]).correlation
        out[dt] = val
    return pd.Series(out, name="rank_ic")


def summarize_ic(ic: pd.Series) -> dict[str, float]:
    x = ic.dropna()
    if x.empty:
        return {"mean_ic": float("nan"), "ic_tstat": float("nan"), "ic_positive_rate": float("nan")}
    return {
        "mean_ic": float(x.mean()),
        "ic_tstat": float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 1 and x.std(ddof=1) else float("nan"),
        "ic_positive_rate": float((x > 0).mean()),
    }


def regime_labels(market_returns: pd.Series, vol_lookback: int = 21) -> pd.Series:
    """Assign calm/normal/stress labels by rolling market volatility and drawdown."""
    mr = market_returns.fillna(0.0)
    rolling_vol = mr.rolling(vol_lookback, min_periods=10).std()
    wealth = (1 + mr).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0

    vol_q70 = rolling_vol.quantile(0.70)
    vol_q90 = rolling_vol.quantile(0.90)
    labels = pd.Series("normal", index=mr.index, dtype="object")
    labels.loc[rolling_vol <= rolling_vol.quantile(0.35)] = "calm"
    labels.loc[rolling_vol >= vol_q70] = "volatile"
    labels.loc[(rolling_vol >= vol_q90) | (drawdown <= -0.08)] = "stress"
    labels.loc[rolling_vol.isna()] = "warmup"
    return labels


def summarize_by_regime(bt: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    aligned = bt.join(regimes.rename("regime"), how="left")
    rows = []
    for regime, g in aligned.groupby("regime"):
        if regime == "warmup" or len(g) < 20:
            continue
        s = summarize_backtest(g)
        s["regime"] = regime
        rows.append(s)
    return pd.DataFrame(rows).set_index("regime").sort_index() if rows else pd.DataFrame()


def compare_summaries(baseline: dict[str, float], improved: dict[str, float]) -> dict[str, float]:
    return {
        "sharpe_before": baseline.get("sharpe", np.nan),
        "sharpe_after": improved.get("sharpe", np.nan),
        "sharpe_delta": improved.get("sharpe", np.nan) - baseline.get("sharpe", np.nan),
        "turnover_reduction_pct": 100 * (baseline.get("avg_daily_turnover", np.nan) - improved.get("avg_daily_turnover", np.nan)) / baseline.get("avg_daily_turnover", np.nan),
        "drawdown_before_pct": 100 * abs(baseline.get("max_drawdown", np.nan)),
        "drawdown_after_pct": 100 * abs(improved.get("max_drawdown", np.nan)),
        "drawdown_reduction_pct": 100 * (abs(baseline.get("max_drawdown", np.nan)) - abs(improved.get("max_drawdown", np.nan))) / abs(baseline.get("max_drawdown", np.nan)),
        "cost_reduction_bps": baseline.get("avg_daily_cost_bps", np.nan) - improved.get("avg_daily_cost_bps", np.nan),
    }
