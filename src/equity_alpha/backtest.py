from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 1.0
    half_spread_bps: float = 2.0
    slippage_bps: float = 4.0
    slippage_exponent: float = 0.50

    @property
    def fixed_bps(self) -> float:
        return self.commission_bps + self.half_spread_bps


def estimate_transaction_costs(
    weights: pd.DataFrame,
    adv20: pd.DataFrame | None = None,
    cost_model: CostModel = CostModel(),
) -> pd.Series:
    """Estimate daily costs as fixed bps on turnover plus liquidity-scaled slippage.

    weights are portfolio weights. A weight change of 0.10 means trading 10% NAV.
    If ADV is supplied, slippage is reduced for the most liquid names and increased
    for less-liquid names by a cross-sectional liquidity multiplier.
    """
    trades = weights.fillna(0.0).diff().fillna(weights.fillna(0.0)).abs()
    fixed_cost = trades.sum(axis=1) * (cost_model.fixed_bps / 10_000.0)
    if adv20 is None:
        slip = trades.sum(axis=1) * (cost_model.slippage_bps / 10_000.0)
        return fixed_cost + slip

    liq = adv20.reindex_like(weights).replace(0, np.nan)
    # More liquid names get multiplier below 1, less liquid names above 1.
    rel_liq = liq.div(liq.median(axis=1).replace(0, np.nan), axis=0)
    liq_multiplier = (1.0 / rel_liq.clip(lower=0.10)) ** cost_model.slippage_exponent
    liq_multiplier = liq_multiplier.clip(0.25, 4.0).fillna(1.0)
    slip = (trades * liq_multiplier).sum(axis=1) * (cost_model.slippage_bps / 10_000.0)
    return fixed_cost + slip


def run_backtest(
    weights: pd.DataFrame,
    next_returns: pd.DataFrame,
    adv20: pd.DataFrame | None = None,
    cost_model: CostModel = CostModel(),
) -> pd.DataFrame:
    """Run daily close-to-close backtest.

    weights at date t are multiplied by next_returns at t, which should be return from
    close t to close t+1. This avoids look-ahead when signals use close-t data.
    """
    w = weights.reindex_like(next_returns).fillna(0.0)
    r = next_returns.reindex_like(w).fillna(0.0)
    gross_ret = (w * r).sum(axis=1)
    costs = estimate_transaction_costs(w, adv20=adv20, cost_model=cost_model).reindex(gross_ret.index).fillna(0.0)
    net_ret = gross_ret - costs
    turnover = w.diff().fillna(w).abs().sum(axis=1)
    gross_exposure = w.abs().sum(axis=1)
    out = pd.DataFrame(
        {
            "gross_return": gross_ret,
            "transaction_cost": costs,
            "net_return": net_ret,
            "turnover": turnover,
            "gross_exposure": gross_exposure,
        }
    )
    out["equity_curve"] = (1 + out["net_return"]).cumprod()
    return out.dropna()
