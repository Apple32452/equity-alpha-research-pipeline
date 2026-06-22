from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_gross(weights: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
    denom = weights.abs().sum(axis=1).replace(0, np.nan)
    return weights.div(denom, axis=0).fillna(0.0) * gross


def apply_liquidity_filter(signal: pd.DataFrame, liquidity_rank: pd.DataFrame, min_rank: float = 0.25) -> pd.DataFrame:
    filtered = signal.where(liquidity_rank >= min_rank)
    return filtered


def build_long_short_weights(
    signal: pd.DataFrame,
    long_quantile: float = 0.80,
    short_quantile: float = 0.20,
    gross: float = 1.0,
) -> pd.DataFrame:
    """Equal-gross long/short portfolio from cross-sectional signal ranks."""
    long_cut = signal.quantile(long_quantile, axis=1)
    short_cut = signal.quantile(short_quantile, axis=1)
    longs = signal.ge(long_cut, axis=0).astype(float)
    shorts = -signal.le(short_cut, axis=0).astype(float)

    long_w = longs.div(longs.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * (gross / 2)
    short_w = shorts.div(shorts.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * (gross / 2)
    return (long_w + short_w).fillna(0.0)


def apply_volatility_scaled_sizing(
    raw_weights: pd.DataFrame,
    volatility: pd.DataFrame,
    min_vol: float = 0.005,
    max_leverage_multiplier: float = 2.0,
    gross: float = 1.0,
) -> pd.DataFrame:
    """Downweight high-volatility names and re-normalize gross exposure."""
    inv_vol = 1.0 / volatility.clip(lower=min_vol)
    inv_vol = inv_vol.div(inv_vol.median(axis=1).replace(0, np.nan), axis=0).clip(0, max_leverage_multiplier)
    adjusted = raw_weights * inv_vol
    return _normalize_gross(adjusted, gross=gross)


def apply_trade_threshold(weights: pd.DataFrame, threshold: float = 0.0025) -> pd.DataFrame:
    """Suppress tiny position changes to reduce churn."""
    out = weights.copy().fillna(0.0)
    prev = pd.Series(0.0, index=out.columns)
    rows = []
    for dt, row in out.iterrows():
        delta = row - prev
        row = row.where(delta.abs() >= threshold, prev)
        rows.append(row)
        prev = row
    return pd.DataFrame(rows, index=out.index, columns=out.columns).fillna(0.0)


def apply_turnover_cap(weights: pd.DataFrame, max_daily_turnover: float = 0.35) -> pd.DataFrame:
    """Linearly scale trades when daily turnover exceeds max_daily_turnover."""
    out = weights.copy().fillna(0.0)
    prev = pd.Series(0.0, index=out.columns)
    rows = []
    for dt, target in out.iterrows():
        delta = target - prev
        turnover = delta.abs().sum()
        if turnover > max_daily_turnover and turnover > 0:
            target = prev + delta * (max_daily_turnover / turnover)
        rows.append(target)
        prev = target
    return pd.DataFrame(rows, index=out.index, columns=out.columns).fillna(0.0)


def downside_risk_filter(
    weights: pd.DataFrame,
    market_returns: pd.Series,
    lookback: int = 21,
    drawdown_threshold: float = -0.06,
    risk_scale: float = 0.50,
) -> pd.DataFrame:
    """Reduce gross exposure when recent market drawdown breaches a threshold."""
    wealth = (1 + market_returns.fillna(0.0)).cumprod()
    rolling_peak = wealth.rolling(lookback, min_periods=max(5, lookback // 3)).max()
    rolling_dd = wealth / rolling_peak - 1.0
    scale = pd.Series(1.0, index=weights.index)
    scale.loc[rolling_dd.reindex(weights.index) <= drawdown_threshold] = risk_scale
    return weights.mul(scale, axis=0)


def build_baseline_weights(signal: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
    # Naive baseline: more concentrated top/bottom decile book with no explicit
    # liquidity filter, volatility scaling, or turnover control.
    return build_long_short_weights(signal, long_quantile=0.90, short_quantile=0.10, gross=gross)


def build_improved_weights(
    signal: pd.DataFrame,
    liquidity_rank: pd.DataFrame,
    volatility: pd.DataFrame,
    market_returns: pd.Series,
    gross: float = 1.0,
    min_liquidity_rank: float = 0.30,
    trade_threshold: float = 0.0060,
    max_daily_turnover: float = 0.16,
) -> pd.DataFrame:
    sig = apply_liquidity_filter(signal, liquidity_rank, min_rank=min_liquidity_rank)
    weights = build_long_short_weights(sig, long_quantile=0.82, short_quantile=0.18, gross=gross)
    weights = apply_volatility_scaled_sizing(weights, volatility=volatility, gross=gross)
    weights = apply_trade_threshold(weights, threshold=trade_threshold)
    weights = apply_turnover_cap(weights, max_daily_turnover=max_daily_turnover)
    weights = downside_risk_filter(weights, market_returns=market_returns)
    return weights.fillna(0.0)
