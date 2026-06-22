from __future__ import annotations

import pandas as pd


def build_baseline_signal(features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Simple medium-term momentum signal used as the baseline."""
    sig = 0.70 * features["rank_mom63"] + 0.30 * features["rank_mom21"]
    return sig.rename_axis(index="date", columns="ticker")


def build_improved_signal(features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Composite alpha signal: momentum + short reversal + low-volatility adjustment.

    This is intentionally simple and defensible for a resume/project conversation:
    - medium-term momentum captures trend persistence,
    - short-term reversal reduces chasing recent jumps,
    - low-volatility rank improves risk-adjusted exposure.
    """
    sig = (
        0.66 * features["rank_mom63"]
        + 0.27 * features["rank_mom21"]
        + 0.05 * features["rank_reversal5"]
        + 0.02 * features["rank_low_vol21"]
    )
    return sig.rename_axis(index="date", columns="ticker")


def zscore_signal(signal: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    mean = signal.mean(axis=1)
    std = signal.std(axis=1).replace(0, pd.NA)
    return signal.sub(mean, axis=0).div(std, axis=0).clip(-clip, clip)
