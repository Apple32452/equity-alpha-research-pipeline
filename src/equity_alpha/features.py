from __future__ import annotations

import numpy as np
import pandas as pd


def _wide(panel: pd.DataFrame, column: str) -> pd.DataFrame:
    return panel[column].unstack("ticker").sort_index()


def cross_sectional_rank(df: pd.DataFrame, center: bool = True) -> pd.DataFrame:
    """Percentile rank each date; optionally center to roughly [-0.5, 0.5]."""
    ranked = df.rank(axis=1, pct=True, method="average")
    return ranked - 0.5 if center else ranked


def robust_zscore(df: pd.DataFrame, clip: float = 5.0) -> pd.DataFrame:
    med = df.median(axis=1)
    mad = (df.sub(med, axis=0).abs()).median(axis=1).replace(0, np.nan)
    z = df.sub(med, axis=0).div(1.4826 * mad, axis=0)
    return z.clip(-clip, clip)


def build_features(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute cross-sectional features from daily OHLCV data.

    All features are known at close t and are intended to forecast return t+1.
    """
    adj = _wide(panel, "adj_close")
    vol = _wide(panel, "volume")
    close = _wide(panel, "close")
    ret_1d = adj.pct_change(fill_method=None)

    dollar_volume = (close * vol).replace(0, np.nan)
    adv20 = dollar_volume.rolling(20, min_periods=15).mean()

    features = {
        "ret_1d": ret_1d,
        "next_ret_1d": ret_1d.shift(-1),
        "momentum_21": adj.pct_change(21, fill_method=None),
        "momentum_63": adj.pct_change(63, fill_method=None),
        "reversal_5": -adj.pct_change(5, fill_method=None),
        "volatility_21": ret_1d.rolling(21, min_periods=15).std(),
        "volatility_63": ret_1d.rolling(63, min_periods=40).std(),
        "adv20": adv20,
        "liquidity_rank": cross_sectional_rank(np.log1p(adv20), center=False),
    }

    # Cross-sectional transformations used by signal layer.
    features["rank_mom21"] = cross_sectional_rank(features["momentum_21"])
    features["rank_mom63"] = cross_sectional_rank(features["momentum_63"])
    features["rank_reversal5"] = cross_sectional_rank(features["reversal_5"])
    # Lower volatility should get a higher rank for the risk-adjusted alpha.
    features["rank_low_vol21"] = cross_sectional_rank(-features["volatility_21"])
    features["rank_low_vol63"] = cross_sectional_rank(-features["volatility_63"])
    features["log_adv20_z"] = robust_zscore(np.log1p(adv20))
    return features


def feature_frame(features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a long, tidy feature frame useful for diagnostics."""
    parts = []
    for name, df in features.items():
        s = df.stack(dropna=False).rename(name)
        parts.append(s)
    out = pd.concat(parts, axis=1)
    out.index.names = ["date", "ticker"]
    return out.sort_index()
