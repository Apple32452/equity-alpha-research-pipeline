from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


OHLCV_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


@dataclass(frozen=True)
class DataConfig:
    start: str = "2023-01-01"
    end: str | None = None
    min_history: int = 252
    seed: int = 7


def load_universe(path: str | Path, max_tickers: int | None = None) -> list[str]:
    """Load a ticker universe from a CSV file with a `ticker` column or one ticker per line."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")
    try:
        df = pd.read_csv(path)
        if "ticker" in df.columns:
            tickers = df["ticker"].astype(str).str.strip().str.upper().tolist()
        else:
            tickers = df.iloc[:, 0].astype(str).str.strip().str.upper().tolist()
    except Exception:
        tickers = [x.strip().upper() for x in path.read_text().splitlines() if x.strip()]
    tickers = [t for t in tickers if t and t != "TICKER"]
    # Preserve order while removing duplicates.
    tickers = list(dict.fromkeys(tickers))
    return tickers[:max_tickers] if max_tickers else tickers


def download_yfinance_panel(
    tickers: Iterable[str],
    start: str,
    end: str | None = None,
    min_history: int = 252,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """Download daily OHLCV data with yfinance.

    Returns a long panel indexed by date/ticker with columns:
    open, high, low, close, adj_close, volume.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance or run with --data-source synthetic.") from exc

    tickers = list(tickers)
    if not tickers:
        raise ValueError("Ticker universe is empty.")

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        group_by="column",
        actions=False,
        threads=True,
        progress=False,
    )
    if raw.empty:
        raise ValueError("No data downloaded. Check dates, tickers, and internet connection.")

    # yfinance returns either single-index columns for one ticker or MultiIndex for multiple tickers.
    if not isinstance(raw.columns, pd.MultiIndex):
        t = tickers[0]
        raw.columns = pd.MultiIndex.from_product([raw.columns, [t]])

    mapping = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    frames: list[pd.DataFrame] = []
    for field, out_name in mapping.items():
        if field in raw.columns.get_level_values(0):
            part = raw[field].copy()
            part.columns.name = "ticker"
            frames.append(part.stack(future_stack=True).rename(out_name))
    panel = pd.concat(frames, axis=1)
    panel.index.names = ["date", "ticker"]
    panel = panel.sort_index()
    panel = panel.dropna(subset=["close", "volume"], how="any")
    if "adj_close" not in panel.columns:
        panel["adj_close"] = panel["close"]
    panel = panel[OHLCV_COLUMNS]

    counts = panel.groupby(level="ticker").size()
    keep = counts[counts >= min_history].index
    panel = panel.loc[pd.IndexSlice[:, keep], :]
    if panel.empty:
        raise ValueError("No tickers met min_history. Lower --min-history or expand dates.")
    return panel


def generate_synthetic_panel(
    n_tickers: int = 300,
    n_days: int = 756,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a realistic-enough daily equity panel for code testing.

    This is only for pipeline validation. Do not use synthetic performance numbers on a resume.
    The generator creates market regimes, ticker-specific volatility/liquidity, and a mild
    medium-term momentum effect so alpha evaluation has a measurable but not guaranteed edge.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    tickers = [f"STK{i:03d}" for i in range(n_tickers)]

    # Regime process: calm, volatile, stress.
    regimes = rng.choice([0, 1, 2], size=n_days, p=[0.68, 0.22, 0.10])
    market_vol = np.choose(regimes, [0.008, 0.014, 0.024])
    market_drift = np.choose(regimes, [0.00035, 0.00005, -0.00045])
    market = market_drift + market_vol * rng.standard_normal(n_days)

    betas = rng.normal(1.0, 0.25, n_tickers)
    idio_vol = rng.uniform(0.010, 0.030, n_tickers)
    log_prices = np.zeros((n_days, n_tickers))
    log_prices[0] = np.log(rng.uniform(15, 250, n_tickers))
    latent_quality = rng.normal(0, 0.0007, n_tickers)
    low_vol_score = (idio_vol.mean() - idio_vol) / idio_vol.std()
    low_vol_alpha = 0.00025 * low_vol_score
    eps = rng.standard_normal((n_days, n_tickers)) * idio_vol

    # Create returns with medium-term continuation, short-term reversal,
    # and a modest low-volatility premium. This makes the demo useful for
    # testing whether the enhanced signal/risk controls add value.
    rets = np.zeros((n_days, n_tickers))
    for t in range(1, n_days):
        past_21 = rets[max(0, t - 21):t].mean(axis=0) if t > 1 else 0.0
        past_5 = rets[max(0, t - 5):t].mean(axis=0) if t > 1 else 0.0
        continuation = 0.10 * past_21
        reversal = -0.08 * past_5
        rets[t] = betas * market[t] + eps[t] + latent_quality + low_vol_alpha + continuation + reversal
        # Stress regime punishes high beta and high vol names more.
        if regimes[t] == 2:
            rets[t] -= 0.0015 * (betas - betas.mean()) + 0.0018 * (idio_vol - idio_vol.mean()) / idio_vol.std()
        log_prices[t] = log_prices[t - 1] + rets[t]

    close = np.exp(log_prices)
    intraday_noise = rng.normal(0, 0.003, close.shape)
    open_ = close * (1 + intraday_noise)
    high = np.maximum(open_, close) * (1 + rng.uniform(0.000, 0.010, close.shape))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.000, 0.010, close.shape))
    avg_dollar_volume = rng.lognormal(mean=17.0, sigma=0.8, size=n_tickers)
    volume = avg_dollar_volume / np.maximum(close, 1.0) * rng.lognormal(mean=0, sigma=0.25, size=close.shape)

    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    panel = pd.DataFrame(
        {
            "open": open_.ravel(),
            "high": high.ravel(),
            "low": low.ravel(),
            "close": close.ravel(),
            "adj_close": close.ravel(),
            "volume": volume.ravel().astype(int),
        },
        index=idx,
    )
    return panel


def save_panel(panel: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.reset_index().to_parquet(path, index=False)


def load_panel(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["date", "ticker"]).sort_index()
