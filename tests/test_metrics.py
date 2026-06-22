import numpy as np
import pandas as pd

from equity_alpha.metrics import max_drawdown, sharpe_ratio
from equity_alpha.portfolio import build_long_short_weights


def test_max_drawdown_simple():
    r = pd.Series([0.10, -0.10, -0.10, 0.05])
    dd = max_drawdown(r)
    assert np.isclose(dd, (0.9 * 0.9) - 1.0)


def test_long_short_weights_gross_one():
    idx = pd.date_range("2024-01-01", periods=3)
    sig = pd.DataFrame(
        [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1, 1, 1, 1, 1]],
        index=idx,
        columns=list("ABCDE"),
    )
    w = build_long_short_weights(sig, gross=1.0)
    assert (w.abs().sum(axis=1) <= 1.000001).all()
    assert (w.sum(axis=1).abs() < 1e-9).all()


def test_sharpe_finite():
    r = pd.Series([0.001, 0.002, -0.001, 0.0005] * 50)
    assert np.isfinite(sharpe_ratio(r))
