from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponentially weighted moving average."""
    span = int(span)
    if span <= 0:
        raise ValueError("EMA span must be > 0")
    return series.ewm(span=span, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD components as a DataFrame with columns: macd, signal, hist."""
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    out = pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})
    return out

