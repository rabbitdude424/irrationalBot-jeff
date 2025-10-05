from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore

import pandas as pd

NY_TZ = ZoneInfo("America/New_York")

INTERVAL_TO_DELTA: Dict[str, timedelta] = {
    "OneMinute": timedelta(minutes=1),
    "TwoMinutes": timedelta(minutes=2),
    "ThreeMinutes": timedelta(minutes=3),
    "FourMinutes": timedelta(minutes=4),
    "FiveMinutes": timedelta(minutes=5),
    "TenMinutes": timedelta(minutes=10),
    "FifteenMinutes": timedelta(minutes=15),
    "TwentyMinutes": timedelta(minutes=20),
    "HalfHour": timedelta(minutes=30),
    "OneHour": timedelta(hours=1),
    "TwoHours": timedelta(hours=2),
    "FourHours": timedelta(hours=4),
    # Daily+ supported by API, but intraday workflows generally use the ones above.
}

# A conservative lookback window per interval (keeps results < 2,000 bars)
LOOKBACK_BY_INTERVAL: Dict[str, timedelta] = {
    "OneMinute": timedelta(days=2),        # approx 780-1200 bars (with pre/post) < 2000
    "TwoMinutes": timedelta(days=3),
    "ThreeMinutes": timedelta(days=4),
    "FourMinutes": timedelta(days=5),
    "FiveMinutes": timedelta(days=6),
    "TenMinutes": timedelta(days=8),
    "FifteenMinutes": timedelta(days=10),  # approx 260 bars
    "TwentyMinutes": timedelta(days=14),
    "HalfHour": timedelta(days=21),
    "OneHour": timedelta(days=30),
    "TwoHours": timedelta(days=45),
    "FourHours": timedelta(days=90),
}

_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_DATE_COMPACT_RE = re.compile(r"(\d{4})(\d{2})(\d{2})")


@dataclass
class Candle:
    """Structured representation of a single OHLCV candle from the API."""

    start: str
    end: str
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class Quote:
    """Normalized view of a Questrade Level-1 quote payload."""

    symbol: str
    symbol_id: int
    bid: Optional[float]
    bid_size: Optional[int]
    ask: Optional[float]
    ask_size: Optional[int]
    last: Optional[float]
    last_rth: Optional[float]
    last_size: Optional[int]
    last_tick: Optional[str]
    last_time: Optional[str]
    volume: Optional[int]
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    delay: bool
    is_halted: bool


def _to_float(value: Any) -> Optional[float]:
    """Return ``value`` coerced to ``float`` or ``None`` when conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    """Return ``value`` coerced to ``int`` or ``None`` when conversion fails."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def coerce_date(d: date | str) -> date:
    """Normalize supported date inputs into a ``datetime.date`` instance."""
    if isinstance(d, date):
        return d
    match = _DATE_RE.fullmatch(d) or _DATE_COMPACT_RE.fullmatch(d)
    if not match:
        raise ValueError("day must be a datetime.date or 'YYYY-MM-DD' (or 'YYYYMMDD')")
    year, month, day = map(int, match.groups())
    return date(year, month, day)


TRADING_OPEN = time(4, 0)
TRADING_CLOSE = time(20, 0)


def day_bounds_iso(day: date | str, tz: ZoneInfo = NY_TZ) -> Tuple[str, str]:
    """Return ISO8601 bounds covering 04:00 through 20:00 local time."""
    d = coerce_date(day)
    start_dt = datetime.combine(d, TRADING_OPEN).replace(tzinfo=tz)
    # The final 15m candle opens at 19:45 and closes at 20:00 inclusive.
    end_dt = datetime.combine(d, TRADING_CLOSE).replace(tzinfo=tz)
    return (
        start_dt.isoformat(timespec="microseconds"),
        end_dt.isoformat(timespec="microseconds"),
    )


def floor_to_interval(dt: datetime, interval: str) -> datetime:
    """Floor ``dt`` to the nearest completed interval boundary understood by Questrade."""
    if interval not in INTERVAL_TO_DELTA:
        raise ValueError(f"Unsupported interval '{interval}'")
    delta = INTERVAL_TO_DELTA[interval]
    base = dt.replace(second=0, microsecond=0)
    day_start = base.replace(hour=0, minute=0)
    minutes = int((base - day_start).total_seconds() // 60)
    step = int(delta.total_seconds() // 60)
    floored_minutes = (minutes // step) * step
    return day_start + timedelta(minutes=floored_minutes)


def as_candles(rows: List[Dict[str, Any]]) -> List[Candle]:
    """Cast raw candle dictionaries into ``Candle`` instances for stronger typing."""
    return [
        Candle(
            start=row["start"],
            end=row["end"],
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=int(row["volume"]),
        )
        for row in rows
    ]


def parse_quote(row: Dict[str, Any]) -> Quote:
    """Translate a raw quote dictionary into a ``Quote`` dataclass."""
    return Quote(
        symbol=row.get("symbol", ""),
        symbol_id=int(row.get("symbolId", 0)),
        bid=_to_float(row.get("bidPrice")),
        bid_size=_to_int(row.get("bidSize")),
        ask=_to_float(row.get("askPrice")),
        ask_size=_to_int(row.get("askSize")),
        last=_to_float(row.get("lastTradePrice")),
        last_rth=_to_float(row.get("lastTradePriceTrHrs")),
        last_size=_to_int(row.get("lastTradeSize")),
        last_tick=row.get("lastTradeTick") or None,
        last_time=row.get("lastTradeTime"),
        volume=_to_int(row.get("volume")),
        open=_to_float(row.get("openPrice")),
        high=_to_float(row.get("highPrice")),
        low=_to_float(row.get("lowPrice")),
        delay=bool(row.get("delay", 0)),
        is_halted=bool(row.get("isHalted", False)),
    )


def quote_price(quote: Quote, mode: str = "last") -> Optional[float]:
    """Pick a deterministic price from ``quote`` according to ``mode``."""
    mode = mode.lower()

    if mode == "bid":
        return quote.bid
    if mode == "ask":
        return quote.ask
    if mode == "mid":
        if quote.bid is not None and quote.ask is not None:
            return (quote.bid + quote.ask) / 2.0
        mode = "last"

    if mode == "last":
        for price in (
            quote.last,
            quote.last_rth,
            (quote.bid + quote.ask) / 2.0 if (quote.bid is not None and quote.ask is not None) else None,
            quote.bid,
            quote.ask,
        ):
            if price is not None:
                return price
        return None

    raise ValueError("mode must be one of: 'last', 'mid', 'bid', 'ask'")


def candles_to_df(rows: List[Dict[str, Any]], tz: ZoneInfo = NY_TZ) -> pd.DataFrame:
    """Convert candle rows into a tidy, timezone-aware pandas DataFrame."""
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "end"]).set_index(
            pd.DatetimeIndex([], name="start")
        )

    df["start"] = pd.to_datetime(df["start"], utc=True).dt.tz_convert(tz)
    df["end"] = pd.to_datetime(df["end"], utc=True).dt.tz_convert(tz)

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.dropna(subset=["open", "high", "low", "close"])
          .sort_values("start")
          .set_index("start")
    )
    return df


__all__ = [
    "NY_TZ",
    "INTERVAL_TO_DELTA",
    "LOOKBACK_BY_INTERVAL",
    "coerce_date",
    "day_bounds_iso",
    "floor_to_interval",
    "Candle",
    "as_candles",
    "Quote",
    "parse_quote",
    "quote_price",
    "candles_to_df",
]
