import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Callable, Tuple
from datetime import date as _date, datetime as _dt, time as _time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from questrade.data import candles_to_df, NY_TZ
from technicals import ema as _ema
from plot_format import PlotFormatter

# Fixed color palette per sector (extend as needed)
SECTOR_COLORS: Dict[str, str] = {
    "energy": "#e45756",
    "materials": "#f58518",
    "industrials": "#54a24b",
    "consumer discretionary": "#4c78a8",
    "consumer staples": "#72b7b2",
    "health care": "#b279a2",
    "financials": "#f2cf5b",
    "information technology": "#ff9da7",
    "communication services": "#9d755d",
    "utilities": "#bab0ab",
    "real estate": "#59a14f",
}

@dataclass
class SymbolDir:
    """Lightweight object representing one symbol directory under data/symbol."""
    name: str
    path: Path
    metadata: Optional[Dict[str, Any]]
    candle_files: List[Path]
    company: Optional[Dict[str, Any]] = None

@dataclass
class SymbolPlotter:
    """
    Temporarily store plottable data keyed by symbol, with optional lazy loading
    from `data/symbol/<SYMBOL>/candles_15m_*.jsonl` on first use.
    """

    symbols: set[str]
    root: Path
    _dirs: Dict[str, SymbolDir]
    _cache: Dict[str, pd.DataFrame]
    _companies: Optional[pd.DataFrame] = None
    _extra_metrics: Optional[set[str]] = None
    _extra_cache: Dict[tuple[str, str], pd.Series] | None = None

    @classmethod
    def from_scan(cls, root: str | Path = Path("data") / "symbol") -> "SymbolPlotter":
        objs = load_symbol_objects(root)
        # Attach company info if available
        companies = _read_current_companies()
        mapping: Dict[str, SymbolDir] = {}
        for o in objs:
            key = o.name.upper()
            if companies is not None and key in companies.index:
                # Shallow copy with company info
                o.company = companies.loc[key].to_dict()
            mapping[key] = o
        extra = _discover_extra_metrics(Path(root))
        return cls(symbols=set(mapping.keys()), root=Path(root), _dirs=mapping, _cache={}, _companies=companies, _extra_metrics=extra, _extra_cache={})

    @classmethod
    def from_symbol_names(cls, names: Iterable[str], root: str | Path = Path("data") / "symbol") -> "SymbolPlotter":
        symbols = {s.upper() for s in names}
        # Best-effort directory map (only for names that exist under root)
        companies = _read_current_companies()
        dirs: Dict[str, SymbolDir] = {}
        for o in load_symbol_objects(root):
            key = o.name.upper()
            if key not in symbols:
                continue
            if companies is not None and key in companies.index:
                o.company = companies.loc[key].to_dict()
            dirs[key] = o
        extra = _discover_extra_metrics(Path(root))
        return cls(symbols=symbols, root=Path(root), _dirs=dirs, _cache={}, _companies=companies, _extra_metrics=extra, _extra_cache={})

    def ingest(self, symbol: str, df: pd.DataFrame) -> None:
        """Provide prebuilt data for a symbol (overrides lazy load)."""
        sym = symbol.upper()
        self.symbols.add(sym)
        self._cache[sym] = df

    def known(self) -> set[str]:
        return set(self.symbols)

    def clear(self) -> None:
        self._cache.clear()

    def _load_df_if_needed(self, symbol: str) -> Optional[pd.DataFrame]:
        sym = symbol.upper()
        if sym in self._cache:
            return self._cache[sym]
        info = self._dirs.get(sym)
        if not info:
            return None
        rows: List[Dict[str, Any]] = []
        for path in info.candle_files:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            rows.append(json.loads(line))
                        except Exception:
                            continue
            except Exception:
                continue
        if not rows:
            return None
        df = candles_to_df(rows)
        self._cache[sym] = df
        return df

    def _load_extra_series(self, symbol: str, metric: str) -> Optional[pd.Series]:
        key = (symbol.upper(), metric)
        if self._extra_cache is not None and key in self._extra_cache:
            return self._extra_cache[key]
        info = self._dirs.get(symbol.upper())
        if not info:
            return None
        # Expect file: metric_<metric>.jsonl
        target = None
        for p in (info.path).glob(f"metric_{metric}.jsonl"):
            target = p
            break
        if target is None or not target.exists():
            return None
        records: List[Dict[str, Any]] = []
        try:
            with target.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            return None
        if not records:
            return None
        # Heuristics: find a time key and a numeric value key
        time_keys = ("time", "timestamp", "start", "date", "datetime")
        tkey = None
        for k in time_keys:
            if k in records[0]:
                tkey = k
                break
        if tkey is None:
            return None
        # Prefer explicit 'value' else first numeric column not time-like
        vkey = "value" if "value" in records[0] else None
        if vkey is None:
            for k, v in records[0].items():
                if k == tkey:
                    continue
                if isinstance(v, (int, float)):
                    vkey = k
                    break
        if vkey is None:
            return None
        df = pd.DataFrame(records)
        try:
            idx = pd.to_datetime(df[tkey], utc=True).dt.tz_convert(NY_TZ)
        except Exception:
            try:
                idx = pd.to_datetime(df[tkey]).tz_localize(NY_TZ)
            except Exception:
                return None
        ser = pd.Series(pd.to_numeric(df[vkey], errors="coerce"), index=idx, name=metric).dropna()
        ser = ser.sort_index()
        if self._extra_cache is not None:
            self._extra_cache[key] = ser
        return ser

    def _passes_company_filters(self, sym: str, filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        info = self._dirs.get(sym)
        if not info or info.company is None:
            return False
        row = info.company
        for key, expected in filters.items():
            actual = _get_company_value(row, key)
            if isinstance(expected, (list, set, tuple)) and not isinstance(expected, tuple):
                # Membership (case-insensitive for strings)
                if isinstance(actual, str):
                    opts = {str(x).strip().lower() for x in expected}
                    if actual.strip().lower() not in opts:
                        return False
                else:
                    if actual not in expected:
                        return False
                continue
            if isinstance(expected, tuple) and len(expected) == 2:
                # Numeric range
                lo = _coerce_number(expected[0])
                hi = _coerce_number(expected[1])
                val = _coerce_number(actual)
                if val is None:
                    return False
                if lo is not None and val < lo:
                    return False
                if hi is not None and val > hi:
                    return False
                continue
            if callable(expected):
                try:
                    if not bool(expected(row)):
                        return False
                except Exception:
                    return False
                continue
            # Fallback to case-insensitive equality for strings
            if isinstance(actual, str) and isinstance(expected, str):
                if actual.strip().lower() != expected.strip().lower():
                    return False
            else:
                if actual != expected:
                    return False
        return True

    def _infer_latest_day(self, symbols: Iterable[str]) -> Optional[_date]:
        from datetime import date as _date_t
        latest: Optional[_date_t] = None
        for sym in symbols:
            info = self._dirs.get(sym)
            if not info:
                continue
            for p in info.candle_files:
                name = p.stem  # candles_15m_YYYY-MM-DD
                try:
                    day_str = name.split("candles_15m_")[1]
                    y, m, d = map(int, day_str.split("-"))
                    cur = _date_t(y, m, d)
                    if latest is None or cur > latest:
                        latest = cur
                except Exception:
                    continue
        return latest

    def _day_bounds(self, day: _date) -> tuple[_dt, _dt]:
        # Market hours: 04:00–20:00 NY time
        start = _dt.combine(day, _time(4, 0)).replace(tzinfo=NY_TZ)
        end = _dt.combine(day, _time(20, 0)).replace(tzinfo=NY_TZ)
        return start, end

    def plot_matching(
        self,
        requested: Iterable[str],
        y: str = "close",
        ax: Optional[plt.Axes] = None,
        company_filters: Optional[Dict[str, Any]] = None,
        *,
        metric: Optional[str] = None,
        day: Optional[_date] = None,
        normalize: bool | str = False,
    ) -> plt.Axes:
        # Back-compat: if metric not provided, use y
        metric = metric or y or "close"
        want = {s.upper() for s in requested}
        matches = sorted(s for s in self.symbols.intersection(want) if self._passes_company_filters(s, company_filters))
        # Default to latest available day across matches
        if day is None:
            day = self._infer_latest_day(matches)
        bounds = self._day_bounds(day) if day is not None else None
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
        shown: List[str] = []
        # Determine if normalization is requested
        do_percent = bool(normalize) and str(normalize).lower() in ("true", "1", "yes", "percent", "pct")
        for sym in matches:
            df = self._cache.get(sym) or self._load_df_if_needed(sym)
            series: Optional[pd.Series] = None
            if df is not None and metric in df.columns and not df.empty:
                sub = df
                if bounds is not None:
                    start, end = bounds
                    sub = sub.loc[(sub.index >= start) & (sub.index <= end)]
                if not sub.empty:
                    series = sub[metric]
            elif self._extra_metrics and metric in self._extra_metrics:
                series = self._load_extra_series(sym, metric)
                if series is not None and bounds is not None:
                    start, end = bounds
                    series = series.loc[(series.index >= start) & (series.index <= end)]
                    if series.empty:
                        series = None
            if series is None or series.empty:
                continue
            if do_percent:
                base = series.iloc[0]
                series = (series / base) - 1.0
            # Color by sector if available
            info = self._dirs.get(sym)
            color = None
            if info and info.company is not None:
                sector_val = _get_company_value(info.company, "Sector")
                if isinstance(sector_val, str):
                    color = SECTOR_COLORS.get(sector_val.strip().lower())
            ax.plot(series.index, series.values, label=sym, linewidth=1.2, color=color)
            shown.append(sym)
        title_metric = f"% {metric}" if do_percent else metric
        day_str = day.isoformat() if isinstance(day, _date) else "all"
        ax.set_title(f"{title_metric} for: {', '.join(shown) if shown else 'no matches'} ({day_str})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Percent change" if do_percent else metric)
        #if do_percent:
        #    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        if shown:
            ax.legend(loc="best", frameon=False)
        ax.grid(True, alpha=0.2)
        return ax

    def list_metrics(self) -> List[str]:
        base = ["open", "high", "low", "close", "volume"]
        extra = sorted(self._extra_metrics) if self._extra_metrics else []
        return base + extra

    # -------------------------
    # Attribute-driven plotting
    # -------------------------
    def displayCurrentPlot(self, plot_attributes: List[str], formatter: Optional[PlotFormatter] = None) -> None:
        """
        High-level entry point that takes a list of attribute strings describing
        selection, filtering, transforms and output options, then displays the plot.

        Example attributes:
          - "sector=Energy minerals"
          - "exchange=NYSE,NASDAQ"
          - "market_cap>1e9"
          - "metric=close" or "metric=ema20"
          - "days=3"
          - "normalize=percent"
          - "volume>2000000"
          - "symbols=AA,AMAT" (optional explicit subset)
        """
        cfg = self._parse_attributes(plot_attributes)

        # Resolve symbol set by company filters and optional explicit symbols
        candidate = set(self.known())
        if cfg["symbols"]:
            candidate &= {s.upper() for s in cfg["symbols"]}
        matches = sorted(s for s in candidate if self._passes_company_filters(s, cfg["company_filters"]))

        # Prepare axes/formatter
        if formatter is None:
            formatter = PlotFormatter()
        fig, ax = plt.subplots(figsize=(12, 6))

        shown: List[str] = []
        do_percent = cfg["normalize"]
        metric = cfg["metric"]
        label_metric = metric

        for sym in matches:
            series = self._resolve_series(sym, metric)
            if series is None or series.empty:
                continue
            # slice by last N days
            series = self._slice_last_n_days(series, cfg["days"]) if cfg["days"] else series
            if series is None or series.empty:
                continue

            # series-level filters (e.g., volume>.. on metric=='volume' or ema)
            if cfg["series_filters"]:
                if not self._series_filters_pass(sym, series, cfg["series_filters"]):
                    continue

            plot_vals = series
            if do_percent:
                base = plot_vals.iloc[0]
                if base == 0:
                    nz = plot_vals[plot_vals != 0]
                    if nz.empty:
                        continue
                    base = nz.iloc[0]
                plot_vals = (plot_vals / base) - 1.0

            # color by sector
            color = None
            info = self._dirs.get(sym)
            if info and info.company is not None:
                sector_val = _get_company_value(info.company, "Sector")
                if isinstance(sector_val, str):
                    color = SECTOR_COLORS.get(sector_val.strip().lower())

            ax.plot(plot_vals.index, plot_vals.values, label=sym, linewidth=formatter.linewidth, alpha=formatter.alpha, color=color)
            shown.append(sym)

        title_metric = (f"% {label_metric}" if do_percent else label_metric)
        title_days = f"last {cfg['days']} day(s)" if cfg["days"] else "all"
        formatter.apply_axes_basics(
            ax,
            title=f"{title_metric} for: {', '.join(shown) if shown else 'no matches'} ({title_days})",
            xlabel="Time",
            ylabel=("Percent change" if do_percent else label_metric),
            percent_y=do_percent,
        )
        plt.show()

    # ---- internals for attribute pipeline ----
    def _slice_last_n_days(self, series: pd.Series, n: int) -> pd.Series:
        if n <= 0:
            return series
        days = pd.Index(series.index.date)
        uniq = pd.unique(days)
        if len(uniq) == 0:
            return series.iloc[0:0]
        last_n = list(pd.Series(uniq).sort_values().iloc[-n:])
        mask = days.isin(last_n)
        return series.loc[mask]

    def _resolve_series(self, symbol: str, metric: str) -> Optional[pd.Series]:
        """Return a single timeseries for the requested metric.
        Supports built-ins (open/high/low/close/volume), discovered extras, and emaNN.
        """
        sym = symbol.upper()
        # EMA pattern: ema20, ema50, etc.
        if metric.lower().startswith("ema"):
            try:
                span = int(metric[3:])
            except Exception:
                return None
            df = self._cache.get(sym) or self._load_df_if_needed(sym)
            if df is None or df.empty or "close" not in df.columns:
                return None
            return _ema(df["close"], span)

        # Built-ins from candles
        if metric in {"open", "high", "low", "close", "volume"}:
            df = self._cache.get(sym) or self._load_df_if_needed(sym)
            if df is None or df.empty or metric not in df.columns:
                return None
            return df[metric]

        # Discovered extras
        if self._extra_metrics and metric in self._extra_metrics:
            ser = self._load_extra_series(sym, metric)
            return ser
        return None

    def _parse_attributes(self, attrs: List[str]) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "company_filters": {},
            "series_filters": [],  # list of (key, op, value)
            "metric": "close",
            "normalize": False,
            "days": 1,
            "symbols": [],
        }

        def _add_company_range(key: str, lo: Optional[float], hi: Optional[float]) -> None:
            # Combine with any existing tuple range
            existing = cfg["company_filters"].get(key)
            if isinstance(existing, tuple) and len(existing) == 2:
                lo = max(existing[0], lo) if existing[0] is not None and lo is not None else (existing[0] if existing and existing[0] is not None else lo)
                hi = min(existing[1], hi) if existing[1] is not None and hi is not None else (existing[1] if existing and existing[1] is not None else hi)
            cfg["company_filters"][key] = (lo, hi)

        for raw in attrs:
            if not raw or not isinstance(raw, str):
                continue
            s = raw.strip()
            # Split on first = or comparison operator
            for op in (">=", "<=", ">", "<", "="):
                if op in s:
                    left, right = s.split(op, 1)
                    key = left.strip()
                    val = right.strip()
                    # known config keys
                    lk = key.lower()
                    if lk == "metric":
                        cfg["metric"] = val
                        break
                    if lk in ("normalize", "compare"):
                        cfg["normalize"] = val.lower() in ("1", "true", "yes", "percent", "pct")
                        break
                    if lk == "days":
                        try:
                            cfg["days"] = max(1, int(float(val)))
                        except Exception:
                            cfg["days"] = 1
                        break
                    if lk == "symbols":
                        cfg["symbols"] = [x.strip().upper() for x in val.split(",") if x.strip()]
                        break

                    # Determine if this is a company or series filter later.
                    # For now, stage all numeric comparisons into both buckets; we’ll resolve during evaluation.
                    if op == "=":
                        # equality: could be categorical (company) or exact series selector; treat as company equality first
                        # support comma for membership
                        if "," in val:
                            vals = {v.strip() for v in val.split(",") if v.strip()}
                            cfg["company_filters"][key] = vals
                        else:
                            cfg["company_filters"][key] = val
                    elif op in (">=", "<=", ">", "<"):
                        num = _coerce_number(val)
                        if num is None:
                            # leave as series filter that will likely fail, but don't crash
                            cfg["series_filters"].append((key, op, val))
                        else:
                            # Assume numeric comparisons likely target company numeric fields (e.g., market cap, revenue)
                            if op == ">":
                                _add_company_range(key, num, None)
                            elif op == ">=":
                                _add_company_range(key, num, None)
                            elif op == "<":
                                _add_company_range(key, None, num)
                            elif op == "<=":
                                _add_company_range(key, None, num)
                            # Also keep as series filter in case it references a time-series like volume
                            cfg["series_filters"].append((key, op, num))
                    break
            else:
                # No operator; ignore
                continue

        return cfg

    def _series_filters_pass(self, symbol: str, series: pd.Series, filters: List[Tuple[str, str, Any]]) -> bool:
        """Evaluate series-level filters like volume>2e6 on the chosen metric.
        Currently uses 'any-bar passes' semantics within the sliced window.
        If the filter refers to a different metric name than current, try to resolve it.
        """
        sym = symbol.upper()
        for key, op, value in filters:
            target: Optional[pd.Series]
            if key.lower() == series.name.lower():
                target = series
            elif key.lower().startswith("ema"):
                try:
                    span = int(key[3:])
                except Exception:
                    continue
                base = self._resolve_series(sym, "close")
                target = _ema(base, span) if base is not None else None
            else:
                # Try built-in or extra metric
                target = self._resolve_series(sym, key)
            if target is None or target.empty:
                continue
            # Align to series window
            if target is not series:
                target = target.reindex(series.index).dropna()
                if target.empty:
                    continue
            if isinstance(value, (int, float)):
                if op == ">":
                    ok = bool((target > value).any())
                elif op == ">=":
                    ok = bool((target >= value).any())
                elif op == "<":
                    ok = bool((target < value).any())
                elif op == "<=":
                    ok = bool((target <= value).any())
                else:
                    ok = bool((target == value).any())
            else:
                ok = bool((target.astype(str) == str(value)).any())
            if not ok:
                return False
        return True


def _read_current_companies(account_dir: Path | str = Path("data") / "account") -> Optional[pd.DataFrame]:
    """
    Load current companies from CSV and index by uppercased ticker.

    Supports either `current_companies.csv` (expected) or a misspelled
    `current_companiee.csv` if present.
    """
    base = Path(account_dir)
    candidates = [base / "current_companies.csv", base / "current_companiee.csv"]
    path = next((p for p in candidates if p.exists()), None)
    if not path:
        return None
    try:
        df = pd.read_csv(path, dtype=str, engine="python")
    except Exception:
        return None
    if df is None or df.empty:
        return None
    # Normalize columns and ticker index
    df.columns = [str(c).strip() for c in df.columns]
    if "Ticker" not in df.columns:
        # Assume first column is ticker if header is missing/unknown
        first_col = df.columns[0]
        if first_col != "Ticker":
            df.rename(columns={first_col: "Ticker"}, inplace=True)
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df["Ticker_norm"] = df["Ticker"].str.upper()
    df = df.set_index("Ticker_norm", drop=False)
    return df

def _coerce_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            v = value.strip().replace(",", "").replace("$", "")
            if v == "":
                return None
            return float(v)
        return float(value)
    except (ValueError, TypeError):
        return None

def _get_company_value(row: Dict[str, Any], key: str) -> Any:
    lk = key.lower()
    for k, v in row.items():
        if str(k).lower() == lk:
            return v
    return None


    
def _discover_extra_metrics(root: Path) -> set[str]:
    names: set[str] = set()
    base = Path(root)
    if not base.exists():
        return names
    for sym_dir in base.iterdir():
        if not sym_dir.is_dir():
            continue
        for p in sym_dir.glob("metric_*.jsonl"):
            # metric_<name>.jsonl -> <name>
            stem = p.stem
            if stem.startswith("metric_"):
                names.add(stem[len("metric_"):])
    return names

def load_symbol_objects(root: str | Path = Path("data") / "symbol") -> List[SymbolDir]:
    base = Path(root)
    if not base.exists() or not base.is_dir():
        return []

    out: List[SymbolDir] = []
    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        meta_path = entry / "metadata.json"
        metadata: Optional[Dict[str, Any]] = None
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = None
        candle_files = sorted(entry.glob("candles_15m_*.jsonl"))
        out.append(SymbolDir(name=name, path=entry.resolve(), metadata=metadata, candle_files=candle_files))

    # Sort for stable output: alphabetical by symbol name
    out.sort(key=lambda s: s.name)
    return out



__all__ = [
    "SymbolDir",
    "load_symbol_objects",
    "SymbolPlotter",
]
