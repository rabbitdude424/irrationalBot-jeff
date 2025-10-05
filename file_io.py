from __future__ import annotations

import json
import logging
import os
import re
from datetime import date
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from questrade.data import candles_to_df, coerce_date

log = logging.getLogger(__name__)

# Default root for data artifacts
DATA_ROOT = Path("data")
_SYMBOL_SAFE_RE = re.compile(r"[^A-Z0-9._-]")

def ensure_data_dirs(base_dir: Path | str = DATA_ROOT) -> Dict[str, Path]:
    base = Path(base_dir)
    account = base / "account"
    symbol = base / "symbol"
    market = base / "market"

    account.mkdir(parents=True, exist_ok=True)
    symbol.mkdir(parents=True, exist_ok=True)
    market.mkdir(parents=True, exist_ok=True)

    log.debug("Data directories ensured: base=%s account=%s symbol=%s market=%s", base, account, symbol, market)
    return {
        "base": base.resolve(),
        "account": account.resolve(),
        "symbol": symbol.resolve(),
        "market": market.resolve(),
    }

def normalize_symbol(symbol: str) -> str:
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")
    s = symbol.strip().upper()
    s = _SYMBOL_SAFE_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")  # collapse repeats and trim edges
    if not s:
        raise ValueError("symbol is empty after normalization")
    return s

def ensure_symbol_dir(symbol: str, base_dir: Path | str = DATA_ROOT) -> Path:
    paths = ensure_data_dirs(base_dir)
    sym = normalize_symbol(symbol)
    sym_dir = paths["symbol"] / sym
    sym_dir.mkdir(parents=True, exist_ok=True)
    log.debug("Symbol directory ensured: %s (from input '%s')", sym_dir, symbol)
    return sym_dir.resolve()

def ensure_temp_dir(base_dir: Path | str = DATA_ROOT) -> Path:
    """Ensure <base_dir>/temp exists and return its resolved path."""
    base = Path(base_dir)
    temp = base / "temp"
    temp.mkdir(parents=True, exist_ok=True)
    return temp.resolve()

def _stage_raw_day_candles(symbol: str, day: date | str, rows: List[Dict[str, Any]], base_dir: Path | str = DATA_ROOT) -> Path:
    """Persist raw API output to a temp staging file for debugging."""
    temp_dir = ensure_temp_dir(base_dir) / normalize_symbol(symbol)
    temp_dir.mkdir(parents=True, exist_ok=True)
    day_str = coerce_date(day).isoformat()
    stage_path = temp_dir / f"candles_15m_{day_str}_{int(time.time()*1_000_000)}.json"
    stage_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return stage_path

def persist_day_candles(symbol: str, day: date | str, rows: List[Dict[str, Any]], base_dir: Path | str = DATA_ROOT) -> Path:
    """Write validated candle rows to their canonical location."""
    return save_day_candles_jsonl(symbol, day, rows, base_dir=base_dir)

def ensure_account_dir(base_dir: Path | str = DATA_ROOT) -> Path:
    """Ensure <base_dir>/account exists and return its resolved path."""
    paths = ensure_data_dirs(base_dir)
    return paths["account"]

def api_usage_path(base_dir: Path | str = DATA_ROOT) -> Path:
    """Return the JSONL path used for persistent API usage telemetry."""
    account_dir = ensure_account_dir(base_dir)
    return account_dir / "api_usage.jsonl"

def candle_error_path(base_dir: Path | str = DATA_ROOT) -> Path:
    """Return the log path used for candle gap fallbacks."""
    return ensure_account_dir(base_dir) / "candle_errors.jsonl"
def _atomic_write_lines(path: Path, lines: List[str]) -> None:
    """Atomically write pre-formatted JSONL lines to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)

def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically write a JSON payload to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)

def candles_day_path(symbol: str, day: date | str, base_dir: Path | str = DATA_ROOT) -> Path:
    """Return canonical JSONL path for a symbol/day 15m candle file."""
    sym_dir = ensure_symbol_dir(symbol, base_dir)
    day_str = coerce_date(day).isoformat()
    return sym_dir / f"candles_15m_{day_str}.jsonl"

def save_day_candles_jsonl(symbol: str,
                            day: date | str,
                            rows: List[Dict[str, Any]],
                            base_dir: Path | str = DATA_ROOT,
                            overwrite: bool = True) -> Path:
    """Persist raw candle rows as JSONL (one object per line)."""
    path = candles_day_path(symbol, day, base_dir)
    payload = [json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n" for r in rows]
    if overwrite or not path.exists():
        _atomic_write_lines(path, payload)
    elif payload:
        with path.open("a", encoding="utf-8") as handle:
            for line in payload:
                handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
    return path

def load_day_candles_df(symbol: str,
                        day: date | str,
                        base_dir: Path | str = DATA_ROOT) -> pd.DataFrame:
    """Read a day's JSONL candles file and convert to a tidy DataFrame."""
    path = candles_day_path(symbol, day, base_dir)
    if not path.exists():
        raise FileNotFoundError(str(path))
    raw = pd.read_json(path, lines=True)
    return candles_to_df(raw.to_dict("records"))

def ensure_market_dir(base_dir: Path | str = DATA_ROOT) -> Path:
    """Ensure <base_dir>/market exists and return its resolved path."""
    paths = ensure_data_dirs(base_dir)
    return paths["market"]

def market_days_path(base_dir: Path | str = DATA_ROOT) -> Path:
    """Canonical JSONL path for historic open days."""
    mdir = ensure_market_dir(base_dir)
    return mdir / "historic_days_open.jsonl"

def read_symbol_metadata(symbol: str, base_dir: Path | str = DATA_ROOT) -> Optional[Dict[str, Any]]:
    """Load cached symbol metadata if present; return ``None`` when missing."""
    sym_dir = ensure_symbol_dir(symbol, base_dir)
    meta_path = sym_dir / "metadata.json"
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def write_symbol_metadata(symbol: str, metadata: Dict[str, Any], base_dir: Path | str = DATA_ROOT) -> Path:
    """Persist symbol metadata under the symbol directory and return the path."""
    sym_dir = ensure_symbol_dir(symbol, base_dir)
    meta_path = sym_dir / "metadata.json"
    _atomic_write_json(meta_path, metadata)
    return meta_path

def append_jsonl_record(path: Path, record: Dict[str, Any]) -> None:
    """Append a single JSON line with durability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())

def append_candle_error(record: Dict[str, Any], base_dir: Path | str = DATA_ROOT) -> None:
    """Append candle gap diagnostics (temporary workaround)."""
    append_jsonl_record(candle_error_path(base_dir), record)

def load_jsonl_key_set(path: Path, key: str) -> set[str]:
    """
    Load a JSONL file and return a set of unique values for `key`.
    Safe to call when file doesn't exist (returns empty set).
    """
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                obj = json.loads(line)
                v = obj.get(key)
                if isinstance(v, str):
                    out.add(v)
            except Exception:
                continue
    return out

__all__ = [
    "ensure_data_dirs",
    "ensure_symbol_dir",
    "ensure_temp_dir",
    "ensure_market_dir",
    "normalize_symbol",
    "candles_day_path",
    "save_day_candles_jsonl",
    "persist_day_candles",
    "load_day_candles_df",
    "market_days_path",
    "candle_error_path",
    "ensure_account_dir",
    "api_usage_path",
    "read_symbol_metadata",
    "write_symbol_metadata",
    "append_candle_error",
    "append_jsonl_record",
    "load_jsonl_key_set",
    "DATA_ROOT",
]

