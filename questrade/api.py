import time
import json
import logging
import os
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import quote, urljoin

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore

import requests
from file_io import (
    _stage_raw_day_candles,
    append_candle_error,
    append_jsonl_record,
    api_usage_path,
    ensure_data_dirs,
    persist_day_candles,
    read_symbol_metadata,
    write_symbol_metadata,
)

from questrade.data import (
    INTERVAL_TO_DELTA,
    LOOKBACK_BY_INTERVAL,
    NY_TZ,
    TRADING_CLOSE,
    TRADING_OPEN,
    coerce_date,
    day_bounds_iso,
    floor_to_interval,
    parse_quote,
    Quote,
)

DEFAULT_TOKEN_PATH = Path("questrade") / "token.json"
DEFAULT_LOGIN_SERVER = os.getenv("QTRADE_LOGIN_SERVER", "https://login.questrade.com")
REQUEST_TIMEOUT = float(os.getenv("QTRADE_TIMEOUT", "10"))  # seconds
REFRESH_FRACTION = float(os.getenv("QTRADE_REFRESH_FRACTION", "0.15"))
REFRESH_MIN_BUFFER = int(os.getenv("QTRADE_REFRESH_MIN_BUFFER", "45"))
MAX_BACKOFF = int(os.getenv("QTRADE_MAX_BACKOFF", "5"))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("questrade")

class QuestradeConfigError(RuntimeError):
    """Raised when required configuration or token files are missing or invalid."""

class QuestradeAuthError(RuntimeError):
    """Raised when refresh or authentication attempts with Questrade fail."""

class QuestradeHTTPError(RuntimeError):
    """Raised for unexpected HTTP/MIME responses from the Questrade REST API."""

@dataclass
class QTToken:
    """Persisted OAuth token fields required for authenticated Questrade access."""

    access_token: str
    refresh_token: str
    token_type: str
    api_server: str
    active_time: int         # seconds (a.k.a. expires_in)
    issued_at: int           # epoch seconds
    login_server: str        # https://login.questrade.com or practice

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "QTToken":
        """Build a QTToken from the persisted JSON payload."""
        login_server = d.get("login_server") or DEFAULT_LOGIN_SERVER
        issued_at = int(d.get("issued_at", 0))
        active_time = int(d.get("active_time", d.get("expires_in", 0)))
        api_root = (d.get("api_server") or "").rstrip("/")
        if not all([d.get("refresh_token"), d.get("token_type"), api_root]):
            raise QuestradeConfigError("Token file missing required fields.")
        return QTToken(
            access_token=d["access_token"],
            refresh_token=d["refresh_token"],
            token_type=d["token_type"],
            api_server=f"{api_root}/",
            active_time=active_time,
            issued_at=issued_at,
            login_server=login_server,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the token into a dict ready for JSON persistence."""
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
            "api_server": self.api_server,
            "active_time": self.active_time,
            "issued_at": self.issued_at,
            "login_server": self.login_server,
        }

@dataclass
class SymbolHit:
    """Result of resolving a ticker symbol to its underlying Questrade identifier."""
    symbol_id: int
    symbol: str
    listing_exchange: Optional[str] = None
    currency: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the symbol resolution payload as a standard dict."""
        return {
            "symbol_id": self.symbol_id,
            "symbol": self.symbol,
            "listing_exchange": self.listing_exchange,
            "currency": self.currency,
        }

@dataclass
class RateLimitInfo:
    """Normalized view of rate-limiting state (best-effort, header-agnostic)."""
    limit: Optional[int] = None
    remaining: Optional[int] = None
    reset_epoch: Optional[int] = None  # epoch seconds if provided
    retry_after_s: Optional[int] = None
    raw: Dict[str, str] = field(default_factory=dict)

    def seconds_until_reset(self) -> Optional[int]:
        if self.reset_epoch is None:
            return None
        return max(0, self.reset_epoch - int(time.time()))

@dataclass
class APIMetrics:
    """Lightweight, in-memory telemetry for API health/usage."""
    started_at: int = field(default_factory=lambda: int(time.time()))
    total_requests: int = 0
    total_errors: int = 0
    last_status: Optional[int] = None
    last_url: Optional[str] = None
    last_latency_ms: Optional[int] = None
    ema_latency_ms: Optional[float] = None
    token_refreshes: int = 0
    last_token_refresh: Optional[int] = None
    rate_limit: RateLimitInfo = field(default_factory=RateLimitInfo)
    per_route_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def _parse_rate_limit(self, headers: Dict[str, str]) -> RateLimitInfo:
        rl = RateLimitInfo(raw={k: v for k, v in headers.items() if 'ratelimit' in k.lower()})
        # Common patterns (best-effort, non-fatal if absent)
        for k, v in headers.items():
            kl = k.lower()
            if kl.endswith('remaining') and 'ratelimit' in kl:
                try: rl.remaining = int(v)
                except: pass
            elif kl.endswith('limit') and 'ratelimit' in kl:
                try: rl.limit = int(v)
                except: pass
            elif 'reset' in kl and 'ratelimit' in kl:
                try:
                    # Some APIs send seconds-from-now; others epoch seconds.
                    iv = int(v)
                    # Heuristic: treat large numbers as epoch seconds.
                    rl.reset_epoch = iv if iv > 10_000_000 else int(time.time()) + iv
                except: pass
            elif kl == 'retry-after':
                try: rl.retry_after_s = int(v)
                except: pass
        return rl

    def record(self, method: str, url: str, status: int, latency_ms: int, headers: Dict[str, str]) -> None:
        self.total_requests += 1
        self.last_status = status
        self.last_url = url
        self.last_latency_ms = latency_ms
        self.ema_latency_ms = latency_ms if self.ema_latency_ms is None else (0.2*latency_ms + 0.8*self.ema_latency_ms)
        if status >= 400:
            self.total_errors += 1
        route_key = f"{method} {url.split('?', 1)[0]}"
        self.per_route_counts[route_key] += 1
        self.rate_limit = self._parse_rate_limit(headers)

    def snapshot(self, token_ttl_s: Optional[int]) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "uptime_s": int(time.time()) - self.started_at,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "last_status": self.last_status,
            "last_url": self.last_url,
            "last_latency_ms": self.last_latency_ms,
            "ema_latency_ms": None if self.ema_latency_ms is None else round(self.ema_latency_ms, 1),
            "token_refreshes": self.token_refreshes,
            "last_token_refresh": self.last_token_refresh,
            "token_ttl_remaining_s": token_ttl_s,
            "rate_limit": {
                "limit": self.rate_limit.limit,
                "remaining": self.rate_limit.remaining,
                "reset_epoch": self.rate_limit.reset_epoch,
                "retry_after_s": self.rate_limit.retry_after_s,
                "seconds_until_reset": self.rate_limit.seconds_until_reset(),
                "raw": self.rate_limit.raw,
            },
            "per_route_counts": dict(self.per_route_counts),
        }

class QuestradeSession:
    """Authenticated HTTP client for the Questrade REST API."""

    def __init__(self, token_path: Optional[Path] = None):
        """Initialize the session using the token stored at ``token_path``."""
        self.token_path = token_path or DEFAULT_TOKEN_PATH
        if not self.token_path.exists():
            raise QuestradeConfigError(f"Token file not found at: {self.token_path}")
        ensure_data_dirs()
        self.api_usage_path = api_usage_path()
        self.http = requests.Session()
        self.token = self._load_token()
        self.metrics = APIMetrics()

    def close(self) -> None:
        """Close the underlying ``requests.Session``."""
        self.http.close()

    def __enter__(self) -> "QuestradeSession":
        """Enter the context manager returning ``self``."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Ensure resources are released when leaving the context manager."""
        self.close()

    def _load_token(self) -> QTToken:
        """Load and validate the persisted token from disk."""
        try:
            with self.token_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError as exc:
            raise QuestradeConfigError(f"Token file not found at: {self.token_path}") from exc
        except json.JSONDecodeError as exc:
            raise QuestradeConfigError("Token file is not valid JSON.") from exc

        token = QTToken.from_dict(data)
        if token.issued_at <= 0:
            log.info("Token has no issued_at; forcing immediate refresh.")
            token = self._refresh_token(token)
        return token

    def _persist(self, token: QTToken) -> None:
        """Persist ``token`` back to disk, replacing the previous file."""
        _atomic_write_json(self.token_path, token.to_dict())
        
    def _token_ttl_remaining_s(self) -> Optional[int]:
        if self.token.active_time <= 0 or self.token.issued_at <= 0:
            return None
        return max(0, self.token.active_time - (_utc_now_s() - self.token.issued_at))
    
    def get_api_usage(self) -> Dict[str, Any]:
        """Return a point-in-time snapshot of API usage & rate-limit state."""
        return self.metrics.snapshot(self._token_ttl_remaining_s())

    def save_api_usage_json(self, path: Path) -> None:
        """Persist the current usage/limit snapshot to disk."""
        _atomic_write_json(path, self.get_api_usage())

    def _needs_refresh(self, token: QTToken) -> bool:
        """Return ``True`` when the supplied token is close to expiring."""
        if token.active_time <= 0:
            return True
        elapsed = _utc_now_s() - token.issued_at
        buffer_s = max(REFRESH_MIN_BUFFER, int(token.active_time * REFRESH_FRACTION))
        return elapsed >= max(0, token.active_time - buffer_s)

    def _refresh_token(self, token: QTToken) -> QTToken:
        """Exchange the refresh token for a new access token."""
        login_server = (token.login_server or DEFAULT_LOGIN_SERVER).rstrip("/")
        endpoint = f"{login_server}/oauth2/token"
        params = {"grant_type": "refresh_token", "refresh_token": token.refresh_token}
        try:
            resp = self.http.get(endpoint, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            raise QuestradeAuthError(f"Network error during token refresh: {exc}") from exc

        if resp.status_code != 200:
            raise QuestradeAuthError(
                f"Failed to refresh token (HTTP {resp.status_code}): {resp.text}"
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise QuestradeAuthError("Malformed refresh response: not valid JSON.") from exc

        access_token = data.get("access_token")
        refresh_token = data.get("refresh_token") or token.refresh_token
        token_type = data.get("token_type") or token.token_type or "Bearer"
        api_root = (data.get("api_server") or token.api_server).rstrip("/")
        expires_in = int(data.get("expires_in", token.active_time or 0))

        if not access_token or not api_root:
            raise QuestradeAuthError("Malformed refresh response: missing access_token/api_server.")

        new_token = QTToken(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type=token_type,
            api_server=f"{api_root}/",
            active_time=expires_in,
            issued_at=_utc_now_s(),
            login_server=login_server,
        )
        self._persist(new_token)
        if token.issued_at > 0:
            self.metrics.token_refreshes += 1
            self.metrics.last_token_refresh = int(time.time())
        log.info("Token refreshed; api_server=%s TTL=%ss", new_token.api_server, expires_in)
        return new_token

    def _request_once(self, method: str, url: str, **kwargs) -> requests.Response:
        """Issue a single HTTP request, normalizing request exceptions."""
        try:
            return self.http.request(method, url, **kwargs)
        except requests.RequestException as exc:
            raise QuestradeHTTPError(f"HTTP error calling {url}: {exc}") from exc

    def ensure_valid(self) -> None:
        """Refresh the current token if it is nearing expiration."""
        if self._needs_refresh(self.token):
            self.token = self._refresh_token(self.token)

    def auth_headers(self) -> Dict[str, str]:
        """Return the Authorization headers for the active access token."""
        return {"Authorization": f"{self.token.token_type} {self.token.access_token}"}

    def get_token(self) -> Dict[str, Any]:
        """Return the current token as a plain dict (safe for logging/tests)."""
        return self.token.to_dict()

    def request(self, method: str, path: str, *, usage_tags: Optional[Dict[str, Any]] = None, **kwargs) -> requests.Response:
        """Issue an authenticated HTTP request with usage/limit tracking."""
        self.ensure_valid()
        url = urljoin(self.token.api_server, path.lstrip("/"))

        timeout = kwargs.pop("timeout", REQUEST_TIMEOUT)
        custom_headers = kwargs.pop("headers", {})
        request_kwargs = dict(kwargs)
        headers = {**custom_headers, **self.auth_headers()}

        def do_once(hdrs):
            start = time.perf_counter()
            resp = self._request_once(method, url, headers=hdrs, timeout=timeout, **request_kwargs)
            latency_ms = int((time.perf_counter() - start) * 1000)
            try:
                self.metrics.record(method, url, resp.status_code, latency_ms, resp.headers or {})
            except Exception:
                pass
            self._append_api_usage(method, path, usage_tags, resp.status_code, latency_ms)
            return resp

        resp = do_once(headers)
        if resp.status_code == 401:
            log.info("401 received; attempting token refresh and retry once.")
            self.token = self._refresh_token(self.token)
            headers = {**custom_headers, **self.auth_headers()}
            resp = do_once(headers)

        # Handle rate limit / service unavailable once with small, capped backoff.
        if resp.status_code in (429, 503):
            rl = self.metrics.rate_limit
            delay = rl.retry_after_s or (rl.seconds_until_reset() or 1)
            delay = max(1, min(delay, MAX_BACKOFF))
            log.warning("Rate limited or service unavailable (%s). Backing off %ss and retrying once.", resp.status_code, delay)
            time.sleep(delay)
            resp = do_once(headers)

        if not (200 <= resp.status_code < 300):
            raise QuestradeHTTPError(f"{method} {url} failed ({resp.status_code}): {resp.text}")

        return resp


    def request_json(self, method: str, path: str, *, usage_tags: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """Convenience wrapper that returns the decoded JSON body."""
        resp = self.request(method, path, usage_tags=usage_tags, **kwargs)
        try:
            return resp.json()
        except ValueError as exc:
            raise QuestradeHTTPError(f"Response from {resp.url} is not valid JSON: {exc}") from exc

    def get_accounts(self) -> Dict[str, Any]:
        """Retrieve the authenticated user's account metadata from Questrade."""
        return self.request_json("GET", "/v1/accounts", usage_tags={"endpoint": "/v1/accounts"})


    def _append_api_usage(self, method: str, path: str, usage_tags: Optional[Dict[str, Any]], status: int, latency_ms: int) -> None:
        """Persist API call telemetry to disk without interrupting primary flow."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": method,
            "endpoint": path,
            "status": status,
            "duration_ms": latency_ms,
        }
        if usage_tags:
            record.update({k: v for k, v in usage_tags.items() if v is not None})
        try:
            append_jsonl_record(self.api_usage_path, record)
        except Exception:
            log.debug("Failed to append API usage record", exc_info=True)

def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write ``payload`` as JSON to ``path`` atomically to avoid partial writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _utc_now_s() -> int:
    """Return the current UTC timestamp as integer seconds."""
    return int(datetime.now(timezone.utc).timestamp())





def _normalize_timestamp(ts: str, tz: ZoneInfo) -> datetime:
    """Parse an ISO8601 timestamp into the provided timezone."""
    clean = ts.replace('Z', '+00:00')
    dt = datetime.fromisoformat(clean)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz)


def _synthesize_candle(symbol: str, start_dt: datetime, tz: ZoneInfo,
                       prev_row: Optional[Dict[str, Any]], next_row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Temporary workaround: synthesize a missing candle by averaging neighbours."""
    # TODO: Replace once upstream data gaps are resolved.
    template = prev_row or next_row or {}
    new_row: Dict[str, Any] = {k: template.get(k) for k in template.keys()}
    new_row['start'] = start_dt.isoformat(timespec='microseconds')
    new_row['end'] = (start_dt + timedelta(minutes=15)).isoformat(timespec='microseconds')
    numeric_fields = ('open', 'high', 'low', 'close')

    def as_float(row: Optional[Dict[str, Any]], field: str) -> Optional[float]:
        if not row:
            return None
        value = row.get(field)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    for field in numeric_fields:
        lo = as_float(prev_row, field)
        hi = as_float(next_row, field)
        if lo is not None and hi is not None:
            new_row[field] = (lo + hi) / 2.0
        elif lo is not None:
            new_row[field] = lo
        elif hi is not None:
            new_row[field] = hi

    vol_lo = as_float(prev_row, 'volume')
    vol_hi = as_float(next_row, 'volume')
    if vol_lo is not None and vol_hi is not None:
        new_row['volume'] = int(round((vol_lo + vol_hi) / 2.0))
    elif vol_lo is not None:
        new_row['volume'] = int(round(vol_lo))
    elif vol_hi is not None:
        new_row['volume'] = int(round(vol_hi))

    new_row.setdefault('source', 'synthetic')
    new_row['synthetic_note'] = 'averaged' if prev_row and next_row else 'copied'
    return new_row


def _ensure_15m_continuity(symbol: str, target_date: date, tz: ZoneInfo,
                            rows: List[Dict[str, Any]], start_times: List[datetime]) -> tuple[List[Dict[str, Any]], List[datetime]]:
    start_local = datetime.combine(target_date, TRADING_OPEN).replace(tzinfo=tz)
    end_local = datetime.combine(target_date, TRADING_CLOSE).replace(tzinfo=tz)
    expected_times: List[datetime] = []
    current = start_local
    while current < end_local:
        expected_times.append(current)
        current += timedelta(minutes=15)

    index = {dt: row for dt, row in zip(start_times, rows)}
    repaired: List[Dict[str, Any]] = []
    repaired_times: List[datetime] = []
    patches: List[Dict[str, Any]] = []
    existing_set = set(index.keys())

    for idx, expected_dt in enumerate(expected_times):
        row = index.get(expected_dt)
        if row is None:
            prev_row = repaired[-1] if repaired else None
            next_dt = next((t for t in expected_times[idx + 1:] if t in existing_set), None)
            next_row = index.get(next_dt) if next_dt else None
            row = _synthesize_candle(symbol, expected_dt, tz, prev_row, next_row)
            patches.append({'start': expected_dt.isoformat(), 'note': row.get('synthetic_note', 'synthetic')})
        else:
            row = {k: v for k, v in row.items()}
            row['start'] = expected_dt.isoformat(timespec='microseconds')
            row['end'] = (expected_dt + timedelta(minutes=15)).isoformat(timespec='microseconds')
        repaired.append(row)
        repaired_times.append(expected_dt)

    if patches:
        append_candle_error({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': symbol.upper(),
            'date': target_date.isoformat(),
            'missing': patches,
            'note': 'temporary candle synth workaround',
        })

    return repaired, repaired_times

def _validate_15m_candles(symbol: str, target_date: date, start_times: List[datetime]) -> None:
    """Ensure candle timestamps cover 04:00-19:45 at 15-minute resolution."""
    expected = 64
    if len(start_times) != expected:
        raise RuntimeError(f"{symbol.upper()} {target_date}: expected {expected} candles, got {len(start_times)}")
    first, last = start_times[0], start_times[-1]
    if (first.hour, first.minute) != (TRADING_OPEN.hour, TRADING_OPEN.minute):
        raise RuntimeError(f"{symbol.upper()} {target_date}: first candle starts at {first.time()}, expected {TRADING_OPEN}")
    expected_last = (datetime.combine(target_date, TRADING_CLOSE) - timedelta(minutes=15)).time()
    if (last.hour, last.minute) != (expected_last.hour, expected_last.minute):
        raise RuntimeError(f"{symbol.upper()} {target_date}: last candle starts at {last.time()}, expected {expected_last}")
    previous = first
    for current in start_times[1:]:
        if current - previous != timedelta(minutes=15):
            raise RuntimeError(f"{symbol.upper()} {target_date}: gap detected between {previous.time()} and {current.time()}")
        previous = current


def _extract_symbol_id(payload: Any, want: str) -> Optional[SymbolHit]:
    """Search both 'symbols' and top-level arrays for the desired ticker."""
    want_upper = want.upper()
    if isinstance(payload, dict):
        rows = payload.get("symbols", [])
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []

    candidates: List[SymbolHit] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        sym = (row.get("symbol") or row.get("ticker") or "").upper()
        sid = row.get("symbolId") or row.get("symbolID") or row.get("id")
        if not sym or sid is None:
            continue
        hit = SymbolHit(int(sid), sym, row.get("listingExchange"), row.get("currency"))
        if sym == want_upper:
            return hit
        candidates.append(hit)
    return candidates[0] if candidates else None

def _fetch_candles(session: "QuestradeSession", symbol_id: int, start_iso: str, end_iso: str,
                   interval: str, *, symbol_label: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch raw candle rows for ``symbol_id`` between two ISO timestamps."""
    path = (
        f"/v1/markets/candles/{symbol_id}"
        f"?startTime={start_iso}&endTime={end_iso}&interval={interval}"
    )
    usage = {"endpoint": "/v1/markets/candles", "interval": interval, "symbol_id": symbol_id}
    if symbol_label is not None:
        usage["symbol"] = symbol_label
    payload = session.request_json("GET", path, usage_tags=usage)
    candles = [row for row in payload.get("candles", []) if isinstance(row, dict)]
    candles.sort(key=lambda c: c.get("start", ""))
    return candles


def resolve_symbol_id(session: "QuestradeSession", symbol: str) -> SymbolHit:
    """Resolve a ticker to its Questrade symbolId, using cached metadata when possible."""
    symbol = symbol.strip()
    if not symbol:
        raise ValueError("symbol must be a non-empty string")

    cached = read_symbol_metadata(symbol)
    if cached:
        try:
            cached_symbol = cached.get("symbol") or symbol
            if isinstance(cached_symbol, str) and cached_symbol.strip():
                cached_symbol = cached_symbol.strip()
            else:
                cached_symbol = symbol
            if cached_symbol.upper() != symbol.upper():
                cached_symbol = symbol.upper()
            symbol_id = int(cached.get("symbol_id"))
            return SymbolHit(
                symbol_id=symbol_id,
                symbol=cached_symbol,
                listing_exchange=cached.get("listing_exchange"),
                currency=cached.get("currency"),
            )
        except (TypeError, ValueError, KeyError):
            pass

    encoded = quote(symbol)
    payload = session.request_json("GET", f"/v1/symbols?names={encoded}", usage_tags={"endpoint": "/v1/symbols", "symbol": symbol.upper()})
    hit = _extract_symbol_id(payload, symbol)
    if not hit:
        payload = session.request_json("GET", f"/v1/symbols/search?prefix={encoded}", usage_tags={"endpoint": "/v1/symbols/search", "symbol": symbol.upper()})
        hit = _extract_symbol_id(payload, symbol)
        if not hit:
            raise RuntimeError(f"Could not resolve symbolId for {symbol!r}")

    metadata = hit.to_dict()
    metadata["resolved_at"] = datetime.now(timezone.utc).isoformat()
    write_symbol_metadata(symbol, metadata)
    return hit

def get_15m_candles_for_day(session: "QuestradeSession", symbol: str, day: date | str,
                            tz: ZoneInfo = NY_TZ) -> List[Dict[str, Any]]:
    """Retrieve 15-minute OHLCV candles for a given ticker and local day."""
    hit = resolve_symbol_id(session, symbol)
    metadata = hit.to_dict()
    metadata["resolved_at"] = datetime.now(timezone.utc).isoformat()
    write_symbol_metadata(symbol, metadata)

    target_date = coerce_date(day)
    start_iso, end_iso = day_bounds_iso(target_date, tz)
    raw_rows = _fetch_candles(session, hit.symbol_id, start_iso, end_iso, "FifteenMinutes", symbol_label=hit.symbol)

    start_local = datetime.combine(target_date, TRADING_OPEN).replace(tzinfo=tz)
    end_local = datetime.combine(target_date, TRADING_CLOSE).replace(tzinfo=tz)
    filtered: List[Dict[str, Any]] = []
    start_times: List[datetime] = []
    for row in raw_rows:
        start_value = row.get("start")
        if not isinstance(start_value, str):
            continue
        start_dt = _normalize_timestamp(start_value, tz)
        if start_dt < start_local or start_dt >= end_local:
            continue
        filtered.append(row)
        start_times.append(start_dt)

    _stage_raw_day_candles(symbol, target_date, raw_rows)
    filtered, start_times = _ensure_15m_continuity(symbol, target_date, tz, filtered, start_times)
    _validate_15m_candles(symbol, target_date, start_times)
    persist_day_candles(symbol, target_date, filtered)
    return filtered

def get_latest_intraday_candle(session: "QuestradeSession",
                               symbol: str,
                               interval: str = "FifteenMinutes",
                               tz: ZoneInfo = NY_TZ) -> Optional[Dict[str, Any]]:
    """Return the most recent *completed* intraday candle for `symbol` at `interval`."""
    if interval not in INTERVAL_TO_DELTA:
        raise ValueError(f"Unsupported interval '{interval}'")

    hit = resolve_symbol_id(session, symbol)
    now_local = datetime.now(tz).replace(second=0, microsecond=0)
    boundary = floor_to_interval(now_local, interval)
    lookback = LOOKBACK_BY_INTERVAL.get(interval, timedelta(days=10))

    start_dt = boundary - lookback
    start_iso = start_dt.isoformat(timespec="microseconds")
    end_iso = boundary.isoformat(timespec="microseconds")

    candles = _fetch_candles(session, hit.symbol_id, start_iso, end_iso, interval, symbol_label=hit.symbol)
    if not candles:
        return None

    latest = candles[-1]
    try:
        latest_end = datetime.fromisoformat(latest["end"])
        if latest_end.replace(tzinfo=None) > boundary.replace(tzinfo=None) and len(candles) >= 2:
            return candles[-2]
    except Exception:
        pass
    return latest

def get_latest_15m_candle(session: "QuestradeSession", symbol: str,
                           tz: ZoneInfo = NY_TZ) -> Optional[Dict[str, Any]]:
    """Convenience wrapper for the common 15-minute case."""
    return get_latest_intraday_candle(session, symbol, "FifteenMinutes", tz)

def get_quote(session: "QuestradeSession", symbol: str) -> Quote:
    """Retrieve a Level-1 quote for `symbol` (delay-aware)."""
    hit = resolve_symbol_id(session, symbol)
    payload = session.request_json("GET", f"/v1/markets/quotes/{hit.symbol_id}", usage_tags={"endpoint": "/v1/markets/quotes", "symbol": symbol.upper(), "symbol_id": hit.symbol_id})
    rows = payload.get("quotes", [])
    if not rows:
        raise RuntimeError(f"No quote returned for symbol '{symbol}' (id {hit.symbol_id})")
    return parse_quote(rows[0])

def get_quotes(session: "QuestradeSession", symbols: List[str]) -> Dict[str, Quote]:
    """Batch-quote multiple symbols in a single request."""
    if not symbols:
        return {}

    cleaned: List[str] = []
    seen: Set[str] = set()
    for sym in symbols:
        if not isinstance(sym, str):
            continue
        normalized = sym.strip()
        if not normalized:
            continue
        upper = normalized.upper()
        if upper not in seen:
            seen.add(upper)
            cleaned.append(normalized)

    if not cleaned:
        return {}

    names_param = quote(",".join(cleaned))
    resolved: Dict[str, int] = {}

    try:
        resp = session.request_json("GET", f"/v1/symbols?names={names_param}", usage_tags={"endpoint": "/v1/symbols", "symbols": names_param})
        arr = resp.get("symbols", [])
        for row in arr:
            if not isinstance(row, dict):
                continue
            sym = (row.get("symbol") or "").upper()
            sid = row.get("symbolId")
            if sym and sid is not None:
                resolved[sym] = int(sid)
    except Exception:
        pass

    missing = [s for s in cleaned if s.upper() not in resolved]
    for s in missing:
        hit = resolve_symbol_id(session, s)
        resolved[hit.symbol.upper()] = hit.symbol_id

    if not resolved:
        return {}

    ids = ",".join(str(sid) for sid in resolved.values())
    payload = session.request_json("GET", f"/v1/markets/quotes?ids={ids}", usage_tags={"endpoint": "/v1/markets/quotes", "ids": ids})
    rows = payload.get("quotes", [])

    out: Dict[str, Quote] = {}
    for row in rows:
        quote_obj = parse_quote(row)
        out[quote_obj.symbol.upper()] = quote_obj

    return out

def collect_symbol_data(session: "QuestradeSession", symbols: List[str], *, days: int = 30, delay_s: float = 0.5, tz: ZoneInfo = NY_TZ) -> None:
    """Fetch and persist the last `days` 15-minute sessions for each symbol."""
    if not isinstance(symbols, list):
        raise TypeError("symbols must be a list of strings")
    today = date.today()
    for raw in symbols:
        if raw is None:
            continue
        symbol = str(raw).strip()
        if not symbol:
            continue
        log.info("Collecting %s (%d days)", symbol.upper(), days)
        for offset in range(1, days + 1):
            target_day = today - timedelta(days=offset)
            if target_day.weekday() >= 5:
                log.debug("Skipping weekend date %s for %s", target_day, symbol.upper())
                continue
            rows = get_15m_candles_for_day(session, symbol, target_day, tz=tz)
            log.debug("Collected %s %s (%d candles)", symbol.upper(), target_day, len(rows))
            time.sleep(delay_s)
