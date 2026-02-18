import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Optional

from .config import UPSTREAM_QUOTA_STATE_FILE, get_base_model_name


_LOCK = Lock()
_STATE_CACHE: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class QuotaBlock:
    model: str
    next_available_at: datetime
    source_message: Optional[str]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_rfc3339(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _from_rfc3339(value: str) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    try:
        value = value.strip()
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _normalize_model(model: Optional[str]) -> str:
    if not model:
        return ""
    model = model.strip()
    if model.startswith("models/"):
        model = model[len("models/") :]
    model = get_base_model_name(model)
    return model


_RESET_AFTER_RE = re.compile(r"reset after\s+((?:\d+\s*[dhms]\s*)+)", re.IGNORECASE | re.MULTILINE)
_DURATION_PART_RE = re.compile(r"(\d+)\s*([dhms])", re.IGNORECASE)


def parse_quota_reset_after(message: str) -> Optional[timedelta]:
    """
    Parse messages like:
      "Your quota will reset after 14h28m8s."
    Returns a timedelta if found, else None.
    """
    if not message:
        return None

    match = _RESET_AFTER_RE.search(message)
    if not match:
        return None

    duration_text = match.group(1).strip().rstrip(".")
    total_seconds = 0
    for amount_str, unit in _DURATION_PART_RE.findall(duration_text):
        amount = int(amount_str)
        unit = unit.lower()
        if unit == "d":
            total_seconds += amount * 86400
        elif unit == "h":
            total_seconds += amount * 3600
        elif unit == "m":
            total_seconds += amount * 60
        elif unit == "s":
            total_seconds += amount
    if total_seconds <= 0:
        return None
    return timedelta(seconds=total_seconds)


def format_duration(delta: timedelta) -> str:
    seconds = int(max(0, delta.total_seconds()))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return "".join(parts)


def _default_state() -> dict[str, Any]:
    return {"version": 1, "models": {}}


def _load_state_unlocked() -> dict[str, Any]:
    global _STATE_CACHE
    if _STATE_CACHE is not None:
        return _STATE_CACHE

    try:
        with open(UPSTREAM_QUOTA_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                data = _default_state()
    except FileNotFoundError:
        data = _default_state()
    except Exception:
        data = _default_state()

    if "models" not in data or not isinstance(data.get("models"), dict):
        data["models"] = {}
    _STATE_CACHE = data
    return data


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-quota-", suffix=".json", dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


def get_quota_block(model: Optional[str]) -> Optional[QuotaBlock]:
    key = _normalize_model(model)
    if not key:
        return None
    now = _utcnow()
    with _LOCK:
        state = _load_state_unlocked()
        entry = state.get("models", {}).get(key)
        if not isinstance(entry, dict):
            return None
        next_available_at = _from_rfc3339(entry.get("next_available_at", ""))
        if not next_available_at:
            return None
        if next_available_at <= now:
            return None
        return QuotaBlock(model=key, next_available_at=next_available_at, source_message=entry.get("source_message"))


def record_upstream_429(
    model: Optional[str],
    message: Optional[str],
    *,
    retry_after_s: Optional[float] = None,
) -> Optional[datetime]:
    """
    When upstream returns a 429 with a message that includes a reset duration,
    persist the computed next-available datetime.
    """
    key = _normalize_model(model)
    if not key:
        return None

    delta = None
    if message:
        delta = parse_quota_reset_after(message)
    if delta is None and retry_after_s is not None:
        try:
            retry_after_s = float(retry_after_s)
        except Exception:
            retry_after_s = None
        if retry_after_s is not None and retry_after_s > 0:
            delta = timedelta(seconds=retry_after_s)
    if delta is None:
        return None

    next_available_at = _utcnow() + delta
    with _LOCK:
        state = _load_state_unlocked()
        models = state.setdefault("models", {})
        entry = models.get(key)
        if not isinstance(entry, dict):
            entry = {}
            models[key] = entry
        entry["next_available_at"] = _to_rfc3339(next_available_at)
        entry["updated_at"] = _to_rfc3339(_utcnow())
        entry["source_message"] = message
        _atomic_write_json(UPSTREAM_QUOTA_STATE_FILE, state)
    return next_available_at
