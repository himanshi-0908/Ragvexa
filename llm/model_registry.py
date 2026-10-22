"""
model_registry.py — Which NVIDIA models can a given API key actually use?

NVIDIA retires models (HTTP 410) and not every key is entitled to every model
(HTTP 404 "Function ... not found for account"). This module validates a key,
probes candidate model ids, caches the result per key fingerprint (never the
key itself) and suggests a working replacement when a configured model fails.

Nothing here logs or returns the key.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("ragvexa")

NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
PROBE_TIMEOUT = 60
CACHE_TTL_SECONDS = 3600

# Ordered by preference for ANSWER generation: fast, concise models first,
# reasoning models (slower, token-heavy) later. Verified 2026-10-22.
CANDIDATE_MODELS: List[str] = [
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-nano-3-30b-a3b",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "mistralai/mistral-large-2-instruct",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
]
REASONING_MODELS = {"nvidia/nemotron-3.5-lightning-30b-a3b"}


@dataclass
class ProbeResult:
    model: str
    ok: bool
    status: object            # int HTTP status or "ERR"
    latency_ms: int
    output_tokens: Optional[int]
    note: str                 # human-readable reason on failure


def key_fingerprint(api_key: Optional[str]) -> str:
    """Stable, non-reversible id for caching and reports. Never the key."""
    if not api_key:
        return "none"
    return hashlib.sha256(api_key.encode()).hexdigest()[:12]


def key_display(api_key: Optional[str]) -> str:
    """Last four characters for the UI/report, e.g. '…ab12'."""
    if not api_key or len(api_key) < 8:
        return "none"
    return "…" + api_key[-4:]


def _request(api_key: str, path: str, body: Optional[dict] = None, timeout: int = PROBE_TIMEOUT):
    req = urllib.request.Request(
        NVIDIA_BASE + path,
        data=json.dumps(body).encode() if body else None,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode())
            msg = detail.get("detail") or detail.get("title") or (detail.get("error") or {}).get("message") or ""
        except Exception:
            msg = ""
        return e.code, str(msg)[:160]
    except Exception as e:  # network / timeout
        return "ERR", str(e)[:160]


def validate_key(api_key: Optional[str]) -> Tuple[bool, str, List[str]]:
    """
    Check a key against GET /models.
    Returns (ok, message, visible_model_ids). The message never contains the key.
    """
    if not api_key or not api_key.strip():
        return False, "No API key provided.", []
    api_key = api_key.strip()
    if not api_key.startswith("nvapi-"):
        return False, "NVIDIA API keys start with 'nvapi-'.", []
    status, data = _request(api_key, "/models", timeout=30)
    if status == 200 and isinstance(data, dict):
        ids = sorted(m.get("id", "") for m in data.get("data", []) if m.get("id"))
        return True, f"Key is valid. {len(ids)} models visible.", ids
    if status == 401:
        return False, "NVIDIA rejected this key (401 Unauthorized).", []
    return False, f"Could not validate key (HTTP {status}). {data if isinstance(data, str) else ''}".strip(), []


def probe_model(api_key: str, model: str) -> ProbeResult:
    """One short chat completion to see whether this key can invoke `model`."""
    t0 = time.time()
    status, data = _request(api_key, "/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word OK."}],
        "max_tokens": 200,
    })
    latency = int((time.time() - t0) * 1000)
    if status == 200 and isinstance(data, dict):
        usage = data.get("usage") or {}
        return ProbeResult(model, True, 200, latency, usage.get("completion_tokens"), "")
    hint = {
        410: "retired by NVIDIA",
        404: "not enabled for this key",
        403: "authorization failed for this key",
        401: "invalid key",
        429: "rate limited",
    }.get(status, "")
    return ProbeResult(model, False, status, latency, None, (hint or str(data))[:120])


# ─────────────────────────────────────────────
# Cache: fingerprint → {model: ProbeResult}
# ─────────────────────────────────────────────

_cache: Dict[str, Tuple[float, Dict[str, ProbeResult]]] = {}
_cache_lock = threading.Lock()


def _cached(fp: str) -> Dict[str, ProbeResult]:
    with _cache_lock:
        entry = _cache.get(fp)
        if entry and time.time() - entry[0] < CACHE_TTL_SECONDS:
            return entry[1]
        _cache.pop(fp, None)
        return {}


def _remember(fp: str, results: Dict[str, ProbeResult]) -> None:
    with _cache_lock:
        ts, existing = _cache.get(fp, (time.time(), {}))
        existing.update(results)
        _cache[fp] = (ts if time.time() - ts < CACHE_TTL_SECONDS else time.time(), existing)


def clear_cache(api_key: Optional[str] = None) -> None:
    with _cache_lock:
        if api_key is None:
            _cache.clear()
        else:
            _cache.pop(key_fingerprint(api_key), None)


def probe_models(api_key: str, models: List[str], use_cache: bool = True) -> Dict[str, ProbeResult]:
    """Probe several models concurrently; cached per key fingerprint for an hour."""
    fp = key_fingerprint(api_key)
    known = _cached(fp) if use_cache else {}
    todo = [m for m in dict.fromkeys(models) if m and m not in known]
    results: Dict[str, ProbeResult] = {m: known[m] for m in models if m in known}
    if todo:
        with ThreadPoolExecutor(max_workers=min(6, len(todo))) as pool:
            for res in pool.map(lambda m: probe_model(api_key, m), todo):
                results[res.model] = res
        _remember(fp, {m: results[m] for m in todo})
    return results


def working_models(api_key: str, extra: Optional[List[str]] = None, use_cache: bool = True) -> List[str]:
    """Candidate (plus `extra`) models this key can invoke, in preference order."""
    candidates = list(dict.fromkeys([*(extra or []), *CANDIDATE_MODELS]))
    results = probe_models(api_key, candidates, use_cache=use_cache)
    ordered = [m for m in CANDIDATE_MODELS if results.get(m) and results[m].ok]
    ordered += [m for m in candidates if m not in ordered and results.get(m) and results[m].ok]
    return ordered


def suggest_model(api_key: str, exclude: Optional[List[str]] = None, role: str = "answer") -> Optional[str]:
    """
    First working model for this key that is not in `exclude`.
    role="answer": prefer non-reasoning models. role="judge": prefer a model
    different from the answer model when possible (caller passes it in exclude).
    """
    exclude = set(exclude or [])
    options = [m for m in working_models(api_key) if m not in exclude]
    if not options:
        return None
    if role == "answer":
        non_reasoning = [m for m in options if m not in REASONING_MODELS]
        return (non_reasoning or options)[0]
    return options[0]


def mark_failed(api_key: str, model: str, status: object, note: str) -> None:
    """Record a failure observed during a real call so we do not retry it for an hour."""
    _remember(key_fingerprint(api_key), {model: ProbeResult(model, False, status, 0, None, note)})
