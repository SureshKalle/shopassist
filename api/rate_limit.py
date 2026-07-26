# api/rate_limit.py
"""
Simple in-memory sliding-window rate limiter, keyed on the caller's API key
if present (see api/security.py) else client IP. One counter per process -
not shared across replicas/workers; swap for Redis-backed limiting before
running more than one.

RATE_LIMIT_PER_MINUTE <= 0 disables this entirely.
"""

import logging
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException, Request

from api.config import settings

logger = logging.getLogger("api.rate_limit")

_WINDOW_SECONDS = 60.0
_lock = Lock()
_hits: dict[str, deque] = defaultdict(deque)
# Amortized cleanup counter/interval - see the sweep at the end of
# rate_limit() below for why this exists.
_calls_since_sweep = 0
_SWEEP_EVERY = 200


def rate_limit(request: Request) -> None:
    global _calls_since_sweep
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return

    key = request.headers.get("X-API-Key") or (request.client.host if request.client else "unknown")
    now = time.monotonic()

    with _lock:
        hits = _hits[key]
        while hits and now - hits[0] > _WINDOW_SECONDS:
            hits.popleft()
        if len(hits) >= limit:
            logger.warning("Rate limit exceeded for %s (%d/min)", key, limit)
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")
        hits.append(now)

        # A key only gets its deque trimmed when that same key is seen again -
        # a caller who stops appearing entirely leaves a stale (but
        # non-empty) deque in _hits forever otherwise, so process memory
        # grows with the number of distinct callers/IPs ever seen, not just
        # currently-active ones. Amortized rather than every call, since this
        # is a full dict scan; `hits` (this request's own key) was just
        # trimmed/appended to above, so it never gets removed here even if
        # this call happens to trigger the sweep.
        _calls_since_sweep += 1
        if _calls_since_sweep >= _SWEEP_EVERY:
            _calls_since_sweep = 0
            for other_key in list(_hits.keys()):
                other_hits = _hits[other_key]
                while other_hits and now - other_hits[0] > _WINDOW_SECONDS:
                    other_hits.popleft()
                if not other_hits:
                    del _hits[other_key]
