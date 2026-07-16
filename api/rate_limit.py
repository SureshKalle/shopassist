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


def rate_limit(request: Request) -> None:
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
