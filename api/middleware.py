# api/middleware.py
"""
HTTP access logging - method, path, client host, status, latency for
every request (INFO), plus full request/response bodies at DEBUG
(LOG_LEVEL=DEBUG) for debugging client integration issues.

Also assigns each request a correlation ID (api/request_context.py) -
reused from an incoming X-Request-ID header if present, echoed back on
the response - so every log line for one request can be grepped out of an
interleaved stream by that ID.

Bodies log verbatim, unmasked (PII masking happens later, in the
orchestrator) - keep DEBUG off outside local dev.
"""

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from api.config import settings
from api.request_context import new_request_id, reset_request_id, set_request_id

logger = logging.getLogger("api.access")


def _decode_and_truncate(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace")
    max_chars = settings.log_body_max_chars
    if len(text) > max_chars:
        return f"{text[:max_chars]}...({len(text)} chars total)"
    return text


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = new_request_id(request.headers.get("X-Request-ID"))
        token = set_request_id(request_id)
        try:
            return await self._dispatch(request, call_next, request_id)
        finally:
            reset_request_id(token)

    async def _dispatch(self, request: Request, call_next, request_id: str):
        start = time.perf_counter()
        client_host = request.client.host if request.client else "-"
        debug = logger.isEnabledFor(logging.DEBUG)

        if debug:
            request_body = await request.body()
            logger.debug(
                "--> %s %s from %s body=%s",
                request.method, request.url.path, client_host,
                _decode_and_truncate(request_body),
            )
        else:
            logger.info("--> %s %s from %s", request.method, request.url.path, client_host)

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "<-- %s %s failed after %.1fms", request.method, request.url.path, duration_ms
            )
            raise

        duration_ms = (time.perf_counter() - start) * 1000

        if debug:
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
            logger.debug(
                "<-- %s %s %d (%.1fms) body=%s",
                request.method, request.url.path, response.status_code, duration_ms,
                _decode_and_truncate(response_body),
            )
            # body_iterator is consumed above - rebuild the response so the
            # client still gets it (content-length is recomputed, not copied).
            headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
            response = Response(
                content=response_body,
                status_code=response.status_code,
                headers=headers,
                media_type=response.media_type,
            )
        else:
            logger.info(
                "<-- %s %s %d (%.1fms)",
                request.method, request.url.path, response.status_code, duration_ms,
            )

        response.headers["X-Request-ID"] = request_id
        return response
