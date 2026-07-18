# api/request_context.py
"""
Per-request correlation ID, so one request's log lines (access log, auth,
orchestrator errors, ...) can be grepped out of an interleaved log stream
by a single ID, and handed back to the client for support/debugging.

Uses a ContextVar (not a manually-threaded parameter) so any logger call
anywhere during a request picks it up automatically via RequestIDLogFilter,
without every function in the call chain needing a request_id argument.
"""

import logging
import uuid
from contextvars import ContextVar

_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def new_request_id(incoming: str | None) -> str:
    """Reuse a client-supplied X-Request-ID (cross-service tracing) or mint one."""
    return incoming if incoming else uuid.uuid4().hex[:12]


def set_request_id(value: str):
    """Returns a reset token - caller must reset it when the request ends."""
    return _request_id.set(value)


def reset_request_id(token) -> None:
    _request_id.reset(token)


def get_request_id() -> str:
    return _request_id.get()


class RequestIDLogFilter(logging.Filter):
    """Attaches the current request ID to every LogRecord as %(request_id)s."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True
