# api/security.py
"""
Dummy API-key auth - a stand-in for real JWT bearer-token auth: one shared
secret instead of per-caller signed tokens with claims. The seam is
deliberate - callers depend on verify_api_key, not on how it checks the
caller, so swapping in real JWT verification later (decode, check
signature/expiry/claims) means changing this file only.

Controlled by two env vars (see .env.example):
  API_KEY          - the expected secret. Empty disables the check entirely.
  API_KEY_ENFORCE  - false (default): missing/invalid key is logged, request
                     still proceeds. true: missing/invalid key gets a 401.
"""

import logging
import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from api.config import settings

logger = logging.getLogger("api.auth")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str | None = Security(_api_key_header)) -> str | None:
    """FastAPI dependency - returns the validated key (or None if auth is
    disabled/advisory), raises 401 if enforcement is on and the key is
    missing or wrong."""
    if not settings.api_key:
        if settings.api_key_enforce:
            # Enforcement on but nothing configured to check against -
            # fail loud here rather than silently letting everyone in.
            logger.error("API_KEY_ENFORCE=true but no API_KEY configured - rejecting all requests")
            raise HTTPException(status_code=500, detail="API key auth misconfigured")
        return None

    if api_key and secrets.compare_digest(api_key, settings.api_key):
        logger.info("Request authenticated (key ...%s)", api_key[-4:])
        return api_key

    logger.warning(
        "Missing or invalid API key (mode=%s, key_provided=%s)",
        "enforced" if settings.api_key_enforce else "advisory",
        bool(api_key),
    )
    if settings.api_key_enforce:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
    return None
