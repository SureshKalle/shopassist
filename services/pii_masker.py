# services/pii_masker.py
"""
Masks PII before any customer text reaches an LLM - the very first step of
every request (services/orchestrator.py step 1, services/data_pipeline.py's
ingestion).

Regex-based, not a full NER model - same class of trade-off as
services/guardrails.py's output-side patterns (which this mirrors for
email/phone, kept as separate compiled patterns rather than imported since
the two files mask different directions - input vs. output - and are
allowed to drift independently). Two categories use a conservative,
phrase-anchored heuristic instead of a bare pattern match, specifically to
avoid mangling the Title-Case product/agent/city names an e-commerce bot's
text is full of:
  - name: only fires on an explicit self-introduction ("my name is X",
    "I'm X", "I am X", "this is X") - a bare two-capitalized-words scan
    would also hit "Gaming Laptop", "Maximus", or a shipping city name.
  - address: only fires on the "<number> <words> <street-suffix>" shape
    (e.g. "123 Main St") - the same shape the old hardcoded stub covered
    (a single literal string), generalized to any street name/number.

Swapping in a real NER model (spaCy, a cloud DLP API) later is a drop-in
replacement as long as mask_text() keeps returning a MaskedQuery - nothing
downstream depends on *how* the masking happened.
"""
import logging
import re
import uuid
from typing import Optional

from common.models import MaskedQuery

logger = logging.getLogger(__name__)

_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

# (?<!\d) / (?<!-) keep this from matching a piece of a longer alphanumeric
# ID - order_id/product_id/session_id are all "<prefix>-<hex chars>"
# (clients/ecommerce_api_client.py mints e.g. "ord-<10 hex chars>"), and a
# hex run can occasionally be all-decimal-digit by chance. A real phone
# number in free text is never directly preceded by a bare hyphen with no
# space, so this costs nothing in false negatives.
_PHONE_PATTERN = re.compile(r"(?<!\d)(?<!-)\+?(?:\d[-.\s]?){9,11}\d(?!\d)")

# Anchored on an explicit self-introduction phrase rather than a bare
# two-capitalized-words scan - see module docstring for why. The (?i:...)
# scopes case-insensitivity to the phrase alternation only ("My name is" /
# "MY NAME IS" both count) - the captured name itself must stay
# case-SENSITIVELY capitalized, or a global re.IGNORECASE would let the
# capture group swallow trailing lowercase words too (e.g. "Jane Smith
# and" - "and" starting with lowercase 'a' would otherwise still satisfy
# a case-insensitive `[A-Z]`).
_NAME_PATTERN = re.compile(
    r"\b(?i:my name is|i'?m|i am|this is)\s+"
    r"([A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+){0,2})"
)

# "<number> <word(s)> <street-suffix>" - generalizes the old stub's one
# hardcoded example ("123 Main St") to any street name/number.
_ADDRESS_PATTERN = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9.'-]+(?:\s+[A-Za-z0-9.'-]+){0,3}\s+"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd|Court|Ct|Place|Pl)\b\.?",
    re.IGNORECASE,
)


def _mask_name(match: re.Match) -> str:
    # group(1) (the captured name) is always the tail end of group(0) (the
    # full "my name is <Name>" match) - slice it off and keep the
    # introductory phrase, rather than replacing the whole match, so the
    # masked text still reads naturally to the router/agent LLMs.
    name = match.group(1)
    return match.group(0)[: -len(name)] + "[NAME]"


class PIIMasker:
    """Regex-based PII detection/masking. Stateless, safe to share one
    instance process-wide (same as services/guardrails.py's GuardrailService).
    See module docstring for what each category catches and why."""

    def mask_text(self, text: str, session_id: Optional[str] = None, user_id: Optional[str] = None) -> MaskedQuery:
        masked_text = text or ""
        if masked_text:
            masked_text = _EMAIL_PATTERN.sub("[EMAIL]", masked_text)
            masked_text = _PHONE_PATTERN.sub("[PHONE]", masked_text)
            masked_text = _NAME_PATTERN.sub(_mask_name, masked_text)
            masked_text = _ADDRESS_PATTERN.sub("[ADDRESS]", masked_text)

        was_masked = masked_text != (text or "")
        logger.info("PII masking complete: session_id=%s masked=%s", session_id, was_masked)

        # Original text hash helps for audit without exposing PII to LLMs.
        # We generate mock session/user IDs if not provided, for pipeline usage.
        return MaskedQuery(
            session_id=session_id if session_id else f"mock_session_{uuid.uuid4().hex[:8]}",
            user_id=user_id if user_id else f"mock_user_{uuid.uuid4().hex[:8]}",
            masked_text=masked_text,
            original_text_hash=str(hash(text)),  # Simple hash for demonstration purposes
        )

    def is_pii_present(self, text: str) -> bool:
        """True if any of the categories above would fire on `text` - same
        detection logic mask_text() itself uses, not a separate heuristic."""
        if not text:
            return False
        return bool(
            _EMAIL_PATTERN.search(text)
            or _PHONE_PATTERN.search(text)
            or _NAME_PATTERN.search(text)
            or _ADDRESS_PATTERN.search(text)
        )


# Example of how this masker might be used:
if __name__ == "__main__":
    masker = PIIMasker()
    sample_text = (
        "Hi, my name is Alex Carter and my email is alex.carter@gmail.com. "
        "You can also reach me at +91 93717 22926 or at 42 Lakeview Road. "
        "My order ID is ord-4f9c2e1a8b, can you check on it?"
    )
    masked_data = masker.mask_text(sample_text)
    print(f"Original: '{sample_text}'")
    print(f"Masked:   '{masked_data.masked_text}'")
    print(f"PII Present (in original): {masker.is_pii_present(sample_text)}")
    print(f"PII Present (in masked):   {masker.is_pii_present(masked_data.masked_text)}")
