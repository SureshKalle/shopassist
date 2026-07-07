# services/pii_masker.py
"""
PII Masking Service.

Two backends are supported, controlled by PII_MASKER_MODE in .env:
  - "regex"    : Fast, zero-dependency, rule-based patterns (default).
                 Covers email, phone, SSN, credit card, name-like tokens,
                 URLs, and IP addresses.
  - "presidio" : Microsoft Presidio NLP-based detection (more accurate).
                 Requires: pip install presidio-analyzer presidio-anonymizer
                 and a spaCy language model (python -m spacy download en_core_web_sm).

In either mode the service:
  1. Detects PII in raw text.
  2. Replaces each entity with a typed placeholder, e.g. [EMAIL], [PHONE].
  3. Returns a MaskedQuery with the masked text and a hash of the original
     (for audit without re-exposing PII).
"""

import re
import uuid
import hashlib
import logging
from typing import Optional, TYPE_CHECKING

from common.config import PII_MASKER_MODE
from common.models import MaskedQuery

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regex patterns (ordered: more specific patterns first)
# ---------------------------------------------------------------------------
_REGEX_PATTERNS: list[tuple[str, str]] = [
    # Email
    (r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", "[EMAIL]"),
    # US/international phone  (+1-800-555-1234 / (800) 555-1234 / 800.555.1234)
    (r"(\+?\d{1,3}[\s\-.])?(\(?\d{3}\)?[\s\-.])\d{3}[\s\-\.]\d{4}", "[PHONE]"),
    # SSN
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
    # Credit card (major formats)
    (r"\b(?:\d[ \-]?){13,16}\b", "[CREDIT_CARD]"),
    # IPv4
    (r"\b\d{1,3}(\.\d{1,3}){3}\b", "[IP_ADDRESS]"),
    # URLs (http / https / www)
    (r"https?://[^\s]+|www\.[^\s]+", "[URL]"),
    # Street address patterns  (e.g. "123 Main St", "456 Oak Avenue")
    (
        r"\b\d{1,5}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+"
        r"(?:St(?:reet)?|Ave(?:nue)?|Blvd|Rd|Road|Dr(?:ive)?|Ln|Lane|"
        r"Ct|Court|Pl|Place|Way|Pkwy|Parkway)\b",
        "[ADDRESS]",
    ),
    # Common name patterns caught by simple heuristic:
    # "John Doe", "Jane Smith" — title-case word followed by title-case word
    # (This is intentionally conservative to avoid false positives on product names.)
    (r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b", "[NAME]"),
]

_COMPILED_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(pat), repl) for pat, repl in _REGEX_PATTERNS
]


def _mask_with_regex(text: str) -> str:
    """Apply all regex patterns sequentially."""
    for pattern, replacement in _COMPILED_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Presidio backend (loaded lazily so regex mode has zero import overhead)
# ---------------------------------------------------------------------------
_presidio_analyzer: Optional["AnalyzerEngine"] = None
_presidio_anonymizer: Optional["AnonymizerEngine"] = None


def _get_presidio():
    """Lazy-load Presidio engines. Raises ImportError with a helpful message."""
    global _presidio_analyzer, _presidio_anonymizer
    if _presidio_analyzer is None:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            _presidio_analyzer = AnalyzerEngine()
            _presidio_anonymizer = AnonymizerEngine()
            logger.info("Presidio PII engines loaded.")
        except ImportError as exc:
            raise ImportError(
                "Presidio is not installed or spaCy model is missing. "
                "Run: pip install presidio-analyzer presidio-anonymizer && "
                "python -m spacy download en_core_web_sm"
            ) from exc
    return _presidio_analyzer, _presidio_anonymizer


def _mask_with_presidio(text: str) -> str:
    """Use Microsoft Presidio for NLP-based PII detection and anonymisation."""
    analyzer, anonymizer = _get_presidio()
    results = analyzer.analyze(text=text, language="en")
    # NOTE: presidio-analyzer's RecognizerResult and presidio-anonymizer's
    # RecognizerResult are structurally identical but distinct classes across
    # the two packages, so static type-checkers (Pylance/mypy) flag this as
    # invariant-list mismatch even though it works correctly at runtime.
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)  # type: ignore[arg-type]
    return anonymized.text


# ---------------------------------------------------------------------------
# Public service class
# ---------------------------------------------------------------------------
class PIIMasker:
    """
    PII detection and masking service.

    Usage:
        masker = PIIMasker()
        masked = masker.mask_text("Hi, I'm John Doe. Email: john@example.com")
        print(masked.masked_text)   # "Hi, I'm [NAME]. Email: [EMAIL]"
    """

    def __init__(self, mode: Optional[str] = None):
        self.mode = (mode or PII_MASKER_MODE).lower()
        if self.mode not in ("regex", "presidio"):
            logger.warning(
                "Unknown PII_MASKER_MODE '%s'. Falling back to 'regex'.", self.mode
            )
            self.mode = "regex"
        logger.info("PIIMasker initialised in '%s' mode.", self.mode)

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def mask_text(
        self,
        text: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> MaskedQuery:
        """
        Detect and mask PII in *text*.

        Returns a MaskedQuery containing:
          - masked_text          : text with PII replaced by typed tokens
          - original_text_hash   : SHA-256 of the raw text (for audit trail)
          - session_id / user_id : echoed back (generated if not supplied)
        """
        logger.debug("[PIIMasker] Masking text (mode=%s): %.60s…", self.mode, text)

        try:
            if self.mode == "presidio":
                masked_text = _mask_with_presidio(text)
            else:
                masked_text = _mask_with_regex(text)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[PIIMasker] Masking failed (%s). Falling back to regex. Error: %s",
                self.mode,
                exc,
            )
            masked_text = _mask_with_regex(text)

        original_hash = hashlib.sha256(text.encode()).hexdigest()

        return MaskedQuery(
            session_id=session_id or f"mock_session_{uuid.uuid4().hex[:8]}",
            user_id=user_id or f"mock_user_{uuid.uuid4().hex[:8]}",
            masked_text=masked_text,
            original_text_hash=original_hash,
        )

    def is_pii_present(self, text: str) -> bool:
        """
        Returns True if the text appears to contain PII.
        Uses the same backend as mask_text().
        """
        if self.mode == "presidio":
            try:
                analyzer, _ = _get_presidio()
                results = analyzer.analyze(text=text, language="en")
                return len(results) > 0
            except Exception:  # noqa: BLE001
                pass  # fall through to regex
        # Regex fallback: check whether any pattern would match
        return any(pat.search(text) for pat, _ in _COMPILED_PATTERNS)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    masker = PIIMasker(mode="regex")

    samples = [
        "My name is John Doe and my email is john.doe@example.com.",
        "Call me at +1-800-555-1234 or (212) 555-0198.",
        "I live at 123 Main St. My SSN is 123-45-6789.",
        "Card number 4111 1111 1111 1111 expires next month.",
        "Check out https://www.example.com for more info.",
    ]
    for s in samples:
        result = masker.mask_text(s)
        print(f"IN : {s}")
        print(f"OUT: {result.masked_text}")
        print(f"PII present: {masker.is_pii_present(s)}")
        print()
