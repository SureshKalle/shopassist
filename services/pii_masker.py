# services/pii_masker.py
import uuid
from typing import List, Optional
from common.models import MaskedQuery

class PIIMasker:
    """
    Handles PII detection and masking. In a real deployment, this might be a dedicated
    microservice or a highly optimized shared library.
    """
    def mask_text(self, text: str, session_id: Optional[str] = None, user_id: Optional[str] = None) -> MaskedQuery:
        print(f"  [PII Masker] Masking PII in text: '{text}'...")
        
        # --- Advanced PII Detection (Conceptual) ---
        # In a production system, this would involve:
        # 1. Named Entity Recognition (NER) models specifically fine-tuned for PII.
        #    e.g., spaCy with a custom PII model, or specialized libraries.
        # 2. Rule-based regex patterns for common formats (email, phone, SSN, etc.).
        # 3. Cloud-based DLP (Data Loss Prevention) APIs (e.g., Google Cloud DLP, AWS Comprehend PII).
        # 4. Contextual analysis to distinguish true PII from similar-looking non-PII words.

        # --- Simple Mock Masking for Demonstration ---
        masked_text = text.replace("john.doe@example.com", "[EMAIL]") \
                          .replace("John Doe", "[NAME]") \
                          .replace("123-456-7890", "[PHONE]") \
                          .replace("123 Main St", "[ADDRESS]") \
                          .replace("Jane Smith", "[NAME]") # Add more mock patterns

        # Original text hash helps for audit without exposing PII to LLMs.
        # We generate mock session/user IDs if not provided, for pipeline usage.
        return MaskedQuery(
            session_id=session_id if session_id else f"mock_session_{uuid.uuid4().hex[:8]}",
            user_id=user_id if user_id else f"mock_user_{uuid.uuid4().hex[:8]}",
            masked_text=masked_text,
            original_text_hash=str(hash(text)) # Simple hash for demonstration purposes
        )

    def is_pii_present(self, text: str) -> bool:
        """
        Simulates PII detection. Useful for validation or conditional masking.
        """
        # In a real system, this would be backed by the same detection logic as mask_text.
        common_pii_keywords = ["email", "phone", "address", "name"]
        return any(keyword in text.lower() for keyword in common_pii_keywords)

# Example of how this masker might be used:
if __name__ == "__main__":
    masker = PIIMasker()
    sample_text = "My name is John Doe and my email is john.doe@example.com. My order ID is 12345."
    masked_data = masker.mask_text(sample_text)
    print(f"Original: '{sample_text}'")
    print(f"Masked: '{masked_data.masked_text}'")
    print(f"PII Present (in original): {masker.is_pii_present(sample_text)}")
    print(f"PII Present (in masked): {masker.is_pii_present(masked_data.masked_text)}")
