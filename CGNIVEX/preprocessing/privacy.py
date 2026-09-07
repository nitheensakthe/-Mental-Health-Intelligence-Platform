import re
import hashlib

class PrivacyScrubber:
    """
    Privacy and Ethics module for removing Personally Identifiable Information (PII)
    and anonymizing user identifiers.
    """
    
    # Regular expressions for PII elements
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE_PATTERN = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    USER_MENTION_PATTERN = re.compile(r'@\w+|u/\w+')

    @staticmethod
    def anonymize_user_id(user_id: str, salt: str = "cgnivex_salt_2026") -> str:
        """Hashes a user ID using SHA-256 to ensure anonymity."""
        if not user_id or str(user_id).strip() == "":
            return "usr_anonymous"
        
        raw_string = f"{user_id}_{salt}".encode('utf-8')
        hashed = hashlib.sha256(raw_string).hexdigest()[:12]
        return f"usr_{hashed}"

    def scrub_text(self, text: str) -> str:
        """
        Removes emails, phone numbers, URLs, and user mentions from post text.
        """
        if not isinstance(text, str):
            return ""

        # Replace PII elements with anonymized placeholders
        text = self.EMAIL_PATTERN.sub('[EMAIL_REMOVED]', text)
        text = self.PHONE_PATTERN.sub('[PHONE_REMOVED]', text)
        text = self.URL_PATTERN.sub('[URL_REMOVED]', text)
        text = self.USER_MENTION_PATTERN.sub('[USER_REMOVED]', text)

        return text.strip()
