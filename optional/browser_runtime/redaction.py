"""
Sensitive Data Redaction
========================

Utilities for redacting sensitive data from browser logs.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set


# Field names that should be redacted
SENSITIVE_FIELD_NAMES: Set[str] = {
    # Passwords
    "password", "passwd", "pwd", "pass",
    "current_password", "new_password", "confirm_password",
    "old_password", "user_password", "user_pass",

    # Secrets/Tokens
    "secret", "secret_key", "secretkey",
    "token", "access_token", "refresh_token", "auth_token",
    "api_key", "apikey", "api_secret",
    "private_key", "privatekey",

    # Authentication
    "auth", "authorization", "authentication",
    "credential", "credentials",
    "session_id", "sessionid", "session_key",

    # Personal Identifiable Information (PII)
    "ssn", "social_security", "socialsecurity",
    "credit_card", "creditcard", "card_number", "cardnumber",
    "cvv", "cvc", "security_code",
    "pin", "pin_code",

    # Banking
    "account_number", "accountnumber", "routing_number",
    "bank_account", "iban", "swift",
}

# Patterns that suggest sensitive content
SENSITIVE_PATTERNS: List[re.Pattern] = [
    # API keys (common formats)
    re.compile(r"[a-zA-Z0-9_-]{20,}"),
    # Credit card numbers (basic)
    re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
    # SSN format
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # Email (sometimes considered PII)
    re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
]

# Redaction placeholder
REDACTED = "[REDACTED]"


def is_sensitive_field(field_name: str) -> bool:
    """
    Check if a field name suggests sensitive content.

    Args:
        field_name: The field name to check

    Returns:
        True if field appears to be sensitive
    """
    normalized = field_name.lower().replace("-", "_").replace(" ", "_")
    return normalized in SENSITIVE_FIELD_NAMES


def redact_value(value: Any) -> str:
    """
    Redact a sensitive value.

    Args:
        value: The value to redact

    Returns:
        Redacted placeholder string
    """
    return REDACTED


def redact_string_patterns(text: str) -> str:
    """
    Redact sensitive patterns from a string.

    Args:
        text: Text to scan and redact

    Returns:
        Text with sensitive patterns redacted
    """
    result = text

    for pattern in SENSITIVE_PATTERNS:
        result = pattern.sub(REDACTED, result)

    return result


def redact_dict(data: Dict[str, Any], deep: bool = True) -> Dict[str, Any]:
    """
    Redact sensitive fields from a dictionary.

    Args:
        data: Dictionary to redact
        deep: Whether to recursively redact nested dicts

    Returns:
        New dictionary with sensitive fields redacted
    """
    result = {}

    for key, value in data.items():
        if is_sensitive_field(key):
            result[key] = REDACTED
        elif deep and isinstance(value, dict):
            result[key] = redact_dict(value, deep=True)
        elif deep and isinstance(value, list):
            result[key] = redact_list(value, deep=True)
        elif isinstance(value, str):
            result[key] = redact_string_patterns(value)
        else:
            result[key] = value

    return result


def redact_list(data: List[Any], deep: bool = True) -> List[Any]:
    """
    Redact sensitive data from a list.

    Args:
        data: List to redact
        deep: Whether to recursively redact nested structures

    Returns:
        New list with sensitive data redacted
    """
    result = []

    for item in data:
        if isinstance(item, dict):
            result.append(redact_dict(item, deep=deep))
        elif isinstance(item, list):
            result.append(redact_list(item, deep=deep))
        elif isinstance(item, str):
            result.append(redact_string_patterns(item))
        else:
            result.append(item)

    return result


def redact_url(url: str) -> str:
    """
    Redact sensitive query parameters from a URL.

    Args:
        url: URL to redact

    Returns:
        URL with sensitive query params redacted
    """
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

    try:
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query, keep_blank_values=True)

        # Redact sensitive parameters
        redacted_params = {}
        for key, values in query_params.items():
            if is_sensitive_field(key):
                redacted_params[key] = [REDACTED]
            else:
                redacted_params[key] = [redact_string_patterns(v) for v in values]

        # Reconstruct URL
        new_query = urlencode(redacted_params, doseq=True)
        new_parsed = parsed._replace(query=new_query)

        return urlunparse(new_parsed)
    except Exception:
        # If parsing fails, return original
        return url


def redact_form_data(selector: str, text: str) -> str:
    """
    Redact text if selector suggests a sensitive field.

    Args:
        selector: CSS selector for the input field
        text: Text being entered

    Returns:
        Redacted text if field appears sensitive, original otherwise
    """
    # Check if selector contains sensitive field names
    selector_lower = selector.lower()

    for sensitive in SENSITIVE_FIELD_NAMES:
        if sensitive in selector_lower:
            return REDACTED

    # Check for common password input patterns
    if any(pattern in selector_lower for pattern in [
        "type=password", "type=\"password\"", "type='password'",
        "[type=password]", "input[type=\"password\"]"
    ]):
        return REDACTED

    return text


def redact_task_log(log_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Redact sensitive data from a task log.

    This is the main entry point for log redaction.

    Args:
        log_data: Task log dictionary

    Returns:
        Redacted log dictionary
    """
    result = redact_dict(log_data, deep=True)

    # Additional URL redaction if present
    if "plan" in result and "steps" in result.get("plan", {}):
        for step in result["plan"]["steps"]:
            if "params" in step and "url" in step["params"]:
                step["params"]["url"] = redact_url(step["params"]["url"])

    return result


class LogRedactor:
    """
    Class for configurable log redaction.
    """

    def __init__(
        self,
        additional_fields: List[str] = None,
        redact_patterns: bool = True,
        redact_urls: bool = True
    ):
        """
        Initialize log redactor.

        Args:
            additional_fields: Additional field names to consider sensitive
            redact_patterns: Whether to redact common sensitive patterns
            redact_urls: Whether to redact URL query parameters
        """
        self.sensitive_fields = SENSITIVE_FIELD_NAMES.copy()
        if additional_fields:
            self.sensitive_fields.update(f.lower() for f in additional_fields)

        self.redact_patterns = redact_patterns
        self.redact_urls = redact_urls

    def redact(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Redact sensitive data.

        Args:
            data: Data to redact

        Returns:
            Redacted data
        """
        return redact_task_log(data)
