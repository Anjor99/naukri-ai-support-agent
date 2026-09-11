"""
Input-side guardrails.

Guardrails:
1. Fixed-format PII masking (email + Indian phone number).
2. Prompt-injection detection using LLM Guard.
3. Toxic-input detection using Detoxify.

The public entry point is `guard_input()`.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from detoxify import Detoxify
from llm_guard import scan_prompt
from llm_guard.input_scanners import PromptInjection


PROMPT_INJECTION_REPLY = (
    "For security reasons, I can't process that request. Sorry."
)

TOXIC_INPUT_REPLY = (
    "I am unable to process this request. Please contact customer care."
)


EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)


# Indian mobile number formats:
# +91 98765 43210
# +91-9876543210
# 9876543210
INDIAN_PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"
)


@lru_cache(maxsize=1)
def _prompt_injection_scanner() -> PromptInjection:
    """Load the prompt-injection scanner once."""
    return PromptInjection(threshold=0.5)


@lru_cache(maxsize=1)
def _toxicity_model() -> Detoxify:
    """Load Detoxify once and reuse it."""
    return Detoxify("original")


def mask_pii(text: str) -> tuple[str, list[str]]:
    """
    Mask fixed-format PII.

    Returns:
        (sanitized_text, detected_pii_types)
    """

    detected: list[str] = []

    def mask_email(match: re.Match[str]) -> str:
        if "email" not in detected:
            detected.append("email")

        return "[EMAIL_REDACTED]"

    def mask_phone(match: re.Match[str]) -> str:
        if "phone" not in detected:
            detected.append("phone")

        return "[PHONE_REDACTED]"

    sanitized = EMAIL_PATTERN.sub(mask_email, text)

    sanitized = INDIAN_PHONE_PATTERN.sub(
        mask_phone,
        sanitized,
    )

    return sanitized, detected


def detect_prompt_injection(text: str) -> dict[str, Any]:
    """
    Detect prompt injection using the installed LLM Guard API.

    The installed version returns:
        (sanitized_prompt, is_valid)

    It does not return a separate risk score.
    """

    scanner = _prompt_injection_scanner()

    sanitized_text, valid = scanner.scan(text)

    return {
        "detected": not valid,
        "valid": valid,
        "sanitized_text": sanitized_text,
    }

def detect_toxic_input(
    text: str,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """
    Detect toxic input using Detoxify.
    """

    scores = _toxicity_model().predict(text)

    toxicity = float(
        scores.get("toxicity", 0.0)
    )

    return {
        "detected": toxicity >= threshold,
        "toxicity": toxicity,
        "threshold": threshold,
        "scores": {
            key: float(value)
            for key, value in scores.items()
        },
    }


def guard_input(
    text: str,
    toxicity_threshold: float = 0.50,
) -> dict[str, Any]:
    """
    Apply all input-side guardrails.

    Processing order:

    1. Mask PII.
    2. Check prompt injection.
    3. Check toxicity.

    PII does not block the request.
    Prompt injection and toxicity block the request.
    """

    sanitized_text, pii_types = mask_pii(text)

    injection = detect_prompt_injection(
        sanitized_text
    )

    if injection["detected"]:
        return {
            "allowed": False,
            "query": sanitized_text,
            "reason": "prompt_injection",
            "response": PROMPT_INJECTION_REPLY,
            "pii_detected": pii_types,
            "prompt_injection": injection,
            "toxicity": None,
        }

    toxicity = detect_toxic_input(
        sanitized_text,
        threshold=toxicity_threshold,
    )

    if toxicity["detected"]:
        return {
            "allowed": False,
            "query": sanitized_text,
            "reason": "toxic_input",
            "response": TOXIC_INPUT_REPLY,
            "pii_detected": pii_types,
            "prompt_injection": injection,
            "toxicity": toxicity,
        }

    return {
        "allowed": True,
        "query": sanitized_text,
        "reason": None,
        "response": None,
        "pii_detected": pii_types,
        "prompt_injection": injection,
        "toxicity": toxicity,
    }