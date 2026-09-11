"""
Output-side guardrails.

Guardrails:
1. Groundedness check using the calibrated RAG similarity threshold.
2. Toxicity detection using LLM Guard.

The public entry point is `guard_output()`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from llm_guard import scan_output
from llm_guard.output_scanners import Toxicity


UNGROUNDED_REPLY = (
    "I don't know based on the available knowledge base."
)

UNSAFE_OUTPUT_REPLY = (
    "Sorry, I cannot answer."
)


# Calibrated earlier from the RAG retrieval scores.
DEFAULT_GROUNDEDNESS_THRESHOLD = 0.30


@lru_cache(maxsize=1)
def _output_scanners() -> tuple:
    """Load output scanners once and reuse them."""

    return (
        Toxicity(threshold=0.70),
    )


def check_groundedness(
    top_similarity: float | None,
    threshold: float = DEFAULT_GROUNDEDNESS_THRESHOLD,
) -> dict[str, Any]:
    """
    Check whether retrieved evidence is strong enough
    to support the generated response.
    """

    if top_similarity is None:
        return {
            "grounded": False,
            "top_similarity": None,
            "threshold": threshold,
        }

    similarity = float(top_similarity)

    return {
        "grounded": similarity >= threshold,
        "top_similarity": similarity,
        "threshold": threshold,
    }


def scan_output_safety(
    prompt: str,
    output: str,
) -> dict[str, Any]:
    """
    Scan generated output for toxicity using LLM Guard.
    """

    sanitized_output, valid = scan_output(
        list(_output_scanners()),
        prompt,
        output,
    )

    blocked = not all(valid.values())

    return {
        "allowed": not blocked,
        "output": sanitized_output,
        "valid": valid,
    }


def guard_output(
    prompt: str,
    output: str,
    *,
    top_similarity: float | None = None,
    groundedness_threshold: float = (
        DEFAULT_GROUNDEDNESS_THRESHOLD
    ),
) -> dict[str, Any]:
    """
    Apply output-side guardrails.

    Groundedness is checked first.

    If the retrieved context does not support
    the answer, return the required fallback.
    """

    groundedness = check_groundedness(
        top_similarity,
        threshold=groundedness_threshold,
    )

    # Grounding failure takes priority.
    if not groundedness["grounded"]:
        return {
            "allowed": False,
            "response": UNGROUNDED_REPLY,
            "reason": "ungrounded",
            "groundedness": groundedness,
            "safety": None,
        }

    safety = scan_output_safety(
        prompt,
        output,
    )

    if not safety["allowed"]:
        return {
            "allowed": False,
            "response": UNSAFE_OUTPUT_REPLY,
            "reason": "unsafe_output",
            "groundedness": groundedness,
            "safety": safety,
        }

    return {
        "allowed": True,
        "response": safety["output"],
        "reason": None,
        "groundedness": groundedness,
        "safety": safety,
    }