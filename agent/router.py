from enum import Enum
import re


class PossibleRoutes(Enum):
    """Enum-like class to define possible routes for the agent."""
    RAG = "rag"
    STATUS = "status"


_RECORD_ID_PATTERN = re.compile(r"\bapp-\d{4}\b", re.IGNORECASE)

# Multi-word phrases that anchor "status" to ONE specific application.
# A bare "status" is dropped entirely -- it's the thing that was causing
# false positives on general/policy questions.
_STATUS_ANCHOR_PHRASES = (
    "status of my application",
    "status of the application",
    "status of application",
    "application status",
    "check my application",
    "check the application",
    "check my status",
    "update on my application",
)

# Phrases that signal a general/policy/FAQ question -- i.e. RAG, even if
# the word "status" or "application" happens to appear nearby.
_RAG_SIGNAL_PHRASES = (
    "in general",
    "policy",
    "process",
    "how do i apply",
    "how does",
    "what stages",
    "hiring process",
    "eligibility",
    "salary range",
)

_FOLLOW_UP_PHRASES = (
    "its status",
    "its salary",
    "its expected salary",
    "its application",
    "this application",
    "that application",
    "this status",
    "that status",
    "what about its",
    "how about its",
)


def route_query(
    query: str,
    remembered_record_id: str | None = None
) -> PossibleRoutes:

    query_lower = query.lower()

    # Explicit application ID
    if _RECORD_ID_PATTERN.search(query_lower):
        return PossibleRoutes.STATUS

    # Follow-up to a previously remembered application
    if remembered_record_id:
        if any(phrase in query_lower for phrase in _FOLLOW_UP_PHRASES):
            return PossibleRoutes.STATUS

    # General RAG / policy question
    status_score = sum(
        1 for phrase in _STATUS_ANCHOR_PHRASES
        if phrase in query_lower
    )

    rag_score = sum(
        1 for phrase in _RAG_SIGNAL_PHRASES
        if phrase in query_lower
    )

    if status_score > rag_score:
        return PossibleRoutes.STATUS

    return PossibleRoutes.RAG