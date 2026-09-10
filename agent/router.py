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


def route_query(query: str) -> PossibleRoutes:
    """
    Determines the route for a given query.

    Priority order:
      1. An explicit record id (APP-0001 etc.) is decisive -> STATUS.
      2. Otherwise, score status-anchor phrases vs. rag-signal phrases.
      3. Ties or no signal at all default to RAG (the safer fallback --
         a wrong RAG answer just cites the wrong doc; a wrong STATUS
         route fails outright with no record id to look up).

    Args:
        query (str): The input query string.
    """
    query_lower = query.lower()

    if _RECORD_ID_PATTERN.search(query_lower):
        return PossibleRoutes.STATUS
    
    # Disabled status queries without ids till persisted memory, they will go to rag
    # status_score = sum(1 for p in _STATUS_ANCHOR_PHRASES if p in query_lower)
    # rag_score = sum(1 for p in _RAG_SIGNAL_PHRASES if p in query_lower)

    # if status_score > rag_score:
    #     return PossibleRoutes.STATUS
    return PossibleRoutes.RAG