"""
agent/field_selector.py

Deterministically identifies which application fields the user is asking for.
Returns a list because a query can request multiple fields.
"""

from typing import List


FIELD_PHRASES = {
    "status": (
        "status",
        "current status",
        "application status",
    ),
    "expected_salary_inr": (
        "salary",
        "expected salary",
        "salary expectation",
        "expected compensation",
    ),
    "days_since_created": (
        "days since created",
        "how old",
        "how long ago",
        "application age",
    ),
    "recommend_escalation": (
        "should it be escalated",
        "should this be escalated",
        "recommend escalation",
        "escalation recommend",
        "escalation"
    ),
}


def select_fields(query: str) -> List[str]:
    """
    Identify application fields requested by the user.

    Returns:
        List of field names. Multiple fields are returned when
        the query asks for multiple pieces of information.
    """

    query_lower = query.lower()
    selected_fields = []

    for field, phrases in FIELD_PHRASES.items():
        if any(phrase in query_lower for phrase in phrases):
            selected_fields.append(field)

    return selected_fields