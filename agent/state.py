from typing import TypedDict

class AgentState(TypedDict):
    """
    State carried through the LangGraph agent.
    """
    query: str
    route: str
    rag_result: str
    status_result: dict
    response: str