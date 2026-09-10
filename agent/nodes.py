from agent.router import route_query, _RECORD_ID_PATTERN, PossibleRoutes
from agent.tools import check_job_application_status
from agent.state import AgentState
from rag.vector_store import VectorStore
from rag.generator import GroundedGenerator
from rag.embeddings import EmbeddingModel

def router_node(state: AgentState) -> dict:
    """Route the query to the appropriate handler.

    Args:
        state (AgentState): The current agent state.

    Returns:
        dict: The updated agent state with route.
    """
    route = route_query(state['query']).value
    
    return {
        "route": route
    }
    
def status_node(state: AgentState) -> dict:
    """Query status based on application / record id

    Args:
        state (AgentState): The current agent state.

    Returns:
        dict: The updated agent state with status_result
    """
    match = _RECORD_ID_PATTERN.search(state['query'])
    record_id = match.group()
    record = check_job_application_status(record_id)
    return {
        "status_result": record
    }
    
    
def rag_node(state: AgentState) -> dict:
    """ Retrieving chunks using rag related to query

    Args:
        state (AgentState): The current agent state.

    Returns:
        dict: The updated agent state with rag_result
    """
    vector_store = VectorStore(EmbeddingModel())
    grounded_generator = GroundedGenerator(vector_store)
    result = grounded_generator.generate(
        query=state["query"],
        collection=vector_store.create_collection("fixed_size_collection"),
        top_k=5
    )
    return {
        "rag_result": result
    }
    
def _format_status_response(status_result: dict) -> str:
    """Convert a status_result dict into a natural language response."""
    record_id = status_result.get("record_id")
    status = status_result.get("status")
    salary = status_result.get("expected_salary_inr")
    recommend_escalation = status_result.get("recommend_escalation")

    if not record_id:
        return "I couldn't find any application matching that record ID. Could you double-check the ID and try again?"

    lines = [f"Here's the status for application **{record_id}**:"]
    lines.append(f"- Current status: **{status}**")

    if salary is not None:
        lines.append(f"- Expected salary: ₹{salary:,}")

    if recommend_escalation:
        lines.append(
            "- This application has been flagged for escalation based on its escalation score, "
            "so it's being prioritized for review."
        )
    else:
        lines.append("- No escalation is currently recommended for this application.")

    return "\n".join(lines)


def response_node(state: AgentState) -> dict:
    """

    Args:
        state (AgentState): _description_

    Returns:
        dict: _description_
    """
    if state["route"] == PossibleRoutes.STATUS.value:
        return {
            "response": _format_status_response(state["status_result"])
        }
    else:
        return {
            "response": f"Following is related information for your query : {state['rag_result']}"
        }