from agent.router import route_query, _RECORD_ID_PATTERN, PossibleRoutes
from agent.tools import check_job_application_status
from agent.state import AgentState
from agent.field_selector import select_fields
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
    route = route_query(
        state["query"],
        state.get("record_id")
    ).value
    
    return {
        "route": route
    }
    
def status_node(state: AgentState) -> dict:
    """Query application data using the current or remembered record ID."""

    match = _RECORD_ID_PATTERN.search(state["query"])

    if match:
        record_id = match.group().upper()
    else:
        record_id = state.get("record_id")

    if not record_id:
        return {
            "status_result": {
                "record_id": None,
                "error": "No application record ID was provided or remembered."
            }
        }

    record = check_job_application_status(record_id)

    return {
        "status_result": record,
        "record_id": record_id,
    }
    
def field_selector_node(state: AgentState) -> dict:
    """Determine which application fields the user requested."""

    fields = select_fields(state["query"])

    # Generic status/application query
    if not fields:
        fields = ["status"]

    return {
        "requested_fields": fields
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
    
def _format_status_response(
    status_result: dict,
    requested_fields: list[str],
) -> str:
    """Format only the application fields requested by the user."""

    record_id = status_result.get("record_id")

    if not record_id:
        return (
            "I couldn't find any application matching that record ID. "
            "Could you double-check the ID and try again?"
        )

    lines = []

    if "status" in requested_fields:
        status = status_result.get("status")
        lines.append(f"Current status: **{status}**")

    if "expected_salary_inr" in requested_fields:
        salary = status_result.get("expected_salary_inr")
        if salary is not None:
            lines.append(f"Expected salary: **₹{salary:,}**")

    if "days_since_created" in requested_fields:
        days = status_result.get("days_since_created")
        if days is not None:
            lines.append(f"Application was created **{days} days ago**.")

    if "flagged_priority_review" in requested_fields:
        flagged = status_result.get("flagged_priority_review")
        if flagged is not None:
            answer = "Yes" if flagged else "No"
            lines.append(
                f"Flagged for priority review: **{answer}**"
            )

    if "recommend_escalation" in requested_fields:
        recommend = status_result.get("recommend_escalation")

        if recommend:
            lines.append(
                "Escalation is **recommended** for this application."
            )
        else:
            lines.append(
                "Escalation is **not currently recommended** for this application."
            )

    return f"Application **{record_id}**:\n" + "\n".join(lines)


def response_node(state: AgentState) -> dict:
    """Generate the final response based on the selected route."""

    if state["route"] == PossibleRoutes.STATUS.value:
        return {
            "response": _format_status_response(
                state["status_result"],
                state.get("requested_fields", ["status"]),
            )
        }

    return {
        "response": (
            f"Following is related information for your query : "
            f"{state['rag_result']}"
        )
    }