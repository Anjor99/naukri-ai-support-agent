from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.nodes import (
    router_node,
    rag_node,
    status_node,
    response_node,
)
from agent.router import PossibleRoutes


def route_decision(state: AgentState) -> str:
    """Read the route set by router_node and pick the next node."""
    return state["route"]


def build_graph():
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("router", router_node)
    graph.add_node("rag", rag_node)
    graph.add_node("status", status_node)
    graph.add_node("response", response_node)

    # Entry point
    graph.add_edge(START, "router")

    # Conditional branch out of router
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            PossibleRoutes.RAG.value: "rag",
            PossibleRoutes.STATUS.value: "status",
        },
    )

    # Both branches converge on response
    graph.add_edge("rag", "response")
    graph.add_edge("status", "response")

    # Exit
    graph.add_edge("response", END)

    return graph.compile()


app = build_graph()