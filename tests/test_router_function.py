from agent.router import route_query, PossibleRoutes

queries = {
    # Explicit application IDs → STATUS
    "What is the status of APP-0001?": PossibleRoutes.STATUS,
    "Can you tell me about APP-0031?": PossibleRoutes.STATUS,
    "Check app-0040 for me.": PossibleRoutes.STATUS,
    "What happened with APP-9999?": PossibleRoutes.STATUS,

    # Application-specific queries without an ID → RAG
    "What is the status of my application?": PossibleRoutes.RAG,
    "What is the status of the application?": PossibleRoutes.RAG,
    "Check my application for me.": PossibleRoutes.RAG,
    "Check the application for me.": PossibleRoutes.RAG,
    "Can you check my status?": PossibleRoutes.RAG,
    "Can you give me an update on my application?": PossibleRoutes.RAG,

    # General knowledge / policy questions → RAG
    "What is the remote work policy?": PossibleRoutes.RAG,
    "How does interview scheduling work?": PossibleRoutes.RAG,
    "What is the notice period policy?": PossibleRoutes.RAG,
    "What is the offer negotiation policy?": PossibleRoutes.RAG,
    "What is the background verification process?": PossibleRoutes.RAG,
    "What are the eligibility requirements for a job application?": PossibleRoutes.RAG,
    "How does the hiring process work?": PossibleRoutes.RAG,
    "What stages does a job application go through?": PossibleRoutes.RAG,
    "What is the application status policy?": PossibleRoutes.RAG,
    "What is the escalation process?": PossibleRoutes.RAG,
    "What does priority review mean?": PossibleRoutes.RAG,

    # General/ambiguous queries → RAG
    "I need help with my application.": PossibleRoutes.RAG,
    "Tell me about applications.": PossibleRoutes.RAG,
    "How do I apply for a job?": PossibleRoutes.RAG,
    "Can you explain the recruitment process?": PossibleRoutes.RAG,
    "How does the system work?": PossibleRoutes.RAG,
    "I need some information about hiring.": PossibleRoutes.RAG,
}

for query, expected_route in queries.items():
    route = route_query(query)
    assert route == expected_route, f"Expected {expected_route}, but got {route} for query: '{query}'"
    print(f"Query: '{query}' correctly routed to {route.value}.")