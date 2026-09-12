from fastapi.testclient import TestClient

from agent.api.app import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_validation() -> None:
    response = client.post("/ask", json={"query": ""})

    assert response.status_code == 422


def test_ask_real_query() -> None:
    response = client.post(
        "/ask",
        json={"query": "What is applicant data retention?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["route"] == "rag"
    assert data["response"]
    assert "retention" in data["response"].lower()


def test_add_document() -> None:
    response = client.post(
        "/add-document",
        json={
            "document_name": "api_test_policy.txt",
            "content": (
                "Document: API Test Policy\n"
                "Employees may request a four day remote work schedule "
                "after completing their probation period."
            ),
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["document_name"] == "api_test_policy.txt"
    assert data["message"] == "Document added to the knowledge base."


def test_end_to_end_add_then_ask() -> None:
    add_response = client.post(
        "/add-document",
        json={
            "document_name": "api_e2e_policy.txt",
            "content": (
                "Document: API E2E Policy\n"
                "Employees can request a four day remote work schedule "
                "after completing six months of employment."
            ),
        },
    )

    assert add_response.status_code == 200

    ask_response = client.post(
        "/ask",
        json={
            "query": "What does the API E2E Policy say about remote work?"
        },
    )

    assert ask_response.status_code == 200

    data = ask_response.json()

    assert data["route"] == "rag"
    assert "six months" in data["response"].lower()


if __name__ == "__main__":
    test_health()
    test_ask_validation()
    test_ask_real_query()
    test_add_document()
    test_end_to_end_add_then_ask()

    print("All FastAPI Task 11 tests passed.")