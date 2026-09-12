from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Any

from agent.conversation import run_turn
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore


MOCK_LLM = True

REPORT_FILE = Path("reports/rag_triad_evaluation.json")

COLLECTION_NAME = "fixed_size_collection"
TOP_K = 3


# ---------------------------------------------------------------------------
# Evaluation dataset
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvaluationQuery:
    query_id: str
    topic: str
    query: str
    expected_keywords: tuple[str, ...]
    out_of_scope: bool = False


EVALUATION_QUERIES = [
    EvaluationQuery(
        "Q01",
        "job-application-eligibility",
        "What are the eligibility requirements for a job application?",
        ("eligibility", "application", "requirements"),
    ),
    EvaluationQuery(
        "Q02",
        "interview-scheduling",
        "How is an interview scheduled?",
        ("interview", "schedule", "scheduling"),
    ),
    EvaluationQuery(
        "Q03",
        "offer-negotiation-policy",
        "What is the policy for negotiating a job offer?",
        ("offer", "negotiat"),
    ),
    EvaluationQuery(
        "Q04",
        "background-verification-process",
        "What is the background verification process?",
        ("background", "verification"),
    ),
    EvaluationQuery(
        "Q05",
        "notice-period-policy",
        "What is the notice period policy?",
        ("notice", "period"),
    ),
    EvaluationQuery(
        "Q06",
        "referral-bonus-policy",
        "How does the employee referral bonus policy work?",
        ("referral", "bonus"),
    ),
    EvaluationQuery(
        "Q07",
        "internal-transfer-eligibility",
        "Who is eligible for an internal transfer?",
        ("internal", "transfer", "eligibility"),
    ),
    EvaluationQuery(
        "Q08",
        "probation-period-policy",
        "What is the probation period policy?",
        ("probation", "period"),
    ),
    EvaluationQuery(
        "Q09",
        "remote-work-eligibility",
        "Who is eligible for remote work?",
        ("remote", "work", "eligibility"),
    ),
    EvaluationQuery(
        "Q10",
        "diversity-hiring-guidelines",
        "What are the diversity hiring guidelines?",
        ("diversity", "hiring", "guidelines"),
    ),
    EvaluationQuery(
        "Q11",
        "exit-interview-process",
        "What happens during the exit interview process?",
        ("exit", "interview", "process"),
    ),
    EvaluationQuery(
        "Q12",
        "applicant-data-retention-policy",
        "How long is applicant data retained?",
        ("applicant", "data", "retention"),
    ),
    EvaluationQuery(
        "Q13",
        "out-of-scope",
        "What is the capital of France?",
        ("france", "capital"),
        out_of_scope=True,
    ),
    EvaluationQuery(
        "Q14",
        "out-of-scope",
        "How do I bake a chocolate cake?",
        ("chocolate", "cake"),
        out_of_scope=True,
    ),
    EvaluationQuery(
        "Q15",
        "edge-case",
        "Can you tell me today's weather in Mumbai?",
        ("weather", "mumbai"),
        out_of_scope=True,
    ),
]


# ---------------------------------------------------------------------------
# MOCK LLM judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """
You are an evaluation judge for a grounded Recruitment & HR support agent.

Evaluate the agent using three dimensions:

1. Context Relevance:
   Does the retrieved context contain information relevant to the user's query?

2. Groundedness:
   Is the answer supported by the retrieved context?
   The answer must not introduce unsupported facts.

3. Answer Relevance:
   Does the answer directly address the user's query?

Return a score from 0 to 1 for each dimension.

The evaluation must be based only on:
- user query
- retrieved context
- agent answer

Do not use outside knowledge.
"""


def _normalise(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())

    # Small amount of stemming so "retained" can match "retention",
    # "negotiation" can match "negotiating", etc.
    normalised = set()

    for word in words:
        normalised.add(word)

        for suffix in (
            "ing",
            "tion",
            "ions",
            "ment",
            "ments",
            "ed",
            "es",
            "s",
        ):
            if len(word) > len(suffix) + 3 and word.endswith(suffix):
                normalised.add(word[: -len(suffix)])

    return normalised


def _keyword_match_score(
    text: str,
    expected_keywords: tuple[str, ...],
) -> float:
    text_words = _normalise(text)

    matched = 0

    for keyword in expected_keywords:
        keyword_normalised = _normalise(keyword)

        if text_words.intersection(keyword_normalised):
            matched += 1

    if not expected_keywords:
        return 0.0

    return matched / len(expected_keywords)


def mock_llm_judge(
    *,
    query: str,
    context: str,
    answer: str,
    expected_keywords: tuple[str, ...],
    out_of_scope: bool,
) -> dict[str, float]:
    """
    Deterministic MOCK_LLM implementation of the RAG triad judge.

    In MOCK_LLM mode there is no external model call. The judge prompt
    is represented by JUDGE_PROMPT, while scoring is deterministic so
    evaluation results remain reproducible.
    """

    if out_of_scope:
        context_relevance = 0.0
        answer_relevance = 1.0 if (
            "don't know" in answer.lower()
            or "do not know" in answer.lower()
            or "available knowledge base" in answer.lower()
        ) else 0.0

        groundedness = answer_relevance

        return {
            "context_relevance": context_relevance,
            "groundedness": groundedness,
            "answer_relevance": answer_relevance,
        }

    context_score = _keyword_match_score(
        context,
        expected_keywords,
    )

    answer_score = _keyword_match_score(
        answer,
        expected_keywords,
    )

    # If retrieval returned relevant context and the answer uses
    # the same relevant terminology, treat it as grounded.
    groundedness = min(context_score, answer_score)

    return {
        "context_relevance": round(context_score, 4),
        "groundedness": round(groundedness, 4),
        "answer_relevance": round(answer_score, 4),
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_context(
    query: str,
    *,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """
    Retrieve top-k chunks from the same Chroma collection used by
    the fixed-size RAG pipeline.
    """

    embedding_model = EmbeddingModel()
    vector_store = VectorStore(embedding_model)

    collection = vector_store.create_collection(
        COLLECTION_NAME,
    )

    query_embedding = embedding_model.embed_text([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    chunks = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        similarity = 1.0 - float(distance)

        chunks.append(
            {
                "text": document,
                "metadata": metadata or {},
                "distance": float(distance),
                "similarity": round(similarity, 4),
            }
        )

    context = "\n\n".join(
        chunk["text"]
        for chunk in chunks
    )

    top_similarity = (
        chunks[0]["similarity"]
        if chunks
        else None
    )

    return {
        "context": context,
        "chunks": chunks,
        "top_similarity": top_similarity,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_query(
    item: EvaluationQuery,
) -> dict[str, Any]:
    """
    Run one query through the actual agent and evaluate the RAG triad.
    """

    state = run_turn(
        query=item.query,
    )

    answer = state.get("response", "")
    route = state.get("route")

    retrieval = retrieve_context(
        item.query,
        top_k=TOP_K,
    )

    context = retrieval["context"]

    scores = mock_llm_judge(
        query=item.query,
        context=context,
        answer=answer,
        expected_keywords=item.expected_keywords,
        out_of_scope=item.out_of_scope,
    )

    return {
        "query_id": item.query_id,
        "topic": item.topic,
        "query": item.query,
        "out_of_scope": item.out_of_scope,
        "route": route,
        "top_similarity": retrieval["top_similarity"],
        "retrieved_chunks": len(retrieval["chunks"]),
        "context_relevance": scores["context_relevance"],
        "groundedness": scores["groundedness"],
        "answer_relevance": scores["answer_relevance"],
        "answer": answer,
    }


def evaluate_all() -> dict[str, Any]:
    if not MOCK_LLM:
        raise RuntimeError(
            "This evaluation must run with MOCK_LLM=True."
        )

    results = []

    print("=" * 90)
    print("RAG TRIAD EVALUATION")
    print("=" * 90)

    print("\nJudge mode: MOCK_LLM")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Top-k: {TOP_K}")
    print(f"Queries: {len(EVALUATION_QUERIES)}")

    print("\nJudge prompt:")
    print(JUDGE_PROMPT.strip())

    print("\n" + "-" * 90)

    for item in EVALUATION_QUERIES:
        result = evaluate_query(item)
        results.append(result)

        print(
            f"\n{result['query_id']} | "
            f"{result['topic']}"
        )
        print(f"Query: {result['query']}")
        print(f"Route: {result['route']}")
        print(
            f"Top similarity: "
            f"{result['top_similarity']}"
        )
        print(
            "Scores: "
            f"context_relevance="
            f"{result['context_relevance']:.2f}, "
            f"groundedness="
            f"{result['groundedness']:.2f}, "
            f"answer_relevance="
            f"{result['answer_relevance']:.2f}"
        )

    context_average = mean(
        result["context_relevance"]
        for result in results
    )

    groundedness_average = mean(
        result["groundedness"]
        for result in results
    )

    answer_average = mean(
        result["answer_relevance"]
        for result in results
    )

    averages = {
        "context_relevance": round(
            context_average,
            4,
        ),
        "groundedness": round(
            groundedness_average,
            4,
        ),
        "answer_relevance": round(
            answer_average,
            4,
        ),
    }

    report = {
        "evaluation": {
            "name": "RAG Triad Evaluation",
            "judge": "MOCK_LLM",
            "collection": COLLECTION_NAME,
            "top_k": TOP_K,
            "query_count": len(results),
            "required_topic_count": 12,
            "out_of_scope_query_count": sum(
                result["out_of_scope"]
                for result in results
            ),
        },
        "judge_prompt": JUDGE_PROMPT.strip(),
        "results": results,
        "averages": averages,
    }

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 90)
    print("AVERAGES")
    print("=" * 90)

    print(
        f"Context Relevance : "
        f"{averages['context_relevance']:.4f}"
    )
    print(
        f"Groundedness      : "
        f"{averages['groundedness']:.4f}"
    )
    print(
        f"Answer Relevance  : "
        f"{averages['answer_relevance']:.4f}"
    )

    print("\nReport written to:")
    print(REPORT_FILE)

    return report


if __name__ == "__main__":
    evaluate_all()