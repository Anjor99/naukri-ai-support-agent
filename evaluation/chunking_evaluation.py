"""
evaluation/chunking_evaluation.py

Step 1: prints top-3 retrieval (chunk_id -> parent doc_id -> similarity)
for the same 5 queries against both collections, so retrieval quality
can be sanity-checked by eye.

Step 2 (added below): Precision@3 and Recall@3 at the DOCUMENT level for
those same 5 queries, computed separately for each collection. Chunks
are mapped back to their parent doc_id via metadata and DEDUPED before
scoring, per-query arithmetic is printed, and both collections'
per-query results plus averages are shown side by side at the end.
"""

import sys
import os

# Allow running as `python evaluation/chunking_evaluation.py` from the
# project root without needing to install the package. This MUST run
# before any `rag.*` imports below, or they'll fail with
# ModuleNotFoundError when this script is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.vector_store import VectorStore
from rag.chunking import ChunkingStrategy
from rag.embeddings import EmbeddingModel
from settings import settings

DB_PATH = settings.chroma_db_path
TOP_K = 3

FIXED_COLLECTION_NAME = "fixed_size_collection"
SENTENCE_COLLECTION_NAME = "sentence_based_collection"

# --- Task 4/5 queries -------------------------------------------------
# Each query maps to the doc_id we EXPECT to be the top parent document.
# Replace these with your real 12-doc knowledge base's queries/doc_ids.
QUERIES_WITH_EXPECTED_DOC_ID = {
    "Who is eligible for remote work?": "remote_work",
    "What is the probation period of the company?": "probation_period",
    "What is the diversity and inclusion policy?": "diversity_hiring",
    "What is the applicant data retention policy?": "applicant_data_retention",
    "What is the offer negotiation process?": "offer_negotiation",
}

# --- Ground truth for Precision@3/Recall@3 -----------------------------
# A query can legitimately be relevant to MORE than one document — if so,
# list every relevant doc_id here. Defaults to a single-doc set built
# from QUERIES_WITH_EXPECTED_DOC_ID above so nothing needs to be repeated
# for the common case, but you can override any query with an explicit
# set of doc_ids if it should map to several documents.
QUERIES_WITH_RELEVANT_DOC_IDS = {
    query: {expected_doc_id}
    for query, expected_doc_id in QUERIES_WITH_EXPECTED_DOC_ID.items()
}


def dedup_preserve_order(doc_ids):
    seen = []
    for d in doc_ids:
        if d not in seen:
            seen.append(d)
    return seen


def precision_recall_at_k(retrieved_doc_ids, relevant_doc_ids):
    """
    retrieved_doc_ids: doc_ids from top-k chunks, IN ORDER, before dedup.
    Dedup happens here, preserving first-seen order.
    """
    deduped = dedup_preserve_order(retrieved_doc_ids)
    if not deduped:
        return 0.0, 0.0

    relevant_retrieved = [d for d in deduped if d in relevant_doc_ids]
    precision = len(relevant_retrieved) / len(deduped)
    recall = len(relevant_retrieved) / len(relevant_doc_ids) if relevant_doc_ids else 0.0
    return precision, recall, deduped, relevant_retrieved


def main():
    vector_store = VectorStore(embedding_model=EmbeddingModel())
    fixed_collection = vector_store.client.get_collection(FIXED_COLLECTION_NAME)
    sentence_collection = vector_store.client.get_collection(SENTENCE_COLLECTION_NAME)

    collections = [
        (FIXED_COLLECTION_NAME, fixed_collection),
        (SENTENCE_COLLECTION_NAME, sentence_collection),
    ]

    # --- Step 1: raw retrieval printout (unchanged) ---
    for query, expected_doc_id in QUERIES_WITH_EXPECTED_DOC_ID.items():
        print(f"\nQuery: {query}")
        print(f"Expected top doc_id: {expected_doc_id}\n")

        for collection_name, collection in collections:
            results = vector_store.search(collection, query, top_k=TOP_K)
            print(f"Collection: {collection_name}")
            for i in range(TOP_K):
                chunk_id = results["ids"][0][i]
                similarity = 1 - results["distances"][0][i]
                chunk_metadata = results["metadatas"][0][i]
                parent_doc_id = chunk_metadata.get("doc_id", "N/A")
                print(f"  {i+1}. Chunk ID: {chunk_id} -> Parent Doc ID: {parent_doc_id} -> Similarity: {similarity:.4f}")

    # --- Step 2: Precision@3 / Recall@3, document-level, per collection ---
    print("\n" + "=" * 70)
    print(f"PRECISION@{TOP_K} / RECALL@{TOP_K} (document-level, deduped)")
    print("=" * 70)

    collection_averages = {}

    for collection_name, collection in collections:
        print(f"\n--- {collection_name} ---")
        precisions, recalls = [], []

        for query, relevant_doc_ids in QUERIES_WITH_RELEVANT_DOC_IDS.items():
            results = vector_store.search(collection, query, top_k=TOP_K)
            retrieved_doc_ids = [meta.get("doc_id", "N/A") for meta in results["metadatas"][0]]

            precision, recall, deduped, relevant_retrieved = precision_recall_at_k(
                retrieved_doc_ids, relevant_doc_ids
            )
            precisions.append(precision)
            recalls.append(recall)

            print(f"\nQuery: {query!r}")
            print(f"  retrieved (top-{TOP_K}, deduped docs): {deduped}")
            print(f"  relevant (ground truth):               {sorted(relevant_doc_ids)}")
            print(f"  relevant ∩ retrieved:                   {relevant_retrieved}")
            print(f"  Precision@{TOP_K} = {len(relevant_retrieved)}/{len(deduped)} = {precision:.3f}")
            print(f"  Recall@{TOP_K}    = {len(relevant_retrieved)}/{len(relevant_doc_ids)} = {recall:.3f}")

        avg_p = sum(precisions) / len(precisions)
        avg_r = sum(recalls) / len(recalls)
        collection_averages[collection_name] = (avg_p, avg_r)
        print(f"\n  {collection_name} AVERAGES -> Precision@{TOP_K}: {avg_p:.3f}  Recall@{TOP_K}: {avg_r:.3f}")

    # --- Final side-by-side comparison ---
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    for collection_name, (avg_p, avg_r) in collection_averages.items():
        print(f"{collection_name:28s}  Precision@{TOP_K}={avg_p:.3f}  Recall@{TOP_K}={avg_r:.3f}")


if __name__ == "__main__":
    main()