# Task 5 — Chunking Strategy Evaluation

## Setup

* **Queries evaluated:** 5 Task 4 in-scope queries
* **Retrieval:** Top-3 chunks
* **Evaluation:** Document-level Precision@3 and Recall@3
* Retrieved chunks were mapped to their parent `doc_id` and duplicate documents were removed before scoring.

## Per-Query Results

| Query                    |       Fixed P@3 |       Fixed R@3 |    Sentence P@3 |    Sentence R@3 |
| ------------------------ | --------------: | --------------: | --------------: | --------------: |
| Remote work              | 1/2 = **0.500** | 1/1 = **1.000** | 1/2 = **0.500** | 1/1 = **1.000** |
| Probation period         | 1/2 = **0.500** | 1/1 = **1.000** | 1/3 = **0.333** | 1/1 = **1.000** |
| Diversity & inclusion    | 1/3 = **0.333** | 1/1 = **1.000** | 1/3 = **0.333** | 1/1 = **1.000** |
| Applicant data retention | 1/2 = **0.500** | 1/1 = **1.000** | 1/2 = **0.500** | 1/1 = **1.000** |
| Offer negotiation        | 1/2 = **0.500** | 1/1 = **1.000** | 1/2 = **0.500** | 1/1 = **1.000** |
| **Average**              |       **0.467** |       **1.000** |       **0.433** |       **1.000** |

## Recommendation

Fixed-size-overlap performed slightly better, with **Precision@3 = 0.467** compared with **0.433** for sentence-based chunking. Both achieved **Recall@3 = 1.000**, so fixed-size-overlap is selected for the RAG pipeline because it provides marginally better precision while maintaining perfect recall on these five queries.
