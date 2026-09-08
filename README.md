# Naukri.com — AI Recruitment & HR Support Agent

## Part 1 — Dataset Design & RAG Core

### Task 1 — Dataset

Created a seeded, deterministic job-application dataset in `dataset.py` with **50 records**.

Each record contains:

* `record_id`
* `category`
* `status`
* `expected_salary_inr`
* `days_since_created`
* `flagged_priority_review`

The generator validates category/status coverage and the required priority-review percentage.

### Task 2 — Knowledge Base

Created **12 policy documents** covering all required HR/recruitment topics in the `knowledge_base/` directory.

### Task 3 — RAG Indexing

Implemented two chunking strategies:

* **Fixed-size with overlap:** 250-character chunks, 50-character overlap
* **Sentence-based:** 2 sentences per chunk

Chunks are embedded locally using `all-MiniLM-L6-v2` and stored in separate persistent ChromaDB collections:

* `fixed_size_collection`
* `sentence_based_collection`

### Task 4 — Grounded Generation

Implemented similarity-based retrieval gating with an empirically calibrated threshold.

* In-scope similarity range: **0.4859–0.7930**
* Out-of-scope similarity range: **-0.0160–0.0733**
* Selected threshold: **0.30**

Queries below the threshold return an "I don't know" fallback.

The generator supports both deterministic `MOCK_LLM` execution and an optional Groq-based real LLM.

Detailed calibration results and measured values are documented in [`evaluation/calibration.md`](evaluation/caliberation_results.md).

### Task 5 — Chunking Evaluation

Evaluated both chunking strategies on the same five Task 4 queries using document-level Precision@3 and Recall@3.

| Strategy           | Precision@3 |  Recall@3 |
| ------------------ | ----------: | --------: |
| Fixed-size-overlap |   **0.467** | **1.000** |
| Sentence-based     |   **0.433** | **1.000** |

Fixed-size-overlap was selected because it achieved slightly higher Precision@3 while maintaining the same perfect Recall@3.

Detailed per-query calculations and retrieval results are available in [`evaluation/chunking_evaluation.md`](evaluation/chunking_evaluation.md).
