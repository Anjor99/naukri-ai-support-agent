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

## Part 2 — LangGraph Agent with Tools, Memory & Guardrails

### Task 6 — Job Application Status Tool

Implemented `check_job_application_status(record_id)` in `agent/tools.py`. The tool looks up an application from the deterministic `JOB_APPLICATIONS` dataset using its `record_id` and returns the application status, expected salary, escalation score, and escalation recommendation.

#### Escalation Score

The escalation score combines the `flagged_priority_review` signal with a normalized recency signal derived from `days_since_created`.

The normalized recency score is calculated over the valid 0–30 day range:

`recency_score = (30 - days_since_created) / 30`

A value of `1.0` represents an application created today, while `0.0` represents an application created 30 days ago.

The escalation score is:

`escalation_score = 0.5 × priority_flag + 0.5 × (1 - recency_score)`

where `priority_flag` is `1.0` when `flagged_priority_review` is `True`, otherwise `0.0`. The inverse of the recency score is used because applications that have been waiting longer should contribute more to escalation urgency.

The resulting score is rounded to four decimal places and remains within the `[0, 1]` range.

#### Escalation Threshold

The escalation threshold is **0.88**.

In the generated 50-record dataset, the **80th percentile of `days_since_created` is 23 days**. Therefore, applications at or above 23 days represent approximately the oldest 20% of the generated applications.

At exactly 23 days, a priority-flagged application receives an escalation score of `0.8833`, which is just above the `0.88` threshold. In the generated dataset, **4 applications** meet both conditions: they are priority-flagged and at least 23 days old. All four receive an escalation recommendation.

#### Testing

Task 6 was tested using:

* `APP-0001` — 23-day priority application at the threshold boundary.
* `APP-0031` — older priority application with a higher escalation score.
* `APP-9999` — invalid record ID, which correctly raises `ValueError`.

These tests verify the score calculation, escalation recommendation, and invalid-record handling.


