from dataset import JOB_APPLICATIONS


def check_job_application_status(record_id: str) -> dict:
    record = next(
        (a for a in JOB_APPLICATIONS if a["record_id"] == record_id),
        None
    )

    if record is None:
        raise ValueError(f"No application found with record_id: {record_id}")

    flag_component = 1.0 if record["flagged_priority_review"] else 0.0

    # Normalized recency over the valid 0–30 day range.
    # 1.0 = created today, 0.0 = created 30 days ago.
    recency_score = (30 - record["days_since_created"]) / 30

    escalation_score = round(
        0.5 * flag_component + 0.5 * (1.0 - recency_score),
        4
    )

    return {
        "record_id": record["record_id"],
        "status": record["status"],
        "expected_salary_inr": record["expected_salary_inr"],
        "escalation_score": escalation_score,
        "recommend_escalation": escalation_score >= 0.88,
    }