from __future__ import annotations

import json

from agent.api.logging_utils import LOG_FILE, log_request


def test_logging_masks_pii() -> None:
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    log_request(
        trace_id="test-trace-001",
        query=(
            "My email is demo@example.com and my phone is "
            "+91 98765 43210. What is the notice period?"
        ),
        route="rag",
        status_code=200,
        elapsed_ms=12.34,
    )

    lines = LOG_FILE.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1

    record = json.loads(lines[0])

    assert record["trace_id"] == "test-trace-001"
    assert "[EMAIL_REDACTED]" in record["query"]
    assert "[PHONE_REDACTED]" in record["query"]

    assert "demo@example.com" not in lines[0]
    assert "+91 98765 43210" not in lines[0]

    print("Logging PII test passed.")
    print(record)


if __name__ == "__main__":
    test_logging_masks_pii()