from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.guardrails.input_guardrails import mask_pii


LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "requests.jsonl"


def log_request(
    *,
    trace_id: str,
    query: str,
    route: str,
    status_code: int,
    elapsed_ms: float,
) -> None:
    """
    Write exactly one PII-safe JSON object per request.
    """
    sanitized_query, pii_types = mask_pii(query)

    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "query": sanitized_query,
        "route": route,
        "status_code": status_code,
        "elapsed_ms": elapsed_ms,
        "pii_detected": pii_types,
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")