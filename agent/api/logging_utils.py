from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.guardrails.input_guardrails import mask_pii


LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "requests.jsonl"


def _sanitize_value(value: Any) -> Any:
    """
    Recursively PII-mask string values before they are written to disk.
    """
    if isinstance(value, str):
        sanitized, _ = mask_pii(value)
        return sanitized

    if isinstance(value, dict):
        return {
            str(key): _sanitize_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]

    if isinstance(value, tuple):
        return [_sanitize_value(item) for item in value]

    return value


def _detect_pii(value: Any) -> list[str]:
    """
    Return unique PII types found in any textual value.
    """
    detected: set[str] = set()

    def inspect(item: Any) -> None:
        if isinstance(item, str):
            _, pii_types = mask_pii(item)
            detected.update(pii_types)

        elif isinstance(item, dict):
            for nested in item.values():
                inspect(nested)

        elif isinstance(item, (list, tuple)):
            for nested in item:
                inspect(nested)

    inspect(value)

    return sorted(detected)


def log_event(
    *,
    conversation_id: str | None,
    trace_id: str,
    event: str,
    step: str,
    elapsed_ms: float | None = None,
    **fields: Any,
) -> None:
    """
    Write exactly one structured JSON object per line.

    All textual fields are recursively PII-masked before being
    persisted to disk.
    """

    sanitized_fields = _sanitize_value(fields)

    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "conversation_id": conversation_id,
        "trace_id": trace_id,
        "event": event,
        "step": step,
        "elapsed_ms": elapsed_ms,
        **sanitized_fields,
        "pii_detected": _detect_pii(fields),
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )