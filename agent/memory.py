"""
agent/memory.py

Simple JSON-file persistence layer for conversation state.

This module is intentionally standalone - it knows nothing about
LangGraph, AgentState, or app.invoke(). It just knows how to take a
Python dict, write it to disk, read it back, and reset it. Wiring this
into the graph (e.g. as a checkpointer or as pre/post-invoke hooks)
happens in a later step.

    save_conversation(state)   -> writes `state` to memory.json
    load_conversation()        -> reads memory.json back into a dict
    clear_conversation()       -> resets memory.json to a fresh/empty state
"""
import json
from pathlib import Path
from typing import Any, Dict

# Default location for the persisted conversation state.
MEMORY_FILE = Path("data/memory.json")


def save_conversation(state: Dict[str, Any], filepath: str | Path = MEMORY_FILE) -> None:
    """Persist the given conversation state to a JSON file.

    Args:
        state (Dict[str, Any]): The conversation/agent state to persist.
            Must be JSON-serializable (plain dicts, lists, str, int,
            float, bool, None).
        filepath (str | Path): Where to write the state. Defaults to
            MEMORY_FILE ("memory.json" in the current working directory).

    Raises:
        TypeError: If `state` contains values that can't be JSON-serialized.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with filepath.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def load_conversation(filepath: str | Path = MEMORY_FILE) -> Dict[str, Any]:
    """Load conversation state from a JSON file.

    Args:
        filepath (str | Path): Where to read the state from. Defaults to
            MEMORY_FILE ("memory.json" in the current working directory).

    Returns:
        Dict[str, Any]: The loaded state, or an empty dict if the file
        doesn't exist yet (e.g. first run, nothing saved so far).

    Raises:
        json.JSONDecodeError: If the file exists but contains invalid JSON.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        return {}

    with filepath.open("r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        # File exists but is empty - treat like "nothing saved yet"
        # rather than raising a JSONDecodeError on empty string.
        return {}

    return json.loads(content)


def clear_conversation(filepath: str | Path = MEMORY_FILE) -> Dict[str, Any]:
    """Reset conversation state to a fresh, empty state.

    Overwrites the file with an empty JSON object rather than deleting
    it, so callers can always assume the file exists after this call.

    Args:
        filepath (str | Path): Which memory file to clear. Defaults to
            MEMORY_FILE ("memory.json" in the current working directory).

    Returns:
        Dict[str, Any]: The fresh, empty state (always `{}`), for
        convenience if the caller wants to immediately use it as their
        new in-memory state.
    """
    fresh_state: Dict[str, Any] = {}
    save_conversation(fresh_state, filepath=filepath)
    return fresh_state