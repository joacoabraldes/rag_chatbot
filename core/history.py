# -*- coding: utf-8 -*-
"""Chat history persistence and token budget helpers."""

import os, json, datetime


def now_iso() -> str:
    return datetime.datetime.now().isoformat()


def save_history(path: str, record: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1 token per 4 characters for mixed Spanish/English."""
    return max(1, len(text) // 4)


def trim_history_to_budget(history: list, token_budget: int = 2000) -> list:
    """Return the most recent history messages that fit within token_budget."""
    total = 0
    result = []
    for msg in reversed(history):
        tokens = estimate_tokens(msg.get("content", ""))
        if total + tokens > token_budget:
            break
        result.insert(0, msg)
        total += tokens
    return result
