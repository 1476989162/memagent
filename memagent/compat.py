"""Compatibility helpers for pluggable responder implementations."""

from __future__ import annotations

import inspect
from typing import Any


def call_responder(
    responder,
    query: str,
    *,
    memories=None,
    persona_extras: str | None = None,
    timeout: float | None = None,
    max_tokens: int | None = None,
) -> Any:
    """Call responders without requiring every optional keyword to be supported."""

    method = responder.respond
    kwargs = {
        "memories": memories,
        "persona_extras": persona_extras,
        "timeout": timeout,
        "max_tokens": max_tokens,
    }
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        accepts_kwargs = any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
        if not accepts_kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}

    return method(query, **kwargs)
