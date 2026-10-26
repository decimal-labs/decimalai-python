"""Nonempty, bounded error evidence without changing existing exception messages."""
from __future__ import annotations


def exception_message(error: BaseException, limit: int = 2000) -> str:
    """Keep a useful message; message-less exceptions still carry their type.

    TimeoutError and asyncio.CancelledError often have an empty string. Recording that string
    alone makes their trace indistinguishable from an error whose cause was never captured.
    A broken user-defined __str__ must not replace the original exception either.
    """
    try:
        message = str(error)
    except Exception:
        message = ""
    if message.strip():
        return message[:limit]
    return type(error).__name__[:limit]
