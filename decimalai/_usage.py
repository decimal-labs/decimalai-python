"""Prompt-cache token counts in the convention LlmCallRecord keeps for a Claude call."""

from __future__ import annotations

from typing import Optional, Tuple


def anthropic_cache_split(
    total_input: Optional[int],
    cache_read: Optional[int],
    cache_creation: Optional[int],
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """``(input_tokens, cache_read_tokens, cache_creation_tokens)`` for a Claude call whose
    input count already includes the cache.

    Anthropic bills a cached prompt in three parts, and LlmCallRecord keeps them the same way:
    ``input_tokens`` is the uncached remainder, and the cache read and the cache write sit
    apart from it, each at its own price. Most frameworks hand over one input count with the
    cache already in it: LangChain's ``usage_metadata``, OpenInference's spans and the OTel
    GenAI ones (Pydantic AI's among them), ADK's ``prompt_token_count``. Recorded as it comes,
    that count bills every cached token at the full input price a second time.

    A total smaller than its own cache parts cannot include them, so it comes back as it was.
    """
    if total_input is None:
        return total_input, cache_read, cache_creation
    cached = (cache_read or 0) + (cache_creation or 0)
    if cached and total_input >= cached:
        return total_input - cached, cache_read, cache_creation
    return total_input, cache_read, cache_creation
