"""Small provider-error policy for opt-in, non-streaming model request recovery."""
from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

logger = logging.getLogger("decimalai.provider_recovery")
_BILLING_MARKERS = (
    "prepayment credits are depleted", "insufficient_quota", "billing_disabled",
    "billing_not_active", "billing_hard_limit_reached", "billing account is disabled",
    "billing account is not enabled",
)


def _failure(exc: Exception) -> tuple[str, float | None]:
    chain = []
    current: BaseException | None = exc
    while current is not None and all(current is not e for e in chain):
        chain.append(current)
        current = current.__cause__ or current.__context__
    text = " ".join(str(e).lower() for e in chain)
    if any(marker in text for marker in _BILLING_MARKERS):
        return "billing", None
    delay = None
    for e in chain:
        headers = getattr(e, "headers", None) or getattr(getattr(e, "response", None), "headers", None)
        if not headers:
            continue
        raw = next((v for k, v in headers.items() if k.lower() == "retry-after"), None)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (ValueError, TypeError, OverflowError):
            try:
                date = parsedate_to_datetime(raw)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                value = (date - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                continue
        if math.isfinite(value):
            delay = max(0.0, value)
            break
    for e in reversed(chain):
        status = getattr(e, "status_code", None) or getattr(e, "code", None)
        status = status or getattr(getattr(e, "response", None), "status_code", None)
        if isinstance(status, int):
            if status == 429:
                return "throttle", delay
            return ("transient" if status in (408, 500, 502, 503, 504) else "permanent"), delay
    if any(isinstance(e, (TimeoutError, ConnectionError, httpx.TransportError)) for e in chain):
        return "transient", delay
    return "permanent", delay


def retry_model_requests(model: Any, *, max_attempts: int = 3, timeout_s: float = 30.0) -> Any:
    """Wrap a Pydantic AI model's non-streaming requests in bounded recovery.

    Tools and the agent loop are never retried. Each request has one total time
    budget, including provider SDK retries, waits and all our attempts. Attempt
    counts describe model.request calls; a provider SDK may make multiple HTTP
    attempts inside one call. Streaming is delegated unchanged: emitted chunks
    cannot safely be replayed. Failures and cancellation still propagate.

    Import Pydantic AI only when used so the SDK's core install stays thin.
    """
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be positive and finite")

    from opentelemetry import trace
    from pydantic_ai import models
    from pydantic_ai.models.wrapper import WrapperModel

    class RequestRetryModel(WrapperModel):
        async def request(self, *args: Any, **kwargs: Any) -> Any:
            deadline = time.monotonic() + timeout_s
            for attempt in range(1, max_attempts + 1):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Model request recovery time budget exhausted")
                span = trace.get_current_span()
                try:
                    result = await asyncio.wait_for(
                        self.wrapped.request(*args, **kwargs), timeout=remaining,
                    )
                except Exception as exc:
                    kind, retry_after = _failure(exc)
                    delay = max(retry_after or 0.0, random.uniform(0.25, 0.5) * 2 ** (attempt - 1))
                    retry = (kind in ("throttle", "transient") and attempt < max_attempts
                             and delay + 0.05 < deadline - time.monotonic())
                    outcome = "retry" if retry else "failed"
                    span.add_event("decimalai.model_request", {
                        "attempt": attempt, "outcome": outcome, "failure_kind": kind,
                    })
                    logger.warning("model_request attempt=%s outcome=%s kind=%s", attempt, outcome, kind)
                    if not retry:
                        raise
                    await asyncio.sleep(delay)
                else:
                    span.add_event("decimalai.model_request", {
                        "attempt": attempt, "outcome": "success" if attempt == 1 else "recovered",
                    })
                    return result

    return RequestRetryModel(models.infer_model(model))
