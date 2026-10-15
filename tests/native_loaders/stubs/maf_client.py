"""A chat client for agent-framework-core that returns scripted function calls.

Built from the framework's own public layers (FunctionInvocationLayer outermost,
then ChatMiddlewareLayer, ChatTelemetryLayer, BaseChatClient), so the real
function-invocation loop, approval flow and middleware run as they would against
a provider. Each model call pops the next step:

    ("calls", [(name, {args}), ...])   an assistant message with function calls
    ("text", "...")                    a final answer

Every request the framework sends is kept in ``requests``.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any, List, Tuple

from agent_framework import (
    BaseChatClient,
    ChatMiddlewareLayer,
    ChatResponse,
    Content,
    FunctionInvocationLayer,
    Message,
)
from agent_framework.observability import ChatTelemetryLayer

_ids = itertools.count(1)


class ScriptedChatClient(FunctionInvocationLayer, ChatMiddlewareLayer, ChatTelemetryLayer,
                         BaseChatClient):
    OTEL_PROVIDER_NAME = "scripted"

    def __init__(self, steps: Sequence[Tuple[str, Any]], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._steps = list(steps)
        self.requests: List[dict] = []

    async def _inner_get_response(  # type: ignore[override]
        self, *, messages: Sequence[Message], stream: bool, options: Mapping[str, Any],
        **kwargs: Any,
    ) -> ChatResponse:
        if stream:
            raise NotImplementedError("the check runs non-streaming only")
        self.requests.append({"messages": list(messages), "options": dict(options)})
        if not self._steps:
            return ChatResponse(messages=[Message(role="assistant", contents=["(script exhausted)"])])
        kind, payload = self._steps.pop(0)
        if kind == "text":
            return ChatResponse(messages=[Message(role="assistant", contents=[payload])],
                                response_id=f"resp-{next(_ids)}", model="scripted-model")
        contents = [Content.from_function_call(call_id=f"call_{next(_ids)}", name=name, arguments=args)
                    for name, args in payload]
        return ChatResponse(messages=[Message(role="assistant", contents=contents)],
                            response_id=f"resp-{next(_ids)}", model="scripted-model")


def function_results(messages: Sequence[Message]) -> List[Tuple[str, Any]]:
    """``(call_id, result)`` for every function result in ``messages``."""
    out = []
    for m in messages:
        for c in m.contents:
            if c.type == "function_result":
                out.append((c.call_id, c.result if c.exception is None else f"EXCEPTION: {c.exception}"))
    return out
