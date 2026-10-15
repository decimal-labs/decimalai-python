"""A LangChain chat model for deepagents that follows a script.

Each step is ``step(messages) -> AIMessage``, so a step can read what the agent was
shown (the skills listing in the system prompt, a tool result) and build its call
from that, the way a model would.
"""

from __future__ import annotations

from typing import Any, Callable, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

Step = Callable[[List[BaseMessage]], AIMessage]


class ScriptedModel(BaseChatModel):
    steps: List[Any] = []
    seen_requests: List[List[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":  # type: ignore[override]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        self.seen_requests.append(list(messages))
        i = len(self.seen_requests) - 1
        message = self.steps[i](list(messages)) if i < len(self.steps) else final("done")
        return ChatResult(generations=[ChatGeneration(message=message)])


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id,
                                              "type": "tool_call"}])


def final(text: str) -> AIMessage:
    return AIMessage(content=text)
