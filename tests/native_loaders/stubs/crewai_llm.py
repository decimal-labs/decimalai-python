"""A CrewAI ``BaseLLM`` that returns scripted native tool calls and records requests.

A step is a list of OpenAI-shaped tool calls (native function calling) or a string
(the final answer).
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List

from crewai.llms.base_llm import BaseLLM


def tool_call(name: str, args: Dict[str, Any], call_id: str) -> Dict[str, Any]:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


class ScriptedLLM(BaseLLM):
    script: List[Any] = []
    requests: List[Dict[str, Any]] = []

    def supports_function_calling(self) -> bool:
        return True

    def supports_stop_words(self) -> bool:
        return False

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        self.requests.append({"messages": copy.deepcopy([dict(m) for m in messages]),
                              "tools": copy.deepcopy(tools)})
        if not self.script:
            return "FINAL: (script exhausted)"
        return self.script.pop(0)
