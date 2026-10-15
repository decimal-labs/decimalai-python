"""A Strands ``Model`` that returns scripted tool calls.

Each step is ``step(messages) -> ("tool", name, input) | ("text", str)``, so a step
can build its call from what the previous tool returned, the way a model would.
Never build a Strands ``Agent`` without a model: its default is Bedrock, which
goes to the network.
"""

from __future__ import annotations

import copy
import json
from collections.abc import AsyncGenerator, Callable
from typing import Any, Dict, List

from strands.models.model import Model

Step = Callable[[List[Dict[str, Any]]], tuple]


def last_tool_result_text(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        for block in msg.get("content", []):
            if "toolResult" in block:
                parts = []
                for c in block["toolResult"].get("content", []):
                    if "text" in c:
                        parts.append(c["text"])
                    elif "json" in c:
                        parts.append(json.dumps(c["json"]))
                return "\n".join(parts)
    return ""


class ScriptedModel(Model):
    def __init__(self, steps: List[Step]) -> None:
        self.steps = steps
        self.turn = 0
        self.seen: List[Dict[str, Any]] = []

    def update_config(self, **model_config: Any) -> None:
        pass

    def get_config(self) -> Dict[str, Any]:
        return {"model_id": "scripted-offline"}

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):  # pragma: no cover
        raise NotImplementedError
        yield  # noqa

    async def stream(self, messages, tool_specs=None, system_prompt=None, *, tool_choice=None,
                     system_prompt_content=None, **kwargs: Any) -> AsyncGenerator[Dict[str, Any], None]:
        self.seen.append({"system_prompt": system_prompt, "tool_specs": copy.deepcopy(tool_specs),
                          "messages": copy.deepcopy(messages)})
        action = self.steps[self.turn](messages) if self.turn < len(self.steps) else ("text", "(done)")
        self.turn += 1
        yield {"messageStart": {"role": "assistant"}}
        if action[0] == "tool":
            _, name, tool_input = action
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": f"call_{self.turn}",
                                                               "name": name}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(tool_input)}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockDelta": {"delta": {"text": action[1]}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        yield {"metadata": {"usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                            "metrics": {"latencyMs": 0}}}
