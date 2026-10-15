"""A google-adk ``BaseLlm`` that follows a script and keeps what it was sent.

Not LiteLLM: the native-loader check needs ADK's own skill tools, not the
conformance driver's provider path, so it runs in the openai 3 environment. Every
response carries usage metadata, or ADK logs a token-usage warning per call.
"""

from __future__ import annotations

from typing import Any, List

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types
from pydantic import PrivateAttr


def call(name: str, **args: Any) -> types.Content:
    return types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name=name,
                                                                                          args=args))])


def answer(text: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part(text=text)])


class ScriptedLlm(BaseLlm):
    _steps: List[types.Content] = PrivateAttr(default_factory=list)
    _responses: List[list] = PrivateAttr(default_factory=list)
    _instructions: List[str] = PrivateAttr(default_factory=list)

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False):
        results = []
        if llm_request.contents:
            for part in llm_request.contents[-1].parts or []:
                if part.function_response is not None:
                    results.append(part.function_response.response)
        self._responses.append(results)
        config = llm_request.config
        self._instructions.append(str(getattr(config, "system_instruction", "") or ""))
        i = len(self._responses) - 1
        step = self._steps[i] if i < len(self._steps) else answer("done")
        yield LlmResponse(
            content=step, turn_complete=True,
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=1, candidates_token_count=1, total_token_count=2),
        )
