"""The live native check's scenario side: what a real model is asked, which model, and how a
provider's billing wall is told apart from a failure.

Each framework's scenario module has a ``live(ctx)`` function that runs ``ctx["trials"]``
independent trials, each one :func:`column.Session.run` of a FRESH agent built with the
framework's own loader (its DecimalAI helper) and a real model, asked :data:`QUESTION` with
:data:`INSTRUCTIONS` and nothing that names the skill. Each trial records every tool call the
model made (``run.called``), what the model was given (``run.saw``) and its answer
(``run.answered``). ``live_contract`` grades them. No assertion lives here.
"""

from __future__ import annotations

import os
from typing import Optional

#: What the agent is told. It does not name the skill: the model must find it in the loader's
#: own listing, which is the behaviour under test.
INSTRUCTIONS = ("You are a customer support agent for Acme. Use your skills when one applies "
                "to the customer's question, and follow it.")
#: A question the refund-policy skill answers (31-60 days: store credit) and a model cannot
#: answer from general knowledge.
QUESTION = ("A customer bought headphones from us for $100 and asks for a refund 45 days "
            "after purchase. What does our refund policy allow?")
#: The fact the answer must carry when the model used the skill.
ANSWER_FACT = "store credit"

DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"   # the release gate's budget tier


def openai_model() -> str:
    return os.environ.get("LIVE_LLM_OPENAI_MODEL") or DEFAULT_OPENAI_MODEL


def trials(ctx: dict) -> int:
    return int(ctx.get("trials") or 3)


#: Error text that means the provider refused for money or rate, not that the code failed.
_PROVIDER_WALLS = ("insufficient_quota", "credit balance is too low", "billing",
                   "rate limit", "rate_limit", "429", "quota", "RESOURCE_EXHAUSTED")


def provider_wall(error: Optional[str]) -> bool:
    return bool(error) and any(marker.lower() in error.lower() for marker in _PROVIDER_WALLS)
