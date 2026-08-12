"""Replay module — re-run stale traces and link results.

Usage::

    from decimalai import replay

    # Single-turn replay
    results = replay.run(agent_fn=my_agent, agent_name="support-agent")

    # Multi-turn session replay
    results = replay.run_sessions(agent_fn=my_agent, agent_name="support-agent")

    # Manual: get prompts / sessions
    prompts = replay.get_prompts("support-agent")
    sessions = replay.get_sessions("support-agent")

    # Link traces manually
    replay.link("original-id", "replayed-id")
"""

from .tasks import (
    ReplayPrompt,
    ReplayResults,
    ReplaySession,
    ReplayTaskResult,
    ReplayTurn,
    get_prompts,
    get_sessions,
    link,
    load_agent_fn,
    run,
    run_sessions,
)

__all__ = [
    "run",
    "run_sessions",
    "get_prompts",
    "get_sessions",
    "link",
    "load_agent_fn",
    "ReplayResults",
    "ReplayTaskResult",
    "ReplayPrompt",
    "ReplaySession",
    "ReplayTurn",
]
