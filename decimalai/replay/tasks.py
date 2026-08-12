"""Replay module — re-run stale traces and link results.

Provides three user-facing functions:

- ``run()`` — Automated replay: fetches prompts, runs agent, submits results
- ``get_prompts()`` — Export prompts for manual replay
- ``link()`` — Manually link a replayed trace to its original

Usage::

    from decimalai import replay

    # Automated replay
    results = replay.run(
        agent_fn=my_agent,
        agent_name="support-agent",
    )

    # Manual: get prompts to replay yourself
    prompts = replay.get_prompts("support-agent")

    # Manual: link a new trace to the original
    replay.link("original-trace-id", "replayed-trace-id")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("decimalai.replay")


# ── Result Types ──────────────────────────────────────────────────


@dataclass
class ReplayTaskResult:
    """Result of a single replay task."""

    original_trace_id: str
    replayed_trace_id: Optional[str] = None
    user_input: str = ""
    status: str = "pending"  # completed / failed / skipped
    eval_score: Optional[float] = None
    eval_verdict: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ReplayResults:
    """Summary of a replay.run() execution."""

    total: int = 0
    completed: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    tasks: List[ReplayTaskResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Fraction of completed tasks that passed."""
        return self.passed / self.completed if self.completed > 0 else 0.0


# ── Trace correlation helpers ─────────────────────────────────────


def _latest_trace_id(client: Any, agent_name: str) -> Optional[str]:
    """Return the id of the most recent trace for ``agent_name``, or None."""
    recent = client.list_traces(limit=1, agent_name=agent_name)
    traces = recent.get("traces", [])
    return traces[0]["id"] if traces else None


def _wait_for_new_trace(
    client: Any,
    agent_name: str,
    baseline_trace_id: Optional[str],
    attempts: int = 10,
    interval_s: float = 0.2,
) -> Optional[str]:
    """Wait for a trace NEWER than ``baseline_trace_id`` to appear.

    Replaying re-runs the agent, which (via install()) exports a fresh
    trace asynchronously. Naively taking ``list_traces(limit=1)[0]`` after
    a fixed sleep is racy: a concurrent trace for the same agent, or an
    export slower than the sleep, makes us link the WRONG trace and
    silently corrupt the replay eval scores.

    We capture the most-recent id BEFORE running the agent and accept a
    result here only when the most-recent id has CHANGED from that
    baseline — i.e. it's genuinely the trace this replay just produced.
    Returns the new id, or None if no new trace appeared in time.
    """
    import time

    for _ in range(max(1, attempts)):
        time.sleep(interval_s)
        latest = _latest_trace_id(client, agent_name)
        if latest is not None and latest != baseline_trace_id:
            return latest
    return None


# ── Prompt Type ───────────────────────────────────────────────────


@dataclass
class ReplayPrompt:
    """A prompt that needs to be replayed.

    The ``replay_context`` dict (when present) explains *why* this trace
    needs replay — what surfaces changed, which skills/tools were used
    in the original trace, and the compat reason.  This metadata is
    informational and does not control agent behavior during replay.
    """

    trace_id: str
    user_input: str
    original_output: str = ""
    verdict: str = "unknown"
    agent_name: str = ""
    manifest_id: Optional[str] = None
    session_id: Optional[str] = None
    created_at: Optional[str] = None
    replay_context: Optional[Dict[str, Any]] = None


@dataclass
class ReplayTurn:
    """A single turn in a multi-turn replay session."""

    turn_index: int
    trace_id: str
    user_input: str
    original_output: str = ""
    verdict: str = "keep"
    is_trigger: bool = False


@dataclass
class ReplaySession:
    """A session (conversation) that needs replay.

    Contains all turns in chronological order. ``is_trigger`` marks
    which turns are replay-eligible; non-trigger turns provide context.
    """

    session_id: Optional[str]
    turns: List[ReplayTurn] = field(default_factory=list)
    total_turns: int = 0
    trigger_count: int = 0
    verdict: str = "keep"
    agent_name: str = ""


# ── Core Functions ────────────────────────────────────────────────


def get_prompts(
    agent_name: str,
    verdict: Optional[str] = None,
    limit: int = 500,
) -> List[ReplayPrompt]:
    """Get stale prompts that need to be replayed.

    Fetches prompts from traces classified as needing replay after a
    manifest change. Use this for manual replay workflows where you
    run the prompts yourself and call ``link()`` to connect them.

    Args:
        agent_name: Agent name to get replay prompts for.
        verdict: Filter by verdict (``"replay"``, ``"drop"``).
            Defaults to replay + drop.
        limit: Maximum number of prompts (default 500, max 5000).

    Returns:
        List of ReplayPrompt objects.

    Example::

        from decimalai import replay

        prompts = replay.get_prompts("support-agent")
        for p in prompts:
            print(f"[{p.trace_id}] {p.user_input}")
    """
    from .._config import _get_client

    client = _get_client()
    data = client.get_replay_prompts(agent_name, verdict=verdict, limit=limit)

    return [
        ReplayPrompt(
            trace_id=p["trace_id"],
            user_input=p.get("user_input", ""),
            original_output=p.get("original_output", ""),
            verdict=p.get("verdict", "unknown"),
            agent_name=p.get("agent_name", agent_name),
            manifest_id=p.get("manifest_id"),
            session_id=p.get("session_id"),
            created_at=p.get("created_at"),
            replay_context=p.get("replay_context"),
        )
        for p in data.get("prompts", [])
    ]


def get_sessions(
    agent_name: str,
    verdict: Optional[str] = None,
    limit: int = 50,
) -> List[ReplaySession]:
    """Get multi-turn sessions that need replay.

    Groups replay-eligible traces by ``session_id`` and includes all
    turns (including context-only ones) in chronological order.
    Standalone traces appear as single-turn sessions.

    Args:
        agent_name: Agent name to get replay sessions for.
        verdict: Filter trigger turns by verdict.
        limit: Maximum number of sessions (default 50, max 500).

    Returns:
        List of ReplaySession objects.

    Example::

        from decimalai import replay

        sessions = replay.get_sessions("support-agent")
        for s in sessions:
            print(f"Session {s.session_id}: {s.total_turns} turns, {s.trigger_count} triggers")
            for turn in s.turns:
                marker = "→" if turn.is_trigger else " "
                print(f"  {marker} [{turn.turn_index}] {turn.user_input[:60]}")
    """
    from .._config import _get_client

    client = _get_client()
    data = client.get_replay_sessions(agent_name, verdict=verdict, limit=limit)

    return [
        ReplaySession(
            session_id=s.get("session_id"),
            turns=[
                ReplayTurn(
                    turn_index=t.get("turn_index", i),
                    trace_id=t["trace_id"],
                    user_input=t.get("user_input", ""),
                    original_output=t.get("original_output", ""),
                    verdict=t.get("verdict", "keep"),
                    is_trigger=t.get("is_trigger", False),
                )
                for i, t in enumerate(s.get("turns", []))
            ],
            total_turns=s.get("total_turns", 0),
            trigger_count=s.get("trigger_count", 0),
            verdict=s.get("verdict", "keep"),
            agent_name=s.get("agent_name", agent_name),
        )
        for s in data.get("sessions", [])
    ]


def link(
    original_trace_id: str,
    replayed_trace_id: str,
) -> Dict[str, Any]:
    """Link a replayed trace to its original.

    Creates a replay task connecting the two traces and triggers
    auto-scoring on the backend. Use this when you've manually
    re-run a prompt and want DecimalAI to score the comparison.

    The backend will:
    1. Link the two traces in a replay batch
    2. Auto-score the replayed trace against the original
    3. Store the result for the Replay dashboard

    Args:
        original_trace_id: ID of the original trace (from before the agent update).
        replayed_trace_id: ID of the new trace (from re-running the prompt).

    Returns:
        Dict with task_id, eval_score, eval_verdict, and batch_id.

    Example::

        from decimalai import replay

        # After re-running a prompt and getting a new trace
        result = replay.link(
            original_trace_id="abc-123",
            replayed_trace_id="def-456",
        )
        print(f"Score: {result['eval_score']}, Verdict: {result['eval_verdict']}")
    """
    from .._config import _get_client

    client = _get_client()
    return client.link_replay(original_trace_id, replayed_trace_id)


def run(
    agent_fn: Callable[[str], Any],
    agent_name: str,
    verdict: Optional[str] = None,
    limit: int = 100,
    pairwise_scoring: bool = False,
    on_progress: Optional[Callable[[int, int, ReplayTaskResult], None]] = None,
) -> ReplayResults:
    """Run automated replay: fetch prompts, execute agent, submit results.

    This is the primary replay function. It:
    1. Fetches prompts needing replay from the backend
    2. Calls your ``agent_fn`` for each prompt
    3. Links the new trace to the original via ``replay.link()``
    4. Returns a summary of results

    **Important:** Your agent must be instrumented with DecimalAI's
    ``install()`` so that new traces are auto-captured. Call
    ``decimalai.init()`` and your framework's ``install()`` before
    calling ``replay.run()``.

    Args:
        agent_fn: Your agent function. Takes a user input string,
            returns the agent's output (string or any). The SDK's
            tracing pipeline auto-captures the trace.
        agent_name: Agent name to fetch replay prompts for.
        verdict: Filter prompts by verdict (``"replay"``, ``"drop"``).
            Defaults to replay + drop.
        limit: Maximum number of prompts to replay (default 100).
        pairwise_scoring: If True, use LLM-based pairwise comparison
            for scoring (requires GEMINI_API_KEY on the backend). Default False.
        on_progress: Optional callback ``(completed, total, task_result)``
            called after each prompt is processed.

    Returns:
        ReplayResults with total, passed, failed, skipped counts
        and per-task details.

    Example::

        import decimalai
        from decimalai import replay

        decimalai.init(langchain=True)

        from my_agent import agent

        async def run_agent(user_input: str) -> str:
            result = await agent.ainvoke({"input": user_input})
            return result["output"]

        results = replay.run(
            agent_fn=run_agent,
            agent_name="support-agent",
            limit=50,
        )
        print(f"Replayed {results.total}, passed {results.passed}")
    """
    import asyncio

    from .._config import _get_client

    client = _get_client()

    # 1. Fetch prompts
    prompts = get_prompts(agent_name, verdict=verdict, limit=limit)

    if not prompts:
        logger.info("No prompts found for replay (agent=%s, verdict=%s)", agent_name, verdict)
        return ReplayResults()

    logger.info("Starting replay: %d prompts for agent '%s'", len(prompts), agent_name)

    results = ReplayResults(total=len(prompts))

    for i, prompt in enumerate(prompts):
        task_result = ReplayTaskResult(
            original_trace_id=prompt.trace_id,
            user_input=prompt.user_input,
        )

        if not prompt.user_input:
            task_result.status = "skipped"
            task_result.error = "No user input available"
            results.skipped += 1
            results.tasks.append(task_result)
            if on_progress:
                on_progress(i + 1, len(prompts), task_result)
            continue

        try:
            # 2. Log replay context
            ctx = prompt.replay_context or {}
            changed = ctx.get("changed_surfaces", [])
            if changed:
                changes_str = ", ".join(
                    f"{s['name']} ({s.get('reason', '?')})" for s in changed
                )
                logger.info(
                    "  [%d/%d] %s — changes: %s",
                    i + 1, len(prompts), prompt.verdict, changes_str,
                )

            # 3. Capture the latest trace id BEFORE running the agent so we
            #    can tell the replay's freshly-exported trace apart from any
            #    pre-existing/concurrent trace for the same agent.
            baseline_trace_id = _latest_trace_id(client, agent_name)

            # 4. Call the user's agent function
            logger.debug("Replaying %d/%d: %s...", i + 1, len(prompts), prompt.user_input[:60])

            # Handle both sync and async agent functions
            if asyncio.iscoroutinefunction(agent_fn):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    # Already in an async context — create a task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        output = pool.submit(
                            asyncio.run, agent_fn(prompt.user_input)
                        ).result()
                else:
                    output = asyncio.run(agent_fn(prompt.user_input))
            else:
                output = agent_fn(prompt.user_input)

            # 5. Wait for the trace this replay just produced — a trace
            #    NEWER than the baseline. Avoids linking a stale/concurrent
            #    trace when async export lags or another trace lands first.
            replayed_trace_id = _wait_for_new_trace(
                client, agent_name, baseline_trace_id
            )

            if replayed_trace_id:
                task_result.replayed_trace_id = replayed_trace_id

                # 5. Link original → replayed and auto-score
                link_result = link(prompt.trace_id, replayed_trace_id)
                task_result.eval_score = link_result.get("eval_score")
                task_result.eval_verdict = link_result.get("eval_verdict")
                task_result.status = "completed"

                if task_result.eval_verdict == "pass":
                    results.passed += 1
                else:
                    results.failed += 1
                results.completed += 1
            else:
                task_result.status = "failed"
                task_result.error = "No trace captured — is install() configured?"
                results.failed += 1
                results.completed += 1

        except Exception as exc:
            task_result.status = "failed"
            task_result.error = str(exc)
            results.failed += 1
            results.completed += 1
            logger.warning("Replay failed for trace %s: %s", prompt.trace_id[:8], exc)

        results.tasks.append(task_result)
        if on_progress:
            on_progress(i + 1, len(prompts), task_result)

    logger.info(
        "Replay complete: %d total, %d passed, %d failed, %d skipped (%.0f%% pass rate)",
        results.total, results.passed, results.failed, results.skipped,
        results.pass_rate * 100,
    )

    return results


def run_sessions(
    agent_fn: Callable,
    agent_name: str,
    verdict: Optional[str] = None,
    limit: int = 50,
    on_progress: Optional[Callable[[int, int, ReplayTaskResult], None]] = None,
) -> ReplayResults:
    """Replay multi-turn sessions with history accumulation.

    For each session, replays all turns in order, passing accumulated
    conversation history to the agent function so it has full context.

    **Agent function signature:**

    Your ``agent_fn`` can accept either:

    - ``(user_input: str)`` — single-turn (history is ignored)
    - ``(user_input: str, history: List[Dict])`` — multi-turn, where
      ``history`` is a list of ``{"role": "user"|"assistant", "content": ...}``
      dicts from prior turns

    The SDK detects the signature automatically.

    Args:
        agent_fn: Your agent function.
        agent_name: Agent name to fetch sessions for.
        verdict: Filter sessions by verdict.
        limit: Maximum sessions to replay (default 50).
        on_progress: Optional ``(completed, total, result)`` callback.

    Returns:
        ReplayResults with per-turn task details.

    Example::

        from decimalai import replay

        async def my_agent(user_input: str, history: list) -> str:
            messages = history + [{"role": "user", "content": user_input}]
            result = await llm.chat(messages)
            return result

        results = replay.run_sessions(
            agent_fn=my_agent,
            agent_name="support-agent",
        )
        print(f"Replayed {results.total}, passed {results.passed}")
    """
    import inspect

    from .._config import _get_client

    client = _get_client()

    # 1. Fetch sessions
    sessions = get_sessions(agent_name, verdict=verdict, limit=limit)

    if not sessions:
        logger.info("No sessions found for replay (agent=%s)", agent_name)
        return ReplayResults()

    # Count total trigger turns across all sessions
    total_triggers = sum(s.trigger_count for s in sessions)
    logger.info(
        "Starting session replay: %d sessions, %d trigger turns for agent '%s'",
        len(sessions), total_triggers, agent_name,
    )

    # Detect if agent_fn accepts history parameter
    sig = inspect.signature(agent_fn)
    accepts_history = len(sig.parameters) >= 2

    results = ReplayResults(total=total_triggers)
    completed_count = 0

    for session in sessions:
        history: List[Dict[str, str]] = []

        logger.info(
            "  Session %s: %d turns (%d triggers)",
            session.session_id or "standalone",
            session.total_turns,
            session.trigger_count,
        )

        for turn in session.turns:
            if not turn.is_trigger:
                # Context-only turn — add to history but don't replay
                history.append({"role": "user", "content": turn.user_input})
                history.append({"role": "assistant", "content": turn.original_output})
                continue

            task_result = ReplayTaskResult(
                original_trace_id=turn.trace_id,
                user_input=turn.user_input,
            )

            if not turn.user_input:
                task_result.status = "skipped"
                task_result.error = "No user input available"
                results.skipped += 1
                results.tasks.append(task_result)
                completed_count += 1
                if on_progress:
                    on_progress(completed_count, total_triggers, task_result)
                continue

            try:
                logger.debug(
                    "    Turn %d: %s...", turn.turn_index, turn.user_input[:60]
                )

                # Capture latest trace id BEFORE running the agent so we can
                # distinguish this turn's freshly-exported trace from any
                # pre-existing/concurrent trace for the same agent.
                baseline_trace_id = _latest_trace_id(client, agent_name)

                # Call agent with or without history
                if accepts_history:
                    raw_output = _call_agent(agent_fn, turn.user_input, history)
                else:
                    raw_output = _call_agent(agent_fn, turn.user_input)

                output = str(raw_output) if raw_output is not None else ""

                # Update history for subsequent turns
                history.append({"role": "user", "content": turn.user_input})
                history.append({"role": "assistant", "content": output})

                # Wait for the trace this turn just produced — a trace NEWER
                # than the baseline — rather than blindly taking the most
                # recent one (which may be stale or another concurrent run).
                replayed_trace_id = _wait_for_new_trace(
                    client, agent_name, baseline_trace_id
                )

                if replayed_trace_id:
                    task_result.replayed_trace_id = replayed_trace_id

                    link_result = link(turn.trace_id, replayed_trace_id)
                    task_result.eval_score = link_result.get("eval_score")
                    task_result.eval_verdict = link_result.get("eval_verdict")
                    task_result.status = "completed"

                    if task_result.eval_verdict == "pass":
                        results.passed += 1
                    else:
                        results.failed += 1
                    results.completed += 1
                else:
                    task_result.status = "failed"
                    task_result.error = "No trace captured — is install() configured?"
                    results.failed += 1
                    results.completed += 1

            except Exception as exc:
                task_result.status = "failed"
                task_result.error = str(exc)
                results.failed += 1
                results.completed += 1
                logger.warning(
                    "    Replay failed for turn %d (trace %s): %s",
                    turn.turn_index, turn.trace_id[:8], exc,
                )

                # Still add to history so subsequent turns have context
                history.append({"role": "user", "content": turn.user_input})
                history.append({"role": "assistant", "content": "[replay failed]"})

            results.tasks.append(task_result)
            completed_count += 1
            if on_progress:
                on_progress(completed_count, total_triggers, task_result)

    logger.info(
        "Session replay complete: %d total, %d passed, %d failed, %d skipped (%.0f%% pass rate)",
        results.total, results.passed, results.failed, results.skipped,
        results.pass_rate * 100,
    )

    return results


def _call_agent(agent_fn: Callable, *args: Any) -> Any:
    """Call an agent function, handling both sync and async."""
    import asyncio

    if asyncio.iscoroutinefunction(agent_fn):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, agent_fn(*args)).result()
        else:
            return asyncio.run(agent_fn(*args))
    else:
        return agent_fn(*args)


def load_agent_fn(dotted_path: str) -> Callable:
    """Load an agent function from a dotted Python path.

    Format: ``"module.path:function_name"``

    Example::

        fn = load_agent_fn("my_app.agent:run")
        # Equivalent to: from my_app.agent import run as fn

    Args:
        dotted_path: Import path in "module:function" format.

    Returns:
        The callable agent function.

    Raises:
        ValueError: If the path format is invalid.
        ImportError: If the module can't be imported.
        AttributeError: If the function doesn't exist.
    """
    import importlib
    import os
    import sys

    if ":" not in dotted_path:
        raise ValueError(
            f"Invalid agent function path: {dotted_path!r}. "
            "Expected format: 'module.path:function_name'"
        )

    module_path, fn_name = dotted_path.rsplit(":", 1)

    # Add CWD to sys.path so local modules can be imported
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    module = importlib.import_module(module_path)
    fn = getattr(module, fn_name)

    if not callable(fn):
        raise TypeError(f"{dotted_path} is not callable")

    return fn
