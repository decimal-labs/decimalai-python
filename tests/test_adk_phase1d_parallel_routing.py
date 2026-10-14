"""ADK adapter, Phase 1d (2026-10-14): one user turn of a ParallelAgent is ONE routing decision.

The branches of an ADK ``ParallelAgent`` are sub-agents of ONE invocation. They
share the plugin's per-invocation state, so they route on the same scope (the
invocation's trace id), under the same agent label, on the same user text: the
same question, asked by every branch at the same moment.

While the router call ran on ADK's event loop, the first branch's call blocked
the loop and filled the router's fragment cache, so every later branch got a
cache hit and the same ``routing_id``. Once the call moved to a worker thread
(so a slow platform stalls one turn, not the loop), the branches all missed the
cache at once and each made its own platform call. The platform persisted N
routing decisions for one turn, and the trace — which has ONE ``routing_id`` —
named only the last: N-1 orphan decisions per turn, each one deflating
acceptance and inflating ``router_decisions_count``.

The fix is one decision per question per invocation while it is being made:
branches that ask the same question while the call is in flight wait for it and
share its answer. Later model calls of the turn (a tool step) still go through
the router and hit its cache under the same scope, as they always did. A
different question (another invocation, other user text) still routes on its
own.

Every test drives the REAL google-adk 2.12 package: a real ``ParallelAgent`` of
real ``LlmAgent`` branches, a real ``InMemoryRunner``, and a scripted
``BaseLlm`` per branch. The router is the real ``SkillRouter`` with only its
HTTP call (``_request``) replaced, so its cache, its rails and its body
delivery are the ones users run. No key, no network, no backend.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import itertools
import threading
import time
import warnings
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest


def _real_adk_installed() -> bool:
    # Asked at import time: sibling ADK test files stub `google.adk` in
    # sys.modules inside fixtures, and a stub must never satisfy this.
    try:
        return importlib.util.find_spec("google.adk.agents.parallel_agent") is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(
    not _real_adk_installed(), reason="google-adk is not installed in this environment",
)

AGENT_LABEL = "review_desk"
CALLER_PROMPT = "You are one reviewer of a parallel review."
QUERY = "What fee applies to an opened box?"
SENTINEL = "SENTINEL-PHASE1D-ADK-7c41e2"
BODY = f"Opened boxes carry a 23.5% restocking fee. {SENTINEL}"
BODY_HASH = hashlib.sha256(BODY.encode()).hexdigest()
TAIL = "Most relevant for this request: refund-policy."
#: How long the stub platform holds a routing call open waiting for the other
#: branches. Only a run that never sends the other branches through the model
#: hook waits it out, so it bounds the red run rather than slowing the green one.
GATE_TIMEOUT_S = 3.0
#: How long a routing call stays open AFTER every branch reached the model hook.
#: A branch that routes on its own does so from a worker thread, a moment after
#: it entered the hook; this is that moment, so its cache lookup lands while the
#: first call is still open (and misses) instead of after it filled the cache.
#: A branch that shares the call needs none of it: it joins before the hook
#: hands control back to the loop.
GRACE_S = 0.2


# ── fixtures ────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def client(monkeypatch):
    """A fresh adapter, a mock DecimalAI client, and ADK's Runner restored after."""
    from google.adk.runners import Runner

    import decimalai
    import decimalai._config as cfg
    import decimalai.adk as adk
    from decimalai import skill_router as sr
    from decimalai._config import DecimalConfig

    # instrument() patches the constructor process-wide; put it back.
    monkeypatch.setattr(Runner, "__init__", Runner.__init__)
    mock = MagicMock()
    mock.register_manifest.return_value = {"manifest_id": "phase1d-manifest", "status": "active"}
    monkeypatch.setattr(cfg, "_config", DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True,
    ))
    monkeypatch.setattr(cfg, "_client", mock)
    monkeypatch.setattr(decimalai, "_global_client", mock, raising=False)
    for name, value in (
        ("_PluginClass", None),
        ("_manifest_ids", {}),
        ("_manifest_trackers", {}),
        ("_pending_manifests", {}),
        ("_skill_loader_enabled", False),
        ("_skill_router_singleton", None),
        ("_instrument_config", adk._InstrumentationConfig()),
        ("_runner_patched", False),
        ("_install_agent_name", None),
    ):
        monkeypatch.setattr(adk, name, value)
    sr._last_delivered_versions_ctx.set(None)
    yield mock
    sr._last_delivered_versions_ctx.set(None)


def _flushed(mock: MagicMock) -> List[Any]:
    from decimalai._config import _sender

    _sender.flush()
    return [call.args[0] for call in mock.ingest_trace.call_args_list]


class _Platform:
    """The two platform endpoints a routed turn touches, behind a real SkillRouter.

    A routing call cannot answer until ``expected`` model turns have entered the
    DecimalAI plugin's ``before_model_callback`` (or ``GATE_TIMEOUT_S`` passes),
    and then ``GRACE_S`` more. So every branch of the turn has reached the
    routing step while the first routing call is still open, and a branch
    that routes on its own has had time to miss the cache: the concurrency is
    held open by the platform, not left to how fast a thread starts.
    """

    def __init__(self, *, fail_route: bool = False) -> None:
        self.fail_route = fail_route
        self.routes: List[str] = []           # routing ids issued, in order
        self.route_queries: List[Optional[str]] = []
        self.body_fetches = 0
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._expected = 0
        self._entries = 0
        self._all_entered = threading.Event()

    def arm(self, expected: int) -> None:
        self._expected, self._entries = expected, 0
        self._all_entered.clear()

    def entered_model_hook(self) -> None:  # on the loop, before the plugin runs
        self._entries += 1
        if self._entries >= self._expected:
            self._all_entered.set()

    def request(self, method, path, json=None, params=None, *, hot=None):  # noqa: ANN001
        from decimalai.skill_router import SkillRouterError

        if path.endswith("/route"):
            with self._lock:
                routing_id = "rt_%024d" % next(self._ids)
                self.routes.append(routing_id)
                self.route_queries.append((json or {}).get("query"))
            self._all_entered.wait(GATE_TIMEOUT_S)
            time.sleep(GRACE_S)
            if self.fail_route:
                raise SkillRouterError("platform answered 503", status_code=503)
            return {
                "prompt_fragment": "Available skill: refund-policy",
                "routing_id": routing_id,
                "skills": [{"name": "refund-policy", "description": "Refund rules"}],
                "stable_menu": "Available skill: refund-policy",
                "stable_menu_skills": ["refund-policy"],
                "routing_hint": TAIL,
                "strategy": "smart_routing",
                "query_ranked": True,
            }
        if path.endswith("/body"):
            with self._lock:
                self.body_fetches += 1
            return {"body": BODY, "version": 3, "content_hash": BODY_HASH}
        raise AssertionError(f"unexpected platform call: {method} {path}")


@pytest.fixture
def platform(monkeypatch):
    """A real SkillRouter on a stub platform, installed as the adapter's router,
    and a count of the model turns that reach the plugin's model hook."""
    import decimalai.adk as adk
    from decimalai.skill_router import SkillRouter

    stub = _Platform()
    router = SkillRouter(api_key="dai_sk_test", base_url="http://localhost:8000", inject_body=True)
    monkeypatch.setattr(router, "_request", stub.request)
    monkeypatch.setattr(adk, "_skill_router_singleton", router)

    cls = adk._plugin_class()
    original = cls.before_model_callback

    async def counted(self, *, callback_context, llm_request):  # noqa: ANN001
        stub.entered_model_hook()
        return await original(self, callback_context=callback_context, llm_request=llm_request)

    monkeypatch.setattr(cls, "before_model_callback", counted)
    return stub


def _system_text(llm_request: Any) -> str:
    import decimalai.adk as adk

    return adk._system_instruction_text(llm_request)


def _branch_model(label: str, seen: List[Tuple[str, str]],
                  script: Optional[List[Callable[[Any], Any]]] = None) -> Any:
    """A BaseLlm that records what its branch was sent and answers from a script."""
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    turns = list(script or [])

    class ScriptedLlm(BaseLlm):
        async def generate_content_async(self, llm_request, stream=False):
            seen.append((label, _system_text(llm_request)))
            content = turns.pop(0)(types) if turns else types.Content(
                role="model", parts=[types.Part(text=f"{label}: a restocking fee applies")],
            )
            yield LlmResponse(content=content)

    return ScriptedLlm(model=f"scripted-{label}")


def _parallel_root(labels: List[str], seen: List[Tuple[str, str]], *,
                   scripts: Optional[Dict[str, list]] = None,
                   tools: Optional[Dict[str, list]] = None) -> Any:
    from google.adk.agents import LlmAgent, ParallelAgent

    branches = [
        LlmAgent(
            name=label, model=_branch_model(label, seen, (scripts or {}).get(label)),
            instruction=CALLER_PROMPT, tools=(tools or {}).get(label, []),
        )
        for label in labels
    ]
    with warnings.catch_warnings():
        # ADK 2.12 deprecates ParallelAgent in favour of Workflow, which cannot
        # yet be an LlmAgent sub-agent; it is still what people run.
        warnings.simplefilter("ignore", DeprecationWarning)
        return ParallelAgent(name="review", sub_agents=branches)


def _runner(root: Any, install: str) -> Any:
    import decimalai.adk as adk
    from google.adk.apps import App
    from google.adk.runners import InMemoryRunner

    if install == "global":
        adk.instrument(agent_name=AGENT_LABEL, enable_skill_loader=True)
        return InMemoryRunner(agent=root, app_name="review")
    return InMemoryRunner(app=App(
        name="review", root_agent=root,
        plugins=[adk.DecimalaiPlugin(agent_name=AGENT_LABEL, enable_skill_loader=True)],
    ))


async def _turn(runner: Any, text: str = QUERY, user: str = "u") -> None:
    from google.genai import types

    session = await runner.session_service.create_session(app_name=runner.app_name, user_id=user)
    async for _ in runner.run_async(
        user_id=user, session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=text)]),
    ):
        pass


def _system_inputs(trace: Any) -> List[str]:
    return [
        m["content"] for call in trace.llm_calls for m in (call.rendered_input or [])
        if m["role"] == "system"
    ]


# ── 1. one turn, one decision, every branch skilled ─────────


class TestOneTurnIsOneRoutingDecision:
    @pytest.mark.parametrize("install", ["explicit-app-plugin", "global"])
    @pytest.mark.parametrize("width", [2, 3])
    def test_parallel_branches_share_one_routing_decision(self, client, platform, install, width):
        labels = ["policy", "tone", "facts"][:width]
        seen: List[Tuple[str, str]] = []
        runner = _runner(_parallel_root(labels, seen), install)
        platform.arm(width)

        asyncio.run(_turn(runner))

        assert platform.route_queries == [QUERY], (
            f"one user turn made {len(platform.routes)} routing calls "
            f"({platform.routes}); every one but the trace's is an orphan decision"
        )
        assert platform.body_fetches == 1
        # Every branch was sent the skill, not only the branch that routed.
        assert sorted(label for label, _ in seen) == sorted(labels)
        for label, system in seen:
            assert system.startswith(CALLER_PROMPT), label
            assert SENTINEL in system and TAIL in system, f"branch {label} was sent no skill"
        traces = _flushed(client)
        assert len(traces) == 1
        trace = traces[0]
        assert trace.routing_id == platform.routes[0]
        assert trace.skills_offered_in_prompt == ["refund-policy"]
        assert trace.skills_delivered == ["refund-policy"]
        assert trace.skills_delivered_versions == [
            {"name": "refund-policy", "hash": BODY_HASH, "routing_id": platform.routes[0]},
        ]
        # What each branch was shown is on the trace too.
        assert len(trace.llm_calls) == width
        systems = _system_inputs(trace)
        assert len(systems) == width and all(SENTINEL in s for s in systems)

    def test_a_tool_step_inside_one_branch_reuses_the_turns_decision(self, client, platform):
        """A branch that calls a tool reaches the model hook twice. Its second
        call comes after the shared call has landed, goes to the router, and is
        answered from the router's cache under the same scope — one decision for
        the whole turn, as on a single LlmAgent."""

        def lookup_order(order_id: str) -> dict:
            """Look up an order."""
            return {"order_id": order_id, "state": "opened"}

        def ask_for_the_order(types):
            return types.Content(role="model", parts=[types.Part(
                function_call=types.FunctionCall(name="lookup_order", args={"order_id": "A1"}),
            )])

        seen: List[Tuple[str, str]] = []
        root = _parallel_root(
            ["policy", "tone"], seen,
            scripts={"policy": [ask_for_the_order]}, tools={"policy": [lookup_order]},
        )
        runner = _runner(root, "explicit-app-plugin")
        platform.arm(2)

        asyncio.run(_turn(runner))

        assert platform.route_queries == [QUERY]
        assert platform.body_fetches == 1
        assert [label for label, _ in seen].count("policy") == 2
        assert len(seen) == 3 and all(SENTINEL in system for _, system in seen)
        trace = _flushed(client)[0]
        assert trace.routing_id == platform.routes[0]
        assert trace.skills_delivered_versions == [
            {"name": "refund-policy", "hash": BODY_HASH, "routing_id": platform.routes[0]},
        ]
        assert len(trace.llm_calls) == 3
        tool_calls = [tc for call in trace.llm_calls for tc in call.tool_calls]
        assert [tc.tool_name for tc in tool_calls] == ["lookup_order"]
        assert tool_calls[0].result == {"order_id": "A1", "state": "opened"}


# ── 2. other turns are other decisions ──────────────────────


class TestOtherTurnsStillRouteOnTheirOwn:
    def test_two_turns_in_a_row_are_two_decisions(self, client, platform):
        seen: List[Tuple[str, str]] = []
        runner = _runner(_parallel_root(["policy", "tone"], seen), "explicit-app-plugin")

        async def go():
            platform.arm(2)
            await _turn(runner)
            platform.arm(2)
            await _turn(runner)

        asyncio.run(go())

        assert platform.route_queries == [QUERY, QUERY]
        traces = _flushed(client)
        assert [t.routing_id for t in traces] == platform.routes
        assert len(set(platform.routes)) == 2
        assert len(seen) == 4 and all(SENTINEL in system for _, system in seen)

    def test_two_concurrent_turns_of_one_parallel_agent_do_not_share_a_decision(
        self, client, platform,
    ):
        """Same question, same agent, same moment, but two users: two
        invocations, two scopes, two decisions, and each trace names its own."""
        seen: List[Tuple[str, str]] = []
        runner = _runner(_parallel_root(["policy", "tone"], seen), "explicit-app-plugin")
        platform.arm(4)

        async def go():
            await asyncio.gather(_turn(runner, user="u1"), _turn(runner, user="u2"))

        asyncio.run(go())

        assert platform.route_queries == [QUERY, QUERY]
        traces = _flushed(client)
        assert len(traces) == 2
        assert sorted(t.routing_id for t in traces) == sorted(platform.routes)
        for trace in traces:
            assert trace.skills_delivered_versions == [
                {"name": "refund-policy", "hash": BODY_HASH, "routing_id": trace.routing_id},
            ]
        assert len(seen) == 4 and all(SENTINEL in system for _, system in seen)

    def test_two_questions_in_one_invocation_are_not_merged(self, client):
        """The shared call is keyed by the question. Two model turns of one
        invocation that route on DIFFERENT text, at the same moment, are two
        routing calls — sharing must never hand one question another's skills."""
        import decimalai.adk as adk

        router = _StandInRouter(release_after=2)
        plugin, agent, ic, cc = _plugin_harness(router)
        first, second = _request("How do refunds work?"), _request("Is my order shipped?")

        async def go():
            await plugin.before_run_callback(invocation_context=ic)
            await plugin.before_agent_callback(agent=agent, callback_context=cc)
            await asyncio.gather(
                plugin.before_model_callback(callback_context=cc, llm_request=first),
                plugin.before_model_callback(callback_context=cc, llm_request=second),
            )

        asyncio.run(go())

        assert sorted(router.queries) == ["How do refunds work?", "Is my order shipped?"]
        assert router.released_in_time == [True, True]
        for request in (first, second):
            assert adk._system_instruction_text(request).endswith(TAIL)
            assert SENTINEL in adk._system_instruction_text(request)


# ── 3. the shared call's failure and cancellation ───────────


class TestTheSharedCallFailsAndIsCancelledSafely:
    def test_a_failed_routing_call_is_one_failure_for_the_turn(self, client, platform):
        """Every branch asked at the same moment; the answer for that moment is
        the platform's failure. Each branch retrying it at once would multiply
        the load on a platform that is already failing (three branches are the
        three consecutive failures that open the circuit breaker for every
        later turn). The run still completes, unskilled, and claims nothing."""
        platform.fail_route = True
        seen: List[Tuple[str, str]] = []
        runner = _runner(_parallel_root(["policy", "tone", "facts"], seen), "explicit-app-plugin")
        platform.arm(3)

        asyncio.run(_turn(runner))

        assert platform.route_queries == [QUERY]
        assert platform.body_fetches == 0
        assert len(seen) == 3
        for label, system in seen:
            assert system.startswith(CALLER_PROMPT), label
            assert "refund-policy" not in system and TAIL not in system, label
        trace = _flushed(client)[0]
        assert trace.routing_id is None
        assert trace.skills_offered_in_prompt == []
        assert trace.skills_delivered == []
        assert trace.skills_delivered_versions == []

    def test_a_cancelled_branch_does_not_take_the_decision_from_the_branch_waiting_on_it(
        self, client,
    ):
        """ADK cancels the other branches when one escalates. The branch whose
        call is in flight can be the one cancelled; the branch waiting on that
        call still gets its answer, and the call is not made a second time."""
        import decimalai.adk as adk
        from google.adk.models.llm_response import LlmResponse
        from google.genai import types

        router = _StandInRouter(release_after=None)
        plugin, agent, ic, cc = _plugin_harness(router)
        cancelled_request, waiting_request = _request(QUERY), _request(QUERY)

        async def go():
            await plugin.before_run_callback(invocation_context=ic)
            await plugin.before_agent_callback(agent=agent, callback_context=cc)
            leader = asyncio.create_task(
                plugin.before_model_callback(callback_context=cc, llm_request=cancelled_request),
            )
            while not router.entered.is_set():
                await asyncio.sleep(0.005)
            follower = asyncio.create_task(
                plugin.before_model_callback(callback_context=cc, llm_request=waiting_request),
            )
            await asyncio.sleep(0.05)  # the follower is now waiting on the leader's call
            leader.cancel()
            router.gate.set()
            await follower
            with pytest.raises(asyncio.CancelledError):
                await leader
            await plugin.after_model_callback(callback_context=cc, llm_response=LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text="a fee applies")]),
            ))
            await plugin.after_agent_callback(agent=agent, callback_context=cc)

        asyncio.run(go())

        assert router.queries == [QUERY], "the waiting branch made the call a second time"
        assert SENTINEL in adk._system_instruction_text(waiting_request)
        assert SENTINEL not in adk._system_instruction_text(cancelled_request)
        trace = _flushed(client)[0]
        assert trace.routing_id == router.routing_ids[0]
        assert trace.skills_delivered == ["refund-policy"]
        assert len(trace.llm_calls) == 1 and SENTINEL in _system_inputs(trace)[0]


# ── helpers for the plugin-level tests ──────────────────────


class _StandInRouter:
    """A router stand-in with a gate: each call waits until ``release_after``
    calls have arrived (``None``: until the test opens ``gate``)."""

    def __init__(self, *, release_after: Optional[int]) -> None:
        self.release_after = release_after
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.queries: List[Optional[str]] = []
        self.routing_ids: List[str] = []
        self.released_in_time: List[bool] = []
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def build_prompt_parts(self, query=None, *, agent_name=None, scope=None, **kw):  # noqa: ANN001
        from decimalai import skill_router as sr

        with self._lock:
            routing_id = "rt_%024d" % next(self._ids)
            self.queries.append(query)
            self.routing_ids.append(routing_id)
            if self.release_after is not None and len(self.queries) >= self.release_after:
                self.gate.set()
        self.entered.set()
        self.released_in_time.append(self.gate.wait(GATE_TIMEOUT_S))
        sr._last_offered_names_ctx.set(["refund-policy"])
        sr._last_delivered_names_ctx.set(["refund-policy"])
        sr._last_delivered_versions_ctx.set(
            [{"name": "refund-policy", "hash": BODY_HASH, "routing_id": routing_id}],
        )
        return f"Available skill: refund-policy\n\n## Skill: refund-policy\n\n{BODY}", TAIL, routing_id


def _plugin_harness(router: Any) -> Tuple[Any, Any, Any, Any]:
    import decimalai.adk as adk
    from google.adk.agents import LlmAgent

    adk._skill_router_singleton = router
    plugin = adk.DecimalaiPlugin(agent_name=AGENT_LABEL, enable_skill_loader=True)
    agent = LlmAgent(name="policy", model=_branch_model("policy", []), instruction=CALLER_PROMPT)
    ic = SimpleNamespace(agent=agent, invocation_id="inv-1d", user_content=None, run_config=None)
    cc = SimpleNamespace(invocation_id="inv-1d", agent_name="policy")
    return plugin, agent, ic, cc


def _request(text: str) -> Any:
    from google.adk.models.llm_request import LlmRequest
    from google.genai import types

    return LlmRequest(
        model="scripted",
        contents=[types.Content(role="user", parts=[types.Part(text=text)])],
        config=types.GenerateContentConfig(system_instruction=CALLER_PROMPT),
    )
