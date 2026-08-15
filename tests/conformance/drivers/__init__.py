"""The Driver contract — and the registry of every framework under conformance.

A driver's ONLY job is to run its framework's documented snippet against a stub
model, pointed at the probe. It declares which contract items apply to it, and
that is the whole of its authority: a driver contains **no assertions**. If a
framework seems to need its own assertion, the contract is wrong — fix
``contract.py``, not the driver.

Adding a framework
------------------
1. Write ``drivers/<name>.py``: a stub model, a ``run(ctx)`` that executes the
   documented snippet, and a module-level ``DRIVER = Driver(...)``.
2. Add the module name to ``DRIVER_MODULES`` below.
3. Run ``pytest tests/conformance``. Anything the adapter gets wrong shows up as
   a failing contract item — which is the point, not a problem.

Everything the run needs comes in on the ``Ctx``; everything the contract checks
is asserted against the SAME ``Ctx``. That is what keeps assertions shared: a
driver that invents its own prompt text instead of using ``ctx.prompt_sentinel``
makes C4 unprovable for its framework, and C4 will say so.
"""

from __future__ import annotations

import importlib
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

# ── What a driver is handed ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Ctx:
    """Everything a driver needs to run one phase. Nothing framework-specific.

    The sentinels are load-bearing. ``prompt_sentinel`` must appear in the text
    the model is asked, ``reply_sentinel`` must be what the stub model answers,
    and ``tool_name`` must be the tool the run calls. The contract asserts the
    wire carries those exact strings — which is how "the preview said
    ``system``" and "the preview was ``<object at 0x7f…>``" get caught without
    the contract knowing anything about the framework.
    """

    base_url: str
    api_key: str
    agent_name: str
    prompt_sentinel: str
    reply_sentinel: str
    tool_name: str
    tool_sentinel: str
    workdir: str
    skills: Tuple[Mapping[str, Any], ...] = ()
    lane: int = 0

    def derive(self, lane: int, *, rename: bool = True) -> "Ctx":
        """A per-lane variant — distinct sentinels, and by default a distinct agent.

        Concurrency defects are cross-contamination defects, so every lane must
        be distinguishable on the wire. Derive, don't reuse.

        ``rename=False`` keeps the agent name: it is for phases run under
        PROCESS-WIDE instrumentation, where the framework only lets you name one
        agent, so per-lane names would be asking for something the API does not
        offer.
        """
        suffix = f"-lane{lane}"
        return Ctx(
            base_url=self.base_url,
            api_key=self.api_key,
            agent_name=self.agent_name + suffix if rename else self.agent_name,
            prompt_sentinel=self.prompt_sentinel + suffix,
            reply_sentinel=self.reply_sentinel + suffix,
            tool_name=self.tool_name,
            tool_sentinel=self.tool_sentinel + suffix,
            workdir=self.workdir,
            skills=self.skills,
            lane=lane,
        )


RunFn = Callable[[Ctx], Any]
FanoutFn = Callable[[Sequence[Ctx]], Any]


# ── the shared stub script ───────────────────────────────────────────────────


@dataclass(frozen=True)
class StubTurn:
    """One scripted model turn, framework-agnostic.

    Every framework's stub model answers the SAME script, so a contract item
    means the same thing everywhere: same tool call, same completion text, same
    token counts. A driver's stub only has to map these fields onto whatever
    message object its framework wants — which is the framework-specific part,
    and the only part.
    """

    #: ``(tool_name, args)`` when this turn asks for a tool, else None.
    tool_call: Optional[Tuple[str, Dict[str, Any]]]
    #: The assistant text for this turn (empty on a pure tool-call turn, which
    #: is what real providers do — and the case that shakes out preview bugs).
    content: str
    input_tokens: int
    output_tokens: int


def stub_script(ctx: Ctx, *, use_tool: bool = True) -> List[StubTurn]:
    """The turns a conformance stub model must produce, in order."""
    reply = StubTurn(None, ctx.reply_sentinel, 23, 7)
    if not use_tool:
        return [reply]
    return [
        StubTurn((ctx.tool_name, {"query": ctx.prompt_sentinel}), "", 17, 5),
        reply,
    ]


def tool_result(ctx: Ctx, query: str) -> str:
    """What every conformance stub tool returns."""
    return f"{ctx.tool_sentinel} for {query}"


def user_message(ctx: Ctx) -> str:
    """The user turn. Carries ``prompt_sentinel``, which C4 asserts on the wire."""
    return f"Please look up {ctx.prompt_sentinel} and report it."


#: The system prompt every driver uses. Static, so the auto-detected manifest
#: hash is stable across runs (which is what C7 grades).
SYSTEM_PROMPT = "You are a conformance fixture. Use the tool, then answer."

#: Model name every stub reports, so C3 reads the same field everywhere.
STUB_MODEL_NAME = "conformance-stub-1"


# ── What a driver declares ───────────────────────────────────────────────────

#: Every capability flag, and the contract items it gates. Declared here so a
#: driver author sees the cost of turning a flag off: the items it silences.
CAPABILITY_ITEMS: Mapping[str, Tuple[str, ...]] = {
    "has_tools": ("C5",),
    "has_skills_rail": ("C8",),
    "supports_concurrency": ("C9",),
    "supports_error_path": ("C10",),
    "supports_degenerate": ("C7b",),
}


@dataclass(frozen=True)
class Capabilities:
    """Which contract items apply to this framework.

    A False flag is a claim that the framework *genuinely cannot* do the thing —
    not that the driver author did not get round to it. Every False flag needs a
    ``reasons`` entry, enforced at construction, and that reason is printed in
    the conformance matrix next to the N/A. Nothing is ever silently skipped.
    """

    # Every flag defaults True on purpose: the default position is "this
    # framework is held to the whole contract", and dropping an item is an
    # explicit, reasoned act rather than something you get by not typing it.
    has_tools: bool = True
    has_skills_rail: bool = True
    supports_concurrency: bool = True
    supports_error_path: bool = True
    supports_degenerate: bool = True
    reasons: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for flag in CAPABILITY_ITEMS:
            if not getattr(self, flag) and not self.reasons.get(flag):
                raise ValueError(
                    f"Capabilities.{flag}=False needs reasons[{flag!r}] — an item a "
                    f"framework cannot do is declared N/A once, with a reason that "
                    f"gets printed. Silent skips are the failure mode this suite exists "
                    f"to remove."
                )

    def na_reason(self, item: str) -> Optional[str]:
        """Why ``item`` does not apply here, or None if it does apply."""
        for flag, items in CAPABILITY_ITEMS.items():
            if item in items and not getattr(self, flag):
                return self.reasons[flag]
        return None


@dataclass(frozen=True)
class Driver:
    """One framework's way of running its documented snippet. No assertions."""

    #: Short id, used in test ids and the matrix.
    name: str
    #: Docs capability-table slugs this driver covers (see the coverage guard).
    covers: frozenset
    #: Importable module names the driver needs. Missing → driver unavailable.
    requires: Tuple[str, ...]
    #: The SDK entry point exercised, for the report (e.g. "decimalai.langchain.CallbackHandler").
    entrypoint: str
    #: Executes the documented snippet once. Return value is ignored by the contract.
    run: RunFn
    capabilities: Capabilities = field(default_factory=Capabilities)
    #: N lanes at once, natively (threads, asyncio, whatever the framework does).
    run_concurrent: Optional[FanoutFn] = None
    #: A run that fails partway through.
    run_error: Optional[RunFn] = None
    #: A run with no model and no tools — the manifest-gate case.
    run_degenerate: Optional[RunFn] = None
    #: The skills rail, one run per lane, as concurrently as the framework
    #: allows. Takes a lane list (like ``run_concurrent``) so routing-id leakage
    #: between runs is visible. Runs LAST, because on several adapters enabling
    #: the rail is an irreversible process-wide monkey-patch.
    run_skills: Optional[FanoutFn] = None

    def __post_init__(self) -> None:
        """A claimed capability must come with the hook that exercises it.

        Without this, ``supports_error_path=True`` and no ``run_error`` fails C10
        with "a failing run produced 0 traces" — a true statement about a phase
        that never ran, which sends the reader hunting an adapter bug that is
        really a driver omission. Fail at construction with the real reason.
        """
        required = {
            "has_skills_rail": "run_skills",
            "supports_concurrency": "run_concurrent",
            "supports_error_path": "run_error",
            "supports_degenerate": "run_degenerate",
        }
        for flag, hook in required.items():
            if getattr(self.capabilities, flag) and getattr(self, hook) is None:
                raise ValueError(
                    f"driver {self.name!r} declares {flag}=True but has no {hook}(). "
                    f"Either implement the hook or set {flag}=False with a reason."
                )

    @property
    def available(self) -> bool:
        return all(_importable(m) for m in self.requires)

    @property
    def missing_requirements(self) -> List[str]:
        return [m for m in self.requires if not _importable(m)]


def _importable(module: str) -> bool:
    """Whether ``module`` can be found, without importing it.

    ``find_spec`` RAISES rather than returning None when a dotted name's
    PARENT is absent — so a driver requiring
    ``openinference.instrumentation.openai`` in a checkout that has no
    ``openinference`` at all would crash driver DISCOVERY, taking the whole
    conformance suite down with it instead of marking one driver unavailable.
    Several drivers declare dotted requirements (``google.adk``,
    ``opentelemetry.sdk``, both openinference instrumentors), so this is the
    ordinary case, not an edge one.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def fanout_threads(run: RunFn) -> FanoutFn:
    """Run one lane per OS thread. The default for anything not natively async.

    Here so a driver gets concurrency for free:
    ``run_concurrent=fanout_threads(run)``.
    """

    def _run(ctxs: Sequence[Ctx]) -> List[Any]:
        with ThreadPoolExecutor(max_workers=max(1, len(ctxs))) as pool:
            return list(pool.map(run, list(ctxs)))

    return _run


class DriverError(RuntimeError):
    """Raised by ``run_error`` drivers to make a run fail on purpose."""


# ── Registry ─────────────────────────────────────────────────────────────────

#: Every driver module under ``tests/conformance/drivers/``.
DRIVER_MODULES: Tuple[str, ...] = (
    "langchain",
    "openai_agents",
    "llamaindex",
    "adk",
    "crewai",
    "ag2",
    "autogen_ms",
    "generic_otel",
    "claude_agent_sdk",
    # Last on purpose: this one enables the OpenInference Anthropic
    # instrumentor, which is a process-wide singleton — running it before the
    # other OTel-routed drivers would bind its tracer to their pipeline.
    "anthropic",
    # Last for the same reason, one SDK over: Pydantic AI does no tracing of
    # its own, so its documented setup pairs it with `init(openai=True)` —
    # which turns on the OpenInference instrumentor for the `openai` SDK
    # process-wide and irreversibly. Run before `openai_agents` it would
    # capture every OpenAI Agents run a SECOND time, as one extra trace per
    # LLM call, and grade that driver on traffic its adapter never sent.
    "pydantic_ai",
)

#: The frameworks the product ADVERTISES, as slugs derived from the capability
#: table in decimalai-docs/sdk/python/frameworks.mdx. Vendored here so the
#: coverage guard still bites in a checkout that has only this repo;
#: ``test_advertised_snapshot_matches_docs`` re-derives it whenever the docs repo
#: IS on disk, so the snapshot cannot quietly go stale.
#:
#: There is deliberately no companion "known missing" ledger. An earlier draft
#: had one, and a ledger that suppresses the coverage failure turns the guard
#: into a formality — add the docs row, add the ledger line, stay green. A
#: framework that genuinely cannot satisfy part of the contract declares those
#: ITEMS N/A in its driver's ``Capabilities``, with a printed reason; it never
#: opts out of having a driver. See ``tests/conformance/test_coverage.py``.
ADVERTISED_SNAPSHOT: frozenset = frozenset({
    "langchain", "langgraph",
    "openai-agents",
    "claude-agent-sdk", "claude-code",
    "pydantic-ai",
    "google-adk",
    "llamaindex",
    "autogen", "ag2",
    "crewai",
    "generic-otel",
})


def all_drivers() -> List[Driver]:
    drivers: List[Driver] = []
    for mod_name in DRIVER_MODULES:
        mod = importlib.import_module(f"{__name__}.{mod_name}")
        driver = getattr(mod, "DRIVER", None)
        if not isinstance(driver, Driver):
            raise RuntimeError(
                f"tests/conformance/drivers/{mod_name}.py must define a module-level "
                f"DRIVER = Driver(...)"
            )
        drivers.append(driver)
    return drivers
