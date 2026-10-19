"""Google ADK (Agent Development Kit) integration.

First-class tracing for ``google-adk`` via a native ADK ``BasePlugin``. One
DecimalAI :class:`RunTrace` is captured per ADK *invocation* (one
``Runner.run`` / ``run_async``), with LLM generations, tool calls, and
multi-agent (sub-agent) activity recorded against it. The root agent's
model / instruction / tools / sub-agents are auto-registered as a manifest.

Simple path (global — every ``Runner`` is traced)::

    import decimalai
    decimalai.init(adk=True)          # or: from decimalai.adk import instrument; instrument()

    from google.adk.runners import Runner
    runner = Runner(agent=my_agent, app_name="support", session_service=svc)
    # all runner.run_async(...) invocations are now traced

The global install covers every way ADK 2.12 builds a runner:
``Runner(agent=...)``, ``Runner(app=App(...))``, ``InMemoryRunner(app=...)``,
and the ``Runner(app=...)`` that ``adk web`` / ``adk run`` / ``api_server``
construct. For an ``app=`` runner the plugin goes into the runner's copy of
``App.plugins``; the caller's ``App`` is not modified.

Explicit path (attach the plugin yourself — ADK's ``App`` is where plugins go)::

    from google.adk.apps import App
    from decimalai.adk import DecimalaiPlugin
    app = App(name="support", root_agent=my_agent,
              plugins=[DecimalaiPlugin(agent_name="support")])
    runner = Runner(app=app, session_service=svc)

An explicit DecimalAI plugin answers for its runner: with the global install
also on, the shared plugin is not added beside it, so an invocation is traced
once.

Skills (opt-in)::

    decimalai.init(api_key=..., agent_name="support")
    from decimalai.adk import instrument
    instrument(agent_name="support", enable_skill_loader=True)

With the loader on, every model turn is routed through ``SkillRouter`` and the
result is appended to ``llm_request.config.system_instruction`` from the
plugin's ``before_model_callback`` — the same hook, and the same field, ADK's
own ``GlobalInstructionPlugin`` uses (``google/adk/plugins/
global_instruction_plugin.py:86-121``). ``flows/llm_flows/core/_model_call.py``
runs that hook (line 194) and hands the same request object to
``llm.generate_content_async`` (line 261), so what is appended here is what the
model is sent. The router call is a blocking HTTP round-trip, so it runs on a
worker thread and never on ADK's event loop. Model turns of one invocation
that ask the same question while that call is in flight — the branches of a
``ParallelAgent`` — wait for it and share its answer, so one user turn is one
routing decision and the trace names it.

Native mode: ADK's own skill loader
-----------------------------------
ADK ships a skill loader of its own — ``google.adk.tools.skill_toolset.
SkillToolset``, with ``list_skills`` / ``load_skill`` / ``load_skill_resource``
/ ``run_skill_script`` tools. Pointed at the skills ``decimalai skills add``
installed::

    from google.adk.tools.skill_toolset import SkillToolset
    import decimalai.adk

    agent = LlmAgent(..., tools=[SkillToolset(skills=decimalai.adk.skills(),
                                              code_executor=...)])

the plugin sees each ``load_skill`` result and records it as an activation
carrying the registry identity of the body the MODEL received
(``decimalai._native``): the stamp in the frontmatter ADK handed over, checked
against that body's hash, else the lock. ADK parses the folders once, when
``skills()`` runs, so after ``decimalai skills update`` an agent built earlier
keeps serving the old version, and its activations say so. A skill loaded twice
in one invocation is one activation. The names ``list_skills`` returns (or the
catalog an ``EAGER`` toolset writes into the system instruction) are recorded
as offered.

The router's injection stays off for an agent that carries a ``SkillToolset``:
the toolset already offers the skills, and a second catalog in the same prompt
would give the model two sets of instructions for one job. That is said once
per agent at INFO. Every other agent of the same runner is routed as before.

The toolset's calls are traced like every other tool call, results included:
each result lands on the ``ToolCallRecord`` of the model turn that asked for it
(matched by function-call id), as a bounded JSON-safe copy. A ``load_skill`` /
``load_skill_resource`` / ``run_skill_script`` result keeps the skill's name,
the file path and the frontmatter's ``name`` and ``metadata`` whole, so the
trace says which skill was read; the body, a resource's content and a script's
output are bounded like any other tool output.

ADK is built around Gemini; the release gate pairs this adapter with the
``google`` provider only.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

from ._native import Load, NativeLoads, record
from ._skill_witness import copy_delivered_versions
from .schema.common import CallRole, FinishReason, SpanType, Status
from .schema.manifest import ManifestTracker, extract_from_config
from .schema.trace import LlmCallRecord, RunTrace, ToolCallRecord, TraceSpan

logger = logging.getLogger("decimalai.adk")

# ── Module-level manifest state (mirrors langchain.py) ─────────
# A manifest is registered once per distinct (agent, tools, prompt, model,
# subagents) shape and reused across traces until that shape changes.
#
# Keyed BY AGENT NAME, not process-wide. Two ADK agents in one process very
# often share a manifest shape — same model, same instruction, same tool
# names, differing only in the label the traces are filed under — so a single
# tracker answered "same hash, already registered" for the second agent and
# handed back the FIRST agent's manifest_id. The trace then pointed at a
# manifest registered for somebody else, which is a cross-agent identity leak,
# not a cache hit. Per-agent state keeps the dedup (an agent that runs twice
# still mints one version) while making the reuse agent-scoped, which is also
# how the backend itself scopes manifest hashes.
_manifest_trackers: Dict[str, ManifestTracker] = {}
_manifest_ids: Dict[str, str] = {}
_manifest_lock = threading.Lock()

#: agent -> (snapshot, refusing exception) for a registration the platform REFUSED.
#: Without it the trace ships under `snapshot.id` — an id the platform never stored —
#: and ingest answers 400 "manifest_id ... does not exist", losing the trace. The
#: caller-thread ladder cannot outlast a Cloud Run revision with no available
#: instance, so the retry belongs on the background sender. Same shape as
#: langchain.py and generic.py; ADK needed it too the moment `decimalai init
#: --framework adk` started writing agents people run.
_pending_manifests: Dict[str, tuple] = {}


def _reregister_refused(agent: str) -> Optional[str]:
    """Re-attempt ONE agent's refused registration, on the sender's thread.

    Returns the id the platform stored, or None to keep holding the trace.
    """
    # `_config` is imported per-function in this module, never at module level.
    from . import _config

    with _manifest_lock:
        pending = _pending_manifests.get(agent)
    if not pending:
        return _manifest_ids.get(agent)
    snapshot, exc = pending
    # A credential refusal will not become a different answer by waiting.
    if exc is not None and ("401" in str(exc) or "403" in str(exc)):
        return None
    try:
        result = _config._get_client().register_manifest(snapshot)
    except Exception:  # noqa: BLE001 — the sender retries; a raise here would drop it
        return None
    manifest_id = result.get("manifest_id")
    if not manifest_id:
        return None
    with _manifest_lock:
        _manifest_ids[agent] = manifest_id
        _pending_manifests.pop(agent, None)
    return manifest_id

# Set by instrument(); used as the fallback agent_name when a trace can't
# resolve one from the ADK agent itself.
_install_agent_name: Optional[str] = None
_runner_patched = False

# Cached plugin class (built lazily so importing this module never requires
# google-adk to be installed).
_PluginClass: Any = None


# Set by instrument(enable_skill_loader=True). Module-level (not per-plugin)
# for the same reason langchain keeps `_skill_loader_installed` module-level:
# the global install path shares ONE plugin across every Runner, so there is no
# per-Runner object to hang the flag on. `DecimalaiPlugin(enable_skill_loader=)`
# overrides it for one explicitly-constructed plugin.
_skill_loader_enabled = False
_skill_router_singleton: Any = None

#: Logged when a caller asks for ``enable_load_skill_tool``. The leading clause
#: is the refusal marker the conformance suite matches on the run, so it stays
#: word for word.
_LOAD_SKILL_TOOL_REFUSAL = (
    "enable_load_skill_tool is not supported on the adk adapter (this adapter "
    "registers no load_skill tool and does not bridge ADK's SkillToolset, so "
    "the model cannot ask for a DecimalAI skill body); staying on prompt "
    "injection. Use openai_agents or pydantic_ai for the native load_skill tool."
)


@dataclass(frozen=True)
class _InstrumentationConfig:
    agent_name: Optional[str] = None
    on_trace: Optional[Callable[[RunTrace], None]] = None
    enable_skill_loader: bool = False


_instrument_lock = threading.Lock()
_instrument_config = _InstrumentationConfig()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _content_to_text(content: Any) -> str:
    """Flatten a genai ``Content`` (or plain str) into text for previews."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = getattr(content, "parts", None) or []
    chunks: List[str] = []
    for p in parts:
        text = getattr(p, "text", None)
        if text:
            chunks.append(text)
    return "\n".join(chunks).strip()


def _provider_for_model(model: Optional[str]) -> Optional[str]:
    """Best-effort provider id from a model string (ADK is Gemini-first)."""
    if not model:
        return None
    m = str(model).lower()
    if "gemini" in m or "google" in m:
        return "google"
    if "claude" in m:
        return "anthropic"
    if "gpt" in m or m.startswith(("o1", "o3", "o4")):
        return "openai"
    return None


def _map_finish_reason(raw: Any, has_tool_calls: bool) -> FinishReason:
    if has_tool_calls:
        return FinishReason.TOOL_CALLS
    s = str(raw or "").upper()
    if "MAX_TOKEN" in s or "LENGTH" in s:
        return FinishReason.LENGTH
    return FinishReason.STOP


def _function_calls_from_content(content: Any) -> List[Any]:
    """Every named ``function_call`` part of a genai Content, as the objects.

    The objects themselves, not copies of their name and args: the record a
    model turn creates has to be found again when that call's tool finishes
    (see ``_ModelToolCall``), and the object is what carries the call's id.
    """
    out: List[Any] = []
    parts = getattr(content, "parts", None) or []
    for p in parts:
        fc = getattr(p, "function_call", None)
        if fc is not None and getattr(fc, "name", None):
            out.append(fc)
    return out


# ── Tool results ────────────────────────────────────────────
#
# A tool's result is recorded on the ToolCallRecord of the model turn that
# asked for it. What is recorded is a bounded, JSON-safe copy, never the object
# the tool returned: a tool may return anything, and the exporter serializes
# the whole trace with ``model_dump(mode="json")`` (and the ``on_trace``
# observer receives a deep copy), so ONE result with no JSON form, or one that
# refuses to be copied, used to be able to lose the entire trace.

#: Longest preview a tool span carries.
_PREVIEW_LIMIT = 500
#: Characters kept from any one string inside a recorded tool result.
_TOOL_RESULT_STR_LIMIT = 2000
#: Entries kept from any one dict or list inside a recorded tool result.
_TOOL_RESULT_ITEM_LIMIT = 50
#: Containers nested deeper than this are recorded as their text.
_TOOL_RESULT_DEPTH_LIMIT = 8
#: Most characters one recorded result may take, measured as JSON text.
_TOOL_RESULT_TOTAL_LIMIT = 8000
#: Most values one conversion visits, whatever the shape of what it walks.
_TOOL_RESULT_NODE_LIMIT = 5000

#: google-adk's ``SkillToolset`` tools whose results say which skill was read.
#: Matched by class (``google.adk.tools.skill_toolset``), not by tool name: a
#: caller's own function tool may be called ``load_skill`` too.
_SKILL_TOOL_MODULE = "google.adk.tools.skill_toolset"
_SKILL_TOOL_CLASSES = frozenset({"LoadSkillTool", "LoadSkillResourceTool", "RunSkillScriptTool"})


def _clip(text: str, limit: Optional[int]) -> str:
    """``text`` cut to ``limit`` characters, saying how much was cut."""
    if limit is None or len(text) <= limit:
        return text
    return f"{text[:limit]}…[truncated, {len(text) - limit} more chars]"


def _as_text(value: Any) -> str:
    try:
        return str(value)
    except Exception:  # noqa: BLE001 — a broken __str__ must not cost the record
        return f"<{type(value).__name__}>"


def _json_safe(
    value: Any, *, str_limit: Optional[int], item_limit: Optional[int],
    node_limit: int = _TOOL_RESULT_NODE_LIMIT,
) -> Any:
    """A copy of ``value`` made only of JSON types, cut to the given limits.

    ``None`` for a limit keeps everything at that level, which is how the parts
    of a skill tool's result that identify the skill are kept whole. Nesting
    and the number of values visited are always bounded, so a value that
    references itself, or shares one child many times, cannot make the walk
    run away.
    """
    remaining = [node_limit]

    def walk(v: Any, depth: int) -> Any:
        remaining[0] -= 1
        if remaining[0] < 0:
            return "…"
        if v is None or isinstance(v, (bool, int, float)):
            return v
        if isinstance(v, str):
            # A str subclass (a str Enum, say) is recorded as the plain string.
            return _clip(v if type(v) is str else str.__str__(v), str_limit)
        if isinstance(v, (bytes, bytearray, memoryview)):
            return f"<{memoryview(v).nbytes} bytes>"
        if depth >= _TOOL_RESULT_DEPTH_LIMIT:
            return _clip(_as_text(v), str_limit)
        if isinstance(v, dict):
            items = list(v.items())
            kept = items if item_limit is None else items[:item_limit]
            out: Dict[str, Any] = {str(k): walk(child, depth + 1) for k, child in kept}
            if len(kept) < len(items):
                out["…"] = f"{len(items) - len(kept)} more entries"
            return out
        if isinstance(v, (list, tuple, set, frozenset)):
            seq = list(v)
            kept_seq = seq if item_limit is None else seq[:item_limit]
            out_list = [walk(child, depth + 1) for child in kept_seq]
            if len(kept_seq) < len(seq):
                out_list.append(f"… {len(seq) - len(kept_seq)} more items")
            return out_list
        dump = getattr(v, "model_dump", None)
        if callable(dump):  # pydantic: genai types, ADK's own models
            try:
                return walk(dump(mode="json"), depth + 1)
            except Exception:  # noqa: BLE001 — fall through to the text form
                pass
        return _clip(_as_text(v), str_limit)

    return walk(value, 0)


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _bounded_tool_result(result: Any) -> Any:
    """``result`` as JSON types, at most ``_TOOL_RESULT_TOTAL_LIMIT`` chars of JSON.

    Keeps the result's structure where it can: strings are cut first, then cut
    harder, and only a result that is still too big becomes its cut JSON text.
    """
    text = ""
    for str_limit in (_TOOL_RESULT_STR_LIMIT, _TOOL_RESULT_STR_LIMIT // 10):
        safe = _json_safe(result, str_limit=str_limit, item_limit=_TOOL_RESULT_ITEM_LIMIT)
        text = _json_text(safe)
        if len(text) <= _TOOL_RESULT_TOTAL_LIMIT:
            return safe
    return _clip(text, _TOOL_RESULT_TOTAL_LIMIT)


def _is_skill_tool(tool: Any) -> bool:
    """True for ADK's ``load_skill`` / ``load_skill_resource`` / ``run_skill_script``."""
    return any(
        cls.__module__ == _SKILL_TOOL_MODULE and cls.__name__ in _SKILL_TOOL_CLASSES
        for cls in type(tool).__mro__
    )


def _skill_tool_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """A SkillToolset result with what identifies the skill kept whole and first.

    ``skill_name``, ``file_path`` and the frontmatter's ``name`` and
    ``metadata`` are what say WHICH skill (and, through metadata, which version
    of it) the model read, so none of them is cut. They also go first, so the
    span preview, which is cut at ``_PREVIEW_LIMIT``, still names the skill.
    Everything else — the body, a resource's content, a script's output — is
    bounded like any other tool output.
    """
    head: Dict[str, Any] = {
        key: _json_safe(result[key], str_limit=None, item_limit=None)
        for key in ("skill_name", "file_path") if key in result
    }
    frontmatter = result.get("frontmatter")
    if isinstance(frontmatter, dict):
        identity = {
            key: _json_safe(frontmatter[key], str_limit=None, item_limit=None)
            for key in ("name", "metadata") if key in frontmatter
        }
        rest_fm = _bounded_tool_result(
            {k: v for k, v in frontmatter.items() if k not in identity}
        )
        head["frontmatter"] = {
            **identity, **(rest_fm if isinstance(rest_fm, dict) else {"…": rest_fm}),
        }
    rest = _bounded_tool_result({k: v for k, v in result.items() if k not in head})
    return {**head, **(rest if isinstance(rest, dict) else {"…": rest})}


def _recorded_tool_result(tool: Any, result: Any) -> Any:
    """What a ToolCallRecord carries for ``result``: bounded, JSON-safe."""
    if isinstance(result, dict) and _is_skill_tool(tool):
        return _skill_tool_result(result)
    return _bounded_tool_result(result)


# ── native mode: ADK's own SkillToolset ─────────────────────
#
# Recognised by class name and module, through the MRO, like `_is_skill_tool`:
# importing google-adk here would make `import decimalai.adk` need it, and a
# subclass of ADK's toolset is still ADK's loader.


def _is_adk_class(obj: Any, name: str) -> bool:
    return any(
        cls.__module__ == _SKILL_TOOL_MODULE and cls.__name__ == name
        for cls in type(obj).__mro__
    )


def _has_skill_toolset(agent: Any) -> bool:
    """True when ``agent`` offers skills through ADK's own ``SkillToolset``."""
    try:
        return any(_is_adk_class(t, "SkillToolset") for t in getattr(agent, "tools", None) or ())
    except Exception:  # noqa: BLE001 - an agent shape we do not know is not a native one
        return False


_CATALOG_RE = re.compile(r"<available_skills>(.*?)</available_skills>", re.S)
_CATALOG_NAME_RE = re.compile(r"<name>\s*(.*?)\s*</name>", re.S)


def _catalog_names(text: Any) -> List[str]:
    """The skill names in ADK's ``<available_skills>`` listing(s) inside ``text``.

    The XML ``google.adk.skills.prompt.format_skills_as_xml`` writes, for both the
    ``list_skills`` result and an ``EAGER`` toolset's system instruction. Names
    are HTML-escaped there, so they are unescaped here.
    """
    if not isinstance(text, str) or "<available_skills>" not in text:
        return []
    names: List[str] = []
    for block in _CATALOG_RE.findall(text):
        for raw in _CATALOG_NAME_RE.findall(block):
            name = html.unescape(raw)
            if name and name not in names:
                names.append(name)
    return names


def _wire_names(names: List[str]) -> List[str]:
    """Each listed name as the trace reports that skill (decimalai._native, D6): an
    installed skill by its wire name, so a listing and an activation of one skill
    cannot arrive under two names. Anything else as it was listed."""
    from .skillpkg.identity import current_index

    index = current_index()
    out: List[str] = []
    for name in names:
        managed = index.get(name)
        wire = managed.wire_name if managed is not None else name
        if wire not in out:
            out.append(wire)
    return out


#: Agents the router's injection was turned off for, named once each at INFO.
_router_off_logged: set = set()
#: Said once per process: a native load in a run the router also delivered into.
_both_catalogs_warned = False
_native_log_lock = threading.Lock()


def _say_router_off(agent_name: Optional[str]) -> None:
    key = agent_name or ""
    with _native_log_lock:
        if key in _router_off_logged:
            return
        _router_off_logged.add(key)
    logger.info(
        "DecimalAI ADK: agent %r carries ADK's own SkillToolset, so DecimalAI's skill "
        "router does not inject its menu or bodies into that agent's requests: the "
        "toolset already offers the installed skills, and its load_skill calls are "
        "recorded as activations. Other agents are routed as before.", agent_name,
    )


def _warn_both_catalogs(skill_name: Optional[str]) -> None:
    global _both_catalogs_warned
    with _native_log_lock:
        if _both_catalogs_warned:
            return
        _both_catalogs_warned = True
    logger.warning(
        "DecimalAI ADK: ADK's SkillToolset loaded %r in an invocation whose requests "
        "also carried DecimalAI's routed skills, so two skill catalogs reached the "
        "model. The router is turned off only for an agent whose own tools include "
        "the SkillToolset; give the toolset to the agent that uses it, or turn the "
        "skill loader off. Logged once per process.", skill_name,
    )


def _native_listing(state: "_RunState", text: Any) -> None:
    names = _catalog_names(text)
    if names:
        state.native.listed(_wire_names(names))


def _observe_native_tool(state: "_RunState", tool: Any, result: Any) -> None:
    """Turn a SkillToolset ``list_skills`` / ``load_skill`` result into the run's
    native loads. What is recorded is what the tool returned, which is what ADK
    hands the model: the skill is never re-read from disk (decimalai._native, D1)."""
    if _is_adk_class(tool, "ListSkillsTool"):
        # ADK returns the XML string; a wrapped {"result": ...} is accepted too.
        text = result.get("result") if isinstance(result, dict) else result
        _native_listing(state, text)
        return
    if not _is_adk_class(tool, "LoadSkillTool") or not isinstance(result, dict):
        return
    instructions = result.get("instructions")
    if "error" in result or not isinstance(instructions, str):
        return  # a refused load delivered nothing
    frontmatter = result.get("frontmatter")
    metadata = frontmatter.get("metadata") if isinstance(frontmatter, dict) else None
    name = result.get("skill_name")
    record(Load(framework="adk", name=name if isinstance(name, str) else None,
                body=instructions,
                metadata=metadata if isinstance(metadata, dict) else None),
           loads=state.native)
    if state.skills_delivered:
        _warn_both_catalogs(name if isinstance(name, str) else None)


def skills(directory: Optional[Union[str, "os.PathLike[str]"]] = None) -> List[Any]:
    """The project's installed skills, read by ADK's own loader, for ``SkillToolset``::

        SkillToolset(skills=decimalai.adk.skills(), code_executor=...)

    ADK's ``load_skills_from_dir`` fails the whole load when ANY folder is bad (a
    name that does not match its folder, frontmatter it cannot parse). This reads
    the same folders with ADK's ``load_skill_from_dir``, one at a time, in the
    same order, and skips a folder ADK refuses with one WARNING naming it, so one
    broken skill cannot take every other skill away from the agent. A folder with
    no ``SKILL.md`` is skipped silently, as ADK does.

    ADK keeps what it read: an agent built from this list serves these bodies
    until it is rebuilt, also after ``decimalai skills update``, and the
    activations DecimalAI records name that version (see the module docstring).

    Args:
        directory: where to start looking for the project (default: the working
            directory), as for :func:`decimalai.skills_dir`.

    Returns:
        ``google.adk.skills.Skill`` objects, sorted by folder name.

    Raises:
        Whatever :func:`decimalai.skills_dir` raises for a project with no
        installed skills; ``ImportError`` without google-adk.
    """
    from google.adk.skills import load_skill_from_dir

    from .skillpkg.project import skills_dir

    loaded: List[Any] = []
    for folder in sorted(skills_dir(directory).iterdir()):
        if not folder.is_dir():
            continue
        if not ((folder / "SKILL.md").exists() or (folder / "skill.md").exists()):
            continue
        try:
            loaded.append(load_skill_from_dir(folder))
        except Exception as exc:  # noqa: BLE001 - one bad folder must not cost the rest
            reason = (str(exc).splitlines() or [""])[0][:300]
            logger.warning(
                "decimalai.adk.skills(): skipped the skill folder %r: ADK cannot load it "
                "(%s: %s). The other skills are loaded; `decimalai skills verify` "
                "checks the installed folders.", folder.name, type(exc).__name__, reason,
            )
    return loaded


@dataclass
class _ModelToolCall:
    """A tool call a model turn asked for, still waiting for its result.

    ``after_model_callback`` sees the model's ``FunctionCall`` before ADK has
    given it an id when the model supplied none: ADK assigns its ``adk-…`` id
    afterwards, IN PLACE on the same object, and that id is the
    ``tool_context.function_call_id`` the tool callbacks see. So the object is
    kept and its id read when the tool finishes.
    """

    record: ToolCallRecord
    function_call: Any
    call_id: Optional[str]

    def current_id(self) -> Optional[str]:
        fc_id = self.call_id or getattr(self.function_call, "id", None)
        return str(fc_id) if fc_id else None


def _introspect_agent(agent: Any) -> Dict[str, Any]:
    """Pull manifest-relevant config off an ADK agent (best-effort, never raises)."""
    info: Dict[str, Any] = {}
    try:
        model = getattr(agent, "model", None)
        # ADK models can be a string id or a model object with a `.model` attr.
        if model is not None and not isinstance(model, str):
            model = getattr(model, "model", None) or str(model)
        if model:
            info["model"] = {"name": str(model), "provider": _provider_for_model(str(model))}

        instruction = getattr(agent, "instruction", None)
        if isinstance(instruction, str) and instruction.strip():
            info["prompts"] = {"system": instruction.strip()}

        tools = []
        for t in getattr(agent, "tools", None) or []:
            name = getattr(t, "name", None) or getattr(t, "__name__", None)
            if not name:
                continue
            entry: Dict[str, Any] = {"name": str(name)}
            desc = getattr(t, "description", None) or getattr(t, "__doc__", None)
            if isinstance(desc, str) and desc.strip():
                entry["description"] = desc.strip()
            tools.append(entry)
        if tools:
            info["tools"] = tools

        subagents = []
        for sub in getattr(agent, "sub_agents", None) or []:
            sub_name = getattr(sub, "name", None)
            if sub_name:
                subagents.append({"name": str(sub_name)})
        if subagents:
            info["subagents"] = subagents
    except Exception:
        logger.debug("ADK agent introspection failed (non-fatal)", exc_info=True)
    return info


# ── SkillRouter delivery ────────────────────────────────────
#
# ADK's prompt seam, established from the installed package rather than from
# memory (google-adk 2.12.0):
#
#   * ``BasePlugin.before_model_callback(*, callback_context, llm_request)``
#     receives the live ``LlmRequest`` BY REFERENCE.
#     ``flows/llm_flows/core/_model_call.py::call_llm_async`` runs it (line 194)
#     and then passes that same object to
#     ``llm.generate_content_async(llm_request, ...)`` (line 261). Nothing
#     copies it in between.
#   * ``LlmRequest.append_instructions(list[str])`` is PUBLIC API and appends to
#     ``config.system_instruction`` with a ``"\n\n"`` join
#     (``models/llm_request.py:266-283``).
#   * ADK's own ``GlobalInstructionPlugin`` writes
#     ``llm_request.config.system_instruction`` from this hook, which is the
#     first-party proof that the seam is a supported extension point rather
#     than something we discovered by poking at internals.
#   * The agent's own ``instruction`` is already in ``system_instruction`` by
#     then — ``_preprocess_async`` runs the instructions processor before
#     ``_call_llm_async`` — so appending puts the caller's stable prompt FIRST
#     and the routed fragment behind it, which is the ordering the prefix/tail
#     split exists for.
#
# The cache caveat, stated rather than hidden: ADK serializes the whole
# instruction as ONE ``system_instruction`` string, so the per-query tail
# changes those bytes every turn and a provider's implicit prefix cache stops
# matching at the start of that field. This is the same tradeoff LangChain
# makes against Gemini (``langchain_google_genai`` also joins system messages
# into one ``system_instruction``); the stable half still leads, so the split is
# not wasted, but ADK has no second cacheable slot to put the tail in.


def _get_skill_router() -> Any:
    """Lazily construct a SkillRouter using the SDK's global config."""
    global _skill_router_singleton
    # The kill switch: a later init(enabled=False) withholds the router (kept
    # for a re-enable) from a loader an earlier init installed.
    from ._config import _disabled_by_init

    if _disabled_by_init():
        return None
    if _skill_router_singleton is not None:
        return _skill_router_singleton
    try:
        from ._config import _get_config
        from .skill_router import SkillRouter
        config = _get_config()
        _skill_router_singleton = SkillRouter(
            api_key=config.api_key,
            base_url=config.base_url,
            # Every routing call names its agent; no router default (see
            # `skill_router._AGENT_FROM_INIT`).
            agent_name=None,
            # This adapter registers no `load_skill` tool (and does not bridge
            # ADK's own SkillToolset), so there is no on-demand fetch to defer
            # to: injection is the only body channel. That is exactly the case
            # `resolve_inject_body` answers True for.
            inject_body=config.resolve_inject_body(has_tool_loop=False),
        )
        return _skill_router_singleton
    except Exception:
        logger.debug("SkillRouter singleton init failed", exc_info=True)
        return None


def _query_from_request(llm_request: Any, fallback: Optional[str]) -> Optional[str]:
    """The user text this turn should be routed on.

    Walks ``llm_request.contents`` backward for the most recent user content
    that carries TEXT. The text filter is load-bearing on ADK specifically:
    ADK files tool results as ``role="user"`` contents whose only part is a
    ``function_response``, so a naive "last user content" picks the tool result
    on step 2 of a turn — a different query, a different cache key, and
    therefore a second routing decision for one user turn.

    Falls back to the invocation's own ``user_content`` (captured in
    ``before_run_callback``) when the request carries no text at all.
    """
    for content in reversed(list(getattr(llm_request, "contents", None) or [])):
        if getattr(content, "role", None) != "user":
            continue
        text = _content_to_text(content)
        if text:
            return text
    return (fallback or "").strip() or None


def _system_instruction_text(llm_request: Any) -> str:
    """Flatten whatever is in ``config.system_instruction`` to text.

    Used to VERIFY that an append actually landed before anything is claimed
    as delivered — see ``_deliver_routed_turn``.
    """
    config = getattr(llm_request, "config", None)
    si = getattr(config, "system_instruction", None)
    if si is None:
        return ""
    if isinstance(si, str):
        return si
    if isinstance(si, (list, tuple)):
        return "\n\n".join(
            part if isinstance(part, str) else _content_to_text(part) or str(
                getattr(part, "text", "") or ""
            )
            for part in si
        )
    return _content_to_text(si) or str(getattr(si, "text", "") or "")


def _append_system_text(llm_request: Any, texts: List[str]) -> bool:
    """Append ``texts`` to the request's system instruction. True if it landed.

    The str case is ``LlmRequest.append_instructions``, ADK's public API.

    The non-str case is here because ``append_instructions`` DROPS the text and
    only logs a warning when ``system_instruction`` is a ``types.Content`` or a
    part list (google-adk 2.12.0, ``models/llm_request.py:276-282``) — a caller who set
    ``generate_content_config=GenerateContentConfig(system_instruction=Content(...))``
    would otherwise get a beautifully traced run with none of their skills in
    it, silently, which is the exact failure this rail exists to prevent.
    """
    config = getattr(llm_request, "config", None)
    if config is None:
        return False
    si = getattr(config, "system_instruction", None)
    if si is None or isinstance(si, str):
        llm_request.append_instructions(list(texts))
        return True
    try:
        from google.genai import types
    except ImportError:  # pragma: no cover - google-adk always brings genai
        return False
    if isinstance(si, types.Content):
        config.system_instruction = types.Content(
            role=getattr(si, "role", None),
            parts=[*(si.parts or []), *(types.Part(text=t) for t in texts)],
        )
        return True
    if isinstance(si, (list, tuple)):
        # A ``list[PartUnion]``; a bare ``str`` is a valid PartUnion.
        config.system_instruction = [*si, *texts]
        return True
    return False


@dataclass(frozen=True)
class _RoutedTurn:
    """One routing decision: what the router returned, and its drained rails.

    Built on the worker thread that made the router call, because the rails
    are ContextVars of that call (see ``_route_turn``). Applied on the loop by
    ``_deliver_routed_turn`` to every model request that shares the decision,
    so it is read-only once built.
    """

    prefix: Any
    tail: Any
    routing_id: Optional[str]
    offered: Tuple[str, ...]
    delivered: Tuple[str, ...]
    delivered_versions: Tuple[Dict[str, str], ...]


def _inject_skills_into_request(state: "_RunState", llm_request: Any) -> None:
    """Route this turn's skills and append them to the system instruction.

    Blocking, for a caller that is not on ADK's event loop. The plugin itself
    goes through ``_route_turn_shared`` instead, which runs ``_route_turn`` on
    a worker thread and shares one call between the model turns of an
    invocation that ask the same question at the same moment.
    """
    routed = _route_turn(state, _query_from_request(llm_request, state.user_input_preview))
    if routed is not None:
        _deliver_routed_turn(state, llm_request, routed)


async def _route_turn_shared(state: "_RunState", query: Optional[str]) -> Optional[_RoutedTurn]:
    """This invocation's routing decision for ``query``, made once while in flight.

    The branches of an ADK ``ParallelAgent`` are model turns of ONE invocation:
    one ``_RunState``, so one scope (the trace id), one agent label, and the
    same user text. They ask the router the same question at the same moment,
    and would get the same answer: those are the inputs the router's cache is
    keyed by. While the router call ran on the loop, the first branch blocked the
    loop and filled that cache, and the others hit it. Off the loop they all
    missed it at once, and each made its own platform call: N routing
    decisions for one user turn, of which the trace (it has ONE
    ``routing_id``) could name only one. The rest were orphans, deflating
    acceptance and inflating the decision count.

    So a call in flight is shared: a model turn that asks the question an
    in-flight call is already asking waits for that call and gets its answer,
    failure included. N branches retrying a failing platform at the same
    instant would only multiply its load, and three of them are the three
    consecutive failures that open the router's circuit breaker, which then
    skips routing for every turn of its cooldown. Sharing ends when the call
    lands. A later model turn (a tool step) goes to the router as it always
    did, and the router's cache answers it with the same decision for the rest
    of the turn.

    Keyed by the question alone, since the scope and the label are this
    invocation's (the router's other cache inputs are fixed for this adapter):
    two different questions are two calls, never one answer.

    The call runs as its own task, and every model turn — the one that started
    it too — awaits it through ``asyncio.shield``: ADK cancels the other
    branches when one escalates, and a cancelled branch must not take the
    answer from a branch still waiting on it. The task is created here, so
    ``asyncio.to_thread`` still runs the router in a copy of the context of the
    turn that started it (OpenTelemetry's span, the enclosing generic trace).
    Only ADK's event loop touches ``routing_flights``, so it needs no lock.
    """
    key = query or ""
    flight = state.routing_flights.get(key)
    if flight is None:
        flight = asyncio.ensure_future(asyncio.to_thread(_route_turn, state, query))
        state.routing_flights[key] = flight

        def _landed(done: "asyncio.Future[Optional[_RoutedTurn]]") -> None:
            if state.routing_flights.get(key) is done:
                del state.routing_flights[key]
            if not done.cancelled():
                # Retrieved, so a failure no surviving turn awaited is not
                # reported by asyncio as an exception nobody handled.
                done.exception()

        flight.add_done_callback(_landed)
    return await asyncio.shield(flight)


def _route_turn(state: "_RunState", query: Optional[str]) -> Optional[_RoutedTurn]:
    """Ask the router for ``query``'s skills, and drain the call's rails.

    Blocking: the router call is a synchronous HTTP round-trip. The plugin
    therefore never calls this on ADK's event loop — ``_route_turn_shared``
    runs it through ``asyncio.to_thread``, which runs it on a worker thread in
    a COPY of the caller's context (OpenTelemetry's current span and the
    generic tracer's enclosing trace come with it). The router call and the
    drain of its per-call rails below MUST stay in this one function for that
    reason: the rails are ContextVars, and a value the router sets in the
    worker's copy is gone once the worker returns.

    None when there is no router, or the router raised: an unskilled turn.
    """
    router = _get_skill_router()
    if router is None:
        return None

    scope = str(state.trace_id)
    from .skill_router import (
        _release_scoped_routing_rail,
        consume_last_delivered_names,
        consume_last_delivered_versions,
        consume_last_offered_names,
    )
    # Custom and older routers need not initialize the newly added rail.
    # Only this prompt build may supply this turn's immutable body witness.
    consume_last_delivered_versions()
    try:
        parts_fn = getattr(router, "build_prompt_parts", None)
        if callable(parts_fn):
            # The prefix/tail split: `prefix` is byte-identical turn to turn
            # (and carries the skill BODIES), `tail` is the one sentence that
            # depends on this query.
            prefix, tail, routing_id = parts_fn(
                query=query, agent_name=state.agent_name, scope=scope,
            )
        else:
            # A router object that predates the split still has to deliver.
            # Without this branch the AttributeError is swallowed below and the
            # adapter traces perfectly while injecting nothing — the silent
            # no-op that cost LangChain 21 tests before its fallback existed.
            prefix, routing_id = router.build_prompt_fragment(
                query=query, agent_name=state.agent_name, scope=scope,
            )
            tail = ""
    except Exception:
        # A partially built prompt must not hand off its body witness to a
        # later invocation, even when a custom router fails after setting it.
        consume_last_delivered_versions()
        logger.debug("build_prompt_parts failed (non-fatal)", exc_info=True)
        return None
    finally:
        # The router keeps its own copy of this decision under the run's scope,
        # for adapters that read it back from there. This one reads the
        # per-call rails below, so the copy is released here or never.
        _release_scoped_routing_rail(router, scope)

    # Drain the router's per-call rails FIRST, whatever happens next: they are
    # contextvars scoped to the call that just ran, and leaving them full would
    # attribute this turn's skills to the next one.
    offered = consume_last_offered_names()
    delivered = consume_last_delivered_names()
    delivered_versions = consume_last_delivered_versions()
    return _RoutedTurn(
        prefix=prefix, tail=tail, routing_id=routing_id,
        offered=tuple(offered), delivered=tuple(delivered),
        delivered_versions=tuple(delivered_versions),
    )


def _deliver_routed_turn(state: "_RunState", llm_request: Any, routed: _RoutedTurn) -> None:
    """Append a routing decision to one model request, and stamp what landed.

    Called once for each model turn that shares the decision, each with its
    own request. Stamping the run again for a second branch changes nothing:
    the rails are sets, and the decision is the same one.

    Nothing is claimed that was not verified: the offered / delivered / routing
    rails are stamped onto the run only after the appended prefix has been read
    back out of ``llm_request``. A router call that produced no text, or an
    append the request refused, leaves the trace saying nothing happened —
    because nothing did.
    """
    prefix, tail, routing_id = routed.prefix, routed.tail, routed.routing_id
    offered, delivered = routed.offered, routed.delivered

    texts = [t for t in (prefix, tail) if t]
    if not texts:
        return
    if not _append_system_text(llm_request, texts):
        logger.debug(
            "ADK system_instruction is a shape append_instructions cannot extend; "
            "skills not delivered for this turn"
        )
        return
    if prefix and prefix not in _system_instruction_text(llm_request):
        # Belt and braces. Reaching here means the append silently no-opped,
        # and the one thing we must not do is report the skills as delivered.
        logger.warning(
            "DecimalAI ADK: skill prefix did not survive the append; reporting "
            "nothing offered or delivered for this turn"
        )
        return

    if routing_id:
        state.routing_id = routing_id
    state.skills_offered.update(n for n in offered if n)
    state.skills_delivered.update(n for n in delivered if n)
    for version in copy_delivered_versions(routed.delivered_versions, names=delivered):
        name, digest = version.get("name"), version.get("hash")
        if name not in delivered or not digest:
            continue
        # A long invocation may receive more than one immutable version.
        # Preserve each actual delivery rather than replacing it with latest.
        state.skills_delivered_versions.add((
            name, digest, version.get("routing_id") or "",
            version.get("skill_id") or "", version.get("version_id") or "",
        ))


class _RunState:
    """Per-invocation trace accumulator, keyed by ADK ``invocation_id``."""

    __slots__ = (
        "trace_id", "agent_name", "root_agent_name", "started_at", "ended_at",
        "user_input_preview", "final_output_preview",
        "llm_calls", "spans", "pending_llm", "pending_tools", "agent_stack",
        "open_tool_calls", "failed_tool_calls",
        "status", "error_code", "error_message", "manifest",
        "routing_id", "skills_offered", "skills_delivered",
        "skills_delivered_versions", "routing_flights",
        "parent_trace_id", "on_trace", "enable_skill_loader",
        "native", "native_agents",
    )

    def __init__(self, *, agent_name: Optional[str], started_at: datetime):
        self.trace_id: UUID = uuid4()
        self.agent_name = agent_name
        self.parent_trace_id: Optional[str] = None
        self.on_trace: Optional[Callable[[RunTrace], None]] = None
        self.enable_skill_loader = False
        # The ADK root agent's own ``.name`` (a Python identifier). Distinct
        # from ``agent_name`` (the DecimalAI label, which an explicit plugin
        # name can override). Used only to detect when the *root* agent
        # finishes — that ``after_agent`` is our end-of-invocation signal.
        self.root_agent_name: Optional[str] = None
        self.started_at = started_at
        self.ended_at: Optional[datetime] = None
        self.user_input_preview: Optional[str] = None
        self.final_output_preview: Optional[str] = None
        self.llm_calls: List[LlmCallRecord] = []
        self.spans: List[TraceSpan] = []
        # before_model pushes; after_model / on_model_error pops (LIFO; calls
        # are sequential within an invocation, so depth is ~1). Each entry is
        # the record AND the LLM span it is attached to, so the two can never
        # drift apart.
        self.pending_llm: List[Tuple[LlmCallRecord, TraceSpan]] = []
        # key -> (ToolCallRecord, TraceSpan, started_at)
        self.pending_tools: Dict[str, Tuple[ToolCallRecord, TraceSpan, datetime]] = {}
        # Tool calls the model asked for whose result has not arrived yet, in
        # the order the model asked. A finished tool claims its own entry (by
        # call id, else by name and args), and its result goes on that entry's
        # record — the record the model turn already put in `llm_calls`.
        self.open_tool_calls: List[_ModelToolCall] = []
        # call id -> record of a call whose tool raised. ADK runs after_tool
        # for it too when another plugin answers the error, and that answer is
        # what the model got; it belongs on the same record, not a second one.
        self.failed_tool_calls: Dict[str, ToolCallRecord] = {}
        # Open agent spans, outermost first: before_agent pushes, after_agent
        # pops. ADK nests sub-agents inside their parent's agent callback, so
        # the top of this stack is the span everything else hangs off.
        self.agent_stack: List[TraceSpan] = []
        self.status: Status = Status.SUCCESS
        self.error_code: Optional[str] = None
        self.error_message: Optional[str] = None
        # Manifest config introspected from the root agent in before_run.
        self.manifest: Dict[str, Any] = {}
        # Skill-delivery rails for this invocation. Accumulated per model turn
        # (a turn that calls a tool routes more than once) and stamped onto the
        # RunTrace in `_finalize`. Sets, so a multi-step turn reports each
        # skill once.
        self.routing_id: Optional[str] = None
        self.skills_offered: set = set()
        self.skills_delivered: set = set()
        self.skills_delivered_versions: set[Tuple[str, str, str, str, str]] = set()
        # Routing calls in flight, by question: the model turns of this
        # invocation that ask one at the same moment (a ParallelAgent's
        # branches) share it. See `_route_turn_shared`.
        self.routing_flights: Dict[str, "asyncio.Future[Optional[_RoutedTurn]]"] = {}
        # Skills ADK's own SkillToolset listed and loaded in this invocation, as
        # decimalai._native resolved them; applied onto the RunTrace in `_finalize`.
        self.native = NativeLoads()
        # Names of the agents of this invocation that carry a SkillToolset: the
        # router does not inject into their requests (see `_say_router_off`).
        self.native_agents: set = set()


def _plugin_class() -> Any:
    """Build (once) and return the concrete BasePlugin subclass.

    Deferred so that ``import decimalai.adk`` works without google-adk; the
    google-adk import only happens when a plugin is actually constructed.
    """
    global _PluginClass
    if _PluginClass is not None:
        return _PluginClass

    from google.adk.plugins.base_plugin import BasePlugin

    class _DecimalaiPlugin(BasePlugin):
        """ADK BasePlugin that emits one DecimalAI RunTrace per invocation."""

        def __init__(
            self,
            agent_name: Optional[str] = None,
            *,
            name: str = "decimalai",
            project: Optional[str] = None,
            parent_trace_id: Optional[str] = None,
            enable_skill_loader: Optional[bool] = None,
            enable_load_skill_tool: bool = False,
            on_trace: Optional[Callable[[RunTrace], None]] = None,
        ):
            super().__init__(name=name)
            # Accepted and DORMANT — see `instrument()`'s docstring. Refused
            # here too, not only there, because the documented
            # `plugins=[DecimalaiPlugin(...)]` form never calls instrument(),
            # and a flag that is silently dropped on one of two documented
            # paths is the same silent no-op this adapter's own comments keep
            # warning about.
            if enable_load_skill_tool:
                logger.warning(_LOAD_SKILL_TOOL_REFUSAL)
            self.agent_name = agent_name
            self.project = project
            self.parent_trace_id = parent_trace_id
            self.on_trace = on_trace
            self._follow_instrument_config = False
            # None = follow the module flag `instrument()` set. An explicit
            # bool is this plugin's own answer, for the documented
            # `plugins=[DecimalaiPlugin(...)]` path where no instrument() call
            # exists to carry it.
            self.enable_skill_loader = enable_skill_loader
            self._runs: Dict[str, _RunState] = {}
            self._lock = threading.Lock()

        # ── helpers ────────────────────────────────────────
        def _skill_loader_on(self) -> bool:
            if self.enable_skill_loader is not None:
                return bool(self.enable_skill_loader)
            return _skill_loader_enabled

        def _get_run(self, invocation_id: Optional[str]) -> Optional[_RunState]:
            if not invocation_id:
                return None
            with self._lock:
                return self._runs.get(invocation_id)

        def _pop_run(self, invocation_id: Optional[str]) -> Optional[_RunState]:
            if not invocation_id:
                return None
            with self._lock:
                return self._runs.pop(invocation_id, None)

        @staticmethod
        def _parent_id(state: _RunState) -> Optional[UUID]:
            """The span everything opened right now hangs off: the innermost
            agent still running, or None when there is no agent span (which is
            what makes that span the trace root)."""
            return state.agent_stack[-1].id if state.agent_stack else None

        @staticmethod
        def _close_span(
            span: TraceSpan, *, output: Optional[str] = None, error: bool = False
        ) -> None:
            """Stamp a span's end. Previews are only set when they carry text —
            an empty preview reads as "the model said nothing" rather than as
            "there was nothing to preview"."""
            if span.ended_at is None:
                span.ended_at = _now()
            if error:
                span.status = Status.ERROR
            if output and span.output_preview is None:
                span.output_preview = output[:500]

        def _close_open_spans(self, state: _RunState) -> None:
            """Ship the spans of an invocation that ended abnormally.

            A run that raises out of the model never reaches ``after_agent``, so
            its agent span (and any in-flight generation) is still open. Ingest
            rejects the WHOLE trace over one span with no ``ended_at``, so these
            are closed here — they did end, abnormally, now.
            """
            error = state.status == Status.ERROR
            while state.pending_llm:
                _, span = state.pending_llm.pop()
                self._close_span(span, output=state.error_message, error=error)
                state.spans.append(span)
            for _, span, _started in list(state.pending_tools.values()):
                self._close_span(span, output=state.error_message, error=error)
                state.spans.append(span)
            state.pending_tools.clear()
            while state.agent_stack:
                span = state.agent_stack.pop()
                self._close_span(
                    span,
                    output=state.error_message if error else state.final_output_preview,
                    error=error,
                )
                state.spans.append(span)

        # ── run boundary ───────────────────────────────────
        async def before_run_callback(self, *, invocation_context: Any):  # noqa: ANN001
            from . import _config
            if not _config._is_enabled():
                return None
            agent = getattr(invocation_context, "agent", None)
            # An explicit label wins over ADK's internal node name (a
            # constrained Python identifier).
            # Capture configuration and context at invocation START. Finalization
            # runs on an executor thread, where the generic ContextVar is absent;
            # reinstrumenting meanwhile must not move an in-flight observer.
            with _instrument_lock:
                settings = _instrument_config if self._follow_instrument_config else _InstrumentationConfig(
                    self.agent_name, self.on_trace, self._skill_loader_on(),
                )
            agent_name = settings.agent_name or getattr(agent, "name", None) or _install_agent_name
            state = _RunState(agent_name=agent_name, started_at=_now())
            state.on_trace = settings.on_trace
            state.enable_skill_loader = settings.enable_skill_loader
            state.parent_trace_id = self.parent_trace_id
            if not state.parent_trace_id:
                # The caller's enclosing generic trace, read from the context
                # this callback runs in. That holds for the sync entry point
                # too: google-adk 2.12's ``Runner.run`` starts its worker
                # thread under ``contextvars.copy_context().run``, so the
                # caller's ContextVars arrive here intact.
                from .generic import _get_current_trace
                enclosing = _get_current_trace()
                if enclosing is not None:
                    state.parent_trace_id = enclosing.get_trace_id()
            state.root_agent_name = getattr(agent, "name", None)
            state.user_input_preview = _content_to_text(
                getattr(invocation_context, "user_content", None)
            ) or None
            if agent is not None:
                state.manifest = _introspect_agent(agent)
            inv_id = getattr(invocation_context, "invocation_id", None)
            if inv_id:
                with self._lock:
                    self._runs[inv_id] = state
            return None

        async def _complete(self, inv_id: Optional[str]) -> None:
            """Pop the run and build+send its trace. Idempotent: the pop means
            a second caller for the same invocation gets ``None`` and no-ops."""
            state = self._pop_run(inv_id)
            if state is None:
                return
            state.ended_at = _now()
            self._close_open_spans(state)
            # Finalize off the event loop: manifest registration is a sync
            # network round-trip, and the trace ingest is queued there too.
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._finalize, state)
            except RuntimeError:
                self._finalize(state)

        # ── agent boundary ─────────────────────────────────
        async def before_agent_callback(self, *, agent: Any, callback_context: Any):  # noqa: ANN001
            # The agent span is the spine of the trace: without it every model
            # turn and every tool call is a root, so an invocation arrives as a
            # flat list the UI cannot lay out as a waterfall — and an agent that
            # calls no tools arrives as an EMPTY timeline, because tool spans
            # used to be the only spans this plugin ever built.
            state = self._get_run(getattr(callback_context, "invocation_id", None))
            if state is None:
                return None
            # The agent object is only handed to this callback; the model
            # callback knows the agent by name. Noted here so the router can
            # stay out of an agent whose own SkillToolset offers the skills.
            if _has_skill_toolset(agent):
                state.native_agents.add(getattr(agent, "name", None))
            span = TraceSpan(
                parent_span_id=self._parent_id(state),
                span_type=SpanType.AGENT,
                name=str(getattr(agent, "name", None) or state.agent_name or "agent"),
                started_at=_now(),
                input_preview=(state.user_input_preview or None),
            )
            state.agent_stack.append(span)
            return None

        def _close_agent_span(
            self, state: _RunState, agent: Any, *, error: Optional[Exception] = None
        ) -> None:
            """Pop this agent's span off the stack and record it.

            Matched by name from the top of the stack rather than blindly
            popping: a callback that never fired (an agent ADK abandoned) must
            not close somebody else's span.
            """
            name = getattr(agent, "name", None)
            span = None
            for candidate in reversed(state.agent_stack):
                if name is None or candidate.name == str(name):
                    span = candidate
                    break
            if span is None:
                return
            state.agent_stack.remove(span)
            self._close_span(
                span,
                output=str(error)[:500] if error else state.final_output_preview,
                error=error is not None,
            )
            state.spans.append(span)

        async def after_agent_callback(self, *, agent: Any, callback_context: Any):  # noqa: ANN001
            # Primary end-of-invocation signal. The root agent's after_agent is
            # the last agent callback of a successful invocation, so finalize
            # here — but only for the *root*: in a multi-agent run each
            # sub-agent also fires after_agent, and we want exactly one trace
            # per invocation. google-adk 2.12 also fires after_run on success
            # on both runtime paths (the node runtime that runs an LlmAgent
            # root, and ``_run_with_trace`` for other BaseAgent roots), later;
            # ``_complete``'s pop makes whichever arrives second a no-op.
            inv_id = getattr(callback_context, "invocation_id", None)
            state = self._get_run(inv_id)
            if state is None:
                return None
            # Close the span BEFORE finalizing: _complete builds the trace from
            # state.spans, and a root span appended afterwards would be lost.
            self._close_agent_span(state, agent)
            if state.root_agent_name is not None and (
                getattr(agent, "name", None) != state.root_agent_name
            ):
                return None  # a sub-agent finished, not the root
            await self._complete(inv_id)
            return None

        async def on_agent_error_callback(self, *, agent: Any, callback_context: Any, error: Exception):  # noqa: ANN001
            state = self._get_run(getattr(callback_context, "invocation_id", None))
            if state is None:
                return None
            self._close_agent_span(state, agent, error=error)
            state.status = Status.ERROR
            if not state.error_message:
                state.error_message = str(error)
            return None

        async def after_run_callback(self, *, invocation_context: Any):  # noqa: ANN001
            # Fallback for an invocation whose root after_agent never fired
            # (e.g. a before_run plugin that ended the run early). No-ops when
            # after_agent already finalized this run.
            await self._complete(getattr(invocation_context, "invocation_id", None))
            return None

        async def on_run_error_callback(self, *, invocation_context: Any, error: Exception):  # noqa: ANN001
            # End-of-invocation signal for the *error* path. When an exception
            # escapes the runner (e.g. a model 429), the root agent never
            # completes, so neither after_agent nor after_run fires — without
            # this the run state would be orphaned and no trace sent. ADK only
            # notifies on_run_error when the exception actually escapes (a
            # recovered model/tool error never reaches here), and it re-raises
            # afterwards, so finalizing here can't fire on success — and
            # _complete's pop makes a second finalize a no-op regardless.
            inv_id = getattr(invocation_context, "invocation_id", None)
            state = self._get_run(inv_id)
            if state is None:
                return None
            state.status = Status.ERROR
            if not state.error_message:
                state.error_message = str(error)
            await self._complete(inv_id)
            return None

        # ── model boundary ─────────────────────────────────
        async def before_model_callback(self, *, callback_context: Any, llm_request: Any):  # noqa: ANN001
            state = self._get_run(getattr(callback_context, "invocation_id", None))
            if state is None:
                return None
            # Delivery FIRST, tracing after. The order matters: everything
            # below is bookkeeping, and a bookkeeping change must never be able
            # to push the one line that actually reaches the model off the end
            # of the callback. Never raises out — a Router failure degrades to
            # an unskilled turn, it does not break the caller's model call.
            # Off the loop: the router call blocks on HTTP, and a slow
            # platform must stall this turn only — not every other coroutine
            # on ADK's event loop. Shared: a sibling branch asking the same
            # question meanwhile waits for this call instead of making its own,
            # so the turn is one routing decision. The decision is appended to
            # THIS request here, on the loop, by the turn that owns it.
            # An agent whose own SkillToolset offers the skills is not routed at
            # all (D7 of the Phase 3 design): its toolset is the catalog.
            native = getattr(callback_context, "agent_name", None) in state.native_agents
            if state.enable_skill_loader and native:
                _say_router_off(getattr(callback_context, "agent_name", None))
            elif state.enable_skill_loader:
                try:
                    routed = await _route_turn_shared(
                        state, _query_from_request(llm_request, state.user_input_preview),
                    )
                    if routed is not None:
                        _deliver_routed_turn(state, llm_request, routed)
                except Exception:
                    logger.debug("Skill injection failed (non-fatal)", exc_info=True)
            model = getattr(llm_request, "model", None)
            rendered_input: List[Dict[str, Any]] = []
            # The system instruction FIRST, and it is not optional bookkeeping.
            #
            # `llm_request.contents` holds the conversation turns only. ADK puts
            # the agent's own `instruction` — and, three lines above, every
            # routed skill body — in `config.system_instruction`, which is a
            # separate field. Recording only `contents` meant the trace omitted
            # the entire system prompt the model was actually shown.
            #
            # That is not a cosmetic gap. Every delivery signal downstream reads
            # the rendered input: the SDK's own `infer_prompt_rungs`, the
            # platform's skill-activation detection, and the conformance
            # contract's C8/C14. So this adapter could route a skill, append its
            # BODY to the system instruction, verify the append survived, stamp
            # `skills_delivered` on the run — and then emit a trace in which no
            # skill text appears anywhere. Delivery was real and unwitnessable,
            # which is the one combination that cannot be distinguished from
            # delivery never happening. It is why ADK was carried as a
            # non-witnessable framework, and why C8/C14 read "claims to have
            # offered [...] but those names are not in the prompt the model was
            # shown" the first time the rail was graded (2026-09-03).
            #
            # Must stay AFTER `_deliver_routed_turn` above, or it records
            # the instruction as it was before the skills were appended.
            sys_text = _system_instruction_text(llm_request)
            if sys_text:
                rendered_input.append({"role": "system", "content": sys_text})
            if native:
                # An EAGER toolset writes its catalog into the system
                # instruction instead of answering list_skills.
                try:
                    _native_listing(state, sys_text)
                except Exception:
                    logger.debug("native skill listing not recorded (non-fatal)", exc_info=True)
            for c in getattr(llm_request, "contents", None) or []:
                txt = _content_to_text(c)
                if txt:
                    rendered_input.append({"role": getattr(c, "role", "user"), "content": txt})
            started = _now()
            # One span per generation, parented to the agent that asked for it.
            # `LlmCallRecord.span_id` used to be a fresh uuid4 that named no
            # span at all, so the call record could not be joined back to the
            # timeline it belonged to.
            span = TraceSpan(
                parent_span_id=self._parent_id(state),
                span_type=SpanType.LLM,
                name=str(model) if model else "llm",
                started_at=started,
                input_preview=(
                    "\n".join(e["content"] for e in rendered_input)[:500] or None
                    if rendered_input else None
                ),
            )
            rec = LlmCallRecord(
                span_id=span.id,
                agent_name=getattr(callback_context, "agent_name", None) or state.agent_name,
                call_role=CallRole.OTHER,
                provider=_provider_for_model(str(model) if model else None),
                model_name=str(model) if model else None,
                rendered_input=rendered_input or None,
                started_at=started,
            )
            state.pending_llm.append((rec, span))
            return None

        async def after_model_callback(self, *, callback_context: Any, llm_response: Any):  # noqa: ANN001
            state = self._get_run(getattr(callback_context, "invocation_id", None))
            if state is None or not state.pending_llm:
                return None
            rec, span = state.pending_llm.pop()
            rec.ended_at = _now()
            content = getattr(llm_response, "content", None)
            text = _content_to_text(content)
            if text:
                rec.output = {"role": "model", "content": text}
                state.final_output_preview = text
            # Tool calls requested by the model in this turn. Each record is
            # the ONE record of its call: the tool callbacks fill in its result
            # when the tool finishes, through `open_tool_calls`.
            fcs = _function_calls_from_content(content)
            for fc in fcs:
                fargs = getattr(fc, "args", None)
                tool_rec = ToolCallRecord(
                    tool_name=str(fc.name), args=dict(fargs) if isinstance(fargs, dict) else {},
                )
                rec.tool_calls.append(tool_rec)
                fc_id = getattr(fc, "id", None)
                state.open_tool_calls.append(
                    _ModelToolCall(tool_rec, fc, str(fc_id) if fc_id else None)
                )
            rec.finish_reason = _map_finish_reason(
                getattr(llm_response, "finish_reason", None), bool(fcs)
            )
            usage = getattr(llm_response, "usage_metadata", None)
            if usage is not None:
                rec.input_tokens = getattr(usage, "prompt_token_count", None)
                rec.output_tokens = getattr(usage, "candidates_token_count", None)
            err_msg = getattr(llm_response, "error_message", None)
            if err_msg:
                rec.status = Status.ERROR
            state.llm_calls.append(rec)
            # A pure tool-call turn answers with no text, so `text` is empty;
            # name what the model DID ask for instead of previewing "".
            self._close_span(
                span,
                output=text or (
                    "→ " + ", ".join(str(fc.name) for fc in fcs) if fcs else None
                ),
                error=bool(err_msg),
            )
            state.spans.append(span)
            return None

        async def on_model_error_callback(self, *, callback_context: Any, llm_request: Any, error: Exception):  # noqa: ANN001
            state = self._get_run(getattr(callback_context, "invocation_id", None))
            if state is None:
                return None
            if state.pending_llm:
                rec, span = state.pending_llm.pop()
                rec.ended_at = _now()
                rec.status = Status.ERROR
                rec.finish_reason = FinishReason.ERROR
                state.llm_calls.append(rec)
                self._close_span(span, output=str(error), error=True)
                state.spans.append(span)
            state.status = Status.ERROR
            state.error_message = str(error)
            return None

        # ── tool boundary ──────────────────────────────────
        def _tool_key(self, tool: Any, tool_context: Any) -> str:
            fcid = getattr(tool_context, "function_call_id", None)
            return str(fcid) if fcid else getattr(tool, "name", None) or "tool"

        async def before_tool_callback(self, *, tool: Any, tool_args: Any, tool_context: Any):  # noqa: ANN001
            state = self._get_run(getattr(tool_context, "invocation_id", None))
            if state is None:
                return None
            started = _now()
            tool_name = getattr(tool, "name", None) or "tool"
            rec = ToolCallRecord(
                tool_name=str(tool_name),
                args=dict(tool_args) if isinstance(tool_args, dict) else {},
            )
            span = TraceSpan(
                parent_span_id=self._parent_id(state),
                span_type=SpanType.TOOL,
                name=str(tool_name),
                started_at=started,
                input_preview=str(tool_args)[:500] if tool_args else None,
            )
            state.pending_tools[self._tool_key(tool, tool_context)] = (rec, span, started)
            return None

        @staticmethod
        def _claim_model_tool_call(
            state: _RunState, tool_name: str, call_id: Optional[str], args: Dict[str, Any],
        ) -> Optional[ToolCallRecord]:
            """The record the model turn created for this call, taken off the
            open list so no later call can claim it too.

            By call id first: the model's own id when it gave one, else the id
            ADK assigned (see ``_ModelToolCall``). Without an id to go on, the
            oldest open call of this tool, preferring one whose args are these
            args, so two parallel calls of one tool that finish out of order
            still get their own results. An open call carrying a DIFFERENT id
            is never taken.
            """
            open_calls = state.open_tool_calls
            if call_id:
                for i, entry in enumerate(open_calls):
                    if entry.current_id() == call_id:
                        return open_calls.pop(i).record
            candidates = [
                i for i, entry in enumerate(open_calls)
                if entry.record.tool_name == tool_name
                and not (call_id and entry.current_id())
            ]
            if not candidates:
                return None
            for i in candidates:
                if open_calls[i].record.args == args:
                    return open_calls.pop(i).record
            return open_calls.pop(candidates[0]).record

        def _record_tool_outcome(
            self, state: _RunState, tool: Any, tool_args: Any, tool_context: Any,
            *, result: Any = None, error: Optional[BaseException] = None,
        ) -> None:
            """Put a finished tool call's result (or error) on its ONE record,
            and close its span.

            The record is the one the model turn created. Only a call no model
            turn recorded (a turn this plugin never saw complete) gets a record
            of its own, attached to the latest model call, so the call is never
            lost and never counted twice.
            """
            key = self._tool_key(tool, tool_context)
            pending = state.pending_tools.pop(key, None)
            ended = _now()
            tool_name = str(getattr(tool, "name", None) or "tool")
            args = dict(tool_args) if isinstance(tool_args, dict) else {}
            raw_id = getattr(tool_context, "function_call_id", None)
            call_id = str(raw_id) if raw_id else None

            if error is not None:
                recorded: Any = {
                    "error": _clip(f"{type(error).__name__}: {error}", _TOOL_RESULT_STR_LIMIT),
                }
                preview: Optional[str] = str(error)[:_PREVIEW_LIMIT]
            else:
                recorded = _recorded_tool_result(tool, result)
                preview = _json_text(recorded)[:_PREVIEW_LIMIT] if result is not None else None

            # A tool that raised, answered afterwards by another plugin's
            # on_tool_error: same call, same record, now with that answer.
            target = state.failed_tool_calls.pop(call_id, None) if call_id else None
            if target is None:
                target = self._claim_model_tool_call(state, tool_name, call_id, args)
            if target is None:
                target = pending[0] if pending is not None else ToolCallRecord(
                    tool_name=tool_name, args=args,
                )
                if state.llm_calls:
                    state.llm_calls[-1].tool_calls.append(target)
            target.result = recorded
            if error is not None:
                target.status = Status.ERROR
                if call_id:
                    state.failed_tool_calls[call_id] = target

            if pending is not None:
                _rec, span, started = pending
                target.latency_ms = int((ended - started).total_seconds() * 1000)
                span.ended_at = ended
                if error is not None:
                    span.status = Status.ERROR
                span.output_preview = preview
                state.spans.append(span)

        async def after_tool_callback(self, *, tool: Any, tool_args: Any, tool_context: Any, result: Any):  # noqa: ANN001
            state = self._get_run(getattr(tool_context, "invocation_id", None))
            if state is None:
                return None
            try:
                self._record_tool_outcome(state, tool, tool_args, tool_context, result=result)
            except Exception:
                # Never out of the callback: ADK's PluginManager re-raises a
                # plugin's exception, which fails the caller's run. Loud,
                # because this call's result is now missing from the trace.
                logger.warning(
                    "DecimalAI ADK: could not record the result of tool %r; the trace "
                    "will not carry it", getattr(tool, "name", None), exc_info=True,
                )
            try:
                _observe_native_tool(state, tool, result)
            except Exception:  # noqa: BLE001 - observing must never break the run
                logger.debug("native skill load not recorded (non-fatal)", exc_info=True)
            return None

        async def on_tool_error_callback(self, *, tool: Any, tool_args: Any, tool_context: Any, error: Exception):  # noqa: ANN001
            state = self._get_run(getattr(tool_context, "invocation_id", None))
            if state is None:
                return None
            try:
                self._record_tool_outcome(state, tool, tool_args, tool_context, error=error)
            except Exception:
                logger.warning(
                    "DecimalAI ADK: could not record the error of tool %r; the trace "
                    "will not carry it", getattr(tool, "name", None), exc_info=True,
                )
            state.status = Status.ERROR
            state.error_message = str(error)
            return None

        # ── finalize (runs in executor thread) ─────────────
        def _finalize(self, state: _RunState) -> None:
            from . import _config
            if not _config._is_enabled():
                return
            manifest_prepared = True
            try:
                manifest_id = self._maybe_register_manifest(state)
            except Exception as exc:
                # Manifest metadata is optional. A failed current extraction
                # must neither lose a completed run nor attach an older shape.
                manifest_prepared = False
                manifest_id = None
                logger.warning(
                    "ADK manifest preparation failed for trace %s (%s); exporting without manifest",
                    state.trace_id, type(exc).__name__,
                )
            try:
                client = _config._get_client()
                config = _config._get_config()
                trace = RunTrace(
                    id=state.trace_id,
                    project=self.project or (config.project if config else None),
                    agent_name=state.agent_name,
                    parent_trace_id=state.parent_trace_id,
                    status=state.status,
                    source_type="production",
                    started_at=state.started_at,
                    ended_at=state.ended_at or _now(),
                    user_input_preview=state.user_input_preview,
                    final_output_preview=state.final_output_preview,
                    error_code=state.error_code,
                    error_message=state.error_message,
                    spans=list(state.spans),
                    llm_calls=list(state.llm_calls),
                    manifest_id=manifest_id,
                    session_metadata={'framework': 'adk'},
                    # Skill-delivery rails. `routing_id` closes the
                    # routing_decision × trace_skill_activation join;
                    # `skills_delivered` is the subset whose BODY reached the
                    # model, and on ADK it is only ever populated after the
                    # injected prefix was read back out of the request.
                    routing_id=state.routing_id,
                    skills_offered_in_prompt=sorted(state.skills_offered),
                    skills_delivered=sorted(state.skills_delivered),
                    skills_delivered_versions=[
                        {"name": name, "hash": digest,
                         **({"routing_id": route_id} if route_id else {}),
                         **({"skill_id": skill_id, "version_id": version_id} if skill_id and version_id else {})}
                        for name, digest, route_id, skill_id, version_id in sorted(state.skills_delivered_versions)
                    ],
                )
                try:
                    # Before the observer, so it sees the trace that is exported.
                    state.native.apply(trace)
                except Exception:  # noqa: BLE001 - a native load must not cost the trace
                    logger.debug("native skill loads not applied (non-fatal)", exc_info=True)
                if state.on_trace is not None:
                    try:
                        # Observers cannot mutate the trace that will be exported.
                        state.on_trace(trace.model_copy(deep=True))
                    except Exception:
                        logger.exception("ADK trace observer failed for %s", trace.id)
                if manifest_prepared and state.agent_name in _pending_manifests:
                    # Registration was refused: hold the trace, re-register on the
                    # sender's thread, and stamp the real id before it ships.
                    _config.submit_trace_pending_manifest(
                        client, trace, lambda: _reregister_refused(state.agent_name)
                    )
                else:
                    _config._sender.submit(client.ingest_trace, trace)
                logger.debug(
                    "Queued ADK trace %s (%d spans, %d llm_calls, manifest=%s)",
                    trace.id, len(trace.spans), len(trace.llm_calls),
                    trace.manifest_id or "none",
                )
            except Exception:
                logger.exception("Failed to queue ADK trace %s", state.trace_id)

        def _maybe_register_manifest(self, state: _RunState) -> Optional[str]:
            """Register the root agent's manifest if its shape is new.

            Every lookup and every write is scoped to ``state.agent_name``: the
            manifest this trace carries must belong to the agent this trace
            names, and a second agent's identical shape is a different agent's
            manifest, not a cache hit.
            """
            from . import _config
            agent = state.agent_name or "unknown"
            if not _config._is_enabled():
                return _manifest_ids.get(agent)

            man = state.manifest or {}
            tools = man.get("tools")
            prompts = man.get("prompts")
            model = man.get("model")
            subagents = man.get("subagents")
            if not (tools or prompts or model or subagents):
                # Nothing to declare — ride whatever THIS agent already has
                # rather than re-declaring a shape we did not observe.
                return _manifest_ids.get(agent)

            snapshot = extract_from_config(
                agent_name=agent,
                tools=tools,
                prompts=prompts,
                models={"default": model} if model else None,
                subagents=subagents,
            )
            with _manifest_lock:
                tracker = _manifest_trackers.get(agent)
                if tracker is None:
                    tracker = _manifest_trackers[agent] = ManifestTracker()
                if not tracker.check_and_update(snapshot):
                    return _manifest_ids.get(agent)  # same hash, already registered
                try:
                    client = _config._get_client()
                    result = client.register_manifest(snapshot)
                    _manifest_ids[agent] = result.get("manifest_id", snapshot.id)
                    _pending_manifests.pop(agent, None)
                    logger.info(
                        "Registered ADK manifest %s for %s (hash=%s, components=%d)",
                        _manifest_ids[agent], agent, snapshot.manifest_hash[:12],
                        len(snapshot.components),
                    )
                except Exception as exc:
                    logger.warning("Failed to register ADK manifest, continuing", exc_info=True)
                    _manifest_ids[agent] = snapshot.id
                    # Hand the send side something to retry WITH, so the trace is held
                    # rather than posted under an id the platform never stored.
                    _pending_manifests[agent] = (snapshot, exc)
                    # The hash is already committed to this agent's tracker;
                    # without the reset every later run short-circuits on it and
                    # registration is never retried.
                    tracker.reset()
            return _manifest_ids.get(agent)

    _PluginClass = _DecimalaiPlugin
    return _PluginClass


def DecimalaiPlugin(  # noqa: N802 — factory presents as a class for ergonomics
    agent_name: Optional[str] = None,
    *,
    name: str = "decimalai",
    project: Optional[str] = None,
    parent_trace_id: Optional[str] = None,
    enable_skill_loader: Optional[bool] = None,
    enable_load_skill_tool: bool = False,
    on_trace: Optional[Callable[[RunTrace], None]] = None,
) -> Any:
    """Construct a DecimalAI ADK plugin to add to a ``Runner``.

    Args:
        agent_name: Fallback agent name when the ADK agent doesn't supply one.
        name: ADK plugin name (must be unique among a Runner's plugins).
        project: Optional project grouping for the traces.
        parent_trace_id: When this Runner runs as a sub-agent of another,
            the parent's trace id — links the child traces in the backend.
        on_trace: Optional observer called once on the exporter worker thread
            with a detached completed trace, before export. Exceptions are
            isolated from tracing and execution.
        enable_skill_loader: Route this Runner's turns through SkillRouter and
            append the result to the system instruction. ``None`` (default)
            follows whatever ``instrument()`` was told; pass ``True``/``False``
            to answer for this plugin alone, which is what the explicit
            ``plugins=[DecimalaiPlugin(...)]`` path needs — it never calls
            ``instrument()``, so there is no module flag to inherit. An agent
            carrying ADK's own ``SkillToolset`` is never routed: the toolset
            is its catalog, and its loads are recorded as activations.
        enable_load_skill_tool: Accepted and DORMANT — this adapter
            registers no ``load_skill`` tool (ADK's own ``SkillToolset`` is not
            wired to DecimalAI skills), so the model cannot ask for a body.
            True logs a warning naming the adapters that do have the tool loop
            and stays on prompt injection. See ``instrument()`` for why it is
            accepted rather than rejected.

    Attach it through ADK's ``App`` — ``App(..., plugins=[DecimalaiPlugin()])``
    — which is ADK's supported place for plugins; ``Runner(plugins=...)`` still
    works but ADK deprecates it.

    ⚠ This factory's signature is a SECOND place every parameter must be
    listed. It forwards by explicit keyword, so a parameter added to
    ``_DecimalaiPlugin.__init__`` and not to this signature does not merely
    get ignored — the call raises TypeError and the plugin is never built, so
    the Runner traces NOTHING. That is not hypothetical: adding
    ``enable_load_skill_tool`` to the class alone took every ADK conformance
    item from green to "the documented snippet ran and NOTHING was POSTed".
    """
    return _plugin_class()(
        agent_name=agent_name, name=name, project=project,
        parent_trace_id=parent_trace_id, enable_skill_loader=enable_skill_loader,
        enable_load_skill_tool=enable_load_skill_tool, on_trace=on_trace,
    )


def instrument(
    agent_name: Optional[str] = None, *, enable_skill_loader: bool = False,
    enable_load_skill_tool: bool = False,
    on_trace: Optional[Callable[[RunTrace], None]] = None,
) -> None:
    """Install DecimalAI tracing globally for google-adk.

    Monkeypatches ``Runner.__init__`` so a single shared DecimalAI plugin is
    auto-injected into every ``Runner`` created afterwards — ``InMemoryRunner``
    and the runners ADK's own CLI and server build included. Idempotent.

    How it is injected depends on the shape ADK was given, because ADK refuses
    ``app=`` together with ``plugins=``:

    * ``Runner(app=...)``: the runner is built from a copy of the ``App`` whose
      ``plugins`` list carries the shared plugin first. The caller's ``App`` is
      not modified (the same move ADK's own evaluator makes).
    * ``Runner(agent=...)`` / ``Runner(node=...)``: the runner is built from
      the ``App`` ADK itself would wrap that agent in, with the shared plugin
      in it — not from ``plugins=``, which ADK deprecates and warns about on
      every construction.
    * ``Runner(..., plugins=[...])`` (the caller's own, deprecated): the shared
      plugin is put first in that list.

    Either way, a runner that already carries a DecimalAI plugin — by type, or
    by the shared plugin's name — is left alone, so an explicit
    ``DecimalaiPlugin`` answers for its runner and nothing is traced twice.

    Args:
        agent_name: Optional DecimalAI trace label. When omitted, the agent
            ``decimalai.init()`` named (or ``DECIMAL_AGENT_NAME``), else the
            ADK agent's own name.
        on_trace: Optional observer called once per real invocation, including
            failed invocations. Receives a detached trace with witnessed delivery
            rails on the exporter worker thread. Exceptions are isolated from
            tracing and execution. Updating instrument() affects new invocations only.
        enable_skill_loader: When True, every model turn is routed through
            ``SkillRouter`` and the routed skills — bodies included, since this
            adapter registers no ``load_skill`` tool to fetch them on demand —
            are appended to ``llm_request.config.system_instruction`` before
            the request goes out. Off by default, like every other adapter's
            loader. An agent carrying ADK's own ``SkillToolset`` is never
            routed (said once per agent at INFO): the toolset is its catalog,
            and its ``load_skill`` calls are recorded as activations whether
            or not this is on.
        enable_load_skill_tool: Accepted and DORMANT, exactly as on the
            langchain adapter. This adapter registers no ``load_skill`` tool
            and does not connect DecimalAI skills to ADK's own
            ``SkillToolset``, so the model has no way to ASK for a DecimalAI
            body and the strongest rung this rail can reach is DELIVERED.
            Passing True logs a warning naming the adapters that do have the
            tool loop, and stays on prompt injection.

            It is accepted rather than rejected on purpose: silently ignoring
            the flag is how an adapter ends up graded as if it had a channel it
            does not have. The conformance suite ASKS for the tool loop in the
            ``tool_loaded`` cell precisely so the refusal has to happen out
            loud, on that run, instead of being excused by a sentence in a
            table.
    """
    global _install_agent_name, _runner_patched, _skill_loader_enabled, _instrument_config
    if enable_load_skill_tool:
        logger.warning(_LOAD_SKILL_TOOL_REFUSAL)
    # Not named here: the agent init() named (or DECIMAL_AGENT_NAME). Each call
    # replaces the name, so a bare `instrument(enable_skill_loader=True)` after
    # `init(adk=True, agent_name=...)` used to drop it back to ADK's node name.
    from ._config import default_agent_name
    agent_name = default_agent_name(agent_name)
    with _instrument_lock:
        _install_agent_name = agent_name
        # Preserve the loader's existing sticky opt-in, and atomically publish
        # the observer/name configuration for subsequent invocations.
        if enable_skill_loader:
            _skill_loader_enabled = True
        _instrument_config = _InstrumentationConfig(agent_name, on_trace, _skill_loader_enabled)
        if _runner_patched:
            return

    try:
        from google.adk.runners import Runner
    except ImportError:
        logger.warning(
            "decimalai.adk.instrument() called but google-adk is not installed; "
            "ADK tracing not active. "
            "Install it with: pip install \"decimalai[adk]\""
        )
        return

    with _instrument_lock:
        if _runner_patched:
            return
        shared_plugin = DecimalaiPlugin()
        shared_plugin._follow_instrument_config = True
        original_init = Runner.__init__

        def carries_decimalai(plugins: Any) -> bool:
            # By type, so a renamed explicit DecimalaiPlugin counts; by name,
            # because ADK refuses two plugins with one name.
            return any(
                isinstance(p, type(shared_plugin))
                or getattr(p, "name", None) == shared_plugin.name
                for p in plugins or ()
            )

        def _wraps_cleanly(kwargs: dict) -> bool:
            # Exactly one of agent / node (ADK refuses the rest itself), and an
            # app_name when it is an agent (ADK requires one; for a node it
            # defaults to the node's name).
            agent, node = kwargs.get("agent"), kwargs.get("node")
            if (agent is None) == (node is None):
                return False
            return node is not None or bool(kwargs.get("app_name"))

        def patched_init(self, **kwargs):  # Runner.__init__ is keyword-only
            app = kwargs.get("app")
            if app is not None:
                # ADK refuses app= with plugins= (`Runner._resolve_app`), so the
                # plugin goes INTO the app. A `plugins=` the caller passed
                # alongside it is left as given, and ADK refuses it with its
                # own message. A copy, so the caller's App is not mutated.
                if not carries_decimalai(app.plugins):
                    kwargs["app"] = app.model_copy(
                        update={"plugins": [shared_plugin, *app.plugins]},
                    )
            elif kwargs.get("plugins") is None and _wraps_cleanly(kwargs):
                # A bare agent or node, and no `plugins=`: build the App ADK
                # would build from it (`Runner._resolve_app`), with the plugin
                # in it. Passing `plugins=` instead made ADK warn that the
                # argument is deprecated on every runner, to a user who never
                # passed one.
                from google.adk.apps.app import App

                root = kwargs.pop("agent", None)
                if root is None:
                    root = kwargs.pop("node")
                    kwargs.pop("agent", None)
                    name = kwargs.get("app_name") or getattr(root, "name", "default")
                else:
                    kwargs.pop("node", None)
                    name = kwargs["app_name"]
                kwargs.pop("plugins", None)
                kwargs["app"] = App.model_construct(
                    name=name, root_agent=root, plugins=[shared_plugin],
                )
            elif kwargs.get("plugins") is not None:
                # The caller passed `plugins=`: ADK's deprecated path, and its
                # warning, are theirs. Add the plugin where they put theirs.
                plugins = list(kwargs.pop("plugins"))
                if not carries_decimalai(plugins):
                    plugins.insert(0, shared_plugin)
                kwargs["plugins"] = plugins
            # Anything else is a combination ADK refuses (both agent and node,
            # neither, an agent with no app_name): handed over untouched, so the
            # caller gets ADK's own error and no warning ADK would not give.
            original_init(self, **kwargs)

        Runner.__init__ = patched_init  # type: ignore[method-assign]
        _runner_patched = True
    logger.info("DecimalAI ADK tracing installed globally (agent_name=%s)", agent_name)
