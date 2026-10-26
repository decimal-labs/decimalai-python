"""Google ADK (Agent Development Kit): an agent's DecimalAI prompt and skills, and its traces.

::

    from google.adk.agents import LlmAgent
    from google.adk.apps import App
    from google.adk.runners import Runner
    from decimalai.adk import DecimalAIPlugin, agent_config

    support = agent_config("support", fallback=BASE_INSTRUCTION)
    agent = LlmAgent(name="support", model="gemini-3.5-flash",
                     instruction=support.instruction, tools=[lookup_order, *support.tools])
    app = App(name="support", root_agent=agent, plugins=[DecimalAIPlugin()])
    runner = Runner(app=app, session_service=session_service)

``support.instruction`` is an ADK instruction provider. Before each model request it
returns the agent's saved prompt (``{{variables}}`` filled from the session's state),
then the skill list routed on the invocation's user message, then one sentence naming the
tool. ``support.tools`` is ``load_skill``, which the model calls for a listed skill's text.
Each invocation is routed on its own message: the first one before its first model
request, later ones from a local copy while their own decision is recorded in the
background, and every one of an agent with more skills than fit the list. ADK fills no
``{var}`` of its own into an instruction provider's text, so ``{{name}}`` is the only
template syntax in play.

On a Claude model, a session keeps its first invocation's instruction and ``load_skill``:
Claude rejects a conversation whose system prompt or tools changed once a thinking block
is in its history (HTTP 400, "bound to a different conversation"). A newer saved prompt or
skill list reaches the next session. ``DecimalAIPlugin`` keeps what the session was shown
in its state (``_decimalai_conversation``), so any process serving the session shows the
same; without the plugin the copy is this process's own.

``DecimalAIPlugin`` traces: one DecimalAI trace per ADK invocation, with its model calls,
tool calls and sub-agents, and the root agent's manifest (model, instruction, tools,
sub-agents). An invocation whose agent was given ``agent_config()``'s instruction is filed
under that DecimalAI agent and carries the prompt version, its variable names and the
skills offered and loaded. Nothing in ADK is patched: the plugin goes where ADK puts
plugins, ``App(plugins=[...])``, so only the runners built from that app are traced. It is
the one line ADK adds to ``agent_config()``: a run of an agent config by a runner without
it is not traced, and logs one warning per agent saying so.

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

An agent that carries a ``SkillToolset`` gets no DecimalAI skills from ``agent_config()``,
and one warning says so: the toolset already offers the skills, and a second catalog in
the same prompt would give the model two sets of instructions for one job.

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
from collections import OrderedDict
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, NoReturn, Optional, Tuple, Union
from uuid import UUID, uuid4

from ._agent_binding import (
    AgentBinding,
    ConversationRecord,
    attach_prompt_records,
    is_claude,
)
from ._error_message import exception_message
from ._native import Load, NativeLoads, record
from ._skill_witness import copy_delivered_versions
from ._usage import anthropic_cache_split
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

# Cached plugin class (built lazily so importing this module never requires
# google-adk to be installed).
_PluginClass: Any = None

#: The SkillRouter behind every agent config. `init()` drops it when a re-init
#: changes the key or workspace (`decimalai._drop_adapter_skill_routers`).
_skill_router_singleton: Any = None


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


def _function_responses(content: Any) -> List[str]:
    """The tool results in a genai ``Content``, as the text the model reads.

    ADK wraps a tool's plain return value as ``{"result": value}``; that value is
    the text. Any other response is its JSON form.
    """
    out: List[str] = []
    for p in getattr(content, "parts", None) or []:
        response = getattr(getattr(p, "function_response", None), "response", None)
        if response is None:
            continue
        if isinstance(response, dict) and set(response) == {"result"} and isinstance(response["result"], str):
            out.append(response["result"])
        else:
            out.append(_json_text(response))
    return [text for text in out if text]


def _usage_int(usage: Any, name: str) -> Optional[int]:
    value = getattr(usage, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


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


def _output_tokens(usage: Any) -> Optional[int]:
    """Every output token a model call was billed for: the answer and the thinking.

    ADK's usage is Gemini's shape, which keeps the two apart: ``candidates_token_count``
    is the answer and ``thoughts_token_count`` the thinking, and both are billed as
    output. ADK's AnthropicLlm takes the thinking out of Anthropic's ``output_tokens``
    to fit that shape, so the candidates alone fall short on every model that thinks
    (Claude Sonnet and Opus 5.5 always do, as do Gemini's thinking models).

    ADK's LiteLlm does not split them: its ``candidates_token_count`` is LiteLLM's
    ``completion_tokens``, which already holds the reasoning, and
    ``thoughts_token_count`` repeats it. Its ``total_token_count`` is then prompt plus
    candidates, and that is how the thinking is kept from being counted twice.
    """
    def _count(name: str) -> Optional[int]:
        value = getattr(usage, name, None)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    candidates, thoughts = _count("candidates_token_count"), _count("thoughts_token_count")
    if candidates is None and thoughts is None:
        return None
    total, prompt = _count("total_token_count"), _count("prompt_token_count")
    if (thoughts and candidates is not None and total is not None and prompt is not None
            and total == prompt + candidates + (_count("tool_use_prompt_token_count") or 0)):
        return candidates
    return (candidates or 0) + (thoughts or 0)


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


#: Said once per process: a native load in a run an agent config also listed skills in.
_both_catalogs_warned = False
_native_log_lock = threading.Lock()


def _warn_both_catalogs(skill_name: Optional[str]) -> None:
    global _both_catalogs_warned
    with _native_log_lock:
        if _both_catalogs_warned:
            return
        _both_catalogs_warned = True
    logger.warning(
        "DecimalAI ADK: ADK's SkillToolset loaded %r in an invocation that also "
        "listed DecimalAI skills from agent_config(), so two skill catalogs reached the "
        "model. agent_config() leaves an agent alone only when its own tools include the "
        "SkillToolset; give the toolset to the agent that uses it, or that agent no "
        "agent_config() instruction. Logged once per process.", skill_name,
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
    if state.skills_offered or state.skills_delivered:
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
        if callable(instruction):
            # An agent config's instruction is its bound method: declare the saved
            # prompt TEMPLATE (variables unfilled), which is what the dashboard versions.
            owner = getattr(instruction, "__self__", None)
            instruction = owner._template() if isinstance(owner, ADKAgentConfig) else None
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


# ── where an agent config's record goes ──────────────

def _get_skill_router() -> Any:
    """The SkillRouter behind every agent config, built on first use; None when off."""
    global _skill_router_singleton
    # The kill switch: a later init(enabled=False) withholds the router (kept for a
    # re-enable) from configs built before it.
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
            # Skill text reaches the model through load_skill, not the prompt.
            inject_body=False,
        )
        return _skill_router_singleton
    except Exception:
        logger.debug("SkillRouter singleton init failed", exc_info=True)
        return None


def _system_instruction_text(llm_request: Any) -> str:
    """Flatten whatever is in ``config.system_instruction`` to text, for the trace."""
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


#: The invocations a DecimalAIPlugin is tracing right now, by invocation id: where an
#: agent config's record goes. Bound in `before_run_callback`, unbound once the
#: trace is built, so a record that arrives after that is dropped, not misfiled.
_live_runs: Dict[str, "_RunState"] = {}
_live_runs_lock = threading.Lock()

#: How long building a trace waits for its invocation's background routing request.
#: An invocation normally outlasts that request by seconds; this covers one faster
#: than one HTTP round trip, which would otherwise ship with no routing id.
_PENDING_ROUTING_WAIT_S = 3.0


def _live_run(run_key: Optional[str]) -> Optional["_RunState"]:
    if not run_key:
        return None
    with _live_runs_lock:
        return _live_runs.get(run_key)


def _clean_names(names: Any) -> List[str]:
    return [n for n in (names or ()) if isinstance(n, str) and n]


def _record_adk(run_key: Optional[str], fields: Dict[str, Any]) -> bool:
    """Put one piece of an agent config's record on its invocation's trace.

    False when no DecimalAIPlugin is tracing that invocation: the record is dropped.
    Names offered or delivered in a composed instruction count only when they are in
    its text, the way the OpenTelemetry rail checks them.
    """
    state = _live_run(run_key)
    if state is None:
        return False
    text = fields.get("system_prompt")

    def shown(names: Any) -> List[str]:
        kept = _clean_names(names)
        return kept if not isinstance(text, str) else [n for n in kept if n in text]

    with _live_runs_lock:
        routing_id = fields.get("routing_id")
        if routing_id and not state.routing_id:
            state.routing_id = routing_id
        state.skills_offered.update(shown(fields.get("offered")))
        delivered = shown(fields.get("delivered"))
        state.skills_delivered.update(delivered)
        loaded = _clean_names(fields.get("loaded"))
        state.skills_loaded.update(loaded)
        for name in loaded:
            digest = (fields.get("loaded_hashes") or {}).get(name)
            if digest and state.loaded_hashes.get(name, digest) != digest:
                digest = None  # one name, two versions: a gap beats a wrong version
            if name not in state.loaded_hashes or not digest:
                state.loaded_hashes[name] = digest
        for version in copy_delivered_versions(fields.get("delivered_versions"), names=delivered + loaded):
            name, digest = version.get("name"), version.get("hash")
            if name and digest:
                state.skills_delivered_versions.add((
                    name, digest, version.get("routing_id") or "",
                    version.get("skill_id") or "", version.get("version_id") or "",
                ))
        prompt = fields.get("prompt")
        if isinstance(prompt, dict) and prompt.get("agent"):
            # First record per agent wins: an invocation reads its prompt once.
            state.decimal_prompts.setdefault(prompt["agent"], dict(prompt))
            if not state.decimal_agent and prompt.get("decimal_agent"):
                # The first config in the invocation names its trace: the root agent's,
                # whose instruction is resolved first.
                state.decimal_agent = prompt["decimal_agent"]
    return True


#: Said once per agent when an agent config's run finds no DecimalAIPlugin tracing it:
#: `_record_adk` drops the record, and nothing else would say so.
_UNTRACED = (
    "this run is not traced, so its prompt version and skills are recorded nowhere: no "
    "DecimalAIPlugin is in the App that ran it. Add one: App(name=..., root_agent=..., "
    "plugins=[DecimalAIPlugin()]), from decimalai.adk."
)


def _note_adk_pending(run_key: Optional[str], pending: Any) -> None:
    state = _live_run(run_key)
    if state is not None and pending is not None:
        with _live_runs_lock:
            state.pending_routing.append(pending)


class _RunState:
    """Per-invocation trace accumulator, keyed by ADK ``invocation_id``."""

    __slots__ = (
        "trace_id", "agent_name", "root_agent_name", "started_at", "ended_at",
        "user_input_preview", "final_output_preview",
        "llm_calls", "spans", "pending_llm", "pending_tools", "agent_stack",
        "open_tool_calls", "failed_tool_calls",
        "status", "error_code", "error_message", "manifest",
        "routing_id", "skills_offered", "skills_delivered",
        "skills_delivered_versions", "skills_loaded", "loaded_hashes",
        "decimal_prompts", "decimal_agent", "pending_routing",
        "parent_trace_id", "on_trace", "native", "native_agents",
    )

    def __init__(self, *, agent_name: Optional[str], started_at: datetime):
        self.trace_id: UUID = uuid4()
        self.agent_name = agent_name
        self.parent_trace_id: Optional[str] = None
        self.on_trace: Optional[Callable[[RunTrace], None]] = None
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
        # The skills rails of this invocation, from the agent configs of its
        # agents (`_record_adk`), stamped onto the RunTrace in `_finalize`. Sets, so
        # a model turn that calls a tool reports each skill once.
        self.routing_id: Optional[str] = None
        self.skills_offered: set = set()
        self.skills_delivered: set = set()
        self.skills_delivered_versions: set[Tuple[str, str, str, str, str]] = set()
        self.skills_loaded: set = set()
        self.loaded_hashes: Dict[str, Optional[str]] = {}
        # Each config's prompt record, by ADK agent name, for that agent's span; and
        # the DecimalAI agent the trace is filed under, from the first config.
        self.decimal_prompts: Dict[str, Dict[str, Any]] = {}
        self.decimal_agent: Optional[str] = None
        # Background routing requests of this invocation, which `_finalize` waits for.
        self.pending_routing: List[Any] = []
        # Skills ADK's own SkillToolset listed and loaded in this invocation, as
        # decimalai._native resolved them; applied onto the RunTrace in `_finalize`.
        self.native = NativeLoads()
        # Names of the agents of this invocation that carry a SkillToolset, whose
        # system instruction may hold an EAGER toolset's catalog.
        self.native_agents: set = set()


def _plugin_class() -> Any:
    """Build (once) and return the concrete BasePlugin subclass.

    Deferred so that ``import decimalai.adk`` works without google-adk; the
    google-adk import only happens when a plugin is actually constructed.
    """
    global _PluginClass
    if _PluginClass is not None:
        return _PluginClass

    try:
        from google.adk.plugins.base_plugin import BasePlugin
    except ImportError:
        raise ImportError(
            'DecimalAIPlugin needs Google ADK: pip install "decimalai[adk]"'
        ) from None

    class _DecimalAIPlugin(BasePlugin):
        """ADK BasePlugin that emits one DecimalAI RunTrace per invocation."""

        def __init__(
            self,
            agent_name: Optional[str] = None,
            *,
            name: str = "decimalai",
            parent_trace_id: Optional[str] = None,
            on_trace: Optional[Callable[[RunTrace], None]] = None,
        ):
            super().__init__(name=name)
            self.agent_name = agent_name
            self.parent_trace_id = parent_trace_id
            self.on_trace = on_trace
            self._runs: Dict[str, _RunState] = {}
            self._lock = threading.Lock()

        # ── helpers ────────────────────────────────────────
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
            # The DecimalAI agent the trace is filed under, unless an agent config
            # names it during the run: this plugin's label, else the agent
            # init() named (or DECIMAL_AGENT_NAME), else ADK's own agent name (a
            # constrained Python identifier).
            from ._config import default_agent_name
            agent_name = self.agent_name or default_agent_name(None) or getattr(agent, "name", None)
            state = _RunState(agent_name=agent_name, started_at=_now())
            state.on_trace = self.on_trace
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
                with _live_runs_lock:
                    _live_runs[inv_id] = state
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
                await loop.run_in_executor(None, self._finalize_and_unbind, inv_id, state)
            except RuntimeError:
                self._finalize_and_unbind(inv_id, state)

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
            # callback knows the agent by name. Noted here so an EAGER toolset's
            # catalog in that agent's system instruction is recorded as offered.
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
                output=exception_message(error, 500) if error is not None else state.final_output_preview,
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
                state.error_message = exception_message(error)
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
                state.error_message = exception_message(error)
            await self._complete(inv_id)
            return None

        # ── model boundary ─────────────────────────────────
        async def before_model_callback(self, *, callback_context: Any, llm_request: Any):  # noqa: ANN001
            try:
                _save_conversations(callback_context)
            except Exception:  # noqa: BLE001 — keeping a record must never cost the request
                logger.debug("could not keep the conversation in session state", exc_info=True)
            state = self._get_run(getattr(callback_context, "invocation_id", None))
            if state is None:
                return None
            native = getattr(callback_context, "agent_name", None) in state.native_agents
            model = getattr(llm_request, "model", None)
            rendered_input: List[Dict[str, Any]] = []
            # The system instruction FIRST, and it is not optional bookkeeping.
            #
            # `llm_request.contents` holds the conversation turns only. ADK puts
            # the agent's own `instruction` — an agent config's prompt and
            # skill list included — in `config.system_instruction`, which is a
            # separate field. Recording only `contents` meant the trace omitted
            # the entire system prompt the model was actually shown.
            #
            # That is not a cosmetic gap. Every skill signal downstream reads the
            # rendered input: the SDK's own `infer_prompt_rungs`, the platform's
            # skill-activation detection, and the conformance contract's C8/C14.
            # Without it a trace claims a skill list it does not itself show,
            # which cannot be told apart from a list that was never shown (C8/C14
            # read exactly that the first time ADK's rail was graded, 2026-09-03).
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
                # A tool's result is what the model reads next: load_skill's body
                # reaches it here, so the trace shows the body the model was handed.
                for result in _function_responses(c):
                    rendered_input.append({"role": "tool", "content": result})
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
                rec.output_tokens = _output_tokens(usage)
                if rec.provider == "anthropic":
                    # ADK's AnthropicLlm folds the cache into prompt_token_count; it keeps the
                    # read as cached_content_token_count and the write as
                    # cache_creation_input_tokens. A Claude call keeps Anthropic's split.
                    rec.input_tokens, rec.cache_read_tokens, rec.cache_creation_tokens = (
                        anthropic_cache_split(
                            rec.input_tokens,
                            _usage_int(usage, "cached_content_token_count"),
                            _usage_int(usage, "cache_creation_input_tokens")))
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
                self._close_span(span, output=exception_message(error), error=True)
                state.spans.append(span)
            state.status = Status.ERROR
            state.error_message = exception_message(error)
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
                preview: Optional[str] = exception_message(error, _PREVIEW_LIMIT)
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
            state.error_message = exception_message(error)
            return None

        # ── finalize (runs in executor thread) ─────────────
        def _finalize_and_unbind(self, inv_id: Optional[str], state: _RunState) -> None:
            """Build and send the trace, then stop taking agent_config() records for it."""
            try:
                self._finalize(state)
            finally:
                if inv_id:
                    with _live_runs_lock:
                        if _live_runs.get(inv_id) is state:
                            del _live_runs[inv_id]

        def _finalize(self, state: _RunState) -> None:
            from . import _config
            if not _config._is_enabled():
                return
            # The invocation's background routing requests, so its trace names its
            # own decision (bounded: a slow platform must not hold the trace back).
            deadline = _now().timestamp() + _PENDING_ROUTING_WAIT_S
            with _live_runs_lock:
                pending = list(state.pending_routing)
            for future in pending:
                try:
                    future.result(timeout=max(0.0, deadline - _now().timestamp()))
                except Exception:
                    logger.debug("background routing did not finish before the trace was built", exc_info=True)
            with _live_runs_lock:
                if state.decimal_agent:
                    state.agent_name = state.decimal_agent
                # A body the model pulled with load_skill was delivered; a body in
                # the prompt was offered there too.
                loaded = set(state.skills_loaded)
                delivered = set(state.skills_delivered) | loaded
                offered = set(state.skills_offered) | set(state.skills_delivered)
                loaded_hashes = dict(state.loaded_hashes)
                prompts = {k: dict(v) for k, v in state.decimal_prompts.items()}
                versions = sorted(state.skills_delivered_versions)
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
                trace = RunTrace(
                    id=state.trace_id,
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
                    # `skills_delivered` is the subset whose text reached the
                    # model, `skills_loaded_by_agent` the part it asked for.
                    routing_id=state.routing_id,
                    skills_offered_in_prompt=sorted(offered),
                    skills_delivered=sorted(delivered),
                    skills_loaded_by_agent=sorted(loaded),
                    skills_delivered_versions=[
                        {"name": name, "hash": digest,
                         **({"routing_id": route_id} if route_id else {}),
                         **({"skill_id": skill_id, "version_id": version_id} if skill_id and version_id else {})}
                        for name, digest, route_id, skill_id, version_id in versions
                    ],
                    # The version of each body the model loaded, where the load
                    # carried one: what joins a measured lift to its version.
                    active_skills=[
                        {"name": name, "hash": loaded_hashes[name]}
                        for name in sorted(loaded) if loaded_hashes.get(name)
                    ],
                )
                # Each config's prompt version and variable names (never values)
                # on its agent's span.
                attach_prompt_records(trace.spans, prompts)
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
                    # One line: an unreachable platform is handled, not a crash.
                    logger.warning(
                        "decimalai: could not register %s's manifest (%s: %s); its traces "
                        "are held and retried in the background.",
                        agent, type(exc).__name__, str(exc)[:200],
                    )
                    logger.debug("ADK manifest registration for %s failed", agent, exc_info=True)
                    _manifest_ids[agent] = snapshot.id
                    # Hand the send side something to retry WITH, so the trace is held
                    # rather than posted under an id the platform never stored.
                    _pending_manifests[agent] = (snapshot, exc)
                    # The hash is already committed to this agent's tracker;
                    # without the reset every later run short-circuits on it and
                    # registration is never retried.
                    tracker.reset()
            return _manifest_ids.get(agent)

    _PluginClass = _DecimalAIPlugin
    return _PluginClass


def DecimalAIPlugin(  # noqa: N802 — factory presents as a class for ergonomics
    agent_name: Optional[str] = None,
    *,
    name: str = "decimalai",
    parent_trace_id: Optional[str] = None,
    on_trace: Optional[Callable[[RunTrace], None]] = None,
) -> Any:
    """Construct a DecimalAI ADK plugin, for ``App(..., plugins=[DecimalAIPlugin()])``.

    Every invocation of a runner built from that app becomes one DecimalAI trace.
    An ``agent_config()`` agent run without it is not traced, and says so once.

    Args:
        agent_name: The DecimalAI agent the traces are filed under. Default: the
            agent ``decimalai.init()`` named (or ``DECIMAL_AGENT_NAME``), else the
            ADK root agent's name. An invocation whose agent was given
            ``agent_config()``'s instruction is filed under that DecimalAI agent.
        name: ADK plugin name (must be unique among a Runner's plugins).
        parent_trace_id: When this Runner runs as a sub-agent of another,
            the parent's trace id — links the child traces in the backend.
        on_trace: Optional observer called once on the exporter worker thread
            with a detached completed trace, before export. Exceptions are
            isolated from tracing and execution.

    ADK's ``App`` is its supported place for plugins; ``Runner(plugins=...)`` still
    works, but ADK deprecates it.

    ⚠ This factory's signature is a SECOND place every parameter must be
    listed. It forwards by explicit keyword, so a parameter added to
    ``_DecimalAIPlugin.__init__`` and not to this signature does not merely
    get ignored — the call raises TypeError and the plugin is never built, so
    the Runner traces NOTHING. That is not hypothetical: adding a parameter to
    the class alone once took every ADK conformance item from green to "the
    documented snippet ran and NOTHING was POSTed".
    """
    return _plugin_class()(
        agent_name=agent_name, name=name,
        parent_trace_id=parent_trace_id, on_trace=on_trace,
    )


def DecimalaiPlugin(*args: Any, **kwargs: Any) -> NoReturn:  # noqa: N802 — the factory's old name
    """Renamed in decimalai 0.21: :func:`DecimalAIPlugin`."""
    raise TypeError(
        "decimalai.adk.DecimalaiPlugin() was renamed DecimalAIPlugin() in decimalai 0.21; "
        "it takes the same arguments: from decimalai.adk import DecimalAIPlugin"
    )



# ── agent_config(): an agent's prompt and skills, as ADK objects ─────────

#: The ADK invocation whose instruction or load_skill call is running in this context.
#: Set around each call ADK makes into a config, so its record and its loads reach that
#: invocation's trace (`_record_adk`).
_RUN: "ContextVar[Optional[str]]" = ContextVar("decimalai_adk_run", default=None)

#: The names `load_skill` declares for the request being built (the run's own).
_DECLARED_NAMES: "ContextVar[Optional[List[str]]]" = ContextVar("decimalai_adk_load_skill_names", default=None)

#: The session state key a Claude conversation's records live under, by DecimalAI agent.
_STATE_KEY = "_decimalai_conversation"

#: Records `DecimalAIPlugin` has yet to write into their session's state, by invocation.
_UNSAVED: "OrderedDict[str, Dict[str, Dict[str, Any]]]" = OrderedDict()
_UNSAVED_MAX = 256
_unsaved_lock = threading.Lock()

#: Conversations one config remembers in this process, newest last.
_MAX_CONVERSATIONS = 1024


def _model_name(agent: Any) -> Optional[str]:
    """The model an ADK agent calls, by name (inherited from its parent when unset)."""
    for attr in ("canonical_model", "model"):
        try:
            model = getattr(agent, attr, None)
        except Exception:  # noqa: BLE001 — canonical_model raises when no ancestor has one
            continue
        name = model if isinstance(model, str) else getattr(model, "model", None)
        if isinstance(name, str) and name:
            return name
    return None


def _session_id(ctx: Any) -> Optional[str]:
    session = getattr(ctx, "session", None)
    if session is None:
        session = getattr(getattr(ctx, "_invocation_context", None), "session", None)
    session_id = getattr(session, "id", None)
    return session_id if isinstance(session_id, str) and session_id else None


def _note_unsaved(invocation_id: Optional[str], agent_name: str, record: ConversationRecord) -> None:
    """Hand a record to the plugin, which writes it into the session's state."""
    if not invocation_id:
        return
    with _unsaved_lock:
        _UNSAVED.setdefault(invocation_id, {})[agent_name] = record.to_state(agent_name)
        _UNSAVED.move_to_end(invocation_id)
        while len(_UNSAVED) > _UNSAVED_MAX:
            _UNSAVED.popitem(last=False)


def _save_conversations(callback_context: Any) -> None:
    """Write this invocation's records into its session's state (a plugin callback's
    state changes travel with the model's response event, so they are persisted)."""
    with _unsaved_lock:
        unsaved = _UNSAVED.pop(getattr(callback_context, "invocation_id", None), None)
    if not unsaved:
        return
    state = callback_context.state
    kept = dict(state.get(_STATE_KEY) or {})
    if any(kept.get(name) != record for name, record in unsaved.items()):
        kept.update(unsaved)
        state[_STATE_KEY] = kept


def _invocation_agent(ctx: Any) -> Any:
    """The ADK agent a context belongs to.

    ADK's context names the agent (``agent_name``) and hands the object over only
    through its invocation context; without that, a stand-in carrying the name.
    """
    agent = getattr(getattr(ctx, "_invocation_context", None), "agent", None)
    return agent if agent is not None else SimpleNamespace(name=getattr(ctx, "agent_name", None))


def _other_load_skill(agent: Any, config: Any) -> bool:
    """Whether ``agent`` has a tool named load_skill that is not ``config``'s own.

    A function tool by its ``__name__``, any other tool by its ``name``. Toolsets
    resolve their tools per request and are not looked into.
    """
    from .skill_router import LOAD_SKILL_TOOL_NAME

    ours = _load_skill_tool_cls
    for tool in getattr(agent, "tools", None) or ():
        if ours is not None and isinstance(tool, ours) and tool.config is config:
            continue
        name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
        if name == LOAD_SKILL_TOOL_NAME:
            return True
    return False


_load_skill_tool_cls: Any = None
_load_skill_tool_lock = threading.Lock()


def _load_skill_tool_class() -> Any:
    """The ADK tool class behind ``support.tools``, built once (google-adk is optional)."""
    global _load_skill_tool_cls
    with _load_skill_tool_lock:
        if _load_skill_tool_cls is not None:
            return _load_skill_tool_cls
        from google.adk.tools.base_tool import BaseTool
        from google.genai import types

        from .skill_router import LOAD_SKILL_TOOL_DESCRIPTION, LOAD_SKILL_TOOL_NAME

        class DecimalaiLoadSkillTool(BaseTool):
            """``load_skill`` for one agent config: a listed skill's text."""

            def __init__(self, config: Any) -> None:
                super().__init__(name=LOAD_SKILL_TOOL_NAME, description=LOAD_SKILL_TOOL_DESCRIPTION)
                self.config = config

            def _get_declaration(self) -> Any:
                names = _DECLARED_NAMES.get()
                if names is None:
                    with self.config._lock:
                        names = list(self.config._tool_names)
                return types.FunctionDeclaration(
                    name=self.name, description=self.description,
                    parameters_json_schema=self.config._tool_schema(names),
                )

            async def process_llm_request(self, *, tool_context: Any, llm_request: Any) -> None:
                """Offer load_skill only beside a skill list (ADK asks before each request)."""
                run_key = getattr(tool_context, "invocation_id", None)
                token = _RUN.set(run_key)
                try:
                    state = await self.config._prepare_run(
                        tool_context, _invocation_agent(tool_context), run_key)
                finally:
                    _RUN.reset(token)
                if state["names"] and state["tool"]:
                    # The run's own names: a Claude conversation keeps its first tool.
                    names = _DECLARED_NAMES.set(list(state.get("tool_names") or []))
                    try:
                        llm_request.append_tools([self])
                    finally:
                        _DECLARED_NAMES.reset(names)

            async def run_async(self, *, args: Dict[str, Any], tool_context: Any) -> Any:
                token = _RUN.set(getattr(tool_context, "invocation_id", None))
                try:
                    return await self.config._load(str((args or {}).get("name") or "").strip())
                finally:
                    _RUN.reset(token)

        DecimalaiLoadSkillTool.__module__ = __name__
        _load_skill_tool_cls = DecimalaiLoadSkillTool
        return _load_skill_tool_cls


class ADKAgentConfig(AgentBinding):
    """One agent's prompt and skills in DecimalAI, as ADK objects.

    Made by :func:`agent_config`. Put its two pieces in your own ``LlmAgent``::

        support = agent_config("support", fallback=BASE_INSTRUCTION)
        agent = LlmAgent(name="support", model="gemini-3.5-flash",
                         instruction=support.instruction, tools=[lookup_order, *support.tools])

    Attributes:
        name: The agent's canonical DecimalAI name.
    """

    _log = logger
    _VALUES_HINT = "Put them in the session's state, as attributes or dict keys."

    def __init__(self, prompt: Any, *, fallback: Optional[str], on_unavailable: str) -> None:
        super().__init__(prompt, fallback=fallback, on_unavailable=on_unavailable)
        self._tool_names: List[str] = []
        self._tool: Any = None
        #: Claude conversations this process has seen: (session id, agent) -> record.
        self._chats: "OrderedDict[Tuple[str, Optional[str]], ConversationRecord]" = OrderedDict()

    def __repr__(self) -> str:
        return (
            f"ADKAgentConfig(name={self.name!r}, prompt_version={self.prompt_version!r}, "
            f"skills={self.skill_names!r})"
        )

    # A config is one agent's live binding (its prompt copy, its runs, its locks):
    # a copied LlmAgent shares it rather than splitting it in two.
    def __copy__(self) -> "ADKAgentConfig":
        return self

    def __deepcopy__(self, memo: Any) -> "ADKAgentConfig":
        return self

    # ── what the developer hands to ADK ─────────────────────
    async def instruction(self, ctx: Any) -> str:
        """The agent's prompt, filled, then its skill list: an ADK instruction provider.

        Pass it as ``LlmAgent(instruction=support.instruction)``; ADK calls it before
        each model request. ``{{variables}}`` are filled from the session's state.
        """
        run_key = getattr(ctx, "invocation_id", None)
        agent = _invocation_agent(ctx)
        token = _RUN.set(run_key)
        try:
            state = await self._prepare_run(ctx, agent, run_key)
            if state["text"] is None:
                state["text"] = self._compose(ctx, agent, run_key, state)
            return state["text"]
        finally:
            _RUN.reset(token)

    @property
    def tools(self) -> List[Any]:
        """``[load_skill]``, for ``LlmAgent(tools=[..., *support.tools])``."""
        with self._lock:
            if self._tool is None:
                self._tool = _load_skill_tool_class()(self)
            return [self._tool]

    # ── the seams AgentBinding asks for ─────────────────────
    def _router(self) -> Any:
        return _get_skill_router()

    def _run_key(self) -> Optional[str]:
        return _RUN.get()

    def _message(self, context: Any, agent: Any) -> Optional[str]:
        return _content_to_text(getattr(context, "user_content", None)) or None

    def _values(self, context: Any) -> Any:
        try:
            return dict(getattr(context, "state", None) or {})
        except Exception:  # noqa: BLE001 — a state we cannot read fills nothing
            return {}

    def _skills_blocked(self, agent: Any, *, tool: bool = True, inject: bool = False) -> Optional[str]:
        base = super()._skills_blocked(agent, tool=tool, inject=inject)
        if base:
            return base
        if _has_skill_toolset(agent):
            return ("it carries ADK's own SkillToolset, a catalog of its own, so DecimalAI adds "
                    "neither its skill list nor load_skill. Each skill the toolset loads is "
                    "recorded as an activation.")
        if tool and _other_load_skill(agent, self):
            return ("it has another tool named load_skill, so DecimalAI adds neither its skill "
                    "list nor its load_skill. Rename that tool to give the agent its skills.")
        return None

    def _record(self, **fields: Any) -> bool:
        kept = _record_adk(_RUN.get(), fields)
        if not kept and "prompt" in fields:
            # The run's own record, made as its instruction is composed, and no plugin is
            # tracing its invocation. A late background decision or load finds its run gone
            # too, but carries no prompt. With tracing off the plugin records nothing either.
            from ._config import _is_enabled

            if _is_enabled():
                self._warn_once(fields.get("agent"), _UNTRACED)
        return kept

    def _note_pending(self, run_key: Optional[str], pending: Any) -> None:
        _note_adk_pending(run_key, pending)

    def _stamp_routing(self, run_key: Optional[str], routing_id: str) -> None:
        _record_adk(run_key, {"routing_id": routing_id})

    def _on_menu(self, names: List[str]) -> None:
        with self._lock:
            self._tool_names = list(names)

    def _conversation(self, context: Any, agent: Any) -> Optional[ConversationRecord]:
        session_id = _session_id(context)
        if session_id is None or not is_claude(_model_name(agent)):
            return None
        kept = getattr(context, "state", None)
        kept = kept.get(_STATE_KEY) if kept is not None else None
        if isinstance(kept, Mapping):
            record = ConversationRecord.from_state(kept.get(self.name), self.name)
            if record is not None:
                return record
        with self._lock:
            record = self._chats.get((session_id, self._agent_label(agent)))
        if record is not None:
            _note_unsaved(getattr(context, "invocation_id", None), self.name, record)
        return record

    def _keep_conversation(self, context: Any, agent: Any, run_key: Optional[str],
                           record: ConversationRecord) -> None:
        session_id = _session_id(context)
        if session_id is None or not is_claude(_model_name(agent)):
            return
        with self._lock:
            self._chats[(session_id, self._agent_label(agent))] = record
            while len(self._chats) > _MAX_CONVERSATIONS:
                self._chats.popitem(last=False)
        _note_unsaved(run_key, self.name, record)

    def _serve_load(self, name: str) -> str:
        from ._agent import _serve_and_record

        run_key = self._run_key()
        return _serve_and_record(
            self._router(), name, scope=run_key, agent_name=self.name,
            record=lambda **fields: _record_adk(run_key, fields),
        )


def agent_config(
    agent_name: str,
    *,
    fallback: Optional[str] = None,
    on_unavailable: str = "continue",
) -> ADKAgentConfig:
    """Pull an agent's prompt and skills from DecimalAI, for your own ``LlmAgent``.

    ::

        from google.adk.agents import LlmAgent
        from decimalai.adk import agent_config

        support = agent_config("support", fallback=BASE_INSTRUCTION)
        agent = LlmAgent(name="support", model="gemini-3.5-flash",
                         instruction=support.instruction, tools=[lookup_order, *support.tools])

    Reads the agent's saved prompt here and keeps a
    local copy, re-read in the background at most once a minute. Each invocation is
    routed on its own user message: the first before its first model request, later
    ones from the local copy while their decision is recorded in the background, and
    every one of an agent with more skills than fit the list. ``{{variables}}`` are
    filled from the session's state (attributes or dict keys). Traces come from
    :func:`DecimalAIPlugin` in ``App(plugins=[...])``: an invocation of this agent is
    filed under ``support.name`` with its prompt version and its skills. A run by a
    runner without the plugin is not traced and logs one warning saying so. Initializes
    the SDK from ``DECIMAL_API_KEY`` when ``decimalai.init()`` has not been called.

    With the ``load_skill`` tool switched off (``DECIMALAI_LOAD_SKILL_TOOL=0`` or
    ``init(load_skill_tool=False)``), every invocation is routed and the text of the
    skills picked for it goes into the instruction instead (``inject_skill_body``).

    With no API key and no ``init()``, the agent runs keyless, as
    :func:`decimalai.agent_config` describes: its ``fallback`` is the prompt, each
    run lists the skills the project lists in ``.decimal/skills.toml``, and nothing
    is sent. With a key, an agent the workspace has no agent by is created.

    Args:
        agent_name: The agent's name in DecimalAI.
        fallback: The instruction to use when the agent has no saved prompt, when no
            API key is set, or when DecimalAI cannot be reached at startup. With a key
            and without one, an unreachable DecimalAI raises here rather than starting
            an agent with no instruction.
        on_unavailable: ``"continue"`` (default) runs without skills when DecimalAI
            never returned them, and the trace says so; ``"fail"`` raises before the
            model request instead.

    Raises:
        ImportError: Google ADK is not installed.
        DecimalConfigError: no API key is set in a project ``.decimal/link.toml``
            connects to a workspace.
        ValueError: ``on_unavailable`` is neither ``"continue"`` nor ``"fail"``.
    """
    if on_unavailable not in ("continue", "fail"):
        raise ValueError("on_unavailable must be 'continue' or 'fail'")
    try:
        import google.adk  # noqa: F401 — before any setup or network call
    except ImportError:
        raise ImportError('agent_config() needs Google ADK: pip install "decimalai[adk]"') from None
    from ._agent import _open_prompt

    prompt = _open_prompt(agent_name, fallback)
    return ADKAgentConfig(prompt, fallback=fallback, on_unavailable=on_unavailable)


def load_agent(*args: Any, **kwargs: Any) -> NoReturn:
    """Renamed in decimalai 0.21: :func:`agent_config`."""
    raise TypeError(
        "decimalai.adk.load_agent() was renamed agent_config() in decimalai 0.21; "
        "it takes the same arguments and returns the same config: "
        "from decimalai.adk import agent_config"
    )
