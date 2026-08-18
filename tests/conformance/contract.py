"""The conformance contract — one spec, every framework.

This file is the specification. Every assertion the suite makes lives here, and
nowhere else. Drivers run their framework's documented snippet; these functions
grade the result. A framework does not get its own assertions, ever: if an item
seems wrong for one adapter, either the contract is wrong (fix it here, for
everybody) or the adapter is wrong (which is a finding, not a test bug).

Each item is one function taking the :class:`~tests.conformance.harness.Observation`
— the payloads captured ON THE WIRE by the probe, plus what the driver was asked
to produce — and returning a :class:`Result` with a precise message. Nothing
reads adapter internals: only bytes that crossed a socket, and public SDK
surfaces (``decimalai.export_status()``).

Item list (v1)
--------------
==========  ==========================================================
C1          at least one trace reaches the wire
C2          every trace passes the backend's own required-field validation
C3          llm_calls carry a model name and plausible token counts
C4          previews and rendered_input carry the real prompt/completion text
C5          a multi-step run has >1 span, a parent link, and distinct names
C6          agent_name is the one asked for, and the manifest belongs to it
C7          the same agent run twice does not mint a second manifest version
C7b         a degenerate run does not fabricate a manifest change
C8          skills rail: routing_id, offered names, and the prompt agrees
C9          N concurrent runs produce N uncontaminated traces
C10         a failing run produces exactly ONE trace, marked errored
C11         a run writes nothing into the working directory
C12         when the adapter cannot do what was asked, it says so
C13         nothing is recorded as activated that the model did not ask for
C13b        a body the model DID pull is not silently dropped
==========  ==========================================================

C7b is the second clause of C7 ("a degenerate run does not fabricate a breaking
change") split into its own function so a framework that has no degenerate form
can declare that one clause N/A without silencing the first.

C13b is the mirror of C13 rather than a clause of it, for the same reason: a
prompt-injection rail genuinely has no model-initiated body pull to record, and
must be able to declare that one half N/A without silencing the half that
forbids fabricating an activation — which applies to every rail there is.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .harness import Observation, Phase

PASS = "pass"
FAIL = "fail"
NA = "na"


@dataclass(frozen=True)
class Result:
    item: str
    title: str
    status: str
    message: str


def _pass(item: str, title: str, message: str = "") -> Result:
    return Result(item, title, PASS, message)


def _fail(item: str, title: str, message: str) -> Result:
    return Result(item, title, FAIL, message)


def _summarize(problems: Iterable[str], limit: int = 3) -> str:
    """Join distinct problems, capped — a failure repeated 8 times reads once.

    Ordering is preserved (first-seen wins) so the message leads with the first
    thing that went wrong rather than whatever sorts first.
    """
    seen: List[str] = []
    for p in problems:
        if p not in seen:
            seen.append(p)
    head = "; ".join(seen[:limit])
    extra = len(seen) - limit
    return f"{head} (+{extra} more of the same kind)" if extra > 0 else head


# ── payload readers (generic; no framework knowledge) ────────────────────────

_ROLE_WORDS = {"system", "user", "assistant", "human", "ai", "tool", "function"}
#: Two unambiguous "this is a Python object, not text" shapes:
#: ``<pkg.Class object at 0x7f…>`` and ``Class(kwarg=…``.
#: Deliberately narrow. A preview that embeds structured JSON still carries the
#: text and is NOT flagged — inventing a stricter bar than every incumbent
#: adapter can meet turns C4 into noise. See README, "what v1 does not cover".
_REPR_RE = re.compile(
    r"<[\w.]+ object at 0x[0-9a-f]+>|\b[A-Za-z_][\w.]*\([a-z_]\w*="
)


def _strings(value: Any, out: List[str]) -> None:
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            _strings(v, out)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _strings(v, out)


def _rendered_text(trace: Dict[str, Any]) -> str:
    """Everything the model was actually shown, as one string."""
    out: List[str] = []
    for call in trace.get("llm_calls") or []:
        _strings(call.get("rendered_input"), out)
    return "\n".join(out)


#: Role stamped on a message that arrived as a bare string, so it carries no
#: role of its own. Deliberately a name no provider uses, so it can never
#: collide with a real role and never satisfy a role-scoped clause (C13's
#: clause 2 in particular: text with no attributable speaker is not evidence
#: that the MODEL asked for anything).
UNSTRUCTURED_ROLE = "<unstructured>"


def _rendered_messages(trace: Dict[str, Any]) -> List[Any]:
    """Every message object the model was shown, in order, across all llm_calls.

    Unlike :func:`_rendered_text` this keeps the message boundaries, because
    "which SPEAKER said this" is the whole difference between a skill being
    pasted into the system prompt and the model asking for it.
    """
    out: List[Any] = []
    for call in trace.get("llm_calls") or []:
        rendered = call.get("rendered_input")
        if isinstance(rendered, (list, tuple)):
            out.extend(rendered)
        elif rendered is not None:
            out.append(rendered)
    return out


def _role_messages(trace: Dict[str, Any]) -> List[Tuple[str, str]]:
    """``(role, text)`` per message. ``_rendered_text`` discards the role.

    ``text`` is every string reachable inside the message, joined — so a
    provider that nests content as ``[{"type": "text", "text": …}]`` is covered,
    and so is the openai-agents ``{"role": "assistant", "type": "function_call",
    "name": "load_skill", "content": "{…}"}`` shape.

    A ``rendered_input`` that is a bare string yields ``UNSTRUCTURED_ROLE``: it
    carries no role, so it can never satisfy a clause that asks who spoke. That
    is correct, not a gap — an adapter that flattens its prompt to one string
    has destroyed the evidence, and inferring a role from flattened text is
    exactly the guess C13 exists to forbid.
    """
    out: List[Tuple[str, str]] = []
    for msg in _rendered_messages(trace):
        parts: List[str] = []
        _strings(msg, parts)
        text = "\n".join(parts)
        role = (
            str(msg.get("role", "")).lower() if isinstance(msg, dict)
            else UNSTRUCTURED_ROLE
        )
        out.append((role or UNSTRUCTURED_ROLE, text))
    return out


def _tool_call_entries(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every ``llm_calls[].tool_calls[]`` entry — the model's own tool requests."""
    out: List[Dict[str, Any]] = []
    for call in trace.get("llm_calls") or []:
        for entry in call.get("tool_calls") or []:
            if isinstance(entry, dict):
                out.append(entry)
    return out


def _tool_spans(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Spans recording a tool execution."""
    return [
        s for s in (trace.get("spans") or [])
        if isinstance(s, dict) and s.get("span_type") == "tool"
    ]


def _flat(value: Any) -> str:
    """Every string inside ``value``, joined — for searching a nested preview."""
    out: List[str] = []
    _strings(value, out)
    return "\n".join(out)


def _output_text(trace: Dict[str, Any]) -> str:
    out: List[str] = []
    for call in trace.get("llm_calls") or []:
        _strings(call.get("output"), out)
    return "\n".join(out)


def _degenerate_preview(text: Optional[str]) -> Optional[str]:
    """Why this preview is junk, or None if it looks like real content."""
    if text is None:
        return "is null"
    if not text.strip():
        return "is empty"
    if text.strip().strip("'\"").lower() in _ROLE_WORDS:
        return f"is a bare role name ({text!r})"
    if _REPR_RE.search(text):
        return f"is a Python repr ({text[:80]!r})"
    return None


def _agent_names_expected(phase: Phase) -> set:
    return {c.agent_name for c in phase.ctxs}


def _ran_phases(obs: Observation) -> List[Phase]:
    return [p for p in obs.phases.values() if p.ran]


def _minted_for(obs: Observation, phases: Sequence[str], agent_name: str) -> List[str]:
    """Manifest ids MINTED (is_new) for ``agent_name`` during ``phases``."""
    minted: List[str] = []
    for name in phases:
        phase = obs.phases[name]
        for r in phase.manifest_posts:
            body = r.body if isinstance(r.body, dict) else {}
            resp = r.response if isinstance(r.response, dict) else {}
            if body.get("agent_name") == agent_name and resp.get("is_new"):
                minted.append(resp.get("manifest_id"))
    return minted


# ── the items ────────────────────────────────────────────────────────────────


def c1_emits(obs: Observation) -> Result:
    """At least one trace reaches the wire.

    Reaching the wire means the POST happened at all — whether the backend would
    have kept it is C2's question. This is the item that would have caught the
    LlamaIndex adapter shipping for months emitting nothing with a green suite.
    """
    item, title = "C1", "emits"
    main = obs.phase("main")
    if not main.attempted:
        seen = {f"{r.method} {r.path}" for r in main.requests}
        return _fail(
            item, title,
            f"the documented snippet ran and NOTHING was POSTed to /api/v1/traces. "
            f"Requests the adapter did make: {sorted(seen) or 'none at all'}",
        )
    return _pass(item, title, f"{len(main.attempted)} trace(s) POSTed")


def c2_ingest_valid(obs: Observation) -> Result:
    """Every trace passes the backend's own required-field validation.

    The probe applies the port of ``trace_service._validate_payload``. A failure
    here means the trace left the SDK and the real backend would have answered
    400 — the "it works locally, nothing shows up in the UI" defect.
    """
    item, title = "C2", "ingest_valid"
    rejected = obs.all_rejected
    if rejected:
        lines = []
        for r in rejected[:5]:
            lines.append(f"  HTTP {r.status} {r.path}: {'; '.join(r.errors)}")
        return _fail(
            item, title,
            f"{len(rejected)} trace POST(s) the backend would REJECT:\n" + "\n".join(lines),
        )
    if not obs.all_attempted:
        return _fail(item, title, "no trace was POSTed at all — nothing to validate (see C1)")
    return _pass(item, title, f"{len(obs.all_attempted)} trace(s) pass backend validation")


def c3_llm_calls(obs: Observation) -> Result:
    """``llm_calls`` present, with a model name and plausible token counts.

    Exact counts belong to the live tier; a stub model reports deterministic
    numbers, so here we assert presence and plausibility only.
    """
    item, title = "C3", "llm_calls"
    traces = obs.phase("main").attempted
    if not traces:
        return _fail(item, title, "no trace to inspect (see C1)")
    with_calls = [t for t in traces if t.get("llm_calls")]
    if not with_calls:
        return _fail(
            item, title,
            "no trace carried any llm_calls — the model turn never reached the wire",
        )
    problems: List[str] = []
    for t in with_calls:
        for i, call in enumerate(t["llm_calls"]):
            if not call.get("model_name"):
                problems.append(f"llm_calls[{i}]: model_name missing")
            for field in ("input_tokens", "output_tokens"):
                value = call.get(field)
                if value is None:
                    problems.append(f"llm_calls[{i}]: {field} absent")
                elif not isinstance(value, int) or value <= 0:
                    problems.append(f"llm_calls[{i}]: {field}={value!r} is not a positive int")
    if problems:
        return _fail(item, title, _summarize(problems))
    total = sum(len(t["llm_calls"]) for t in with_calls)
    return _pass(item, title, f"{total} llm_call(s) with model + token fields")


def c4_content(obs: Observation) -> Result:
    """Previews and ``rendered_input`` carry the real prompt/completion TEXT.

    Asserted against the sentinels the driver was told to put in the prompt and
    to have the stub model answer, which is how ``"system"`` / ``"assistant"`` /
    ``<object at 0x…>`` previews get caught without knowing the framework.
    """
    item, title = "C4", "content"
    traces = obs.phase("main").attempted
    if not traces:
        return _fail(item, title, "no trace to inspect (see C1)")
    ctx = obs.ctx
    problems: List[str] = []

    rendered = "\n".join(_rendered_text(t) for t in traces)
    outputs = "\n".join(_output_text(t) for t in traces)
    if ctx.prompt_sentinel not in rendered:
        problems.append(
            f"rendered_input does not contain the prompt text ({ctx.prompt_sentinel!r}); "
            f"got {rendered[:200]!r}"
        )
    if ctx.reply_sentinel not in outputs:
        problems.append(
            f"llm_call output does not contain the completion text "
            f"({ctx.reply_sentinel!r}); got {outputs[:200]!r}"
        )

    ins = [t.get("user_input_preview") for t in traces]
    outs = [t.get("final_output_preview") for t in traces]
    if not any(p and ctx.prompt_sentinel in p for p in ins):
        junk = [_degenerate_preview(p) for p in ins]
        problems.append(
            f"no user_input_preview contains the prompt text; previews: "
            f"{[j or 'looks like content but not the prompt' for j in junk]}"
        )
    if not any(p and ctx.reply_sentinel in p for p in outs):
        junk = [_degenerate_preview(p) for p in outs]
        problems.append(
            f"no final_output_preview contains the completion text; previews: "
            f"{[j or 'looks like content but not the completion' for j in junk]}"
        )

    for t in traces:
        for name, value in (
            ("user_input_preview", t.get("user_input_preview")),
            ("final_output_preview", t.get("final_output_preview")),
        ):
            why = _degenerate_preview(value)
            if why:
                problems.append(f"{name} {why}")
        for i, call in enumerate(t.get("llm_calls") or []):
            # The completion itself, not a preview of it: a consumer reading
            # llm_call.output.content to show "what the model said" must not get
            # a repr of the framework's response object.
            content = (call.get("output") or {}).get("content")
            if isinstance(content, str) and _REPR_RE.search(content):
                problems.append(
                    f"llm_calls[{i}].output.content is a Python repr, not the "
                    f"completion ({content[:120]!r})"
                )
            for j, entry in enumerate(call.get("rendered_input") or []):
                shown = entry.get("content") if isinstance(entry, dict) else None
                if isinstance(shown, str) and _REPR_RE.search(shown):
                    problems.append(
                        f"llm_calls[{i}].rendered_input[{j}].content is a Python repr, "
                        f"not the message text ({shown[:120]!r})"
                    )
        for span in t.get("spans") or []:
            for name in ("input_preview", "output_preview"):
                value = span.get(name)
                if value is None:
                    continue  # absence is C5's problem, not junk
                why = _degenerate_preview(value)
                if why:
                    problems.append(f"span {span.get('name')!r} {name} {why}")

    if problems:
        return _fail(item, title, _summarize(problems))
    return _pass(item, title, "prompt and completion text present in previews and rendered_input")


def c5_structure(obs: Observation) -> Result:
    """A multi-step run produces >1 span, a parent link, and distinguishable names."""
    item, title = "C5", "structure"
    traces = obs.phase("main").attempted
    if not traces:
        return _fail(item, title, "no trace to inspect (see C1)")
    best = max(traces, key=lambda t: len(t.get("spans") or []))
    spans = best.get("spans") or []
    if len(spans) < 2:
        return _fail(
            item, title,
            f"a multi-step run produced {len(spans)} span(s) — the trace is flat, so the "
            f"steps cannot be told apart in the waterfall",
        )
    ids = {s.get("id") for s in spans}
    linked = [s for s in spans if s.get("parent_span_id") in ids]
    if not linked:
        return _fail(
            item, title,
            f"{len(spans)} spans and not one parent link — every span claims to be a root",
        )
    names = [s.get("name") for s in spans]
    if len({n for n in names if n}) < 2:
        return _fail(item, title, f"span names are indistinguishable: {names}")
    return _pass(
        item, title,
        f"{len(spans)} spans, {len(linked)} parent link(s), {len(set(names))} distinct names",
    )


def c6_identity(obs: Observation) -> Result:
    """The trace's agent_name is the one asked for, and its manifest belongs to it.

    Two defects at once: an adapter that keeps shipping the FIRST agent's name
    after a second agent runs, and one that stamps agent A's manifest_id on
    agent B's trace (a process-global manifest id).
    """
    item, title = "C6", "identity"
    problems: List[str] = []
    checked = 0
    for phase in _ran_phases(obs):
        expected = _agent_names_expected(phase)
        for t in phase.attempted:
            checked += 1
            name = t.get("agent_name")
            if name not in expected:
                problems.append(
                    f"[{phase.name}] agent_name={name!r}, driver asked for one of {sorted(expected)}"
                )
                continue
            mid = t.get("manifest_id")
            if not mid:
                continue  # absence is C2's rejection, not an identity mix-up
            owner = obs.probe.manifest_owner(mid)
            if owner is None:
                problems.append(
                    f"[{phase.name}] {name!r} carries manifest_id {mid} which was never "
                    f"registered — it belongs to no agent"
                )
            elif owner != name:
                problems.append(
                    f"[{phase.name}] {name!r} carries manifest_id {mid}, registered for "
                    f"{owner!r}"
                )
    if problems:
        return _fail(item, title, _summarize(problems))
    return _pass(item, title, f"{checked} trace(s) name their own agent and manifest")


def c7_manifest_stable(obs: Observation) -> Result:
    """Running the same agent twice does not mint a second manifest version."""
    item, title = "C7", "manifest_stable"
    minted = _minted_for(obs, ("main", "repeat"), obs.ctx.agent_name)
    if not minted:
        return _fail(
            item, title,
            "the agent ran twice and no manifest was ever registered for it — there is "
            "nothing for a trace's manifest_id to point at",
        )
    if len(minted) > 1:
        return _fail(
            item, title,
            f"two identical runs minted {len(minted)} manifest versions ({minted}) — the "
            f"version history now shows a change that never happened",
        )
    return _pass(item, title, f"one manifest ({minted[0]}) across both runs")


def c7b_manifest_no_fabrication(obs: Observation) -> Result:
    """A degenerate run (no model, no tools) does not fabricate a manifest change."""
    item, title = "C7b", "manifest_no_fabrication"
    phase = obs.phases["degenerate"]
    minted = _minted_for(obs, ("degenerate",), obs.ctx.agent_name)
    if minted:
        return _fail(
            item, title,
            f"a run with nothing to declare registered {len(minted)} new manifest "
            f"version(s) ({minted}) — the diff will read the absent model/tools as "
            f"deletions the user never made",
        )
    return _pass(item, title, f"{len(phase.attempted)} degenerate trace(s), no new manifest")


def c8_skills_rail(obs: Observation) -> Result:
    """Routing id present, offered names recorded, and the prompt actually agrees.

    Run across several lanes, because the rail's characteristic defect is a
    shared one: a router singleton whose per-call rails (routing_id, offered
    names) leak from one run into the next, so a trace claims a routing decision
    that belongs to somebody else's turn.
    """
    item, title = "C8", "skills_rail"
    phase = obs.phase("skills")
    traces = phase.attempted
    if not traces:
        return _fail(
            item, title,
            "the skills-rail run produced no trace at all — the rail cannot be graded "
            "because nothing reached the wire",
        )
    offered_by_probe = {s["name"] for s in obs.probe.skills}
    problems: List[str] = []
    for t in traces:
        if not t.get("routing_id"):
            problems.append(
                "routing_id absent — the offered→activated join cannot close, so this "
                "run contributes nothing to skill effectiveness"
            )
        recorded = set(t.get("skills_offered_in_prompt") or [])
        if recorded != offered_by_probe:
            problems.append(
                f"skills_offered_in_prompt={sorted(recorded)} but the router offered "
                f"{sorted(offered_by_probe)}"
            )
        prompt = _rendered_text(t)
        missing_in_prompt = sorted(n for n in recorded if n not in prompt)
        if missing_in_prompt:
            problems.append(
                f"claims to have offered {missing_in_prompt} but those names are not in "
                f"the prompt the model was shown"
            )

    # One routing decision per run: a shared routing_id double-counts one
    # decision across several runs and corrupts the effectiveness join.
    routing_ids = [t.get("routing_id") for t in traces if t.get("routing_id")]
    if len(set(routing_ids)) != len(routing_ids):
        problems.append(
            f"{len(routing_ids)} runs share {len(set(routing_ids))} routing_id(s) — a "
            f"routing decision leaked between runs: {routing_ids}"
        )
    if len(traces) != len(phase.ctxs):
        problems.append(
            f"{len(phase.ctxs)} rail runs produced {len(traces)} trace(s)"
        )
    # Each lane's trace must carry its OWN prompt and nobody else's, and the
    # routing_id it reports must be the one the router minted FOR THAT prompt.
    # The provenance check is what turns "two lanes happened to collide" into a
    # deterministic statement about whose routing decision this trace stole.
    for t in traces:
        text = _rendered_text(t)
        mine = [c for c in phase.ctxs if c.prompt_sentinel in text]
        if len(mine) != 1:
            problems.append(
                f"a rail trace carries {len(mine)} lane prompts "
                f"({[c.prompt_sentinel for c in mine]}) — expected exactly its own"
            )
            continue
        query = obs.probe.routing_queries.get(t.get("routing_id"))
        if query is not None and mine[0].prompt_sentinel not in query:
            problems.append(
                f"the run asking about {mine[0].prompt_sentinel!r} reports routing_id "
                f"{t.get('routing_id')}, which the router minted for a DIFFERENT run's "
                f"query ({query[:80]!r}) — the routing rail crossed runs"
            )

    # Any skill whose BODY the probe actually served must be recorded as delivered.
    served = {
        r.path.rsplit("/", 2)[-2]
        for r in phase.requests
        if r.method == "GET" and r.path.endswith("/body") and r.accepted
    }
    if served:
        delivered = set()
        for t in traces:
            delivered |= set(t.get("skills_delivered") or [])
            delivered |= set(t.get("skills_loaded_by_agent") or [])
        missing = sorted(served - delivered)
        if missing:
            problems.append(
                f"the router served the body of {missing} but no trace records them as "
                f"delivered/loaded"
            )
    if problems:
        return _fail(item, title, _summarize(problems))
    return _pass(
        item, title,
        f"{len(traces)} rail run(s): distinct routing_ids, "
        f"{sorted(offered_by_probe)} offered and in-prompt",
    )


def c9_isolation(obs: Observation) -> Result:
    """N concurrent runs produce N traces with no cross-contamination."""
    item, title = "C9", "isolation"
    phase = obs.phases["concurrent"]
    lanes = {c.agent_name: c for c in phase.ctxs}
    traces = phase.attempted
    problems: List[str] = []

    if len(traces) != len(lanes):
        problems.append(
            f"{len(lanes)} concurrent runs produced {len(traces)} trace(s) "
            f"(agent_names: {sorted(t.get('agent_name') for t in traces)})"
        )
    names = [t.get("agent_name") for t in traces]
    if sorted(n for n in names if n) != sorted(lanes):
        problems.append(f"agent_names on the wire {sorted(names)} != lanes {sorted(lanes)}")

    seen_spans: Dict[str, str] = {}
    for t in traces:
        name = t.get("agent_name")
        lane = lanes.get(name)
        if lane is not None:
            text = _rendered_text(t) + "\n" + _output_text(t)
            foreign = [
                c.prompt_sentinel for c in phase.ctxs
                if c.agent_name != name and c.prompt_sentinel in text
            ]
            if foreign:
                problems.append(f"{name}'s trace carries another lane's prompt: {foreign}")
            if lane.prompt_sentinel not in text:
                problems.append(f"{name}'s trace does not carry its OWN prompt")
        for span in t.get("spans") or []:
            sid = span.get("id")
            if sid in seen_spans and seen_spans[sid] != name:
                problems.append(f"span {sid} appears in both {seen_spans[sid]} and {name}")
            seen_spans[sid] = name

    routing_ids = [t.get("routing_id") for t in traces if t.get("routing_id")]
    if len(set(routing_ids)) != len(routing_ids):
        problems.append(f"routing_ids are shared across lanes: {routing_ids}")

    if problems:
        return _fail(item, title, _summarize(problems))
    return _pass(item, title, f"{len(traces)} lanes, no contamination")


def c10_error_path(obs: Observation) -> Result:
    """A failing run produces exactly ONE trace, marked errored."""
    item, title = "C10", "error_path"
    phase = obs.phases["error"]
    traces = phase.attempted
    if len(traces) != 1:
        return _fail(
            item, title,
            f"a failing run produced {len(traces)} trace(s); exactly 1 is the contract "
            f"(0 = the failure is invisible, >1 = the same failure is double-counted)",
        )
    status = traces[0].get("status")
    if status != "error":
        return _fail(
            item, title,
            f"the failing run's trace is status={status!r} — a failed run recorded as a "
            f"success is worse than no trace",
        )
    return _pass(item, title, "one trace, status=error")


def c11_no_side_effects(obs: Observation) -> Result:
    """The run writes nothing into the working directory the user did not ask for."""
    item, title = "C11", "no_side_effects"
    dirty = {p.name: p.new_paths for p in _ran_phases(obs) if p.new_paths}
    if dirty:
        detail = "; ".join(f"[{k}] {sorted(v)[:8]}" for k, v in dirty.items())
        return _fail(
            item, title,
            f"the run left files in the working directory: {detail}",
        )
    return _pass(item, title, "cwd unchanged in every phase")


def c12_loud_failure(obs: Observation) -> Result:
    """When the adapter cannot do what was asked, it says so.

    Two generic cases, both derived from what the wire shows rather than from
    any framework's internals: a phase that emitted nothing, and a trace the
    backend refused. Either is allowed to happen — silently is not.
    """
    item, title = "C12", "loud_failure"
    problems: List[str] = []
    for phase in _ran_phases(obs):
        loud = bool(phase.logs or phase.warnings or phase.exception)
        if not phase.attempted and not loud:
            problems.append(
                f"[{phase.name}] emitted no trace and raised/logged nothing — a silent no-op"
            )
        if phase.rejected and not (phase.export_failed_delta or phase.logs):
            problems.append(
                f"[{phase.name}] {len(phase.rejected)} trace(s) were rejected by the "
                f"backend and neither export_status() nor a warning reported it"
            )
    if problems:
        return _fail(item, title, "; ".join(problems))
    return _pass(item, title, "no silent no-ops")


# ── activation (C13, C13b) ───────────────────────────────────────────────────
#
# The three rungs of the skills ladder are OFFERED (the menu row was in the
# prompt), DELIVERED (the body reached the model) and ACTIVATED (the model
# actually asked for it). C8 grades the first two. These two grade the third,
# from opposite sides: C13 forbids recording an activation the model never
# asked for, C13b forbids dropping one it did.
#
# Asymmetry is deliberate. A fabricated activation is strictly worse than a
# missing one: it is indistinguishable downstream from a real one, it becomes a
# TraceSkillActivation row, it feeds router_activated_count and the activation
# rate the product reports, and it is blended back into ranking — so a skill
# that was merely pasted into a prompt gets PROMOTED over one that was used.
# An omission only under-reports. When only one of the two can hold, C13 wins.
#
# The ceiling is stated rather than hidden: the strongest thing any of this
# proves is that the MODEL ASKED FOR THE BODY. A model can pull a body and
# ignore it. Neither item claims the skill changed the output.

#: Roles whose text is never evidence of activation. A body in a system or
#: developer message is the router DELIVERING it; a name in a user message is
#: the user typing it. Counting either promotes a rung the model never climbed.
_NOT_EVIDENCE_ROLES = ("system", "developer", "user", "human")

#: Roles a tool RESULT arrives under. This message exists only because
#: something asked for it, which is why it counts where a system message does not.
_TOOL_RESULT_ROLES = ("tool", "function")


def _activated_names(trace: Dict[str, Any]) -> set:
    """The activation set the platform will actually persist for this trace.

    Both wire fields, unioned and deduped by name — which is precisely what the
    backend does before writing ``TraceSkillActivation`` rows. Grading
    ``active_skills`` alone would grade a field; this grades the number.
    """
    names = {
        (e.get("name") if isinstance(e, dict) else e)
        for e in (trace.get("active_skills") or [])
    }
    names |= set(trace.get("skills_loaded_by_agent") or [])
    return {n for n in names if isinstance(n, str) and n.strip()}


def _body_signature(body: str) -> str:
    """A short string that appears in this skill's body and in no menu row.

    The longest line, capped. Menu rows carry the NAME and the description; only
    a delivered or loaded body carries the prose. Using the name alone would
    make a menu row look like a body, which is the exact confusion these items
    exist to prevent.
    """
    lines = [ln.strip() for ln in (body or "").splitlines() if ln.strip()]
    return max(lines, key=len)[:40] if lines else ""


def _activation_evidence(trace: Dict[str, Any], name: str, sig: str) -> List[str]:
    """Every model-initiated channel in THIS trace that asked for ``name``.

    Per-trace on purpose. C8's delivered check is phase-level, so a rail that
    leaked a loaded name between two concurrent runs still satisfies it; here
    the corroboration has to live in the same payload as the claim.

    Four accepted forms, strongest last-resort first:

    * a tool/function-role message carrying the body signature — the tool RESULT
      coming back. The only form present on every tool loop, which is why it is
      accepted at all;
    * a tool span that names the skill on the way IN and returns the body on the
      way OUT;
    * a ``tool_calls`` entry whose arguments name the skill — the model's own output;
    * an assistant ``function_call`` message naming it — likewise.

    NOT evidence, ever: the name or the body appearing in a system, developer or
    user message. That is the router delivering, or the user typing. Several
    channels are accepted rather than one because they differ per framework
    (openai-agents has the span and the function_call but no ``tool_calls``;
    pydantic-ai has ``tool_calls`` but no tool span), and because an item that
    hardcodes one channel breaks the day an adapter legitimately changes it.
    """
    found: List[str] = []
    if sig and any(
        role in _TOOL_RESULT_ROLES and sig in text
        for role, text in _role_messages(trace)
    ):
        found.append("tool-role body")
    for span in _tool_spans(trace):
        if not sig:
            break
        if name in _flat(span.get("input_preview")) and sig in _flat(
            span.get("output_preview")
        ):
            found.append(f"{span.get('name') or 'tool'} span input+output")
            break
    for entry in _tool_call_entries(trace):
        args = entry.get("args") if "args" in entry else entry.get("arguments")
        if name in _flat(args):
            found.append("model tool_call args")
            break
    for msg in _rendered_messages(trace):
        if not isinstance(msg, dict):
            continue
        if str(msg.get("role", "")).lower() != "assistant":
            continue
        if str(msg.get("type", "")).lower() != "function_call":
            continue
        if name in _flat(msg):
            found.append("assistant function_call")
            break
    return found


def _served_bodies(phase: Phase) -> set:
    """Skills whose body the probe's router actually handed over on this rail."""
    return {
        r.path.rsplit("/", 2)[-2]
        for r in phase.requests
        if r.method == "GET" and r.path.endswith("/body") and r.accepted
    }


def c13_skills_activation(obs: Observation) -> Result:
    """Nothing is recorded as activated that the model did not itself ask for."""
    item, title = "C13", "skills_activation"
    phase = obs.phase("skills")
    traces = phase.attempted
    if not traces:
        return _fail(
            item, title,
            "the skills-rail run produced no trace at all — activation cannot be "
            "graded because nothing reached the wire",
        )
    bodies = {s["name"]: s.get("body", "") for s in obs.probe.skills}
    offered_by_probe = set(bodies)
    sigs = {n: _body_signature(b) for n, b in bodies.items()}
    served = _served_bodies(phase)
    never_served = offered_by_probe - served

    problems: List[str] = []
    activated_total: set = set()
    per_trace: List[int] = []
    forms: List[str] = []

    for t in traces:
        activated = _activated_names(t)
        per_trace.append(len(activated))
        activated_total |= activated
        recorded_offered = set(t.get("skills_offered_in_prompt") or [])
        roles = _role_messages(t)

        for name in sorted(activated):
            # ── clause 1: offered first ──────────────────────
            if name not in recorded_offered:
                problems.append(
                    f"{name} is recorded as activated but is not in this trace's "
                    f"skills_offered_in_prompt ({sorted(recorded_offered)}) — an "
                    f"activation the router never offered can never join a "
                    f"routing_decision, so it inflates the activated numerator "
                    f"against an offered denominator that does not contain it"
                )
            if name not in offered_by_probe:
                problems.append(
                    f"{name} is recorded as activated but the router never offered it "
                    f"at all — it offered {sorted(offered_by_probe)}"
                )

            # ── clause 2: model-initiated evidence, in THIS trace ──
            sig = sigs.get(name, "")
            found = _activation_evidence(t, name, sig)
            if found:
                forms.extend(found)
                continue
            where = sorted(
                {
                    role for role, text in roles
                    if name in text or (sig and sig in text)
                }
            )
            if where:
                seen = (
                    f"the only place it appears in this trace is a "
                    f"{'/'.join(where)}-role message: that is "
                    f"{'DELIVERED' if any(r in _NOT_EVIDENCE_ROLES for r in where) else 'PRESENT'}"
                    f", not activated"
                )
            else:
                seen = "it appears nowhere in this trace's messages at all"
            problems.append(
                f"{name} is recorded as activated but nothing in this trace shows the "
                f"MODEL asking for it — no tool-role message carrying its body, no tool "
                f"span naming it, no tool_call argument, no assistant function_call; "
                f"{seen}"
            )

        # ── clause 4: hash fidelity ──────────────────────────
        for entry in t.get("active_skills") or []:
            if not isinstance(entry, dict):
                continue
            name, digest = entry.get("name"), entry.get("hash")
            if not name or not digest or name not in served:
                continue
            expected = hashlib.sha256(
                (bodies.get(name) or "").encode("utf-8")
            ).hexdigest()
            got = str(digest)
            # The prefix is an encoding detail the SDK normalises in both
            # directions; the DIGEST is the thing that resolves a version.
            got_bare = got[len("sha256:"):] if got.startswith("sha256:") else got
            if got_bare != expected:
                problems.append(
                    f"{name} is recorded as activated with hash {got!r}, but the body "
                    f"the router served hashes to sha256:{expected} — a wrong hash "
                    f"resolves the activation to a skill VERSION that never ran, "
                    f"silently misattributing per-version effectiveness"
                )

    # ── clause 3: never-served control ───────────────────────
    # conformance-skill-beta is offered on every lane and its body is served on
    # none, so any offered→activated promotion shows up here with no fixture change.
    fabricated = sorted(activated_total & never_served)
    if fabricated:
        problems.append(
            f"{fabricated} recorded as activated although the router never served a "
            f"body for them on this rail (offered {sorted(offered_by_probe)}, served "
            f"{sorted(served)}) — an offered menu row promoted to an activation"
        )

    if problems:
        return _fail(item, title, _summarize(problems))
    if not activated_total:
        return _pass(
            item, title,
            f"{len(traces)} rail run(s) recorded NO activation; correct for a "
            f"prompt-injection rail, where the strongest observable rung is delivered",
        )
    counts = set(per_trace)
    shape = (
        f"{counts.pop()} activation each" if len(counts) == 1
        else f"{sum(per_trace)} activations across the phase"
    )
    return _pass(
        item, title,
        f"{len(traces)} run(s), {shape} ({sorted(activated_total)}), corroborated by "
        f"{' + '.join(sorted(set(forms)))}",
    )


def c13b_skills_activation_recorded(obs: Observation) -> Result:
    """The mirror of C13: a body the model DID pull is not silently dropped.

    C13 stops a rung being invented. This stops one being lost. A model-initiated
    body pull that never reaches ``active_skills`` / ``skills_loaded_by_agent``
    biases effectiveness the other way — a skill that was genuinely used scores
    as though it were never touched.
    """
    item, title = "C13b", "skills_activation_recorded"
    phase = obs.phase("skills")
    traces = phase.attempted
    if not traces:
        return _fail(
            item, title,
            "the skills-rail run produced no trace at all — a dropped activation "
            "cannot be graded because nothing reached the wire",
        )
    bodies = {s["name"]: s.get("body", "") for s in obs.probe.skills}
    sigs = {n: _body_signature(b) for n, b in bodies.items()}

    problems: List[str] = []
    pulled_total: set = set()
    for t in traces:
        activated = _activated_names(t)
        pulled = {
            name
            for role, text in _role_messages(t)
            if role in _TOOL_RESULT_ROLES
            for name, sig in sigs.items()
            if sig and sig in text
        }
        pulled_total |= pulled
        missing = sorted(pulled - activated)
        if missing:
            problems.append(
                f"the model pulled the body of {missing} into this run — it comes back "
                f"in a tool-role message — but the trace records "
                f"{sorted(activated) or 'nothing'} as activated; a real activation "
                f"dropped on the floor under-reports the skill that was actually used"
            )
    if problems:
        return _fail(item, title, _summarize(problems))
    if not pulled_total:
        return _pass(
            item, title,
            f"{len(traces)} rail run(s), no model-initiated body pull to record",
        )
    return _pass(
        item, title,
        f"{len(traces)} rail run(s): every body the model pulled "
        f"({sorted(pulled_total)}) is recorded as activated",
    )


# ── registry ─────────────────────────────────────────────────────────────────

ITEMS: Dict[str, Callable[[Observation], Result]] = {
    "C1": c1_emits,
    "C2": c2_ingest_valid,
    "C3": c3_llm_calls,
    "C4": c4_content,
    "C5": c5_structure,
    "C6": c6_identity,
    "C7": c7_manifest_stable,
    "C7b": c7b_manifest_no_fabrication,
    "C8": c8_skills_rail,
    "C9": c9_isolation,
    "C10": c10_error_path,
    "C11": c11_no_side_effects,
    "C12": c12_loud_failure,
    "C13": c13_skills_activation,
    "C13b": c13b_skills_activation_recorded,
}

#: Display order — dict order is insertion order, but be explicit.
ITEM_ORDER: List[str] = list(ITEMS)


def grade(item: str, obs: Observation) -> Result:
    """Grade one contract item, honouring the driver's N/A declarations."""
    reason = obs.driver.capabilities.na_reason(item)
    if reason:
        return Result(item, ITEMS[item].__name__.split("_", 1)[1], NA, reason)
    return ITEMS[item](obs)
