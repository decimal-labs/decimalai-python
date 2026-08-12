"""Live-LLM (heavy) — 3-level nested agent tree + a combined "kitchen-sink" trace.

The flat suite proves each trace shape in isolation (deep chains, wide fan-out,
tool errors, 2-level handoff). These two cells cover what production agents
actually look like — combinations:

  * NESTED — a 3-level tree (orchestrator → inventory analyst → price auditor).
    The 2-level multi-agent cells prove parent/child linkage once; this proves
    the MIDDLE node works as both a subagent and a parent at the same time —
    manifest subagent declarations, the parent_trace_id chain, and the
    topology / SubAgent-Health surfaces at depth 3.

  * KITCHEN-SINK — one trace mixing every shape at once: a dependent sequential
    chain, a fan-out over multi-KB payloads, and a tool error the model must
    recover from. Stresses ingestion + rendering the way a messy real agent
    does, in a single run.

Both stay deterministic (known integer totals) so a wrong answer is a real bug,
not model variance. Gemini-only: trace shape is provider-independent (same
rationale as the scale cells). To keep the sub-tasks deterministic, the
``consult_*`` handlers run the CANONICAL sub-question rather than the model's
paraphrase — the model still drives every delegation and relays every number.

Marker ``heavy``: runs in the release gate BY DEFAULT (``--skip-heavy`` opts
out); excluded from the routine live sweep (``-m 'live_llm and not heavy'``).

Marker: live_llm + heavy (+ multi_agent on the nested cell)
"""

from __future__ import annotations

import hashlib

import pytest

from . import _live_helpers as h


# ═══════════════════════════════════════════════════════════════════
# NESTED: deterministic 3-level fixtures
# ═══════════════════════════════════════════════════════════════════
STOCK = {"east": 20, "west": 30}                 # level-2 tool data
UNIT_PRICE = {"sku-a": 5, "sku-b": 7}            # level-3 tool data
PRICE_SUM = sum(UNIT_PRICE.values())             # 12
GRAND_TOTAL = sum(STOCK.values()) + PRICE_SUM    # 62

AUDITOR_QUERY = (
    "Call lookup_unit_price for 'sku-a' and for 'sku-b', then reply with ONLY "
    "the sum of the two prices as a number."
)
ANALYST_QUERY = (
    "Two steps: (1) call count_stock for 'east' and for 'west' and add the two "
    "counts. (2) call consult_price_auditor once to get the unit-price sum. "
    "Then reply with ONLY stock total + price sum as a single number."
)
ORCH_QUERY = (
    "Call consult_inventory_analyst exactly once to get the inventory grand "
    "total, then reply with ONLY the number it returns."
)


@pytest.mark.live_llm
@pytest.mark.heavy
@pytest.mark.multi_agent
@pytest.mark.parametrize("provider, model", h.matrix("generic", only=("google",)))
def test_three_level_subagent_tree(provider, model):
    """orchestrator → analyst → auditor: 3 linked traces, topology at depth 3.

    The analyst is simultaneously a subagent (of the orchestrator) and a parent
    (of the auditor) — the case no 2-level cell exercises.
    """
    h.require_key_for(provider)
    pytest.importorskip("google.genai")
    import decimalai

    orch_name = h.unique_agent(f"nested-{provider}-orchestrator")
    analyst_name = h.unique_agent(f"nested-{provider}-analyst")
    auditor_name = h.unique_agent(f"nested-{provider}-auditor")

    # ── level 3: price auditor (leaf agent) ───────────────────────────
    def run_auditor(parent_id: str) -> str:
        with decimalai.start_trace(agent_name=auditor_name,
                                   parent_trace_id=parent_id) as ctx:
            ctx.set_input(AUDITOR_QUERY)
            answer = h.gemini_tool_loop(
                model, AUDITOR_QUERY,
                tool_declarations=[{
                    "name": "lookup_unit_price",
                    "description": "Unit price of a SKU. Returns {'price': int}.",
                    "parameters": {"type": "OBJECT",
                                   "properties": {"sku": {"type": "STRING"}},
                                   "required": ["sku"]},
                }],
                handlers={"lookup_unit_price": lambda sku: {"price": UNIT_PRICE[sku]}},
                log_llm=ctx.log_llm_call,
                log_tool=ctx.log_tool_call,
            )
            ctx.set_output(answer)
            return answer

    # ── level 2: inventory analyst (subagent AND parent) ──────────────
    def run_analyst(parent_id: str) -> str:
        with decimalai.start_trace(agent_name=analyst_name,
                                   parent_trace_id=parent_id,
                                   subagents=[{"name": auditor_name}]) as ctx:
            ctx.set_input(ANALYST_QUERY)
            answer = h.gemini_tool_loop(
                model, ANALYST_QUERY,
                tool_declarations=[
                    {"name": "count_stock",
                     "description": "Units in stock for a region ('east' or 'west'). "
                                    "Returns {'count': int}.",
                     "parameters": {"type": "OBJECT",
                                    "properties": {"region": {"type": "STRING"}},
                                    "required": ["region"]}},
                    {"name": "consult_price_auditor",
                     "description": "Delegate the unit-price question to the "
                                    "price-auditor agent; returns its numeric answer.",
                     "parameters": {"type": "OBJECT",
                                    "properties": {"question": {"type": "STRING"}},
                                    "required": ["question"]}},
                ],
                handlers={
                    "count_stock": lambda region: {"count": STOCK[region]},
                    # Canonical sub-question for determinism (see module docstring).
                    "consult_price_auditor": lambda question: {
                        "answer": run_auditor(ctx.get_trace_id())},
                },
                log_llm=ctx.log_llm_call,
                log_tool=ctx.log_tool_call,
                max_iters=10,
            )
            ctx.set_output(answer)
            return answer

    # ── level 1: orchestrator ──────────────────────────────────────────
    with decimalai.start_trace(agent_name=orch_name,
                               subagents=[{"name": analyst_name}]) as orch_ctx:
        orch_ctx.set_input(ORCH_QUERY)
        final = h.gemini_tool_loop(
            model, ORCH_QUERY,
            tool_declarations=[{
                "name": "consult_inventory_analyst",
                "description": "Delegate the inventory question to the "
                               "inventory-analyst agent; returns its numeric answer.",
                "parameters": {"type": "OBJECT",
                               "properties": {"question": {"type": "STRING"}},
                               "required": ["question"]},
            }],
            handlers={"consult_inventory_analyst": lambda question: {
                "answer": run_analyst(orch_ctx.get_trace_id())}},
            log_llm=orch_ctx.log_llm_call,
            log_tool=orch_ctx.log_tool_call,
        )
        orch_ctx.set_output(final)

    assert str(GRAND_TOTAL) in final.replace(",", ""), (
        f"3-level tree didn't produce {GRAND_TOTAL}: {final!r}"
    )

    h.flush_sdk_sender()
    orch = h.get_trace_detail(h.poll_for_trace(orch_name)[0]["id"])
    analyst = h.get_trace_detail(h.poll_for_trace(analyst_name)[0]["id"])
    auditor = h.get_trace_detail(h.poll_for_trace(auditor_name)[0]["id"])

    # The parent/child chain holds at depth 3.
    assert analyst.get("parent_trace_id") == orch["id"], (
        f"analyst.parent_trace_id={analyst.get('parent_trace_id')!r}, "
        f"expected orchestrator id {orch['id']!r}"
    )
    assert auditor.get("parent_trace_id") == analyst["id"], (
        f"auditor.parent_trace_id={auditor.get('parent_trace_id')!r}, "
        f"expected analyst id {analyst['id']!r}"
    )

    # Every level is a real agent run.
    h.assert_rich_agent_trace(orch, min_llm_calls=1, min_tool_calls=1)
    h.assert_rich_agent_trace(analyst, min_llm_calls=2, min_tool_calls=2)
    h.assert_rich_agent_trace(auditor, min_llm_calls=1, min_tool_calls=2)

    # Three distinct manifests — each level registered its own.
    manifest_ids = {orch.get("manifest_id"), analyst.get("manifest_id"),
                    auditor.get("manifest_id")}
    assert None not in manifest_ids and len(manifest_ids) == 3, (
        f"Expected 3 distinct manifests, got {manifest_ids}"
    )

    # Topology + SubAgent-Health surfaces hold at BOTH links — the analyst
    # is a subagent and a parent at the same time.
    h.assert_topology_declared(orch_name, analyst_name)
    h.assert_topology_declared(analyst_name, auditor_name)
    h.assert_subagent_resolved(analyst_name, orch_name)
    h.assert_subagent_resolved(auditor_name, analyst_name)


# ═══════════════════════════════════════════════════════════════════
# KITCHEN-SINK: one trace combining chain + fan-out + payload + error-recovery
# ═══════════════════════════════════════════════════════════════════
KS_CHAIN_LEN = 4


def _ks_nid(i: int) -> str:
    return "k_" + hashlib.sha1(f"decimalai-kitchen-{i}".encode()).hexdigest()[:10]


KS_CHAIN = {
    _ks_nid(i): {"value": (i + 1) * 4, "next": _ks_nid(i + 1) if i < KS_CHAIN_LEN - 1 else None}
    for i in range(KS_CHAIN_LEN)
}
KS_CHAIN_START = _ks_nid(0)
KS_CHAIN_TOTAL = sum(n["value"] for n in KS_CHAIN.values())          # 40

KS_RECORD_COUNT = 4
_KS_BLOB = "synthetic payload text for preview-truncation coverage " * 40  # ~2 KB
KS_RECORDS = {
    f"krec-{i}": {
        "id": f"krec-{i}",
        "value": (i + 1) * 9,
        "blob": f"RECORD krec-{i} VALUE={(i + 1) * 9}\n{_KS_BLOB}",
    }
    for i in range(KS_RECORD_COUNT)
}
KS_RECORDS_TOTAL = sum(r["value"] for r in KS_RECORDS.values())      # 90
KS_BONUS = 25
KS_TOTAL = KS_CHAIN_TOTAL + KS_RECORDS_TOTAL + KS_BONUS              # 155

KS_QUERY = (
    "Complete ALL THREE tasks, then reply with ONLY the grand total as a number.\n"
    f"(1) CHAIN: start at node '{KS_CHAIN_START}'. Call next_hop on it, read its "
    "'value' and 'next', keep calling next_hop on each 'next' until it is null, "
    "summing every 'value'.\n"
    "(2) RECORDS: call fetch_record for each of: "
    + ", ".join(sorted(KS_RECORDS)) + ". Sum their 'value' fields.\n"
    "(3) BONUS: call unlock_bonus with code='start'. If it returns an error, "
    "follow the instructions in the error message and call it again. Add the "
    "'bonus' value.\n"
    "Grand total = chain sum + records sum + bonus."
)


def _ks_next_hop(node_id: str) -> dict:
    if node_id not in KS_CHAIN:
        raise ValueError(f"unknown node {node_id!r}")
    return KS_CHAIN[node_id]


def _ks_fetch_record(record_id: str) -> dict:
    if record_id not in KS_RECORDS:
        raise ValueError(f"unknown record {record_id!r}")
    return KS_RECORDS[record_id]


def _ks_unlock_bonus(code: str = "") -> dict:
    # Deterministic error-recovery: the first (instructed) code fails with a
    # message telling the model the correct one — recovery is in the error text.
    if code != "retry-amber":
        raise ValueError("locked: call unlock_bonus again with code='retry-amber'")
    return {"bonus": KS_BONUS}


_KS_DECLS = [
    {"name": "next_hop",
     "description": "Look up a chain node. Returns {'value': int, 'next': string|null}.",
     "parameters": {"type": "OBJECT",
                    "properties": {"node_id": {"type": "STRING"}},
                    "required": ["node_id"]}},
    {"name": "fetch_record",
     "description": "Fetch a record by id. Returns its large text 'blob' and an integer 'value'.",
     "parameters": {"type": "OBJECT",
                    "properties": {"record_id": {"type": "STRING"}},
                    "required": ["record_id"]}},
    {"name": "unlock_bonus",
     "description": "Unlock the bonus value with an access code. Returns {'bonus': int}.",
     "parameters": {"type": "OBJECT",
                    "properties": {"code": {"type": "STRING"}},
                    "required": ["code"]}},
]


def _has_error_evidence(detail: dict) -> bool:
    """At least one tool span recorded the failed unlock (status or output)."""
    for s in detail.get("spans", []):
        if s.get("span_type") != "tool":
            continue
        if str(s.get("status", "")).lower() in ("error", "failed", "fail"):
            return True
        if s.get("error_message") or s.get("error"):
            return True
        out = s.get("output_preview") or s.get("output") or ""
        if isinstance(out, str) and "locked" in out.lower():
            return True
    return False


@pytest.mark.live_llm
@pytest.mark.heavy
@pytest.mark.parametrize("provider, model", h.matrix("generic", only=("google",)))
def test_kitchen_sink_combined_trace(provider, model):
    """Chain + fan-out + multi-KB payloads + error-recovery in ONE trace."""
    h.require_key_for(provider)
    pytest.importorskip("google.genai")
    import decimalai

    agent_name = h.unique_agent(f"kitchen-{provider}-sink")

    @decimalai.trace(agent_name=agent_name)
    def run() -> str:
        return h.gemini_tool_loop(
            model, KS_QUERY,
            tool_declarations=_KS_DECLS,
            handlers={
                "next_hop": _ks_next_hop,
                "fetch_record": _ks_fetch_record,
                "unlock_bonus": _ks_unlock_bonus,
            },
            log_llm=decimalai.log_llm_call,
            log_tool=decimalai.log_tool_call,
            max_iters=16,
        )

    answer = run()
    assert str(KS_TOTAL) in answer.replace(",", ""), (
        f"Kitchen-sink didn't total {KS_TOTAL}: {answer!r}"
    )

    h.flush_sdk_sender()
    detail = h.get_trace_detail(h.poll_for_trace(agent_name)[0]["id"])

    # A genuinely mixed trace: ~4 chain + ~4 fetch + 2 unlock tool calls across
    # several model turns. Thresholds sit below the ideal to tolerate batching.
    h.assert_rich_agent_trace(detail, min_llm_calls=4, min_tool_calls=8)

    # The failed unlock is *captured* in the trace (the error-recovery surface).
    assert _has_error_evidence(detail), (
        f"No tool span carries the failed unlock_bonus attempt. "
        f"Trace id={detail['id']}"
    )

    # Multi-KB payloads were captured (the output_preview truncation path).
    tool_spans = [s for s in detail.get("spans", []) if s.get("span_type") == "tool"]
    with_output = [s for s in tool_spans if (s.get("output_preview") or "").strip()]
    assert len(with_output) >= 3, (
        f"Expected ≥3 tool spans with a captured output_preview, got "
        f"{len(with_output)} of {len(tool_spans)}. Trace id={detail['id']}"
    )
