"""Live-LLM (heavy) — deep + large trace stress.

The rest of the live suite proves the *shapes* of agent traces (tool loops,
parallel fan-out, errors, multi-agent, drift), but every individual trace is
small by design: deterministic answers, 2-5 tools, single-digit LLM calls.
These two cells deliberately go big, to stress the parse + store + render path
at scale:

  * DEEP — a long chain of *dependent* tool calls (model->tool->model->tool for
    ~14 hops). Each hop's id is an opaque token only knowable from the previous
    hop's result, so the model cannot guess ahead or batch: the trace is a tall
    sequence of ~14 LLM turns + ~14 tool calls. Stresses long-trace ingestion +
    ordering + the waterfall rendering many rows.

  * LARGE/WIDE — a fan-out to ~12 tool calls, each returning a multi-KB blob. A
    wide trace carrying heavy payloads. Stresses wide-tree ingestion and big-
    payload handling — the backend stores a *preview* of each payload (see the
    span ``output_preview`` field), so this also pins that truncation path.

Both stay deterministic (a known integer total) so a wrong answer is a real bug,
not model variance. Both run on Gemini only: trace scale is provider-independent,
and a deep run is many model calls (kept moderate to ride out the rate limit;
``gemini_tool_loop`` retries transient 429/503 with backoff).

Marker ``heavy``: runs in the release gate BY DEFAULT (``--skip-heavy`` opts out);
excluded from the routine live sweep (``-m 'live_llm and not heavy'``).
Alone: ``-m heavy``.

Marker: live_llm + heavy
"""

from __future__ import annotations

import hashlib

import pytest

from . import _live_helpers as h


# ── DEEP: a dependent chain the agent must traverse hop-by-hop ───────────────
# Opaque ids (a hash, not node-00/01/...) so the only way to know hop N+1 is to
# have read hop N's result — this forces genuinely SEQUENTIAL tool calls, no
# guessing the chain or batching it.
CHAIN_LEN = 14


def _nid(i: int) -> str:
    return "n_" + hashlib.sha1(f"decimalai-chain-{i}".encode()).hexdigest()[:10]


CHAIN = {
    _nid(i): {"value": (i + 1) * 3, "next": _nid(i + 1) if i < CHAIN_LEN - 1 else None}
    for i in range(CHAIN_LEN)
}
CHAIN_START = _nid(0)
CHAIN_TOTAL = sum(n["value"] for n in CHAIN.values())  # 3*(1+...+14) = 315


def _next_hop(node_id: str) -> dict:
    """One hop: the node's integer ``value`` + the ``next`` node id (null at end)."""
    if node_id not in CHAIN:
        raise ValueError(f"unknown node {node_id!r}")
    return CHAIN[node_id]


_NEXT_HOP_DECL = {
    "name": "next_hop",
    "description": "Look up a chain node. Returns {'value': int, 'next': string|null}.",
    "parameters": {"type": "OBJECT",
                   "properties": {"node_id": {"type": "STRING"}},
                   "required": ["node_id"]},
}

DEEP_QUERY = (
    f"Start at node '{CHAIN_START}'. Call next_hop on it to read its 'value' and "
    f"its 'next' node id. Then call next_hop on that 'next' id, and keep going — "
    f"following each 'next' — adding up every 'value' you see along the way. Stop "
    f"when 'next' is null. Then reply with ONLY the final total sum as a number."
)


# ── LARGE/WIDE: many independent records, each a multi-KB blob with a value ──
RECORD_COUNT = 12
_BLOB = "lorem ipsum dolor sit amet consectetur adipiscing elit " * 60  # ~3 KB
RECORDS = {
    f"rec-{i:02d}": {
        "id": f"rec-{i:02d}",
        "value": (i + 1) * 7,
        # The value marker sits at the FRONT of the blob so it survives the
        # backend's preview truncation and the model can read it cheaply.
        "blob": f"RECORD rec-{i:02d} VALUE={(i + 1) * 7}\n{_BLOB}",
    }
    for i in range(RECORD_COUNT)
}
RECORD_IDS = sorted(RECORDS)
RECORDS_TOTAL = sum(r["value"] for r in RECORDS.values())  # 7*(1+...+12) = 546


def _fetch_record(record_id: str) -> dict:
    """Return a full record — a multi-KB ``blob`` plus its integer ``value``."""
    if record_id not in RECORDS:
        raise ValueError(f"unknown record {record_id!r}")
    return RECORDS[record_id]


_FETCH_DECL = {
    "name": "fetch_record",
    "description": "Fetch a record by id. Returns its large text 'blob' and an integer 'value'.",
    "parameters": {"type": "OBJECT",
                   "properties": {"record_id": {"type": "STRING"}},
                   "required": ["record_id"]},
}

LARGE_QUERY = (
    "Fetch every one of these records and add up their 'value' fields: "
    + ", ".join(RECORD_IDS)
    + ". Call fetch_record for each id (you may request several at once). "
    "Reply with ONLY the total of all the values as a number."
)


@pytest.mark.live_llm
@pytest.mark.heavy
@pytest.mark.parametrize("provider, model", h.matrix("generic", only=("google",)))
def test_deep_sequential_chain(provider, model):
    """~14 dependent hops -> a tall trace of many sequential LLM + tool calls."""
    h.require_key_for(provider)
    pytest.importorskip("google.genai")
    import decimalai

    agent_name = h.unique_agent(f"deep-{provider}-chain")

    @decimalai.trace(agent_name=agent_name)
    def run() -> str:
        return h.gemini_tool_loop(
            model, DEEP_QUERY,
            tool_declarations=[_NEXT_HOP_DECL],
            handlers={"next_hop": _next_hop},
            log_llm=decimalai.log_llm_call,
            log_tool=decimalai.log_tool_call,
            max_iters=CHAIN_LEN + 6,
        )

    answer = run()
    assert str(CHAIN_TOTAL) in answer.replace(",", ""), (
        f"Deep chain didn't sum to {CHAIN_TOTAL}: {answer!r}"
    )

    h.flush_sdk_sender()
    detail = h.get_trace_detail(h.poll_for_trace(agent_name)[0]["id"])
    # Depth: the opaque chain forces a long sequence — roughly one tool call per
    # hop plus a final answer turn. Thresholds sit below the ideal (~15 LLM /
    # 14 tool) to tolerate the odd double-step, while still proving a deep trace.
    h.assert_rich_agent_trace(detail, min_llm_calls=10, min_tool_calls=12)


@pytest.mark.live_llm
@pytest.mark.heavy
@pytest.mark.parametrize("provider, model", h.matrix("generic", only=("google",)))
def test_wide_large_payload(provider, model):
    """~12 tool calls each returning a multi-KB blob -> a wide, heavy-payload trace."""
    h.require_key_for(provider)
    pytest.importorskip("google.genai")
    import decimalai

    agent_name = h.unique_agent(f"large-{provider}-fanout")

    @decimalai.trace(agent_name=agent_name)
    def run() -> str:
        return h.gemini_tool_loop(
            model, LARGE_QUERY,
            tool_declarations=[_FETCH_DECL],
            handlers={"fetch_record": _fetch_record},
            log_llm=decimalai.log_llm_call,
            log_tool=decimalai.log_tool_call,
            max_iters=RECORD_COUNT + 6,
        )

    answer = run()
    assert str(RECORDS_TOTAL) in answer.replace(",", ""), (
        f"Wide fan-out didn't sum to {RECORDS_TOTAL}: {answer!r}"
    )

    h.flush_sdk_sender()
    detail = h.get_trace_detail(h.poll_for_trace(agent_name)[0]["id"])
    # Wide: roughly one tool call per record. (min below 12 to tolerate a merge.)
    h.assert_rich_agent_trace(detail, min_llm_calls=1, min_tool_calls=10)

    # Heavy payloads: each fetch returned a multi-KB blob. The backend stores a
    # *preview* of each tool output, so assert the outputs were captured (not
    # dropped) — ≥10 tool spans carry a non-empty output_preview.
    tool_spans = [s for s in detail.get("spans", []) if s.get("span_type") == "tool"]
    with_output = [s for s in tool_spans if (s.get("output_preview") or "").strip()]
    assert len(with_output) >= 10, (
        f"Expected ≥10 tool spans with a captured output_preview, got "
        f"{len(with_output)} of {len(tool_spans)}. Trace id={detail['id']}"
    )
