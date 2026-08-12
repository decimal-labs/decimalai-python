"""Live-LLM Layer 6 — manifest drift detection.

Run the SAME agent twice with a different system prompt between runs. Asserts:

  * Two distinct manifest_ids land in the backend for that agent_name.
  * Their manifest_hash values differ (auto-detected drift, not luck).
  * POST /api/v1/regression-check against the candidate (v2) returns a
    non-`first_run` verdict — proving the backend saw v1 as a baseline and
    computed an impact report against it.
  * The impact report's diff_summary contains a prompt-surface change.

This is the live end-to-end proof of the manifest-aware versioning claim.

Marker: live_llm + manifest_drift
"""

from __future__ import annotations

import pytest

from . import _live_helpers as h


SYSTEM_PROMPT_V1 = (
    "You are a customer support agent. Use your tools to look up the "
    "customer's order and compute any refund they're entitled to. Reply "
    "concisely with the refund amount."
)

# Materially different surface text — manifest hash MUST change.
SYSTEM_PROMPT_V2 = (
    "You are a customer support agent with strict policy guardrails. "
    "Before issuing any refund, you must verify the customer's tier AND "
    "the order status. Cite the relevant policy in your final answer. "
    "Reply with the refund amount, store credit, and the policy citation."
)


@pytest.mark.live_llm
@pytest.mark.manifest_drift
@pytest.mark.parametrize("provider, model", h.matrix("langchain"))
def test_langchain_manifest_drift_triggers_regression_check(provider, model):
    """Two runs, same agent_name, different system prompts → two manifests +
    a regression-check that recognizes the second as a candidate."""
    h.require_key_for(provider)
    pytest.importorskip("langgraph")
    from langchain_core.tools import tool
    from decimalai.langchain import CallbackHandler

    # Build the provider's chat model. Drift detection hashes the manifest's
    # prompt/tool surface, so it must hold across adapters: Gemini's
    # ChatGoogleGenerativeAI carries the system prompt differently than
    # ChatOpenAI, and this proves the auto-detector sees the change either way.
    if provider == "google":
        pytest.importorskip("langchain_google_genai")
        from langchain_google_genai import ChatGoogleGenerativeAI
        def _make_llm():
            return h.chat_google_genai(model)
    elif provider == "anthropic":
        pytest.importorskip("langchain_anthropic")
        from langchain_anthropic import ChatAnthropic
        def _make_llm():
            return ChatAnthropic(model=model, temperature=0)
    else:
        pytest.importorskip("langchain_openai")
        from langchain_openai import ChatOpenAI
        def _make_llm():
            return ChatOpenAI(model=model)

    @tool
    def get_order_details(order_id: str) -> dict | str:
        """Return order details — status, items, total — by order ID."""
        try:
            return h.get_order_details(order_id)
        except ValueError as e:
            return f"ERROR: {e}"

    @tool
    def calculate_refund(order_total: float, condition: str) -> dict:
        """Compute refund + store credit for an order given its condition."""
        return h.calculate_refund(order_total, condition)

    # Same agent_name across runs — this is what couples the manifests for
    # the regression-check to find a baseline.
    agent_name = h.unique_agent(f"langchain-{provider}-drift")
    tools = [get_order_details, calculate_refund]

    # ── Run 1: register manifest v1 ───────────────────────────────────
    llm_v1 = _make_llm()
    agent_v1 = h.make_react_agent(llm_v1, tools=tools, prompt=SYSTEM_PROMPT_V1)
    handler_v1 = CallbackHandler(agent_name=agent_name)
    agent_v1.invoke(
        {"messages": [{"role": "user", "content": h.SUPPORT_QUERY}]},
        config={"callbacks": [handler_v1], "run_name": "drift-v1"},
    )
    h.flush_sdk_sender()

    # ── Run 2: same agent_name, different prompt — should register v2 ─
    llm_v2 = _make_llm()
    agent_v2 = h.make_react_agent(llm_v2, tools=tools, prompt=SYSTEM_PROMPT_V2)
    handler_v2 = CallbackHandler(agent_name=agent_name)
    agent_v2.invoke(
        {"messages": [{"role": "user", "content": h.SUPPORT_QUERY}]},
        config={"callbacks": [handler_v2], "run_name": "drift-v2"},
    )
    h.flush_sdk_sender()

    # Both traces should land — proves both runs reached the backend.
    h.poll_for_trace(agent_name, expected_count=2)

    # ── Manifests: two distinct, different hashes ─────────────────────
    manifests = h.list_manifests(agent_name)
    assert len(manifests) >= 2, (
        f"Expected ≥ 2 manifests for {agent_name!r}, got {len(manifests)}. "
        f"This means the prompt-driven drift was NOT detected by the adapter."
    )
    hashes = {m.get("manifest_hash") for m in manifests}
    assert len(hashes) >= 2, (
        f"All {len(manifests)} manifests share the same hash {hashes!r} — "
        f"prompt change did not produce a new manifest hash. "
        f"Auto-detection is broken or the prompts are being normalized away."
    )

    # Backend orders newest-first. v2 is the newer manifest (current state);
    # v1 is the older one (the prior production manifest).
    v2 = manifests[0]
    v1 = manifests[1]
    assert v2["id"] != v1["id"]

    # ── Auto-detected drift: v2 is active, v1 is superseded ───────────
    # This is the central moat assertion: registering a new manifest with
    # a changed prompt automatically supersedes the prior one. That state
    # transition IS the drift signal — surfaces the timeline UI, the
    # version-vs-version compare, and the prompt-changed regression alert.
    assert v2.get("status") == "active", (
        f"Expected newer manifest to be 'active', got {v2.get('status')!r}"
    )
    assert v1.get("status") == "superseded", (
        f"Expected older manifest to be 'superseded' after v2 was registered, "
        f"got {v1.get('status')!r}. Drift was not auto-detected on registration."
    )
    assert v2.get("parent_manifest_id") == v1["id"], (
        f"Lineage broken: v2.parent_manifest_id={v2.get('parent_manifest_id')!r}, "
        f"expected {v1['id']!r}. The version timeline UI relies on this."
    )

    # ── Regression-check: pass v1 (superseded) as candidate; backend
    #    resolves v2 (active) as baseline → real diff. This mirrors the
    #    "what if I revert to v1?" preview surface. Verdict ≠ first_run
    #    proves the regression-check pipeline saw both manifests.
    report = h.post_regression_check(agent_name, v1["id"])
    verdict = report.get("verdict")
    assert verdict and verdict != "first_run", (
        f"Regression-check returned verdict={verdict!r}. Expected the "
        f"backend to find v2 (active) as baseline against v1 candidate. "
        f"v1 id: {v1['id']}, v2 id: {v2['id']}"
    )
    assert report.get("baseline_manifest_id") == v2["id"], (
        f"Regression-check resolved baseline as "
        f"{report.get('baseline_manifest_id')!r}, expected {v2['id']!r} (active)"
    )

    diff = report.get("diff_summary") or {}
    total_changes = diff.get("total_changes")
    if total_changes is None:
        total_changes = sum(
            v if isinstance(v, int) else 0
            for k, v in diff.items()
            if k != "first_run"
        )
    assert total_changes and total_changes > 0, (
        f"Regression-check diff_summary reported 0 changes between v2 and v1: "
        f"{diff!r}. Expected the prompt change to surface as a structural diff."
    )

    # Localization: the ONLY edit between v1 and v2 was the system prompt, so the
    # diff must attribute the change to the prompt surface — not merely report
    # "something changed". This guards against a diff that fires on the wrong
    # surface (e.g. flags a tool/model change when only the prompt moved): a
    # total_changes>0 that points at the wrong thing is a silent correctness bug.
    changes = diff.get("changes") or []
    prompt_localized = [
        c for c in changes
        if "prompt" in str(c.get("type", "")).lower()
        or str(c.get("name", "")).lower() == "system"
    ]
    assert prompt_localized, (
        f"Regression-check diff did not localize the change to the prompt/system "
        f"surface. Only the system prompt changed between v1 and v2, so a change "
        f"with type~='prompt' or name=='system' was expected. changes={changes!r}"
    )
