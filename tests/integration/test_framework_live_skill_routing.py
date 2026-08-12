"""Live-LLM — skill routing: real model picks from the SkillRouter menu.

The 22-cell live tier proves the adapter→backend trace wire works, but NONE of
it exercises skill routing end to end: given a router-built menu, does a *real*
model pick the right skill, and does the offered→activated routing join close?
Everything that touches that path elsewhere uses deterministic fixtures or a
stubbed model. This layer closes that gap.

What each cell does, end to end:
  1. Seed two unambiguous skills in a unique category — a refunds skill and a
     flights skill — so the menu is a clean 2-row discriminator.
  2. Call the REAL router: ``SkillRouter.build_prompt_fragment`` (full-menu
     strategy, category-scoped) → ``(prompt_fragment, routing_id)``. This is the
     exact primitive every framework adapter calls.
  3. Feed that fragment as the system prompt to a REAL model (Gemini / GPT /
     Claude) with a refund-intent query, and read back which skill it named.
  4. Assert the model routed correctly — it referenced the refunds skill and NOT
     the flights skill (skill SELECTION by description, not topic echo).
  5. Post a trace carrying the real ``routing_id`` + the model-chosen
     ``active_skills``, then assert the join closed: the refunds skill is
     offered AND activated, the flights skill is offered-not-activated, and the
     per-skill router-stats roll it up.

Deterministic by construction: a unique category means the only routing
decision that offers these skills is this cell's menu call, and the only trace
that activates one is the one we post. The single model-dependent assertion
(step 4) is made reliable by a 2-skill menu + an unambiguous query.

Marker: live_llm + skill_routing
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from uuid import uuid4

import pytest

from . import _live_helpers as h


# ─── HTTP helpers (urllib, matching _live_helpers style) ─────────────

def _api(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        f"{h.BACKEND_URL}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {h.API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def _create_skill(name: str, description: str, body_markdown: str, category: str) -> str:
    """Create an active skill; return its skill_id. Body must clear the
    backend's 50-character minimum body length."""
    resp = _api("POST", "/api/v1/skills", {
        "name": name,
        "description": description,
        "body_markdown": body_markdown,
        "category": category,
    })
    skill_id = resp.get("skill_id") or resp.get("id") or ""
    assert skill_id, f"skill create returned no id: {resp}"
    return skill_id


def _delete_skill(skill_id: str) -> None:
    try:
        _api("DELETE", f"/api/v1/skills/{skill_id}")
    except Exception:
        pass  # best-effort cleanup — never fail the test on teardown


def _seed_manifest(agent_name: str) -> str:
    """Register a minimal manifest so trace ingest passes its required-manifest
    gate (DECIMAL_REQUIRE_MANIFEST_ON_INGEST). Returns the manifest_id."""
    suffix = uuid4().hex[:12]
    resp = _api("POST", "/api/v1/manifests", {
        "agent_name": agent_name,
        "manifest_hash": f"sha_skillroute_{suffix}",
        "version_label": "v1",
        "components": [
            {"component_type": "model", "component_name": "gpt-4o", "content_hash": "m_gpt4o"},
            {"component_type": "prompt", "component_name": "system",
             "content_hash": f"p_sys_{suffix}"},
        ],
    })
    manifest_id = resp.get("manifest_id") or resp.get("id") or ""
    assert manifest_id, f"manifest seed returned no id: {resp}"
    return manifest_id


def _post_trace(agent_name: str, manifest_id: str, routing_id: str,
                active_skills: list[str], user_input: str, final_output: str) -> str:
    trace_id = str(uuid4())
    _api("POST", "/api/v1/traces", {
        "id": trace_id,
        "agent_name": agent_name,
        "manifest_id": manifest_id,
        "status": "success",
        "user_input": user_input,
        "final_output": final_output[:2000],
        "routing_id": routing_id,
        "active_skills": active_skills,
        "llm_calls": [],
        "spans": [],
    })
    return trace_id


def _poll_join(routing_id: str, *, want_activated: str, attempts: int = 6) -> dict:
    """GET /skills/routing/{id}, retrying until the activation commits."""
    last: dict = {}
    for _ in range(attempts):
        last = _api("GET", f"/api/v1/skills/routing/{routing_id}")
        if want_activated in (last.get("activated") or []):
            return last
        time.sleep(1)
    return last


# ─── Real-model generation (no tools — just skill selection) ─────────

def _ask_model(provider: str, model: str, *, system: str, query: str) -> str:
    """Single generation with `system` as the system prompt. Returns text."""
    if provider == "google":
        from google import genai
        from google.genai import types
        # Bind the client to a local — `genai.Client().models...` GC-closes its
        # httpx transport mid-call.
        client = h.gemini_client()
        resp = client.models.generate_content(
            model=model,
            contents=query,
            config=types.GenerateContentConfig(system_instruction=system),
        )
        return resp.text or ""
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
        )
        return resp.choices[0].message.content or ""
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": query}],
        )
        return "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        )
    raise ValueError(f"unknown provider {provider!r}")


_GENERIC_IMPORT = {"google": "google.genai", "openai": "openai", "anthropic": "anthropic"}

# The discriminating query — unambiguously a refund task, never a flight task.
_QUERY = (
    "A customer emailed: their order arrived damaged and they want a full refund "
    "back to their original card. Which skill applies here, and what will you do?"
)


@pytest.mark.live_llm
@pytest.mark.skill_routing
@pytest.mark.parametrize("provider, model", h.matrix("generic"))
def test_real_model_routes_to_correct_skill(provider, model):
    h.require_key_for(provider)
    pytest.importorskip(_GENERIC_IMPORT[provider])

    from decimalai.skill_router import SkillRouter

    stamp = uuid4().hex[:8]
    category = f"liveroute-{stamp}"
    expected = f"liveroute-refunds-{stamp}"     # the right answer
    distractor = f"liveroute-flights-{stamp}"   # the wrong answer
    agent_name = h.unique_agent(f"skillroute-{provider}")

    expected_id = _create_skill(
        expected,
        "Handle product return, refund, and money-back requests from customers.",
        "Refund runbook: verify the order, confirm the item condition, then issue "
        "the money back to the customer's original payment method.",
        category,
    )
    distractor_id = _create_skill(
        distractor,
        "Search for and book airline flights and travel itineraries.",
        "Flight-booking runbook: search routes by date and destination, compare "
        "fares, then confirm the airline booking for the traveler.",
        category,
    )
    try:
        # ── 1. Real router call (full menu, scoped to our 2-skill category) ──
        router = SkillRouter(
            api_key=h.API_KEY, base_url=h.BACKEND_URL,
            agent_name=agent_name, strategy="menu",
        )
        fragment, routing_id = router.build_prompt_fragment(query=_QUERY, category=category)
        assert routing_id and routing_id.startswith("rt_"), f"no routing_id: {routing_id!r}"
        assert expected in fragment, f"refunds skill missing from menu:\n{fragment}"
        assert distractor in fragment, f"flights skill missing from menu:\n{fragment}"

        # ── 2-3. Real model picks from the menu ──
        answer = _ask_model(provider, model, system=fragment, query=_QUERY)
        assert answer.strip(), "model returned empty output"

        # ── 4. Routing correctness: referenced the refunds skill, not flights.
        # Substring on the prefixed stem is robust to whether the model echoed
        # the random stamp, and (unlike asserting on the word "refund") proves
        # it picked the SKILL by name rather than just parroting the topic.
        assert "liveroute-refunds" in answer, (
            f"model did not select the refunds skill.\nQuery: {_QUERY}\nAnswer: {answer}"
        )
        assert "liveroute-flights" not in answer, (
            f"model wrongly selected the flights skill.\nAnswer: {answer}"
        )

        # ── 5. Close + assert the offered→activated join ──
        chosen = [s for s in (expected, distractor) if s in answer]  # model-derived
        assert chosen == [expected]
        manifest_id = _seed_manifest(agent_name)
        _post_trace(agent_name, manifest_id, routing_id, chosen, _QUERY, answer)
        h.flush_sdk_sender()

        join = _poll_join(routing_id, want_activated=expected)
        assert expected in (join.get("offered") or []), f"join offered missing expected: {join}"
        assert expected in (join.get("activated") or []), f"join activated missing expected: {join}"
        assert distractor in (join.get("offered") or []), f"join offered missing distractor: {join}"
        assert distractor in (join.get("offered_not_activated") or []), (
            f"distractor should be offered-not-activated: {join}"
        )

        # ── 6. Per-skill rollup ──
        stats = _api("GET", f"/api/v1/skills/{expected}/router-stats?window_days=30")
        assert stats.get("decisions_count", 0) >= 1, stats
        assert stats.get("activated_count", 0) >= 1, stats
    finally:
        _delete_skill(expected_id)
        _delete_skill(distractor_id)
