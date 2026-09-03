"""SDK-side manifest extraction and change detection.

Extracts an AgentManifest from a LangGraph graph or manual configuration,
hashes it, and detects changes to trigger registration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import time
import warnings
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger("decimalai.manifest")

# `ComponentSnapshot.schema_json` shadows `BaseModel.schema_json` (a pydantic
# method that JSON-serializes the schema), so pydantic v2 emits a
# UserWarning at class-creation time. The shadow
# is harmless under v2 — pydantic gives precedence to the user field —
# and `schema_json` is the established wire-format field name (the
# backend DB column is also `schema_json`, the HTTP API echoes it back).
# Renaming would require a coordinated migration on both ends. Instead
# scope a warnings filter narrowly to this specific class-shadow message
# so callers don't see noise on every `import decimalai`.
warnings.filterwarnings(
    "ignore",
    message=r'Field name "schema_json" in "ComponentSnapshot" shadows an attribute in parent "BaseModel"',
    category=UserWarning,
    module=__name__,
)


def quantize_float(value: Optional[float], step: float = 0.1) -> Optional[float]:
    """Quantize a float to the nearest step for threshold-based hashing.

    0.71 → 0.7, 0.75 → 0.8. Same quantized value → same hash.
    Returns None if input is None.
    """
    if value is None:
        return None
    # Round to step precision, then round to 10 decimal places to avoid
    # IEEE 754 floating point artifacts (e.g. 0.7000000000000001)
    return round(round(value / step) * step, 10)


class ComponentSnapshot(BaseModel):
    """A snapshot of a single component for registration."""

    component_type: str  # tool / prompt / model / subagent / workflow / output_schema
    component_name: str
    component_version: Optional[str] = None
    content_hash: Optional[str] = None
    schema_json: Optional[Dict[str, Any]] = None


class ManifestSnapshot(BaseModel):
    """A snapshot of the agent's manifest for registration."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_name: str
    version_label: Optional[str] = None
    manifest_hash: str = ""
    detection_source: str = "auto"
    components: List[ComponentSnapshot] = Field(default_factory=list)
    agent_models_json: Optional[Dict[str, Any]] = None
    graph_topology_hash: Optional[str] = None
    component_summary_json: Optional[Dict[str, Any]] = None
    # Closed-world contracts enable runtime violation detection.
    # Defaults to False (descriptive) for backwards compatibility: SDK versions
    # in the field that don't set this field will continue to register
    # manifests in descriptive mode (no violations fired). Set by the explicit
    # register_manifest() path (via contract_mode="closed", the default there).
    is_closed_world: bool = False

    def to_a2a_card(self, *, url: Optional[str] = None) -> Dict[str, Any]:
        """Export as an A2A (Agent2Agent) Agent Card.

        Delegates to ``agentversion.a2a.manifest_to_agent_card`` (one shared
        implementation) so the card is the A2A-canonical shape — ``capabilities``
        as a feature dict, ``skills``/``defaultOutputModes`` descriptors — and
        carries an ``x-agentversion`` provenance block (``manifest_id`` +
        ``overall_hash`` + ``spec_version``). That block lets a card consumer pin
        the exact versioned contract the card describes, the capability A2A itself
        does not provide. Requires the ``agentversion`` package.

        Args:
            url: the agent's A2A service endpoint (optional; a deployment concern,
                not part of the internal contract).

        Returns:
            An A2A Agent Card dict.
        """
        from agentversion.a2a import manifest_to_agent_card

        return manifest_to_agent_card(self.to_agentversion(), url=url)

    def to_agentversion(
        self,
        *,
        version_label: Optional[str] = None,
        created_by: Optional[Dict[str, str]] = None,
        parent_manifest_id: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Export as an ``agentversion`` AgentManifest dict (the OSS spec shape).

        The SDK stores a manifest as a flat component list; ``agentversion``'s
        ``diff`` / ``validate`` read the contract-keyed shape (``contract.*``
        surfaces). This regroups the components by surface, fills the surfaces'
        required fields, and mints a canonical ``amf_<ULID>`` manifest id — i.e.
        it is the SDK→agentversion seam.

        The returned dict validates against ``agentversion.AgentManifest`` and is
        diffable by ``agentversion diff``. When ``agentversion`` (and its ``jcs``
        dependency) is importable, the canonical ``jcs-sha256`` overall hash is
        computed so the manifest validates with **zero** warnings; otherwise the
        SDK's own surface hash is carried under an honest ``decimalai-surface-
        sha256`` algorithm label (the validator downgrades a non-standard hash
        algorithm to a warning, never an error, and ``diff`` ignores the hash).

        This is a pure function — no network, and ``agentversion`` is an optional
        dependency, never imported at module load.
        """
        by_type: Dict[str, List[ComponentSnapshot]] = {}
        for c in self.components:
            by_type.setdefault(c.component_type, []).append(c)

        def _ref(c: ComponentSnapshot) -> Dict[str, str]:
            return {
                "id": c.component_name,
                "version": c.component_version or "1",
                "hash": c.content_hash or "",
            }

        # prompt_stack — agentversion exposes exactly two ref slots (system /
        # developer). Named prompts claim their slot; anything left fills the
        # open slots in order. Extra prompts beyond two have no ref slot in the
        # spec (a target-schema limit, not a converter one).
        prompt_stack: Dict[str, Any] = {}
        leftover: List[ComponentSnapshot] = []
        for c in by_type.get("prompt", []):
            key = c.component_name.lower()
            if key in ("system", "system_prompt"):
                prompt_stack.setdefault("system_prompt", _ref(c))
            elif key in ("developer", "developer_prompt", "instruction", "instructions"):
                prompt_stack.setdefault("developer_prompt", _ref(c))
            else:
                leftover.append(c)
        for c in leftover:
            if "system_prompt" not in prompt_stack:
                prompt_stack["system_prompt"] = _ref(c)
            elif "developer_prompt" not in prompt_stack:
                prompt_stack["developer_prompt"] = _ref(c)

        # model_runtime (required surface: provider + model). First model
        # component by name is primary; provider is read from config or inferred
        # from the model id; "unknown"/"unknown" only as a last resort so the
        # surface stays structurally valid even for a model-less manifest.
        model_comps = sorted(by_type.get("model", []), key=lambda c: c.component_name)
        if model_comps:
            primary = model_comps[0]
            cfg = primary.schema_json or {}
            model = cfg.get("model") or primary.component_version or "unknown"
            provider = cfg.get("provider") or _infer_provider(
                cfg.get("model") or primary.component_version or primary.component_name
            )
            model_runtime: Dict[str, Any] = {"provider": provider, "model": model}
            if primary.content_hash:
                model_runtime["model_config_hash"] = primary.content_hash
            # Type-guard each generation param: agentversion's GenerationConfig
            # types are strict (response_format is str, not dict), so a mistyped
            # value would fail validation — skip it instead.
            #
            # Export the QUANTIZED temperature/top_p (same steps used when the
            # model_config_hash is computed: 0.1 for temperature, 0.05 for
            # top_p). Otherwise the exported generation_config disagrees with
            # the hash — e.g. temp 0.71 vs 0.79 both hash to 0.7 but would
            # export different temperatures, confusing diff tools that treat
            # the hash as identity.
            gen: Dict[str, Any] = {}
            if isinstance(cfg.get("temperature"), (int, float)):
                gen["temperature"] = quantize_float(cfg["temperature"], 0.1)
            if isinstance(cfg.get("top_p"), (int, float)):
                gen["top_p"] = quantize_float(cfg["top_p"], 0.05)
            if isinstance(cfg.get("max_tokens"), int):
                gen["max_tokens"] = cfg["max_tokens"]
            if isinstance(cfg.get("response_format"), str):
                gen["response_format"] = cfg["response_format"]
            if gen:
                model_runtime["generation_config"] = gen
            # tool_calling_mode + runtime_version are TOP-LEVEL model_runtime
            # fields (not under generation_config); the diff classifies a
            # tool_calling_mode change as breaking. Must stay byte-identical with
            # agentversion.contract.contract_from_components.
            if isinstance(cfg.get("tool_calling_mode"), str):
                model_runtime["tool_calling_mode"] = cfg["tool_calling_mode"]
            if isinstance(cfg.get("runtime_version"), str):
                model_runtime["runtime_version"] = cfg["runtime_version"]
        else:
            model_runtime = {"provider": "unknown", "model": "unknown"}

        # tool_registry (required). registry_version is a constant "1": the SDK
        # doesn't version surfaces independently, so the registry_hash carries
        # the change signal (and agentversion diff compares the tools list too).
        tools_out: List[Dict[str, Any]] = []
        for c in by_type.get("tool", []):
            sj = c.schema_json or {}
            td: Dict[str, Any] = {"name": c.component_name, "hash": c.content_hash or ""}
            if c.component_version:
                td["version"] = c.component_version
            if sj.get("description"):
                td["description"] = sj["description"]
            if sj.get("input_schema_hash"):
                td["input_schema_hash"] = sj["input_schema_hash"]
            if sj.get("output_schema_hash"):
                td["output_schema_hash"] = sj["output_schema_hash"]
            if sj.get("stability") in ("experimental", "stable", "deprecated"):
                td["stability"] = sj["stability"]
            if isinstance(sj.get("annotations"), dict):
                td["annotations"] = sj["annotations"]
            tools_out.append(td)
        tool_registry = {
            "registry_version": "1",
            "registry_hash": _compute_surface_hash("tool_registry", self.components),
            "tools": tools_out,
        }

        # workflow (required surface, all fields optional → {} is valid).
        workflow: Dict[str, Any] = {}
        wf_comps = by_type.get("workflow", [])
        if wf_comps:
            wf = wf_comps[0]
            workflow["graph_name"] = wf.component_name
            if wf.component_version:
                workflow["graph_version"] = wf.component_version
            if wf.content_hash:
                workflow["graph_hash"] = wf.content_hash
        elif self.graph_topology_hash:
            workflow["graph_hash"] = self.graph_topology_hash

        # output_contract (required: version + schema_hash + format). With no
        # declared output schema, synthesize the "none" sentinel so the surface
        # is present and valid.
        out_comps = by_type.get("output_schema", [])
        if out_comps:
            oc = out_comps[0]
            sj = oc.schema_json or {}
            fmt = sj.get("format")
            output_contract = {
                "version": oc.component_version or "1",
                "schema_hash": oc.content_hash or "",
                "format": fmt if isinstance(fmt, str) else "json",
                "strict": bool(sj.get("strict", False)),
                "modalities": sj["modalities"] if isinstance(sj.get("modalities"), list) else [],
            }
        else:
            output_contract = {
                "version": "0", "schema_hash": "", "format": "none",
                "strict": False, "modalities": [],
            }

        contract: Dict[str, Any] = {
            "prompt_stack": prompt_stack,
            "model_runtime": model_runtime,
            "tool_registry": tool_registry,
            "workflow": workflow,
            "output_contract": output_contract,
        }

        # Optional surfaces — only emitted when the SDK manifest declares them.
        skill_comps = by_type.get("skill", [])
        if skill_comps:
            skills_out: List[Dict[str, Any]] = []
            for c in skill_comps:
                sj = c.schema_json or {}
                sd: Dict[str, Any] = {"name": c.component_name, "hash": c.content_hash or ""}
                if c.component_version:
                    sd["version"] = c.component_version
                if sj.get("description"):
                    sd["description"] = sj["description"]
                if sj.get("stability") in ("experimental", "stable", "deprecated"):
                    sd["stability"] = sj["stability"]
                skills_out.append(sd)
            contract["skill_registry"] = {
                "registry_version": "1",
                "registry_hash": _compute_surface_hash("skill_registry", self.components),
                "skills": skills_out,
            }

        subagents_out = [
            {"name": c.component_name, "version": c.component_version or "1",
             "hash": c.content_hash or ""}
            for c in by_type.get("subagent", [])
        ]
        if subagents_out:
            contract["subagents"] = subagents_out

        if by_type.get("guardrail"):
            contract["guardrails"] = {
                "bundle_version": "1",
                "bundle_hash": _compute_surface_hash("guardrails", self.components),
            }

        ctx_comps = by_type.get("context_config", [])
        if ctx_comps:
            ctx = {"retrieval_config_version": "1"}
            if ctx_comps[0].content_hash:
                ctx["retrieval_config_hash"] = ctx_comps[0].content_hash
            contract["context_config"] = ctx

        # behavioral_policy / environment are single structured surfaces (the
        # surface IS the config), emitted verbatim from schema_json. Must stay
        # byte-identical with agentversion.contract.contract_from_components.
        bp_comps = by_type.get("behavioral_policy", [])
        if bp_comps:
            contract["behavioral_policy"] = dict(bp_comps[0].schema_json or {})

        env_comps = by_type.get("environment", [])
        if env_comps:
            contract["environment"] = dict(env_comps[0].schema_json or {})

        vlabel = (
            version_label or self.version_label
            or (self.manifest_hash[:12] if self.manifest_hash else None)
            or "untagged"
        )

        manifest: Dict[str, Any] = {
            "kind": "agent_manifest",
            "manifest_id": _mint_amf_id(),
            "agent_name": self.agent_name,
            "version_label": vlabel,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": created_by or {"type": "system", "id": "decimalai-sdk"},
            "identity": {
                "overall_hash": self.manifest_hash or "",
                "hash_algorithm": "decimalai-surface-sha256",
            },
            "contract": contract,
        }

        try:
            from agentversion.constants import SPEC_VERSION as _spec
            manifest["spec_version"] = _spec
        except Exception:
            pass

        if parent_manifest_id:
            manifest["parent_manifest_id"] = parent_manifest_id
        if description:
            manifest["description"] = description

        # When agentversion (+ its jcs dep) is importable, recompute the
        # canonical jcs-sha256 hash so the export validates with zero warnings.
        # Otherwise keep the SDK surface hash under an honest algorithm label:
        # the validator downgrades a non-standard algorithm to a warning (never
        # an error), and `agentversion diff` ignores the overall hash entirely.
        try:
            from agentversion.hasher import compute_and_set_hashes
            compute_and_set_hashes(manifest)
        except Exception:
            pass

        return manifest


def _infer_provider(model_name: Optional[str]) -> str:
    """Best-effort provider inference from a model id (used when the model
    component carries no explicit ``provider``)."""
    name = (model_name or "").lower()
    if "gemini" in name or "google" in name:
        return "google"
    if "gpt" in name or "openai" in name or name.startswith(("o1", "o3", "o4")):
        return "openai"
    if "claude" in name or "anthropic" in name:
        return "anthropic"
    return "unknown"


_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _crockford_encode(value: int, length: int) -> str:
    """Encode ``value`` as a fixed-``length`` Crockford-base32 string (MSB
    first, zero-padded)."""
    out: List[str] = []
    for _ in range(length):
        value, rem = divmod(value, 32)
        out.append(_CROCKFORD[rem])
    return "".join(reversed(out))


def _mint_amf_id() -> str:
    """Mint an agentversion-canonical ``agent_manifest`` id: ``amf_<26 ULID>``.

    Mirrors agentversion's id grammar (``<prefix>_<26-char Crockford-base32
    ULID>``) without importing agentversion: a 48-bit millisecond timestamp
    (10 chars) followed by 80 bits of randomness (16 chars). The Crockford
    alphabet excludes I/L/O/U, matching agentversion's canonical id regex.
    """
    now_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand = int.from_bytes(secrets.token_bytes(10), "big")
    return f"amf_{_crockford_encode(now_ms, 10)}{_crockford_encode(rand, 16)}"


def _hash_content(content: Any) -> str:
    """SHA-256 hash of JSON-serialized content."""
    serialized = json.dumps(content, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _compute_surface_hash(surface_name: str, components: List[ComponentSnapshot]) -> str:
    """Compute a deterministic hash for one contract surface."""
    surface_comps = sorted(
        [c for c in components if _surface_for_type(c.component_type) == surface_name],
        key=lambda c: c.component_name,
    )
    if not surface_comps:
        return ""
    # Include component_version in the per-component dict so a version-only
    # bump changes the surface hash. Skills/guardrails set content_hash to a
    # caller-supplied `hash` when present, so without version here a caller
    # who bumps only the version (keeping the same explicit hash) would
    # produce an identical manifest_hash — and the new version would never
    # be registered. None vs "1.0" still differs, so the common case where
    # no version is supplied is unaffected.
    data = [
        {
            "name": c.component_name,
            "hash": c.content_hash,
            "version": c.component_version,
        }
        for c in surface_comps
    ]
    return _hash_content(data)


def _surface_for_type(component_type: str) -> str:
    """Map component_type to contract surface name."""
    mapping = {
        "tool": "tool_registry",
        "skill": "skill_registry",
        "prompt": "prompt_stack",
        "model": "model_runtime",
        "subagent": "subagents",
        "output_schema": "output_contract",
        "workflow": "workflow",
        "guardrail": "guardrails",
        "context_config": "context_config",
    }
    return mapping.get(component_type, component_type)


def _compute_overall_hash(components: List[ComponentSnapshot]) -> str:
    """Compute the overall manifest hash from sorted per-surface hashes.

    This ensures multi-instance convergence: same components → same hash,
    regardless of insertion order or which pod computes it.
    """
    surfaces = ["tool_registry", "skill_registry", "prompt_stack", "model_runtime",
                "subagents", "output_contract", "workflow", "guardrails",
                "context_config", "behavioral_policy", "environment"]
    surface_hashes = {}
    for surface in surfaces:
        h = _compute_surface_hash(surface, components)
        if h:
            surface_hashes[surface] = h
    return _hash_content(surface_hashes)


def extract_from_config(
    agent_name: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    prompts: Optional[Dict[str, str]] = None,
    models: Optional[Dict[str, Dict[str, Any]]] = None,
    subagents: Optional[List[Dict[str, Any]]] = None,
    output_schema: Optional[Dict[str, Any]] = None,
    workflow: Optional[Dict[str, Any]] = None,
    skills: Optional[List[Dict[str, Any]]] = None,
    guardrails: Optional[List[Dict[str, Any]]] = None,
    context_config: Optional[Dict[str, Any]] = None,
    behavioral_policy: Optional[Dict[str, Any]] = None,
    environment: Optional[Dict[str, Any]] = None,
    version_label: Optional[str] = None,
    is_closed_world: bool = False,
) -> ManifestSnapshot:
    """Extract a manifest snapshot from explicit configuration.

    This is the primary extraction method. Each parameter adds
    components to the snapshot.

    Args:
        agent_name: Name of the agent.
        tools: List of tool descriptors [{name, schema, ...}].
        prompts: Dict of prompt_name → prompt_text.
        models: Dict of node_name → {provider, model, ...}.
        subagents: List of subagent descriptors.
        output_schema: Output schema dict.
        workflow: Workflow/graph metadata dict.
        skills: List of skill descriptors [{name, hash, description, ...}].
        guardrails: List of validator/guardrail descriptors
            ``[{"name": "pii_filter", "kind": "pii", "scope": "output", ...}]``.
            Populates the ``guardrails`` contract surface — contract
            enforcement uses it to detect ``guardrail_missing`` (declared
            but never ran in production) and ``guardrail_failed``
            (declared, ran, errored).
        context_config: Retrieval + memory config
            ``{"retrieval": {"source": "pinecone", "index_name": "...", "top_k": 5},
               "memory": {"policy": "session", "max_turns": 10}}``.
            Populates the ``context_config`` contract surface — contract
            enforcement uses it to detect ``context_source_undeclared``.
        behavioral_policy: Versioned policy-document surface
            ``{"policy_id": ..., "policy_hash": ..., "rules": {...}?}`` — binds
            the agent to a named policy artifact (refund rules, a safety
            guardrail set, an escalation SOP, …) by hash. The dict is opaque to
            the SDK: it is hashed whole (or by the supplied ``policy_hash``), so
            ANY change to the bound policy diffs as breaking (replay/drop) and a
            policy flip no longer hides in the prompt hash; an unchanged
            ``policy_hash`` is non-breaking.
        environment: Deployment/infra surface
            ``{"deployment_id", "region", "infra_image_hash", "runtime_versions",
               "secret_refs", "external_service_pins", "feature_flags",
               "resource_limits"}``. Note ``runtime_versions`` (plural, the infra
            map) is distinct from ``model_runtime.runtime_version`` (the SDK/runtime
            version passed via ``models[...]``).
        version_label: Human-readable version label.
        is_closed_world: When True, the manifest is treated as a closed contract —
            the backend will flag runtime contract violations (calls to undeclared
            tools, out-of-scope models, off-schema outputs, etc.) when traces are
            ingested under this manifest. Default False (descriptive mode); the
            top-level :func:`decimalai.register_manifest` defaults this to True
            since explicit declaration IS the act of declaring a contract.

    Returns:
        A ManifestSnapshot ready for registration.
    """
    components: List[ComponentSnapshot] = []
    summary: Dict[str, int] = {}

    # Tools
    if tools:
        summary["tools"] = len(tools)
        for tool in tools:
            name = tool.get("name", "unknown")
            schema = tool.get("schema") or tool.get("input_schema")
            components.append(ComponentSnapshot(
                component_type="tool",
                component_name=name,
                component_version=tool.get("version"),
                content_hash=_hash_content(tool),
                schema_json={
                    "input_schema_hash": _hash_content(schema) if schema else None,
                    "output_schema_hash": _hash_content(tool.get("output_schema")) if tool.get("output_schema") else None,
                    "description": tool.get("description"),
                    "annotations": tool.get("annotations"),
                    "stability": tool.get("stability"),
                },
            ))

    # Prompts
    if prompts:
        summary["prompts"] = len(prompts)
        for prompt_name, prompt_text in prompts.items():
            components.append(ComponentSnapshot(
                component_type="prompt",
                component_name=prompt_name,
                content_hash=_hash_content(prompt_text),
                schema_json={"content": prompt_text, "text_preview": prompt_text[:200]},
            ))

    # Models — quantize floats for threshold-based hashing
    if models:
        summary["models"] = len(models)
        for node_name, model_config in models.items():
            # Quantize generation params for hashing
            hash_config = dict(model_config)
            if "temperature" in hash_config:
                hash_config["temperature"] = quantize_float(hash_config["temperature"], 0.1)
            if "top_p" in hash_config:
                hash_config["top_p"] = quantize_float(hash_config["top_p"], 0.05)
            # Exclude runtime-only settings from hash
            for runtime_key in ("max_retries", "timeout", "rate_limit", "batch_size"):
                hash_config.pop(runtime_key, None)
            components.append(ComponentSnapshot(
                component_type="model",
                component_name=node_name,
                component_version=model_config.get("model"),
                content_hash=_hash_content(hash_config),
                schema_json=model_config,  # Store original (un-quantized) for display
            ))

    # Subagents
    if subagents:
        summary["subagents"] = len(subagents)
        for sa in subagents:
            components.append(ComponentSnapshot(
                component_type="subagent",
                component_name=sa.get("name", "unknown"),
                component_version=sa.get("version"),
                content_hash=_hash_content(sa),
            ))

    # Output schema
    if output_schema:
        summary["output_schemas"] = 1
        components.append(ComponentSnapshot(
            component_type="output_schema",
            component_name="output",
            content_hash=_hash_content(output_schema),
            schema_json={
                **output_schema,
                "modalities": output_schema.get("modalities", []),
            },
        ))

    # Workflow
    if workflow:
        summary["workflows"] = 1
        components.append(ComponentSnapshot(
            component_type="workflow",
            component_name=workflow.get("name", "graph"),
            content_hash=_hash_content(workflow),
            schema_json=workflow,
        ))

    # Skills
    if skills:
        summary["skills"] = len(skills)
        for skill in skills:
            name = skill.get("name", "unknown")
            components.append(ComponentSnapshot(
                component_type="skill",
                component_name=name,
                component_version=skill.get("version"),
                content_hash=skill.get("hash") or _hash_content(skill),
                schema_json={
                    "description": skill.get("description", ""),
                    "stability": skill.get("stability"),
                },
            ))

    # Guardrails — input/output validators, PII checks, toxicity classifiers, etc.
    # Each entry is hashed independently so a validator change re-fires the
    # surface hash. Used at runtime to detect `guardrail_missing`
    # (declared but never ran) and `guardrail_failed` (declared, ran, errored).
    if guardrails:
        summary["guardrails"] = len(guardrails)
        for guard in guardrails:
            name = guard.get("name", "unknown")
            components.append(ComponentSnapshot(
                component_type="guardrail",
                component_name=name,
                component_version=guard.get("version"),
                content_hash=guard.get("hash") or _hash_content(guard),
                schema_json={
                    "description": guard.get("description", ""),
                    "kind": guard.get("kind"),  # e.g., "pii_filter", "toxicity_check"
                    "scope": guard.get("scope"),  # "input" | "output" | "both"
                },
            ))

    # Context config — retrieval source + memory policy.
    # The manifest declares ONE context config (the surface is single-row).
    # Used at runtime to detect `context_source_undeclared`
    # (retrieval span with a source the manifest didn't declare).
    if context_config:
        summary["context_config"] = 1
        components.append(ComponentSnapshot(
            component_type="context_config",
            component_name="context",
            content_hash=_hash_content(context_config),
            schema_json=context_config,  # {retrieval: {source, index_name, top_k, ...}, memory: {policy, max_turns}}
        ))

    # behavioral_policy — versioned policy-document surface (agentversion 0.2.0).
    # The dict is opaque; it's hashed whole (or by the supplied policy_hash), so ANY
    # change to the bound policy artifact diffs as breaking → replay/drop, and a
    # policy flip no longer hides in the prompt hash.
    if behavioral_policy:
        summary["behavioral_policy"] = 1
        components.append(ComponentSnapshot(
            component_type="behavioral_policy",
            component_name="behavioral_policy",
            content_hash=behavioral_policy.get("policy_hash") or _hash_content(behavioral_policy),
            schema_json=behavioral_policy,
        ))

    # environment — deployment/infra surface (region, infra_image_hash,
    # runtime_versions, external_service_pins, …); changes diff as non_breaking
    # but drive replay decisions.
    if environment:
        summary["environment"] = 1
        components.append(ComponentSnapshot(
            component_type="environment",
            component_name="environment",
            content_hash=_hash_content(environment),
            schema_json=environment,
        ))

    # Compute overall hash from sorted per-surface hashes
    manifest_hash = _compute_overall_hash(components)

    return ManifestSnapshot(
        agent_name=agent_name,
        version_label=version_label,
        manifest_hash=manifest_hash,
        detection_source="sdk",
        components=components,
        agent_models_json={k: v for k, v in (models or {}).items()},
        graph_topology_hash=_hash_content(workflow) if workflow else None,
        component_summary_json=summary,
        is_closed_world=is_closed_world,
    )


class ManifestTracker:
    """Tracks manifest changes and triggers registration.

    Use this in the SDK to detect when the agent's manifest changes
    and automatically register new versions.
    """

    #: How many distinct hashes one process remembers. Bounded because a
    #: long-lived worker could otherwise accumulate one entry per config change
    #: for the life of the process; 64 is far more than any real agent produces
    #: and costs a few KB.
    _MAX_REMEMBERED = 64

    def __init__(self) -> None:
        self._last_hash: Optional[str] = None
        self._last_manifest: Optional[ManifestSnapshot] = None
        # EVERY hash this process has already registered, not just the previous
        # one. A single slot re-registers whenever the snapshot OSCILLATES —
        # A -> B -> A sends three requests for two manifests, and the third is
        # one the server already has. Measured on prod 2026-09-03:
        # `POST /api/v1/manifests` is 17.9% of ALL backend traffic and ~99.4% of
        # it dedupes server-side.
        #
        # ⚠ This does NOT suppress a genuinely new hash. An agent that discovers
        # a tool mid-run has a different manifest and the platform must be told;
        # re-registering there is the product working, not waste. Only the
        # already-seen case is dropped.
        self._seen_hashes: "OrderedDict[str, None]" = OrderedDict()

    @property
    def last_hash(self) -> Optional[str]:
        return self._last_hash

    @property
    def last_manifest(self) -> Optional[ManifestSnapshot]:
        return self._last_manifest

    def check_and_update(self, snapshot: ManifestSnapshot) -> bool:
        """Check if the manifest has changed.

        Returns True if the manifest is new or changed (should be registered).
        Returns False if it's the same as the last known version.
        """
        h = snapshot.manifest_hash
        if self._last_hash == h or h in self._seen_hashes:
            # Keep `last_*` truthful about what was most recently OFFERED, so a
            # caller reading them after an oscillation sees the current shape.
            self._last_hash = h
            self._last_manifest = snapshot
            self._seen_hashes.move_to_end(h, last=True)
            return False

        self._last_hash = h
        self._last_manifest = snapshot
        self._seen_hashes[h] = None
        while len(self._seen_hashes) > self._MAX_REMEMBERED:
            self._seen_hashes.popitem(last=False)   # LRU
        return True

    def reset(self) -> None:
        """Reset the tracker state — this process forgets everything it registered.

        ⚠ `_seen_hashes` MUST be cleared here, and the first version of it was not.
        `reset()` is the ROLLBACK the caller uses when a registration fails
        (`generic.py`: "Roll back the tracker so a transient first-trace failure
        does not suppress every later attempt"). A remembered hash that survived
        the rollback would suppress the retry, and the next trace citing a
        manifest the backend never stored is exactly the "manifest_id does not
        exist" 400 this whole area exists to avoid.

        Caught by test_failed_registration_does_not_poison_tracker and
        test_reset, which is what they are for.
        """
        self._last_hash = None
        self._last_manifest = None
        self._seen_hashes.clear()
