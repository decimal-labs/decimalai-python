"""Shared enums and base types for the DecimalAI SDK."""

from __future__ import annotations

from enum import Enum


class SpanType(str, Enum):
    """Type of span in a trace."""
    AGENT = "agent"
    LLM = "llm"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    ROUTER = "router"
    JUDGE = "judge"
    HANDOFF = "handoff"        # Agent-to-agent transfer
    GUARDRAIL = "guardrail"    # Validation / safety check
    MEMORY = "memory"          # Memory read / write
    PLANNING = "planning"      # Agent planning step
    OTHER = "other"


class Status(str, Enum):
    """Execution status."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    RUNNING = "running"


class SourceType(str, Enum):
    """How this trace was generated."""
    PRODUCTION = "production"
    REPLAYED = "replayed"
    SYNTHETIC = "synthetic"
    DISTILLATION = "distillation"
    MANUAL = "manual"


class CallRole(str, Enum):
    """Role of an LLM call within the agent."""
    PLANNER = "planner"
    RESPONDER = "responder"
    SUBAGENT = "subagent"
    JUDGE = "judge"
    ROUTER = "router"
    OTHER = "other"


class FinishReason(str, Enum):
    """Why the LLM stopped generating."""
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    ERROR = "error"
