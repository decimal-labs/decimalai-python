"""A scenario for the runner's own test: no framework, just the contract of a child."""
from __future__ import annotations

import os


def echo(ctx: dict) -> dict:
    return {"cwd": os.getcwd(), "ctx": ctx,
            "tracing_off": os.environ.get("OTEL_SDK_DISABLED"),
            "decimal_env": sorted(k for k in os.environ if k.startswith("DECIMAL"))}


def boom(ctx: dict) -> dict:
    raise ValueError("scenario failed on purpose")
