"""Run ONE generated agent file, exactly as ``python agent.py`` would.

    python tests/conformance/_journey_run.py <agent.py>

Invoked by ``journey.run_journey``; not a test module (pytest collects
``test_*.py`` only). It exists for a single reason, and it is worth stating
plainly because a wrapper around the thing under test is exactly where a suite
starts fooling itself.

**It does not touch the file, the SDK, the model or the skills.** ``runpy`` runs
the generated source unmodified, under ``__main__``, so ``if __name__ ==
"__main__":`` fires and the file's own ``print(run(...))`` is what lands on
stdout. Everything the journey grades comes from that run.

The one thing it changes is FOREIGN TELEMETRY — a framework's own exporter that
would leave this machine:

* The OpenAI Agents SDK installs a default ``BackendSpanExporter`` whose endpoint
  is hardcoded to ``https://api.openai.com/v1/traces/ingest`` and whose key falls
  back to ``OPENAI_API_KEY``. Pointing the MODEL at a local stub does not move it.
  Left alone, every journey run makes a real outbound request to OpenAI carrying
  a stub key — measured: ``[non-fatal] Tracing client error 401`` — which is not
  hermetic, is slow behind a firewall, and is not ours to send.

Clearing that processor list is not a shortcut past anything the tier grades: the
DecimalAI adapter installs its own processor when the generated file calls
``instrument()``, AFTER this runs, and the trace still reaches the probe (verified
— the ``POST /api/v1/traces`` is in ``run_phase.requests``). If the neutralisation
is ever impossible it exits LOUDLY rather than running anyway, because a silent
fallback here would mean an unnoticed network call in every future run.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

#: Exit codes distinct from anything the generated file can produce, so
#: "the runner could not set up" is never read as "the agent failed".
EXIT_USAGE = 2
EXIT_NEUTRALISE_FAILED = 4


def _neutralise_openai_agents_tracing() -> str:
    """Drop the Agents SDK's default trace exporter. Returns what happened."""
    try:
        import agents  # noqa: F401
    except ImportError:
        return "openai-agents not installed — nothing to neutralise"
    try:
        from agents.tracing import set_trace_processors
    except ImportError as exc:
        print(
            "[conformance] openai-agents is installed but "
            f"agents.tracing.set_trace_processors is gone ({exc}). Refusing to run: "
            "its default exporter posts to https://api.openai.com/v1/traces/ingest, "
            "and this tier must make no outbound provider call. Find the current "
            "way to replace the processor list and update this runner.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_NEUTRALISE_FAILED) from exc
    # Empty, not "replace with a no-op exporter": the DecimalAI adapter adds its
    # own processor when the generated file runs instrument(), so the list is
    # rebuilt by the code under test rather than by this file.
    set_trace_processors([])
    return "cleared the openai-agents default trace processors"


def main(argv: list) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <generated-agent-file>", file=sys.stderr)
        return EXIT_USAGE
    target = Path(argv[1])
    if not target.is_file():
        print(f"[conformance] no such generated file: {target}", file=sys.stderr)
        return EXIT_USAGE

    # On stderr, and captured: the journey's evidence should say what was done to
    # the environment it ran in, not just what came out of it.
    print(f"[conformance] {_neutralise_openai_agents_tracing()}", file=sys.stderr)

    # `python agent.py` puts the script's directory on sys.path; do the same, so
    # a generated file that ever imports a sibling behaves identically here.
    sys.path.insert(0, str(target.parent))
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
