"""The generated file EXECUTES — not just parses.

`tests/test_cli_init_scaffold.py` proves the template is valid Python and that
`enable_skill_loader=True` and the agent name are really there. That is the
cheap half. It cannot catch the expensive half: a template that compiles fine
while calling an `instrument()` keyword the adapter does not accept, importing
a symbol that moved, or constructing an `Agent(...)` whose signature changed.
Every one of those ships a file that dies on line one for the user, and the
product claim is that the file RUNS.

So this module writes the generated file to a temp directory and runs it in a
SUBPROCESS, with exactly one thing stubbed: the LLM call. Everything else is
the real code path — the real imports, the real `instrument()`, the real
`Agent(...)` / `init_chat_model(...)`, the real entry point.

A subprocess and not an in-process exec, because `instrument()` is a
process-wide install: it monkey-patches `Agent.__init__` (openai-agents) and
`BaseChatModel.invoke` (langchain) and registers a global trace processor.
Running that inside pytest would leak those patches into every test that came
after it.

`decimalai.init()` is the one other stub. It is not what is under test here,
and left alone it makes a real network call to verify the (fake) key.

Each framework skips when its runtime package is absent, so a core-only
checkout still runs the suite. `agents` is in the dev extras and therefore
runs on every commit; `langchain` is not, and runs wherever it is installed.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from decimalai.cli.scaffold import render_agent_file

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT = "refund-bot"
SKILLS = [{"skill_name": "refund-policy", "description": "When a refund is allowed",
           "scope": "agent"}]

# What each framework needs installed, and how to stub its one LLM call.
# The stub replaces the model, never the DecimalAI seam — patching the seam
# would make the test pass on a template that never reaches it.
RUNTIMES = {
    "openai-agents": {
        "requires": "agents",
        "stub": """
import agents
class _Result:
    final_output = "stubbed answer"
_stack.enter_context(patch.object(agents.Runner, "run_sync", return_value=_Result()))
""",
        "probe": """
import decimalai.openai_agents as adapter
print("LOADER:", adapter._skill_loader_installed)
# Read the name off the processor the adapter actually registered. The
# earlier version of this probe printed the literal "refund-bot", which
# made the assertion below unfalsifiable — it passed whatever instrument()
# did with the name. openai-agents has no module-level equivalent of
# langchain's _install_agent_name; the name lives on the processor as
# default_agent_name, so that is what gets read.
from agents.tracing import get_trace_provider
_procs = get_trace_provider()._multi_processor._processors
_ours = [p for p in _procs if hasattr(p, "default_agent_name")]
print("PROCESSORS:", len(_ours))
print("BOUND:", _ours[0].default_agent_name if _ours else "<no processor>")
""",
    },
    "langchain": {
        "requires": "langchain",
        "stub": """
from langchain_core.messages import AIMessage
class _Fake:
    def invoke(self, x, *a, **k):
        return AIMessage(content="stubbed answer")
_stack.enter_context(patch("langchain.chat_models.init_chat_model", return_value=_Fake()))
""",
        "probe": """
import decimalai.langchain as adapter
print("LOADER:", adapter._skill_loader_installed)
print("BOUND:", adapter._install_agent_name)
""",
    },
}

DRIVER = """\
import contextlib, runpy
from unittest.mock import patch

_stack = contextlib.ExitStack()
{stub}
# init() is stubbed only because it would make a real network call to verify a
# fake key. The adapter seam below it is NOT stubbed — that is the point.
_init = _stack.enter_context(patch("decimalai.init"))
_flush = _stack.enter_context(patch("decimalai.flush"))
with _stack:
    runpy.run_path({path!r}, run_name="__main__")
    print("INIT:", _init.called)
    print("FLUSH:", _flush.called)
{probe}
"""


def _have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


@pytest.mark.parametrize("framework", sorted(RUNTIMES))
def test_generated_file_executes(framework, tmp_path):
    spec = RUNTIMES[framework]
    if not _have(spec["requires"]):
        pytest.skip(f"{spec['requires']} not installed")

    agent_py = tmp_path / "agent.py"
    agent_py.write_text(render_agent_file(AGENT, framework, SKILLS))
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER.format(
        stub=spec["stub"], probe=spec["probe"], path=str(agent_py),
    ))

    proc = subprocess.run(
        [sys.executable, str(driver)],
        capture_output=True, text=True, cwd=tmp_path, timeout=120,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(REPO_ROOT),
            "DECIMAL_API_KEY": "dai_sk_not_a_real_key",
            # The generated file legitimately turns the loader on; inside a
            # disk-runtime harness that prints a duplicate-skills warning to
            # stderr, which is correct advice and noise here.
            "DECIMALAI_SUPPRESS_DISK_RUNTIME_WARNING": "1",
            "DECIMAL_AUTOINIT": "false",
        },
    )
    assert proc.returncode == 0, (
        f"generated {framework} file failed to run:\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    out = proc.stdout

    # The example call ran and produced the model's answer.
    assert "stubbed answer" in out, out
    assert "INIT: True" in out and "FLUSH: True" in out, out

    # The load-bearing one: `instrument(enable_skill_loader=True)` did not
    # merely parse — the adapter accepted it and the loader is INSTALLED.
    # A template that passes an argument the adapter silently ignores would
    # satisfy every AST assertion in the sibling module and fail here.
    assert "LOADER: True" in out, out

    # And the name reached the adapter, so traces file against the agent the
    # user configured rather than an auto-detected one.
    assert f"BOUND: {AGENT}" in out, out

    # Exactly one DecimalAI processor. Two would mean every span is sent
    # twice — the failure `init(openai_agents=True)` plus `instrument()`
    # produces, and the reason the template passes a bare `init()`.
    if framework == "openai-agents":
        assert "PROCESSORS: 1" in out, out
