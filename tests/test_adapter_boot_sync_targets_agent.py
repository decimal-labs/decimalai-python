"""An adapter's boot sync puts the skills it creates on its own agent, and only those.

New skills on the platform are opt-in since 2026-10-22: a skill reaches only the agents it is
added to. An adapter syncs the skills you wrote up on every agent boot, so it names its agent
(the skill was written for it) with ``add_to_agent="unless_removed"``: every synced skill goes on
the agent except one a person removed from it on the dashboard, which a restart must never undo.
("created", the first version, added only the skills that boot created, so a second agent booting
from the same skills dir never got them.) A static check: the boot sync runs on a background
sender with process-wide state, and what matters here is the call's arguments.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "decimalai"


def _sync_calls(module: str):
    tree = ast.parse((PKG / module).read_text(encoding="utf-8"))
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "sync_skills"
    ]


@pytest.mark.parametrize("module", ["langchain.py", "openai_agents.py"])
def test_the_boot_sync_names_its_agent_and_never_undoes_a_removal(module):
    calls = _sync_calls(module)
    assert len(calls) == 1, f"{module}: expected one boot sync, found {len(calls)}"
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert "agent_name" in kw, f"{module}: the boot sync does not name its agent"
    assert isinstance(kw.get("add_to_agent"), ast.Constant) and kw["add_to_agent"].value == "unless_removed", (
        f"{module}: the boot sync must send add_to_agent='unless_removed', or every restart re-adds "
        "skills removed from the agent on the dashboard"
    )
