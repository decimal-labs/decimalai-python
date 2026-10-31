"""Run the linked Colab notebooks' actual action checks and loops without a model or keys."""
from __future__ import annotations

import ast
import asyncio
import inspect
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1] / "examples" / "tutorial"
SDKS = ("openai_agents", "openai", "anthropic")
QUESTION = "I was charged twice for order ORD-1002. Can you fix it?"
SAVED = "An older saved prompt says billing will follow up."
ANSWER = "Charge: $129\nCharge: $129\nI've flagged ORD-1002 for billing review."


def notebook(sdk):
    return json.loads((HERE / f"order_support_{sdk}.ipynb").read_text())


def program(sdk):
    return "\n\n".join("".join(cell["source"]) for cell in notebook(sdk)["cells"]
                       if cell["cell_type"] == "code" and "def checked_reply" in "".join(cell["source"]))


def load_program(sdk):
    """Execute the real notebook definitions, substituting only SDK/network objects."""
    source = program(sdk)
    tree = ast.parse(source)
    names = {"ORDERS", "ACTION_RULES", "confirmed", "TOOLS", "BY_NAME", "agent"}
    functions = {"checked_reply", "lookup_order", "flag_for_billing", "lock_account",
                 "start_account_deletion", "schema", "instructions", "answer"}
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in names for target in node.targets):
            body.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in functions:
            node.decorator_list = []  # call the actual tool body without importing a provider SDK
            body.append(node)

    async def saved_instructions(context, agent):
        return SAVED

    env = {"inspect": inspect, "json": json,
           "support": SimpleNamespace(name="order-support", tools=[], instructions=saved_instructions),
           "Agent": lambda **kwargs: SimpleNamespace(**kwargs)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), "notebook program", "exec"), env)
    if sdk == "openai_agents":
        ask_source = next("".join(cell["source"]) for cell in notebook(sdk)["cells"]
                          if cell["cell_type"] == "code" and "async def reply" in "".join(cell["source"]))
        reply = next(node for node in ast.parse(ask_source).body if isinstance(node, ast.AsyncFunctionDef) and node.name == "reply")
        exec(compile(ast.Module(body=[reply], type_ignores=[]), "notebook reply", "exec"), env)
    return env


@pytest.mark.parametrize("sdk", SDKS)
def test_notebook_cells_compile_without_outputs_or_literal_credentials(sdk):
    nb = notebook(sdk)
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None and cell["outputs"] == []
        text = "\n".join(line for line in "".join(cell["source"]).splitlines()
                         if not line.lstrip().startswith(("!", "%")))
        compile(text, f"{sdk} cell {index}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    assert not re.search(r"dai_sk_[A-Za-z0-9_]{12,}|sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}", json.dumps(nb))


@pytest.mark.parametrize("sdk", SDKS)
@pytest.mark.parametrize("outcome", ("missing", "failed", "wrong_order", "success"))
def test_actual_notebook_loop_checks_current_run_and_keeps_saved_prompt_policy(sdk, outcome):
    env = load_program(sdk)
    # A previous successful run cannot authorize the next model answer.
    env["confirmed"].update({("lookup_order", "ORD-1002"), ("flag_for_billing", "ORD-1002")})
    seen = []

    def calls():
        if outcome == "missing":
            return []
        order = {"failed": "ORD-9999", "wrong_order": "ORD-1001", "success": "ORD-1002"}[outcome]
        return [("lookup_order", {"order_id": "ORD-1002"}),
                ("flag_for_billing", {"order_id": order, "note": "check the duplicate"})]

    def perform_tools():
        for name, args in calls():
            env[name](**args)

    if sdk == "openai_agents":
        class Runner:
            @staticmethod
            async def run(agent, question):
                seen.append(await agent.instructions(None, agent))
                perform_tools()
                return SimpleNamespace(final_output=ANSWER)
        env["Runner"] = Runner
        def run():
            return asyncio.run(env["reply"](QUESTION))
    else:
        system = SAVED if sdk == "openai" else [{"type": "text", "text": SAVED}]
        value = SimpleNamespace(system=system, message={"role": "user", "content": QUESTION},
                                response_tools=[], tools=[], tool_result=lambda call: None)

        class Run:
            def __enter__(self):
                return value
            def __exit__(self, *args):
                return False
        env["support"].run = lambda question: Run()
        if sdk == "openai":
            requests = []
            def create(**kwargs):
                seen.append(kwargs["instructions"])
                requests.append(kwargs)
                items = [SimpleNamespace(type="function_call", name=name, arguments=json.dumps(args), call_id=str(i))
                         for i, (name, args) in enumerate(calls())] if len(requests) == 1 else []
                return SimpleNamespace(output=items, output_text=ANSWER)
            env["OpenAI"] = lambda: SimpleNamespace(responses=SimpleNamespace(create=create))
        else:
            def tool_runner(**kwargs):
                seen.append(kwargs["system"])
                def until_done():
                    perform_tools()
                    return SimpleNamespace(content=[SimpleNamespace(type="text", text=ANSWER)])
                return SimpleNamespace(until_done=until_done)
            env["client"] = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(tool_runner=tool_runner)))
        def run():
            return env["answer"](QUESTION)

    if outcome == "success":
        assert run() == ANSWER  # formatting is preserved, not replaced with a repaired answer
    else:
        with pytest.raises(RuntimeError, match="reply not accepted"):
            run()
    assert seen  # the actual callback/model loop ran
    for instructions in seen:
        if isinstance(instructions, list):
            assert instructions == [{"type": "text", "text": SAVED}, {"type": "text", "text": env["ACTION_RULES"]}]
        else:
            assert instructions == SAVED + "\n\n" + env["ACTION_RULES"]


@pytest.mark.parametrize("sdk", SDKS)
def test_deletion_requires_the_actual_link_tool_not_only_an_account_lock(sdk):
    env = load_program(sdk)
    question = "Please delete my account and all my data today."
    env["lock_account"]()
    with pytest.raises(RuntimeError):
        env["checked_reply"]("The confirmation link was sent.", question)
    env["start_account_deletion"]()
    assert env["checked_reply"]("The confirmation link was sent.", question) == "The confirmation link was sent."


def test_openai_notebook_can_start_and_skip_with_a_blank_provider_key(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    constructors = []

    def OpenAI():
        constructors.append(True)
        raise RuntimeError("A provider client cannot be constructed with a blank key")

    env = {"inspect": inspect, "json": json, "os": os, "asyncio": asyncio, "OpenAI": OpenAI,
           "agent_config": lambda name, **kwargs: SimpleNamespace(name=name, tools=[])}
    tree = ast.parse(program("openai"))
    body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(ast.Module(body=body, type_ignores=[]), "blank-key notebook setup", "exec"), env)
    ask_source = next("".join(cell["source"]) for cell in notebook("openai")["cells"]
                      if cell["cell_type"] == "code" and "async def ask" in "".join(cell["source"]))
    body = [node for node in ast.parse(ask_source).body if isinstance(node, ast.AsyncFunctionDef)]
    exec(compile(ast.Module(body=body, type_ignores=[]), "blank-key notebook ask", "exec"), env)
    asyncio.run(env["ask"](QUESTION))
    assert "Skipped: no OPENAI_API_KEY" in capsys.readouterr().out
    assert constructors == []
