"""Run the linked Colab notebooks' actual tools and loops without a model or keys."""
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
QUESTION = "I opened my kettle from ORD-1001 and it works, but I don't want it. How much do I get back if I return it?"
SAVED = "An older saved prompt says billing will follow up."
ANSWER = "You'd get back **$41.65**.\n\nStart at acme.example/returns."


def notebook(sdk):
    return json.loads((HERE / f"order_support_{sdk}.ipynb").read_text())


def program(sdk):
    return "\n\n".join("".join(cell["source"]) for cell in notebook(sdk)["cells"]
                       if cell["cell_type"] == "code" and "def lookup_order" in "".join(cell["source"]))


def load_program(sdk):
    """Execute the real notebook definitions, substituting only SDK/network objects."""
    source = program(sdk)
    tree = ast.parse(source)
    names = {"ORDERS", "TOOLS", "BY_NAME", "agent"}
    functions = {"lookup_order", "flag_for_billing", "lock_account",
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
def test_actual_notebook_loop_passes_the_saved_prompt_as_it_is_and_returns_the_reply(sdk):
    env = load_program(sdk)
    seen = []

    if sdk == "openai_agents":
        class Runner:
            @staticmethod
            async def run(agent, question):
                seen.append(await agent.instructions(None, agent))
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
                items = [SimpleNamespace(type="function_call", name="lookup_order",
                                         arguments=json.dumps({"order_id": "ORD-1001"}), call_id="0")] if len(requests) == 1 else []
                return SimpleNamespace(output=items, output_text=ANSWER)
            env["OpenAI"] = lambda: SimpleNamespace(responses=SimpleNamespace(create=create))
        else:
            def tool_runner(**kwargs):
                seen.append(kwargs["system"])
                return SimpleNamespace(until_done=lambda: SimpleNamespace(content=[SimpleNamespace(type="text", text=ANSWER)]))
            env["client"] = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(tool_runner=tool_runner)))
        def run():
            return env["answer"](QUESTION)

    assert run() == ANSWER
    assert seen  # the actual callback/model loop ran
    for instructions in seen:
        assert instructions in (SAVED, [{"type": "text", "text": SAVED}])


@pytest.mark.parametrize("sdk", SDKS)
def test_tools_report_what_happened_and_an_unknown_order_flags_nothing(sdk):
    env = load_program(sdk)
    assert "electric kettle" in env["lookup_order"]("ORD-1001")
    assert env["lookup_order"]("ORD-9999") == "No order with that id."
    assert env["flag_for_billing"]("ORD-9999", "check") == "No order with that id."
    assert env["flag_for_billing"]("ORD-1002", "check") == "Flagged ORD-1002 for the billing team: check"


@pytest.mark.parametrize("sdk", SDKS)
def test_the_model_is_told_not_to_claim_an_action_and_the_reader_is_pointed_at_the_check(sdk):
    assert "Only describe an action as done after" in program(sdk)
    text = "".join("".join(cell["source"]) for cell in notebook(sdk)["cells"] if cell["cell_type"] == "markdown")
    assert "A model can say it did something without doing it" in text and "/measure-it" in text


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
