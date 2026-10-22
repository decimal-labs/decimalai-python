"""Deep Agents native-mode scenarios (child side; no assertions).

The documented native mode: ``decimalai.init(langchain=True)`` (the LangChain adapter
traces the graph; since 0.25 that is all it does), then
``create_deep_agent(model, **decimalai.deepagents.skills(backend=...))``. The model is
``tests.native_loaders.stubs.deepagents_model.ScriptedModel``: each step reads what the
agent was shown (the skills listing in the system prompt, a tool result) and acts on it,
using only paths the listing gave it.

``column`` is the native column's scenario (``column.py`` describes the shape).
``checks`` runs a Deep Agents agent and a plain LangGraph agent side by side with skills in
the workspace, a pinned skill, and a Deep Agents agent that also has ``agent_config()``'s
middleware, and returns what each model call was shown and what each run logged.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List

from tests.native import column as col
from tests.native.column import RESOURCE, SCRIPT, SCRIPT_ARGS, TARGET, Run, Session

DISTRIBUTIONS = ["deepagents", "langchain-core", "langgraph"]
USER = "refund for $100 after 45 days?"


def _system_text(messages) -> str:
    from langchain_core.messages import SystemMessage

    texts = []
    for m in messages:
        if isinstance(m, SystemMessage):
            texts.append(m.text if isinstance(m.text, str) else m.text())
    return "\n".join(texts)


def listed_path(messages, name: str = TARGET) -> str:
    """The ``SKILL.md`` path the listing gives ``name``, as a model reads it."""
    m = re.search(r"(?m)^- \*\*" + re.escape(name) + r"\*\*:.*(?:\n  -> (?!Read).*)*"
                  r"\n  -> Read `([^`]+)`", _system_text(messages))
    return m.group(1) if m else ""


def listed_names(messages) -> List[str]:
    return sorted(set(re.findall(r"(?m)^- \*\*([^*]+)\*\*:", _system_text(messages))))


def _model(steps):
    """The native-loader suite's scripted model, with a model name.

    LangChain reports a model's name from its identifying params, and ingest refuses an
    LLM call without one; the suite's stub has none (it never sends a trace).
    """
    from tests.native_loaders.stubs.deepagents_model import ScriptedModel

    class Named(ScriptedModel):
        @property
        def _identifying_params(self):
            return {"model_name": "scripted-deepagents"}

    return Named(steps=steps)


def _shell_backend(root: str):
    from deepagents.backends import LocalShellBackend

    return LocalShellBackend(root_dir=root, virtual_mode=False,
                             env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")})


class _Script:
    """The scripted model's turns, one list for both runs (one model object)."""

    def __init__(self) -> None:
        from tests.native_loaders.stubs.deepagents_model import final, tool_call

        self.final, self.tool_call = final, tool_call
        self.calls: List[Dict[str, Any]] = []     # every tool call made, in order
        self.listed: List[List[str]] = []          # the listing each turn was shown
        self.steps: List[Any] = []

    def call(self, name: str, args: dict):
        call_id = f"c{len(self.calls) + 1}"
        self.calls.append({"id": call_id, "name": name, "args": args})
        return self.tool_call(name, args, call_id)

    def read_skill(self, messages):
        self.listed.append(listed_names(messages))
        path = listed_path(messages)
        if not path:
            return self.final(f"{TARGET} is not in the skills listing")
        return self.call("read_file", {"file_path": path, "limit": 1000})

    def read_resource(self, messages):
        path = listed_path(messages)
        return self.call("read_file", {"file_path": f"{path.rsplit('/', 1)[0]}/{RESOURCE}"})

    def run_script(self, messages):
        folder = listed_path(messages).rsplit("/", 1)[0]
        cmd = " ".join((sys.executable, f"{folder}/{SCRIPT}", *SCRIPT_ARGS))
        return self.call("execute", {"command": cmd})

    def answer(self, messages):
        return self.final("done")


def _tool_results(out: Any, call_ids: List[str]) -> Dict[str, str]:
    from langchain_core.messages import ToolMessage

    wanted = set(call_ids)
    return {m.tool_call_id: str(m.content) for m in (out or {}).get("messages", [])
            if isinstance(m, ToolMessage) and m.tool_call_id in wanted}


def column(ctx: Dict[str, Any]) -> Dict[str, Any]:
    import decimalai.deepagents
    from deepagents import create_deep_agent
    from langgraph.checkpoint.memory import InMemorySaver


    s = Session("deepagents", distributions=DISTRIBUTIONS, langchain=True)
    script = _Script()
    # Run 1: list, load, load again, read the resource, run the script, answer.
    # Run 2 (after the update): load, answer.
    script.steps = [script.read_skill, script.read_skill, script.read_resource,
                    script.run_script, script.answer, script.read_skill, script.answer]
    model = _model(script.steps)
    project = str(col.project().root)
    agent = create_deep_agent(model=model, checkpointer=InMemorySaver(),
                              **decimalai.deepagents.skills(backend=_shell_backend(project)))
    # One conversation: the second run continues the first's thread, so Deep Agents keeps
    # the listing it parsed in run 1 (its skills middleware caches it in the thread state).
    config = {"configurable": {"thread_id": "native-column"}}
    received: Dict[str, str] = {}

    def one_run(run: Run) -> None:
        before = len(script.calls)
        out = agent.invoke({"messages": [{"role": "user", "content": USER}]}, config=config)
        made = script.calls[before:]
        results = _tool_results(out, [c["id"] for c in made])
        for c in made:
            text = results.get(c["id"])
            run.saw(text)
            received[c["id"]] = text
            if c["name"] == "read_file" and c["args"]["file_path"].endswith("/SKILL.md"):
                run.loaded_target()

    s.run(one_run)
    s.notes["v2"] = col.update_to_v2()
    s.run(one_run)
    s.notes["calls"] = script.calls
    s.notes["listed"] = script.listed
    # What read_file handed the model for the target, first lines: the format the hook
    # reconstructs the delivered SKILL.md from.
    s.notes["delivered_head"] = [
        "\n".join((received.get(c["id"]) or "").splitlines()[:3]) for c in script.calls
        if c["name"] == "read_file" and c["args"]["file_path"].endswith("/SKILL.md")]
    return s.result()


def checks(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Five runs with skills in the workspace: a Deep Agents agent built with the pointer
    helper alone (its default backend), a plain LangGraph agent, the Deep Agents agent again
    with the target pinned, and twice a Deep Agents agent that also has ``agent_config()``'s
    middleware. Each makes one model call; ``warned`` is what a run logged at WARNING."""
    import logging

    import decimalai.deepagents
    from deepagents import create_deep_agent
    from decimalai.langchain import agent_config
    from langchain_core.messages import HumanMessage
    from langgraph.graph import END, START, MessagesState, StateGraph

    from tests.conformance.isolation import CONFORMANCE_SKILLS
    from tests.native_loaders.stubs.deepagents_model import final

    logs: List[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            logs.append(f"{record.levelname} {record.name}: {record.getMessage()}")

    logging.getLogger("decimalai").addHandler(_Capture(level=logging.INFO))
    logging.getLogger("decimalai").setLevel(logging.INFO)

    s = Session("deepagents", distributions=DISTRIBUTIONS, langchain=True)
    s.probe.skills = [dict(sk) for sk in CONFORMANCE_SKILLS]
    out: Dict[str, Any] = {}

    def injected(messages) -> List[str]:
        return [str(m.content)[:200] for m in messages
                if (getattr(m, "additional_kwargs", None) or {}).get("decimalai_injected")]

    def logged_run(fn) -> List[str]:
        before = len(logs)
        s.run(fn)
        return [line for line in logs[before:] if line.startswith("WARNING")]

    # 1. Deep Agents, through the pointer helper with its default backend.
    da_model = _model([lambda m: final("done")])
    agent = create_deep_agent(model=da_model, **decimalai.deepagents.skills())
    warned = logged_run(lambda run: agent.invoke({"messages": [{"role": "user", "content": USER}]}))
    seen = da_model.seen_requests[0] if da_model.seen_requests else []
    out["deepagents"] = {"injected": injected(seen), "listed": listed_names(seen),
                         "backend": type(decimalai.deepagents.skills()["backend"]).__name__,
                         "warned": warned}

    # 2. The control: a plain LangGraph agent in the same process. Nothing reaches its model
    # either, and it is the run the adapter tells so: its skills come from nowhere.
    plain_model = _model([lambda m: final("done")])

    def call_model(state):
        return {"messages": [plain_model.invoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("model_turn", call_model)
    builder.add_edge(START, "model_turn")
    builder.add_edge("model_turn", END)
    graph = builder.compile()
    warned = logged_run(lambda run: graph.invoke({"messages": [HumanMessage(content=USER)]}))
    seen = plain_model.seen_requests[0] if plain_model.seen_requests else []
    out["plain"] = {"injected": injected(seen), "warned": warned}

    # 3. A pinned skill: the middleware appends its body to the conversation itself,
    # without the model asking. The body arrives with the frontmatter cut off.
    pin_model = _model([lambda m: final("done")])
    pinned = create_deep_agent(model=pin_model, **decimalai.deepagents.skills())
    s.run(lambda run: pinned.invoke({"messages": [{"role": "user", "content": USER}],
                                     "pinned_skills": [TARGET]}))
    seen = pin_model.seen_requests[0] if pin_model.seen_requests else []
    out["pinned"] = {"messages": [str(m.content)[:120] for m in seen
                                  if (getattr(m, "additional_kwargs", None) or {}).get(
                                      "lc_source") == "pinned_skill"]}

    # 4. Deep Agents' skills and agent_config()'s middleware in one agent, two runs. Deep
    # Agents puts a user's middleware before its skills middleware, so ours runs first.
    s.probe.register_agent("both-lists", system_prompt="You answer refund questions.",
                           skills=[dict(CONFORMANCE_SKILLS[0])])
    support = agent_config("both-lists")
    both_model = _model([lambda m: final("done"), lambda m: final("done")])
    both = create_deep_agent(model=both_model, middleware=[support.middleware],
                             **decimalai.deepagents.skills())
    warned = [logged_run(lambda run: both.invoke({"messages": [{"role": "user", "content": USER}]}))
              for _ in range(2)]
    seen = both_model.seen_requests[0] if both_model.seen_requests else []
    # Deep Agents' skills middleware rebuilds the system message our marker was on, so the
    # agent's prompt is looked for as text.
    out["both"] = {"warned": warned, "listed": listed_names(seen),
                   "prompt_shown": "You answer refund questions." in _system_text(seen)}

    result = s.result()
    out["traces"] = [r["traces"] for r in result["runs"]]
    out["errors"] = [r["error"] for r in result["runs"]]
    out["logs"] = [line for line in logs if "decimalai" in line][:40]
    return out


# ── the live check (tests/native/test_native_live.py) ────────────────────────


def _text_of(message: Any) -> str:
    text = getattr(message, "text", "")
    return text if isinstance(text, str) else text()


def live(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Each trial: a fresh Deep Agents agent built as the column builds it (the pointer
    helper over a shell backend) with a REAL OpenAI model (``ChatOpenAI``), asked the live
    question; every tool call it made, what it was given and its answer (see
    ``tests/native/live.py``)."""
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_openai import ChatOpenAI

    import decimalai.deepagents
    from deepagents import create_deep_agent
    from tests.native import live as lv

    s = Session("deepagents", distributions=DISTRIBUTIONS + ["langchain-openai"],
                langchain=True)
    project = str(col.project().root)

    class FirstPrompt(BaseCallbackHandler):
        """The system prompt of the first model call: the skills listing the model got."""

        def __init__(self, run: Run) -> None:
            self.run, self.done = run, False

        def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
            if not self.done and messages:
                self.done = True
                self.run.saw(_system_text(messages[0]))

    def trial(run: Run) -> None:
        agent = create_deep_agent(model=ChatOpenAI(model=lv.openai_model()),
                                  system_prompt=lv.INSTRUCTIONS,
                                  **decimalai.deepagents.skills(backend=_shell_backend(project)))
        state: Dict[str, Any] = {}
        try:
            # Streamed state, so a run that stops early still reports what it did.
            for state in agent.stream({"messages": [{"role": "user", "content": lv.QUESTION}]},
                                      config={"callbacks": [FirstPrompt(run)],
                                              "recursion_limit": 40},
                                      stream_mode="values"):
                pass
        finally:
            messages = (state or {}).get("messages", [])
            for m in messages:
                if isinstance(m, AIMessage):
                    for call in m.tool_calls or []:
                        run.called(call.get("name"), call.get("args"))
                elif isinstance(m, ToolMessage):
                    run.saw(str(m.content))
            final = [m for m in messages if isinstance(m, AIMessage) and not m.tool_calls]
            if final:
                run.answered(_text_of(final[-1]))

    for _ in range(lv.trials(ctx)):
        s.run(trial)
    s.notes["model"] = lv.openai_model()
    return s.result()
