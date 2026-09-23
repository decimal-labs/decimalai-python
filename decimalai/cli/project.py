"""Generate an independently runnable Support draft project. No skill code executes."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from decimalai import __version__
from decimalai.agent_checks import (
    DEFAULT_OPENAI_JUDGE,
    SUPPORT_SKILLS,
    check_definition,
    judge_model,
)

from .scaffold import env_vars, install_command, normalize_model, provider_key


def render_project(agent_name: str, skills: list[dict[str, Any]], model: str | None,
                   base_url: str, agent_id: str | None = None) -> dict[str, str]:
    model = normalize_model("langchain", model or "gpt-5.4-2026-03-05")
    required_key = provider_key("langchain", model)
    bindings = {}
    for slug in SUPPORT_SKILLS:
        matches = list({s["skill_name"] for s in skills
                        if s.get("source_skill_slug", s["skill_name"]) == slug})
        if len(matches) != 1:
            raise ValueError(f"Support checks need exactly one selected {slug} skill; found {len(matches)}.")
        bindings[slug] = matches[0]
    definition = check_definition(bindings)
    source = f'''"""Support draft agent. Run directly or import run(question) in your service."""
from __future__ import annotations

import os
import sys
from typing import Callable

from dotenv import load_dotenv

load_dotenv()
import decimalai
from decimalai._support_policy import draft_reply_violation
from decimalai.langchain import CallbackHandler, instrument
from decimalai.schema.trace import RunTrace
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

AGENT_NAME = {agent_name!r}
MODEL = {model!r}
BASE_URL = os.environ.get("DECIMAL_BASE_URL", {base_url!r})
MAX_TURNS = 5
MAX_DRAFT_ATTEMPTS = 3
TOOLS: list = []
trace_observer: Callable[[RunTrace], None] | None = None
trace_session_id: str | None = None
PROVIDER_KEY = {required_key!r}
if PROVIDER_KEY and not os.environ.get(PROVIDER_KEY):
    raise ValueError(f"Missing {{PROVIDER_KEY}}. Fill in .env before running the agent.")


def observe(trace: RunTrace) -> None:
    if trace_observer is not None:
        trace_observer(trace)


# The check and your service use this same model, prompt, skill loader and run().
decimalai.init(base_url=BASE_URL)
config = decimalai.load_agent(AGENT_NAME)
CAPABILITY_CONTEXT = (
    "Runtime capabilities: You may perform actions only through the tools actually provided. "
    "Skill instructions describe workflows; they do not grant access to tools. Never claim an "
    "action happened unless a tool result confirms it. "
    + ("Use only the provided tools." if TOOLS else
       "No tools are configured. You draft replies only. You cannot initiate recovery, send mail, "
       "open cases, refund, delete, lock accounts or change billing. Describe what the customer "
       "or a human support operator can do next; never offer or promise to do those actions yourself. "
       "Any authority to route, escalate or apply policy in the dashboard prompt or skills is "
       "conditional on having the necessary tools; this installation has none. Do not promise "
       "that you will have another team review, contact or act for the customer. Give the customer "
       "a next step they can take instead. Describe contacting human support as a request for "
       "review, never a promise that we will review it or act right away.")
    + " Begin the reply by acknowledging the customer's concrete issue. Do not open with "
      "generic thanks for reaching out, contacting support, or sending a message."
      " Use only business facts supplied in the ticket. Skill examples illustrate a workflow; "
      "their actions and promises are not facts about this ticket. Do not invent support channels, "
      "the availability of human review, policies, guarantees or response times."
      " Briefly state the applicable policy path, including its prerequisite and any later "
      "confirmation step, before giving one immediate next step supported by the supplied facts. "
      "Do not omit the later policy step just because a prerequisite must happen first. If a location or procedure is "
      "not supplied, name the flow without inventing where to find it. Do not ask for identity "
      "or recovery details in this chat. End after the next step and a brief neutral signoff; "
      "do not add alternative workflows or an offer to act on the customer's behalf."
      " Preserve supplied numerical ranges and explicitly distinguish estimates from guarantees. "
      "If a requested action is unavailable, say so; do not infer that verification or completing "
      "another process will enable it unless the supplied facts explicitly establish that link."
)
SYSTEM_PROMPT = "\\n\\n".join(part for part in (config.system_prompt, CAPABILITY_CONTEXT) if part)
# Skills may assume tools this application lacks. Repeat the actual constraints
# after their guidance, while retaining the full skill bodies for trace evidence.
instrument(agent_name=AGENT_NAME, enable_skill_loader=True, skill_body_top_k=3,
           priority_skills={[bindings[SUPPORT_SKILLS[1]]]!r}, on_trace=observe,
           runtime_policy=CAPABILITY_CONTEXT)
# Keep generation settings explicit for the tested default and GPT-4.1 overrides.
model_options = {{"temperature": 0}} if MODEL.startswith(("gpt-4.1", "openai:gpt-4.1")) else {{}}
if MODEL in ("gpt-5.4-2026-03-05", "openai:gpt-5.4-2026-03-05"):
    model_options = {{"reasoning_effort": "low"}}


@wrap_model_call
def check_draft(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    # Runs in the SAME graph in checks and service calls. Every attempt is traced.
    # This conservative guard covers known English patterns, not arbitrary claims.
    if TOOLS:
        return handler(request)  # Requalify after adding actual action tools.
    for attempt in range(MAX_DRAFT_ATTEMPTS):
        response = handler(request)
        reply = response.result[-1]
        violation = ("Do not call tools: this runtime has none." if getattr(reply, "tool_calls", None)
                     else draft_reply_violation(reply.text))
        if violation is None:
            return response
        if attempt + 1 == MAX_DRAFT_ATTEMPTS:
            raise ValueError("Draft validation failed after bounded revisions; no reply was returned.")
        # Do not repeat an invalid tool call without its required tool response.
        feedback = HumanMessage(content="Revise the previous draft. " + violation +
                                " Preserve the supplied facts and return only the revised customer reply.")
        previous = [] if getattr(reply, "tool_calls", None) else [reply]
        request = request.override(messages=[*request.messages, *previous, feedback])
    raise RuntimeError("Draft validation ended without a result.")


agent = create_agent(init_chat_model(MODEL, **model_options), tools=TOOLS,
                     system_prompt=SYSTEM_PROMPT, middleware=[check_draft])


def run(question: str) -> str:
    invocation = {{"recursion_limit": 2 * MAX_TURNS + 1}}
    if trace_session_id:
        handler = CallbackHandler(agent_name=AGENT_NAME, session_id=trace_session_id)
        handler.on_trace = observe
        invocation["callbacks"] = [handler]
    state = agent.invoke(
        {{"messages": [HumanMessage(content=question)]}},
        config=invocation,
    )
    answer = state["messages"][-1]
    # AIMessage.text normalizes both string and provider content-block responses.
    return answer.text


if __name__ == "__main__":
    try:
        question = " ".join(sys.argv[1:]) or "What can you help me with?"
        print(run(question))
    finally:
        decimalai.flush()
'''
    dependencies = install_command("langchain", model).removeprefix("pip install ").split()
    dependencies[0] = f"decimalai[langchain,evals]=={__version__}"
    requirements = "\n".join(d.strip('"') for d in dependencies) + "\n"
    keys = env_vars("langchain", model)
    env = "# Fill these in locally; never commit .env.\n" + "\n".join(f"{k}=" for k in keys)
    env += f"\nDECIMAL_BASE_URL={base_url}\n# Optional LiteLLM model identifier for the semantic judge.\n# CHECK_MODEL={judge_model(model)}\n"
    readme = f'''# {agent_name}

A runnable Support reply agent with your selected skills and DecimalAI tracing.
It drafts text; TOOLS is empty. It cannot refund, delete accounts, send mail, or
change billing. Skill instructions do not grant those capabilities.
The synchronous run() checks known English action-claim and generic-opening patterns
before returning a draft. It allows at most two revisions, then raises ValueError;
all attempts stay in the same trace. This is a narrow guard, not comprehensive
fact verification. Review replies before sending them to customers. If you add
TOOLS, the draft-only guard is bypassed: add tool-specific checks and requalify.

## Run and check

Python 3.10+:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# Fill in your DecimalAI and model-provider keys in .env.
python agent.py "Draft a reply to this support ticket: ..."
python check_agent.py
```

The generated requirements target SDK {__version__}. For an unpublished candidate,
install its supplied wheel in this environment first; this version is not a claim
that the candidate is available from PyPI.

The check runs the SAME agent.py twice using explicitly labeled sample business
facts: an authorization hold and an unavailable account-deletion action. It
verifies delivered skill bodies and grades the answers against a versioned,
narrow rubric with a model judge. OpenAI projects use the calibrated
{DEFAULT_OPENAI_JUDGE} judge; other providers default to the agent's model.
CHECK_MODEL can select another LiteLLM model. A changed judge needs calibration;
the included evidence qualifies the OpenAI default only. Provider errors fail the check.
These checks make model calls and send the two sample traces to your workspace.

Exit 0 means both cases, skill delivery, configuration stability, and the exact
persisted traces passed. Each attempt has its own dashboard link. Trace confirmation
waits at most 30 seconds; if delayed, run `python check_agent.py --resume` to retry
confirmation without more model calls. Use a normal run to repeat failed checks.
Every attempt keeps its own check-results.<check-id>.json receipt. To resume a
specific concurrent attempt, use `--resume --check-id <id>`. check-results.json is
a convenience copy of the last local result, including failures. Read the answers,
criteria and run IDs there. These two examples are a starter check, not general
production certification or proof of skill lift. The synthetic connectivity canary
(`decimalai init --test-trace`) is separate.

## Versions and deployment

The dashboard controls the system prompt and skill subscriptions. Latest prompt
is resolved when this process starts; latest skill bodies are resolved at runtime.
Pinned subscriptions resolve their pinned version. The receipt records resolved
versions/hashes, delivered bodies, model names, installed packages and local file
hashes. A prior pass is stale after ANY prompt, skill, tool, model, dependency or
code change. Restart and rerun the check after changes, including latest updates.
Pin prompt/skill versions in the dashboard and lock dependencies for repeatability.

For your deployment, import run(question) into your own worker or service, provide
the same environment variables, and call decimalai.flush() during shutdown. Keep
provider keys in your deployment's secret store. Add only tools you actually
implement to TOOLS, retain a finite turn limit, add tests for those actions, and
rerun checks. Service hosting, authentication, concurrency limits and deployment
configuration belong to your own application. No deployment runs from this project.
'''
    return {
        "agent.py": source,
        "check_agent.py": 'import argparse\nfrom pathlib import Path\nfrom decimalai.agent_checks import run_project_checks\n\nif __name__ == "__main__":\n    parser = argparse.ArgumentParser()\n    parser.add_argument("--resume", action="store_true", help="Retry trace confirmation without running models")\n    parser.add_argument("--check-id", help="Resume a specific concurrent check")\n    args = parser.parse_args()\n    if args.check_id and not args.resume: parser.error("--check-id requires --resume")\n    raise SystemExit(run_project_checks(Path(__file__).resolve().parent, resume=args.resume, check_id=args.check_id))\n',
        "project.json": json.dumps({"agent_name": agent_name, "agent_id": agent_id, "base_url": base_url}, indent=2) + "\n",
        "checks.json": json.dumps(definition, indent=2) + "\n",
        "requirements.txt": requirements,
        ".env.example": env,
        ".gitignore": ".env\n.venv/\n__pycache__/\ncheck-results*.json\n",
        "README.md": readme,
    }


def write_project(destination: Path, files: dict[str, str]) -> None:
    """Publish a complete new directory, refusing existing files and symlinks."""
    if os.path.lexists(destination):
        raise ValueError(f"{destination} already exists; choose a new project directory.")
    if not destination.parent.is_dir():
        raise ValueError(f"Parent directory does not exist: {destination.parent}")
    staging = Path(tempfile.mkdtemp(prefix=".decimal-project-", dir=destination.parent))
    try:
        for name, content in files.items():
            (staging / name).write_text(content, encoding="utf-8")
        # mkdir is the exclusive claim: rename alone could replace an empty directory.
        destination.mkdir()
        try:
            os.replace(staging, destination)
        except Exception:
            destination.rmdir()
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
