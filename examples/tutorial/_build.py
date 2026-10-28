#!/usr/bin/env python3
"""Build the tutorial notebooks: order_support_<sdk>.ipynb, one per runnable SDK tab.

WHAT THEY ARE FOR. The docs tutorial (docs.decimal.ai, "Tutorial: a support agent") runs
`order-support` on its own page, then has the reader run it in their own code (Part 2). These
notebooks are Part 2 with nothing to install: the docs link them as "Open in Colab" beside each
SDK's code. The program is the one the docs show for that SDK, unchanged except where a notebook
forces it (below).

TWO WAYS TO RUN, CHOSEN BY ONE SECRET.
- With DECIMAL_API_KEY (the reader's "Docs tutorial" key from signing in on Part 1, or any key of
  theirs): agent_config() reads THEIR saved prompt and skills from DecimalAI, and the run is traced
  to their dashboard. The keys cell first checks the key reaches an `order-support`: a key that is
  wrong, or a workspace with no copy yet, is said plainly and the notebook falls back to keyless
  rather than let agent_config() create an empty agent.
- Without it: keyless, as Part 2's first half. The demo's prompt is the fallback= text, and the
  setup cell adds the three registry skills to a project folder with `decimalai skills add` (no
  account), which is where keyless agent_config() reads them. Nothing goes to DecimalAI.

WHERE THE NOTEBOOK DIFFERS FROM main.py, AND WHY.
- OpenAI Agents SDK: `await Runner.run(...)`, not Runner.run_sync. A notebook already runs an event
  loop, and run_sync starts its own, which raises there.
- OpenAI SDK and Anthropic API: the loop is a function, `answer(question)`, called in a worker
  thread (asyncio.to_thread). decimalai's sync `with support.run(...)` must not be opened on a
  thread that is running an event loop, and a notebook cell is one.

KEYS. Colab Secrets first (they never enter the notebook file), then the environment, then a
hidden prompt where something can answer it. A provider key is required to get a reply;
DECIMAL_API_KEY is optional. Nothing here prints a key.

WHAT IT MUST NEVER DO. Raise. No key, a wrong key, no network, a provider error: each is a printed
sentence, never a traceback.

    python examples/tutorial/_build.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "decimal-labs/decimalai-python"
DOCS = "https://docs.decimal.ai/tutorials/support-agent"
SKILLS = ["billing-dispute-resolution", "account-deletion-verification-policy", "diagnostic-triage-questions"]
QUESTION = "I was charged twice for order ORD-1002. Can you fix it?"

ORDERS = '''ORDERS = {
    "ORD-1001": {"item": "electric kettle", "status": "delivered 2026-10-20", "charges": ["$49.00 on 2026-10-15"]},
    "ORD-1002": {"item": "mechanical keyboard", "status": "delivered 2026-10-18", "charges": ["$129.00 on 2026-10-12", "$129.00 on 2026-10-12"]},
}'''

CONFIG = '''support = agent_config(
    "order-support",
    fallback=(
        "You are Acme's order support agent. Look the order up before you answer. You cannot issue "
        "refunds or change orders yourself: say what you found and that the billing team will follow up."
    ),
)'''

SDKS = {
    "openai_agents": {
        "label": "OpenAI Agents SDK",
        "extra": "openai-agents",
        "provider": "OPENAI_API_KEY",
        "provider_name": "OpenAI",
        "provider_url": "https://platform.openai.com/api-keys",
        "program": f'''from agents import Agent, Runner, function_tool
from decimalai.openai_agents import agent_config

{ORDERS}


@function_tool
def lookup_order(order_id: str) -> str:
    """Look up an order by its id, for example ORD-1002."""
    return str(ORDERS.get(order_id, "No order with that id."))


{CONFIG}

agent = Agent(
    name=support.name,
    instructions=support.instructions,     # the prompt, then the skill list
    tools=[lookup_order, *support.tools],  # your tools, then load_skill
)''',
        "reply": '''    # main.py says Runner.run_sync(agent, question); a notebook is already running an event
    # loop, so here the run is awaited.
    result = await Runner.run(agent, question)
    return str(result.final_output)''',
    },
    "openai": {
        "label": "OpenAI SDK",
        "extra": "openai",
        "provider": "OPENAI_API_KEY",
        "provider_name": "OpenAI",
        "provider_url": "https://platform.openai.com/api-keys",
        "program": f'''import json

from openai import OpenAI
from decimalai.openai import agent_config

{ORDERS}


def lookup_order(order_id: str) -> str:
    return str(ORDERS.get(order_id, "No order with that id."))


LOOKUP_ORDER = {{
    "type": "function",
    "name": "lookup_order",
    "description": "Look up an order by its id, for example ORD-1002.",
    "parameters": {{
        "type": "object",
        "properties": {{"order_id": {{"type": "string"}}}},
        "required": ["order_id"],
    }},
}}

client = OpenAI()
{CONFIG}


def answer(question):
    # main.py's loop, as a function: the next cell runs it in a worker thread.
    with support.run(question) as run:
        items = [run.message]
        while True:
            response = client.responses.create(
                model="gpt-5.4-mini",
                instructions=run.system,                    # the prompt, then the skill list
                input=items,
                tools=[LOOKUP_ORDER, *run.response_tools],  # your tool, then load_skill
            )
            items += response.output
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                break
            for call in calls:
                items.append(run.tool_result(call) or {{   # load_skill answers itself
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": lookup_order(**json.loads(call.arguments)),
                }})
    return response.output_text''',
        "reply": '''    return await asyncio.to_thread(answer, question)''',
    },
    "anthropic": {
        "label": "Anthropic API",
        "extra": "anthropic",
        "provider": "ANTHROPIC_API_KEY",
        "provider_name": "Anthropic",
        "provider_url": "https://console.anthropic.com/settings/keys",
        "program": f'''import anthropic
from anthropic import beta_tool
from decimalai.anthropic import agent_config

{ORDERS}


@beta_tool
def lookup_order(order_id: str) -> str:
    """Look up an order by its id, for example ORD-1002."""
    return str(ORDERS.get(order_id, "No order with that id."))


client = anthropic.Anthropic()
{CONFIG}


def answer(question):
    # main.py's loop, as a function: the next cell runs it in a worker thread.
    with support.run(question) as run:
        runner = client.beta.messages.tool_runner(
            model="claude-haiku-5-5",
            max_tokens=4096,
            system=run.system,                 # the prompt, then the skill list
            tools=[lookup_order, *run.tools],  # your tool, then load_skill
            messages=[{{"role": "user", "content": question}}],
        )
        reply = runner.until_done()
    return "".join(block.text for block in reply.content if block.type == "text")''',
        "reply": '''    return await asyncio.to_thread(answer, question)''',
    },
}


def _s(source):
    lines = source.split("\n")
    return [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _s(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": _s(text)}


META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.0"},
    "colab": {"provenance": []},
}


def notebook(key, sdk):
    path = f"examples/tutorial/order_support_{key}.ipynb"
    colab = f"https://colab.research.google.com/github/{REPO}/blob/main/{path}"
    provider = sdk["provider"]
    keys = f'''# ── your keys ───────────────────────────────────────────────────────────
import getpass, json, os, sys, urllib.error, urllib.request

def secret(name):
    """A Colab secret, else the environment. Colab raises both for a missing secret and for one
    this notebook hasn't been given access to; either way the answer is "not set"."""
    try:
        from google.colab import userdata  # ImportError anywhere but Colab
        value = (userdata.get(name) or "").strip()
        if value:
            return value, f"Colab secret {{name}}"
    except Exception:
        pass
    value = os.environ.get(name, "").strip()
    return (value, f"environment variable {{name}}") if value else (None, None)

# Only ask where something can answer: Colab and Jupyter route getpass through the kernel.
CAN_PROMPT = sys.stdin.isatty() or "ipykernel" in sys.modules

PROVIDER_KEY, where = secret("{provider}")
if not PROVIDER_KEY and CAN_PROMPT:
    try:  # getpass, never input(): input() would echo the key into this cell's saved output
        PROVIDER_KEY = getpass.getpass("Paste your {sdk["provider_name"]} key (hidden), or press Enter to skip: ").strip() or None
        where = "pasted (memory only)"
    except Exception:
        PROVIDER_KEY = None
if PROVIDER_KEY:
    os.environ["{provider}"] = PROVIDER_KEY
    print(f"{provider}: from {{where}}")
else:
    print("{provider}: not set. Everything below still runs; the reply cell will say it skipped.")

DECIMAL_API_KEY, where = secret("DECIMAL_API_KEY")
os.environ.pop("DECIMAL_API_KEY", None)
if DECIMAL_API_KEY:
    # Does this key reach an order-support? Without one, agent_config() would create an empty agent.
    base = os.environ.get("DECIMAL_BASE_URL", "https://api.decimal.ai").rstrip("/")
    request = urllib.request.Request(f"{{base}}/api/v1/agents/order-support/prompt",
                                     headers={{"Authorization": f"Bearer {{DECIMAL_API_KEY}}"}})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            saved = json.load(response)
        os.environ["DECIMAL_API_KEY"] = DECIMAL_API_KEY
        print(f"DECIMAL_API_KEY: from {{where}}: your order-support, prompt v{{saved.get('version_number')}}")
    except urllib.error.HTTPError as error:
        reason = ("that key isn't valid (revoked or expired?)" if error.code in (401, 403) else
                  "your workspace has no order-support yet: sign in on Part 1 of the tutorial to make your copy"
                  if error.code == 404 else f"DecimalAI answered {{error.code}}")
        print(f"DECIMAL_API_KEY: {{reason}}. Running keyless instead.")
    except Exception:
        print("DECIMAL_API_KEY: DecimalAI couldn't be reached from here. Running keyless instead.")
else:
    print("DECIMAL_API_KEY: not set. Running keyless: the demo's prompt and skills, nothing sent to DecimalAI.")
KEYED = "DECIMAL_API_KEY" in os.environ'''

    setup = f'''# ── the project folder ──────────────────────────────────────────────────
import os, subprocess

# One folder, however often this cell runs. Keyless agent_config() reads its skills from here.
PROJECT = globals().get("PROJECT") or os.path.abspath("order-support")
os.makedirs(PROJECT, exist_ok=True)
os.chdir(PROJECT)

if KEYED:
    print("Keyed: your skills live on DecimalAI, so there is nothing to add here.")
else:
    for name in {SKILLS!r}:
        done = subprocess.run(["decimalai", "skills", "add", name], capture_output=True, text=True)
        print(("added " if done.returncode == 0 else "could not add ") + name
              + ("" if done.returncode == 0 else ": " + (done.stderr or done.stdout).strip()[-300:]))'''

    ask = f'''# ── ask it ──────────────────────────────────────────────────────────────
import asyncio

async def ask(question):
    """Print the agent's reply, and where its prompt came from. Never raises."""
    if not os.environ.get("{provider}"):
        print("Skipped: no {provider}. Add it as a Colab secret (or paste it above) and run the keys cell again.")
        return
    try:
        print(await reply(question))
    except Exception as error:
        print(f"The run didn't finish: {{type(error).__name__}}: {{str(error)[:300]}}")
        return
    source = getattr(support, "prompt_source", None) or ("decimalai" if KEYED else "fallback")
    version = getattr(support, "prompt_version", None)
    print("\\n— prompt: " + ("your saved prompt from DecimalAI" + (f", v{{version}}" if version else "")
                            if source == "decimalai" else "the fallback= text (keyless)"))


async def reply(question):
{sdk["reply"]}'''

    cells = [
        md(f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab})\n"
           "\n"
           f"# order-support, in your own code: {sdk['label']}\n"
           "\n"
           f"This is the DecimalAI tutorial's support agent, the one you ran on [the docs page]({DOCS}/try-it), "
           f"running here in your own code with nothing to install. It is [Part 2]({DOCS}/build-it) of the "
           f"tutorial as a notebook: the same program the docs show for the {sdk['label']}.\n"
           "\n"
           "**It runs one of two ways, depending on one secret:**\n"
           "\n"
           "- **With your `DECIMAL_API_KEY`**: it runs *your* copy of `order-support`, with the prompt and "
           "skills you saved. The run shows up in your DecimalAI dashboard.\n"
           "- **Without it**: keyless. The demo's prompt and its three skills from the registry; nothing is "
           "sent to DecimalAI."),
        md("## 1. Your keys\n"
           "\n"
           "Add them as Colab **Secrets** (the key icon in the left sidebar) and turn on notebook access for "
           "each. Secrets never enter the notebook file.\n"
           "\n"
           f"- **`{provider}`** (required for a reply): your own [{sdk['provider_name']} key]({sdk['provider_url']}). "
           "These model calls are on your account.\n"
           "- **`DECIMAL_API_KEY`** (optional): the *Docs tutorial* key from signing in on Part 1 (Part 2's "
           "`.env` box shows it, in the same browser tab), or any key from **Settings → API keys** in the "
           "DecimalAI dashboard.\n"
           "\n"
           "Not in Colab? The cell reads the same names from the environment, or asks for the provider key."),
        code(keys),
        md("## 2. Install"),
        code(f'!pip install -q "decimalai[{sdk["extra"]}]>=0.26"'),
        md("## 3. The project folder\n"
           "\n"
           "Keyless, `agent_config()` reads the agent's skills from the project's files, so this adds the three "
           "from the registry, as Part 2's `decimalai skills add` step does. No account is needed. With your "
           "key, your skills live on DecimalAI instead and there is nothing to add."),
        code(setup),
        md("## 4. The agent\n"
           "\n"
           f"The program from the docs' **{sdk['label']}** tab. `agent_config()` gives the model the prompt and "
           "the skill menu, and adds one tool, `load_skill`. `lookup_order` is your own tool: here it reads a "
           "dictionary; in your app, your database."),
        code(sdk["program"]),
        md("## 5. Ask it"),
        code(ask),
        code(f"await ask({QUESTION!r})"),
        md("Ask another question: each one needs a different skill, and the model picks it.\n"
           "\n"
           "- *Please delete my account and all my data today.*\n"
           "- *My kettle from order ORD-1001 won't turn on.*"),
        code('await ask("Please delete my account and all my data today.")'),
        md("## What just happened\n"
           "\n"
           "- **With your key**, the reply followed *your* saved prompt and *your* copy of a skill, read from "
           "DecimalAI. Change either one in the dashboard (or on Part 1's page) and run **Ask it** again: "
           "`agent_config()` re-reads them at most once a minute, with nothing to reinstall or redeploy. Each "
           "run is traced at [app.decimal.ai/agents/order-support](https://app.decimal.ai/agents/order-support).\n"
           "- **Keyless**, the prompt was the `fallback=` text and the skills came from this folder. To connect "
           "it, add `DECIMAL_API_KEY` and run the notebook again from the top (**Runtime → Restart session**, "
           "then **Run all**).\n"
           "\n"
           f"Next: [3. Start from a template]({DOCS}/start-from-a-template)."),
    ]
    for i, cell in enumerate(cells):
        cell["id"] = f"{key}-{i:02d}"  # nbformat 4.5 wants one; stable, so a rebuild diffs cleanly
    return path, {"cells": cells, "metadata": META, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(HERE))
    for key, sdk in SDKS.items():
        path, nb = notebook(key, sdk)
        with open(os.path.join(root, path), "w") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print("wrote", path)
