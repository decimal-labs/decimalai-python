"""A real-model check of `decimalai.google_genai.agent_config()` that needs no DecimalAI key.

Run where a working Gemini key is (Machine B, while the key on Machine A is out of credit):

    GEMINI_API_KEY=... python scripts/live_check_gemini_keyless.py [--model gemini-3.5-flash]

It sends nothing to DecimalAI: with no `DECIMAL_API_KEY` the binding runs keyless, so its prompt is
the `fallback=` text and its skills are the ones the project lists. The script makes a temporary
project whose one skill, written here, carries a token made for this run, and asks a question that
only that skill answers, three ways:

  control      the same model, instruction and question, without agent_config(): it must NOT
               answer with the token (else the token reached the model another way)
  chat         a chat with automatic function calling, tools=[*run.tools] (the docs' short form)
  hand-loop    generate_content with tools=[*run.tool_params], each call answered with
               run.tool_result(call) (the shape `decimalai init --framework google-genai` writes)

One line per form (`ok`, or what failed and the start of that answer), then `RESULT PASS` or
`RESULT FAIL`; exit 0 on PASS, 1 on FAIL, 2 when it could not run. Never prints a key, nor an answer
that passed. `--base-url` points the client at a stub (the script's own test).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import uuid
from pathlib import Path

SKILL = "rma-keyless-check"
PROMPT = "You are a customer support agent. Answer using the procedures available to you."
QUESTION = ("A customer's phone arrived with a cracked screen. What return authorization code should I "
            "give them? Reply with the code and nothing else.")


def _project(root: Path, token: str) -> None:
    """A project whose one skill, written here, holds the token: listed under [authored]."""
    skill = root / ".agents" / "skills" / SKILL
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {SKILL}\ndescription: Internal return procedure for phones that arrive with a "
        f"cracked screen, including the return authorization code to give the customer.\n---\n\n"
        f"# Cracked-screen returns\n\nThe return authorization code is `RMA-{token}`. Quote it exactly, "
        f"including the prefix.\n", encoding="utf-8")
    (root / ".decimal").mkdir()
    (root / ".decimal" / "skills.toml").write_text(f'[authored]\nskills = ["{SKILL}"]\n', encoding="utf-8")


def _say(form: str, ok: bool, failure: str, answer: str) -> None:
    """One line per form: "ok", or the failure word and the start of the answer (a synthetic
    answer to the synthetic question above; shown only when the form failed)."""
    print(f"{form:<10} ok" if ok else f"{form:<10} {failure}: {answer.strip()[:80]!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", default=os.environ.get("LIVE_LLM_GEMINI_MODEL", "gemini-3.5-flash"))
    parser.add_argument("--base-url", default=None, help="a stub's URL, for this script's own test")
    args = parser.parse_args()

    for name in ("DECIMAL_API_KEY", "DECIMALAI_API_KEY"):
        os.environ.pop(name, None)  # keyless: nothing is sent to DecimalAI
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("RESULT UNAVAILABLE: set GEMINI_API_KEY (or GOOGLE_API_KEY)")
        return 2
    try:
        from google import genai
        from google.genai import types

        from decimalai.google_genai import agent_config
    except ImportError as exc:
        print(f"RESULT UNAVAILABLE: {exc}")
        return 2

    token = uuid.uuid4().hex[:10].upper()
    want = f"RMA-{token}"
    root = Path(tempfile.mkdtemp(prefix="gemini-keyless-check-"))
    _project(root, token)
    os.chdir(root)  # a keyless run reads the project it runs in

    options = types.HttpOptions(base_url=args.base_url) if args.base_url else None
    client = genai.Client(http_options=options)
    results = {}

    control = client.models.generate_content(
        model=args.model, contents=QUESTION,
        config=types.GenerateContentConfig(system_instruction=PROMPT)).text or ""
    results["control"] = want not in control.upper()
    _say("control", results["control"], "LEAK", control)

    support = agent_config("gemini-keyless-check", fallback=PROMPT)
    with support.run(QUESTION) as run:
        chat = client.chats.create(model=args.model, config=types.GenerateContentConfig(
            system_instruction=run.system, tools=[*run.tools]))
        answer = chat.send_message(run.message).text or ""
    results["chat"] = want in answer.upper()
    _say("chat", results["chat"], "MISSING", answer)

    with support.run(QUESTION) as run:
        contents = [{"role": "user", "parts": [{"text": run.message}]}]
        config = types.GenerateContentConfig(system_instruction=run.system, tools=[*run.tool_params])
        answer = ""
        for _ in range(4):
            response = client.models.generate_content(model=args.model, contents=contents, config=config)
            if not response.function_calls:
                answer = response.text or ""
                break
            contents.append(response.candidates[0].content)
            contents.append({"role": "user", "parts": [
                run.tool_result(call) or {"function_response": {
                    "name": call.name, "id": call.id, "response": {"error": "no such function"}}}
                for call in response.function_calls]})
    results["hand-loop"] = want in answer.upper()
    _say("hand-loop", results["hand-loop"], "MISSING", answer)

    passed = all(results.values())
    print(f"RESULT {'PASS' if passed else 'FAIL'} (model {args.model}, keyless, nothing sent to DecimalAI)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
