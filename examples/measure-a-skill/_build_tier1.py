#!/usr/bin/env python3
"""Build the Tier 1 notebook: reproduce_the_benchmark.ipynb.

TIER 1 OF THREE. Tier 0 showed our number, our transcripts and our
withheld-evidence stats with no credentials at all. This one hands the reader an
instrument: one free model key, the same ablation re-run on their model, and a
judge printed in full.

TWO ARMS, NOT THREE. An earlier draft carried a third arm — a ~200-word FLSA
policy WE wrote, offered as the competitor the skill had to beat. It is gone.
Its strength was a dial we controlled, so a demo that grades our own fabricated
rival is not evidence about the skill in either direction: make the rival strong
and we look humble, make it weak and we look good, and the reader has no way to
tell which we did. The published claim is "+26.09 points WITH the skill versus
WITHOUT it", and reproducing exactly that claim is this notebook's whole job.
With, and without.

WHY THE KEY ASK LANDS HERE AND NOT EARLIER. A credential asked for before there
is anything to measure is a toll booth. Asked for after the reader has seen the
claim and decided they don't believe it, the same credential is the tool for
settling the argument. Nothing above cell 1.0 needs it, and nothing in this
notebook sends it to us — the key goes from the Colab runtime to Google, and the
only DecimalAI traffic is the same anonymous GETs Tier 0 made.

NO FRAMEWORK, ON PURPOSE. `google-genai` is the entire install (~32 packages).
The claim under test is "a skill improves any agent"; teaching LangChain
vocabulary to demonstrate that would put a framework's own behaviour inside the
measurement, and every adapter differs in ways that would then need caveating.
Raw provider calls keep the only moving part the one being argued about.

WHAT THIS NOTEBOOK REFUSES TO DO. Print a number it cannot stand behind, and
raise a traceback. A stack trace in a vendor notebook reads as broken software,
so every failure here — no key, depleted credits, an unreachable registry, a
judge that returns garbage — is a printed sentence and a skipped cell.

ONE HEADLINE, NOT TWO. Cell 1.7 compares the reader's re-run against the
PUBLISHED +26.09, recomputed live from the scorecard's own per-case outcomes so
that both sides of the comparison are scored the same way. It does NOT compute a
second, differently-scored lift figure to set beside the card. A vendor notebook
that publishes two headlines has published none.

THE UNCOMFORTABLE THING IT SHIPS ANYWAY — as per-case evidence, not as a second
score. Two published cases record the skill turning a correct bare answer into a
wrong one, because its absolute rules ("hourly is never exempt", "under $684 is
never exempt") have no carve-out for computer employees (29 CFR 541.400(b):
exempt at $27.63/hour or more) or outside sales (541.500: no salary requirement
at all). Cell 1.4 opens on one of those cases rather than on a flattering one,
and cell 1.8 names both.
"""
import json
import os

RAW_MANIFEST = (
    "https://raw.githubusercontent.com/decimal-labs/decimalai-python/main/"
    "examples/measure-a-skill/manifest.yaml"
)
API = "https://api.decimal.ai/api/v1"


def _s(source):
    lines = source.split("\n")
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": _s(s)}


def code(s):
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": _s(s),
    }


META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.0"},
}

cells = [
    # ── header ─────────────────────────────────────────────────────
    md(
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        "(https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/"
        "examples/measure-a-skill/reproduce_the_benchmark.ipynb)\n"
        "\n"
        "# Re-run the benchmark yourself\n"
        "\n"
        "The [previous notebook](https://colab.research.google.com/github/decimal-labs/"
        "decimalai-python/blob/main/examples/measure-a-skill/measure_a_skill.ipynb) showed you\n"
        "our number, our test cases, our transcripts and our judge. That is a self-graded exam,\n"
        "however faithfully it is reproduced. This one hands you the marking pen.\n"
        "\n"
        "**Runtime → Run all.** One install (`google-genai`), one free key, about 20 model calls.\n"
        "\n"
        "| | needs | you get |\n"
        "|---|---|---|\n"
        "| Previous notebook | nothing | the claim, the suite, the transcripts, and what we withhold |\n"
        "| **You are here** | one free Google AI Studio key | the same ablation on your model, blind-judged |\n"
        "| Next notebook | a DecimalAI account | routing and traces for your own agent |\n"
        "\n"
        "**Two arms, which is the whole claim: with the skill, and without it.** An earlier draft\n"
        "of this notebook had a third — a policy prompt *we* wrote and offered as the competitor.\n"
        "We cut it. Its strength was a dial we controlled, so grading our own invented rival tells\n"
        "you nothing in either direction: write it strong and we look humble, write it weak and we\n"
        "look good, and you cannot tell from here which we did. The number on the scorecard is\n"
        "**+26.09 points with the skill versus without it**, and that is the only thing this\n"
        "notebook tries to reproduce.\n"
        "\n"
        "Three things are set up so that this notebook can lose:\n"
        "\n"
        "1. **The case we open on is one our skill gets wrong.** Cell 1.4 runs `case-22` — a\n"
        "   computer employee paid hourly — where the bare model answers correctly and the skill\n"
        "   does not. No judge is involved: the two answers are parsed and compared to the\n"
        "   published expectation by code, and ours is the losing one.\n"
        "2. **A judge with the labels stripped.** It never learns which response came from which\n"
        "   arm, the order is swapped from case to case so neither arm sits first more often than\n"
        "   the other, and the whole prompt is printed for you to read.\n"
        "3. **A case you write.** The last cell takes a scenario we have never seen, which is the\n"
        "   only part of this where we did not author both the question and the answer key."
    ),

    # ── 1.0 the key ask ────────────────────────────────────────────
    md(
        "## 1.0 — The one thing this notebook asks you for\n"
        "\n"
        "Everything you have read so far was ours: our benchmark, our cases, our judge. Checking\n"
        "it needs a model, and a model needs a key. That is the whole ask, and it arrives here\n"
        "rather than on page one because until now there was nothing for it to do.\n"
        "\n"
        "**Get one:** [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — Google\n"
        "account, **no credit card, about twenty seconds**. The free tier covers this notebook\n"
        "several times over.\n"
        "\n"
        "**Where it goes.** From this Colab runtime straight to Google. **DecimalAI never sees\n"
        "it.** Nothing here sends it anywhere: the only DecimalAI traffic in this notebook is the\n"
        "same anonymous `GET`s the last one made, and there is no vendor client library between\n"
        "you and `requests` to hide a second destination. Read the cells — that is the point of\n"
        "them being short.\n"
        "\n"
        "**Two ways to set it**, in the next cell:\n"
        "\n"
        "- Colab **Secrets** (the key icon in the left sidebar) → name it `GEMINI_API_KEY`, toggle\n"
        "  notebook access on. It never enters the notebook file.\n"
        "- Or paste it at the prompt. Hidden input, held in memory only.\n"
        "\n"
        "**Or set nothing.** The notebook still runs top to bottom. Every model cell prints a skip\n"
        "notice, and the cells that read *our* published run still work — including the one that\n"
        "shows the two cases where our own skill turns a correct answer into a wrong one."
    ),
    code(
        r'''# ── 1.1 model_key ─────────────────────────────────────────────────────────
import getpass, os, sys

API_KEY, KEY_SOURCE = None, None

# Colab Secrets first: the only option here that never puts the key in the
# notebook, the shell history, or a cell output.
try:
    from google.colab import userdata          # ImportError anywhere but Colab
    API_KEY = userdata.get("GEMINI_API_KEY")
    KEY_SOURCE = "Colab secret GEMINI_API_KEY"
except Exception:
    API_KEY = os.environ.get("GEMINI_API_KEY")
    if API_KEY:
        KEY_SOURCE = "environment variable GEMINI_API_KEY"

# Only offer the paste prompt where something can actually answer it. Colab and
# Jupyter route getpass through the kernel (a hidden widget); a plain script has
# a terminal or nothing. Asking anyway in a headless run makes getpass fall back
# to an echoing read and warn about it on stderr, which looks like a defect.
_can_prompt = sys.stdin.isatty() or "ipykernel" in sys.modules

if not API_KEY and _can_prompt:
    try:
        # getpass, never input(). input() echoes what you type into the cell's
        # OUTPUT, and Colab saves outputs into the .ipynb — so an input() prompt
        # here would write your key into a file you might later share or commit.
        API_KEY = getpass.getpass(
            "Paste your AI Studio key (hidden), or press Enter to skip: ").strip()
        KEY_SOURCE = "pasted into this cell (memory only, not saved to the notebook)"
    except Exception:
        # No stdin at all (papermill, nbconvert, CI). Not an error — just no key.
        API_KEY = None

HAS_KEY = bool(API_KEY)

if HAS_KEY:
    print(f"key: loaded from {KEY_SOURCE}")
    print("     it is used for calls to Google only — nothing in this notebook sends it to us")
else:
    print("key: not set.")
    print("     Every cell that needs a model will print a skip notice; the notebook still runs")
    print("     to the end, and the cells that read OUR published run still do their work.")''',
    ),

    # ── setup ──────────────────────────────────────────────────────
    md(
        "## Setup — one install, then the published run\n"
        "\n"
        "`google-genai` is the whole install, and it is the only thing between this notebook and\n"
        "the model. No agent framework: the claim under test is *a skill improves any agent*, and\n"
        "putting a framework inside the measurement would put the framework's own behaviour in\n"
        "there too.\n"
        "\n"
        "Then we pull the published run and do something the previous notebook deliberately\n"
        "declined to do — check the **expectations in the published eval suite against the\n"
        "prompts in the published run**. That notebook skipped the comparison because it fails\n"
        "for a large share of this registry, and burying a check that usually fails inside a demo\n"
        "where it happens to pass is its own kind of lie. It passes for this skill. Watch it pass."
    ),
    code(
        r'''!pip install -q google-genai''',
    ),
    code(
        r'''# ── registry fetch (no key needed — same anonymous endpoints as Tier 0) ────
import hashlib, itertools, json, random, re, textwrap, time
import requests, yaml

API = "''' + API + r'''"
MANIFEST_URL = "''' + RAW_MANIFEST + r'''"


def get(path, **params):
    """GET with backoff. Distinguishes 'rate limited' from 'actually missing'.

    Retries the TRANSPORT failures too, not just the status codes. Anonymous
    registry traffic is rate-limited at the edge, and the shape that arrives is
    often a read timeout rather than a clean 429 — a version of this helper that
    only retried status codes gave up on the first slow second.
    """
    url = path if path.startswith("http") else f"{API}{path}"
    delay = 1.0
    for attempt in range(6):
        try:
            # 60s, not 30: the measured-skill detail endpoint has been observed
            # taking 36s on a cold cache, and a timeout shorter than the server's
            # slow path turns "slow" into "unreachable" for no reason.
            r = requests.get(url, params=params or None, timeout=60)
        except requests.RequestException:
            time.sleep(delay)
            delay *= 2
            continue
        if r.status_code == 200:
            return r.json() if "json" in r.headers.get("content-type", "") else r.text
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(delay)
            delay *= 2
            continue
        raise RuntimeError(f"{r.status_code} on {url}")
    raise RuntimeError(f"gave up after 6 attempts: {url}")


def wrap(text, width=88, indent="    "):
    return textwrap.indent(textwrap.fill(str(text), width), indent)


# Same manifest as Tier 0 — one source of truth for both notebooks, so a demo
# skill that goes stale is repaired in one YAML file rather than in two .ipynb
# blobs. The embedded copy is the fallback for a raw.github blip.
FALLBACK = {"demo_skill": {"slug": "flsa-exemption-test",
                           "skill_id": "7ce7a476-606b-4768-a716-1c9cbf7b24b0",
                           "source_type": "platform", "version": 1, "demo_case": "case-22",
                           "contested_case": "case-01"}}
try:
    M = yaml.safe_load(requests.get(MANIFEST_URL, timeout=10).text)
    if not isinstance(M, dict) or "demo_skill" not in M:
        raise ValueError("manifest did not parse as expected")
except Exception as exc:
    M = FALLBACK
    print(f"manifest: using the embedded copy ({type(exc).__name__})")
DEMO = M["demo_skill"]

# PUB_OK gates every later cell. The registry being unreachable is a bad minute
# for us, not a traceback for you.
PUB_OK, SUMMARY, RUN, RESULTS, EXPECT = False, {}, {}, {}, {}
try:
    skill = get(f"/registry/skills/{DEMO['slug']}")
    SUMMARY = skill["benchmark_summary"]
    RUN = get(f"/registry/skills/{DEMO['slug']}/benchmark")["latest_run"]
    RESULTS = {r["case_name"]: r for r in RUN["results"]}
    suite = {c["name"]: c for c in yaml.safe_load(
        get(f"/registry/skills/{DEMO['slug']}/eval")["eval_yaml_text"])["cases"]}
    PUB_OK = True
except Exception as exc:
    print(f"could not reach the registry ({type(exc).__name__}: {exc}).")
    print("Every cell below will say so and skip. Re-run this cell in a minute.")

if PUB_OK:
    # THE CROSS-CHECK. Two independent endpoints: /eval renders the published
    # suite from the eval-case rows; /benchmark returns the prompt each RESULT
    # was bound to. If a suite was rewritten after its run, these disagree — and
    # then grading your re-run against those expectations would be grading it
    # against a different exam. We check before we use them, not after.
    matched = [n for n in RESULTS if n in suite and suite[n].get("prompt") == RESULTS[n].get("case_prompt")]
    EXPECT = {n: suite[n]["expectations"] for n in matched if suite[n].get("expectations")}
    scored = [n for n in RESULTS if n in EXPECT]

    print(f"{skill['url_slug']}@{DEMO['version']} — published run v{RUN['version_number']}")
    print(f"  headline           {SUMMARY['pass_rate_delta_pts']:+.2f} pts over "
          f"{SUMMARY['total_cases']} cases, judged by {SUMMARY.get('judge_model')}")
    print(f"  prompts served     {sum(1 for r in RESULTS.values() if r.get('case_prompt'))}"
          f"/{len(RESULTS)}   (withheld prompts can't be re-run at all)")
    print(f"  suite ↔ run        {len(matched)}/{len(RESULTS)} prompts identical across the two "
          f"endpoints")
    print(f"  usable here        {len(scored)} cases carry judged expectations we can grade against")
    if len(suite) > len(RESULTS):
        print(f"\n  Note: the published suite has {len(suite)} cases; this judged run covers "
              f"{len(RESULTS)}.")
        print(f"  The other {len(suite) - len(RESULTS)} are graded by scripts rather than a judge and")
        print("  are not part of the number on the scorecard. We are reproducing the run, not the suite.")


def blocked(need_key=False):
    """One place to decide whether a cell can do its work. Returns a reason, or None."""
    if not PUB_OK:
        return "the published run could not be fetched — re-run the setup cell"
    if need_key and not HAS_KEY:
        return "no model key set (cell 1.1 has the two ways to set one)"
    return None''',
    ),

    # ── 1.2 call_model ─────────────────────────────────────────────
    md(
        "## 1.2 — One call, pinned to the model that was measured\n"
        "\n"
        "Lift is model-relative: it says a skill supplied knowledge **that model** lacked. So the\n"
        "run's own model id is read out of the payload and used here — never hardcoded, because a\n"
        "hardcoded model id is the thing that quietly turns a reproduction into a different\n"
        "experiment.\n"
        "\n"
        "Model ids do get retired. The cell asks Google which ones your key is actually served\n"
        "before using one, and if the pinned id has gone, it falls back and **says so** — because\n"
        "that alone is enough to explain a disagreement in cell 1.7.\n"
        "\n"
        "`temperature=0` and `max_output_tokens=2048` match the published runner. They are not a\n"
        "guarantee of identical output — a served model changes underneath a fixed id — but a\n"
        "different temperature would guarantee a different one."
    ),
    code(
        r'''# ── 1.2 call_model ────────────────────────────────────────────────────────
from google import genai
from google.genai import types

CLIENT = genai.Client(api_key=API_KEY) if HAS_KEY else None

# Read from the run, not written by us.
RUN_MODEL = (SUMMARY.get("benchmark_model") or "gemini-flash-latest")
JUDGE_MODEL = (SUMMARY.get("judge_model") or RUN_MODEL)


def preflight(model_id, served):
    """Swap a retired id for a current flash model, loudly."""
    if not served or model_id in served:
        return model_id
    for alt in ("gemini-flash-latest", "gemini-2.5-flash"):
        if alt in served:
            print(f"  '{model_id}' is not served to this key — using '{alt}' instead.")
            print("  Your run is then a DIFFERENT model's result, which is a real and")
            print("  sufficient explanation for any disagreement in cell 1.7.")
            return alt
    return model_id


if CLIENT is not None:
    try:
        # Listing models is metadata and costs nothing — it works even on a key
        # whose paid credits are gone, which is exactly when you want to know.
        _served = {m.name.split("/")[-1] for m in CLIENT.models.list()}
    except Exception as exc:
        _served = None
        print(f"could not list models ({type(exc).__name__}) — using the pinned ids as-is")
    RUN_MODEL = preflight(RUN_MODEL, _served)
    JUDGE_MODEL = preflight(JUDGE_MODEL, _served)


def call_model(prompt, model=None, max_tokens=2048):
    """One raw google-genai call. Returns (text, error). NEVER raises."""
    if CLIENT is None:
        return None, "no model key"
    cfg = types.GenerateContentConfig(temperature=0, max_output_tokens=max_tokens)
    delay = 4.0
    for attempt in range(4):
        try:
            r = CLIENT.models.generate_content(
                model=model or RUN_MODEL, contents=prompt, config=cfg)
            return (r.text or "").strip(), None
        except Exception as exc:
            msg = str(exc)
            low = msg.lower()
            # A 429 has two completely different meanings and only one of them is
            # worth waiting out. "Prepayment credits are depleted" is a billing
            # state: every retry returns it, so backing off just burns minutes.
            if "deplet" in low or "billing" in low:
                return None, ("this key's prepaid credits are gone — a billing state, not a "
                              "rate limit, so retrying will not help")
            if any(k in msg for k in ("429", "500", "502", "503",
                                      "RESOURCE_EXHAUSTED", "UNAVAILABLE")):
                time.sleep(delay)
                delay *= 2
                continue
            return None, f"{type(exc).__name__}: {msg[:200]}"
    return None, "rate limited on four attempts — free-tier requests-per-minute; wait and re-run"


print(f"runner model  {RUN_MODEL}" + ("" if HAS_KEY else "   (nothing will be called — no key)"))
print(f"judge model   {JUDGE_MODEL}")
print("temperature   0     max_output_tokens 2048   (both match the published runner)")''',
    ),

    # ── 1.3 two arms ───────────────────────────────────────────────
    md(
        "## 1.3 — Two arms, and the only thing that differs between them\n"
        "\n"
        "The claim on the scorecard is about one variable, so the experiment moves one variable:\n"
        "\n"
        "| | what goes in the slot | who wrote it |\n"
        "|---|---|---|\n"
        "| **A — bare** | nothing | — |\n"
        "| **B — the skill** | the published `SKILL.md`, verbatim | the skill's author |\n"
        "\n"
        "*The demo skill is a worked example of measurement, not legal guidance. Nothing here —\n"
        "the skill, its answers, or the regulations cited below — is legal advice, and none of it\n"
        "should be relied on for a real classification decision.*\n"
        "\n"
        "**The envelope is not ours to invent.** Both arms are wrapped in the exact text the\n"
        "published runner wrapped *its* two arms in — same preamble, same `User:` line, same\n"
        "trailing instruction. The slot is the only difference, which is what makes the difference\n"
        "in the scores attributable to the skill and nothing else.\n"
        "\n"
        "**Where arm A wins, and why we are telling you before you run it.** The skill states two\n"
        "rules as absolutes, and both of them have exemptions it never mentions. On those cases the\n"
        "bare model is right and the skill is wrong:\n"
        "\n"
        "| the skill's absolute rule | the exemption it misses | what our published run recorded |\n"
        "|---|---|---|\n"
        "| paid hourly ⇒ never exempt | **computer employees**, 29 CFR 541.400(b) — exempt at **$27.63/hour** or more | `case-22`, a $32.00/hr systems analyst. Expected `YES / none`; the skill answered `NO / salary_basis`; **the bare model got it right** |\n"
        "| under $684/week ⇒ never exempt | **outside sales**, 29 CFR 541.500 — **no salary requirement at all** | `case-21`, an outside rep on $500/wk plus commission. Expected `YES / none`; the skill answered `NO / salary_level`; **the bare model got it right** |\n"
        "\n"
        "`case-22` is in the six-case default sample and cell 1.4 runs it first, before anything\n"
        "that flatters us; `case-21` needs `FULL_SUITE = True` in cell 1.6. Cell 1.8 lists every\n"
        "such case in the published run, with what each arm actually answered."
    ),
    code(
        r'''# ── 1.3 the two arms ──────────────────────────────────────────────────────
# Byte-for-byte the published runner's envelope. Reproduced rather than
# reinvented: a "reproduction" that rephrases the wrapper is measuring its own
# rewrite, and would be the first thing worth accusing us of.
def envelope(system, prompt):
    return f"{system}\n\nUser: {prompt}\n\nRespond as the agent. Use the skill if relevant."


BARE = "You are an agent."


def loaded(body):
    return f"You are an agent. Follow this skill:\n\n{body}\n\n"


# ARM B — the skill. body_markdown is the field the runner itself loaded, which
# is the raw file MINUS the provenance front-matter that the /s/ route prepends.
# When the pinned version is not the latest one, that field is the wrong bytes,
# so we take the pinned version from the raw route and strip the header instead.
SKILL_BODY = ""
if PUB_OK:
    SKILL_BODY = skill.get("body_markdown") or ""
    if DEMO["version"] != skill.get("latest_version_number") or not SKILL_BODY:
        try:
            raw = get(f"https://app.decimal.ai/s/{DEMO['slug']}@{DEMO['version']}/SKILL.md")
            SKILL_BODY = re.sub(r"\A---\n.*?\n---\n+", "", raw, flags=re.S)
            print(f"note: pinned v{DEMO['version']} differs from latest "
                  f"v{skill.get('latest_version_number')} — using the pinned bytes")
        except Exception as exc:
            print(f"could not fetch the pinned version ({type(exc).__name__}) — arm B is empty")

# Two arms, because the claim is about one variable. There is no third arm
# holding a prompt we wrote: we would be setting its strength ourselves, and a
# score against our own invented rival is not evidence about the skill.
ARMS = ["A", "B"]
ARM_LABEL = {"A": "bare model", "B": "the skill"}


def arm_input(arm, prompt):
    if arm == "A":
        return envelope(BARE, prompt)
    return envelope(loaded(SKILL_BODY), prompt)


print(f"A  bare model   {0:>6,} bytes   (the slot is empty — no policy text at all)")
print(f"B  the skill    {len(SKILL_BODY):>6,} bytes   " +
      ("(published SKILL.md, verbatim)" if SKILL_BODY else
       "<- EMPTY: the registry fetch above failed, so arm B has nothing loaded"))
print()
print("Everything outside the slot is identical in both arms. Here is arm A in full, which is")
print("also the entire input the published without-skill arm received:\n")
print(textwrap.indent(arm_input("A", "<the case prompt goes here>"), "    | "))''',
    ),

    # ── 1.4 one glance ─────────────────────────────────────────────
    md(
        "## 1.4 — One case, two arms, no judge anywhere\n"
        "\n"
        "This is the cell to read if you read only one, and we picked it against ourselves.\n"
        "\n"
        "The case asks about a computer systems analyst paid **$32.00 an hour**. The published\n"
        "expectation is *exempt* — 29 CFR 541.400(b) exempts a computer employee paid hourly at\n"
        "$27.63 or more, and the prompt invokes that exemption by name. **The bare model gets\n"
        "this right. Our skill gets it wrong**, because it applies \"hourly pay fails the\n"
        "salary-basis prong\" absolutely, with no carve-out.\n"
        "\n"
        "So the first case you watch run is one where the arm we are selling loses to the arm\n"
        "that has nothing in it. Two arms, one of them ours, and ours is the one that is wrong.\n"
        "\n"
        "The verdict below is **code**, not an LLM. Each reply is parsed as JSON and its `exempt`\n"
        "field is compared against the value quoted verbatim from the published expectation. You\n"
        "do not have to trust a judge, or us, to see `YES` and `NO` disagree — and the raw replies\n"
        "are printed underneath so you can check that the parser is not doing the work.\n"
        "\n"
        "**A caveat we owe you about the suite itself.** It contains a second case (`case-01`, a\n"
        "software specialist at $80/hour) with the *same* fact pattern — an hourly computer\n"
        "employee well above $27.63 — and the **opposite** expected answer. Both cannot be right,\n"
        "so one of the two expectations we wrote is wrong, and whichever arm is consistent across\n"
        "the pair gets marked WRONG on one of them. We think `case-01` is the mistaken one. We\n"
        "have left both in rather than quietly deleting the case that embarrasses us."
    ),
    code(
        r'''# ── 1.4 one_glance ────────────────────────────────────────────────────────
def parse_answer(text):
    """Pull the JSON object out of a model reply. Returns a dict, or None. No judge."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)          # tolerate ```json fences and prose
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def expected_field(expectations, field):
    """Read the expected value out of the PUBLISHED expectation strings.

    Deliberately not read out of our own with-skill transcript — grading an arm
    against our arm's answer would be circular. The source string is printed
    beside the verdict so you can see the derivation and disagree with it.
    """
    for e in expectations or []:
        m = re.search(rf"{field} field to ([A-Za-z_]+)", e)
        if m:
            return m.group(1).rstrip("."), e
    return None, None


CASE = DEMO["demo_case"]
_why = blocked()
if _why:
    print("skipped:", _why)
elif CASE not in EXPECT:
    print(f"skipped: {CASE} has no gradable expectations in the published suite")
else:
    prompt = RESULTS[CASE]["case_prompt"]
    want, want_src = expected_field(EXPECT[CASE], "exempt")
    print(f"CASE {CASE}\n"); print(wrap(prompt)); print()
    print(f"EXPECTED  exempt = {want}")
    print(f"          from the published expectation, verbatim: {want_src!r}\n")

    replies = {}
    if not HAS_KEY:
        print("skipped the two live calls:", blocked(need_key=True))
        print("Our own published transcripts for this case are still shown at the bottom.\n")
    else:
        errs = {}
        for arm in ARMS:
            txt, err = call_model(arm_input(arm, prompt))
            replies[arm] = txt
            if err:
                errs[arm] = err

        if len(errs) == len(ARMS):
            # Both arms failed for the same reason. Printing an empty results
            # table under it would suggest two answers were compared and found
            # wanting; nothing was called.
            print(f"neither arm ran: {list(errs.values())[0]}")
        else:
            for arm, err in errs.items():
                print(f"  {arm}: {err}")
            print(f"{'':<4}{'arm':<14}{'exempt':<10}{'verdict'}")
            for arm in ARMS:
                got = (parse_answer(replies.get(arm)) or {}).get("exempt", "—")
                verdict = ("unparsed" if got == "—" else
                           "correct" if str(got).upper() == str(want).upper() else "WRONG")
                print(f"    {arm} {ARM_LABEL[arm]:<12}{str(got):<10}{verdict}")

            print("\n--- the raw replies, so you can check the parser ---")
            for arm in ARMS:
                body = (replies.get(arm) or "(no reply)").strip()
                print(f"\n  [{arm} {ARM_LABEL[arm]}]")
                print(textwrap.indent(body[:600], "    "))

    print("\n--- and what OUR published run recorded for the same case ---")
    print(f"  without the skill  {(RESULTS[CASE].get('without_skill_output') or '').strip()[:120]!r}")
    print(f"  with the skill     {(RESULTS[CASE].get('with_skill_output') or '').strip()[:120]!r}")
    print(f"  recorded outcome   {RESULTS[CASE]['outcome']}")''',
    ),

    # ── 1.5 blind judge ────────────────────────────────────────────
    md(
        "## 1.5 — The judge, printed in full\n"
        "\n"
        "An LLM judge is the easiest place in a benchmark to hide a thumb on the scale, so here is\n"
        "the entire prompt, and here are the three controls it runs under — each one is a specific\n"
        "way a judge can be rigged, closed off:\n"
        "\n"
        "1. **Labels stripped.** The judge sees `Response 1` and `Response 2`. It is never told\n"
        "   which arm produced which, so it cannot favour \"the one with the skill\".\n"
        "2. **Order swapped case by case, from a printed seed.** Position bias is real: judges\n"
        "   favour the first or the last thing they read, and with two responses that bias lands\n"
        "   entirely on one arm. A per-case coin flip is unbiased *on average* and still comes out\n"
        "   lopsided on any single run. So the two orders are dealt in strict alternation instead,\n"
        "   with the seed choosing only which one goes first. Each arm sits in each position the\n"
        "   same number of times, within one. The counts are printed below rather than asserted;\n"
        "   change the seed and check that the verdicts do not move.\n"
        "3. **Criteria verbatim from the published suite.** Not paraphrased for this notebook —\n"
        "   the exact strings the published run was graded on, cross-checked against the run's own\n"
        "   prompts in the setup cell. The judge grades against a rubric we wrote *before* your\n"
        "   answers existed and cannot retune now.\n"
        "\n"
        "One call grades both responses to a case together: the same judge, the same context, one\n"
        "pass — a within-case comparison rather than two separate gradings that only meet in a\n"
        "spreadsheet, and one call against the free tier instead of two.\n"
        "\n"
        "The grading rule afterwards is code, not judgement: **a case passes for an arm only if\n"
        "every criterion for that case passes.** Partial credit is where soft benchmarks live."
    ),
    code(
        r'''# ── 1.5 blind_judge ───────────────────────────────────────────────────────
SEED = 20260810          # change it; the verdicts should not move


JUDGE_TEMPLATE = """You are grading two candidate answers to the same task.

You are NOT told which system produced which answer, and their order was randomized.
Grade each answer independently and ONLY against the numbered criteria below.
Judge substance, not style: ignore length, tone, formatting and explanation quality.

TASK GIVEN TO BOTH:
{task}

CRITERIA:
{criteria}

RESPONSES:
{responses}

Return ONLY a JSON object and nothing else, in exactly this form:
{schema}
where each value is true if that response satisfies that criterion, and false otherwise."""


def build_judge_prompt(task, expectations, texts_in_order):
    criteria = "\n".join(f"{i}. {e}" for i, e in enumerate(expectations, 1))
    responses = "\n\n".join(
        f"--- Response {i} ---\n{(t or '(no response)').strip()}"
        for i, t in enumerate(texts_in_order, 1))
    schema = "{" + ", ".join(
        '"%d": {%s}' % (i, ", ".join('"c%d": true|false' % k
                                     for k in range(1, len(expectations) + 1)))
        for i in range(1, len(texts_in_order) + 1)) + "}"
    return JUDGE_TEMPLATE.format(task=task, criteria=criteria,
                                 responses=responses, schema=schema)


# Counterbalanced, not free-shuffled. With two arms there are exactly two orders,
# and they are dealt in strict alternation, so each arm sits in each position the
# same number of times (within one) across the suite. A per-case coin flip is
# unbiased in expectation but lands lopsided on any single run, and with only two
# responses in the prompt, position bias falls entirely on one arm — "the skill
# was shown first on most cases" is a fair thing to be suspicious of. The seed
# decides only which order the alternation starts on, so it is not our thumb.
PERMS = [list(p) for p in itertools.permutations(ARMS)]
_START = random.Random(SEED).randrange(len(PERMS))
_ORDERED_CASES = sorted(EXPECT) if PUB_OK else []
ORDER = {n: PERMS[(i + _START) % len(PERMS)] for i, n in enumerate(_ORDERED_CASES)}


def shuffled_order(case_name):
    """Per-case presentation order. Deterministic, balanced, and printed below."""
    return ORDER.get(case_name, PERMS[_START])


def judge_case(case_name, task, expectations, replies):
    """One call, both responses, blind. Returns ({arm: [bool per criterion]}, error)."""
    order = shuffled_order(case_name)
    prompt = build_judge_prompt(task, expectations, [replies.get(a) for a in order])
    raw, err = call_model(prompt, model=JUDGE_MODEL, max_tokens=1024)
    if err:
        return None, err
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        verdicts = json.loads(m.group(0))
    except Exception:
        # A judge that returns prose is an UNGRADED case, never a silent zero:
        # scoring a parse failure as "failed" would quietly favour whichever arm
        # confuses the judge least.
        return None, f"judge returned unparseable output: {(raw or '')[:120]!r}"
    out = {}
    for i, arm in enumerate(order, 1):
        row = verdicts.get(str(i)) or {}
        out[arm] = [bool(row.get(f"c{k}")) for k in range(1, len(expectations) + 1)]
    return out, None


print(JUDGE_TEMPLATE)
print("\n" + "=" * 88)
print(f"seed {SEED}, alternation starts on order {_START + 1} of {len(PERMS)}\n")
_why = blocked()
if _why:
    print("  (case list unavailable:", _why + ")")
else:
    for n in _ORDERED_CASES[:6]:
        print(f"  {n}   Response 1/2  =  " + " / ".join(shuffled_order(n)))
    print("  ...and so on, alternating per case. The judge never sees this mapping.\n")
    firsts = {a: sum(1 for n in _ORDERED_CASES if shuffled_order(n)[0] == a) for a in ARMS}
    print(f"  position balance over {len(_ORDERED_CASES)} cases — times each arm is shown FIRST:")
    print("    " + "   ".join(f"{a} {firsts[a]}" for a in ARMS) +
          "     (equal within one, by construction rather than by luck)")''',
    ),

    # ── pre-commitment before the run ──────────────────────────────
    md(
        "## Before you press run — what we are predicting, and what would sink us\n"
        "\n"
        "Written down **before** the numbers exist, because a band drawn after seeing the result\n"
        "is not a band, it is an excuse.\n"
        "\n"
        "**The sample.** Six cases, chosen by `sha256(case_name) % total < 6`. Not \"the six\n"
        "hardest\", not \"the six that flipped\" — a hash has no idea what is in a case, and the\n"
        "rule is printed with the selection so you can verify it picked what it says it picked.\n"
        "Set `FULL_SUITE = True` to run all of them.\n"
        "\n"
        "**The band.** Your re-run is *paired*: both arms see the same cases, so the honest\n"
        "interval is a McNemar band computed from the published flip counts, and the cell prints\n"
        "it before your result. Be warned what it looks like at six cases: **one case is 16.7\n"
        "points**, and the 95% band comes out around ±35 — wider than the 26-point effect being\n"
        "claimed. At the full 23 it is about ±18, which finally clears the effect, but not by\n"
        "much. (Both are recomputed live below from the published flip counts, so they move if\n"
        "the skill is re-benchmarked.) We would rather tell you now that our own demo is\n"
        "underpowered than have you find it on your own time.\n"
        "\n"
        "**So the aggregate cannot settle this, and we are not asking it to.** What carries the\n"
        "argument is the per-case flips: specific cases where the bare arm says a person is exempt\n"
        "and the skill arm says they are not.\n"
        "\n"
        "**What disagreement would mean**, in advance:\n"
        "\n"
        "| you see | most likely because |\n"
        "|---|---|\n"
        "| delta inside the band | sampling — expected, at six cases almost anything is |\n"
        "| delta far below, flips still happen | judge drift; a newer judge model grades our rubric differently |\n"
        "| no flips at all on the hourly-pay cases | the model moved — a newer one may already know the rule, which shrinks lift honestly. This is the falsifier |\n"
        "| the bare arm ahead on the expected-YES cases | expected, and it is our fault: the skill states \"hourly\" and \"under $684\" as absolute disqualifiers and has no carve-out for computer employees or outside sales. Cell 1.8 names those cases |\n"
        "| the bare arm ahead overall | possible on six cases, and cell 1.8 prints it in those words if it happens. There is no branch in this notebook that suppresses that ending |\n"
        "\n"
        "**What would falsify the claim outright:** the with-skill arm failing to answer\n"
        "`exempt: NO` on the hourly-pay cases. No band rescues that, and this notebook would print\n"
        "it in cell 1.4 before you ever got here."
    ),
    code(
        r'''# ── 1.6 run_the_ablation ──────────────────────────────────────────────────
FULL_SUITE = False        # True = all cases; ~3 model calls each
SAMPLE_SIZE = 6
PACE_S = 1.0              # free-tier requests-per-minute is the binding constraint

MINE, UNGRADED = {}, []
_why = blocked(need_key=True)
if _why:
    print("skipped:", _why)
    print("Cells 1.7 and 1.8 still read OUR published run, which is worth doing even with no")
    print("key — 1.8 shows the two cases where our own skill gets the answer wrong.")
else:
    names = sorted(EXPECT)
    total = len(names)
    # Deterministic and content-blind. The hash cannot see which cases flipped,
    # which is the entire reason to use one: any hand-picked six invites exactly
    # the accusation this notebook exists to answer.
    picked = names if FULL_SUITE else [
        n for n in names
        if int(hashlib.sha256(n.encode()).hexdigest(), 16) % total < SAMPLE_SIZE]

    print(f"rule:     sha256(case_name) % {total} < {SAMPLE_SIZE}"
          if not FULL_SUITE else "rule:     FULL_SUITE — every case")
    print(f"selected: {len(picked)} of {total} — {', '.join(picked)}")
    print(f"demo case {DEMO['demo_case']} in this sample: "
          f"{'yes' if DEMO['demo_case'] in picked else 'no — 1.4 showed it separately'}")
    print(f"calls:    ~{len(picked) * 3} ({len(picked)} cases x 2 arms + 1 judge call each)\n")

    for i, n in enumerate(picked, 1):
        prompt, exps = RESULTS[n]["case_prompt"], EXPECT[n]
        replies, failed = {}, None
        for arm in ARMS:
            txt, err = call_model(arm_input(arm, prompt))
            replies[arm] = txt
            if err:
                failed = err
            time.sleep(PACE_S)
        if failed:
            UNGRADED.append((n, failed))
            print(f"  [{i}/{len(picked)}] {n}  skipped — {failed}")
            if "credits" in failed:
                print("  Stopping: every remaining call would return the same thing.")
                break
            continue

        verdicts, err = judge_case(n, prompt, exps, replies)
        time.sleep(PACE_S)
        if err:
            UNGRADED.append((n, err))
            print(f"  [{i}/{len(picked)}] {n}  ungraded — {err}")
            continue

        # A case passes for an arm only if EVERY criterion passes. No partial credit.
        MINE[n] = {arm: all(verdicts[arm]) for arm in ARMS}
        MINE[n]["_detail"] = verdicts
        MINE[n]["_replies"] = replies
        print(f"  [{i}/{len(picked)}] {n}  " +
              "  ".join(f"{a}={'pass' if MINE[n][a] else 'fail'}" for a in ARMS))

    print(f"\ngraded {len(MINE)} cases" + (f", {len(UNGRADED)} unusable" if UNGRADED else ""))''',
    ),

    # ── 1.7 compare ────────────────────────────────────────────────
    md(
        "## 1.7 — Your number against the published one\n"
        "\n"
        "This cell scores **your** run with the rule from 1.5 (every criterion must pass) and puts\n"
        "it beside the **published +26.09** — recomputed live from the scorecard's own per-case\n"
        "outcomes, so what you are compared against is the number on the card and not some second\n"
        "figure this notebook invented. The recomputation is printed next to the headline as a\n"
        "check on us: if those two ever disagree, this notebook is reading the run wrong and the\n"
        "cell says so.\n"
        "\n"
        "**The one asymmetry, stated plainly.** Your side is graded live, by the judge and the\n"
        "all-criteria rule in 1.5. Our side is not re-graded here at all: the scorecard's own\n"
        "per-case verdict is what the published headline is computed from, and this cell reads\n"
        "that verdict back exactly as recorded. Same cases and same arms, two graders — that is\n"
        "the first thing to suspect if the numbers drift, ahead of anything about the skill.\n"
        "\n"
        "**Read the per-case rows, not the total.** At six cases one flip is 16.7 points, so the\n"
        "aggregate cannot settle anything on its own. The flips can."
    ),
    code(
        r'''# ── 1.7 compare_to_published ──────────────────────────────────────────────
# The scorecard's OWN per-case verdict, read back out of the run. `outcome` is
# the field the published headline is computed from — 12 passes with the skill
# and 6 without, over 23 cases, is the +26.09 on the card. Reading it back is how
# your re-run gets compared against that number instead of against a second,
# differently-scored one this notebook made up. A benchmark that ships two
# headlines has shipped none.
PASSED_FOR = {"B": {"pass_kept", "flip_to_pass"},     # the with-skill arm
              "A": {"pass_kept", "flip_to_fail"}}     # the without-skill arm
KNOWN_OUTCOMES = PASSED_FOR["A"] | PASSED_FOR["B"] | {"fail_kept"}


def pub_pass(result, arm):
    return result.get("outcome") in PASSED_FOR[arm]


def delta_pts(rows, a="A", b="B"):
    n = len(rows)
    if not n:
        return None, 0, 0
    gained = sum(1 for r in rows if r[b] and not r[a])
    lost = sum(1 for r in rows if r[a] and not r[b])
    return 100.0 * (gained - lost) / n, gained, lost


def band95(gained, lost, n_pub, n):
    """95% McNemar band for a re-run of n cases, from the published flip rates.

    Paired, because you re-run the SAME cases in both arms. An unpaired
    two-proportion interval would be narrower and would be the wrong answer —
    the flattering mistake, which is why it is worth naming.
    """
    pg, pl = gained / n_pub, lost / n_pub
    var = (pg + pl - (pg - pl) ** 2) / max(n, 1)
    return 1.96 * (var ** 0.5) * 100


_why = blocked()
pub_rows = []
if _why:
    print("skipped:", _why)
else:
    pub_rows = [{"n": n, "A": pub_pass(r, "A"), "B": pub_pass(r, "B")}
                for n, r in RESULTS.items() if r.get("outcome") in KNOWN_OUTCOMES]
    if not pub_rows:
        print("skipped: this run records no per-case outcomes, so there is nothing to "
              "compare your run against")
if pub_rows:
    pub_d, pub_g, pub_l = delta_pts(pub_rows)
    head = SUMMARY["pass_rate_delta_pts"]

    print("THE PUBLISHED NUMBER — the scorecard headline, and the same thing recomputed here")
    print(f"  on the scorecard                {head:+6.2f} pts over {SUMMARY['total_cases']} cases")
    print(f"  from its own per-case outcomes  {pub_d:+6.2f} pts over {len(pub_rows)} cases"
          f"   ({pub_g} gained, {pub_l} lost)")
    # Those two lines are the SAME quantity by two routes. Printing them side by
    # side is a check on us, not a second headline: if they part company, this
    # notebook is reading the run differently from the card, and you should know
    # that before you weigh your own result against it.
    head_n = SUMMARY.get("total_cases")
    if abs(head - pub_d) > 0.05 or (head_n is not None and len(pub_rows) != head_n):
        print()
        print(wrap(
            "Those two lines should be the same number over the same case count, and here they "
            "are not. That is a defect in this notebook's reading of the run, not a finding "
            "about the skill — treat the comparison below as indicative and tell us.", 88, "  "))
    print()

    if not MINE:
        print("YOUR run: nothing to compare — cell 1.6 graded no cases.")
    else:
        rows = [{"n": n, **{a: MINE[n][a] for a in ARMS}} for n in MINE]
        mine_d, mine_g, mine_l = delta_pts(rows)
        sub = [r for r in pub_rows if r["n"] in MINE]
        sub_d, _, _ = delta_pts(sub)
        band = band95(pub_g, pub_l, len(pub_rows), len(rows))

        print(f"YOUR run on {RUN_MODEL}, {len(rows)} cases, graded by {JUDGE_MODEL}")
        print(f"  your delta (with skill vs without)     {mine_d:+6.2f} pts"
              f"   gained {mine_g}, lost {mine_l}")
        print(f"  the same cases in the published run    {sub_d:+6.2f} pts")
        print(f"  95% band around the published rate     +/-{band:.1f} pts at n={len(rows)}"
              f"   (one case = {100 / len(rows):.1f} pts)")
        print("  your side is graded live by the rule in 1.5; ours is the scorecard's own")
        print("  per-case outcome. Same cases, same arms, two graders — suspect that first.")
        # A published run with no discordant pairs gives a zero-width band, which
        # would make every possible result "outside" it. That is a degenerate
        # estimate, not a verdict on your run.
        inside = abs(mine_d - pub_d) <= band if (pub_g + pub_l) else None
        print("  → " + {True: "inside", False: "OUTSIDE"}.get(
            inside, "no band: the published run has no cases that changed either way, so there "
                    "is nothing to estimate a spread from"))
        if inside is False:
            print(wrap("Outside is not automatically a scandal — at this sample size it is "
                       "mostly sampling, and the pre-run table lists the other three causes. "
                       "The per-case flips below are the evidence that survives a wobbly "
                       "aggregate.", 88, "  "))

        print("\nper case — the part that carries the argument")
        print(f"  {'case':<10}{'published':<14}{'you: A bare':<14}{'B skill':<14}")
        for n in sorted(MINE):
            pub = RESULTS[n]["outcome"]
            cells_ = "".join(f"{('pass' if MINE[n][a] else 'fail'):<14}" for a in ARMS)
            flip = ("  <- flip" if MINE[n]["B"] and not MINE[n]["A"] else
                    "  <- the skill LOST this one" if MINE[n]["A"] and not MINE[n]["B"] else "")
            print(f"  {n:<10}{pub:<14}{cells_}{flip}")
        if UNGRADED:
            print(f"\n  {len(UNGRADED)} case(s) could not be graded and are excluded, not zeroed:")
            for n, why in UNGRADED:
                print(f"    {n}: {why}")''',
    ),

    # ── 1.8 where the gap is ───────────────────────────────────────
    md(
        "## 1.8 — Where the gap actually is, including the ending we would rather not have\n"
        "\n"
        "The suite splits cleanly in two, and the split is not ours to draw — it comes straight\n"
        "out of the published expectations:\n"
        "\n"
        "- **expected NO** — a worker who looks exempt and is not. The trap cases.\n"
        "- **expected YES** — a genuinely exempt worker. Getting these right means *not* crying\n"
        "  overtime.\n"
        "\n"
        "**Our hypothesis, stated before the numbers:** the whole of the skill's lift lives in the\n"
        "expected-NO column. That is where the model's default instinct — \"well-paid office job,\n"
        "must be exempt\" — is wrong, and where the salary-basis rule, the $684 threshold and the\n"
        "tie-break order change the answer. In the expected-YES column we expect the skill to be\n"
        "**level with the bare model at best, and behind it wherever a case turns on one of the\n"
        "two carve-outs from cell 1.3** — `case-22` is in the default six-case sample, `case-21`\n"
        "needs `FULL_SUITE = True` — because the skill's absolute rules have no carve-out to apply.\n"
        "\n"
        "A skill that helps on one half of a suite and hurts on the other is the honest shape of\n"
        "most real ones. The scorecard number is the sum of the two, not the good half.\n"
        "\n"
        "And one more thing this cell prints from our own transcripts: every case where the skill\n"
        "turns a correct bare answer into a wrong one. Shown per case, with what each arm actually\n"
        "answered — not rolled into a second lift figure to argue with the scorecard's."
    ),
    code(
        r'''# ── 1.8 where_the_gap_is ──────────────────────────────────────────────────
def case_class(name):
    """From the published expectation string, not from anybody's answer."""
    want, _ = expected_field(EXPECT.get(name), "exempt")
    return {"NO": "expected NO (looks exempt, is not)",
            "YES": "expected YES (genuinely exempt)"}.get((want or "").upper(), "other")


_why = blocked()
if _why:
    print("skipped:", _why)
else:
    classes = sorted({case_class(n) for n in EXPECT})

    # The scorecard's own per-case outcomes, split by class. Not re-scored here:
    # these are the four verdicts the published run recorded, counted.
    OUTS = ("flip_to_pass", "pass_kept", "fail_kept", "flip_to_fail")
    print("OUR published run, by case class, in the scorecard's own per-case outcomes")
    print(f"  {'class':<34}{'n':>4}" + "".join(f"{o:>14}" for o in OUTS))
    for c in classes:
        rows = [r for n, r in RESULTS.items() if n in EXPECT and case_class(n) == c]
        if not rows:
            continue
        print(f"  {c:<34}{len(rows):>4}" +
              "".join(f"{sum(1 for r in rows if r.get('outcome') == o):>14}" for o in OUTS))
    print("  a fail_kept is a case neither arm passed.")

    # PER-CASE EVIDENCE, NOT A SECOND SCORE. Read straight off the two recorded
    # transcripts with the same parser cell 1.4 used: cases where the arm WITHOUT
    # the skill matched the published expectation and the arm WITH it did not.
    # It produces no aggregate and competes with no headline; it names cases.
    worse = []
    for n in sorted(EXPECT):
        want, _ = expected_field(EXPECT[n], "exempt")
        bare = parse_answer(RESULTS[n].get("without_skill_output")) or {}
        withs = parse_answer(RESULTS[n].get("with_skill_output")) or {}
        if not want or "exempt" not in bare or "exempt" not in withs:
            continue
        if (str(bare["exempt"]).upper() == want.upper()
                and str(withs["exempt"]).upper() != want.upper()):
            worse.append((n, want, withs.get("exempt"), withs.get("failed_prong"),
                          RESULTS[n]["outcome"]))
    print()
    if worse:
        print(f"  {len(worse)} case(s) where our recorded WITH-skill answer is wrong and the bare "
              f"one is right:")
        print(f"    {'case':<10}{'expected':<11}{'bare':<9}{'skill answered':<22}"
              f"{'scorecard outcome'}")
        for n, want, got, prong, out in worse:
            print(f"    {n:<10}{want:<11}{'correct':<9}{f'{got} / {prong}':<22}{out}")
        print()
        print(wrap(
            "These are the two exemptions from cell 1.3 — the computer-employee hourly rate in "
            "29 CFR 541.400(b) and outside sales in 541.500 — neither of which the skill has. "
            "They sit in the same suite the +26.09 is computed over, and the last column above "
            "is the scorecard's own verdict on each, so you can see exactly how the published "
            "run counted a case the skill got wrong. We are showing you the cases rather than "
            "folding them into a second score to argue with the first — the rows above are the "
            "evidence, and they need no badge to interpret.",
            88, "  "))
    else:
        print("  No case in the published run has the bare arm right where the skill arm is wrong.")

    if not MINE:
        print("\nYOUR run: nothing to split — cell 1.6 graded no cases.")
    else:
        print(f"\nYOUR run, {len(MINE)} cases, by class")
        print(f"  {'class':<34}{'n':>4}{'A bare':>9}{'B skill':>9}")
        for c in classes:
            ns = [n for n in MINE if case_class(n) == c]
            if not ns:
                continue
            print(f"  {c:<34}{len(ns):>4}" +
                  "".join(f"{sum(MINE[n][a] for n in ns):>9}" for a in ARMS))

        a_ = sum(MINE[n]["A"] for n in MINE)
        b_ = sum(MINE[n]["B"] for n in MINE)
        print(f"\n  totals   A bare {a_}/{len(MINE)}   B skill {b_}/{len(MINE)}\n")
        if a_ > b_:
            print(wrap(
                "The bare model beat the skill on your run. We said this notebook would print "
                "that ending if it happened, so: on these cases, on this model, the published "
                "file made the answers worse rather than better. Before you conclude the skill "
                "is worthless, check the class split above — if the loss is concentrated in the "
                "expected-YES column it is the two missing carve-outs from cell 1.3 doing it, "
                "and the sample is small enough that which cases the hash picked matters. But "
                "the honest headline for YOUR run is the one printed above, not ours.",
                88, "  "))
        elif a_ == b_:
            print(wrap(
                "The skill and the bare model tied on your run. At this sample size a tie is a "
                "real possibility rather than a surprise — one case is worth 16.7 points — and "
                "the class split above is more informative than the total: a skill that gains on "
                "the expected-NO cases and gives it back on the expected-YES ones nets to zero "
                "while still telling you something true about where it helps. Set FULL_SUITE = "
                "True in cell 1.6 for the version of this table that is worth arguing about.",
                88, "  "))
        else:
            print(wrap(
                "The skill came out ahead on your run, which is the result we predicted, so "
                "treat it with the appropriate suspicion: it is one model, one judge, and this "
                "sample of cases, graded against expectations we wrote. The part of it that is "
                "hard to explain away is in the per-case rows of cell 1.7 — a specific case "
                "where the bare arm gets a worker's exemption wrong and the skill arm gets it "
                "right. The total is the weakest evidence on this page.", 88, "  "))''',
    ),

    # ── 1.9 your own case ──────────────────────────────────────────
    md(
        "## 1.9 — A case we have never seen\n"
        "\n"
        "Everything up to here still has our fingerprints on it, and it is worth saying so plainly:\n"
        "**we wrote the skill, we wrote the test cases, we wrote the expectations the judge grades\n"
        "against, and we picked the judge.** Every control in cell 1.5 constrains how those pieces\n"
        "are *used*; none of them changes who authored them.\n"
        "\n"
        "This cell is the one place that is not true. Type a scenario of your own — ideally one\n"
        "where you already know the answer, or one you would actually be asked at work. Both arms,\n"
        "no judge, no scoring. You read the two replies and decide.\n"
        "\n"
        "Things that are worth trying, because they are where the rule bites: someone paid hourly\n"
        "at a high rate; someone salaried at exactly $684 a week; a salaried manager at $600 a\n"
        "week; a well-paid salaried worker whose actual duties are manual."
    ),
    code(
        r"""# ── 1.9 your_own_case ─────────────────────────────────────────────────────
# Edit this. Keep the JSON instruction so the two answers stay comparable.
YOUR_CASE = '''A veterinary clinic pays its lead technician $46 an hour. She supervises three
assistants, sets the surgery schedule, and has authority to hire and fire them.
She works about 46 hours in a typical week.

Is this role exempt from FLSA overtime? Reply with only a JSON object
{"exempt": "YES" or "NO", "failed_prong": one of salary_basis, salary_level, duties, none}.'''

_why = blocked(need_key=True)
if _why:
    print("skipped:", _why)
    print("\nThe case is still here to read, and it is the cell worth coming back for once you")
    print("have a key — it is the only one where we did not write the question.")
    print("\n" + textwrap.indent(YOUR_CASE, "    | "))
else:
    print(textwrap.indent(YOUR_CASE, "    | ") + "\n")
    for arm in ARMS:
        txt, err = call_model(arm_input(arm, YOUR_CASE))
        time.sleep(PACE_S)
        print(f"[{arm}  {ARM_LABEL[arm]}]")
        if err:
            print(f"    {err}\n")
            continue
        print(textwrap.indent((txt or "(no reply)").strip()[:800], "    "))
        got = (parse_answer(txt) or {}).get("exempt")
        if got:
            print(f"    -> exempt = {got}")
        print()

    print(wrap(
        "No score, on purpose. There is no expectation to grade against because you wrote the "
        "case, and inventing one now would put us back on both sides of the exam. If the two "
        "answers disagree, the disagreement is the result.", 88, "  "))"""
    ),

    # ── close ──────────────────────────────────────────────────────
    md(
        "## What you have now, and what is still ours\n"
        "\n"
        "You have run the same ablation the scorecard reports — with the skill and without it —\n"
        "on your key, on your model, under a judge whose prompt and controls you have read. And\n"
        "you have the cases where our own skill turns a correct answer into a wrong one, named,\n"
        "with what each arm actually said.\n"
        "\n"
        "What is still ours: the skill, the cases, the expectations, the choice of judge model,\n"
        "and the per-case scoring behind the headline. Cell 1.9 is the only part where none of\n"
        "that applies. If you want the rest under your control, that is the next notebook — a\n"
        "DecimalAI account gets you routing and traces over **your** skills and **your** cases,\n"
        "and the same benchmark run on evidence you own.\n"
        "\n"
        "Or take the file and go: `pip install decimalai` then\n"
        "`decimalai skills pull flsa-exemption-test --out .claude/skills/` — anonymous, no\n"
        "account, and your agent picks it up from disk. The measurement was the product; the\n"
        "Markdown was never locked up."
    ),
]

nb = {"cells": cells, "metadata": META, "nbformat": 4, "nbformat_minor": 4}
path = os.path.join(os.path.dirname(__file__), "reproduce_the_benchmark.ipynb")
with open(path, "w") as f:
    json.dump(nb, f, indent=1)
print(f"built {path} — {len(cells)} cells")
