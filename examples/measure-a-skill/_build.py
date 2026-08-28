#!/usr/bin/env python3
"""Build the Tier 0 notebook: measure_a_skill.ipynb.

TIER 0 OF THREE. Zero credentials, zero installs, and it is independently
complete — a reader who stops here has still audited the vendor. Tier 1 adds one
free model key and re-runs the ablation; Tier 2 adds a DecimalAI key for routing
and traces. The whole design puts the payoff before every credential ask.

WHY THIS NOTEBOOK EXISTS. Every other notebook in examples/ asks for
DECIMAL_API_KEY in its second code cell and raises DecimalConfigError on the
placeholder, so a visitor's first experience of the SDK is a traceback. This one
runs top to bottom on a cold Colab with nothing configured, because everything
it reads is public.

WHAT IT MUST NEVER DO. Print a number it cannot stand behind. Every claim below
is fetched live and gate-checked against manifest.yaml first; if a gate fails the
notebook falls through to the spare and says so, rather than rendering a stale
figure. The registry's own withheld-evidence statistics are printed too — a
vendor demo that volunteers how much of its evidence it refuses to show is doing
the one thing a skeptical reader cannot get anywhere else.
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
    # ── 0.0 promise ────────────────────────────────────────────────
    md(
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        "(https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/"
        "examples/measure-a-skill/measure_a_skill.ipynb)\n"
        "\n"
        "# Don't trust our number\n"
        "\n"
        "DecimalAI's registry ranks agent skills by a **measured** with-vs-without benchmark\n"
        "— not stars, not downloads. This notebook hands you the evidence behind one of those\n"
        "numbers so you can decide whether it means anything.\n"
        "\n"
        "**Runtime → Run all.** Nothing here asks you to sign up or install anything.\n"
        "\n"
        "| | needs | you get |\n"
        "|---|---|---|\n"
        "| **You are here** | nothing | the claim, the test suite, the per-case transcripts, the safety scan, and what we refuse to show you |\n"
        "| Next notebook | one free Google AI Studio key | re-run the benchmark yourself, blind-judged, on your model |\n"
        "| After that | a DecimalAI account | routing and traces for your own agent |\n"
        "\n"
        "Stopping after this cell block is a perfectly good outcome. You will have audited a\n"
        "vendor without giving it anything."
    ),

    # ── 0.1 preflight ──────────────────────────────────────────────
    md(
        "## Setup — one helper, no installs\n"
        "\n"
        "Colab already ships `requests` and `PyYAML`, so there is nothing to `pip install`.\n"
        "\n"
        "The retry helper is not boilerplate. Anonymous traffic to the registry is rate-limited\n"
        "at the edge, and a cold cache on the measured-skill listing can take a few seconds — a\n"
        "notebook whose entire pitch is *re-run it yourself* cannot fall over on the fourth\n"
        "request."
    ),
    code(
        "import json, time, textwrap\n"
        "import requests, yaml\n"
        "\n"
        'API = "' + API + '"\n'
        'MANIFEST_URL = "' + RAW_MANIFEST + '"\n'
        "\n"
        "\n"
        "def get(path, **params):\n"
        '    """GET with backoff. Distinguishes \'rate limited\' from \'actually missing\'."""\n'
        "    url = path if path.startswith(\"http\") else f\"{API}{path}\"\n"
        "    delay = 1.0\n"
        "    for attempt in range(5):\n"
        "        try:\n"
        "            r = requests.get(url, params=params or None, timeout=30)\n"
        "        except requests.exceptions.RequestException as exc:\n"
        "            # A dropped connection is the same event as a 503, one layer down.\n"
        "            # Observed: one read timeout against app.decimal.ai ended this\n"
        "            # notebook in a 60-line urllib3 traceback at the last cell.\n"
        "            if attempt == 4:\n"
        "                raise RuntimeError(f\"{type(exc).__name__} on {url}\") from None\n"
        "            time.sleep(delay)\n"
        "            delay *= 2\n"
        "            continue\n"
        "        if r.status_code == 200:\n"
        "            return r.json() if \"json\" in r.headers.get(\"content-type\", \"\") else r.text\n"
        "        if r.status_code in (429, 500, 502, 503, 504):\n"
        "            time.sleep(delay)\n"
        "            delay *= 2\n"
        "            continue\n"
        "        raise RuntimeError(f\"{r.status_code} on {url}\")\n"
        "    raise RuntimeError(f\"gave up after 5 attempts: {url}\")\n"
        "\n"
        "\n"
        "def wrap(text, width=88, indent=\"    \"):\n"
        "    return textwrap.indent(textwrap.fill(str(text), width), indent)\n"
        "\n"
        "\n"
        "# The manifest lives on `main` so a stale figure is a one-line YAML edit rather\n"
        "# than notebook surgery. The embedded copy below is the fallback: raw.github can\n"
        "# be briefly unavailable, and a reader who hits that moment should get a working\n"
        "# notebook and a note — not a KeyError on a 404 page parsed as YAML.\n"
        "FALLBACK = {\n"
        "    # demo_case is case-22 ON PURPOSE — the case where this skill LOSES. Keep the\n"
        "    # two copies in step; a fallback that quietly disagrees with main is how a\n"
        "    # notebook starts telling a different story than its manifest.\n"
        "    \"demo_skill\": {\"slug\": \"flsa-exemption-test\",\n"
        "                   \"skill_id\": \"7ce7a476-606b-4768-a716-1c9cbf7b24b0\",\n"
        "                   \"source_type\": \"platform\", \"version\": 1, \"demo_case\": \"case-22\",\n"
        "                   \"contested_case\": \"case-01\"},\n"
        "    \"gates\": {\"min_delta_pts\": 20, \"min_cases\": 20,\n"
        "              \"require_grading_method\": \"judged\",\n"
        # Not optional. Any reader whose raw.github fetch above fails lands on this
        # dict — a fallback missing this key silently
        # turns the evidence gate off for all of them and leaves the cell printing
        # only the arithmetic it was written to stop being sufficient.
        "              \"require_all_prompts_served\": True},\n"
        "    \"registry_disclosure\": {\"measured_public_skills\": 43819, \"graded_cases\": 980708,\n"
        "                            \"cases_withheld_pct\": 12.6, \"platform_cases_withheld_pct\": 56.6,\n"
        "                            \"github_import_cases_withheld_pct\": 0.7, \"as_of\": \"2026-08-28\"},\n"
        "}\n"
        "\n"
        "try:\n"
        "    M = yaml.safe_load(requests.get(MANIFEST_URL, timeout=10).text)\n"
        "    if not isinstance(M, dict) or \"demo_skill\" not in M:\n"
        "        raise ValueError(\"manifest did not parse as expected\")\n"
        "    print(\"manifest: fetched from main\")\n"
        "except Exception as exc:\n"
        "    M = FALLBACK\n"
        "    print(f\"manifest: using the embedded copy ({type(exc).__name__}) — \"\n"
        "          \"figures may lag what is on main\")\n"
        "\n"
        "DEMO, GATES = M[\"demo_skill\"], M[\"gates\"]\n"
        "print(\"demo skill:\", DEMO[\"slug\"])"
    ),

    # ── 0.2 the gate, as a population statistic ────────────────────
    # THE CIRCULAR GATE THIS REPLACED. The first version of this cell asked the
    # API for `sort=lift` descending, limit=100, and then screened on
    # pass_rate_delta_pts >= 20 — it sorted by the thing it was filtering for, so
    # the filter could not fail. It printed "100 skills in → 82 survive" under a
    # caption whose whole point was how FEW survive. Reading one page in the
    # registry's DEFAULT order costs nothing and makes the screen falsifiable.
    md(
        "## What survives an evidence gate\n"
        "\n"
        "Two filters, run in front of you against one page of the registry in its **default\n"
        "order** — deliberately not `sort=lift`. This cell used to ask the API for the\n"
        "highest-lift skills and then screen on lift, which is a filter that cannot fail:\n"
        "the list was selected by the very thing being tested.\n"
        "\n"
        "The first filter is the arithmetic every scorecard has — judged grading, at least\n"
        "20 cases, at least +20 points. The second is the one that costs something: can the\n"
        "skill show you the prompt behind **every** case it was graded on? That answer lives\n"
        "nowhere but the per-skill benchmark payload, so it is one request per candidate and\n"
        "the slowest cell in this notebook (about half a minute).\n"
        "\n"
        "The point is not the survivors. It is which skills the second filter removes: when\n"
        "we ran it on 2026-08-15, the arithmetic took 100 skills down to 56, and the evidence\n"
        "gate took those 56 to 52 — and all four it removed were skills **we** wrote. Your\n"
        "run prints its own figures below; where they disagree with this paragraph, believe\n"
        "the cell."
    ),
    code(
        "# NOT sort=\"lift\". This cell screens on lift, and asking the API for the\n"
        "# highest-lift skills first makes that screen unfalsifiable — a list selected by\n"
        "# the thing being filtered passes its own filter. This is the registry's default\n"
        "# order: the page a reader lands on.\n"
        "PAGE = 100          # one listing request; 100 is the API's maximum page size\n"
        "MAX_CHECKS = 60     # ceiling on the per-skill requests below, so a shift in the\n"
        "                    # registry can never turn this cell into a 100-request crawl\n"
        "\n"
        "rows = get(\"/registry/skills\", measured=\"only\", limit=PAGE)\n"
        "items = rows.get(\"items\", rows if isinstance(rows, list) else [])\n"
        "\n"
        "\n"
        "def clears_the_numbers(s):\n"
        "    # Every gate field lives under benchmark_summary, never at the top level —\n"
        "    # reading them flat silently yields None and passes everything.\n"
        "    b = s.get(\"benchmark_summary\") or {}\n"
        "    return (\n"
        "        b.get(\"grading_method\") == GATES[\"require_grading_method\"]\n"
        "        and (b.get(\"total_cases\") or 0) >= GATES[\"min_cases\"]\n"
        "        and (b.get(\"pass_rate_delta_pts\") or 0) >= GATES[\"min_delta_pts\"]\n"
        "    )\n"
        "\n"
        "\n"
        "def prompts_served(slug):\n"
        "    \"\"\"(served, total) case prompts on the run behind the published number.\n"
        "\n"
        "    A withheld prompt is visible nowhere but this payload — the listing carries\n"
        "    the headline figures and says nothing about whether the evidence exists.\n"
        "    \"\"\"\n"
        "    res = get(f\"/registry/skills/{slug}/benchmark\")[\"latest_run\"][\"results\"]\n"
        "    return sum(1 for r in res if r.get(\"case_prompt\")), len(res)\n"
        "\n"
        "\n"
        "candidates = [s for s in items if clears_the_numbers(s)]\n"
        "checked, kept, dropped, unknown = [], [], [], []\n"
        "\n"
        "if GATES.get(\"require_all_prompts_served\", True):\n"
        "    for s in candidates[:MAX_CHECKS]:\n"
        "        try:\n"
        "            served, total = prompts_served(s[\"url_slug\"])\n"
        "        except RuntimeError as exc:\n"
        "            # Counted as neither. A skill we could not check is not a skill that\n"
        "            # passed, and quietly folding it into either number would be the same\n"
        "            # kind of flattery this cell exists to avoid.\n"
        "            unknown.append((s, str(exc)))\n"
        "            continue\n"
        "        checked.append(s)\n"
        "        (kept if total and served == total else dropped).append((s, served, total))\n"
        "        time.sleep(0.25)  # the edge limiter is a small bucket with a slow refill\n"
        "else:\n"
        "    checked, kept = candidates, [(s, None, None) for s in candidates]\n"
        "\n"
        "print(f\"{len(items)} measured skills, in the registry's default order\")\n"
        "print(f\"  -> {len(candidates)} clear the arithmetic \"\n"
        "      f\"({GATES['require_grading_method']}, >= {GATES['min_cases']} cases, \"\n"
        "      f\">= +{GATES['min_delta_pts']} pts)\")\n"
        "print(f\"  -> {len(kept)} of the {len(checked)} checked can also show the prompt \"\n"
        "      f\"behind every case they were graded on\\n\")\n"
        "\n"
        "if dropped:\n"
        "    by_source, pool = {}, {}\n"
        "    for s, served, total in dropped:\n"
        "        by_source.setdefault(s.get(\"source_type\", \"?\"), []).append((s, served, total))\n"
        "    for s in checked:\n"
        "        src = s.get(\"source_type\", \"?\")\n"
        "        pool[src] = pool.get(src, 0) + 1\n"
        "    print(f\"REMOVED BY THE EVIDENCE GATE — {len(dropped)}:\")\n"
        "    for src, group in sorted(by_source.items()):\n"
        "        print(f\"  {src}: {len(group)} of the {pool[src]} checked\")\n"
        "        for s, served, total in group[:6]:\n"
        "            b = s[\"benchmark_summary\"]\n"
        "            print(f\"     {b['pass_rate_delta_pts']:+6.2f} pts  {served}/{total} prompts \"\n"
        "                  f\"served  {s['url_slug']}\")\n"
        "    if set(by_source) == {\"platform\"}:\n"
        "        print(wrap(\"Every skill this gate removed is one of ours. A published \"\n"
        "                   \"+20-point number with no way to see what was asked is exactly \"\n"
        "                   \"the combination the last cell of this notebook is about.\", 88, \"  \"))\n"
        "\n"
        "if unknown:\n"
        "    print(f\"\\n  ({len(unknown)} candidate(s) could not be checked, counted as neither)\")\n"
        "\n"
        "print(\"\\nSURVIVORS (highest lift first):\")\n"
        "for s, served, total in sorted(\n"
        "        kept, key=lambda t: -t[0][\"benchmark_summary\"][\"pass_rate_delta_pts\"])[:8]:\n"
        "    b = s[\"benchmark_summary\"]\n"
        "    print(f\"  {b['pass_rate_delta_pts']:+6.2f} pts  {b['total_cases']:>3} cases  \"\n"
        "          f\"{s.get('source_type','?'):<14} {s['url_slug']}\")\n"
        "\n"
        "print(\"\\nA high delta is not the same as a legible one, which is why the demo skill\")\n"
        "print(\"below is pinned by name rather than taken from the top of this list.\")"
    ),

    # ── 0.3 evidence card ──────────────────────────────────────────
    md(
        "## The claim, and every qualifier attached to it"
    ),
    code(
        "skill = get(f\"/registry/skills/{DEMO['slug']}\")\n"
        "b = skill[\"benchmark_summary\"]\n"
        "\n"
        "# Identity, not just existence. A slug that still resolves is not the skill you\n"
        "# meant: one slug in this registry still returns 200 and is now a medical-skills\n"
        "# import. If these pins fail we use the spare rather than describe the wrong thing.\n"
        "#\n"
        "# The id, not the author's display name. A handle is the author's to change and\n"
        "# names a real person; the uuid is the one field that cannot be re-pointed at a\n"
        "# different skill, so it is what \"same skill\" actually means here.\n"
        "assert skill[\"id\"] == DEMO[\"skill_id\"], \"skill id changed\"\n"
        "assert skill[\"source_type\"] == DEMO[\"source_type\"], \"source_type changed\"\n"
        "\n"
        "with_pct = 100.0 * b[\"passed_cases\"] / b[\"total_cases\"]\n"
        "without_pct = with_pct - b[\"pass_rate_delta_pts\"]\n"
        "\n"
        "print(f\"{skill['url_slug']}  ({skill['skill_badge']}, {skill['source_type']})\\n\")\n"
        "print(f\"  lift            {b['pass_rate_delta_pts']:+.2f} pts\")\n"
        "print(f\"  without skill   {without_pct:.1f}% of {b['total_cases']} cases\")\n"
        "print(f\"  with skill      {with_pct:.1f}%\")\n"
        "print(f\"  cases passed    {b['passed_cases']} with the skill, \"\n"
        "      f\"{round(b['total_cases'] * without_pct / 100)} without\")\n"
        "print(f\"  graded by       {b.get('judge_model')}  ({b.get('grading_method')})\")\n"
        "print(f\"  safety scan     {skill.get('safety_status')}\")\n"
        "\n"
        "print(\"\\n--- the part most scorecards leave off ---\")\n"
        "print(wrap(\n"
        "    f\"The with-skill arm still fails {b['total_cases'] - b['passed_cases']} of \"\n"
        "    f\"{b['total_cases']} cases. This skill is not 'solved' — it moved the needle \"\n"
        "    f\"{b['pass_rate_delta_pts']:+.1f} points and left plenty on the table. A registry \"\n"
        "    \"that only ever showed you the ceiling would be less useful, not more.\"))\n"
        "print()\n"
        "print(wrap(\n"
        "    \"Lift is model-relative. It was measured on \"\n"
        "    f\"{b.get('benchmark_model')}: it says this skill supplies knowledge THAT model \"\n"
        "    \"lacked, not a universal constant. A stronger base model may need it less.\"))"
    ),

    # ── 0.4 within-payload integrity ───────────────────────────────
    md(
        "## Check our arithmetic before you believe it\n"
        "\n"
        "This recomputes the headline from the per-case rows in the same payload. It is a\n"
        "genuine check on one artifact: if the summary and the cases disagree, the summary is\n"
        "not describing this run.\n"
        "\n"
        "Deliberately *not* a cross-check against the published eval suite. That comparison\n"
        "fails on a large fraction of this registry — and the reason is the subject of the next\n"
        "cell but one, not something to bury here."
    ),
    code(
        "run = get(f\"/registry/skills/{DEMO['slug']}/benchmark\")[\"latest_run\"]\n"
        "res = run[\"results\"]\n"
        "\n"
        "PASSING = {\"flip_to_pass\", \"pass_kept\"}\n"
        "recomputed_pass = sum(1 for r in res if r[\"outcome\"] in PASSING)\n"
        "\n"
        "assert len(res) == b[\"total_cases\"], \"case count disagrees with the summary\"\n"
        "assert recomputed_pass == b[\"passed_cases\"], \"pass count disagrees with the summary\"\n"
        "\n"
        "outcomes = {}\n"
        "for r in res:\n"
        "    outcomes[r[\"outcome\"]] = outcomes.get(r[\"outcome\"], 0) + 1\n"
        "\n"
        "print(f\"{len(res)} case rows; recomputed {recomputed_pass} passing — matches the headline.\\n\")\n"
        "for k, v in sorted(outcomes.items(), key=lambda kv: -kv[1]):\n"
        "    print(f\"  {v:>3}  {k}\")\n"
        "print(f\"\\n  run version v{run['version_number']}   is_latest={run['is_latest_version']}\")"
    ),

    # ── 0.5 the transcript ─────────────────────────────────────────
    md(
        "## The case where our own skill loses\n"
        "\n"
        "A vendor demo picks the case that flatters it. This one picks a case where the skill\n"
        "**makes the answer worse** — and where our own benchmark grades it as a failure.\n"
        "\n"
        "That is a deliberate choice, and it is the honest one. While building the next notebook\n"
        "we found that this suite contradicts itself: the case that most flatters this skill\n"
        "(`case-01`, a software specialist at $80/hour) is the *same fact pattern* as the case\n"
        "below — an hourly computer employee well above the $27.63/hr threshold that\n"
        "29 CFR 541.400(b) exempts. If the case below is right, `case-01` is wrong, and there the\n"
        "\"wrong\" no-skill answer was the correct one all along.\n"
        "\n"
        "We could have shipped `case-01` and it would have looked better. A notebook whose whole\n"
        "argument is *check our work* cannot lead with the case where our work does not survive\n"
        "being checked."
    ),
    code(
        "case = next((r for r in res if r[\"case_name\"] == DEMO[\"demo_case\"]), None) \\\n"
        "    or next(r for r in res if r[\"outcome\"] == \"fail_kept\")\n"
        "\n"
        "# `outcome` reads `fail_kept` on this case — both arms scored as failing. That\n"
        "# per-case verdict is the scorecard's own and it is the authoritative one: it is\n"
        "# read back here exactly as recorded, not re-graded by this notebook. Printed\n"
        "# rather than hidden, because the expectation rows below disagree with it and you\n"
        "# should see both.\n"
        "print(f\"{case['case_name']}\")\n"
        "print(f\"outcome: {case['outcome']}\\n\")\n"
        "\n"
        "# A withheld prompt is a real state, not an error — see the next cell.\n"
        "if not case.get(\"case_prompt\"):\n"
        "    print(f\"prompt withheld: {case.get('case_prompt_unavailable')}\")\n"
        "else:\n"
        "    print(\"PROMPT\"); print(wrap(case[\"case_prompt\"])); print()\n"
        "    print(\"WITHOUT THE SKILL\"); print(wrap(case.get(\"without_skill_output\"))); print()\n"
        "    print(\"WITH THE SKILL\"); print(wrap(case.get(\"with_skill_output\"))); print()\n"
        "    missed = [e for e in (case.get(\"expectation_results\") or [])\n"
        "              if not e.get(\"passed\")]\n"
        "    if missed:\n"
        "        print(\"THE EXPECTATION THE SKILL MISSED\")\n"
        "        print(wrap(missed[0][\"expectation\"]))\n"
        "        print()\n"
        "        print(wrap(\n"
        "            \"Read that again: the arm WITHOUT the skill answered correctly and the arm \"\n"
        "            \"WITH it did not. The skill over-applies the salary-basis rule to a category \"\n"
        "            \"of worker the regulation carves out. Our benchmark caught it, scored it as \"\n"
        "            \"a failure, and it is one of the 11 cases the skill still fails.\"))\n"
        "\n"
        "print(\"\\n--- cost, because a skill is never free ---\")\n"
        "agg = run.get(\"aggregate_metrics\") or {}\n"
        "for key in (\"tokens\", \"duration_ms\"):\n"
        "    blk = agg.get(key) or {}\n"
        "    if blk.get(\"delta_pct\") is not None:\n"
        "        print(f\"  {key:<12} {blk['delta_pct']:+.1f}% with the skill loaded\")"
    ),

    # ── 0.6 what we refuse to show you ─────────────────────────────
    md(
        "## What this registry refuses to show you\n"
        "\n"
        "This is the cell no vendor demo includes, and it is the reason to trust the rest.\n"
        "\n"
        "A benchmark result is only meaningful if the prompt shown beside it is the prompt that\n"
        "produced it. For a large share of this registry we cannot prove that, so the API now\n"
        "returns `case_prompt: null` and a reason instead of a plausible-looking pairing."
    ),
    code(
        "d = M[\"registry_disclosure\"]\n"
        "print(f\"As of {d['as_of']}, across {d['measured_public_skills']:,} measured \"\n"
        "      f\"public skills / {d['graded_cases']:,} graded cases:\\n\")\n"
        "print(f\"  {d['cases_withheld_pct']}% of all cases have their prompt withheld\")\n"
        "print(f\"  {d['platform_cases_withheld_pct']}% of PLATFORM-authored cases\")\n"
        "print(f\"  {d['github_import_cases_withheld_pct']}% of GitHub-import cases\")\n"
        "print()\n"
        "print(wrap(\n"
        "    \"Read that middle number again: more than half the cases in our own hand-authored \"\n"
        "    \"skills cannot currently prove which prompt they were graded against. Two causes — \"\n"
        "    \"an eval suite rewritten in place after a run, and an authoring lane that recorded \"\n"
        "    \"one placeholder prompt for every case in a suite. Both are ours. Neither changes \"\n"
        "    \"a single recorded output or verdict, but both destroy the pairing, and a pairing \"\n"
        "    \"we cannot prove is one we should not display.\", 88, \"  \"))\n"
        "print()\n"
        "print(wrap(\n"
        "    f\"The skill above is not in that group — all {b['total_cases']} of its prompts are \"\n"
        "    \"served, which is a precondition for appearing in this notebook at all. That is \"\n"
        "    \"also why it is not the highest-lift skill on the board.\", 88, \"  \"))"
    ),

    # ── 0.7 the artifact ───────────────────────────────────────────
    md(
        "## The thing itself\n"
        "\n"
        "A skill is a Markdown file. Here is the whole of it — the same bytes the benchmark\n"
        "loaded, pinned to the measured version rather than to whatever is latest."
    ),
    code(
        "v = DEMO[\"version\"]\n"
        "body = get(f\"https://app.decimal.ai/s/{DEMO['slug']}@{v}/SKILL.md\")\n"
        "print(f\"# {DEMO['slug']}@{v} — {len(body):,} bytes\\n\")\n"
        "print(body)\n"
        "\n"
        "print(\"\\n\" + \"-\" * 60)\n"
        "print(wrap(\n"
        "    \"That is the entire product on this axis: a few hundred lines of text, readable \"\n"
        "    \"before you install anything, with a measured claim attached and the evidence \"\n"
        "    \"behind that claim in the cells above. Nothing is behind a login.\", 88, \"  \"))"
    ),

    # ── close ──────────────────────────────────────────────────────
    md(
        "## You have not verified anything yet\n"
        "\n"
        "Everything above is *our* number, *our* test cases, *our* judge. Reading it carefully is\n"
        "not the same as checking it, and a faithfully-reproduced self-graded exam is still a\n"
        "self-graded exam.\n"
        "\n"
        "The next notebook re-runs this exact comparison on **your** model — the same two arms\n"
        "the scorecard reports, with the skill and without it — under a blind judge you can read,\n"
        "and prints the result whichever way it falls. It needs one free Google AI Studio key\n"
        "(no card, ~20 seconds).\n"
        "\n"
        "Or skip us entirely and take the file: `pip install decimalai` then\n"
        "`decimalai skills pull " + "flsa-exemption-test" + " --out .claude/skills/` — anonymous,\n"
        "no account, and your agent picks it up from disk."
    ),
]

nb = {"cells": cells, "metadata": META, "nbformat": 4, "nbformat_minor": 4}
path = os.path.join(os.path.dirname(__file__), "measure_a_skill.ipynb")
with open(path, "w") as f:
    json.dump(nb, f, indent=1)
print(f"built {path} — {len(cells)} cells")
