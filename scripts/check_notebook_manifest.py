#!/usr/bin/env python3
"""Assert that examples/measure-a-skill/manifest.yaml still describes the live registry.

WHY THIS EXISTS. The Tier-0 notebook hardcodes zero numbers: every figure it
prints is fetched live and gate-checked against manifest.yaml first. That design
only pays off if something *else* fails when the pinned skill rots — otherwise
the first thing a skeptical reader sees is a notebook quietly falling through to
its spare, or a cell built around evidence the API no longer serves.

Registry skills churn hard. Of the eight slugs an earlier docs tutorial named,
three now 404 and a fourth fell from a headline number to +4.5 pts. So this
script is the thing that fails, on a daily cron, before a reader gets there.

WHAT IT CHECKS

1.  MANIFEST vs NOTEBOOK (offline). The notebook carries an embedded copy of the
    manifest as its raw.githubusercontent fallback. A YAML edit that doesn't
    reach that copy silently re-introduces the stale figure for every reader who
    hits a raw.github blip.
2.  IDENTITY, not existence. `crisis-response-protocol` still returns HTTP 200
    today and is now a suicide-prevention import from a medical-skills repo. A
    bare existence check goes green on that, so author_display_name and
    source_type are pinned and compared.
3.  EVERY GATE in `gates:` against the live benchmark_summary. A gate key with
    no implementation here is an error, not a no-op — otherwise tightening the
    YAML would silently do nothing.
4.  PROMPTS SERVED. Since the case-binding guard shipped, a run whose per-case
    bindings can't be proven serves `case_prompt: null`; ~47% of
    platform-authored measured skills are in that state. "Has a great lift" no
    longer implies "we can show the evidence", so drifted_cases == 0,
    unrecorded_prompt_cases == 0, and every per-case prompt non-null are all
    asserted.
5.  THE DEMO CASE. `demo_case` must still exist in the latest run, must still have
    the two arms DISAGREEING, and must still carry both arms' output — that one
    cell is the whole argument. Note the check is disagreement, NOT
    outcome == 'flip_to_pass'. The current demo case is deliberately one where
    the SKILL is wrong and the bare model is right; requiring a flip would have
    encoded "always show a case that flatters us" as a CI rule and filed a drift
    issue every day against a case chosen on purpose.
6.  THE SPARE. `fallback_slug` faces the identical battery, because a warm spare
    discovered broken at the moment it is needed is not a spare.
7.  THE DISCLOSURE FIGURES (`--disclosure`, the daily run). A number about how
    much evidence we withhold is exactly the number that must not go stale.
    measured_public_skills and graded_cases are recomputed exactly from the
    listing; the three withheld percentages are estimated from a per-source
    sample, with a jackknife 95% margin added to the tolerance so sampling noise
    can never be reported as drift.

    BE HONEST ABOUT THE SAMPLE. Withholding is near all-or-nothing within a run,
    so 25 skills of the ~57%-withheld platform arm carry a margin around ±13 pts:
    the daily job catches a REGIME change (a backfill lands, a case-binding
    regression ships) and not a 56.6 → 60 slide. `--disclosure-full` censuses
    every measured skill and reports exact figures; that is the mode to use when
    refreshing the numbers in the manifest.

Usage:
    python scripts/check_notebook_manifest.py                 # gates only (fast, ~8 requests)
    python scripts/check_notebook_manifest.py --disclosure    # + recompute the disclosure figures
    python scripts/check_notebook_manifest.py --disclosure --disclosure-full
                                                              # exact, no sampling (~1.5k requests)

Exit codes. FINDINGS BEAT SILENCE — read this before touching the order.
    0  every check that ran passed, AND the network leg actually ran.
    1  real drift — a skill, a gate, a pin or a figure needs a YAML edit. The
       output names the exact key. Reported whether or not the registry
       answered: an OFFLINE finding (manifest.yaml disagreeing with the copy
       embedded in the notebook, a missing identity pin) is a complete verdict
       on its own and does not need the network to be true.
    2  the registry never answered (edge rate-limit, backend restart, DNS) AND
       nothing was found offline either. ONLY then. A network blip must never be
       reported as a rotted skill — but "I could not reach the registry" must
       never be allowed to erase what this script already knows.

    That precedence is the whole point of the check ledger printed at the end of
    every run, on every exit path. Until 2026-08-10 this script ran its offline
    checks first, collected their findings, and then discarded them unprinted
    the moment a later request timed out. The workflow downgrades exit 2 to
    success — so a pull request that genuinely broke the manifest went GREEN
    whenever the edge limiter happened to be throttling, and the output said
    only "the check DID NOT RUN". Reproduced by pointing API at a dead port with
    a planted manifest/FALLBACK mismatch; it returned 2 and printed nothing
    about the mismatch. A check that cannot say WHICH checks it performed can
    always be mistaken for one that passed.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "examples" / "measure-a-skill" / "manifest.yaml"
DEFAULT_NOTEBOOK = REPO_ROOT / "examples" / "measure-a-skill" / "measure_a_skill.ipynb"

API = "https://api.decimal.ai/api/v1"
# Measured 2026-08-10 against prod: `/registry/skills?measured=only&limit=100`
# answers in 17s warm and has been seen past 60s cold — the page size barely
# matters (limit=10 took 43s, limit=25 took 22s), so the cost is the query, not
# the payload. A 60s timeout turns that endpoint into a permanent "unreachable".
TIMEOUT_S = 150
SKILL_MD_URL = "https://app.decimal.ai/s/{slug}@{version}/SKILL.md"
UA = "decimalai-notebook-freshness/1.0 (+https://github.com/decimal-labs/decimalai-python)"

# Defaults used when manifest.yaml declares no `tolerance:` block, so the check
# still runs (with stated numbers) against an older manifest.
DEFAULT_TOLERANCE = {
    "counts_drift_pct": 15.0,
    "withheld_drift_pts": 3.0,
    "disclosure_sample_per_source": 25,
    "as_of_max_age_days": 45,
}


class Unreachable(RuntimeError):
    """The registry never gave a definitive answer.

    On its own this is exit 2. It is NOT a licence to drop findings the run
    already made — see the exit-code note in the module docstring.
    """


# Printed at the end of every run. A cron whose wall time quietly triples is
# usually being throttled, and without this the only symptom is a slow job.
STATS = {"requests": 0, "rate_limited": 0, "retried": 0, "backoff_s": 0.0}


# ──────────────────────────────────────────────────────────────────────
# Which checks actually ran
# ──────────────────────────────────────────────────────────────────────

OFFLINE, NETWORK = "offline", "network"


class Ledger:
    """Every check this run intended to make, and what became of each one.

    Printed on EVERY exit path, including the unreachable one. "0 checks
    performed, exit 0" and "7 checks performed, exit 0" are the same line of
    output without this, and that ambiguity is the entire failure mode: a green
    tick that means "I asserted nothing" is worse than a red one.

    Checks are declared with `plan()` BEFORE they are attempted and flipped by
    `done()` when they complete, so anything an exception cut short is reported
    as NOT REACHED for free — there is no way to abort past a check and leave it
    looking performed.
    """

    RAN = "ran"
    SKIPPED = "skipped"
    NOT_REACHED = "NOT REACHED"
    NOT_REQUESTED = "not requested"

    _MARK = {RAN: "✓", SKIPPED: "–", NOT_REACHED: "!", NOT_REQUESTED: "·"}

    def __init__(self):
        self._rows = []  # [name, kind, state, detail], in declaration order

    def _row(self, name):
        for row in self._rows:
            if row[0] == name:
                return row
        raise KeyError(f"check {name!r} was never planned")

    def plan(self, name, kind):
        self._rows.append([name, kind, self.NOT_REACHED, ""])
        return name

    def done(self, name, detail=""):
        row = self._row(name)
        row[2], row[3] = self.RAN, detail

    def skip(self, name, why):
        row = self._row(name)
        row[2], row[3] = self.SKIPPED, why

    def not_requested(self, name, why):
        row = self._row(name)
        row[2], row[3] = self.NOT_REQUESTED, why

    def count(self, state):
        return sum(1 for r in self._rows if r[2] == state)

    @property
    def performed(self):
        return self.count(self.RAN)

    @property
    def applicable(self):
        """Checks that were meant to happen — a `--disclosure`-only check on a
        gates-only run is not a hole, so it is not in the denominator."""
        return sum(1 for r in self._rows if r[2] != self.NOT_REQUESTED)

    def ran_any(self, kind):
        return any(r[2] == self.RAN for r in self._rows if r[1] == kind)

    def render(self):
        lines = [f"CHECKS PERFORMED — {self.performed} of {self.applicable} applicable:"]
        for name, kind, state, detail in self._rows:
            suffix = f" — {detail}" if detail else ""
            lines.append(f"  {self._MARK[state]} [{kind:7}] {state:<11} {name}{suffix}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# HTTP
# ──────────────────────────────────────────────────────────────────────

def fetch(url, params=None, *, retries=4, pace=0.0, as_json=True):
    """GET with backoff. Returns (payload, status); a 404 returns (None, 404).

    Everything that is not a definitive answer is retried and then raised as
    Unreachable. The public registry rate-limits anonymous traffic at the edge
    with a `text/html` "Rate exceeded." body, HTTP 429 and no Retry-After, and
    the API can restart under a request — so 429/5xx/timeout/DNS are all
    "ask again later", not "this skill is gone".
    """
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    if pace:
        time.sleep(pace)
    last = ""
    for attempt in range(retries + 1):
        STATS["requests"] += 1
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                raw = resp.read().decode("utf-8", "replace")
                if not as_json:
                    return raw, resp.status
                try:
                    return json.loads(raw), resp.status
                except ValueError:
                    # A JSON endpoint answering with HTML is an edge error page,
                    # not a skill that changed shape.
                    last = f"HTTP {resp.status} with an unparseable body"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None, 404
            if e.code == 429:
                STATS["rate_limited"] += 1
            last = f"HTTP {e.code}"
        except (urllib.error.URLError, OSError) as e:
            last = f"{type(e).__name__}: {e}"
        if attempt < retries:
            # 4s, 12s, 28s, 60s (+jitter): long enough to outlast a Cloud Run
            # cold start and most of an edge rate-limit window.
            delay = min(60, 4 * (2**attempt)) + random.uniform(0, 2)
            STATS["retried"] += 1
            STATS["backoff_s"] += delay
            time.sleep(delay)
    raise Unreachable(f"{url} — {last} after {retries + 1} attempts")


def _stats_line():
    return (
        f"{STATS['requests']} request(s), {STATS['rate_limited']} rate-limited, "
        f"{STATS['retried']} retried, {STATS['backoff_s']:.0f}s spent backing off"
    )


# ──────────────────────────────────────────────────────────────────────
# Manifest ↔ notebook consistency (offline)
# ──────────────────────────────────────────────────────────────────────

def embedded_fallback(notebook_path):
    """The `FALLBACK = {...}` literal the notebook uses when raw.github is down.

    Returns the parsed dict, or None if the notebook has no such literal (which
    is itself reported — the fallback is what keeps a raw.github outage from
    turning into a KeyError on a 404 page parsed as YAML).
    """
    try:
        nb = json.loads(Path(notebook_path).read_text())
    except ValueError as e:
        raise SystemExit(f"✗ {notebook_path} is not valid .ipynb JSON: {e}")
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if "FALLBACK = {" not in src:
            continue
        body = src[src.index("FALLBACK = {") + len("FALLBACK = ") :]
        # Trim to the matching close brace: the literal is written flush-left,
        # so the first line that is exactly "}" ends it.
        lines, depth = [], 0
        for line in body.splitlines():
            lines.append(line)
            depth += line.count("{") - line.count("}")
            if depth == 0:
                break
        try:
            return ast.literal_eval("\n".join(lines))
        except (ValueError, SyntaxError) as e:
            raise SystemExit(f"cannot parse the notebook's FALLBACK literal: {e}")
    return None


def check_embedded_copy(manifest, notebook_path):
    """Compare the notebook's embedded manifest copy against manifest.yaml.

    Only keys the embedded copy actually carries are compared — it is
    deliberately a subset. But every key it does carry must agree, because a
    reader who hits a raw.github blip sees these values and nothing warns them.
    """
    problems = []
    fb = embedded_fallback(notebook_path)
    if fb is None:
        return [
            f"{Path(notebook_path).name}: no `FALLBACK = {{...}}` literal found — the "
            "notebook has no offline copy of the manifest; a raw.github blip becomes a "
            "traceback. Restore it in _build.py and rebuild."
        ]
    for section, values in fb.items():
        live = manifest.get(section)
        if not isinstance(values, dict) or not isinstance(live, dict):
            if values != live:
                problems.append(f"{section}: embedded copy {values!r} != manifest {live!r}")
            continue
        for key, embedded in values.items():
            if key not in live:
                problems.append(
                    f"{section}.{key}: in the notebook's FALLBACK but not in manifest.yaml"
                )
            elif live[key] != embedded:
                problems.append(
                    f"{section}.{key}: manifest says {live[key]!r}, the notebook's embedded "
                    f"FALLBACK still says {embedded!r} — update FALLBACK in "
                    f"examples/measure-a-skill/_build.py and re-run it"
                )
    return problems


# ──────────────────────────────────────────────────────────────────────
# Gates
# ──────────────────────────────────────────────────────────────────────

def _g_min_delta(expected, b, run):
    live = b.get("pass_rate_delta_pts")
    if live is None or float(live) < float(expected):
        return f"pass_rate_delta_pts is {live}, gates.min_delta_pts wants >= {expected}"
    return None


def _g_min_cases(expected, b, run):
    live = b.get("total_cases")
    if live is None or int(live) < int(expected):
        return f"total_cases is {live}, gates.min_cases wants >= {expected}"
    return None


def _g_grading(expected, b, run):
    if b.get("grading_method") != expected:
        return (
            f"grading_method is {b.get('grading_method')!r}, "
            f"gates.require_grading_method wants {expected!r}"
        )
    return None


def _g_prompts_served(expected, b, run):
    """The load-bearing one: can we actually SHOW the evidence?

    Three separate ways to lose the pairing, all checked: the run's own drift
    counter, its unrecorded-prompt counter, and — belt and braces, because the
    counters are a summary and the notebook renders the rows — any per-case
    `case_prompt` that came back null.
    """
    if not expected:
        return None
    drifted = run.get("drifted_cases")
    unrecorded = run.get("unrecorded_prompt_cases")
    if drifted or unrecorded:
        return (
            f"latest run has drifted_cases={drifted}, unrecorded_prompt_cases={unrecorded}; "
            "gates.require_all_prompts_served wants 0 and 0 — the notebook cannot show the "
            "prompt beside the verdict for this skill any more"
        )
    withheld = [r.get("case_name") for r in run.get("results") or [] if not r.get("case_prompt")]
    if withheld:
        return (
            f"{len(withheld)} case(s) serve case_prompt: null ({', '.join(map(str, withheld[:4]))}"
            f"{'…' if len(withheld) > 4 else ''}) despite zero counters; "
            "gates.require_all_prompts_served wants every prompt served"
        )
    return None


# NO never_hurt / regressed_cases GATE, deliberately. Both are derived under the
# headline's calibration rule, which sets aside the very expectations a skill made
# worse — so `never_hurt: true` can be true of a skill that demonstrably turned a
# right answer wrong (live: flsa-exemption-test case-21 and case-22, where the
# no-skill arm passes both expectations and the skill fails both). Gating on a
# badge that can contradict the rows underneath it is worse than not gating.
# What the notebook shows instead is the pair of pass rates, which needs no
# interpretation: N of M cases passed with the skill, K of M without.
GATE_CHECKS = {
    "min_delta_pts": _g_min_delta,
    "min_cases": _g_min_cases,
    "require_grading_method": _g_grading,
    "require_all_prompts_served": _g_prompts_served,
}


# ──────────────────────────────────────────────────────────────────────
# Per-skill assertions
# ──────────────────────────────────────────────────────────────────────

def check_skill(slug, *, role, key_prefix, pins, gates, pace, spare_hint, key_sep="."):
    """Assert one skill against its identity pins and every declared gate.

    `role` is prose for the human ("demo skill" / "fallback"); `key_prefix` is
    the YAML path to name in an error, so the fix is a lookup, not a search.
    Returns (problems, skill_record, latest_run).
    """
    problems = []
    skill, status = fetch(f"{API}/registry/skills/{slug}", pace=pace)
    if skill is None:
        return (
            [
                f"{key_prefix}{key_sep}slug: '{slug}' does not resolve (HTTP {status}) — the {role} is "
                f"gone from the registry. {spare_hint}"
            ],
            None,
            None,
        )

    # Identity before anything else: a slug that still resolves is not the skill
    # you meant.
    for field, expected in pins.items():
        live = skill.get(field)
        if live != expected:
            problems.append(
                f"{key_prefix}{key_sep}{field}: manifest pins {expected!r}, the live '{slug}' says "
                f"{live!r} — the slug resolves but it is not the same skill"
            )

    b = skill.get("benchmark_summary") or {}
    if not b:
        problems.append(
            f"{key_prefix}{key_sep}slug: '{slug}' has no benchmark_summary — it is no longer a "
            "measured skill, so every gate below is unevaluable"
        )
        return problems, skill, None

    bench, status = fetch(f"{API}/registry/skills/{slug}/benchmark", pace=pace)
    run = (bench or {}).get("latest_run") or {}
    if not run:
        problems.append(
            f"{key_prefix}{key_sep}slug: '{slug}' has a benchmark_summary but /benchmark returned no "
            f"latest_run (HTTP {status}) — the notebook's per-case cells have nothing to read"
        )
        return problems, skill, None

    for key, expected in sorted(gates.items()):
        check = GATE_CHECKS.get(key)
        if check is None:
            problems.append(
                f"gates.{key}: declared in manifest.yaml but no check implements it — add one to "
                f"GATE_CHECKS in {Path(__file__).name} or the gate is decorative"
            )
            continue
        failure = check(expected, b, run)
        if failure:
            problems.append(f"{role} '{slug}': {failure}")

    # The summary and the rows must describe the same run — the notebook
    # recomputes the headline from the rows in cell 0.4 and asserts on it.
    if len(run.get("results") or []) != b.get("total_cases"):
        problems.append(
            f"{role} '{slug}': benchmark_summary.total_cases={b.get('total_cases')} but "
            f"/benchmark returned {len(run.get('results') or [])} rows — cell 0.4's "
            "recompute assertion will fail in the reader's face"
        )

    if not any(r.get("outcome") == "flip_to_pass" for r in run.get("results") or []):
        problems.append(
            f"{role} '{slug}': no case has outcome 'flip_to_pass' — cell 0.5 has no "
            "with-vs-without contrast to show, and its fallback `next(...)` raises "
            "StopIteration"
        )
    return problems, skill, run


def check_demo_extras(manifest, slug, run, *, pace):
    """The pins that only apply to the demo skill: version and demo_case."""
    problems = []
    demo = manifest["demo_skill"]
    version = demo.get("version")

    # The notebook prints the body of `version` and calls it "the same bytes the
    # benchmark loaded". If a v2 is published and re-benchmarked, that sentence
    # becomes false without any 404 anywhere.
    if run and run.get("version_number") != version:
        problems.append(
            f"demo_skill.version: manifest pins v{version}, but the latest benchmark run is of "
            f"v{run.get('version_number')} — cell 0.7 would print a body the headline was not "
            "measured on"
        )

    body, status = fetch(SKILL_MD_URL.format(slug=slug, version=version), pace=pace, as_json=False)
    if body is None:
        problems.append(
            f"demo_skill.version: {SKILL_MD_URL.format(slug=slug, version=version)} returned "
            f"HTTP {status} — cell 0.7 has no artifact to print"
        )
    elif len(body.strip()) < 200:
        problems.append(
            f"demo_skill.version: the SKILL.md body for {slug}@{version} is only "
            f"{len(body.strip())} bytes — that is a login wall or an error page, not the skill"
        )
    elif body.lstrip()[:1] == "<" or "<!DOCTYPE html" in body[:500]:
        # HTTP 200 lies in front of an auth wall: a protected path serves the
        # sign-in PAGE with a 200, so the status code proves nothing and the body
        # is the only evidence. This one has cost this org five separate outages
        # of exactly this shape.
        problems.append(
            f"demo_skill.version: {SKILL_MD_URL.format(slug=slug, version=version)} answered 200 "
            "with HTML — the raw body is behind an auth wall or an error page, and cell 0.7 "
            "would print a sign-in page as 'the thing itself'"
        )

    name = demo.get("demo_case")
    case = next((r for r in (run or {}).get("results") or [] if r.get("case_name") == name), None)
    if case is None:
        available = [r.get("case_name") for r in (run or {}).get("results") or []]
        flips = [
            r.get("case_name")
            for r in (run or {}).get("results") or []
            if r.get("outcome") == "flip_to_pass"
        ]
        problems.append(
            f"demo_skill.demo_case: '{name}' is not in the latest run ({len(available)} cases). "
            f"Cases whose arms differ: {', '.join(map(str, flips[:5])) or 'none available'}"
        )
        return problems

    # The cell needs the two arms to DISAGREE. Which one is right is an editorial
    # choice, and the current demo_case is deliberately one where the SKILL is
    # wrong and the bare model is right — see the long note in manifest.yaml.
    #
    # This used to require outcome == 'flip_to_pass', which quietly encoded
    # "always show a case that flatters us" as a CI rule. It would have filed a
    # drift issue every day against a demo case chosen on purpose.
    with_out = (case.get("with_skill_output") or "").strip()
    without_out = (case.get("without_skill_output") or "").strip()
    if with_out and without_out and with_out == without_out:
        problems.append(
            f"demo_skill.demo_case: '{name}' now has both arms producing the SAME output, so "
            "the cell shows two arms that agree and demonstrates nothing. Pick a case whose "
            "arms differ — either direction is fine."
        )
    if not case.get("case_prompt"):
        problems.append(
            f"demo_skill.demo_case: '{name}' serves case_prompt: null "
            f"({case.get('case_prompt_unavailable')!r}) — cell 0.5 prints 'prompt withheld' "
            "instead of the transcript the whole notebook builds to"
        )
    for field in ("with_skill_output", "without_skill_output"):
        if not (case.get(field) or "").strip():
            problems.append(
                f"demo_skill.demo_case: '{name}' has an empty {field} — cell 0.5 prints a blank arm"
            )
    return problems


# ──────────────────────────────────────────────────────────────────────
# Disclosure figures
# ──────────────────────────────────────────────────────────────────────

def walk_measured(pace, max_pages=60):
    """Every measured public skill, as (slug, source_type, total_cases).

    Cursor-paginated; `total_hint` comes back null on this listing, so the count
    is only exact by walking it.

    Paced and retried harder than everything else in this file. Measured
    2026-08-10: the second page of the walk was 429'd through four backoffs
    (~150s) and the run gave up — this endpoint is the expensive one and the
    edge limiter is unforgiving about it. Waiting minutes for a page is fine on
    a daily cron; abandoning the whole disclosure recompute is not.
    """
    rows, cursor, pages = [], None, 0
    while pages < max_pages:
        params = {"measured": "only", "limit": 100}
        if cursor:
            params["cursor"] = cursor
        page, _ = fetch(f"{API}/registry/skills", params, pace=max(pace, 3.0), retries=6)
        page = page or {}
        for item in page.get("items") or []:
            b = item.get("benchmark_summary") or {}
            rows.append(
                (
                    item.get("url_slug"),
                    item.get("source_type") or "unknown",
                    int(b.get("total_cases") or 0),
                )
            )
        pages += 1
        cursor = page.get("next_cursor")
        if not page.get("has_next") or not cursor:
            return rows, True
    return rows, False  # hit the page cap — counts are a floor, not exact


def jackknife_rate(samples):
    """(rate, standard error) for withheld_cases / total_cases over skills.

    Sampling is by SKILL but the figure is per CASE, so the naive binomial
    standard error understates it badly — withholding is close to all-or-nothing
    within a run. A leave-one-skill-out jackknife over the ratio estimator gets
    the clustering right without pretending to a formula this file cannot show
    its work for.
    """
    cases = sum(c for c, _ in samples)
    withheld = sum(w for _, w in samples)
    if not cases:
        return None, None
    rate = withheld / cases
    n = len(samples)
    if n < 2:
        return rate, None
    partials = []
    for i in range(n):
        c = cases - samples[i][0]
        w = withheld - samples[i][1]
        partials.append(w / c if c else rate)
    mean = sum(partials) / n
    var = (n - 1) / n * sum((p - mean) ** 2 for p in partials)
    return rate, math.sqrt(var)


def stratum_margin(censused, se, n):
    """95% margin on a stratum's withheld rate, in rate units (0–1).

    A census has none. Otherwise it is 1.96·SE — floored by the rule of three,
    because a jackknife over a DEGENERATE sample (every sampled skill withheld
    everything, or none of them did) returns exactly zero standard error. That
    zero is an artifact of the estimator, not knowledge: 25 skills that all serve
    their prompts are still consistent with ~12% of the stratum withholding.
    Reporting it as "exact" would be the one lie this whole file exists to
    prevent.
    """
    if censused:
        return 0.0
    if not n:
        return 1.0
    return max(1.96 * (se or 0.0), 3.0 / n)


def measure_disclosure(rows, *, sample_per_source, seed, pace, full):
    """Recompute the registry_disclosure figures from the live registry.

    Exact for the two counts (they come straight off the listing). Sampled for
    the three percentages, because the only place a withheld prompt is visible
    is the per-skill /benchmark payload (~50KB each) and there are ~1,400 of
    them behind an anonymous edge rate limit.
    """
    by_source = {}
    for slug, source, cases in rows:
        by_source.setdefault(source, []).append((slug, cases))

    total_cases = sum(c for _, _, c in rows)
    strata = {}
    rng = random.Random(seed)
    for source, entries in sorted(by_source.items()):
        pool = [e for e in entries if e[1] > 0]
        picked = pool if full or len(pool) <= sample_per_source else rng.sample(pool, sample_per_source)
        samples = []
        for slug, _ in picked:
            bench, _status = fetch(f"{API}/registry/skills/{slug}/benchmark", pace=pace)
            results = ((bench or {}).get("latest_run") or {}).get("results") or []
            if not results:
                continue
            samples.append((len(results), sum(1 for r in results if not r.get("case_prompt"))))
        rate, se = jackknife_rate(samples)
        # A census of the stratum has no sampling error. Every skill in the pool
        # must have ANSWERED, not merely been picked — a fetch that came back
        # empty leaves a real hole.
        censused = bool(pool) and len(samples) == len(pool)
        strata[source] = {
            "skills": len(entries),
            "cases": sum(c for _, c in entries),
            "sampled_skills": len(samples),
            "rate": rate,
            "censused": censused,
            "margin": stratum_margin(censused, se, len(samples)),
        }

    # Overall = the strata's exact case weights times their sampled rates, which
    # is a tighter estimate than sampling the whole population blind: the
    # platform and github-import arms differ by ~50 points, and their sizes are
    # known exactly.
    #
    # Weights are renormalised over the strata we actually measured. Dividing by
    # the full population instead would silently count an unmeasured stratum as
    # 0% withheld — biasing an honesty figure downward, which is the one
    # direction it must never move by accident.
    measured_cases = sum(s["cases"] for s in strata.values() if s["rate"] is not None)
    weighted = [
        (s["cases"] / measured_cases, s) for s in strata.values() if s["rate"] is not None
    ] if measured_cases else []
    overall = sum(w * s["rate"] for w, s in weighted) if weighted else None
    overall_margin = (
        math.sqrt(sum((w * s["margin"]) ** 2 for w, s in weighted)) if weighted else None
    )
    return {
        "measured_public_skills": len(rows),
        "graded_cases": total_cases,
        "coverage_pct": 100.0 * measured_cases / total_cases if total_cases else 0.0,
        "strata": strata,
        "overall_rate": overall,
        "overall_margin": overall_margin,
    }


def compare_disclosure(declared, live, tolerance):
    """Drift messages. Sampling margin is added to the tolerance, never hidden."""
    problems, notes = [], []
    tol_pct = float(tolerance["counts_drift_pct"])
    for key, live_val in (
        ("measured_public_skills", live["measured_public_skills"]),
        ("graded_cases", live["graded_cases"]),
    ):
        old = declared.get(key)
        if not old:
            problems.append(f"registry_disclosure.{key}: missing from manifest.yaml")
            continue
        signed = 100.0 * (live_val - old) / old  # signed for the human, absolute for the gate
        line = f"registry_disclosure.{key}: manifest {old:,} · live {live_val:,} ({signed:+.1f}%)"
        (problems if abs(signed) > tol_pct else notes).append(
            line + (f" — over the {tol_pct:.0f}% tolerance" if abs(signed) > tol_pct else "")
        )

    tol_pts = float(tolerance["withheld_drift_pts"])
    pairs = [
        ("cases_withheld_pct", live["overall_rate"], live["overall_margin"], "all sources"),
        (
            "platform_cases_withheld_pct",
            (live["strata"].get("platform") or {}).get("rate"),
            (live["strata"].get("platform") or {}).get("margin"),
            "platform",
        ),
        (
            "github_import_cases_withheld_pct",
            (live["strata"].get("github_import") or {}).get("rate"),
            (live["strata"].get("github_import") or {}).get("margin"),
            "github_import",
        ),
    ]
    for key, rate, margin_rate, label in pairs:
        old = declared.get(key)
        if rate is None:
            problems.append(
                f"registry_disclosure.{key}: could not measure the {label} arm at all — no "
                "sampled skill returned per-case results"
            )
            continue
        live_pct = 100.0 * rate
        margin = 100.0 * (margin_rate or 0.0)
        drift = abs(live_pct - (old if old is not None else 0.0))
        line = (
            f"registry_disclosure.{key}: manifest {old} · live {live_pct:.1f}"
            f"{f' ±{margin:.1f}' if margin else ' (censused, exact)'} pts"
        )
        if old is None:
            problems.append(f"registry_disclosure.{key}: missing from manifest.yaml — live {live_pct:.1f}")
        elif drift > tol_pts + margin:
            problems.append(
                line + f" — drifted {drift:.1f} pts, past the {tol_pts:.0f} pt tolerance"
                f"{f' + {margin:.1f} pt sampling margin' if margin else ''}"
            )
        else:
            notes.append(line)
    return problems, notes


def paste_block(declared, live, today):
    """The YAML to paste, so a failure is a copy, not an investigation."""
    strata = live["strata"]
    lines = [
        "registry_disclosure:",
        f"  measured_public_skills: {live['measured_public_skills']}   # exact",
        f"  graded_cases: {live['graded_cases']}   # exact",
    ]
    sampled = False
    for key, source in (
        ("cases_withheld_pct", None),
        ("platform_cases_withheld_pct", "platform"),
        ("github_import_cases_withheld_pct", "github_import"),
    ):
        s = strata.get(source) or {}
        rate = live["overall_rate"] if source is None else s.get("rate")
        margin = live["overall_margin"] if source is None else s.get("margin")
        if rate is None:
            lines.append(f"  {key}: {declared.get(key)}   # UNMEASURED this run — left as declared")
            continue
        if margin:
            sampled = True
            note = f"   # ±{100.0 * margin:.1f} pts, sampled"
        else:
            note = "   # exact (censused)"
        lines.append(f"  {key}: {100.0 * rate:.1f}{note}")
    lines.append(f'  as_of: "{today}"')
    if sampled:
        lines.append(
            "  # ^ the ±-marked figures are ESTIMATES from a sample. Re-run with "
            "--disclosure-full before pasting them."
        )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument(
        "--disclosure",
        action="store_true",
        help="also recompute the registry_disclosure figures (the daily run; adds ~70 requests)",
    )
    parser.add_argument(
        "--disclosure-full",
        action="store_true",
        help="with --disclosure: fetch every measured skill's benchmark instead of sampling "
        "(exact percentages, ~1.5k requests, slow)",
    )
    parser.add_argument("--sample", type=int, default=None, help="skills sampled per source_type")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="sampling seed (default: today's date, so a failure is reproducible with --seed)",
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=1.0,
        help="seconds to wait before each request. The public registry rate-limits anonymous "
        "traffic, and one second between requests is cheaper than the 4/12/28/60s backoff that "
        "follows a 429",
    )
    args = parser.parse_args(argv)

    try:
        import yaml
    except ImportError:
        print("✗ PyYAML is required: pip install pyyaml")
        return 1

    if not args.manifest.exists():
        print(f"✗ manifest not found: {args.manifest}")
        return 1
    manifest = yaml.safe_load(args.manifest.read_text())
    demo = manifest.get("demo_skill") or {}
    gates = manifest.get("gates") or {}
    if not demo.get("slug") or not gates:
        print(f"✗ {args.manifest}: `demo_skill.slug` and `gates:` are both required")
        return 1

    tolerance = dict(DEFAULT_TOLERANCE)
    tolerance.update((manifest.get("registry_disclosure") or {}).get("tolerance") or {})
    today = time.strftime("%Y-%m-%d", time.gmtime())
    seed = args.seed if args.seed is not None else int(today.replace("-", ""))

    # Findings are kept in two piles on purpose. An OFFLINE finding is true
    # without the network and stays true when the network dies mid-run; a
    # NETWORK finding is one the registry answered definitively (a 404, a pin
    # that came back different) before it stopped answering. Neither is ever
    # discarded — only "I never got an answer" is, and that is what exit 2 is.
    offline_problems, online_problems, notes = [], [], []
    ledger = Ledger()

    c_pins = ledger.plan("demo_skill identity pins are declared at all", OFFLINE)
    c_embed = ledger.plan("manifest.yaml == the notebook's embedded FALLBACK copy", OFFLINE)
    c_spare_declared = ledger.plan("a warm spare (fallback_slug) is declared", OFFLINE)
    c_demo = ledger.plan(f"demo skill '{demo['slug']}': identity pins + every gate", NETWORK)
    c_demo_extra = ledger.plan(
        f"demo skill '{demo['slug']}': pinned version, SKILL.md body, demo_case", NETWORK
    )
    c_spare = ledger.plan(
        f"spare '{manifest.get('fallback_slug')}': identity pins + every gate", NETWORK
    )
    c_listing = ledger.plan("cell 0.2: the measured listing still has gate survivors", NETWORK)
    c_disclosure = ledger.plan("registry_disclosure figures recomputed from the registry", NETWORK)
    if not args.disclosure:
        # Not a hole — a PR run deliberately skips the ~100-request walk. Marked
        # so it drops out of the "N of M performed" denominator instead of
        # looking like a check that failed to happen.
        ledger.not_requested(c_disclosure, "pass --disclosure (the daily cron does)")

    # 0 ─ the pins have to exist before they can be checked. A manifest missing
    #     an identity pin degrades this whole script to an existence check, which
    #     is precisely the thing that goes green on a slug that is now somebody
    #     else's medical-skills import.
    for key in ("author_display_name", "source_type", "version", "demo_case"):
        if demo.get(key) in (None, ""):
            offline_problems.append(
                f"demo_skill.{key}: missing from manifest.yaml — without it this check cannot "
                "tell 'the slug resolves' from 'it is still the same skill'"
            )
    ledger.done(c_pins)

    # 1 ─ offline: does the notebook's embedded copy still agree with the YAML?
    if args.notebook.exists():
        offline_problems += check_embedded_copy(manifest, args.notebook)
        ledger.done(c_embed, args.notebook.name)
    else:
        ledger.skip(c_embed, f"no notebook at {args.notebook}")

    # 1b ─ also offline: the spare has to be NAMED before it can be checked.
    #      (Was inside the network block, where an unreachable registry hid it.)
    if manifest.get("fallback_slug"):
        ledger.done(c_spare_declared, str(manifest["fallback_slug"]))
    else:
        offline_problems.append(
            "fallback_slug: missing — the notebook has no spare to fall through to"
        )
        ledger.done(c_spare_declared)
        ledger.skip(c_spare, "no fallback_slug declared")

    # `Unreachable` is caught, not returned on: the findings above are already
    # verdicts, and the reporting block at the bottom owns the exit code.
    unreachable = None
    try:
        # 2 ─ the demo skill: identity, gates, prompts served, the demo case.
        demo_problems, _demo_record, run = check_skill(
            demo["slug"],
            role="demo skill",
            key_prefix="demo_skill",
            pins={
                k: v
                for k, v in {
                    "author_display_name": demo.get("author_display_name"),
                    "source_type": demo.get("source_type"),
                }.items()
                if v is not None  # absence is already reported above; don't double-count
            },
            gates=gates,
            pace=args.pace,
            spare_hint=(
                f"Promote the spare: set demo_skill.slug to '{manifest.get('fallback_slug')}' "
                "(and its author/source/version/demo_case pins) or pick a new one."
            ),
        )
        online_problems += demo_problems
        ledger.done(c_demo, f"{len(gates)} gate(s) evaluated")
        if run:
            online_problems += check_demo_extras(manifest, demo["slug"], run, pace=args.pace)
            ledger.done(c_demo_extra)
        else:
            ledger.skip(c_demo_extra, "the demo skill has no latest benchmark run to read")
        # Not a gate — the notebook prints whatever the live scan says, which is
        # the honest behaviour. But a demo skill that has picked up a safety flag
        # is a bad first impression on a page about trust, so say it out loud.
        if _demo_record and _demo_record.get("safety_status") != "clean":
            notes.append(
                f"⚠ demo skill safety_status is {_demo_record.get('safety_status')!r}, not "
                "'clean' — cell 0.3 prints that verbatim to the reader"
            )

        # 3 ─ the spare, facing the same battery.
        if manifest.get("fallback_slug"):
            fb_problems, _fb, _fb_run = check_skill(
                manifest["fallback_slug"],
                role="fallback",
                key_prefix="fallback",
                # The spare's pins are flat top-level keys (`fallback_slug`,
                # `fallback_author_display_name`), not a nested block — so the
                # separator changes, and every message still names a key that
                # exists in the file.
                key_sep="_",
                pins={
                    k: v
                    for k, v in {
                        "author_display_name": manifest.get("fallback_author_display_name"),
                        "source_type": manifest.get("fallback_source_type"),
                    }.items()
                    if v is not None
                },
                gates=gates,
                pace=args.pace,
                spare_hint="There is now no warm spare — pick a new fallback_slug.",
            )
            online_problems += fb_problems
            ledger.done(c_spare, f"{len(gates)} gate(s) evaluated")

        # 4 ─ cell 0.2 renders the gate as a population filter. Zero survivors
        #     makes the notebook's opening move read as a broken query.
        listing, _ = fetch(
            f"{API}/registry/skills",
            {"measured": "only", "sort": "lift", "limit": 100},
            pace=args.pace,
        )
        items = (listing or {}).get("items") or []
        survivors = [
            s
            for s in items
            if (s.get("benchmark_summary") or {}).get("grading_method")
            == gates.get("require_grading_method")
            and ((s.get("benchmark_summary") or {}).get("total_cases") or 0) >= gates.get("min_cases", 0)
            and (s.get("benchmark_summary") or {}).get("never_hurt") is True
            and ((s.get("benchmark_summary") or {}).get("pass_rate_delta_pts") or 0)
            >= gates.get("min_delta_pts", 0)
        ]
        if not items:
            online_problems.append(
                "the measured listing returned zero skills — cell 0.2 prints '0 skills in' and "
                "the notebook's opening move looks broken"
            )
        elif not survivors:
            online_problems.append(
                f"cell 0.2: none of the top {len(items)} measured skills survive the gates — the "
                "filter reads as broken rather than strict. Loosen gates.* or re-check the sort."
            )
        else:
            notes.append(f"cell 0.2: {len(survivors)}/{len(items)} listed skills survive the gates")
        ledger.done(c_listing, f"{len(survivors)}/{len(items)} survive")

        # 5 ─ the disclosure figures (daily run only).
        if args.disclosure:
            rows, complete = walk_measured(args.pace)
            if not complete:
                notes.append("measured listing hit the page cap — counts below are a floor")
            live = measure_disclosure(
                rows,
                sample_per_source=args.sample or int(tolerance["disclosure_sample_per_source"]),
                seed=seed,
                pace=args.pace,
                full=args.disclosure_full,
            )
            declared = manifest.get("registry_disclosure") or {}
            d_problems, d_notes = compare_disclosure(declared, live, tolerance)
            notes += d_notes
            for source, s in sorted(live["strata"].items()):
                if s["rate"] is None:
                    notes.append(
                        f"  {source}: {s['skills']} skills / {s['cases']:,} cases — NOT measured "
                        "(no sampled skill returned per-case results); excluded from the overall"
                    )
                else:
                    how = (
                        "censused"
                        if s["censused"]
                        else f"{s['sampled_skills']} sampled, ±{100 * s['margin']:.1f} pts"
                    )
                    notes.append(
                        f"  {source}: {s['skills']} skills / {s['cases']:,} cases → "
                        f"{100 * s['rate']:.1f}% withheld ({how})"
                    )
            if live["coverage_pct"] < 99.9:
                notes.append(
                    f"  ⚠ the overall figure covers {live['coverage_pct']:.1f}% of graded cases — "
                    "the rest sit in strata nothing answered for"
                )
            age_days = None
            if declared.get("as_of"):
                try:
                    as_of = time.strptime(str(declared["as_of"]), "%Y-%m-%d")
                    age_days = (time.time() - time.mktime(as_of)) / 86400
                except ValueError:
                    d_problems.append(
                        f"registry_disclosure.as_of: {declared['as_of']!r} is not YYYY-MM-DD"
                    )
            if age_days and age_days > float(tolerance["as_of_max_age_days"]):
                notes.append(
                    f"⚠ registry_disclosure.as_of is {age_days:.0f} days old ({declared['as_of']}). "
                    f"The figures still hold, but the notebook prints that date to the reader — "
                    f"bump it to {today}."
                )
            if d_problems:
                print("✗ THE DISCLOSURE FIGURES DRIFTED. These are the numbers about how much")
                print("  evidence this registry refuses to show. They print verbatim in cell 0.6,")
                print("  and they are exactly the numbers that must not go stale.\n")
                for p in d_problems:
                    print(f"  {p}")
                print("\n  Paste over the block in examples/measure-a-skill/manifest.yaml:\n")
                for line in paste_block(declared, live, today).splitlines():
                    print(f"    {line}")
                print(
                    "\n  Then mirror it into FALLBACK in _build.py and re-run the builder.\n"
                    f"  (sample seed {seed} — reproduce with --seed {seed})\n"
                )
                online_problems += d_problems
            ledger.done(c_disclosure, f"{len(rows)} measured skills walked")
    except Unreachable as e:
        # Caught, NOT returned on. Whatever is already in the two piles was
        # established before the registry went quiet and is still true; the
        # single reporting block below decides the exit code with the offline
        # pile taking precedence over the network's silence.
        unreachable = e

    # ── one reporting block, one exit code ────────────────────────────────
    for note in notes:
        print(f"  {note}")
    print()
    print(ledger.render())
    print(f"  {_stats_line()}")

    if unreachable:
        print(f"\n⚠ the registry stopped answering: {unreachable}")
        print(
            "  Every NOT REACHED line above is a check this run could not make. That is a gap in\n"
            "  coverage, not a clean bill of health."
        )

    problems = offline_problems + online_problems

    # The three ways this run can fail to reach a verdict. All of them are exit
    # 2 and none of them is a pass — including the last one, which is the
    # "0 checks performed, exit 0" shape this ledger exists to make impossible.
    if unreachable:
        no_verdict = "the registry never answered"
    elif not ledger.performed:
        no_verdict = "not one check actually ran"
    elif not ledger.ran_any(NETWORK):
        no_verdict = "nothing was checked against the registry — only the offline half ran"
    else:
        no_verdict = ""

    # Machine-readable, for .github/workflows/notebook-freshness.yml. That
    # workflow downgrades exit 2 to success, so it needs to be able to prove
    # from the output alone that the downgrade is not burying a real finding.
    exit_code = 1 if problems else (2 if no_verdict else 0)
    print(
        f"\nVERDICT exit={exit_code} checks_performed={ledger.performed}/{ledger.applicable} "
        f"offline_problems={len(offline_problems)} registry_problems={len(online_problems)} "
        f"registry={'unreachable' if unreachable else 'answered'}"
    )

    if problems:
        print(f"\n✗ examples/measure-a-skill has rotted — {len(problems)} problem(s):\n")
        if offline_problems:
            # Named separately and first, because these are the ones the old
            # code threw away: they need no network to be true, so a throttling
            # registry is not a reason to defer them to tomorrow's run.
            print(f"  OFFLINE — true regardless of the registry ({len(offline_problems)}):")
            for p in offline_problems:
                print(f"    {p}")
        if online_problems:
            if offline_problems:
                print()
            print(f"  FROM THE REGISTRY — answered definitively ({len(online_problems)}):")
            for p in online_problems:
                print(f"    {p}")
        if unreachable:
            print(
                "\n  The registry went unreachable partway through this run, so the list above is "
                "a\n  FLOOR, not a total — but it is real, and exit 1 stands. Re-run when the "
                "registry is\n  answering to see whether anything else drifted too."
            )
        print(
            "\n  Everything above names the manifest.yaml key to change. The notebook itself "
            "hardcodes\n  no numbers, so a YAML edit on main reaches every already-open Colab "
            "tab on its next run."
        )
        return 1

    if no_verdict:
        print(
            f"\n⚠ NO VERDICT — {no_verdict}, and nothing was wrong offline either.\n"
            f"  {ledger.performed} of {ledger.applicable} checks ran; this is exit 2, not a pass. "
            "Nothing rotted as far as\n  this run can tell, and it could not tell very far."
        )
        return 2

    scope = "gates + disclosure figures" if args.disclosure else "gates (pass --disclosure for the figures)"
    skipped = ledger.count(Ledger.SKIPPED)
    print(
        f"\n✓ examples/measure-a-skill is fresh — {scope}. "
        f"'{demo['slug']}' still passes every gate, still serves every prompt, and "
        f"'{demo.get('demo_case')}' still has its two arms disagreeing."
    )
    if skipped:
        print(
            f"  ⚠ …but {skipped} of {ledger.applicable} checks were SKIPPED (see the ledger "
            "above). This is a\n    narrower pass than it looks."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
