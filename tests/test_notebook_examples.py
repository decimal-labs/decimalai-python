"""The example notebooks must run on a cold machine with no credentials.

Three claims are made about examples/measure-a-skill/, and until this file
existed all three were claims rather than tests:

1.  **It runs keyless, top to bottom, without a traceback.** Every other
    notebook in examples/ asks for DECIMAL_API_KEY in its second code cell and
    raises DecimalConfigError on the placeholder, so a visitor's first
    experience of the SDK is a stack trace. "Degrades gracefully" is only true
    if something fails when it stops being true — so the code cells are
    extracted and executed in a subprocess with every credential scrubbed from
    the environment, and the output is checked for a traceback, not just for a
    zero exit code (a bare `except Exception: pass` would satisfy the latter).

2.  **The committed .ipynb is what _build.py produces.** The repo convention is
    that notebooks are generated, and a hand-edit to the JSON is invisible until
    the next person runs the builder and clobbers it.

3.  **It hardcodes zero numbers.** That is the whole reason
    scripts/check_notebook_manifest.py can be the thing that fails instead of
    the reader's first impression. The one legal exception is the embedded
    FALLBACK copy of the manifest, which exists so a raw.github blip degrades to
    a stale figure with a printed warning rather than a KeyError.

WHY NOT nbmake. It is not in this repo's `dev` extra and neither is nbformat or
nbclient, and pulling in a Jupyter stack to run three notebooks that import
`requests` and print text would be a large dependency for a small job. Extracting
the code cells and running them in a subprocess also tests something nbmake
cannot: the process EXIT CODE and the absence of a traceback in combined
stdout+stderr, on a machine with no kernel and no notebook tooling at all —
which is closer to the promise being made ("nothing to install").

HOW TO ACTUALLY RUN THE KEYLESS TESTS. Read this before trusting a green run.

    pytest tests/test_notebook_examples.py                      # ← does NOT run
                                                                #   the live one
    pytest tests/test_notebook_examples.py -m "not live_llm"    # ← runs everything
                                                                #   here, incl. live
    pytest tests/test_notebook_examples.py \
        -k keyless_with_no_network                              # ← the offline twin,
                                                                #   default suite, ~1s

pyproject's `addopts = -m 'not integration and not live_llm'` DESELECTS the
`integration` marker, so a bare `pytest tests/` reports this file green while
never executing a single notebook cell. That is not a bug to fix by deleting the
marker — `test_notebook_runs_keyless_without_traceback` really does need
api.decimal.ai and app.decimal.ai, it takes minutes, and a default unit suite
that silently depends on a rate-limited public API is worse than one that
doesn't. Two things make it un-misreadable instead:

  * `test_notebook_runs_keyless_with_no_network` (added 2026-08-10) carries NO
    marker, so it runs in the default suite. It executes the same extracted
    cells with the same scrubbed environment but with the socket layer
    hard-blocked, and asserts the notebook fails ONLY in its own retry helper's
    controlled way — never a NameError, a KeyError on an unset key, an EOFError
    from a prompt, or a raw transport traceback. Those are the keyless-specific
    defects; none of them need a network to expose. It costs about a second.
  * `.github/workflows/notebook-freshness.yml`'s `keyless` job runs the live one
    with `-m "not live_llm"`, which OVERRIDES addopts (a command-line `-m`
    replaces the one in addopts rather than being ANDed with it) — verified with
    `--collect-only`: 13 items selected by default, 16 with the flag, the three
    extra being exactly `test_notebook_runs_keyless_without_traceback[*]`. That
    job then FAILS if those three were not collected, so the marker cannot
    quietly turn into "nobody executes the notebooks at all".

The live keyless test skips — loudly, never silently passes — when the registry
cannot be reached.
"""

from __future__ import annotations

import ast
import functools
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"
MEASURE_A_SKILL = EXAMPLES / "measure-a-skill"
NOTEBOOK = MEASURE_A_SKILL / "measure_a_skill.ipynb"
MANIFEST = MEASURE_A_SKILL / "manifest.yaml"

# Scrubbed from the child environment so "keyless" means keyless even on a
# developer laptop that has all of these exported.
CREDENTIAL_VARS = (
    "DECIMAL_API_KEY",
    "DECIMALAI_API_KEY",
    "DECIMAL_AI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)

# Both hosts the notebooks read: the registry API, and the app that serves raw
# SKILL.md bodies. They fail independently — the app runs a separate Cloud Run
# service — and neither being up is a statement about the notebook, so both are
# probed and an outage in either one skips rather than fails.
PROBE_URLS = (
    "https://api.decimal.ai/api/v1/registry/skills?measured=only&limit=1",
    "https://app.decimal.ai/s/flsa-exemption-test@1/SKILL.md",
)

# A notebook whose committed JSON deliberately differs from what its builder
# produces, with the reason. An entry here that turns out to be IN sync fails
# the test — the allowlist cleans itself up instead of quietly outliving the
# problem.
KNOWN_BUILDER_DRIFT = {
    # Found 2026-08-10 while adding this test. The committed notebook was
    # hand-corrected to the real SDK surface (`decimalai.pull_dataset(...)`)
    # while _build.py still generates a call that does not exist
    # (`decimalai.export_dataset(...)`). Running the builder would REVERT the
    # correction, which is exactly what this test exists to prevent — fix
    # belongs in that example's _build.py, not here.
    "examples/datasets-and-training/build_sft_dataset.ipynb",
}

# What every tier's `get()` helper raises once its retry budget is spent:
# "gave up after 5 attempts: <url>", "503 on <url>", "ReadTimeout on <url>",
# "ConnectionError on <url>". Defined once and shared by both keyless tests so
# "the notebook handled the outage" means the same thing in each.
RETRY_HELPER_GAVE_UP = re.compile(
    r"RuntimeError: (gave up after|\d{3} on |\w*(?:Timeout|Error) on )"
)

# A frame from these means nothing caught the transport error — a real defect,
# and the one this file found on its first run.
RAW_TRANSPORT_FRAMES = ("requests.exceptions.", "urllib3.exceptions.", "socket.timeout")


def notebooks():
    """Only the tiered onboarding notebooks.

    The rest of examples/ deliberately requires credentials — asking for
    DECIMAL_API_KEY in the second cell is the norm this tier exists to contrast
    with — so a keyless run is a claim about measure-a-skill/ specifically.
    """
    return sorted(MEASURE_A_SKILL.glob("*.ipynb"))


def code_source(nb_path):
    """The notebook's code cells, concatenated into one runnable script.

    Magics and shell escapes are dropped — there is no IPython here — and the
    packages a `!pip install` line would have fetched are returned so the caller
    can check they are present instead of watching an ImportError it caused
    itself. In Colab those lines really do run, so preinstalling them is the
    faithful reproduction, not a shortcut.
    """
    nb = json.loads(nb_path.read_text())
    chunks, needs = [], []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        lines = []
        for line in "".join(cell.get("source", [])).splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("!", "%")):
                if "pip install" in stripped:
                    tail = stripped.split("pip install", 1)[1].split()
                    needs += [t for t in tail if not t.startswith("-")]
                continue
            lines.append(line)
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks), needs


def missing_distributions(names):
    """Which of these pip names are not installed here.

    Checked by distribution metadata rather than by import, so `google-genai`
    → `google.genai` and friends do not need a hand-maintained name map.
    """
    import importlib.metadata as md

    missing = []
    for name in names:
        base = name.split("[")[0].split("==")[0].split(">")[0].split("<")[0]
        try:
            md.distribution(base)
        except md.PackageNotFoundError:
            missing.append(base)
    return missing


@functools.lru_cache(maxsize=1)
def registry_reachable(attempts=3, backoff=20):
    """Are both hosts answering right now?

    Retried, because the anonymous edge rate limit is easy to trip from a
    developer machine that has been poking the API by hand, and a skip that
    happens on every run is a check that has quietly stopped existing. Cached for
    the session: probing once per parametrised notebook would spend minutes
    re-establishing the same fact, and burn the same rate limit doing it.
    """
    for url in PROBE_URLS:
        host = urllib.parse.urlsplit(url).netloc
        detail = ""
        for attempt in range(attempts):
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "decimalai-notebook-test"}
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    if resp.status == 200:
                        break
                    detail = f"{host} answered HTTP {resp.status}"
            except urllib.error.HTTPError as e:
                detail = f"{host} answered HTTP {e.code}"
            except OSError as e:
                detail = f"{host}: {type(e).__name__}: {e}"
            if attempt < attempts - 1:
                time.sleep(backoff)
        else:
            return False, detail
    return True, "both hosts answered"


# ──────────────────────────────────────────────────────────────────────
# 1. Runs keyless, no traceback
# ──────────────────────────────────────────────────────────────────────

# KEEP THE MARKER. This one really does need api.decimal.ai + app.decimal.ai and
# takes minutes; `integration` is what keeps a rate-limited public API out of the
# default unit suite. The consequence — `pytest tests/` NEVER runs this — is real
# and is covered two ways, both named in this module's docstring:
#
#     pytest tests/test_notebook_examples.py -m "not live_llm"    ← runs THIS test
#     pytest tests/test_notebook_examples.py -k keyless_with_no_network
#                                                                 ← the offline twin,
#                                                                   in the default suite
#
# The CI job that runs it is `keyless` in .github/workflows/notebook-freshness.yml,
# and it now fails if these items are not collected — so the marker cannot quietly
# turn into "nobody executes the notebooks at all".
@pytest.mark.integration
@pytest.mark.parametrize("nb_path", notebooks(), ids=lambda p: p.stem)
def test_notebook_runs_keyless_without_traceback(nb_path, tmp_path):
    pytest.importorskip("requests", reason="the notebooks use requests, which Colab preinstalls")
    pytest.importorskip("yaml", reason="the notebooks use PyYAML, which Colab preinstalls")

    ok, detail = registry_reachable()
    if not ok:
        # A skip, never a pass. The public registry rate-limits anonymous
        # traffic at the edge and runs one backend instance, so "unreachable
        # right now" is a real state that must not be reported as a broken
        # notebook — the same distinction scripts/check_notebook_manifest.py
        # draws with its exit code 2.
        pytest.skip(f"registry unreachable ({detail}) — cannot exercise the notebook end to end")

    source, needs = code_source(nb_path)
    missing = missing_distributions(needs)
    if missing:
        # A skip that names what to install, so the freshness workflow can add it
        # and start exercising this notebook — rather than a permanent silent
        # hole that nobody can see from the test output.
        pytest.skip(
            f"{nb_path.name} pip-installs {' '.join(missing)}, which is not present here — "
            f"add it to the 'Install what Colab ships' step to cover this notebook"
        )

    script = tmp_path / "notebook_cells.py"
    script.write_text(source)

    env = {k: v for k, v in os.environ.items() if k not in CREDENTIAL_VARS}
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=tmp_path,
            env=env,
            # A notebook that prompts for a key gets EOF rather than a live tty:
            # it must handle that, not block forever waiting for a human.
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired as e:
        partial = (e.stdout or "") + (e.stderr or "")
        pytest.fail(
            f"{nb_path.name} was still running after 10 minutes with no credentials — a reader "
            f"who hits Run all sits there watching it. Blocked on input, or retrying a call that "
            f"will never succeed?\nLast output:\n" + "\n".join(partial.splitlines()[-20:])
        )
    output = proc.stdout + proc.stderr
    tail = "\n".join(output.splitlines()[-40:])

    # A hostile network is not a broken notebook — but "the notebook has no
    # handling for a dropped connection" IS. The line between them is visible in
    # the output: the notebooks' own `get()` catches transport errors, retries,
    # and re-raises as `RuntimeError: <thing> on <url>` with the chain suppressed.
    # So a bare `requests.exceptions.*` frame means nothing caught it — a real
    # defect, and the one this test found on its first run — while a RuntimeError
    # from the retry helper means the notebook did its job against an API that
    # was down or throttling. Only the second is a skip.
    if "Traceback (most recent call last)" in output:
        raw_transport = any(sig in output for sig in RAW_TRANSPORT_FRAMES)
        gave_up = bool(RETRY_HELPER_GAVE_UP.search(output))
        if gave_up and not raw_transport:
            pytest.skip(
                f"{nb_path.name}'s retry helper exhausted its budget against a throttling or "
                f"restarting registry — infrastructure, not the notebook:\n{tail}"
            )

    assert "Traceback (most recent call last)" not in output, (
        f"{nb_path.name} raised with no credentials configured — a cold reader's first "
        f"experience of this SDK would be a stack trace:\n{tail}"
    )
    assert proc.returncode == 0, f"{nb_path.name} exited {proc.returncode}:\n{tail}"
    # An empty run would satisfy both assertions above.
    assert len(output.strip()) > 500, (
        f"{nb_path.name} produced almost no output ({len(output.strip())} chars) — it exited "
        f"clean without actually showing the reader anything:\n{tail}"
    )


# ──────────────────────────────────────────────────────────────────────
# 1b. The same claim, with no network and no marker — so `pytest tests/` covers it
# ──────────────────────────────────────────────────────────────────────

# Patches the socket layer shut, then runs the extracted cells. Written as a
# separate file rather than a sitecustomize.py on PYTHONPATH because a
# sitecustomize of our own SHADOWS the interpreter's (Homebrew's python ships one
# to add its site-packages), which silently turns every import into
# ModuleNotFoundError and every notebook into a false failure. Cost 20 minutes to
# find; runpy costs nothing and keeps real filenames in the traceback.
OFFLINE_RUNNER = '''\
import runpy, socket, sys, time


def _deny(*a, **k):
    raise OSError(101, "network is unreachable (blocked by the test harness)")


socket.socket.connect = _deny
socket.socket.connect_ex = _deny
socket.create_connection = _deny
socket.getaddrinfo = _deny
# The notebooks back off between retries. Real seconds here buy nothing: the
# next attempt fails for the same reason, and a 1s unit test becomes a 30s one.
time.sleep = lambda *a, **k: None
runpy.run_path(sys.argv[1], run_name="__main__")
'''

# The exceptions that mean "keyless is broken", as opposed to "the network is
# gone". Every one of them is reachable with no network at all, which is exactly
# why this test does not need one.
KEYLESS_DEFECTS = (
    "EOFError",  # blocked on input()/getpass with no tty — a reader watches it hang
    "KeyError",  # os.environ["DECIMAL_API_KEY"] on a machine that has no key
    "NameError",
    "AttributeError",
    "TypeError",
    "IndexError",
)


def _final_exception(output):
    """The `SomeError: message` line that actually ended the run."""
    hits = [
        line
        for line in output.splitlines()
        if re.match(r"^[A-Za-z_][\w.]*(Error|Exception|Exit|Interrupt|Warning):", line)
    ]
    return hits[-1] if hits else ""


@pytest.mark.parametrize("nb_path", notebooks(), ids=lambda p: p.stem)
def test_notebook_runs_keyless_with_no_network(nb_path, tmp_path):
    """No credentials, no network, no marker — this one runs in `pytest tests/`.

    Its live twin above is the real end-to-end proof and is excluded from the
    default suite by the `integration` marker, which means a plain `pytest
    tests/` used to assert NOTHING about whether these notebooks execute. This
    closes that hole for the part that does not need a registry: with the socket
    layer shut, a notebook may finish clean or die through its own retry helper
    ("gave up after 5 attempts: <url>"), and nothing else. A KeyError on an unset
    key, an EOFError from a prompt, a NameError, or a bare
    requests.exceptions.* frame all reproduce offline and all mean a cold
    reader's first cell is a stack trace.
    """
    pytest.importorskip("requests", reason="the notebooks use requests, which Colab preinstalls")
    pytest.importorskip("yaml", reason="the notebooks use PyYAML, which Colab preinstalls")

    source, needs = code_source(nb_path)
    missing = missing_distributions(needs)
    if missing:
        pytest.skip(
            f"{nb_path.name} pip-installs {' '.join(missing)}, which is not present here — "
            f"install it (or add it to the workflow's 'Install what Colab ships' step) to "
            f"cover this notebook"
        )

    script = tmp_path / "notebook_cells.py"
    script.write_text(source)
    runner = tmp_path / "_offline_runner.py"
    runner.write_text(OFFLINE_RUNNER)

    env = {k: v for k, v in os.environ.items() if k not in CREDENTIAL_VARS}
    env["PYTHONUNBUFFERED"] = "1"
    # The socket patches live in this process only, and Tier 2 shells out to
    # `pip install decimalai`. Neuter that child too, so "no network" is true for
    # the whole tree and the notebook's pip-failed branch is what gets exercised.
    env.update(
        PIP_NO_INDEX="1", PIP_NO_INPUT="1", PIP_RETRIES="0", PIP_TIMEOUT="1",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
    )

    try:
        proc = subprocess.run(
            [sys.executable, str(runner), str(script)],
            cwd=tmp_path,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired as e:
        partial = (e.stdout or "") + (e.stderr or "")
        pytest.fail(
            f"{nb_path.name} was still running after 3 minutes with the network blocked and no "
            f"credentials. With every socket refused instantly there is nothing left to wait "
            f"for — this is a blocked prompt or an unbounded retry loop.\nLast output:\n"
            + "\n".join(partial.splitlines()[-20:])
        )

    output = proc.stdout + proc.stderr
    tail = "\n".join(output.splitlines()[-40:])
    final = _final_exception(output)

    raw_transport = [sig for sig in RAW_TRANSPORT_FRAMES if sig in output]
    assert not raw_transport, (
        f"{nb_path.name} let a raw transport exception ({', '.join(raw_transport)}) reach the "
        f"reader — nothing caught it. Every network call belongs behind the notebook's own "
        f"get()/retry helper:\n{tail}"
    )

    defect = next((d for d in KEYLESS_DEFECTS if final.startswith(d + ":")), None)
    assert not defect, (
        f"{nb_path.name} died with {final!r} on a machine with no credentials and no network. "
        f"A {defect} is a keyless bug, not an outage — the notebook has to print a skip notice "
        f"and carry on:\n{tail}"
    )

    if proc.returncode != 0:
        assert RETRY_HELPER_GAVE_UP.search(final), (
            f"{nb_path.name} exited {proc.returncode} with {final!r}. With no network the ONLY "
            f"acceptable ending is the notebook's own retry helper giving up "
            f"(\"gave up after N attempts: <url>\"); anything else is a code path that has "
            f"never been run keyless:\n{tail}"
        )

    assert len(output.strip()) > 200, (
        f"{nb_path.name} produced almost no output ({len(output.strip())} chars) before "
        f"stopping — it is not reaching the reader at all:\n{tail}"
    )


def test_every_notebook_code_cell_parses():
    """Cheap, offline, and catches a hand-edit that no other test would.

    The execution tests above stop at the first cell that raises, so a syntax
    error in a LATER cell of a notebook whose network leg dies early is
    invisible to them. This one compiles every cell of every tier.
    """
    checked = 0
    for nb_path in notebooks():
        nb = json.loads(nb_path.read_text())
        for i, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            # Magics and shell escapes are not Python; code_source() drops them
            # for the same reason.
            lines = [
                line
                for line in "".join(cell.get("source", [])).splitlines()
                if not line.lstrip().startswith(("!", "%"))
            ]
            try:
                ast.parse("\n".join(lines))
            except SyntaxError as e:
                pytest.fail(f"{nb_path.name} cell {i} does not parse: {e}")
            checked += 1
    assert checked, "no code cells found in examples/measure-a-skill — the glob broke"


# ──────────────────────────────────────────────────────────────────────
# 2. The committed .ipynb is what the builder produces
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "builder",
    sorted(EXAMPLES.rglob("_build*.py")),
    ids=lambda p: f"{p.parent.name}:{p.stem}",
)
def test_committed_notebook_matches_builder(builder, tmp_path):
    """Rebuild in a temp dir and diff. Never writes to the working tree."""
    shutil.copy(builder, tmp_path / "_build.py")
    proc = subprocess.run(
        [sys.executable, str(tmp_path / "_build.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"{builder} failed to run:\n{proc.stdout}\n{proc.stderr}"

    built = sorted(tmp_path.glob("*.ipynb"))
    assert built, f"{builder} produced no .ipynb"
    for rebuilt in built:
        committed = builder.parent / rebuilt.name
        assert committed.exists(), f"{builder} builds {rebuilt.name}, which is not committed"
        rel = committed.relative_to(REPO_ROOT).as_posix()
        in_sync = json.loads(rebuilt.read_text()) == json.loads(committed.read_text())
        if rel in KNOWN_BUILDER_DRIFT:
            assert not in_sync, (
                f"{rel} is listed in KNOWN_BUILDER_DRIFT but now matches {builder.name} — "
                f"delete the entry so the next drift is caught."
            )
            continue
        assert in_sync, (
            f"{rel} is out of sync with {builder.name} — someone hand-edited the notebook "
            f"JSON, and the next `python {builder.name}` will silently revert it. Re-run the "
            f"builder and commit the result."
        )


# ──────────────────────────────────────────────────────────────────────
# 3. Zero hardcoded figures, outside the deliberate offline copy
# ──────────────────────────────────────────────────────────────────────

def _fallback_span(text):
    """(start, end) of the `FALLBACK = {...}` literal, or None."""
    start = text.find("FALLBACK = {")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return start, i + 1
    return None


def test_notebook_quotes_no_manifest_figures_outside_the_fallback():
    yaml = pytest.importorskip("yaml")
    manifest = yaml.safe_load(MANIFEST.read_text())
    disclosure = manifest["registry_disclosure"]

    nb = json.loads(NOTEBOOK.read_text())
    text = "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []))

    span = _fallback_span(text)
    assert span, (
        "the notebook has no FALLBACK literal — without it a raw.githubusercontent blip is a "
        "KeyError on a 404 page parsed as YAML, not a printed warning"
    )
    outside = text[: span[0]] + text[span[1] :]

    figures = {
        key: value
        for key, value in disclosure.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    leaked = [f"registry_disclosure.{k} ({v})" for k, v in figures.items() if str(v) in outside]
    assert not leaked, (
        "the notebook prints a figure it should be fetching: "
        + ", ".join(leaked)
        + ". Every number in this notebook comes from manifest.yaml on main or from the live "
        "API, so repairing rot stays a one-line YAML edit; the only legal copy is the FALLBACK "
        "literal used when raw.github is unavailable."
    )


def test_fallback_copy_agrees_with_the_manifest():
    """The offline copy is what a reader sees during a raw.github outage.

    scripts/check_notebook_manifest.py asserts the same thing on a cron; this
    catches it in the PR that causes it, with no network.
    """
    yaml = pytest.importorskip("yaml")
    manifest = yaml.safe_load(MANIFEST.read_text())
    nb = json.loads(NOTEBOOK.read_text())
    text = "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []))
    span = _fallback_span(text)
    assert span
    fallback = ast.literal_eval(text[span[0] + len("FALLBACK = ") : span[1]])

    mismatches = []
    for section, values in fallback.items():
        live = manifest.get(section)
        for key, embedded in (values or {}).items():
            if live is None or key not in live:
                mismatches.append(f"{section}.{key} is in FALLBACK but not in manifest.yaml")
            elif live[key] != embedded:
                mismatches.append(
                    f"{section}.{key}: manifest {live[key]!r} vs embedded {embedded!r}"
                )
    assert not mismatches, (
        "the notebook's embedded manifest copy drifted from manifest.yaml — update FALLBACK in "
        "examples/measure-a-skill/_build.py and re-run it:\n  " + "\n  ".join(mismatches)
    )
