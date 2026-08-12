#!/usr/bin/env python3
"""End-to-end tests for DecimalAI sample apps.

Runs the ACTUAL sample app code — imports modules, executes scenario scripts,
and extracts/runs notebook cells. No reimplementations.
"""

import json
import os
import sys
import traceback
import subprocess
import re

PASS = 0
FAIL = 0

def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        print(f"  ✅ {name}")
    except Exception as e:
        FAIL += 1
        print(f"  ❌ {name}: {e}")
        traceback.print_exc(limit=3)


def extract_notebook_cells(nb_path):
    """Extract code cells from a notebook, filtering pip installs and awaits.
    Returns list of (index, cleaned_source) tuples.
    """
    with open(nb_path) as f:
        nb = json.load(f)
    cells = []
    idx = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        lines = [l for l in src.split("\n")
                 if not l.strip().startswith("!") and not l.strip().startswith("await ")]
        code = "\n".join(lines).strip()
        cells.append((idx, code))
        idx += 1
    return cells


EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
REFAPP_DIR = os.path.join(EXAMPLES_DIR, "reference-app")


# ═══════════════════════════════════════════════════════
# 1. Reference App: Run actual tool modules
# ═══════════════════════════════════════════════════════

print("\n═══ 1. Reference App Tools (imported from agent/tools/) ═══\n")

sys.path.insert(0, REFAPP_DIR)

def test_search_docs_module():
    from agent.tools.search_docs import search_docs

    # Keyword match
    r = search_docs("password reset")
    assert r["total"] > 0
    assert r["results"][0]["title"] == "How to Reset Your Password"
    assert r["results"][0]["relevance"] == 0.95

    # Different keyword
    r = search_docs("shipping information")
    assert r["total"] > 0

    # All keywords
    for kw in ["password", "return", "shipping", "account", "payment"]:
        r = search_docs(kw)
        assert r["total"] > 0, f"No results for '{kw}'"

    # Fuzzy fallback
    r = search_docs("completely random gibberish xyz")
    assert r["total"] == 1
    assert r["results"][0]["relevance"] == 0.5

test("search_docs — all keywords + fuzzy fallback", test_search_docs_module)

def test_check_order_module():
    from agent.tools.check_order import check_order

    # All 5 orders
    orders = {
        "ORD-10001": ("delivered", True),
        "ORD-10002": ("shipped", True),
        "ORD-10003": ("processing", True),
        "ORD-10004": ("cancelled", True),
        "ORD-10005": ("shipped", True),
    }
    for oid, (status, has_eta) in orders.items():
        r = check_order(oid)
        assert r["found"], f"{oid} not found"
        assert r["status"] == status, f"{oid}: expected {status}, got {r['status']}"

    # Not found
    r = check_order("ORD-99999")
    assert not r["found"]
    assert "not found" in r["error"]

test("check_order — all 5 orders + not found", test_check_order_module)

def test_process_refund_module():
    from agent.tools.process_refund import process_refund

    r = process_refund("ORD-10001", "damaged item")
    assert r["status"] == "approved"
    assert r["amount"] == 29.99
    assert r["estimated_days"] == 3
    assert r["refund_id"] == "REF-10001"
    assert "ORD-10001" in r["message"]
    assert "$29.99" in r["message"]

test("process_refund — approval flow", test_process_refund_module)


# ═══════════════════════════════════════════════════════
# 2. Reference App Scenarios (subprocess execution)
# ═══════════════════════════════════════════════════════

print("\n═══ 2. Reference App Scenarios (subprocess execution) ═══\n")

def run_scenario(script_name, checks):
    """Run a scenario script as subprocess with a dummy API key."""
    env = os.environ.copy()
    env["DECIMAL_API_KEY"] = "dai_sk_test_dummy_key_for_testing"
    result = subprocess.run(
        [sys.executable, os.path.join(REFAPP_DIR, "scenarios", script_name)],
        capture_output=True, text=True, timeout=30, cwd=REFAPP_DIR, env=env,
    )
    for check_name, check_str in checks:
        assert check_str in result.stdout, (
            f"'{check_str}' not in output for {check_name}.\n"
            f"stdout: {result.stdout[:200]}\nstderr: {result.stderr[:200]}"
        )
    return result

def test_scenario3():
    run_scenario("03_run_evals.py", [
        ("eval header", "Run Evaluations"),
        ("answered eval", "answered_question"),
        ("hedging check", "hedging"),
    ])

test("Scenario 3: run_evals.py (defines + runs evaluators)", test_scenario3)

def test_scenario4():
    run_scenario("04_build_dataset.py", [
        ("impact report", "Impact Report"),
        ("SFT format", "check_order"),
        ("workflow summary", "Dataset built"),
    ])

test("Scenario 4: build_dataset.py (impact report + SFT format)", test_scenario4)


# ═══════════════════════════════════════════════════════
# 3. Quickstart Notebook: Execute actual cells
# ═══════════════════════════════════════════════════════

print("\n═══ 3. Quickstart Notebook (actual cell execution) ═══\n")

def test_quickstart_cells():
    nb_path = os.path.join(EXAMPLES_DIR, "quickstart", "quickstart.ipynb")
    cells = extract_notebook_cells(nb_path)

    # Find the cell that defines tool functions (search_docs, check_inventory)
    tool_cell = None
    v2_cell = None
    for idx, code in cells:
        if "def search_docs" in code and "def check_inventory" in code:
            tool_cell = code
        if "def lookup_stock" in code or "def process_refund" in code:
            v2_cell = code

    assert tool_cell is not None, "Could not find tool definition cell"

    ns = {}
    exec(tool_cell, ns)
    assert callable(ns.get("search_docs")), "search_docs not defined"
    assert callable(ns.get("check_inventory")), "check_inventory not defined"

    # Test actual tool outputs
    r = ns["search_docs"]("How do I reset my password?")
    assert "Settings" in r or "password" in r.lower()

    r = ns["check_inventory"]("SKU-1234")
    assert r["name"] == "Wireless Mouse"
    assert r["in_stock"] is True

    r = ns["check_inventory"]("SKU-5678")
    assert r["in_stock"] is False

    # V2 cell
    if v2_cell:
        exec(v2_cell, ns)
        if callable(ns.get("process_refund")):
            r = ns["process_refund"]("ORD-001", "damaged")
            assert r["amount"] == 29.99

test("quickstart.ipynb (tool cells extracted + tested)", test_quickstart_cells)


# ═══════════════════════════════════════════════════════
# 4. Evaluations Notebook: Execute actual cells
# ═══════════════════════════════════════════════════════

print("\n═══ 4. Evaluations Notebook (actual cell execution) ═══\n")

def test_evals_notebook_cells():
    nb_path = os.path.join(EXAMPLES_DIR, "evaluations", "builtin_evaluators.ipynb")
    cells = extract_notebook_cells(nb_path)

    # Build a shared namespace, run cells that don't need decimalai.init()
    ns = {}

    for idx, code in cells:
        # Skip cells that call decimalai.init() or are empty
        if "decimalai.init()" in code or not code.strip():
            continue
        # Skip cells that are all comments
        if all(l.strip().startswith("#") for l in code.split("\n") if l.strip()):
            continue

        try:
            exec(code, ns)
        except Exception as e:
            # Some cells may need init — skip gracefully
            if "DecimalConfigError" in str(type(e).__name__) or "API key" in str(e):
                continue
            raise

    # Verify we got the sample trace
    assert "sample" in ns, "TraceData sample not created"
    assert ns["sample"].input == "What is the return policy?"

    # Verify evaluators were created and can run
    assert callable(ns.get("check_answered")), "@eval check_answered not created"
    assert callable(ns.get("quality_score")), "@eval quality_score not created"
    assert callable(ns.get("check_hallucination")), "@eval check_hallucination not created"

    # Run each evaluator on the sample trace
    r = ns["check_answered"](ns["sample"])
    assert r is not None and r.passed

    r = ns["quality_score"](ns["sample"])
    assert r is not None and r.score > 0

    r = ns["check_hallucination"](ns["sample"])
    assert r is not None and r.passed

test("builtin_evaluators.ipynb (evals created + run on sample)", test_evals_notebook_cells)


# ═══════════════════════════════════════════════════════
# 5. Manifest Change Notebook: Validate scenarios
# ═══════════════════════════════════════════════════════

print("\n═══ 5. Manifest Change Notebook (scenario validation) ═══\n")

def test_manifest_notebook():
    nb_path = os.path.join(EXAMPLES_DIR, "version-aware-loop", "manifest_change.ipynb")
    with open(nb_path) as f:
        nb = json.load(f)

    md_text = " ".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")

    # All 4 scenarios present
    assert "Tool Rename" in md_text
    assert "Tool Added" in md_text
    assert "Tool Removed" in md_text
    assert "Schema Change" in md_text

    # Classification table present
    for cls in ["keep", "repair", "replay", "drop"]:
        assert cls.lower() in md_text.lower(), f"Missing classification: {cls}"

    # Code cells reference decimalai APIs
    code_text = " ".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "decimalai.trace" in code_text
    assert "log_tool_call" in code_text

test("manifest_change.ipynb (4 scenarios + classifications)", test_manifest_notebook)


# ═══════════════════════════════════════════════════════
# 6. Datasets Notebook: Execute SFT cell
# ═══════════════════════════════════════════════════════

print("\n═══ 6. Datasets Notebook (actual cell execution) ═══\n")

def test_datasets_notebook_cells():
    nb_path = os.path.join(EXAMPLES_DIR, "datasets-and-training", "build_sft_dataset.ipynb")
    cells = extract_notebook_cells(nb_path)

    # Find the cell that defines sft_example
    ns = {"json": json}  # Pre-inject json module
    sft_cell = None
    for idx, code in cells:
        if "sft_example" in code:
            sft_cell = code
            break

    assert sft_cell is not None, "Could not find sft_example cell"
    exec(sft_cell, ns)
    assert "sft_example" in ns

    msgs = ns["sft_example"]["messages"]
    assert len(msgs) == 5
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "check_order"
    assert msgs[3]["role"] == "tool"
    assert msgs[4]["role"] == "assistant" and msgs[4]["content"] is not None

    # Roundtrip JSONL
    line = json.dumps(ns["sft_example"])
    assert json.loads(line) == ns["sft_example"]

test("build_sft_dataset.ipynb (SFT format cell executed + validated)", test_datasets_notebook_cells)


# ═══════════════════════════════════════════════════════
# 7. All Notebooks: Structure + Colab + Syntax
# ═══════════════════════════════════════════════════════

print("\n═══ 7. All Notebooks (structure + Colab + syntax) ═══\n")

def test_all_notebooks():
    notebooks = []
    for root, dirs, files in os.walk(EXAMPLES_DIR):
        for f in sorted(files):
            if f.endswith(".ipynb"):
                notebooks.append(os.path.join(root, f))

    assert len(notebooks) == 10, f"Expected 10, found {len(notebooks)}"

    for nb_path in notebooks:
        nb = json.load(open(nb_path))
        assert nb["nbformat"] == 4

        # Colab badge with correct org
        first = "".join(nb["cells"][0]["source"])
        assert "decimal-labs/decimalai-python" in first

        # All URLs use correct org
        full = json.dumps(nb)
        for m in re.finditer(r"colab\.research\.google\.com/github/([^/]+)", full):
            assert m.group(1) == "decimal-labs"

        # Syntax check all code cells
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            clean = "\n".join(l for l in src.split("\n")
                            if not l.strip().startswith("!") and not l.strip().startswith("await "))
            if clean.strip():
                compile(clean, f"{os.path.basename(nb_path)}:cell_{i}", "exec")

test("All 10 notebooks (structure + Colab URLs + Python syntax)", test_all_notebooks)


# ═══════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════

print()
print("═" * 55)
print(f"  Results: {PASS} passed, {FAIL} failed")
print("═" * 55)

if FAIL > 0:
    sys.exit(1)
else:
    print("  🎉 All tests use actual sample app code!")
