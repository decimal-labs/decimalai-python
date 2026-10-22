"""``decimalai/_frameworks.py``, the one framework registry, held to the code it describes.

The registry is copied into three places by generators rather than by hand: the
scaffold's constants, the docs' two framework tables, and the dashboard's JSON. A
generator only moves a fact; these tests are what make the fact TRUE — every flag is a
real ``init()`` parameter, every extra is in pyproject, every module, entry point and
native helper exists — and what keep the scaffold's derived values from changing by
accident through an edit made for the docs.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pytest

from decimalai import _frameworks as reg
from decimalai.cli import scaffold

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
REGISTRY_FILE = REPO_ROOT / "decimalai" / "_frameworks.py"
# The sibling docs checkout: absent in GitHub Actions, where the docs comparison skips
# (tests/conformance/test_coverage.py reads the same file the same way).
DOCS_FRAMEWORKS = REPO_ROOT.parent / "decimalai-docs" / "sdk" / "python" / "frameworks.mdx"
DOCS_CHANGELOG = REPO_ROOT.parent / "decimalai-docs" / "changelog.mdx"

#: What scaffold.py typed out by hand until 2026-10-18, when it began deriving these from
#: the registry. Equal by value, on purpose: a registry edit made for the docs or the
#: dashboard must not change what `decimalai init` does. When a change IS meant (a
#: framework gains a template), change this snapshot in the same commit. Anthropic
#: gained one in 0.23, the raw OpenAI SDK in 0.25.
SCAFFOLD_BEFORE = {
    "DEFAULT_MODEL": "gpt-4o-mini",
    "_FRAMEWORK_DEFAULT_MODELS": {
        "pydantic-ai": "openai:gpt-4o-mini", "adk": "gemini-3.5-flash",
        "anthropic": "claude-haiku-5-5",
    },
    "SUPPORTED_FRAMEWORKS": ("langchain", "openai-agents", "pydantic-ai", "adk", "anthropic",
                             "openai"),
    "UNSCAFFOLDED_WITH_SEAM": {"crewai": "CrewAI"},
    "NO_PROMPT_SEAM": {
        "llamaindex": "LlamaIndex",
        "claude-agent-sdk": "The Claude Agent SDK",
        "autogen": "AutoGen / AG2",
        "otel": "The generic OpenTelemetry rail",
    },
    "INSTALL": {
        "langchain": 'pip install "decimalai[langchain]" langchain-openai',
        "openai-agents": 'pip install "decimalai[openai-agents]"',
        "pydantic-ai": 'pip install "decimalai[pydantic-ai]"',
        "adk": 'pip install "decimalai[adk]"',
        "anthropic": 'pip install "decimalai[anthropic]"',
        "openai": 'pip install "decimalai[openai]"',
    },
    "EXTRA_MIN_PYTHON": {"adk": (3, 11), "pydantic-ai": (3, 11)},
    "ENV_VARS": {
        "langchain": ("DECIMAL_API_KEY", "OPENAI_API_KEY"),
        "openai-agents": ("DECIMAL_API_KEY", "OPENAI_API_KEY"),
        "pydantic-ai": ("DECIMAL_API_KEY", "OPENAI_API_KEY"),
        "adk": ("DECIMAL_API_KEY", "GOOGLE_API_KEY"),
        "anthropic": ("DECIMAL_API_KEY", "ANTHROPIC_API_KEY"),
        "openai": ("DECIMAL_API_KEY", "OPENAI_API_KEY"),
    },
}

#: `init()` opt-in flags that are not a framework: the raw provider-SDK rails (the
#: frameworks page covers them in prose, "No framework at all") and evidence export.
#: Same split as tests/conformance/test_coverage.py's PROVIDER_FLAG_DRIVERS.
NON_FRAMEWORK_FLAGS = {"openai", "anthropic", "google", "evidence"}


def _declared_extras() -> List[str]:
    """Extra names from [project.optional-dependencies], hand-parsed (3.10 has no tomllib)."""
    lines = PYPROJECT.read_text(encoding="utf-8").splitlines()
    start = lines.index("[project.optional-dependencies]") + 1
    names = []
    for line in lines[start:]:
        if line.startswith("["):
            break
        m = re.match(r"^([A-Za-z0-9._-]+)\s*=\s*\[", line)
        if m:
            names.append(m.group(1))
    return names


def _resolve(dotted: str):
    module, attr = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(module), attr)


# ── the scaffold reads the registry, and reads the same values ───────


@pytest.mark.parametrize("name", sorted(SCAFFOLD_BEFORE))
def test_scaffold_constants_are_what_they_were(name):
    assert getattr(scaffold, name) == SCAFFOLD_BEFORE[name]


def test_scaffold_order_is_the_registry_order():
    """`decimalai init` offers templates in registry order; langchain is the default."""
    assert scaffold.SUPPORTED_FRAMEWORKS[0] == "langchain"
    assert list(scaffold.SUPPORTED_FRAMEWORKS) == [
        f.key for f in reg.FRAMEWORKS if f.scaffold == reg.TEMPLATE]


def test_every_template_framework_has_a_renderer_and_a_usable_default_model():
    templates = {f.key for f in reg.FRAMEWORKS if f.scaffold == reg.TEMPLATE}
    assert set(scaffold._RENDERERS) == templates
    for f in reg.FRAMEWORKS:
        if f.scaffold == reg.TEMPLATE:
            assert f.default_model, f.key
            assert scaffold.normalize_model(f.key, None) == f.default_model
        else:
            assert f.default_model is None, f"{f.key}: a default model with no template to use it"


def test_init_framework_help_reads_the_scaffold_list():
    from decimalai.cli.main import init

    help_text = next(p.help for p in init.params if p.name == "framework")
    assert help_text == ("Framework to generate for: langchain (default), openai-agents, "
                         "pydantic-ai, adk, anthropic or openai")
    for key in scaffold.SUPPORTED_FRAMEWORKS:
        assert key in help_text


# ── every fact is true of the code ───────────────────────────────────


def test_keys_are_unique_and_statuses_known():
    keys = [f.key for f in reg.FRAMEWORKS]
    assert len(keys) == len(set(keys))
    assert all(f.scaffold in reg.SCAFFOLD_STATUSES for f in reg.FRAMEWORKS)
    assert reg.get("openai_agents") is reg.BY_KEY["openai-agents"]


def test_every_init_flag_is_a_real_init_parameter():
    import decimalai

    params = inspect.signature(decimalai.init).parameters
    for f in reg.FRAMEWORKS:
        if f.init_flag is None:
            assert f.init_flag_note, f"{f.key}: no flag, and no note saying why"
            continue
        p = params.get(f.init_flag)
        assert p is not None, f"{f.key}: decimalai.init() has no {f.init_flag!r} parameter"
        assert p.annotation in (bool, "bool") and p.default is False, f.init_flag


def test_every_framework_flag_on_init_is_in_the_registry():
    """A new `init(<framework>=True)` must get a registry entry, or the docs never list it."""
    import decimalai

    opt_in = {name for name, p in inspect.signature(decimalai.init).parameters.items()
              if p.annotation in (bool, "bool") and p.default is False}
    flags = {f.init_flag for f in reg.FRAMEWORKS if f.init_flag}
    assert opt_in - NON_FRAMEWORK_FLAGS == flags


def test_every_extra_is_declared_in_pyproject():
    extras = set(_declared_extras())
    assert extras, "parsed no extras out of pyproject.toml"
    for f in reg.FRAMEWORKS:
        if f.extra is None:
            assert f.extra_note or f.capability is None, f"{f.key}: no extra, and no note saying why"
            continue
        assert f.extra in extras, f"{f.key}: pyproject has no [{f.extra}] extra"


def test_install_lines_name_the_extra_and_survive_zsh():
    for f in reg.FRAMEWORKS:
        assert f.install.startswith("pip install "), f.key
        for token in f.install.split()[2:]:
            if "[" in token:  # zsh globs a bare bracket and pip never sees it
                assert token.startswith('"') and token.endswith('"'), f"{f.key}: {token}"
        if f.extra:
            assert f'"decimalai[{f.extra}]"' in f.install, f.key
        assert f.env_vars and f.env_vars[0] == "DECIMAL_API_KEY", f.key


def test_every_module_and_entry_point_exists():
    for f in reg.FRAMEWORKS:
        module = importlib.import_module(f.module)
        for name in re.findall(r"`([A-Za-z_]\w*)\(", f.entry_point):
            assert callable(getattr(module, name, None)), f"{f.module} has no {name}()"


def test_every_native_helper_exists():
    for f, n in reg.native_loaders():
        assert n.helpers, n.loader_id
        for dotted in n.helpers:
            assert callable(_resolve(dotted)), dotted
        if n.module != f.module:
            # its own module: the module reference needs all three, the native page names them
            assert n.requires and n.traced_by and n.use_for, n.loader_id


def test_native_loaders_are_the_native_loader_snapshot():
    from tests.native_loaders.loaders import NATIVE_LOADERS_SNAPSHOT

    ids = [n.loader_id for _, n in reg.native_loaders()]
    assert len(ids) == len(set(ids))
    assert set(ids) == set(NATIVE_LOADERS_SNAPSHOT)


def test_min_python_is_set_only_where_the_extra_needs_it():
    for f in reg.FRAMEWORKS:
        if f.min_python is not None:
            assert f.extra == f.key, "install_python_note names decimalai[<key>]"
            assert f.min_python > (3, 10)


# ── the docs' tables ─────────────────────────────────────────────────


def test_capability_table_advertises_the_conformance_snapshot():
    """The rendered first column parses, the way the coverage guard parses it, into the
    frameworks the conformance drivers cover. Runs without the docs checkout."""
    from tests.conformance.drivers import ADVERTISED_SNAPSHOT
    from tests.conformance.test_coverage import advertised_from_docs

    assert advertised_from_docs(reg.render_capability_table()) == set(ADVERTISED_SNAPSHOT)


def test_tables_have_one_row_per_entry_and_no_stray_pipes():
    cap = reg.render_capability_table().splitlines()
    assert len(cap) == 2 + sum(1 for f in reg.FRAMEWORKS if f.capability)
    mod = reg.render_module_reference().splitlines()
    own_module_loaders = sum(1 for f, n in reg.native_loaders() if n.module != f.module)
    assert len(mod) == 2 + sum(1 for f in reg.FRAMEWORKS if f.use_for) + own_module_loaders
    for table, header in ((cap, reg.CAPABILITY_HEADER), (mod, reg.MODULE_HEADER)):
        for line in table:
            assert line.count("|") == len(header) + 1, line
    with pytest.raises(ValueError):
        reg._row(("a|b",))


def test_skills_cells_name_every_native_helper():
    table = reg.render_capability_table()
    for _, n in reg.native_loaders():
        assert n.helpers[0] + "()" in table, n.loader_id


def test_replace_block_round_trips_on_a_page():
    page = ("intro\n\n" + reg.begin_marker("capability-table") + "\n\n| stale |\n\n"
            + reg.end_marker("capability-table") + "\n\noutro\n")
    assert reg.extract_block(page, "capability-table") == "| stale |\n"
    fresh = reg.replace_block(page, "capability-table")
    assert reg.extract_block(fresh, "capability-table") == reg.render_capability_table()
    assert fresh.startswith("intro\n\n") and fresh.endswith("\n\noutro\n")
    assert reg.replace_block(fresh, "capability-table") == fresh
    assert reg.extract_block(page, "module-reference") is None
    with pytest.raises(ValueError):
        reg.extract_block(page.replace(reg.end_marker("capability-table"), ""), "capability-table")


def _documented_release(changelog: Path) -> Optional[str]:
    """The newest SDK version the docs' changelog names: the release the docs describe.

    decimalai-docs' scripts/test_docs_sdk_drift.py reads it the same way: the newest
    ``<Update>`` that names a version (a platform-only entry names none)."""
    text = changelog.read_text(encoding="utf-8")
    for description in re.findall(r'<Update label="[^"]*" description="([^"]*)"', text):
        versions = [tuple(int(x) for x in v.split("."))
                    for v in re.findall(r"\b(\d+\.\d+\.\d+)\b", description)]
        if versions:
            return ".".join(str(x) for x in max(versions))
    return None


def _documented_registry(changelog: Path, tmp_path: Path, monkeypatch) -> Tuple[Any, str]:
    """The registry the docs page describes, and what it is.

    The page tells users what to install from PyPI and what to call, so it describes a
    release, not this tree. Main runs ahead of a release: held to the working tree, an SDK
    commit could land only by publishing an API ``pip install`` could not deliver yet (0.25's
    LangChain middleware landed before 0.25 shipped). So the release is the newest version the
    docs' changelog names, read from its tag, the rule the dashboard's frameworks.json has
    followed since 2026-10-19 (platform scripts/sync_framework_registry.py). A version the
    docs name with no tag here is the release being prepared, and its registry is this tree's.
    """
    version = _documented_release(changelog) if changelog.exists() else None
    if version is None:
        return reg, "this tree (the docs name no SDK version)"
    tag = f"v{version}"
    shown = subprocess.run(["git", "show", f"{tag}:decimalai/_frameworks.py"], cwd=REPO_ROOT,
                           capture_output=True, text=True)
    if shown.returncode != 0:
        return reg, f"this tree ({tag}, the release the docs name, is not tagged yet)"
    path = tmp_path / "_frameworks_released.py"
    path.write_text(shown.stdout, encoding="utf-8")
    name = "_frameworks_" + version.replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)   # @dataclass looks its module up there
    spec.loader.exec_module(module)
    return module, f"{tag}, the release the docs name"


def test_docs_generated_blocks_are_the_documented_releases(tmp_path, monkeypatch):
    """The docs page carries exactly what the registry of the release it documents renders.
    Skips without the docs checkout; decimalai-docs' scripts/check_generated_framework_tables.py
    makes the comparison from that side (`--write` regenerates), and a release's docs change
    regenerates the blocks (RELEASING.md, docs sync)."""
    if not DOCS_FRAMEWORKS.exists():
        pytest.skip(f"docs repo not on disk at {DOCS_FRAMEWORKS}")
    registry, source = _documented_registry(DOCS_CHANGELOG, tmp_path, monkeypatch)
    text = DOCS_FRAMEWORKS.read_text(encoding="utf-8")
    for name in registry.GENERATED_BLOCKS:
        have = registry.extract_block(text, name)
        assert have is not None, f"frameworks.mdx has no generated {name!r} block"
        assert have == registry.render_block(name), (
            f"frameworks.mdx's {name!r} block differs from the registry of {source}. In "
            f"decimalai-docs run `python3 scripts/check_generated_framework_tables.py --write` "
            f"with the docs for that release.")


# ── the JSON, and loading the file on its own ────────────────────────


def test_json_command_prints_the_registry():
    out = subprocess.run([sys.executable, "-m", "decimalai._frameworks", "--json"],
                         capture_output=True, text=True, check=True, cwd=REPO_ROOT).stdout
    assert out == reg.to_json()
    data = json.loads(out)
    assert data["schema_version"] == reg.SCHEMA_VERSION
    assert list(data["frameworks"]) == [f.key for f in reg.FRAMEWORKS]
    assert data["frameworks"]["openai-agents"]["install"] == scaffold.INSTALL["openai-agents"]
    block = subprocess.run([sys.executable, "-m", "decimalai._frameworks", "--block", "module-reference"],
                           capture_output=True, text=True, check=True, cwd=REPO_ROOT).stdout
    assert block == reg.render_module_reference()


def test_registry_imports_only_the_standard_library():
    """The docs checker and the app's check load this file by path, without
    decimalai's dependencies and without decimalai/__init__.py."""
    tree = ast.parse(REGISTRY_FILE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "a relative import: loading the file by path would fail"
            imported.add((node.module or "").split(".")[0])
    stdlib = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}
    if stdlib:
        assert imported <= stdlib, imported - stdlib


def test_registry_loads_by_path_like_its_consumers(tmp_path):
    script = tmp_path / "load.py"
    script.write_text(
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('fw', {str(REGISTRY_FILE)!r})\n"
        "mod = importlib.util.module_from_spec(spec); sys.modules['fw'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "assert 'decimalai' not in sys.modules, 'loading the registry imported decimalai'\n"
        "sys.stdout.write(mod.to_json())\n"
    )
    out = subprocess.run([sys.executable, "-I", str(script)], capture_output=True, text=True,
                         check=True, cwd=tmp_path).stdout
    assert out == reg.to_json()
