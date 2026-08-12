"""`install()` → `instrument()` on the framework integrations.

"install" was doing double duty inside one package. On a framework module it
turned on TRACING; on `SkillRouter` it added a SKILL to a workspace. Two
unrelated actions under one word — and the skill sense is the one users arrive
with, because it is what every extension marketplace means by install. Somebody
reading `decimalai.langchain.install()` has no reason to think it is about
tracing.

`decimalai.providers.instrument()` already used the new name, so this makes the
package consistent rather than inventing a word for it.

The old name keeps working. These tests exist so that stays true: a rename that
silently breaks `install()` for everyone who already wrote it is worse than the
ambiguity it fixes.
"""
import importlib
import inspect
import warnings

import pytest

# Every framework module that had a module-level install().
MODULES = [
    "adk", "anthropic", "autogen", "claude_agent_sdk", "langchain",
    "llamaindex", "openai_agents", "otel", "pydantic_ai",
]


@pytest.mark.parametrize("name", MODULES)
def test_both_names_are_importable(name):
    """New name present, old name still there. Neither half is optional."""
    mod = importlib.import_module(f"decimalai.{name}")
    assert callable(getattr(mod, "instrument", None)), f"{name}.instrument() missing"
    assert callable(getattr(mod, "install", None)), (
        f"{name}.install() disappeared — every user who already wrote it breaks"
    )


@pytest.mark.parametrize("name", MODULES)
def test_the_old_name_warns_and_delegates(name):
    """The shim must actually FORWARD, not quietly do nothing.

    A deprecation shim that warns and returns None is the worst outcome
    available: tracing silently stops, and the warning looks like it explains it.
    """
    mod = importlib.import_module(f"decimalai.{name}")
    sentinel = object()
    called = {}

    def _fake(*a, **k):
        called["args"], called["kwargs"] = a, k
        return sentinel

    real, mod.instrument = mod.instrument, _fake
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = mod.install("positional", agent_name="x")
    finally:
        mod.instrument = real

    assert out is sentinel, f"{name}.install() did not return instrument()'s result"
    assert called["args"] == ("positional",)
    assert called["kwargs"] == {"agent_name": "x"}

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, f"{name}.install() did not warn"
    msg = str(deprecations[0].message)
    assert "instrument()" in msg, "the warning must name the replacement"
    assert name in msg, "the warning must name the module it is about"


@pytest.mark.parametrize("name", MODULES)
def test_the_real_work_moved_to_instrument(name):
    """`instrument` is the implementation; `install` is a thin forwarder.

    Guards the lazy version of this rename — aliasing `instrument = install` and
    leaving the body where it was. That reads identically from the outside and
    leaves the deprecated name as the thing every future edit lands in.
    """
    mod = importlib.import_module(f"decimalai.{name}")
    impl = inspect.getsource(mod.instrument)
    shim = inspect.getsource(mod.install)
    assert len(impl.splitlines()) > len(shim.splitlines()), (
        f"{name}.install() is longer than instrument() — the rename went backwards"
    )
    assert "DeprecationWarning" in shim
    assert "DeprecationWarning" not in impl, (
        f"{name}.instrument() warns — the new name must not tell people off"
    )


def test_init_reaches_instrument_not_the_deprecated_shim():
    """`decimalai.init(langchain=True)` must not trip its own warning.

    If the internal dispatch still imported `install`, every user of the
    one-liner would see a DeprecationWarning for something they never wrote.
    """
    src = inspect.getsource(importlib.import_module("decimalai").init)
    assert "import install as" not in src, (
        "init() still dispatches through the deprecated name"
    )
    assert "import instrument as" in src


def test_skillrouter_install_is_untouched():
    """The OTHER install — adding a skill — keeps its name.

    That one is not ambiguous; it means what a marketplace means. The whole point
    of the rename was to leave exactly one thing called install.
    """
    from decimalai.skill_router import SkillRouter
    assert callable(getattr(SkillRouter, "install", None))
