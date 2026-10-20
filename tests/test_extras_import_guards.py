"""Framework deps are extras — absent-dep paths must name their extra.

Frameworks ARE installed in the dev venv, so each test simulates absence by
poisoning sys.modules with None (the import system raises ImportError when it
finds None under a module's name — blocking the top-level package blocks all
submodule imports too). Idempotency globals are reset via monkeypatch so
instrument() actually reaches its import guard.
"""

import sys

import pytest


def _block(monkeypatch, *names):
    for name in names:
        # Drop already-imported submodules so the block is airtight, then
        # poison the top-level name.
        for mod in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
            monkeypatch.delitem(sys.modules, mod, raising=False)
        monkeypatch.setitem(sys.modules, name, None)


class TestRaisingGuards:
    """instrument() paths that hard-require the framework raise, naming the extra."""

    def test_langchain_install_names_extra(self, monkeypatch):
        import decimalai.langchain as lc

        _block(monkeypatch, "langchain_core")
        monkeypatch.setattr(lc, "_installed", False)
        with pytest.raises(ImportError, match=r'decimalai\[langchain\]'):
            lc.instrument()

    def test_openai_agents_install_names_extra(self, monkeypatch):
        import decimalai.openai_agents as oa

        _block(monkeypatch, "agents")
        with pytest.raises(ImportError, match=r'decimalai\[openai-agents\]'):
            oa.instrument()

    def test_llamaindex_install_names_extra(self, monkeypatch):
        import decimalai.llamaindex as li

        _block(monkeypatch, "llama_index")
        with pytest.raises(ImportError, match=r'decimalai\[llamaindex\]'):
            li.instrument()

    def test_evals_llm_call_names_extra(self, monkeypatch):
        from decimalai.evals import llm_evaluators

        _block(monkeypatch, "litellm")
        with pytest.raises(ImportError, match=r'decimalai\[evals\]'):
            llm_evaluators._call_llm("hi", model="gpt-4o-mini")


class TestWarningGuards:
    """Soft-fail paths warn (never raise), still naming the extra."""

    def test_claude_agent_sdk_install_warns_with_extra(self, monkeypatch, caplog):
        import decimalai.claude_agent_sdk as cas

        _block(monkeypatch, "claude_agent_sdk")
        monkeypatch.setattr(cas, "_query_patched", False)
        with caplog.at_level("WARNING"):
            cas.instrument()
        assert 'decimalai[claude-agent-sdk]' in caplog.text

    def test_adk_install_warns_with_extra(self, monkeypatch, caplog):
        import decimalai.adk as adk

        _block(monkeypatch, "google.adk")
        monkeypatch.setattr(adk, "_runner_patched", False)
        with caplog.at_level("WARNING"):
            adk.instrument()
        assert 'decimalai[adk]' in caplog.text

    def test_pydantic_ai_load_agent_names_the_extra_before_any_call(self, monkeypatch):
        """Pydantic AI's skills come from load_agent(), which needs the framework
        outright: it raises, naming the extra, before init() or any request."""
        import decimalai._agent as agent_mod
        import decimalai.pydantic_ai as pai

        _block(monkeypatch, "pydantic_ai")
        monkeypatch.setattr(agent_mod, "_ensure_initialized",
                            lambda: pytest.fail("set up before the import check"))
        with pytest.raises(ImportError, match=r'decimalai\[pydantic-ai\]'):
            pai.load_agent("support")

    def test_langchain_loader_warns_with_extra(self, monkeypatch, caplog):
        import decimalai.langchain as lc

        _block(monkeypatch, "langchain_core")
        monkeypatch.setattr(lc, "_skill_loader_installed", False)
        with caplog.at_level("WARNING"):
            lc._install_skill_loader()
        assert 'decimalai[langchain]' in caplog.text

    def test_openai_agents_load_agent_names_the_extra_before_any_call(self, monkeypatch):
        """Since 0.18 the skills come from load_agent(), which needs the SDK
        outright: it raises, naming the extra, before init() or any request."""
        import decimalai.openai_agents as oa

        _block(monkeypatch, "agents")
        monkeypatch.setattr(oa, "_ensure_initialized",
                            lambda: pytest.fail("set up before the import check"))
        with pytest.raises(ImportError, match=r'decimalai\[openai-agents\]'):
            oa.load_agent("support")


class TestCoreImportSurvivesMissingFrameworks:
    """`import decimalai` and its lazy attrs work with ALL framework deps absent."""

    def test_package_imports_clean(self, monkeypatch):
        import importlib

        mods = [
            "decimalai.langchain",
            "decimalai.openai_agents",
            "decimalai.llamaindex",
            "decimalai.pydantic_ai",
            "decimalai.claude_agent_sdk",
            "decimalai.adk",
            "decimalai.evals.llm_evaluators",
        ]
        # Pin the parent-package attributes to today's module objects so
        # monkeypatch teardown restores module identity (other tests patch
        # attrs on these module objects and must keep seeing the same ones).
        for mod in mods:
            parent_name, attr = mod.rsplit(".", 1)
            parent = importlib.import_module(parent_name)
            importlib.import_module(mod)
            monkeypatch.setattr(parent, attr, getattr(parent, attr))

        _block(
            monkeypatch,
            "langchain_core",
            "agents",
            "openai",
            "llama_index",
            "pydantic_ai",
            "claude_agent_sdk",
            "litellm",
            "langgraph",
        )
        # google.adk is a namespace child — block just it, not all of google.
        _block(monkeypatch, "google.adk")
        # Force fresh imports of the adapter modules under the block so a
        # module-level framework import regression is caught here.
        for mod in mods:
            monkeypatch.delitem(sys.modules, mod, raising=False)
            importlib.import_module(mod)
