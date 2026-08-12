"""Unit tests for decimalai SDK core: init(), @trace, start_trace()."""

import os
import pytest
from unittest.mock import MagicMock, patch


class TestInit:
    """Tests for decimalai.init()."""

    def setup_method(self):
        """Reset global state before each test."""
        import decimalai._config as cfg
        cfg._config = None
        cfg._client = None
        # Clear env vars
        os.environ.pop("DECIMAL_API_KEY", None)
        os.environ.pop("DECIMAL_BASE_URL", None)

    def test_init_with_explicit_params(self):
        import decimalai
        decimalai.init(api_key="dai_sk_test", base_url="http://localhost:8000")

        from decimalai._config import _get_config, _get_client
        config = _get_config()
        assert config.api_key == "dai_sk_test"
        assert config.base_url == "http://localhost:8000"
        assert config.enabled is True

        client = _get_client()
        assert client is not None

    def test_init_from_env_vars(self):
        os.environ["DECIMAL_API_KEY"] = "dai_sk_from_env"
        os.environ["DECIMAL_BASE_URL"] = "http://envhost:9000"

        import decimalai
        # envhost:9000 is a synthetic fixture URL, not a reachable backend.
        # The init-time verify probe would raise DecimalConfigError on the DNS
        # failure; this test is about env-var resolution, not connectivity,
        # so opt out.
        decimalai.init(verify=False)

        from decimalai._config import _get_config
        config = _get_config()
        assert config.api_key == "dai_sk_from_env"
        assert config.base_url == "http://envhost:9000"

    def test_init_missing_api_key_raises(self):
        import decimalai
        from decimalai._config import DecimalConfigError

        with pytest.raises(DecimalConfigError, match="No API key"):
            decimalai.init()

    def test_init_disabled_mode(self):
        import decimalai
        decimalai.init(api_key="", enabled=False)

        from decimalai._config import _is_enabled
        assert _is_enabled() is False

    def test_init_default_base_url(self):
        import decimalai
        decimalai.init(api_key="dai_sk_test")

        from decimalai._config import _get_config
        config = _get_config()
        assert config.base_url == "https://api.decimal.ai"

    def test_init_with_project(self):
        import decimalai
        decimalai.init(api_key="dai_sk_test", project="my-project")

        from decimalai._config import _get_config
        config = _get_config()
        assert config.project == "my-project"


class TestTrace:
    """Tests for the @trace decorator and start_trace() context manager."""

    def setup_method(self):
        """Init the SDK with a mock client."""
        import decimalai._config as cfg
        from decimalai._config import DecimalConfig

        cfg._config = DecimalConfig(
            api_key="dai_sk_test",
            base_url="http://localhost:8000",
            enabled=True,
        )
        cfg._client = MagicMock()

        # Reset module-level manifest state so stale MagicMock values
        # from previous tests don't cause Pydantic validation errors
        import decimalai.generic as _gen
        _gen._manifest_id = None
        _gen._manifest_tracker = _gen.ManifestTracker()

    def test_trace_decorator_sends_on_return(self):
        import decimalai
        import decimalai._config as cfg

        @decimalai.trace(agent_name="test-agent")
        def my_func(query):
            return "hello"

        result = my_func("world")
        assert result == "hello"

        # Flush background sender before asserting
        from decimalai._config import _sender
        _sender.flush()

        # Verify ingest_trace was called
        cfg._client.ingest_trace.assert_called_once()
        trace = cfg._client.ingest_trace.call_args[0][0]
        assert trace.agent_name == "test-agent"
        assert len(trace.spans) == 0  # no manual spans logged
        assert trace.status.value == "success"

    def test_trace_decorator_captures_input_output(self):
        import decimalai
        import decimalai._config as cfg

        @decimalai.trace(agent_name="test")
        def my_func(query):
            return "response text"

        my_func("input text")

        from decimalai._config import _sender
        _sender.flush()

        trace = cfg._client.ingest_trace.call_args[0][0]
        assert trace.user_input_preview == "input text"
        assert trace.final_output_preview == "response text"

    def test_trace_decorator_handles_errors(self):
        import decimalai
        import decimalai._config as cfg

        @decimalai.trace(agent_name="error-agent")
        def failing_func():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            failing_func()

        # Trace should still be sent with error status
        from decimalai._config import _sender
        _sender.flush()
        cfg._client.ingest_trace.assert_called_once()
        trace = cfg._client.ingest_trace.call_args[0][0]
        assert trace.status.value == "error"

    def test_trace_decorator_auto_send_false(self):
        import decimalai
        import decimalai._config as cfg

        @decimalai.trace(agent_name="manual", auto_send=False)
        def my_func():
            return "ok"

        my_func()
        # Should NOT have sent
        cfg._client.ingest_trace.assert_not_called()

    def test_start_trace_context_manager(self):
        import decimalai
        import decimalai._config as cfg

        with decimalai.start_trace(agent_name="ctx-agent") as ctx:
            ctx.set_input("hello")
            ctx.log_llm_call(
                model="gpt-4o",
                input=[{"role": "user", "content": "hello"}],
                output={"content": "world"},
                input_tokens=10,
                output_tokens=5,
            )
            ctx.log_tool_call(
                name="search",
                input="query",
                output="results",
            )
            ctx.set_output("final answer")

        from decimalai._config import _sender
        _sender.flush()
        cfg._client.ingest_trace.assert_called_once()
        trace = cfg._client.ingest_trace.call_args[0][0]
        assert trace.agent_name == "ctx-agent"
        assert len(trace.llm_calls) == 1
        assert len(trace.spans) == 1  # tool call creates a span
        assert trace.llm_calls[0].model_name == "gpt-4o"
        assert trace.llm_calls[0].input_tokens == 10
        assert trace.spans[0].name == "search"
        assert trace.user_input_preview == "hello"
        assert trace.final_output_preview == "final answer"

    def test_start_trace_error_handling(self):
        import decimalai
        import decimalai._config as cfg

        with pytest.raises(RuntimeError, match="fail"):
            with decimalai.start_trace(agent_name="err") as ctx:
                raise RuntimeError("fail")

        from decimalai._config import _sender
        _sender.flush()
        cfg._client.ingest_trace.assert_called_once()
        trace = cfg._client.ingest_trace.call_args[0][0]
        assert trace.status.value == "error"

    def test_log_llm_call_outside_trace_raises(self):
        import decimalai
        from decimalai._config import DecimalConfigError

        with pytest.raises(DecimalConfigError, match="No active trace"):
            decimalai.log_llm_call(model="gpt-4o")

    def test_log_tool_call_outside_trace_raises(self):
        import decimalai
        from decimalai._config import DecimalConfigError

        with pytest.raises(DecimalConfigError, match="No active trace"):
            decimalai.log_tool_call(name="search")


class TestCallbackHandler:
    """Tests for decimalai.langchain.CallbackHandler."""

    def setup_method(self):
        import decimalai._config as cfg
        from decimalai._config import DecimalConfig

        cfg._config = DecimalConfig(
            api_key="dai_sk_test",
            base_url="http://localhost:8000",
            enabled=True,
        )
        cfg._client = MagicMock()

    def test_handler_creates_spans(self):
        from decimalai.langchain import CallbackHandler
        from uuid import uuid4

        handler = CallbackHandler(agent_name="test", auto_send=False)

        run_id = uuid4()
        handler.on_chain_start(
            {"name": "AgentExecutor"},
            {"input": "hello"},
            run_id=run_id,
        )
        handler.on_chain_end(
            {"output": "world"},
            run_id=run_id,
        )

        trace = handler.get_trace()
        assert len(trace.spans) == 1
        assert trace.spans[0].name == "AgentExecutor"
        assert trace.spans[0].status.value == "success"

    def test_handler_skips_noisy_chains(self):
        from decimalai.langchain import CallbackHandler
        from uuid import uuid4

        handler = CallbackHandler(auto_send=False)

        for name in ["RunnableSequence", "RunnableLambda", "RunnableAssign<x>"]:
            handler.on_chain_start(
                {"name": name},
                {},
                run_id=uuid4(),
            )

        trace = handler.get_trace()
        assert len(trace.spans) == 0

    def test_handler_auto_send_on_root_end(self):
        from decimalai.langchain import CallbackHandler
        from uuid import uuid4
        import decimalai._config as cfg

        handler = CallbackHandler(agent_name="auto", auto_send=True)

        root_id = uuid4()
        handler.on_chain_start(
            {"name": "AgentExecutor"},
            {"input": "test"},
            run_id=root_id,
        )
        handler.on_chain_end(
            {"output": "result"},
            run_id=root_id,
        )

        # Should have auto-sent (via background sender)
        from decimalai._config import _sender
        _sender.flush()
        cfg._client.ingest_trace.assert_called_once()
        trace = cfg._client.ingest_trace.call_args[0][0]
        assert trace.agent_name == "auto"

    def test_handler_reset_after_get_trace(self):
        from decimalai.langchain import CallbackHandler
        from uuid import uuid4

        handler = CallbackHandler(auto_send=False)

        handler.on_chain_start(
            {"name": "Chain1"},
            {},
            run_id=uuid4(),
        )
        trace1 = handler.get_trace()
        assert len(trace1.spans) == 1

        trace2 = handler.get_trace()
        assert len(trace2.spans) == 0  # reset


class TestLangchainInstall:
    """Tests for decimalai.langchain.install() — global tracing registration."""

    def setup_method(self):
        import decimalai._config as cfg
        from decimalai._config import DecimalConfig

        cfg._config = DecimalConfig(
            api_key="dai_sk_test",
            base_url="http://localhost:8000",
            enabled=True,
        )
        cfg._client = MagicMock()

        # Reset the install state so each test starts fresh
        import decimalai.langchain as lc_mod
        lc_mod._installed = False

    def test_install_registers_hook(self):
        """install() should call register_configure_hook with the callback var."""
        with patch(
            "decimalai.langchain.register_configure_hook",
            create=True,
        ) as mock_hook:
            # Patch the import inside install()
            with patch.dict("sys.modules", {
                "langchain_core": MagicMock(),
                "langchain_core.tracers": MagicMock(),
                "langchain_core.tracers.context": MagicMock(
                    register_configure_hook=mock_hook
                ),
            }):
                from decimalai.langchain import install, _decimal_callback_var
                install()

                mock_hook.assert_called_once_with(
                    _decimal_callback_var, inheritable=True
                )

    def test_install_is_idempotent(self):
        """Calling install() twice should only register once."""
        mock_hook = MagicMock()

        with patch.dict("sys.modules", {
            "langchain_core": MagicMock(),
            "langchain_core.tracers": MagicMock(),
            "langchain_core.tracers.context": MagicMock(
                register_configure_hook=mock_hook
            ),
        }):
            from decimalai.langchain import install
            install()
            install()  # second call should be no-op

            mock_hook.assert_called_once()

    def test_install_sets_auto_send_handler(self):
        """install() should create a CallbackHandler with auto_send=True."""
        mock_hook = MagicMock()

        with patch.dict("sys.modules", {
            "langchain_core": MagicMock(),
            "langchain_core.tracers": MagicMock(),
            "langchain_core.tracers.context": MagicMock(
                register_configure_hook=mock_hook
            ),
        }):
            from decimalai.langchain import install, _decimal_callback_var
            install(agent_name="global-agent")

            handler = _decimal_callback_var.get()
            assert handler is not None
            assert handler.auto_send is True
            assert handler.agent_name == "global-agent"

    def test_install_without_langchain_raises(self):
        """install() should raise ImportError if langchain-core is not installed."""
        import sys

        # Remove langchain_core from sys.modules to simulate it not being installed
        saved = {}
        for key in list(sys.modules.keys()):
            if "langchain_core" in key:
                saved[key] = sys.modules.pop(key)

        try:
            with patch.dict("sys.modules", {"langchain_core.tracers.context": None}):
                from decimalai.langchain import install
                with pytest.raises(ImportError, match="langchain-core"):
                    install()
        finally:
            sys.modules.update(saved)
