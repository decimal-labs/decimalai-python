"""Lock in: push_* eval-adapter functions don't require user-passed client.

The client parameter used to leak into the public API: calling the
module-level `decimalai.push_custom_scores(trace_id, scores, source)`
raised `TypeError: missing 1 required positional argument: 'client'`.
The wrappers now auto-fetch the client from `_config._get_client()`,
mirroring the `flush()` wrapper pattern.
"""

import inspect

import pytest


def test_push_custom_scores_does_not_require_client_arg():
    """Signature must not have `client` as required."""
    import decimalai
    sig = inspect.signature(decimalai.push_custom_scores)
    params = sig.parameters
    assert "client" in params
    # client should have a default value (Optional[Any] = None) — NOT required
    assert params["client"].default is None


def test_push_deepeval_results_does_not_require_client_arg():
    import decimalai
    sig = inspect.signature(decimalai.push_deepeval_results)
    assert "client" in sig.parameters
    assert sig.parameters["client"].default is None


def test_push_langsmith_scores_does_not_require_client_arg():
    import decimalai
    sig = inspect.signature(decimalai.push_langsmith_scores)
    assert "client" in sig.parameters
    assert sig.parameters["client"].default is None


def test_raw_adapter_still_requires_client():
    """The advanced raw form `decimalai.evals.adapters.push_*` still
    requires client — only the top-level wrapper is convenience.
    """
    from decimalai.evals.adapters import push_custom_scores as raw
    sig = inspect.signature(raw)
    # The first param IS client and has no default
    first_param = list(sig.parameters.values())[0]
    assert first_param.name == "client"
    assert first_param.default is inspect.Parameter.empty


def test_push_custom_scores_uses_global_client_when_none_passed(mocker):
    """When `client=None`, the wrapper fetches from `_config._get_client()`."""
    import decimalai
    from decimalai import _config

    mock_client = mocker.MagicMock()
    mock_client.push_eval_scores.return_value = {"status": "ok"}

    mocker.patch.object(_config, "_get_client", return_value=mock_client)

    result = decimalai.push_custom_scores(
        trace_id="tr-1",
        scores=[{"name": "x", "score": 0.5}],
    )
    assert result == {"status": "ok"}
    # The adapter normalizes then forwards to client.push_eval_scores
    assert mock_client.push_eval_scores.called
