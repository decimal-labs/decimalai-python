"""Top-level decimalai.repair_preview / repair_apply wrappers.

Like the other top-level convenience wrappers, they fetch the client from
_config._get_client() (raising DecimalConfigError before init()) and delegate.
"""

from unittest.mock import MagicMock

import pytest


def test_repair_preview_requires_init():
    import decimalai
    from decimalai import _config
    from decimalai._config import DecimalConfigError

    _config._config = None
    _config._client = None
    with pytest.raises(DecimalConfigError):
        decimalai.repair_preview("m1", "m2")


def test_repair_apply_requires_init():
    import decimalai
    from decimalai import _config
    from decimalai._config import DecimalConfigError

    _config._config = None
    _config._client = None
    with pytest.raises(DecimalConfigError):
        decimalai.repair_apply("m1", "m2")


def test_repair_preview_delegates_to_client(mocker):
    import decimalai
    from decimalai import _config

    client = MagicMock()
    client.repair_preview.return_value = {"rules": [], "total_eligible": 0}
    mocker.patch.object(_config, "_get_client", return_value=client)

    out = decimalai.repair_preview("m1", "m2", sample_size=3)
    assert out["total_eligible"] == 0
    client.repair_preview.assert_called_once_with("m1", "m2", sample_size=3)


def test_repair_apply_delegates_to_client(mocker):
    import decimalai
    from decimalai import _config

    client = MagicMock()
    client.repair_apply.return_value = {"batch_id": "b1", "status": "completed"}
    mocker.patch.object(_config, "_get_client", return_value=client)

    out = decimalai.repair_apply("m1", "m2", approved_rule_indices=[1])
    assert out["batch_id"] == "b1"
    client.repair_apply.assert_called_once_with("m1", "m2", approved_rule_indices=[1])
