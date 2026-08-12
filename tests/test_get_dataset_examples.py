"""get_dataset_examples() must hit a route that actually exists.

The historical implementation called GET /api/v1/datasets/{id}/examples and
/datasets/{id}/versions/{vid}/examples — NEITHER of which exists on the
backend (both 404/400 live). Example rows are served by the version *export*
route as JSONL. These tests lock the repointed behaviour:

  * the method resolves the version and calls the export route with
    format=jsonl (never the dead /examples paths),
  * the JSONL body is parsed into a list of example dicts,
  * the return is a structured dict (dataset_id / version_id / count /
    examples) rather than a raw 404-prone passthrough.
"""

import json
from unittest.mock import MagicMock, patch

from decimalai._client import DecimalAIClient

# A full version UUID short-circuits resolve_version_id (no get_dataset call).
_VERSION_UUID = "11111111-2222-3333-4444-555555555555"
_DATASET_ID = "ds_abc123"


def _make_client():
    return DecimalAIClient(
        api_key="dai_sk_test",
        base_url="http://localhost:8000",
    )


def _jsonl_response(*rows):
    fake = MagicMock()
    fake.text = "\n".join(json.dumps(r) for r in rows)
    fake.raise_for_status.return_value = None
    return fake


def test_calls_export_route_not_dead_examples_route():
    """Explicit version → GET .../versions/{vid}/export?format=jsonl."""
    client = _make_client()
    fake = _jsonl_response(
        {"messages": [{"role": "user", "content": "hi"}]},
        {"messages": [{"role": "user", "content": "bye"}]},
    )
    with patch.object(client._http, "get", return_value=fake) as mock_get:
        client.get_dataset_examples(_DATASET_ID, version_id=_VERSION_UUID)

    mock_get.assert_called_once_with(
        f"/api/v1/datasets/{_DATASET_ID}/versions/{_VERSION_UUID}/export",
        params={"format": "jsonl"},
    )
    # crucially, the dead /examples path is never used
    called_url = mock_get.call_args[0][0]
    assert called_url.endswith("/export")
    assert "/examples" not in called_url
    client.close()


def test_parses_jsonl_into_examples_list():
    """The JSONL body is parsed into structured rows + a count."""
    client = _make_client()
    fake = _jsonl_response(
        {"messages": [{"role": "user", "content": "a"}]},
        {"messages": [{"role": "user", "content": "b"}], "tools": [{"x": 1}]},
    )
    with patch.object(client._http, "get", return_value=fake):
        result = client.get_dataset_examples(_DATASET_ID, version_id=_VERSION_UUID)

    assert result["dataset_id"] == _DATASET_ID
    assert result["version_id"] == _VERSION_UUID
    assert result["count"] == 2
    assert len(result["examples"]) == 2
    assert result["examples"][0]["messages"][0]["content"] == "a"
    assert result["examples"][1]["tools"] == [{"x": 1}]
    client.close()


def test_blank_lines_are_skipped():
    """Trailing/blank lines in the JSONL must not produce empty rows."""
    client = _make_client()
    fake = MagicMock()
    fake.text = '{"messages": []}\n\n   \n'
    fake.raise_for_status.return_value = None
    with patch.object(client._http, "get", return_value=fake):
        result = client.get_dataset_examples(_DATASET_ID, version_id=_VERSION_UUID)
    assert result["count"] == 1
    client.close()


def test_latest_version_resolves_then_exports():
    """version_id=None resolves the latest version, then exports it."""
    client = _make_client()
    fake = _jsonl_response({"messages": []})
    with patch.object(client._http, "get", return_value=fake) as mock_get, patch.object(
        client,
        "resolve_version_id",
        return_value=_VERSION_UUID,
    ) as mock_resolve:
        result = client.get_dataset_examples(_DATASET_ID)

    mock_resolve.assert_called_once_with(_DATASET_ID, None)
    mock_get.assert_called_once_with(
        f"/api/v1/datasets/{_DATASET_ID}/versions/{_VERSION_UUID}/export",
        params={"format": "jsonl"},
    )
    assert result["version_id"] == _VERSION_UUID
    client.close()
