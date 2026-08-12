"""HuggingFace Hub integration — push datasets and load as HF Datasets.

Enables seamless interop with the open-source training ecosystem:
- Push versioned datasets to HF Hub (usable by Axolotl, Unsloth, TRL)
- Load datasets as huggingface `Dataset` objects for in-memory training

Requires: pip install huggingface_hub datasets
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("decimalai.integrations.huggingface")


def push_to_hub(
    dataset_id: str,
    repo_id: str,
    *,
    version: Optional[str] = None,
    token: Optional[str] = None,
    private: bool = True,
    commit_message: Optional[str] = None,
    split: str = "train",
) -> Dict[str, Any]:
    """Push a DecimalAI dataset to HuggingFace Hub.

    Once pushed, the dataset is loadable by any tool that supports
    ``load_dataset()`` — including Axolotl, Unsloth, and TRL.

    Args:
        dataset_id: The DecimalAI dataset ID.
        repo_id: HuggingFace repo in ``"org/dataset-name"`` format.
        version: Version specifier (``None``/``"latest"``, ``"v3"``, or UUID).
        token: HuggingFace API token. Falls back to ``HF_TOKEN`` env var
               or cached login (``huggingface-cli login``).
        private: Whether to create a private repo (default True).
        commit_message: Custom commit message. Defaults to a descriptive one.
        split: Dataset split name (default ``"train"``).

    Returns:
        Dict with ``repo_url``, ``commit_hash``, ``version_id``,
        ``row_count``, and ``repo_id``.

    Example::

        import decimalai
        decimalai.init()

        result = decimalai.push_to_hub(
            "ds_abc123",
            "my-org/support-agent-sft-v3",
        )
        print(f"Pushed to {result['repo_url']}")

        # Now loadable everywhere:
        # from datasets import load_dataset
        # ds = load_dataset("my-org/support-agent-sft-v3")
    """
    try:
        from huggingface_hub import (
            HfApi,  # create_repo is invoked as api.create_repo() below
        )
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for HF Hub integration. "
            "Install with: pip install huggingface_hub"
        )

    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError(
            "The datasets library is required for HF Hub push. "
            "Install with: pip install datasets"
        )

    from decimalai._config import _get_client

    client = _get_client()

    # Resolve version and download
    resolved_version = client.resolve_version_id(dataset_id, version)
    raw_data = client.export_dataset(dataset_id, resolved_version, format="jsonl")

    # Parse JSONL string to list of dicts
    rows = _parse_jsonl(raw_data)
    if not rows:
        raise ValueError(
            f"Dataset {dataset_id} version {resolved_version} has no rows. "
            f"Build a version first with client.build_dataset()."
        )

    # Create HF Dataset object
    hf_dataset = Dataset.from_list(rows)

    # Resolve token
    import os
    resolved_token = token or os.environ.get("HF_TOKEN")

    # Create repo if it doesn't exist
    api = HfApi(token=resolved_token)
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
    except Exception as e:
        logger.warning("Repo creation note: %s", e)

    # Build commit message
    if not commit_message:
        commit_message = (
            f"DecimalAI: dataset {dataset_id[:12]} "
            f"version {resolved_version[:12]}"
        )

    # Push to hub
    hf_dataset.push_to_hub(
        repo_id,
        split=split,
        token=resolved_token,
        commit_message=commit_message,
        private=private,
    )

    repo_url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info(
        "Pushed %d rows to %s (version=%s)",
        len(rows), repo_url, resolved_version[:12],
    )

    return {
        "repo_url": repo_url,
        "repo_id": repo_id,
        "row_count": len(rows),
        "version_id": resolved_version,
        "dataset_id": dataset_id,
        "split": split,
    }


def load_dataset(
    dataset_id: str,
    *,
    version: Optional[str] = None,
    split: str = "train",
) -> Any:
    """Load a DecimalAI dataset as a HuggingFace Dataset object.

    Returns a ``datasets.Dataset`` that can be plugged directly into
    TRL, Axolotl, Unsloth, or any HuggingFace-compatible trainer.

    Args:
        dataset_id: The DecimalAI dataset ID.
        version: Version specifier (``None``/``"latest"``, ``"v3"``, or UUID).
        split: Name to assign to the split (default ``"train"``).

    Returns:
        A ``datasets.Dataset`` object.

    Example::

        from decimalai.integrations.huggingface import load_dataset

        ds = load_dataset("ds_abc123", version="v2")
        print(ds)        # Dataset({features: ['messages'], num_rows: 500})
        print(ds[0])      # {'messages': [{'role': 'user', ...}, ...]}

        # Use with TRL's SFTTrainer
        from trl import SFTTrainer
        trainer = SFTTrainer(model=model, train_dataset=ds, ...)
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError(
            "The datasets library is required. "
            "Install with: pip install datasets"
        )

    from decimalai._config import _get_client

    client = _get_client()

    resolved_version = client.resolve_version_id(dataset_id, version)
    raw_data = client.export_dataset(dataset_id, resolved_version, format="jsonl")

    rows = _parse_jsonl(raw_data)
    if not rows:
        raise ValueError(
            f"Dataset {dataset_id} version {resolved_version} has no rows."
        )

    hf_dataset = Dataset.from_list(rows)
    logger.info(
        "Loaded %d rows from dataset %s (version=%s)",
        len(rows), dataset_id, resolved_version[:12],
    )
    return hf_dataset


def _parse_jsonl(data: Union[str, bytes, list]) -> List[Dict[str, Any]]:
    """Parse JSONL data into a list of dicts.

    Handles:
    - Raw JSONL string (from API)
    - List of dicts (already parsed)
    - Bytes
    """
    if isinstance(data, list):
        return data

    if isinstance(data, bytes):
        data = data.decode("utf-8")

    rows = []
    for line in data.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed JSONL line: %s...", line[:80])
    return rows
