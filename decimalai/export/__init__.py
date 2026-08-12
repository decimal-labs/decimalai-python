"""Dataset export utilities — download and write datasets to disk.

Provides high-level helpers for pulling datasets from the DecimalAI platform
and writing them to local files in training-ready formats (JSONL, Parquet).

Quick usage::

    import decimalai
    decimalai.init()

    # Pull latest version of a dataset to a local file
    result = decimalai.pull_dataset("ds_abc123", path="./training_data.jsonl")

    # Pull a specific version
    result = decimalai.pull_dataset(
        "ds_abc123",
        version="v2",
        format="parquet",
        path="./data.parquet",
    )
"""

from .jsonl import write_jsonl
from .parquet import write_parquet

__all__ = ["write_jsonl", "write_parquet"]
