"""Parquet export — write dataset rows to Apache Parquet files.

Requires ``pyarrow`` for writing. Falls back to a helpful error message
if not installed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

logger = logging.getLogger("decimalai.export.parquet")


def write_parquet(
    rows: Union[List[Dict[str, Any]], bytes],
    path: Union[str, Path],
) -> Dict[str, Any]:
    """Write dataset rows to a Parquet file.

    Args:
        rows: Either a list of row dicts (converted to Parquet via pyarrow),
              or raw Parquet bytes (as returned by the backend export endpoint).
        path: Output file path. Parent directories are created automatically.

    Returns:
        Summary dict with row_count, file_path, and bytes_written.

    Raises:
        ImportError: If ``pyarrow`` is not installed and rows is a list.

    Example::

        from decimalai.export import write_parquet

        # From API response (raw bytes)
        data = client.export_dataset("ds_abc", "v_xyz", format="parquet")
        result = write_parquet(data, "./training_data.parquet")
        print(f"Wrote {result['row_count']} rows")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(rows, bytes):
        # Raw Parquet bytes from the backend — write directly
        with open(path, "wb") as f:
            f.write(rows)

        file_size = path.stat().st_size
        row_count = _count_parquet_rows(path)

        logger.info("Wrote Parquet file to %s (%d bytes)", path, file_size)
        return {
            "row_count": row_count,
            "file_path": str(path.resolve()),
            "bytes_written": file_size,
            "format": "parquet",
        }

    if isinstance(rows, list):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "pyarrow is required for Parquet export from row data. "
                "Install it with: pip install pyarrow\n"
                "Alternatively, use format='jsonl' which has no extra dependencies."
            )

        if not rows:
            # Write an empty Parquet file with no columns
            table = pa.table({})
            pq.write_table(table, str(path))
            return {
                "row_count": 0,
                "file_path": str(path.resolve()),
                "bytes_written": path.stat().st_size,
                "format": "parquet",
            }

        # Flatten nested dicts/lists to JSON strings for Parquet compatibility
        flat_rows = []
        for row in rows:
            flat = {}
            for k, v in row.items():
                if isinstance(v, (dict, list)):
                    flat[k] = json.dumps(v, ensure_ascii=False, default=str)
                else:
                    flat[k] = v
            flat_rows.append(flat)

        table = pa.Table.from_pylist(flat_rows)
        pq.write_table(table, str(path))

        file_size = path.stat().st_size
        logger.info(
            "Wrote %d rows to %s (%d bytes)", len(rows), path, file_size
        )
        return {
            "row_count": len(rows),
            "file_path": str(path.resolve()),
            "bytes_written": file_size,
            "format": "parquet",
        }

    raise TypeError(
        f"Expected list of dicts or raw Parquet bytes, got {type(rows).__name__}"
    )


def _count_parquet_rows(path: Path) -> int:
    """Try to read the row count from a Parquet file, return -1 if unavailable."""
    try:
        import pyarrow.parquet as pq
        metadata = pq.read_metadata(str(path))
        return metadata.num_rows
    except (ImportError, Exception):
        return -1
