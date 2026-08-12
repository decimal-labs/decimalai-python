"""JSONL export — write dataset rows to newline-delimited JSON files.

Handles both in-memory data (from the API) and streaming writes for
large datasets. Supports OpenAI SFT chat-completion format, Alpaca,
ShareGPT, and raw JSONL.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

logger = logging.getLogger("decimalai.export.jsonl")


def write_jsonl(
    rows: Union[List[Dict[str, Any]], str],
    path: Union[str, Path],
    *,
    append: bool = False,
    ensure_newline: bool = True,
) -> Dict[str, Any]:
    """Write dataset rows to a JSONL file.

    Args:
        rows: Either a list of row dicts, or a raw JSONL string
              (as returned by the backend export endpoint).
        path: Output file path. Parent directories are created automatically.
        append: If True, append to existing file instead of overwriting.
        ensure_newline: If True, ensure the file ends with a newline.

    Returns:
        Summary dict with row_count, file_path, and bytes_written.

    Example::

        from decimalai.export import write_jsonl

        # From API response
        data = client.export_dataset("ds_abc", "v_xyz", format="jsonl")
        result = write_jsonl(data, "./training_data.jsonl")
        print(f"Wrote {result['row_count']} rows to {result['file_path']}")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    row_count = 0

    # Capture pre-existing size so bytes_written reports this call's delta,
    # not the whole file (which in append mode includes prior content).
    size_before = path.stat().st_size if (append and path.exists()) else 0

    with open(path, mode, encoding="utf-8") as f:
        if isinstance(rows, str):
            # Raw JSONL string from the backend
            content = rows.rstrip("\n")
            f.write(content)
            if ensure_newline and content:
                f.write("\n")
            row_count = content.count("\n") + 1 if content else 0
        elif isinstance(rows, list):
            for row in rows:
                line = json.dumps(row, ensure_ascii=False, default=str)
                f.write(line)
                f.write("\n")
                row_count += 1
        else:
            raise TypeError(
                f"Expected list of dicts or JSONL string, got {type(rows).__name__}"
            )

    bytes_written = path.stat().st_size - size_before
    logger.info("Wrote %d rows to %s (%d bytes)", row_count, path, bytes_written)

    return {
        "row_count": row_count,
        "file_path": str(path.resolve()),
        "bytes_written": bytes_written,
        "format": "jsonl",
    }
