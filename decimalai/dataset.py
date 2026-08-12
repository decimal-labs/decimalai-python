"""Dataset primitive — a named, iterable wrapper over example collections.

A `Dataset` is just a labelled iterable of example dicts (each with at
least an `input` field; `id` + `expected` optional). Constructors:

- `Dataset.from_jsonl(path)` — load from a `.jsonl` file (one example per line).
- `Dataset(rows=[...], name="...")` — direct construction from a list.

Future constructors (deferred — require backend round-trip):
- `Dataset.from_manifest(manifest_id)` — pull all traces tagged to a manifest.
- `Dataset.from_trace_query(filter)` — pull traces matching a query.

The type exists so eval code can pass "the examples" around as one object
with a stable name, instead of a bare list plus a string in a variable
somewhere else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class Dataset:
    """Labelled list of evaluation examples.

    Attributes:
        rows: list of example dicts. Each typically has `input`, optional
            `id`, `expected`, plus any free-form metadata.
        name: human-readable label used in eval reports.
    """

    rows: List[Dict[str, Any]] = field(default_factory=list)
    name: str = "dataset"

    @classmethod
    def from_jsonl(cls, path: str, name: Optional[str] = None) -> "Dataset":
        """Load a Dataset from a `.jsonl` file (one JSON example per line).

        Empty lines and comment lines starting with `#` are skipped so users
        can hand-annotate JSONL files. Invalid JSON lines raise immediately
        rather than silently dropping data.
        """
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
                if not isinstance(rec, dict):
                    raise ValueError(f"{path}:{lineno}: expected object, got {type(rec).__name__}")
                rows.append(rec)
        return cls(rows=rows, name=name or Path(path).stem)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]
