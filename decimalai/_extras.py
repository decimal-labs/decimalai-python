"""The ImportError a framework module raises when its extra installed nothing.

Two extras carry a ``python_version >= '3.11'`` marker in pyproject.toml (``adk``, ``pydantic-ai``):
their frameworks need Python 3.11. On 3.10, ``pip install "decimalai[adk]"`` succeeds and installs no
framework, so "pip install decimalai[adk]" would send the reader round in a circle. The message names
the real requirement instead.
"""
from __future__ import annotations

import sys
from typing import Tuple

#: The extras whose framework needs Python 3.11 or newer, by the name a reader types.
PY311_EXTRAS = frozenset({"adk", "pydantic-ai"})


def framework_missing(what: str, framework: str, extra: str,
                      version: Tuple[int, ...] = tuple(sys.version_info[:2])) -> ImportError:
    """``what`` needs ``framework``: the install line, or the Python version it needs."""
    if extra in PY311_EXTRAS and version < (3, 11):
        return ImportError(f"{what} needs {framework}, which needs Python 3.11 or newer; "
                           f"this is Python {version[0]}.{version[1]}.")
    return ImportError(f'{what} needs {framework}: pip install "decimalai[{extra}]"')
