"""Registry skills on disk, managed like packages.

A project lists the skills it wants in ``.decimal/skills.toml``. The engine installs each one
as a real folder, ``.agents/skills/<install-name>/``, and records exactly what it wrote in
``.decimal/skills.lock``. The ``decimalai skills`` commands and this Python API are the same
engine::

    from decimalai import skillpkg

    skillpkg.add("acme/refund-policy")
    skillpkg.install(frozen=True)       # CI and Docker: the locked bytes, or a refusal
    skillpkg.update()
    skillpkg.status()
    skillpkg.verify()
    skillpkg.remove("acme-refund-policy")
    skillpkg.migrate()                  # a version 1 lock -> manifest and version 2 lock

Each call returns a :class:`Result` of ``changes``, ``problems`` and an ``exit_code`` from the
documented table in :mod:`decimalai.skillpkg.errors`, which also holds the exceptions.

The names are imported lazily: importing this package loads none of the modules behind them,
and none of those modules imports an agent framework.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, List

#: Public name -> the module of this package that defines it.
#:
#: No module of this package may share a name with an export. Importing a submodule binds it
#: as an attribute of the package, so a ``verify.py`` would replace the ``verify`` function
#: with the module the first time anything imported it, and ``skillpkg.verify()`` would raise
#: ``TypeError: 'module' object is not callable``. That is why the verifier lives in
#: ``verifier`` and the lock migration in ``migration``.
_EXPORTS: Dict[str, str] = {
    "add": "ops",
    "install": "ops",
    "update": "ops",
    "remove": "ops",
    "status": "ops",
    "verify": "verifier",
    "migrate": "migration",
    "Result": "errors",
}

__all__ = ["add", "install", "update", "remove", "status", "verify", "migrate", "Result"]


def __getattr__(name: str) -> Any:
    """Import the module behind a public name when the name is first used (PEP 562).

    Nothing is cached here, so the value is always the owning module's current attribute.
    """
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))
