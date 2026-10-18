"""The child process: run ONE native-mode scenario, write its JSON, exit.

    python tests/native/_child.py <tests.native.module:function> <ctx.json> <out.json>

Started by ``_runner.run_scenario``; runnable by hand to debug one framework.
"""

from __future__ import annotations

import importlib
import json
import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

# ``python <script>`` puts this directory first on sys.path; import the suite as a package.
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != _HERE]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main(argv: list) -> int:
    if len(argv) != 4 or ":" not in argv[1]:
        print(f"usage: {argv[0]} <module:function> <ctx.json> <out.json>", file=sys.stderr)
        return 2
    target, ctx_path, out_path = argv[1:]
    module_name, func_name = target.split(":", 1)
    ctx = json.loads(Path(ctx_path).read_text())
    try:
        result = getattr(importlib.import_module(module_name), func_name)(ctx)
    except BaseException as exc:  # noqa: BLE001 - a crash is reported, never swallowed
        result = {"crashed": f"{type(exc).__name__}: {str(exc)[:500]}",
                  "traceback": traceback.format_exc()[-6000:]}
    Path(out_path).write_text(json.dumps(result, default=str, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
