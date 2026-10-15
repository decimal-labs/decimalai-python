"""The child process: run ONE loader against ONE project, write the observation, exit.

    python tests/native_loaders/_child.py <loader> <observe|discover> <ctx.json> <out.json>

Started by ``runner.run_child`` with the project as its working directory and an
environment built for that one run (``runner.child_env``). Not a test module, but
runnable by hand when one loader is being debugged.

Why a process per loader: CrewAI's event bus, Strands' plugin registry, ADK's
runners and every OpenTelemetry provider are process-global, so loaders sharing a
process decide what each other can observe. The parent imports no framework.

Nothing is graded here; the observation goes back to the parent, where
``contract.py`` is the only place assertions live.
"""

from __future__ import annotations

import importlib
import json
import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # .../tests/native_loaders
_TESTS = _HERE.parent                            # .../tests
_REPO = _TESTS.parent

# ``python <script>`` puts the script's own directory first on sys.path, which would
# make this directory's modules (``source``, ``oracle``, ``matrix``...) importable
# under bare names a framework might also use. Import the suite as a package instead.
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != _HERE]
for _p in (str(_REPO), str(_TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from native_loaders.loaders import LOADERS_BY_NAME  # noqa: E402
from native_loaders.loaders._observe import Ctx, describe  # noqa: E402


def main(argv: list) -> int:
    if len(argv) != 5 or argv[2] not in ("observe", "discover"):
        print(f"usage: {argv[0]} <loader> <observe|discover> <ctx.json> <out.json>", file=sys.stderr)
        return 2
    name, mode, ctx_path, out_path = argv[1:]
    loader = LOADERS_BY_NAME.get(name)
    if loader is None:
        print(f"no loader named {name!r}; known: {sorted(LOADERS_BY_NAME)}", file=sys.stderr)
        return 2
    raw = json.loads(Path(ctx_path).read_text())
    ctx = Ctx(project=Path(raw["project"]), home=Path(raw["home"]), expect=raw["expect"],
              plugin_name=raw["plugin_name"])
    try:
        module = importlib.import_module(f"native_loaders.loaders.{loader.module}")
        result = getattr(module, mode)(ctx)
    except Exception as exc:  # noqa: BLE001 - a crash is reported, never swallowed
        result = {"loader": name, "crashed": describe(exc),
                  "traceback": traceback.format_exc()[-4000:]}
    Path(out_path).write_text(json.dumps(result, default=str, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
