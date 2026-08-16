"""The child process: run ONE driver's phases, write the capture, exit.

    python tests/conformance/_child.py <driver-name> <out.json>

Invoked by ``isolation.run_driver_in_child``; not a test module (pytest collects
``test_*.py`` only) and not meant to be run by hand, though it is perfectly
runnable by hand when a single framework is being debugged.

The probe is started HERE, in the process that runs the framework, because this
is the process the SDK's HTTP calls come out of. A probe in the parent would
record nothing and every driver would grade as "emits nothing".

Nothing is graded here. The capture goes back to the parent and ``contract.py``
stays the single place assertions live.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # …/tests/conformance
_TESTS_DIR = _HERE.parent                        # …/tests
_REPO_ROOT = _TESTS_DIR.parent                   # the checkout root

# Import the suite the same way pytest does (``conformance.*``, with ``tests/``
# on the path), and keep the checkout root importable so ``decimalai`` resolves
# from a scratch venv that has not installed it.
for _path in (str(_REPO_ROOT), str(_TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from conformance.drivers import all_drivers  # noqa: E402
from conformance.harness import observe  # noqa: E402
from conformance.isolation import (  # noqa: E402
    CONFORMANCE_SKILLS,
    dump_payload,
    encode_observation,
)
from conformance.probe import Probe  # noqa: E402


def main(argv: list) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} <driver-name> <out.json>", file=sys.stderr)
        return 2
    name, out_path = argv[1], argv[2]

    drivers = {d.name: d for d in all_drivers()}
    driver = drivers.get(name)
    if driver is None:
        print(
            f"no driver named {name!r}; known: {sorted(drivers)}", file=sys.stderr
        )
        return 2
    if not driver.available:
        print(
            f"{name}: missing import(s): {', '.join(driver.missing_requirements)}",
            file=sys.stderr,
        )
        return 3

    probe = Probe().start()
    try:
        obs = observe(driver, probe, skills=CONFORMANCE_SKILLS)
        text = dump_payload(encode_observation(obs))
    finally:
        probe.stop()
    Path(out_path).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
