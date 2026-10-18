# Native mode

When a framework loads the skills `decimalai skills add` installed with **its own** skill
loader, does DecimalAI record the load as an activation of the exact version the model
received? This suite answers that for the eight frameworks that have such a loader, on
every `pytest tests/` run, offline, with no keys.

`tests/native_loaders` proves the loaders read our folders. This suite proves DecimalAI
**sees** them do it: the activation reaches the wire, on the trace DecimalAI sends.

## The native column

`test_native_column.py` grades seven items per framework, from one scenario each
(`_scenario_<framework>.py:column`), with every assertion in `column_contract.py`:

| item | asserts |
|---|---|
| NC1 | run 1's trace lists the target among the skills the model was offered |
| NC2 | run 1's trace activates the target with the installed version's exact identity (wire name, skill id, version id, body hash) |
| NC3 | the model loaded the target twice in run 1, and the trace counts it once |
| NC4 | a tool call in run 1 read the target's resource through the framework, and its result holds the resource's bytes |
| NC5 | a tool call in run 1 ran the target's script through the framework, and its result holds the script's output |
| NC6 | after `decimalai skills update`, run 2's activation names the version whose body the model actually received |
| NC7 | the router stays out: with the router holding skills of its own, neither run made a routing request or carries a routing id |

The scenario drives the real framework, its own loader (set up with DecimalAI's pointer
helper), and a scripted model, with `decimalai.init` pointed at the conformance probe
(`tests/conformance/probe.py`). Expected identities come from the lock (`oracle.build`),
never from the trace or the probe. `test_column_contract.py` proves every item fails on the
defect it names. An item a framework cannot satisfy goes in `column_contract.DECLARED_NA`
with its reason (budget: `NA_BUDGET`); a declared item that starts passing fails.

What NC6 measured on 2026-10-18: ADK, Microsoft Agent Framework and Strands keep serving
v1 to the same agent after the update (and the activation says v1); Claude Agent SDK, the
OpenAI sandbox, CrewAI, Deep Agents and the Pydantic AI harness re-read it (v2).

## Rules

- **One child process per scenario** (`_runner.run_scenario`, `_child.py`): event buses,
  plugin registries, OTel providers and every adapter's `instrument()` patch are
  process-global, and this suite rides the default unit run, so nothing it sets may leak
  into the tests after it. Tracing is ON in these children (the native-loader children
  turn it off).
- **Nothing skips silently.** A framework graded in the other environment
  (`LOADER_ENVS` in `tests/conformance/environments.py`) is a declared "elsewhere" skip; a
  framework not installed is a declared "unavailable" skip, which `ledger.skip_declared`
  refuses under a named `DECIMAL_TEST_ENV`. `conftest._framework_present` applies both to
  every framework's own test module.
- **Collected by default**, not behind the `conformance` marker, so CI's floors jobs grade
  it at the lowest declared versions: `floors-native` (openai3's six frameworks) and
  `floors-openai2` (CrewAI, Strands).

## Run it

```bash
DECIMAL_TEST_ENV=openai3 .venv/bin/python -m pytest tests/native -q -rs
DECIMAL_TEST_ENV=openai2 .venv-openai2/bin/python -m pytest tests/native -q -rs
```

About 90 s (openai3) and 30 s (openai2). One framework: `-k <framework>`. Pass test files
from this directory together: pytest stops applying this directory's fixtures to a second
group of its files listed after files from elsewhere.

## Adding a framework

1. Its hook (`decimalai/<framework>.py`) hands what the model received to
   `decimalai._native.record`, with the run's `NativeLoads` when the adapter owns one.
   Never re-read the SKILL.md: after an update the file is the wrong witness.
2. Its pointer helper, public in that module.
3. `_scenario_<framework>.py` with `column(ctx)`, built on `column.Session`.
4. Its entry in `test_native_column.FRAMEWORKS` and `conftest._FRAMEWORK_OF_MODULE`, and
   in `tests/native_loaders/loaders` (the column must cover exactly the loaders graded
   there).
