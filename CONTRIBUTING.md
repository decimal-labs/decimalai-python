# Contributing to decimalai

Thanks for your interest. This is the Python SDK — it instruments agent runs and syncs
skills, so the bar is compatibility: code here runs inside other people's applications.

## Before you open a PR

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q --ignore=tests/test_langchain_compat.py
ruff check decimalai/ --select I,E,W,F --ignore E501,E402,F821,F841
git grep -nE 'decimal[-_]ai'   # must find nothing — the package is 'decimalai', no separator
```

CI runs the same commands, plus the LangChain compatibility matrix in
`tests/test_langchain_compat.py`.

## What a PR is expected to contain

- **A test that fails without the change.** Regression tests are named for the behaviour
  they protect (`test_sync_clobber_newer_wins.py`), not for a ticket.
- **No new required dependency** without discussion. Optional integrations belong behind an
  extra in `pyproject.toml`, imported lazily, so `pip install decimalai` stays small.
- **Python 3.10+.** `requires-python` is `>=3.10`; the test matrix is 3.10–3.12.
- **A CHANGELOG entry** for anything a user would notice.

## Compatibility

The package name is `decimalai` (install name == import name). Public API is anything
importable from `decimalai` without a leading underscore — removing or renaming it is a
breaking change and needs a major version. Adding optional keyword arguments is not.

## Reporting bugs

Open an issue with the smallest reproduction you can manage, the SDK version
(`python -c "import decimalai; print(decimalai.__version__)"`), and the framework you are
instrumenting. For anything security-related see [SECURITY.md](SECURITY.md) instead —
please do not open a public issue.
