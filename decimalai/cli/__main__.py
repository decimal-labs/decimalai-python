"""Entry point for `python -m decimalai.cli`.

Lets users invoke the CLI via the conventional package form
(`python -m decimalai.cli ...`) in addition to the longer
`python -m decimalai.cli.main ...` and the installed `decimal`
console script. All three routes reach the same `cli` entry point.
"""

from .main import cli

if __name__ == "__main__":
    cli()
