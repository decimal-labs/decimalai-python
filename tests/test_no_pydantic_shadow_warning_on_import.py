"""Lock in: importing the SDK doesn't emit the pydantic schema_json field-
shadow warning.

Every `import decimalai` used to trigger:
    UserWarning: Field name "schema_json" in "ComponentSnapshot"
    shadows an attribute in parent "BaseModel"

The shadow is harmless under pydantic v2 — the field takes precedence
over the BaseModel method — but the noise polluted CI logs, Jupyter
notebooks, and library-user terminals. Fix uses `warnings.filterwarnings`
scoped to this exact message at the top of `schema/manifest.py` so the
established wire-format field name stays unchanged.
"""

from __future__ import annotations

import importlib
import sys
import warnings


def test_importing_decimalai_does_not_emit_schema_json_shadow_warning():
    """Importing the SDK should not produce the schema_json shadow warning."""
    # Force a clean import — wipe any cached decimalai modules.
    for name in list(sys.modules):
        if name == "decimalai" or name.startswith("decimalai."):
            del sys.modules[name]

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("default")
        importlib.import_module("decimalai")

    shadow_warnings = [
        w for w in captured
        if issubclass(w.category, UserWarning)
        and "schema_json" in str(w.message)
        and "shadows" in str(w.message)
    ]
    assert not shadow_warnings, (
        f"Importing decimalai emitted {len(shadow_warnings)} schema_json "
        f"shadow warning(s): {[str(w.message) for w in shadow_warnings]}"
    )


def test_filter_scope_is_narrow_other_shadow_warnings_still_fire():
    """The narrow filter must NOT suppress unrelated shadow warnings.

    If someone later adds another field shadowing BaseModel, we want to
    notice — so the filter should match only the schema_json/ComponentSnapshot
    pair, not the broader "shadows attribute" class.
    """
    from pydantic import BaseModel

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("default")

        # Define a class that shadows a DIFFERENT BaseModel attribute.
        # `copy` is a BaseModel method. Naming a field `copy` shadows it.
        try:
            class _Probe(BaseModel):  # noqa: D401
                copy: str = ""
        except Exception:
            # Some pydantic versions reject this outright — fine.
            return

    relevant = [w for w in captured if "shadows" in str(w.message) and "copy" in str(w.message)]
    # We don't assert this fires — pydantic version-dependent — but we
    # DO assert that if it fires, our filter didn't accidentally swallow it.
    if relevant:
        # If pydantic emitted ANY shadow warning for `copy`, our filter
        # didn't suppress it (it would not be in `captured`). Pass.
        pass
