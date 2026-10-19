"""Coverage for the install-path traversal guard.

Threat model: the backend supplies the skill name. A malicious or compromised
registry entry must not be able to write into ~/.bashrc, .git/hooks, cron.d, or an
SSH config on the installing machine.
"""

import pytest

from decimalai._safe_paths import UnsafeSkillPathError, safe_skill_dirname


@pytest.mark.parametrize(
    "name",
    [
        "code-review",
        "playwright-cli",
        "skill.with.dots",
        "Mixed_Case-Name",
        "a",
    ],
)
def test_accepts_a_plain_single_component(name):
    assert safe_skill_dirname(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "../evil",
        "../../etc/passwd",
        "nested/child",
        "acme/example-tool",  # a namespaced registry name is NOT a dirname
        "back\\slash",
        "..\\windows",
        "/absolute",
        "/etc/passwd",
        "has\x00null",
    ],
)
def test_rejects_traversal_and_separators(name):
    with pytest.raises(UnsafeSkillPathError):
        safe_skill_dirname(name)


def test_error_names_the_offending_value_and_kind():
    with pytest.raises(UnsafeSkillPathError) as ei:
        safe_skill_dirname("../evil")
    msg = str(ei.value)
    assert "../evil" in msg
    assert "skill name" in msg
    # Subclasses ValueError so existing `except ValueError` handlers still catch it.
    assert isinstance(ei.value, ValueError)


def test_refuses_rather_than_sanitizing():
    """The guard must never quietly rewrite a hostile name into a safe-looking one —
    a silently sanitized install is an install the user cannot audit."""
    with pytest.raises(UnsafeSkillPathError):
        safe_skill_dirname("../../.ssh/authorized_keys")
