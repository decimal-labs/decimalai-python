"""Names a server hands us that become paths on this machine.

The registry supplies skill names. A malicious or compromised entry must not be able
to write into ~/.bashrc, .git/hooks, cron.d or an SSH config on the machine that
installs it, so a name is checked, never quietly sanitized: a silently rewritten
install is one nobody can audit.
"""

from __future__ import annotations

import os


class UnsafeSkillPathError(ValueError):
    """A server-supplied skill name that would escape the folder it is written to
    (path traversal, or an absolute path). A ``ValueError``, so existing
    ``except ValueError`` handlers catch it."""

    def __init__(self, value: str, kind: str = "path"):
        self.value = value
        super().__init__(
            f"Refusing to write outside the skill directory: unsafe {kind} {value!r} "
            "(path traversal / absolute path). The skill source may be malicious."
        )


def safe_skill_dirname(name: str) -> str:
    """``name``, when it is a single safe path component; else :class:`UnsafeSkillPathError`.

    Skill slugs are ``[A-Za-z0-9._-]``; a path separator, a ``.``/``..`` segment, a null
    byte or an absolute path is a traversal attempt.
    """
    if not name or name in (".", "..") or os.path.isabs(name) or "\x00" in name:
        raise UnsafeSkillPathError(name, "skill name")
    if "/" in name or "\\" in name or ".." in name.replace("\\", "/").split("/"):
        raise UnsafeSkillPathError(name, "skill name")
    return name
