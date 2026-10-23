"""Compare immutable file identity across task and prepared-proof schemas.

Task bindings contain ``path`` and ``sha256``.  A prepared artifact may also
record ``bytes``: this is informational provenance, not a different file
identity.  This module does no filesystem access, including for label paths;
the caller retains responsibility for its existing permitted-file hash checks.
"""
from __future__ import annotations

import re


def _metadata(value, *, artifact):
    expected = {"path", "sha256"}
    if not isinstance(value, dict) or not (set(value) == expected or
            (artifact and set(value) == expected | {"bytes"})):
        raise ValueError("invalid artifact identity schema")
    path = value["path"]
    # Frozen assets live at absolute POSIX server paths.  Validate without
    # resolving, normalizing, statting or opening a potentially sealed label.
    if (not isinstance(path, str) or not path.startswith("/") or
            path.startswith("//") or "\\" in path or
            any(ord(character) < 32 or ord(character) == 127 for character in path) or
            any(part in ("", ".", "..") for part in path.split("/")[1:])):
        raise ValueError("invalid absolute artifact path")
    digest = value["sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("invalid artifact sha256")
    if "bytes" in value and (type(value["bytes"]) is not int or value["bytes"] < 0):
        raise ValueError("invalid informational artifact byte count")
    return path, digest


def same_artifact(binding, artifact):
    """Return exact literal path/hash identity; reject malformed metadata.

    The optional prepared-artifact byte count is validated but never used as a
    substitute for the SHA-256, and no normalization can conceal a path change.
    """
    return _metadata(binding, artifact=False) == _metadata(artifact, artifact=True)


def require_same_artifact(binding, artifact, message):
    """Require schema-valid, identical path/hash metadata without opening it."""
    if not same_artifact(binding, artifact):
        raise ValueError(message)
