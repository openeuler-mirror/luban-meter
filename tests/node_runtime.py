"""Locate Node.js for headless report JavaScript checks."""

from __future__ import annotations

import shutil
from pathlib import Path


def node_binary() -> str | None:
    """Return a Node.js executable, including nvm installs off PATH."""
    found = shutil.which("node")
    if found:
        return found
    candidates = sorted(
        (Path.home() / ".nvm" / "versions" / "node").glob("*/bin/node")
    )
    return str(candidates[-1]) if candidates else None
