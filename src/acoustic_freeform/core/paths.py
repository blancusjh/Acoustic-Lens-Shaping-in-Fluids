"""Locate research data in the repository's study artifact tree.

Runs executed before 25 September 2026 recorded inputs as ``artifacts/<campaign>/...``.
Those directories now live under ``artifacts/studies/<study>/``; ``configs/relocations.json`` maps
each old prefix to its new location so historical configurations stay executable.
"""

import json
from functools import cache
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
RELOCATIONS = REPOSITORY / "configs" / "relocations.json"


@cache
def _relocations():
    if not RELOCATIONS.exists():
        return ()
    table = json.loads(RELOCATIONS.read_text())
    return tuple(sorted(table.items(), key=lambda item: -len(item[0])))


def data_path(value):
    """Resolve a configured input path, following recorded relocations."""
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    direct = REPOSITORY / path
    if direct.exists():
        return direct
    text = path.as_posix()
    for old, new in _relocations():
        if text == old or text.startswith(old + "/"):
            moved = Path(new + text[len(old) :])
            return moved if moved.is_absolute() else REPOSITORY / moved
    return direct
