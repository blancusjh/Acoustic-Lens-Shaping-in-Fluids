"""Snapshot executable sources before numerical work, independently of later edits."""

import hashlib
import json
import platform
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _git(*args):
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def capture_execution(destination):
    out = Path(destination) / "provenance"
    out.mkdir(parents=True, exist_ok=True)
    # The whole acoustic_freeform package, not only this core/ subpackage.
    root = Path(__file__).resolve().parents[1]
    archive = out / "execution-sources.zip"
    hashes = {}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for p in sorted(root.rglob("*.py")):
            name = str(p.relative_to(root.parent))
            content = p.read_bytes()
            hashes[name] = hashlib.sha256(content).hexdigest()
            zipped.writestr(name, content)
    packages = {name: version(name) for name in ("numpy", "scipy", "scikit-fem", "pyvista")}
    for name in ("cvxpy", "clarabel", "scs"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    report = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": hashes,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_uncommitted_changes": _git("status", "--porcelain"),
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "scope": "Sources captured at entry to this numerical execution, before iterations. Rendering can be performed later using separately versioned viewer code.",
    }
    (out / "execution.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
