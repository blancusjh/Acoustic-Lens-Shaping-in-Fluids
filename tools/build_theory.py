"""Compile the private theoretical manuscript; no simulation is executed."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "docs" / "theory"
    output = root / "artifacts" / "theory"
    build = output / "build"
    build.mkdir(parents=True, exist_ok=True)
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise SystemExit("latexmk is required (with pdfLaTeX and biber).")
    command = [
        latexmk,
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        "-jobname=acoustic-fluid-shaping-theory",
        f"-outdir={build}",
        "main.tex",
    ]
    completed = subprocess.run(command, cwd=source, text=True, capture_output=True, check=False)
    log = completed.stdout + completed.stderr
    (build / "latexmk-output.log").write_text(log)
    if completed.returncode:
        print("\n".join(log.splitlines()[-70:]))
        raise SystemExit(completed.returncode)
    tex_log = (build / "acoustic-fluid-shaping-theory.log").read_text(errors="replace")
    problems = [
        line
        for line in tex_log.splitlines()
        if any(
            token in line
            for token in (
                "undefined references",
                "undefined citations",
                "LaTeX Warning: Citation",
                "multiply defined",
                "Overfull \\hbox",
                "Overfull \\vbox",
            )
        )
    ]
    if problems:
        raise SystemExit("Manuscript build needs review:\n" + "\n".join(problems))
    pdf = output / "acoustic-fluid-shaping-theory.pdf"
    shutil.copy2(build / pdf.name, pdf)
    files = sorted(source.rglob("*.tex")) + [source / "references.bib", Path(__file__).resolve()]
    manifest = {
        "built_utc": datetime.now(UTC).isoformat(),
        "artifact_kind": "theoretical_manuscript",
        "physics_simulations_executed": False,
        "command": command,
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
        "pdf": str(pdf.relative_to(root)),
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "pdf_bytes": pdf.stat().st_size,
    }
    (output / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(pdf)
    print(f"{pdf.stat().st_size:,} bytes; references resolved; no overfull boxes.")


if __name__ == "__main__":
    main()
