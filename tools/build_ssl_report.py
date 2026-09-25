"""Build the standalone SSL vision report without touching the main manuscript."""

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "docs/theory/ssl-generator"
    output = root / "artifacts/theory/ssl-generator"
    output.mkdir(parents=True, exist_ok=True)
    name = "ssl-generator-vision"
    final = output / f"{name}.pdf"
    if final.exists():
        raise FileExistsError(
            "Archive the previous report and source milestone before replacing the PDF"
        )
    command = [
        "latexmk",
        "-xelatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-jobname={name}",
        f"-outdir={output / 'build'}",
        "main.tex",
    ]
    built = subprocess.run(command, cwd=source, text=True, capture_output=True, check=False)
    (output / "build-output.log").write_text(built.stdout + built.stderr)
    if built.returncode:
        print((built.stdout + built.stderr)[-7000:])
        raise SystemExit(built.returncode)
    log = (output / "build" / f"{name}.log").read_text(errors="replace")
    tokens = (
        "Overfull \\hbox",
        "Overfull \\vbox",
        "Missing character:",
        "undefined references",
        "undefined citations",
        "LaTeX Warning: Citation",
        "multiply defined",
    )
    problems = [line for line in log.splitlines() if any(t in line for t in tokens)]
    if problems:
        raise SystemExit("\n".join(problems))
    shutil.copy2(output / "build" / f"{name}.pdf", final)
    files = [*sorted(source.glob("*.tex")), source.parent / "references.bib", Path(__file__)]
    manifest = {
        "built_utc": datetime.now(UTC).isoformat(),
        "command": command,
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        "pdf_sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
        "physics_simulations_executed": False,
        "scope": "Standalone SSL design vision and theory sketch; main theory PDF unchanged.",
    }
    (output / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(final)


if __name__ == "__main__":
    main()
