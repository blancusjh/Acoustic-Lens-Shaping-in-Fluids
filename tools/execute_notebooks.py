"""Execute clean source notebooks into artifacts/ and export readable HTML."""

import argparse
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "notebooks", nargs="*", type=Path, help="Selected source notebooks; default: all"
)
parser.add_argument("--timeout", type=int, default=300, help="Execution timeout per cell, seconds")
args = parser.parse_args()
if args.timeout <= 0:
    parser.error("--timeout must be positive")
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
destination = root / "artifacts" / "notebooks"
destination.mkdir(parents=True, exist_ok=True)
for source in args.notebooks or sorted((root / "notebooks").glob("*.ipynb")):
    notebook = nbformat.read(source, as_version=4)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python (acoustic-freeform-lab)",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python"}
    print(f"Executing {source.name}", flush=True)
    NotebookClient(
        notebook, timeout=args.timeout, kernel_name="python3", resources={"metadata": {"path": str(root)}}
    ).execute()
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert not any(output.output_type == "error" for output in cell.get("outputs", []))
    target = destination / (source.stem + ".executed.ipynb")
    nbformat.write(notebook, target)
    html, _ = HTMLExporter().from_notebook_node(notebook)
    source_directory = source.resolve().parent

    def relocate_link(match, source_parent=source_directory):
        url = urlsplit(match[2])
        if url.scheme or url.netloc or not url.path or url.path.startswith("/"):
            return match[0]
        path = os.path.relpath(source_parent / url.path, destination)
        return match[1] + urlunsplit(("", "", path, url.query, url.fragment)) + match[3]

    html = re.sub(r'(href=")([^"]*)(")', relocate_link, html)
    (destination / (source.stem + ".html")).write_text(
        "\n".join(line.rstrip() for line in html.splitlines()) + "\n"
    )
    print(f"Saved {target.name} and HTML", flush=True)
