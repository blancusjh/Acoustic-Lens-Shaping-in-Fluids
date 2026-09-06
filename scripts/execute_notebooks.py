"""Rebuild paired notebooks, execute in the local uv environment, export HTML."""

import os
import sys
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

root = Path(__file__).resolve().parents[1]
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
for source in sorted((root / "notebooks").glob("*.py")):
    notebook = jupytext.read(source)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python (acoustic-freeform-lab)",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python"}
    print(f"Executing {source.name}", flush=True)
    NotebookClient(
        notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(root)}}
    ).execute()
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert not any(output.output_type == "error" for output in cell.get("outputs", []))
    target = source.with_suffix(".ipynb")
    nbformat.write(notebook, target)
    html, _ = HTMLExporter().from_notebook_node(notebook)
    source.with_suffix(".html").write_text(
        "\n".join(line.rstrip() for line in html.splitlines()) + "\n"
    )
    print(f"Saved {target.name} and HTML", flush=True)
