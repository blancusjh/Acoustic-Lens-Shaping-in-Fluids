"""Execute clean source notebooks into artifacts/ and export readable HTML."""

import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

root = Path(__file__).resolve().parents[1]
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
destination = root / "artifacts" / "notebooks"
destination.mkdir(parents=True, exist_ok=True)
for source in sorted((root / "notebooks").glob("*.ipynb")):
    notebook = nbformat.read(source, as_version=4)
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
    target = destination / (source.stem + ".executed.ipynb")
    nbformat.write(notebook, target)
    html, _ = HTMLExporter().from_notebook_node(notebook)
    (destination / (source.stem + ".html")).write_text(
        "\n".join(line.rstrip() for line in html.splitlines()) + "\n"
    )
    print(f"Saved {target.name} and HTML", flush=True)
