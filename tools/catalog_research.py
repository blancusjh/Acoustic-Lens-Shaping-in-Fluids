"""Index existing experiment records without moving or reinterpreting evidence."""

import argparse
import hashlib
import json
from pathlib import Path

from acoustic_freeform.core.provenance import capture_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.out.mkdir(parents=True, exist_ok=False)
    capture_execution(args.out)
    records = []
    candidates = sorted(
        set((root / "artifacts").rglob("config.json"))
        | set((root / "artifacts").rglob("configuration.json"))
    )
    for path in candidates:
        if args.out.resolve() in path.parents:
            continue
        folder = path.parent
        rel = folder.relative_to(root)
        if rel.parts[1].startswith("catalog-"):
            continue  # Inventories are navigation artifacts, not experiments.
        reports = [
            p.name
            for p in folder.iterdir()
            if p.name
            in (
                "report.md",
                "report.json",
                "results.json",
                "validation.json",
                "clarifications.md",
                "spatial-convergence.json",
            )
        ]
        family = (
            "two-face precision program: consult report"
            if "dual-precision" in str(rel)
            else "two-face stationary"
            if "dual-cartesian" in str(rel)
            else "imported single-interface audit"
            if "imported-discussion" in str(rel)
            else "earlier model: consult configuration"
        )
        records.append(
            {
                "directory": str(rel),
                "family": family,
                "configuration": path.name,
                "configuration_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "available_records": sorted(reports),
                "provenance_present": (folder / "provenance").is_dir(),
                "scientific_pass_inferred": False,
            }
        )
    config = {
        "operation": "Read-only inventory of configured experiment directories",
        "scope": "artifacts/",
        "record_count": len(records),
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.out / "catalog.json").write_text(json.dumps(records, indent=2) + "\n")
    (args.out / "validation.json").write_text(
        json.dumps(
            {
                "all_configuration_paths_exist": all(
                    (root / r["directory"] / r["configuration"]).is_file() for r in records
                ),
                "scientific_results_recomputed": False,
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    lines = [
        "# Research evidence catalog",
        "",
        "Inventory only: file presence is not scientific completion or validation.",
        "",
        "| Experiment | Model family | Available report records |",
        "|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| `{r['directory']}` | {r['family']} | {', '.join(r['available_records']) or 'None: inspect partial evidence'} |"
        )
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Indexed {len(records)} configured experiment directories in {args.out}")


if __name__ == "__main__":
    main()
