"""Index preserved reports and render the curated evidence ledger.

The report index is an inventory, not a numerical acceptance test. Scientific
verdicts are reviewed separately in studies/ledger.json.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "artifacts" / "studies"
LEDGER = ROOT / "studies" / "ledger.json"
INDEX = ROOT / "studies" / "run-index.json"
STATUS = ROOT / "STATUS.md"
STATES = {"passed", "failed", "model_limited", "not_assessed", "not_applicable"}


def relative(path):
    return path.relative_to(ROOT).as_posix()


def report_index():
    records = []
    for study in sorted(STUDIES.iterdir()):
        if not study.is_dir():
            continue
        for report in sorted(study.rglob("*")):
            if report.name not in {"report.json", "results.json", "report.md"}:
                continue
            run = report.parent
            config = next(
                (run / name for name in ("config.json", "configuration.json") if (run / name).is_file()),
                None,
            )
            validation = run / "validation.json"
            provenance = run / "provenance"
            report_path = relative(report)
            records.append(
                {
                    "id": "R-" + hashlib.sha256(report_path.encode()).hexdigest()[:12],
                    "study": study.name,
                    "report": report_path,
                    "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                    "config": relative(config) if config else None,
                    "validation": relative(validation) if validation.is_file() else None,
                    "provenance_present": provenance.is_dir() or (run / "provenance.json").is_file(),
                }
            )
    return {"schema_version": 1, "meaning": "File inventory only; no acceptance inference", "records": records}


def render_status(ledger):
    fields = ledger["evidence_fields"]
    if len(fields) != 9 or len(set(fields)) != 9:
        raise ValueError("The evidence ledger must declare nine distinct fields")
    rows = []
    for record in ledger["records"]:
        path = ROOT / record["evidence_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        evidence = record["evidence"]
        if set(evidence) != set(fields) or set(evidence.values()) - STATES:
            raise ValueError(f"Invalid evidence states for {record['id']}")
        label = f"[{record['id']}](studies/{record['study']}/study.md)"
        rows.append(
            "| " + " | ".join(
                [
                    label,
                    evidence["source_candidate_found"],
                    evidence["stationary_root_converged"],
                    evidence["stationary_optical_gates_met"],
                    evidence["fixed_command_convergence_verified"],
                    evidence["formation_verified"],
                ]
            ) + " |"
        )
    parts = [
        "# Research status",
        "",
        "Generated from [the reviewed evidence ledger](studies/ledger.json) by `python3 tools/build_research_index.py`.",
        "Each status applies only to its declared numerical model and evidence file. No experimental validation is recorded.",
        "",
        "| Record | Source | Stationary root | Optical gates | Fixed-command convergence | Formation |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
        "`model_limited` means the reported check exists only within the stated approximation. `not_assessed` is not a failure or a pass. The ideal-load study uses a prescribed traction, so acoustic source and fixed-command fields are `not_applicable`.",
        "",
        "## Reviewed verdicts",
        "",
    ]
    for record in ledger["records"]:
        parts.extend(
            [
                f"- **{record['id']} — {record['title']}:** {record['verdict']} {record['scope']}",
            ]
        )
    parts.extend(
        [
            "",
            "The next theory gates are acoustic reachability under admissible continuous sources and stability of the driven three-dimensional state. Any numerical refinement must hold physical commands fixed before interpreting a new inverse solve. Wall-layer mean streaming and thermal feedback remain open.",
            "",
        ]
    )
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if generated files are stale")
    args = parser.parse_args()
    index_text = json.dumps(report_index(), indent=2) + "\n"
    status_text = render_status(json.loads(LEDGER.read_text()))
    for path, expected in ((INDEX, index_text), (STATUS, status_text)):
        if args.check:
            if not path.exists() or path.read_text() != expected:
                raise SystemExit(f"Stale generated file: {relative(path)}")
        else:
            path.write_text(expected)
    print(f"Indexed {len(json.loads(index_text)['records'])} report files")


if __name__ == "__main__":
    main()
