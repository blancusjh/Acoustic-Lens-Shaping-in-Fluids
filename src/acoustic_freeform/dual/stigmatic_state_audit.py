"""Joint optical gates for saved coupled equilibria, never inverse iterates."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..provenance import capture_execution
from .config import DualConfig
from .optics import trace_pair
from .stigmatic_audit import summary
from .surface import CartesianPatch, DualSurface


def run(campaign, output, ray_counts=(1001, 4001), allow_completed_grids=False):
    campaign, output = Path(campaign), Path(output)
    inputs = json.loads((campaign / "config.json").read_text())
    records = json.loads((campaign / "results.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    capture_execution(output)
    provenance = {}
    rows = []
    for case, record in zip(inputs["cases"], records, strict=True):
        complete = record.get("status", "complete") == "complete"
        if not complete and not allow_completed_grids:
            raise ValueError("Finish the coupled campaign before auditing it")
        cfg = DualConfig(**inputs["apparatus"])
        faces = case["faces"]
        targets = [CartesianPatch(cfg, j, **face) for j, face in enumerate(faces)]
        vertices = np.array(cfg.levels_m[1:3]) + [face["vertex_displacement_m"] for face in faces]
        z0 = vertices[0] + faces[0]["optical"]["z_o_m"]
        z2 = vertices[1] + faces[1]["optical"]["z_i_m"]
        z1 = [
            vertices[0] + faces[0]["optical"]["z_i_m"],
            vertices[1] + faces[1]["optical"]["z_o_m"],
        ]
        if abs(z1[0] - z1[1]) > 1e-14:
            raise ValueError("Independent targets do not share a laboratory conjugate")
        indices = [
            faces[0]["optical"]["n_o"],
            faces[0]["optical"]["n_i"],
            faces[1]["optical"]["n_i"],
        ]
        if indices[1] != faces[1]["optical"]["n_o"]:
            raise ValueError("Middle optical medium is inconsistent")
        for grid in record["grids"]:
            path = campaign / case["name"] / f"{grid['grid']}-state.npz"
            provenance[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            q = np.load(path)["coefficients_m"]
            space = DualSurface(DualConfig(**grid["numerics"]))
            for count in ray_counts:
                r = np.linspace(0, cfg.clear_radius_m, count)
                trace = trace_pair(
                    space,
                    q,
                    indices,
                    z0,
                    z2,
                    launch_radius_m=r,
                    launch_height_m=cfg.levels_m[1] + targets[0].evaluate(r),
                )
                rows.append(
                    {
                        "case": case["name"],
                        "grid": grid["grid"],
                        "entire_campaign_complete": complete,
                        "rays": count,
                        **summary(trace),
                        "surface_max_error_m": [x["max_error_m"] for x in grid["faces"]],
                        "sampled_joint_geometry_pass": bool(
                            trace["sampled_1um_pass"] and grid["mean_10nm_sampled_test"]
                        ),
                        "physical_accuracy_certified": False,
                    }
                )
                np.savez_compressed(
                    output / f"{case['name']}-{grid['grid']}-{count}-rays.npz",
                    spots_m=trace["spots_m"],
                    transmitted=trace["transmitted"],
                    intersections_m=trace["intersections_m"],
                )
    (output / "config.json").write_text(
        json.dumps(
            {
                "campaign": str(campaign),
                "ray_counts": ray_counts,
                "allow_completed_grids": allow_completed_grids,
                "launch": "Exact-target-pupil fixed cone including rim",
                "campaign_inputs": inputs,
            },
            indent=2,
        )
        + "\n"
    )
    (output / "input-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "validation.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output / "report.md").write_text(
        "# Coupled-state optical audit\n\nGeometric rays, not diffraction. "
        "Stationary states, not formation or stability.\n\n```json\n"
        + json.dumps(rows, indent=2)
        + "\n```\n"
    )
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--allow-completed-grids",
        action="store_true",
        help="Audit saved solved grids while the remaining refinements are running.",
    )
    args = parser.parse_args()
    run(args.campaign, args.out, allow_completed_grids=args.allow_completed_grids)
