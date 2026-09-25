"""Reproducible optical prerequisite audit; no acoustically formed-state claim."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..apparatus.config import DualConfig
from ..core.provenance import capture_execution
from ..mechanics.surface import DualSurface
from ..optics.raytrace import trace_graph_pair, trace_pair
from ..optics.stigmatic import pair_prescription, stigmatic_pair


def summary(trace):
    return {
        key: value
        for key, value in trace.items()
        if key not in ("spots_m", "intersections_m", "transmitted")
    }


def run(config_path, output):
    inputs = json.loads(Path(config_path).read_text())
    cfg = DualConfig(**inputs["apparatus"])
    indices, z = inputs["indices"], inputs["stigmatic_z_m"]
    vertices = inputs["vertex_displacement_m"]
    targets = stigmatic_pair(cfg, indices, z, vertices)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(inputs, indent=2) + "\n")
    (output / "resolved-config.json").write_text(json.dumps(cfg.as_dict(), indent=2) + "\n")
    capture_execution(output)
    faces = pair_prescription(cfg, indices, z, vertices)
    (output / "target-case.json").write_text(
        json.dumps({"name": "shared-conjugate-pair", "faces": faces}, indent=2) + "\n"
    )
    exact, projections = [], []
    for count in inputs["ray_counts"]:
        r = np.linspace(0, cfg.clear_radius_m, count)
        launch_z = cfg.levels_m[1] + targets[0].evaluate(r)
        trace = trace_graph_pair(
            cfg,
            lambda j, rr, derivative=0: targets[j].evaluate(rr, derivative),
            indices,
            z[0],
            z[2],
            launch_radius_m=r,
            launch_height_m=launch_z,
        )
        a, b = trace["intersections_m"]
        path = (
            indices[0] * np.linalg.norm(a - [0, z[0]], axis=1)
            + indices[1] * np.linalg.norm(b - a, axis=1)
            + indices[2] * np.linalg.norm(b - [0, z[2]], axis=1)
        )
        exact.append(
            {"rays": count, **summary(trace), "total_optical_path_range_m": float(np.ptp(path))}
        )
    dense_r = np.linspace(0, cfg.radius_m, 16001)
    target_h = np.array([t.evaluate(dense_r) for t in targets])
    for elements in inputs["surface_elements"]:
        space = DualSurface(replace(cfg, surface_elements=elements))
        q = space.project(targets)
        h = np.array([space.evaluate(face, dense_r) for face in q])
        errors = np.max(
            abs(h[:, dense_r <= cfg.clear_radius_m] - target_h[:, dense_r <= cfg.clear_radius_m]),
            axis=1,
        )
        trace = trace_pair(
            space, q, indices, z[0], z[2], launch_radius_m=r, launch_height_m=launch_z
        )
        projections.append(
            {
                "surface_elements": elements,
                **summary(trace),
                "sampled_surface_max_error_m": errors.tolist(),
                "projected_target_not_forward_state": True,
            }
        )
        np.savez_compressed(
            output / f"projection-{elements}.npz",
            coefficients_m=q,
            radius_m=dense_r,
            heights_m=h,
            target_heights_m=target_h,
            spots_m=trace["spots_m"],
            transmitted=trace["transmitted"],
        )
    bounds = np.vstack(
        [
            np.full(len(dense_r), cfg.levels_m[0]),
            np.array(cfg.levels_m[1:3])[:, None] + target_h,
            np.full(len(dense_r), cfg.levels_m[-1]),
        ]
    )
    report = {
        "exact_target": exact,
        "spline_projection": projections,
        "minimum_sampled_layer_depth_m": float(np.min(np.diff(bounds, axis=0))),
        "acoustic_inverse_solved": False,
        "formation_solved": False,
        "experimental_validation": False,
        "ten_nm_physical_accuracy_certified": False,
        "diffraction_computed": False,
        "launch_convention": "Fixed cone through exact front target pupil including rim; not the old flat-reference cone",
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "report.md").write_text(
        "# Joint stigmatic target audit\n\nOptical targets and their numerical representation only. "
        "No source command or physical trajectory is computed.\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.out)
