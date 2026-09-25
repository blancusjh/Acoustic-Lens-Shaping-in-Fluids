"""Separate geometry, source integration and wave resolution at fixed state/drive."""

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from skfem import FacetBasis

from acoustic_freeform.single_interface.acoustics import CavityAcoustics
from acoustic_freeform.single_interface.config import LensConfig
from acoustic_freeform.single_interface.surface import SurfaceSpace


def geometry_diagnostics(space, coefficients, field):
    top = FacetBasis(
        field.basis.mesh,
        field.basis.elem,
        facets=field.basis.mesh.boundaries["surface"],
        intorder=14,
    )
    r, z = top.global_coordinates()
    h = space.evaluate(coefficients, r)
    slope = space.evaluate(coefficients, r, 1)
    expected = np.stack([-slope, np.ones_like(slope)]) / np.sqrt(1 + slope * slope)
    clear = r <= space.config.clear_radius_m / space.config.radius_m
    return {
        "maximum_height_error_m": float(space.config.radius_m * np.max(abs(z - h))),
        "maximum_clear_height_error_m": float(space.config.radius_m * np.max(abs(z - h)[clear])),
        "maximum_normal_error": float(np.max(np.linalg.norm(top.normals - expected, axis=0))),
        "graph_area_projection_error": float(abs(np.sum(top.dx * top.normals[1] * r) - 0.5)),
    }


def fixed_state_study(result_directory, destination):
    source, out = Path(result_directory), Path(destination)
    if (out / "report.json").exists():
        raise FileExistsError(f"Completed study exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    cfg = LensConfig(**json.loads((source / "configuration.json").read_text()))
    saved = np.load(source / "stationary.npz")
    modes = max(cfg.surface_modes, 48)
    coefficients = np.pad(saved["coefficients"], (0, modes - cfg.surface_modes))
    drive = saved["drive_m_s"].copy()
    common = replace(cfg, surface_modes=modes)
    definitions = [
        ("quadratic_uniform", {}),
        ("exact_uniform", {"geometry_mapping": "exact_graph"}),
        (
            "exact_rim",
            {"geometry_mapping": "exact_graph", "mesh_radial_distribution": "rim_clustered"},
        ),
        (
            "exact_rim_aligned",
            {
                "geometry_mapping": "exact_graph",
                "mesh_radial_distribution": "rim_clustered",
                "align_array_mesh": True,
            },
        ),
        (
            "exact_rim_aligned_q14",
            {
                "geometry_mapping": "exact_graph",
                "mesh_radial_distribution": "rim_clustered",
                "align_array_mesh": True,
                "acoustic_quadrature_order": 14,
            },
        ),
        (
            "exact_rim_aligned_fine",
            {
                "geometry_mapping": "exact_graph",
                "mesh_radial_distribution": "rim_clustered",
                "align_array_mesh": True,
                "acoustic_quadrature_order": 14,
                "mesh_radial": 80,
                "mesh_vertical": 128,
            },
        ),
    ]
    records, forces = [], []
    for name, changes in definitions:
        local = replace(common, **changes)
        space = SurfaceSpace(local)
        print(f"fixed-state study: {name}", flush=True)
        start = time.perf_counter()
        field = CavityAcoustics(local, space).solve_basis(coefficients)
        force = field.force(drive)
        pressure = (
            local.density_kg_m3
            * 2
            * np.pi
            * local.frequency_hz
            * local.radius_m
            * (field.solutions @ drive)
        )
        p = field.basis.interpolate(pressure)
        weights = field.basis.dx * field.basis.global_coordinates()[0]
        row = {
            "case": name,
            "configuration": local.as_dict(),
            "acoustic_dofs": int(field.basis.N),
            "triangles": int(field.basis.mesh.nelements),
            "seconds": time.perf_counter() - start,
            "helmholtz_relative_residual": field.residual,
            "cavity_pressure_volume_rms_pa": float(
                np.sqrt(np.sum(weights * abs(p) ** 2) / weights.sum())
            ),
            "cavity_pressure_max_at_dofs_pa": float(abs(pressure).max()),
            **geometry_diagnostics(space, coefficients, field),
        }
        records.append(row)
        forces.append(force)
        np.savez_compressed(
            out / f"{name}.npz",
            coefficients=coefficients,
            drive_m_s=drive,
            force=force,
            radial_samples=field.radial_samples,
            normal_velocity=field.normal_velocity_basis @ drive,
            quadrature_weights=field.quadrature_weights,
        )
        (out / "progress.json").write_text(json.dumps(records, indent=2) + "\n")
        print({k: v for k, v in row.items() if k != "configuration"}, flush=True)
        del field, p, weights, pressure
    space = SurfaceSpace(common)
    q = space.tangent
    _, stiffness = space.derivatives(coefficients)
    inverse = q @ np.linalg.solve(q.T @ stiffness @ q, q.T)
    pupil = cfg.clear_radius_m / cfg.radius_m * np.sqrt((np.arange(801) + 0.5) / 801)
    for i, row in enumerate(records):
        delta = inverse @ (forces[i] - forces[-1])
        row["force_gap_to_finest_capillary_compliance_clear_rms_m"] = float(
            cfg.radius_m * np.sqrt(np.mean(space.evaluate(delta, pupil) ** 2))
        )
        row["force_gap_to_finest_capillary_compliance_full_rms_m"] = float(
            cfg.radius_m * np.sqrt(delta @ space.mass @ delta / space.w.sum())
        )
        if i:
            change = inverse @ (forces[i] - forces[i - 1])
            row["force_change_from_previous_compliance_clear_rms_m"] = float(
                cfg.radius_m * np.sqrt(np.mean(space.evaluate(change, pupil) ** 2))
            )
    report = {
        "kind": "fixed_surface_fixed_drive_resolution_study",
        "scope": "The same modal surface and physical drive in every case. Compliance quantities diagnose force differences; they are not re-solved coupled equilibria or optical validation.",
        "source": str(source.resolve()),
        "source_state_sha256": hashlib.sha256((source / "stationary.npz").read_bytes()).hexdigest(),
        "source_sha256": {
            str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for directory in [Path(__file__).parents[1] / "lens", Path(__file__).parent]
            for p in sorted(directory.glob("*.py"))
        },
        "cases": records,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
