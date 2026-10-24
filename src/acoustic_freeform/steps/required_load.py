"""Full-domain prescribed mean traction; no emitters or formation simulation."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..apparatus.config import DualConfig
from ..core.provenance import capture_execution
from ..mechanics.required_load import quadrature, traction_components
from ..mechanics.surface import DualSurface
from ..optics.raytrace import trace_pair
from ..optics.stigmatic import stigmatic_pair


def run(config_path, output):
    data = json.loads(Path(config_path).read_text())
    cfg = DualConfig(**data["apparatus"])
    targets = stigmatic_pair(
        cfg, data["indices"], data["stigmatic_z_m"], data["vertex_displacement_m"]
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(data, indent=2) + "\n")
    (output / "resolved-config.json").write_text(json.dumps(cfg.as_dict(), indent=2) + "\n")
    capture_execution(output)

    def components(r):
        return np.array(
            [
                traction_components(
                    t,
                    r,
                    cfg.surface_tension_n_m[j],
                    cfg.density_kg_m3[j] - cfg.density_kg_m3[j + 1],
                    cfg.gravity_m_s2,
                )
                for j, t in enumerate(targets)
            ]
        )

    # Gauge and target-volume audit use a mesh independent of all solve grids.
    edges = np.unique(np.r_[np.linspace(0, cfg.radius_m, 257), cfg.clear_radius_m])
    rq, aq = quadrature(edges, 24)
    gauge = components(rq).sum(axis=1) @ aq / aq.sum()
    volume = np.array([t.evaluate(rq) @ aq for t in targets])
    r = np.linspace(0, cfg.radius_m, 16001)
    loads = components(r)
    total = loads.sum(axis=1) - gauge[:, None]
    heights = np.array([t.evaluate(r) for t in targets])
    np.savez_compressed(
        output / "required-load.npz",
        radius_m=r,
        target_displacement_m=heights,
        capillary_pa=loads[:, 0],
        gravity_pa=loads[:, 1],
        mean_traction_pa=total,
        pressure_gauge_pa=gauge,
    )
    rays = np.linspace(0, cfg.clear_radius_m, data["ray_count"])
    launch = cfg.levels_m[1] + targets[0].evaluate(rays)
    studies = []
    for elements in data["surface_elements"]:
        space = DualSurface(replace(cfg, surface_elements=elements))
        mesh = np.unique(space.raw.t) * cfg.radius_m
        for order in data["load_quadrature_orders"]:
            nodes, weights = quadrature(mesh, order)
            prescribed = components(nodes).sum(axis=1) - gauge[:, None]
            force = ((prescribed * weights) @ space.basis(nodes)).ravel()
            # Solve from FLAT, not the projected target; load is independent of q.
            q = space.prescribed_equilibrium(force)
            gradient, stiffness, _ = space.mechanics(q)
            h = np.array([space.evaluate(face, r) for face in q])
            pupil = r <= cfg.clear_radius_m
            trace = trace_pair(
                space,
                q,
                data["indices"],
                data["stigmatic_z_m"][0],
                data["stigmatic_z_m"][2],
                launch_radius_m=rays,
                launch_height_m=launch,
            )
            row = {
                "surface_elements": elements,
                "load_quadrature_order": order,
                "sampled_full_surface_error_m": np.max(abs(h - heights), axis=1).tolist(),
                "sampled_pupil_error_m": np.max(
                    abs(h[:, pupil] - heights[:, pupil]), axis=1
                ).tolist(),
                "force_residual_norm_n": float(np.linalg.norm(gradient - force)),
                "minimum_discrete_mechanical_eigenvalue_n_m": float(
                    np.linalg.eigvalsh(stiffness)[0]
                ),
                "max_geometric_spot_radius_m": trace["max_radius_m"],
                "all_rays_transmitted": trace["all_rays_transmitted"],
            }
            studies.append(row)
            np.savez_compressed(
                output / f"equilibrium-e{elements}-q{order}.npz",
                coefficients_m=q,
                prescribed_force_n=force,
                radius_m=r,
                displacement_m=h,
                spots_m=trace["spots_m"],
                transmitted=trace["transmitted"],
            )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    for j, ax in enumerate(axes):
        ax.plot(r * 1e3, loads[j, 0], label="Capillary (reference gauge)")
        ax.plot(r * 1e3, loads[j, 1], label="Gravity (reference gauge)")
        ax.plot(r * 1e3, total[j], label="Required total (disk mean removed)")
        ax.axvline(cfg.clear_radius_m * 1e3, color="gray", linestyle="--")
        ax.set(
            ylabel="Upward mean traction [Pa]",
            title=["Front / lower interface", "Back / upper interface"][j],
        )
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Radius [mm]; dashed line separates pupil and annulus")
    fig.savefig(output / "required-load.png", dpi=160)
    plt.close(fig)
    report = {
        "scope": "Ideal fixed spatial mean traction, static axisymmetric forward verification only",
        "load_gauge": "Zero horizontal-disk mean on each complete interface",
        "target_volume_displacement_m3": volume.tolist(),
        "target_pinned_displacement_m": heights[:, -1].tolist(),
        "sampled_required_traction_range_pa": [[float(v.min()), float(v.max())] for v in total],
        "studies": studies,
        "formation_simulated": False,
        "viscous_or_acoustic_stability_verified": False,
        "acoustic_realization_verified": False,
        "experimental_validation": False,
        "continuous_maximum_certified": False,
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "report.md").write_text(
        "# Required loads before emitter synthesis\n\n"
        "![Both complete interface loads](required-load.png)\n\n"
        "Loads come from continuous target curvature and gravity; equilibria are solved from flat. "
        "They are not target projections or time trajectories. Positive mechanical eigenvalues "
        "are not an acoustic/streaming stability certificate.\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n"
    )
    print(json.dumps(report, indent=2))
