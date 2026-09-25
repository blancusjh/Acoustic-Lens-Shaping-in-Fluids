"""Forward two-interface creeping-flow formation under fixed ideal mean loads."""

import argparse
import json
from pathlib import Path

import numpy as np

from ..apparatus.config import DualConfig
from ..core.provenance import capture_execution
from ..mechanics.required_load import quadrature, traction_components
from ..mechanics.surface import DualSurface
from ..mechanics.viscous import assemble_mobility
from ..optics.raytrace import trace_pair
from ..optics.stigmatic import stigmatic_pair


def implicit_step(space, q, force, mobility, dt):
    """Backward mechanical gradient, old-geometry Stokes mobility; first order."""
    old = q.ravel()
    new = old.copy()
    for _ in range(20):
        grad, stiffness, _ = space.mechanics(new.reshape(q.shape))
        residual = new - old - dt * mobility @ (force - grad)
        change = np.linalg.solve(np.eye(len(old)) + dt * mobility @ stiffness, residual)
        new -= change
        if np.max(abs(change)) < 1e-14:
            return new.reshape(q.shape)
    raise RuntimeError("Implicit viscous step did not converge")


def run(config_path, output):
    data = json.loads(Path(config_path).read_text())
    cfg = DualConfig(**data["apparatus"])
    mu = data["viscosities_pa_s"]
    space = DualSurface(cfg)
    targets = stigmatic_pair(
        cfg, data["indices"], data["stigmatic_z_m"], data["vertex_displacement_m"]
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(data, indent=2) + "\n")
    (output / "resolved-config.json").write_text(json.dumps(cfg.as_dict(), indent=2) + "\n")
    capture_execution(output)
    nodes, weights = quadrature(np.unique(space.raw.t) * cfg.radius_m, 24)
    loads = np.array(
        [
            sum(
                traction_components(
                    t,
                    nodes,
                    cfg.surface_tension_n_m[j],
                    cfg.density_kg_m3[j] - cfg.density_kg_m3[j + 1],
                    cfg.gravity_m_s2,
                )
            )
            for j, t in enumerate(targets)
        ]
    )
    force = ((loads * weights) @ space.basis(nodes)).ravel()
    np.savez_compressed(
        output / "prescribed-load.npz",
        radius_m=nodes,
        upward_mean_traction_pa=loads,
        generalized_force_n=force,
    )
    q = np.zeros((2, space.count))
    dt, duration = data["step_s"], data["duration_s"]
    steps = round(duration / dt)
    if not np.isclose(steps * dt, duration):
        raise ValueError("Duration must be a multiple of fixed step")
    dense = np.linspace(0, cfg.radius_m, 4001)
    target_h = np.array([t.evaluate(dense) for t in targets])
    pupil = dense <= cfg.clear_radius_m
    rays = np.linspace(0, cfg.clear_radius_m, data["ray_count"])
    launch = cfg.levels_m[1] + targets[0].evaluate(rays)
    rows, states, times, energy = [], [q.copy()], [0.0], [0.0]
    diagnostics = []

    def observe(time, state):
        h = np.array([space.evaluate(face, dense) for face in state])
        trace = trace_pair(
            space,
            state,
            data["indices"],
            data["stigmatic_z_m"][0],
            data["stigmatic_z_m"][2],
            launch_radius_m=rays,
            launch_height_m=launch,
        )
        return {
            "time_s": time,
            "sampled_pupil_error_m": np.max(abs(h[:, pupil] - target_h[:, pupil]), axis=1).tolist(),
            "sampled_full_surface_error_m": np.max(abs(h - target_h), axis=1).tolist(),
            "max_geometric_spot_radius_m": trace["max_radius_m"],
            "all_rays_transmitted": trace["all_rays_transmitted"],
            "transmitted_rays": int(np.sum(trace["transmitted"])),
        }

    rows.append(observe(0, q))
    for step in range(steps):
        fluid = assemble_mobility(
            space, q, mu, data["fluid_radial_cells"], data["fluid_cells_per_layer"]
        )
        new = implicit_step(space, q, force, fluid.mobility, dt)
        grad, _, mechanical = space.mechanics(new)
        loaded_energy = mechanical - float(force @ new.ravel())
        if loaded_energy > energy[-1] + 1e-16:
            raise RuntimeError("Loaded mechanical energy increased")
        u = fluid.velocity_basis.interpolate(fluid.velocity_operator @ (force - grad))
        speed = float(np.max(np.sqrt(np.sum(u**2, axis=0))))
        diagnostics.append(
            {
                **fluid.diagnostics,
                "time_s": (step + 1) * dt,
                "sampled_velocity_peak_m_s": speed,
                "reynolds_radius_estimate": float(
                    max(np.array(cfg.density_kg_m3) / mu) * speed * cfg.radius_m
                ),
            }
        )
        q = new
        times.append((step + 1) * dt)
        states.append(q.copy())
        energy.append(loaded_energy)
        if (step + 1) % data["observation_stride"] == 0 or step + 1 == steps:
            rows.append(observe(times[-1], q))
            print(json.dumps(rows[-1]), flush=True)
        np.savez_compressed(
            output / "trajectory.npz", time_s=times, coefficients_m=states, loaded_energy_j=energy
        )
        (output / "progress.json").write_text(
            json.dumps({"step": step + 1, "steps": steps, "time_s": times[-1]}) + "\n"
        )
    # Higher ray count at the endpoint without changing the fixed illumination.
    final_r = np.linspace(0, cfg.clear_radius_m, data["final_ray_count"])
    final_trace = trace_pair(
        space,
        q,
        data["indices"],
        data["stigmatic_z_m"][0],
        data["stigmatic_z_m"][2],
        launch_radius_m=final_r,
        launch_height_m=cfg.levels_m[1] + targets[0].evaluate(final_r),
    )
    equilibrium = space.prescribed_equilibrium(force)
    distance = np.array([space.evaluate(v, dense) for v in q - equilibrium])
    report = {
        "completed": True,
        "model": "Moving-faceted-domain, three-fluid axisymmetric Stokes; projected graph kinematics",
        "initial_condition": "Flat interfaces, fixed full-target load switched on at t=0",
        "maximum_reynolds_radius_estimate": max(d["reynolds_radius_estimate"] for d in diagnostics),
        "viscous_diffusion_radius_time_s": float(
            max(np.array(cfg.density_kg_m3) / mu) * cfg.radius_m**2
        ),
        "sampled_final_distance_to_discrete_equilibrium_m": np.max(abs(distance), axis=1).tolist(),
        "final_max_geometric_spot_radius_m": final_trace["max_radius_m"],
        "final_all_rays_transmitted": final_trace["all_rays_transmitted"],
        "final_ray_count": data["final_ray_count"],
        "observations": rows,
        "inertia_streaming_heat_nonaxisymmetric_implemented": False,
        "acoustic_realization_verified": False,
        "physical_ten_nm_certified": False,
        "material_properties": "Hypothetical explicit scenario, not a characterized fluid triplet",
        "convergence": "Requires comparison against independently configured runs",
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "fluid-diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    (output / "report.md").write_text(
        "# Prescribed-load viscous formation\n\n"
        "Forward computed trajectory, not shape interpolation. Creeping-flow startup has no inertia. "
        "No acoustic or physical accuracy certificate.\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "observations"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.out)
