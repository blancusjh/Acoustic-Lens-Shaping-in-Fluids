"""Forward emitter-driven Stokes diagnostic; no target shape in the dynamics."""

import hashlib
import json

import numpy as np

from ..acoustics.helmholtz import DualAcoustics
from ..apparatus.config import DualConfig
from ..apparatus.sources import export_sources
from ..core.paths import data_path
from ..core.provenance import capture_execution
from ..forward.formation import envelope, implicit_step, linearized_implicit_step
from ..mechanics.surface import CartesianPatch, DualSurface
from ..mechanics.viscous import assemble_mobility
from ..optics.raytrace import trace_pair


def run(configuration, output):
    data = json.loads(configuration.read_text())
    source = data_path(data["source_directory"])
    synthesis = json.loads(data_path(data["synthesis_config"]).read_text())
    # "frozen" lags the whole acoustic load (historical S05); "linearized" makes
    # its analytic shape tangent implicit in each step.
    acoustic_load = data.get("acoustic_load", "frozen")
    if acoustic_load not in ("frozen", "linearized"):
        raise ValueError("acoustic_load must be frozen or linearized")
    material = json.loads(data_path(synthesis["material_reference"]).read_text())
    cfg = DualConfig(**json.loads((source / "config.json").read_text()))
    space = DualSurface(cfg, allow_volume_offsets=True)
    drive = np.load(source / "state.npz")["source_velocity_m_s"]
    mu = np.array(material["viscosities_pa_s"])
    output.mkdir(parents=True, exist_ok=False)
    capture_execution(output)
    data["state_sha256"] = hashlib.sha256((source / "state.npz").read_bytes()).hexdigest()
    data["apparatus"] = cfg.as_dict()
    data["material"] = material
    (output / "config.json").write_text(json.dumps(data, indent=2) + "\n")
    export_sources(output, cfg, drive)
    targets = [CartesianPatch(cfg, j, **f) for j, f in enumerate(synthesis["case"]["faces"])]
    r = np.linspace(0, cfg.clear_radius_m, data["ray_count"])
    launch = cfg.levels_m[1] + targets[0].evaluate(r)
    target_h = np.array([t.evaluate(r) for t in targets])
    # Start from the acoustics-off equilibrium at the declared volumes (flat
    # for the historical equal-volume convention).
    q = space.unforced_equilibrium()
    f0, k0, _ = space.mechanics(q)
    minimum_stiffness = float(np.linalg.eigvalsh(k0)[0])
    if np.max(abs(np.linalg.solve(k0, f0))) > 1e-12 or minimum_stiffness <= 0:
        raise ValueError("The acoustics-off start is not a stable mechanical equilibrium")
    wave = DualAcoustics(cfg, space, linear_solver="static_condensed")
    times, states, rows, diagnostics, energies = [0.0], [q.copy()], [], [], [0.0]

    def observe():
        trace = trace_pair(
            space,
            q,
            material["indices"],
            material["stigmatic_z_m"][0],
            material["stigmatic_z_m"][2],
            launch_radius_m=r,
            launch_height_m=launch,
        )
        error = np.max(
            abs(np.array([space.height(j, v, r) for j, v in enumerate(q)]) - target_h), axis=1
        )
        return {
            "time_s": times[-1],
            "sampled_pupil_error_m": error.tolist(),
            "max_spot_radius_m": trace["max_radius_m"],
            "transmitted_count": int(trace["transmitted"].sum()),
            "all_rays_transmitted": trace["all_rays_transmitted"],
        }

    def save():
        np.savez_compressed(
            output / "trajectory.npz",
            time_s=times,
            coefficients_m=states,
            mechanical_energy_j=energies,
            source_velocity_m_s=drive,
        )
        (output / "observations.json").write_text(json.dumps(rows, indent=2) + "\n")
        (output / "step-diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")

    rows.append(observe())
    save()
    reason = None
    while times[-1] < data["duration_s"] - 1e-12:
        if len(diagnostics) >= data["maximum_steps"]:
            reason = "Accepted-step budget reached; partial computed trajectory preserved"
            break
        t = times[-1]
        wants_tangent = acoustic_load == "linearized" or data.get("report_acoustic_tangent", False)
        # A tangent from an earlier accepted geometry may be reused for a few
        # steps; the load itself is always freshly solved on the current faces.
        if wants_tangent and len(diagnostics) % data.get("tangent_refresh_steps", 1) == 0:
            response, tangent_unit = wave.shape_force_jacobian(q, drive)
            tangent_time = t
        else:
            response = wave.solve(q, drive)
            if not wants_tangent:
                tangent_unit, tangent_time = None, None
        force_unit = response.force(np.ones(1, complex))
        fluid = assemble_mobility(
            space, q, mu, data["fluid_radial_cells"], data["fluid_cells_per_layer"]
        )
        dt = min(data["step_s"], data["duration_s"] - t)
        while True:
            amplitude = envelope(t + dt / 2, data["ramp_s"])
            force = force_unit * amplitude**2
            tangent = None if tangent_unit is None else tangent_unit * amplitude**2
            try:
                if acoustic_load == "linearized":
                    new = linearized_implicit_step(space, q, force, tangent, fluid.mobility, dt)
                else:
                    new = implicit_step(space, q, force, fluid.mobility, dt)
                increment = max(space.polynomial_maximum(v)["max_abs_m"] for v in new - q)
                heights = np.array([space.height(j, v, space.r) for j, v in enumerate(new)])
                bounds = np.vstack(
                    [
                        np.full(len(space.r), cfg.levels_m[0]),
                        np.array(cfg.levels_m[1:3])[:, None] + heights,
                        np.full(len(space.r), cfg.levels_m[-1]),
                    ]
                )
                accepted = (
                    increment <= data["maximum_step_displacement_m"]
                    and np.min(np.diff(bounds, axis=0)) > 0
                )
            except (RuntimeError, ValueError):
                accepted = False
            if accepted:
                break
            dt /= 2
            if dt < data["minimum_step_s"]:
                reason = "Minimum time step reached at displacement/gap/Newton guard"
                break
        if reason:
            break
        grad, _, energy = space.mechanics(new)
        net = force - grad
        velocity = fluid.velocity_basis.interpolate(fluid.velocity_operator @ net)
        speed = float(np.max(np.sqrt(np.sum(velocity**2, axis=0))))
        step_change = (new - q).ravel()
        if acoustic_load == "linearized":
            # Trapezoidal work of the linearized load: an estimate, not an identity.
            work = float((force + 0.5 * tangent @ step_change) @ step_change)
        else:
            work = float(force @ step_change)
        balance = work - (energy - energies[-1])
        if acoustic_load == "frozen" and balance < -1e-14:
            raise RuntimeError("Frozen-load step violates dissipation inequality")
        tangent_rate = None
        if tangent is not None:
            _, stiffness, _ = space.mechanics(q)
            # Linearized growth rate of the old state, dq/dt = M (A - K) dq.
            tangent_rate = float(
                np.max(np.linalg.eigvals(fluid.mobility @ (tangent - stiffness)).real)
            )
        diagnostics.append(
            {
                **fluid.diagnostics,
                "time_s": t + dt,
                "step_s": dt,
                "midpoint_command_envelope": amplitude,
                "maximum_increment_m": increment,
                "sampled_fluid_speed_m_s": speed,
                "reynolds_radius_upper_estimate": float(
                    max(np.array(cfg.density_kg_m3) / mu) * speed * cfg.radius_m
                ),
                "work_minus_energy_change_j": balance,
                "acoustic_load": acoustic_load,
                "linearized_max_growth_rate_s": tangent_rate,
                "acoustic_tangent_geometry_time_s": tangent_time,
                "full_command_wave": response.diagnostics(np.ones(1, complex)),
            }
        )
        q = new
        times.append(t + dt)
        states.append(q.copy())
        energies.append(energy)
        rows.append(observe())
        save()
        print(json.dumps(rows[-1]), flush=True)
    result = {
        "completed": reason is None,
        "stop_reason": reason,
        "last_time_s": times[-1],
        "accepted_steps": len(diagnostics),
        "unforced_minimum_discrete_stiffness_n_m": minimum_stiffness,
        "unforced_force_n": float(np.max(abs(f0))),
        "maximum_reynolds_radius_upper_estimate": max(
            (d["reynolds_radius_upper_estimate"] for d in diagnostics), default=0
        ),
        "viscous_diffusion_radius_time_s": float(
            max(np.array(cfg.density_kg_m3) / mu) * cfg.radius_m**2
        ),
        "final_observation": rows[-1],
        "target_used_in_rhs": False,
        "acoustic_load": acoustic_load,
        "scope": "Emitter-driven moving-domain Stokes approximation. No inertia, convection, streaming or thermal physics. Unrefined diagnostic, not a settling or 10 nm claim.",
    }
    (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "report.md").write_text(
        "# Emitter-driven viscous trajectory\n\n```json\n"
        + json.dumps(result, indent=2)
        + "\n```\n"
    )
    print(json.dumps(result, indent=2), flush=True)
