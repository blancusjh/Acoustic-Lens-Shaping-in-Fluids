"""Energy, absorption-driven flow and fast-interface diagnostics at a holding state.

Uses the declared bulk attenuation and optional viscous acoustic wall losses.
Mean wall-layer streaming, thermal feedback, photoelasticity and cavitation are absent.
"""

import json
from pathlib import Path

import numpy as np
from skfem import Basis, LinearForm, asm
from skfem.helpers import dot

from acoustic_freeform.single_interface.acoustics import CavityAcoustics, wall_gradient_matrix
from acoustic_freeform.single_interface.config import LensConfig
from acoustic_freeform.single_interface.hydrodynamics import assemble_fluid
from acoustic_freeform.single_interface.surface import SurfaceSpace


def acoustic_budget(source, destination):
    source, out = Path(source), Path(destination)
    if (out / "report.json").exists():
        raise FileExistsError(f"Completed diagnostic exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    cfg = LensConfig(**json.loads((source / "configuration.json").read_text()))
    data = np.load(source / "stationary.npz")
    c, drive = data["coefficients"], data["drive_m_s"]
    space = SurfaceSpace(cfg)
    field = CavityAcoustics(cfg, space).solve_basis(c)
    omega = 2 * np.pi * cfg.frequency_hz
    phi = field.basis.interpolate(field.solutions @ drive)
    p = cfg.density_kg_m3 * omega * cfg.radius_m * phi
    velocity = -1j * phi.grad
    intensity = 0.5 * np.real(p[None, ...] * velocity.conj())
    heat = cfg.attenuation_np_m * abs(p) ** 2 / (cfg.density_kg_m3 * cfg.sound_speed_m_s)
    body = 2 * cfg.attenuation_np_m / cfg.sound_speed_m_s * intensity
    weights = field.basis.dx * field.basis.global_coordinates()[0]
    absorbed_power = 2 * np.pi * cfg.radius_m**3 * np.sum(weights * heat)
    nodal_p = cfg.density_kg_m3 * omega * cfg.radius_m * (field.solutions @ drive)
    source_power = (
        -np.pi * cfg.radius_m**2 * np.real(np.vdot(-1j * (field.wall_loads @ drive), nodal_p))
    )
    delta = np.sqrt(2 * cfg.viscosity_pa_s / (cfg.density_kg_m3 * omega))
    potential = field.solutions @ drive
    wall_power = (
        float(
            np.pi
            * cfg.viscosity_pa_s
            * cfg.radius_m**2
            / delta
            * np.vdot(potential, wall_gradient_matrix(field.basis) @ potential).real
        )
        if cfg.viscous_wall_acoustics
        else 0.0
    )
    fluid = assemble_fluid(space, c)
    integration = Basis(
        field.basis.mesh, fluid.velocity_basis.elem, quadrature=field.basis.quadrature
    )

    @LinearForm
    def absorption_load(v, w):
        return w.x[0] * dot(w.body, v)

    rhs = asm(absorption_load, integration, body=body * cfg.radius_m**2 / cfg.surface_tension_n_m)
    u_body = fluid.solve(rhs)
    response = fluid.solve(fluid.coupling.T)
    mobility = fluid.coupling @ response
    equivalent_force = np.linalg.solve(mobility, fluid.coupling @ u_body)
    # Enforce zero mean interface speed to diagnose the residual circulation.
    circulation = u_body - response @ equivalent_force
    speed = (
        cfg.surface_tension_n_m
        / cfg.viscosity_pa_s
        * np.linalg.norm(circulation.reshape(-1, 2), axis=1)
    )
    _, capillary = space.derivatives(c)
    q = space.tangent
    correction = q @ np.linalg.solve(q.T @ capillary @ q, equivalent_force)
    pupil = cfg.clear_radius_m / cfg.radius_m * np.sqrt((np.arange(801) + 0.5) / 801)
    vn = field.normal_velocity_basis @ drive
    clear = field.radial_samples <= cfg.clear_radius_m / cfg.radius_m
    fast = vn / omega
    report = {
        "scope": "Energy and constrained bulk-circulation diagnostic at the recorded holding geometry. This diagnostic does not independently solve the nonlinear mean-flow equilibrium. Bulk heating and Eckart forcing use the declared constant attenuation; viscous acoustic wall loss is included when configured. Capillary compliance is not the fully coupled optical error.",
        "configuration": cfg.as_dict(),
        "source_power_w": float(source_power),
        "volume_absorbed_power_w": float(absorbed_power),
        "viscous_wall_dissipation_w": wall_power,
        "power_balance_relative_error": float(
            abs(source_power - absorbed_power - wall_power) / max(abs(source_power), 1e-30)
        ),
        "maximum_heat_generation_w_m3": float(heat.max()),
        "volume_mean_heat_generation_w_m3": float(np.sum(weights * heat) / weights.sum()),
        "maximum_intensity_w_m2": float(np.linalg.norm(intensity, axis=0).max()),
        "maximum_bulk_force_n_m3": float(np.linalg.norm(body, axis=0).max()),
        "constrained_absorption_streaming_max_speed_m_s": float(speed.max()),
        "constrained_streaming_reynolds_number": float(
            cfg.density_kg_m3 * speed.max() * cfg.radius_m / cfg.viscosity_pa_s
        ),
        "streaming_force_capillary_compliance_pupil_rms_m": float(
            cfg.radius_m * np.sqrt(np.mean(space.evaluate(correction, pupil) ** 2))
        ),
        "fast_normal_displacement_clear_peak_m": float(abs(fast[clear]).max()),
        "fast_normal_displacement_clear_space_time_rms_m": float(
            np.sqrt(
                np.sum(field.quadrature_weights[clear] * abs(fast[clear]) ** 2)
                / (2 * np.sum(field.quadrature_weights[clear]))
            )
        ),
        "viscous_acoustic_boundary_layer_m": float(
            np.sqrt(2 * cfg.viscosity_pa_s / (cfg.density_kg_m3 * omega))
        ),
        "acoustic_pressure_to_bulk_modulus_max": float(
            abs(p).max() / (cfg.density_kg_m3 * cfg.sound_speed_m_s**2)
        ),
        "reference": "Bach and Bruus (2018), Eq. (55), doi:10.1121/1.5049579. Here Gamma*omega/c^2 = 2*alpha/c. Heat follows directly by integrating -div(I) for the declared complex-k Helmholtz model.",
        "missing_for_thermal_prediction": [
            "liquid heat capacity",
            "liquid thermal conductivity",
            "apparatus thermal boundary conditions",
            "temperature derivatives of acoustic and optical properties",
        ],
    }
    np.savez_compressed(
        out / "absorption-flow.npz",
        coefficients=c,
        drive_m_s=drive,
        velocity_coefficients_m_s=circulation * cfg.surface_tension_n_m / cfg.viscosity_pa_s,
        equivalent_surface_force=equivalent_force,
        capillary_compliance_correction=correction,
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print({k: v for k, v in report.items() if k != "configuration"}, flush=True)
    return report
