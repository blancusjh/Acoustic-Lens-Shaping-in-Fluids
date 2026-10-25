"""Static and creeping-flow linear stability of a saved held-command root.

About the saved geometry q and command g, small axisymmetric shape changes obey
dq/dt = M (A - K) dq in the declared Stokes approximation: K is the discrete
capillary-gravity Hessian, A = dF_ac/dq the analytic acoustic shape stiffness at
the same command and M the three-fluid Stokes mobility. A positive real part
of an eigenvalue of M (A - K) is linear instability of this axisymmetric,
inertia-free model. Non-axisymmetric modes, inertia, streaming base flow and
thermal feedback are not represented and are reported as not assessed.
"""

import json

import numpy as np
from scipy.special import jn_zeros

from ..acoustics.helmholtz import DualAcoustics
from ..core.paths import data_path
from ..core.provenance import capture_execution
from ..mechanics.surface import DualSurface
from ..mechanics.viscous import assemble_mobility
from .force_convergence import load_state


def growth_rates(mobility, acoustic, stiffness, scale=1.0):
    return np.linalg.eigvals(mobility @ (scale * acoustic - stiffness))


def critical_scale(mobility, acoustic, stiffness, upper=4.0, samples=161):
    """Smallest s with a non-negative growth rate of M (s A - K); None below upper.

    A margin of the acoustic shape stiffness at this state, not a statement
    about other commands: only s = 1 linearizes about an actual equilibrium.
    """
    scales = np.linspace(0, upper, samples)
    peak = [np.max(growth_rates(mobility, acoustic, stiffness, s).real) for s in scales]
    crossing = next((i for i, value in enumerate(peak) if value >= 0), None)
    if crossing is None:
        return None
    if crossing == 0:
        return 0.0
    low, high = scales[crossing - 1], scales[crossing]
    for _ in range(40):
        middle = (low + high) / 2
        if np.max(growth_rates(mobility, acoustic, stiffness, middle).real) >= 0:
            high = middle
        else:
            low = middle
    return float(high)


def pinned_capillary_gravity_stiffness(cfg):
    """Analytic flat-state per-area stiffness (Pa/m) of the softest pinned modes."""
    rows = []
    for j in range(2):
        sigma = cfg.surface_tension_n_m[j]
        gravity = (cfg.density_kg_m3[j] - cfg.density_kg_m3[j + 1]) * cfg.gravity_m_s2
        # Volume-preserving axisymmetric mode: J2(kR) = 0. First m = 1 mode: J1(kR) = 0.
        axisymmetric = jn_zeros(2, 1)[0] / cfg.radius_m
        tilt = jn_zeros(1, 1)[0] / cfg.radius_m
        rows.append(
            {
                "face": j,
                "axisymmetric_volume_preserving_pa_m": float(sigma * axisymmetric**2 + gravity),
                "m1_pa_m": float(sigma * tilt**2 + gravity),
            }
        )
    return rows


def run(configuration, output):
    inputs = json.loads(configuration.read_text())
    cfg, q, drive, synthesis, hashes = load_state(inputs["state_directory"])
    material = json.loads(data_path(synthesis["material_reference"]).read_text())
    viscosity = np.array(inputs.get("viscosities_pa_s", material["viscosities_pa_s"]))
    space = DualSurface(cfg, allow_volume_offsets=True)
    output.mkdir(parents=True, exist_ok=False)
    capture_execution(output)
    inputs["input_sha256"] = hashes
    (output / "config.json").write_text(json.dumps(inputs, indent=2) + "\n")
    wave = DualAcoustics(cfg, space, linear_solver="static_condensed")
    response, acoustic = wave.shape_force_jacobian(q, drive)
    force = response.force(np.ones(1, complex))
    mechanical, stiffness, _ = space.mechanics(q)
    unit = np.ones(1, complex)

    # Independent check of the analytic tangent by central differences of fresh solves.
    rng = np.random.default_rng(inputs.get("seed", 0))
    step = inputs.get("difference_step_m", 1e-9)
    tangent_checks = []
    for _ in range(inputs.get("difference_directions", 2)):
        direction = rng.normal(size=q.size)
        direction /= np.linalg.norm(direction)
        plus = wave.solve(q + step * direction.reshape(q.shape), drive).force(unit)
        minus = wave.solve(q - step * direction.reshape(q.shape), drive).force(unit)
        difference = (plus - minus) / (2 * step)
        analytic = acoustic @ direction
        tangent_checks.append(
            float(np.linalg.norm(difference - analytic) / np.linalg.norm(analytic))
        )

    coupled = stiffness - acoustic
    static = np.linalg.eigvals(np.linalg.solve(stiffness, coupled))
    symmetric = np.linalg.eigvalsh((coupled + coupled.T) / 2)
    mobility_rows, primary = [], None
    for cells in inputs["fluid_meshes"]:
        fluid = assemble_mobility(
            space, q, viscosity, cells["radial_cells"], cells["cells_per_layer"]
        )
        rates = growth_rates(fluid.mobility, acoustic, stiffness)
        order = np.argsort(-rates.real)
        mobility_rows.append(
            {
                "fluid_mesh": cells,
                "max_growth_rate_s": float(rates.real[order[0]]),
                "unstable_mode_count": int(np.sum(rates.real > 0)),
                "five_largest_rates_s": [
                    [float(rates[i].real), float(rates[i].imag)] for i in order[:5]
                ],
                "slowest_unforced_decay_rate_s": float(
                    np.max(np.linalg.eigvals(-fluid.mobility @ stiffness).real)
                ),
                "critical_acoustic_scale": critical_scale(fluid.mobility, acoustic, stiffness),
            }
        )
        if primary is None:
            primary = (fluid.mobility, rates, order)
    mobility, rates, order = primary
    values, vectors = np.linalg.eig(mobility @ (acoustic - stiffness))
    lead = vectors[:, np.argmax(values.real)].real.reshape(2, -1)
    radii = np.linspace(0, cfg.radius_m, 401)
    np.savez_compressed(
        output / "linearization.npz",
        coefficients_m=q,
        source_velocity_m_s=drive,
        stiffness_n_m=stiffness,
        acoustic_jacobian_n_m=acoustic,
        mobility=mobility,
        growth_rates_s=rates,
        radius_m=radii,
        leading_mode_height=np.array([space.evaluate(v, radii) for v in lead]),
    )
    report = {
        "force_balance_residual_relative": float(
            np.linalg.norm(force - mechanical) / np.linalg.norm(mechanical)
        ),
        "acoustic_tangent_central_difference_relative_error": tangent_checks,
        "acoustic_to_mechanical_stiffness_norm_ratio": float(
            np.linalg.norm(acoustic, 2) / np.linalg.norm(stiffness, 2)
        ),
        "acoustic_tangent_asymmetry": float(
            np.linalg.norm(acoustic - acoustic.T) / np.linalg.norm(acoustic)
        ),
        "mechanical_min_eigenvalue_n_m": float(np.linalg.eigvalsh(stiffness)[0]),
        "coupled_symmetric_part_negative_count": int(np.sum(symmetric < 0)),
        "static_eigenvalues_of_inverse_K_times_K_minus_A_real_sorted": np.sort(
            static.real
        ).tolist()[:8],
        "static_negative_count": int(np.sum(static.real < 0)),
        "frozen_to_coupled_amplification": float(
            np.linalg.norm(np.linalg.solve(coupled, stiffness), 2)
        ),
        "creeping_flow": mobility_rows,
        "flat_pinned_capillary_gravity_stiffness": pinned_capillary_gravity_stiffness(cfg),
        "wave_diagnostics": response.diagnostics(unit),
        "model": "Axisymmetric, inertia-free three-fluid Stokes mobility about the saved state",
        "not_assessed": [
            "non-axisymmetric (m >= 1) acoustic and flow modes",
            "fluid inertia and convection",
            "acoustic streaming base flow",
            "thermal feedback",
        ],
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    first = mobility_rows[0]
    lines = [
        "# Coupled linear stability of a held-command root",
        "",
        f"- Acoustic tangent check (central difference, relative): {tangent_checks}",
        f"- ‖A‖/‖K‖: {report['acoustic_to_mechanical_stiffness_norm_ratio']:.4g}",
        f"- Static modes with K - A not positive (K-relative): {report['static_negative_count']}",
        f"- Frozen-to-coupled amplification ‖(K - A)^-1 K‖: {report['frozen_to_coupled_amplification']:.4g}",
        (
            f"- Maximum creeping-flow growth rate: {first['max_growth_rate_s']:.4g} 1/s "
            f"({first['unstable_mode_count']} unstable modes)"
        ),
        f"- Critical acoustic scale s_c (M (s A - K) loses stability): {first['critical_acoustic_scale']}",
        "",
        "Axisymmetric and inertia-free; m >= 1 modes, streaming and heating are not assessed.",
        "",
        "```json",
        json.dumps(report, indent=2),
        "```",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:9]), flush=True)
