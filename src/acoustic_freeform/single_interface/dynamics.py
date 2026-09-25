"""Fluid inertia and acoustic shape sensitivity for maintained liquid surfaces.

The velocity reduction uses divergence-free finite-element resolvent snapshots.
It preserves kinetic energy and viscous dissipation; no fitted modal mass or
arbitrary relaxation time is introduced. It is a local, quiescent linear model,
without streaming, thermal evolution or actuator/sensor dynamics.
"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from ..core.provenance import capture_execution
from .acoustics import CavityAcoustics
from .hydrodynamics import assemble_fluid


def acoustic_derivatives(space, coefficients, drive, destination, step=1e-6, radiation_only=False):
    """Central differences of all quadratic load kernels along volume-null modes."""
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "acoustic-derivatives.npz"
    provenance = {
        "configuration": space.config.as_dict(),
        "difference_step": step,
        "radiation_only": radiation_only,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    q = space.tangent
    derivatives = []
    if checkpoint.exists():
        cache = np.load(checkpoint)
        if not (
            np.array_equal(cache["coefficients"], coefficients)
            and float(cache["difference_step"]) == step
            and str(cache["configuration_json"])
            == json.dumps(space.config.as_dict(), sort_keys=True)
            and bool(cache.get("radiation_only", False)) == radiation_only
        ):
            raise ValueError("Derivative cache belongs to a different state/configuration/step.")
        derivatives = list(cache["kernel_derivatives"])
    acoustic_config = (
        replace(space.config, bulk_streaming=False) if radiation_only else space.config
    )
    acoustic = CavityAcoustics(acoustic_config, space)
    field = acoustic.solve_basis(coefficients)
    for j in range(len(derivatives), q.shape[1]):
        delta = step * q[:, j]
        plus = acoustic.solve_basis(coefficients + delta).force_kernels
        minus = acoustic.solve_basis(coefficients - delta).force_kernels
        derivatives.append((plus - minus) / (2 * step))
        if j % 4 == 0 or j == q.shape[1] - 1:
            np.savez_compressed(
                checkpoint,
                coefficients=coefficients,
                difference_step=step,
                configuration_json=json.dumps(space.config.as_dict(), sort_keys=True),
                kernel_derivatives=np.asarray(derivatives),
                radiation_only=radiation_only,
            )
            print(f"acoustic shape sensitivity: {j + 1}/{q.shape[1]}", flush=True)
    dkernel = np.asarray(derivatives)
    dforce = np.einsum("a,jkab,b->kj", drive.conj(), dkernel, drive).real
    product = np.einsum("kab,b->ka", field.force_kernels, drive)
    actuator = 2 * np.concatenate([product.real, product.imag], axis=1)
    gradient, hessian = space.derivatives(coefficients)
    stiffness = q.T @ hessian @ q
    shape_response = q.T @ dforce
    result = {
        "capillary_stiffness": stiffness,
        "acoustic_stiffness": shape_response,
        "effective_stiffness": stiffness - shape_response,
        "actuator_jacobian": q.T @ actuator,
        "force_residual": q.T @ (field.force(drive) - gradient),
        "kernel_derivatives": dkernel,
    }
    np.savez_compressed(
        out / "linearization.npz", coefficients=coefficients, drive_m_s=drive, **result
    )
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return result


def linearize_for_control(space, coefficients, drive, destination, step=1e-6):
    """Radiation/viscous-fluid surrogate at a possibly streaming holding state.

    Body-load and base-circulation shape derivatives are omitted here. This is
    a control-design model, not a stability certificate for the full plant.
    Its commands must be tested in the nonlinear distributed-force simulation.
    """
    out = Path(destination)
    if (out / "report.json").exists():
        raise FileExistsError(f"Completed linearization exists: {out}")
    execution = capture_execution(out)
    derivative = acoustic_derivatives(space, coefficients, drive, out, step, radiation_only=True)
    _fluid, velocities, reduced = fluid_reduction(space, coefficients)
    a, b = state_matrices(
        reduced,
        derivative["effective_stiffness"],
        derivative["actuator_jacobian"],
        space.config.capillary_time_s,
    )
    rates = np.linalg.eigvals(a)
    # An independently directed finite difference checks the assembled Jacobian.
    direction = np.random.default_rng(9062026).normal(size=space.count - 1)
    direction /= np.linalg.norm(direction)
    acoustic = CavityAcoustics(replace(space.config, bulk_streaming=False), space)
    h = step / 2
    plus = acoustic.solve_basis(coefficients + h * space.tangent @ direction).force(drive)
    minus = acoustic.solve_basis(coefficients - h * space.tangent @ direction).force(drive)
    observed = space.tangent.T @ (plus - minus) / (2 * h)
    predicted = derivative["acoustic_stiffness"] @ direction
    check = float(np.linalg.norm(observed - predicted) / max(np.linalg.norm(observed), 1e-30))
    np.savez_compressed(
        out / "fluid-reduction.npz",
        **reduced,
        velocity_basis_coefficients=velocities,
        state_matrix_s_inv=a,
        input_matrix=b,
        coefficients=coefficients,
        drive_m_s=drive,
    )
    report = {
        "scope": linearize_for_control.__doc__,
        "configuration": space.config.as_dict(),
        "frozen_mean_flow_model_largest_real_rate_s_inv": float(rates.real.max()),
        "frozen_mean_flow_model_unstable_modes": int(np.sum(rates.real > 1e-5)),
        "directional_derivative_relative_error": check,
        "static_mobility_relative_error": reduced["static_relative_error"],
        "execution": execution,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print({k: v for k, v in report.items() if k not in ("execution", "configuration")}, flush=True)
    return report


def fluid_reduction(space, coefficients, shifts=(0.0, 1.0, 10.0, 100.0), relative_cutoff=1e-10):
    """Mass-orthonormal Ritz basis from (A + s*Oh^-2*M)^-1 C^T snapshots.

    Shifts are real, nonnegative and dimensionless (physical rate * mu*R/sigma).
    Include zero to reproduce the complete discrete Stokes mobility exactly.
    Retained modes are chosen by the kinetic Gram matrix, after column scaling.
    """
    fluid = assemble_fluid(space, coefficients)
    snapshots = []
    for shift in shifts:
        print(f"fluid resolvent shift: {shift:g}", flush=True)
        snapshots.append(fluid.solve(fluid.coupling.T, fluid.inertia * shift))
    candidates = np.concatenate(snapshots, axis=1)
    gram = candidates.T @ (fluid.mass @ candidates)
    scales = np.sqrt(np.maximum(np.diag(gram), 1e-300))
    gram /= scales[:, None] * scales[None, :]
    eigenvalues, eigenvectors = eigh(0.5 * (gram + gram.T))
    keep = eigenvalues > relative_cutoff * eigenvalues[-1]
    v = (candidates / scales) @ (eigenvectors[:, keep] / np.sqrt(eigenvalues[keep]))
    mass = v.T @ (fluid.mass @ v)
    # A final Cholesky orthogonalization removes roundoff in the smaller modes.
    lower = np.linalg.cholesky(0.5 * (mass + mass.T))
    v = np.linalg.solve(lower, v.T).T
    damping = v.T @ (fluid.viscosity @ v)
    damping = 0.5 * (damping + damping.T)
    coupling = fluid.coupling @ v
    full_static = fluid.coupling @ snapshots[0] if shifts[0] == 0 else None
    static = coupling @ np.linalg.solve(damping, coupling.T)
    relative_error = (
        None
        if full_static is None
        else float(np.linalg.norm(static - full_static) / np.linalg.norm(full_static))
    )
    return (
        fluid,
        v,
        {
            "mass": v.T @ (fluid.mass @ v),
            "damping": damping,
            "coupling": coupling,
            "static_mobility": static,
            "static_relative_error": relative_error,
            "inertia": fluid.inertia,
            "snapshot_shifts": np.asarray(shifts),
            "relative_cutoff": relative_cutoff,
        },
    )


def state_matrices(reduction, stiffness, actuator, time_scale):
    """Physical-time state is [volume-null surface displacement; fluid velocity]."""
    h, a, beta = reduction["coupling"], reduction["damping"], reduction["inertia"]
    ns = h.shape[0]
    system = np.block([[np.zeros((ns, ns)), h], [-h.T @ stiffness / beta, -a / beta]]) / time_scale
    input_matrix = (
        np.vstack([np.zeros((ns, actuator.shape[1])), h.T @ actuator / beta]) / time_scale
    )
    return system, input_matrix


def stability_report(space, coefficients, drive, destination, step=1e-6):
    if space.config.bulk_streaming:
        raise ValueError(
            "This spectral reduction assumes a quiescent base flow. An equivalent stationary surface force cannot replace a distributed inertial body-force model."
        )
    out = Path(destination)
    if (out / "report.json").exists():
        raise FileExistsError(f"Completed stability result exists: {out}")
    execution = capture_execution(out)
    derivative = acoustic_derivatives(space, coefficients, drive, out, step)
    fluid, velocities, reduced = fluid_reduction(space, coefficients)
    a, b = state_matrices(
        reduced,
        derivative["effective_stiffness"],
        derivative["actuator_jacobian"],
        space.config.capillary_time_s,
    )
    values = np.linalg.eigvals(a)
    stokes = (
        -reduced["static_mobility"]
        @ derivative["effective_stiffness"]
        / space.config.capillary_time_s
    )
    stokes_values = np.linalg.eigvals(stokes)
    # Independent resolvents at complex frequencies, not the real snapshot shifts.
    checks = []
    h, damping = reduced["coupling"], reduced["damping"]
    for omega in (0.3, 3.0, 30.0):
        s = 1j * omega
        exact = fluid.coupling @ fluid.solve(fluid.coupling.T.astype(complex), s * fluid.inertia)
        ritz = h @ np.linalg.solve(damping + s * fluid.inertia * np.eye(len(damping)), h.T)
        checks.append(
            {
                "dimensionless_laplace_frequency": omega,
                "relative_mobility_error": float(
                    np.linalg.norm(exact - ritz) / np.linalg.norm(exact)
                ),
            }
        )
    np.savez_compressed(
        out / "fluid-reduction.npz",
        **reduced,
        velocity_basis_coefficients=velocities,
        state_matrix_s_inv=a,
        input_matrix=b,
        coefficients=coefficients,
        drive_m_s=drive,
    )
    report = {
        "execution": execution,
        "scope": "Local axisymmetric stability of a quiescent interface with instantaneous cycle-averaged acoustics, fixed complex wall velocities and unsteady viscous incompressible flow. No streaming, heating or apparatus calibration.",
        "configuration": space.config.as_dict(),
        "inertial_coefficient_Oh_inverse_squared": fluid.inertia,
        "surface_dofs": len(space.tangent.T),
        "fluid_reduced_dofs": len(reduced["damping"]),
        "static_mobility_relative_error": reduced["static_relative_error"],
        "resolvent_checks": checks,
        "largest_real_growth_rate_s_inv": float(values.real.max()),
        "unstable_modes": int(np.count_nonzero(values.real > 1e-5)),
        "growth_rates_real_s_inv": values.real.tolist(),
        "growth_rates_imag_s_inv": values.imag.tolist(),
        "stokes_largest_real_growth_rate_s_inv": float(stokes_values.real.max()),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        {
            k: v
            for k, v in report.items()
            if k not in ("configuration", "growth_rates_real_s_inv", "growth_rates_imag_s_inv")
        },
        flush=True,
    )
    return report
