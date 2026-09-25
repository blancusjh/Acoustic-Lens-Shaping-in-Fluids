"""Independent algebra/finite-difference checks for the SSL report; no simulation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import sympy as sp


def main():
    root = Path(__file__).resolve().parents[1]
    r, z = sp.symbols("r z", real=True)
    a, b, no, ni = sp.symbols("a b no ni", positive=True)
    path = no * sp.sqrt(r * r + (z + a) ** 2) + ni * sp.sqrt(r * r + (z - b) ** 2)
    curvature = -sp.diff(path, r, 2).subs({r: 0, z: 0}) / sp.diff(path, z).subs({r: 0, z: 0})
    assert sp.simplify(curvature - (ni / b + no / a) / (ni - no)) == 0
    h0, h1, zb, zt, g = sp.symbols("h0 h1 zb zt g", real=True)
    rho0, rho1, rho2 = sp.symbols("rho0 rho1 rho2", positive=True)
    potential = (
        g
        / 2
        * (rho0 * (h0 * h0 - zb * zb) + rho1 * (h1 * h1 - h0 * h0) + rho2 * (zt * zt - h1 * h1))
    )
    assert sp.simplify(sp.diff(potential, h0) - (rho0 - rho1) * g * h0) == 0
    assert sp.simplify(sp.diff(potential, h1) - (rho1 - rho2) * g * h1) == 0
    # Independent finite-difference spherical graph, away from rim and axis.
    radius = 0.012
    nodes = np.linspace(0.001, 0.006, 19)
    eps = 1e-6
    cap = lambda x: np.sqrt(radius**2 - x**2)
    slope = (cap(nodes + eps) - cap(nodes - eps)) / (2 * eps)
    second = (cap(nodes + eps) - 2 * cap(nodes) + cap(nodes - eps)) / eps**2
    kappa = -second / (1 + slope * slope) ** 1.5 - slope / (nodes * np.sqrt(1 + slope * slope))
    sphere_error = float(np.max(abs(kappa - 2 / radius)) / (2 / radius))
    assert sphere_error < 1e-6
    rng = np.random.default_rng(20260924)
    raw = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    kernel = (raw + raw.conj().T) / 2
    amplitude = rng.uniform(0.2, 1, size=5)
    phase = rng.normal(size=5)
    command = amplitude * np.exp(1j * phase)
    force = lambda u: float(np.vdot(u, kernel @ u).real)
    kw = kernel @ command
    da = 2 * np.real(kw.conj() * np.exp(1j * phase))
    dp = -2 * np.imag(kw.conj() * command)
    fd_a = []
    fd_p = []
    for unit in np.eye(5):
        fd_a.append(
            (
                force((amplitude + eps * unit) * np.exp(1j * phase))
                - force((amplitude - eps * unit) * np.exp(1j * phase))
            )
            / (2 * eps)
        )
        fd_p.append(
            (
                force(amplitude * np.exp(1j * (phase + eps * unit)))
                - force(amplitude * np.exp(1j * (phase - eps * unit)))
            )
            / (2 * eps)
        )
    error_a = float(np.max(abs(da - fd_a)))
    error_p = float(np.max(abs(dp - fd_p)))
    assert error_a < 1e-8 and error_p < 1e-8
    assert abs(np.sum(dp)) < 1e-12
    # Compartment incidence: sealed three-layer cell versus connected bath.
    incidence = np.array([[1.0, 0.0], [-1.0, 1.0], [0.0, -1.0]])
    sealed_constraints = np.eye(3)[:2] @ incidence
    bath_constraint = np.array([[-1.0, 1.0]])
    assert np.linalg.matrix_rank(sealed_constraints) == 2
    assert np.linalg.matrix_rank(bath_constraint) == 1
    np.testing.assert_allclose(incidence.sum(axis=0), 0)
    # Solve a constrained dissipation saddle system directly and compare
    # its load multiplier with the resistance after eliminating velocities.
    dissipation = np.array([[3.0, 0.2, 0.1], [0.2, 2.0, 0.3], [0.1, 0.3, 1.0]])
    trace = np.array([[1.0, 0.0, 0.4], [0.0, 1.0, -0.2]])
    wanted = np.array([0.3, -0.1])
    saddle = np.block([[dissipation, -trace.T], [trace, np.zeros((2, 2))]])
    solved = np.linalg.solve(saddle, np.r_[np.zeros(3), wanted])
    resistance = np.linalg.inv(trace @ np.linalg.solve(dissipation, trace.T))
    np.testing.assert_allclose(solved[3:], resistance @ wanted, atol=1e-14)
    np.testing.assert_allclose(trace @ solved[:3], wanted, atol=1e-14)
    np.testing.assert_allclose(
        solved[:3] @ dissipation @ solved[:3], wanted @ resistance @ wanted, atol=1e-14
    )
    result = {
        "real_object_real_image_vertex_identity": True,
        "three_phase_gravitational_energy_variation": True,
        "spherical_cap_curvature_fd_relative_error": sphere_error,
        "amplitude_gradient_fd_max_error": error_a,
        "phase_gradient_fd_max_error": error_p,
        "common_phase_null_direction": True,
        "compartment_pressure_gauge_ranks": {"sealed_three_layers": 2, "connected_bath": 1},
        "constrained_dissipation_multiplier_and_work_identity": True,
        "scope": "Algebra and finite-difference consistency checks, not a proof of reachability or physical validation.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "theory_source_sha256": hashlib.sha256(
            (root / "docs/theory/ssl-generator/theory.tex").read_bytes()
        ).hexdigest(),
        "multifluid_source_sha256": hashlib.sha256(
            (root / "docs/theory/ssl-generator/multifluid-variational.tex").read_bytes()
        ).hexdigest(),
    }
    output = root / "artifacts/theory/ssl-generator/algebra-checks.json"
    if output.exists():
        raise FileExistsError("Archive the previous algebra-check milestone first")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
