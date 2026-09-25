"""Fixed-command coupled stationary solve: fresh wave fields on every trial shape."""

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import root


def coupled_equilibrium(
    wave, space, drive, initial, max_evaluations=160, method="hybr", progress=None
):
    """Solve force balance with fresh wave fields on every proposed shape.

    Newton iterations are not physical time. The initial shape is only a root
    solver seed; the target does not appear anywhere in this residual.
    """
    shape = initial.shape
    scale = 1e-4
    _, stiffness, _ = space.mechanics(initial)
    inverse = np.linalg.inv(stiffness)
    calls = 0
    last_x, last_value = None, None
    if method not in ("hybr", "krylov"):
        raise ValueError("Stationary solver must be hybr or krylov.")

    def residual(x):
        nonlocal calls, last_x, last_value
        if last_x is not None and np.array_equal(x, last_x):
            return last_value.copy()
        q = (x * scale).reshape(shape)
        required, _, _ = space.mechanics(q)
        response = wave.solve(q, drive)
        actual = response.force(np.ones(1, complex))
        calls += 1
        value = inverse @ (required - actual) / scale
        last_x, last_value = np.array(x).copy(), value.copy()
        if progress is not None:
            progress(
                {
                    "wave_solves": calls,
                    "wave_linear_residual": response.residual,
                    "wave_krylov_iterations": response.krylov_iterations,
                    "preconditioner_refreshes": response.preconditioner_refreshes,
                    "trial_coefficients_m": q.tolist(),
                    "trial_compliance_residual_max_m": float(
                        np.max(abs(block_diag(space.b, space.b) @ value)) * scale
                    ),
                    "scope": "Trial root residual, not time evolution or accepted equilibrium.",
                }
            )
        return value

    initial_x = initial.ravel() / scale
    initial_value = residual(initial_x)
    initial_residual_m = float(np.max(abs(block_diag(space.b, space.b) @ initial_value)) * scale)
    if np.isfinite(initial_residual_m) and initial_residual_m <= 1e-11:
        return initial.copy(), {
            "solver_success": True,
            "method": method,
            "message": "Initial seed satisfies independently recomputed coupled force balance.",
            "initial_seed_accepted": True,
            "wave_solves": calls,
            "compliance_residual_max_m": initial_residual_m,
            "interpretation": "Stationary force-balance check, not a formation trajectory.",
        }

    options = (
        {"xtol": 1e-8, "maxfev": max_evaluations, "eps": 1e-10}
        if method == "hybr"
        else {"fatol": 1e-8, "maxiter": max_evaluations, "jac_options": {"rdiff": 1e-5}}
    )
    solution = root(
        residual,
        initial_x,
        method=method,
        options=options,
    )
    last = residual(solution.x)
    q = (solution.x * scale).reshape(shape)
    residual_m = float(np.max(abs(block_diag(space.b, space.b) @ last)) * scale)
    if not np.isfinite(residual_m) or not np.all(np.isfinite(q)) or residual_m > 1e-11:
        raise RuntimeError(f"Coupled equilibrium unresolved: {residual_m:g} m, {solution.message}")
    return q, {
        "solver_success": bool(solution.success),
        "method": method,
        "message": solution.message,
        "wave_solves": calls,
        "compliance_residual_max_m": residual_m,
        "interpretation": "Stationary root solve, not a formation trajectory.",
    }
