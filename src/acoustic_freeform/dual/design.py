"""Coherent source synthesis and an independent coupled stationary solve."""

import numpy as np
from scipy.linalg import block_diag, eigh, null_space
from scipy.optimize import least_squares, root


def synthesize(
    wave,
    space,
    target_coefficients,
    seed=20260922,
    initial=None,
    max_evaluations=600,
    effort_weight=1e-6,
    starts=1,
):
    response = wave.solve(target_coefficients)
    required, stiffness, _ = space.mechanics(target_coefficients)
    # A displacement-weighted traction residual avoids treating every pressure
    # mode as equally important to geometric accuracy.
    r = np.linspace(0, space.config.radius_m, 121)
    weight = np.where(r <= space.config.clear_radius_m, 1.0, 0.3)
    observe = block_diag(*(weight[:, None] * space.basis(r) for _ in range(2)))
    compliance_observation = observe @ np.linalg.inv(stiffness) / 1e-5
    channels = space.config.channels
    limit = space.config.max_source_speed_m_s / np.sqrt(2)
    rng = np.random.default_rng(seed)
    if initial is None:
        x0 = rng.normal(scale=0.12, size=2 * channels)
    else:
        x0 = np.r_[initial.real, initial.imag]
    x0 = np.clip(x0, -0.95 * limit, 0.95 * limit)

    def unpack(x):
        return x[:channels] + 1j * x[channels:]

    def residual(x):
        drive = unpack(x)
        error = compliance_observation @ (response.force(drive) - required)
        return np.r_[error, effort_weight * x / limit]

    def jacobian(x):
        derivative = compliance_observation @ response.force_jacobian(unpack(x))
        return np.vstack([derivative, effort_weight * np.eye(2 * channels) / limit])

    candidates = [x0]
    beta = compliance_observation.T @ (compliance_observation @ required)
    spectral = np.einsum("i,iab->ab", beta, response.force_kernels)
    _, eigenvectors = eigh(spectral)
    directions = [eigenvectors[:, -j] for j in range(1, min(5, channels) + 1)]
    directions += [rng.normal(size=channels) + 1j * rng.normal(size=channels) for _ in range(20)]
    target = compliance_observation @ required
    for direction in directions:
        direction = direction / np.linalg.norm(direction)
        response_ray = compliance_observation @ response.force(direction)
        power = max(
            0.0, np.dot(target, response_ray) / max(np.dot(response_ray, response_ray), 1e-30)
        )
        trial = np.r_[direction.real, direction.imag] * np.sqrt(power)
        trial *= min(1.0, 0.95 * limit / max(np.max(abs(trial)), 1e-30))
        candidates.append(trial)
    candidates.sort(key=lambda x: np.dot(residual(x), residual(x)))
    attempts = []
    opt = None
    for index, candidate in enumerate(candidates[:starts]):
        attempt = least_squares(
            residual,
            candidate,
            jac=jacobian,
            bounds=(-limit, limit),
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-11,
            max_nfev=max_evaluations,
        )
        attempts.append(
            {
                "start": index,
                "cost": float(attempt.cost),
                "evaluations": attempt.nfev,
                "success": bool(attempt.success),
            }
        )
        print("source start", attempts[-1], flush=True)
        if opt is None or attempt.cost < opt.cost:
            opt = attempt
    drive = unpack(opt.x)
    displacement = np.linalg.solve(stiffness, response.force(drive) - required)
    return (
        drive,
        response,
        {
            "success": bool(opt.success),
            "message": opt.message,
            "evaluations": opt.nfev,
            "cost": float(opt.cost),
            "linearized_height_residual_max_m": float(np.max(abs(observe @ displacement))),
            "effort_weight": effort_weight,
            "seed": seed,
            "attempts": attempts,
            "amplitude_constraint": "Each Cartesian source component bounded by v_max/sqrt(2).",
        },
    )


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


def synthesize_with_annulus(
    wave,
    space,
    target_coefficients,
    initial=None,
    max_evaluations=100,
    annulus_modes=6,
    seed=20260922,
    checkpoint=None,
):
    """Joint nonlinear wave/source inverse; only the non-optical annuli can move.

    Every residual recomputes acoustic scattering. Shape derivatives on the
    annuli use central finite differences of that complete response. No frozen
    acoustic sensitivity is used as a stability claim.
    """
    cfg = space.config
    count, channels = space.count, cfg.channels
    r_inner = np.linspace(0, cfg.clear_radius_m, 6 * count)
    annulus = null_space(space.basis(r_inner), rcond=1e-11)
    _, hessian, _ = space.mechanics(target_coefficients)
    smoothness = annulus.T @ hessian[:count, :count] @ annulus
    _, rotation = eigh(smoothness)
    annulus = annulus @ rotation[:, : min(annulus_modes, annulus.shape[1])]
    annulus_map = block_diag(annulus, annulus)
    n_annulus = annulus_map.shape[1]
    shape_scale = 1e-4
    limit = cfg.max_source_speed_m_s / np.sqrt(2)
    r_observe = np.linspace(0, cfg.radius_m, 101)
    observe = block_diag(space.basis(r_observe), space.basis(r_observe))
    metric = observe @ np.linalg.inv(hessian) / 1e-5
    rng = np.random.default_rng(seed)
    drive0 = rng.normal(0, 0.15, channels) + 1j * rng.normal(0, 0.15, channels)
    if initial is not None:
        drive0 = initial
    x0 = np.r_[
        np.clip(np.r_[drive0.real, drive0.imag], -0.95 * limit, 0.95 * limit), np.zeros(n_annulus)
    ]
    cache = {}
    calls = 0
    n_source = 2 * channels
    regularization = np.r_[np.full(n_source, 1e-6 / limit), np.full(n_annulus, 1e-6)]

    def unpack(x):
        drive = x[:channels] + 1j * x[channels:n_source]
        q = target_coefficients + (shape_scale * annulus_map @ x[n_source:]).reshape(2, count)
        return drive, q

    def evaluate(x):
        nonlocal calls
        if "x" not in cache or not np.array_equal(cache["x"], x):
            drive, q = unpack(x)
            response = wave.solve(q)
            force, stiffness, _ = space.mechanics(q)
            error = metric @ (response.force(drive) - force)
            cache.update(
                x=x.copy(), response=response, stiffness=stiffness, drive=drive, q=q, error=error
            )
            calls += 1
            print(
                f"joint inverse {calls}: displacement residual {np.max(abs(error)) * 1e-5:.4e} m",
                flush=True,
            )
            if checkpoint is not None:
                checkpoint(q, drive, calls, float(np.max(abs(error)) * 1e-5))
        return cache

    def residual(x):
        data = evaluate(x)
        return np.r_[data["error"], regularization * x]

    def jacobian(x):
        data = evaluate(x)
        source_derivative = data["response"].force_jacobian(data["drive"])
        columns = []
        step = 2e-8
        for direction in annulus_map.T:
            delta = direction.reshape(2, count) * step
            plus = wave.solve(data["q"] + delta, data["drive"]).force(np.ones(1, complex))
            minus = wave.solve(data["q"] - delta, data["drive"]).force(np.ones(1, complex))
            derivative = (plus - minus) / (2 * step) - data["stiffness"] @ direction
            columns.append(derivative * shape_scale)
        derivative = metric @ np.c_[source_derivative, np.stack(columns, axis=1)]
        return np.vstack([derivative, np.diag(regularization)])

    bounds = np.r_[np.full(n_source, limit), np.full(n_annulus, 2.0)]
    opt = least_squares(
        residual,
        x0,
        jac=jacobian,
        bounds=(-bounds, bounds),
        ftol=1e-11,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=max_evaluations,
    )
    final = evaluate(opt.x)
    report = {
        "success": bool(opt.success),
        "message": opt.message,
        "evaluations": opt.nfev,
        "cost": float(opt.cost),
        "annulus_modes_per_face": n_annulus // 2,
        "linearized_height_residual_max_m": float(np.max(abs(final["error"])) * 1e-5),
        "source_geometry": "Fixed ports; non-optical annuli varied with full wave updates.",
        "pupil_variation_basis_max": float(np.max(abs(space.basis(r_inner) @ annulus))),
        "seed": seed,
    }
    return final["drive"], final["q"], report
