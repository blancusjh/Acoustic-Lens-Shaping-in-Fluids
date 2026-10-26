"""Faithful two-face inverse: fit the reachable equilibrium of a modal source.

Implements the formulation of docs/theory/sections/10e_faithful_inverse.tex.
The unknown is a vector of complex modal coefficients c (region values Phi c).
The residual is the pupil height error of the end of the start-up branch, not
a frozen load mismatch at the target; its Jacobian uses the coupled operator
(K - A)^-1. Declared parameter tolerances add first-order equilibrium shifts,
and source power and sampled peak pressure are hard caps on every step.
"""

from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares, minimize

from ..acoustics.helmholtz import DualAcoustics
from ..forward.branch import startup_branch
from ..mechanics.surface import DualSurface


def real_coordinates(c):
    return np.r_[c.real, c.imag]


def complex_coefficients(x):
    half = len(x) // 2
    return x[:half] + 1j * x[half:]


class ModalOperators:
    """Modal force kernels, pressure samples and power form at one geometry."""

    def __init__(self, wave, q, modes, pressure_samples):
        response = wave.solve(q)
        kernels = response.force_kernels
        self.kernels = np.einsum("ai,nab,bj->nij", modes, kernels, modes)
        pressure = response.solution @ modes
        # Boundary absorption equals source work for a passive, lossless interior.
        self.power = np.pi * pressure.conj().T @ (response.boundary_matrix @ pressure)
        self.power = (self.power + self.power.conj().T) / 2
        order = np.argsort(-np.linalg.norm(pressure, axis=1))
        self.pressure_rows = pressure[order[:pressure_samples]]
        self.all_pressure = pressure

    def force(self, c):
        return np.einsum("i,nij,j->n", c.conj(), self.kernels, c).real

    def force_jacobian(self, c):
        """Derivative with respect to (Re c, Im c), rows are forces."""
        kc = np.einsum("nij,j->ni", self.kernels, c)
        return 2 * np.c_[kc.real, kc.imag]

    def source_power(self, c):
        return float(np.real(c.conj() @ self.power @ c))

    def peak_pressure(self, c):
        return float(np.max(abs(self.all_pressure @ c)))


class FaithfulProblem:
    def __init__(
        self,
        cfg,
        modes,
        targets,
        pupil_radii,
        height_scale_m,
        sigmas,
        perturbations=(),
        pressure_samples=400,
        tolerance_m=1e-13,
        mobility=None,
    ):
        self.cfg, self.modes, self.targets = cfg, modes, targets
        self.space = DualSurface(cfg, allow_volume_offsets=True)
        self.start = self.space.unforced_equilibrium()
        self.wave = DualAcoustics(cfg, self.space, linear_solver="static_condensed")
        self.radii = np.asarray(pupil_radii, float)
        self.pupil_basis = self.space.basis(self.radii)
        self.target_heights = np.array([t.evaluate(self.radii) for t in targets])
        self.scale = height_scale_m
        self.sigmas = sigmas
        self.pressure_samples = pressure_samples
        self.tolerance_m = tolerance_m
        self.mobility = mobility
        # Each perturbation declares apparatus overrides of one tolerance.
        self.perturbations = [
            {
                **p,
                "config": replace(
                    cfg,
                    **{
                        k: tuple(v) if isinstance(v, list) else v for k, v in p["overrides"].items()
                    },
                ),
            }
            for p in perturbations
        ]

    def pupil_heights(self, q):
        return np.array([self.space.height(j, v, self.radii) for j, v in enumerate(q)])

    def reachable(self, c, progress=None):
        return startup_branch(
            self.wave,
            self.space,
            self.modes @ c,
            self.sigmas,
            initial=self.start,
            tolerance_m=self.tolerance_m,
            mobility=self.mobility,
            progress=progress,
        )

    def linearize(self, c, branch):
        """Residuals and Jacobians at the end of a completed branch."""
        q = branch["points"][-1]["coefficients_m"]
        stiffness = branch["final_stiffness"]
        coupled = stiffness - branch["final_acoustic_jacobian"]
        operators = ModalOperators(self.wave, q, self.modes, self.pressure_samples)
        observe = np.kron(np.eye(2), self.pupil_basis) / self.scale
        response = np.linalg.solve(coupled, operators.force_jacobian(c))
        blocks = {
            "shape": (
                ((self.pupil_heights(q) - self.target_heights) / self.scale).ravel(),
                observe @ response,
            )
        }
        grad, _, _ = self.space.mechanics(q)
        for p in self.perturbations:
            space = DualSurface(p["config"], allow_volume_offsets=True)
            wave = DualAcoustics(p["config"], space, linear_solver="static_condensed")
            perturbed = ModalOperators(wave, q, self.modes, 1)
            difference = perturbed.kernels - operators.kernels
            force = np.einsum("i,nij,j->n", c.conj(), difference, c).real
            perturbed_grad, _, _ = space.mechanics(q)
            shift = np.linalg.solve(coupled, force - (perturbed_grad - grad))
            kc = np.einsum("nij,j->ni", difference, c)
            jacobian = np.linalg.solve(coupled, 2 * np.c_[kc.real, kc.imag])
            blocks[p["name"]] = (observe @ shift, observe @ jacobian)
        return q, operators, blocks

    @staticmethod
    def stacked(blocks):
        residual = np.concatenate([b[0] for b in blocks.values()])
        jacobian = np.vstack([b[1] for b in blocks.values()])
        return residual, jacobian

    def step(self, c, operators, blocks, radius, limits, fixed):
        """Trust-region Gauss-Newton step under exact power and pressure caps.

        ``fixed`` indexes real coordinates held at zero (one phase). Both caps
        are convex in c and are imposed exactly at the new command.
        """
        residual, jacobian = self.stacked(blocks)
        x = real_coordinates(c)
        free = np.setdiff1d(np.arange(len(x)), fixed)
        reduced = jacobian[:, free]
        scale = max(np.linalg.norm(x), 1e-12)

        # Every term is normalized to order one; SLSQP fails on the raw scales
        # (pressures squared in Pa^2 against powers in W).
        normal = max(0.5 * residual @ residual, 1e-300)

        def merit(d):
            value = residual + reduced @ d
            return 0.5 * value @ value

        def objective(d):
            value = residual + reduced @ d
            return 0.5 * value @ value / normal, reduced.T @ value / normal

        def new(d):
            y = x.copy()
            y[free] += d
            return complex_coefficients(y)

        size = radius * scale
        constraints = [
            {"type": "ineq", "fun": lambda d: 1 - (d @ d) / size**2},
            {
                "type": "ineq",
                "fun": lambda d: 1 - operators.source_power(new(d)) / limits["source_power_w"],
            },
            {
                "type": "ineq",
                "fun": lambda d: (
                    1 - abs(operators.pressure_rows @ new(d)) ** 2 / limits["peak_pressure_pa"] ** 2
                ),
            },
        ]
        solution = minimize(
            objective,
            np.zeros(len(free)),
            jac=True,
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-14},
        )
        return new(solution.x), float(merit(solution.x)), solution


def frozen_start(problem, limits, seeds, rng, evaluations=400, seed_power_fraction=0.01):
    """Initial command from a frozen pupil-only correction of the acoustics-off state.

    Only an initial guess: forces are quadratic, so a Gauss-Newton step from
    c = 0 is degenerate. About the acoustics-off equilibrium q0, the pupil
    heights move by H K0^-1 F_ac(q0; c) to first order; the fit asks this to
    close the pupil residual of the meniscus. The frozen compliance is used
    here and nowhere in acceptance. Candidates are scaled into both caps.
    """
    space = problem.space
    q = problem.start
    _, stiffness, _ = space.mechanics(q)
    operators = ModalOperators(problem.wave, q, problem.modes, problem.pressure_samples)
    observe = np.kron(np.eye(2), problem.pupil_basis) / problem.scale
    compliance = observe @ np.linalg.inv(stiffness)
    target = ((problem.target_heights - problem.pupil_heights(q)) / problem.scale).ravel()
    m = problem.modes.shape[1]
    candidates = []
    for _ in range(seeds):
        c = rng.normal(size=m) + 1j * rng.normal(size=m)
        c *= np.sqrt(
            seed_power_fraction * limits["source_power_w"] / max(operators.source_power(c), 1e-30)
        )

        def fun(x):
            return compliance @ operators.force(complex_coefficients(x)) - target

        def jac(x):
            return compliance @ operators.force_jacobian(complex_coefficients(x))

        fit = least_squares(fun, real_coordinates(c), jac=jac, max_nfev=evaluations)
        c = complex_coefficients(fit.x)
        factor = min(
            1.0,
            np.sqrt(limits["source_power_w"] / max(operators.source_power(c), 1e-30)),
            limits["peak_pressure_pa"] / max(operators.peak_pressure(c), 1e-30),
        )
        c = c * factor
        candidates.append(
            {
                "coefficients": c,
                "frozen_pupil_defect_scaled": float(np.max(abs(fun(real_coordinates(c))))),
                "cap_scale_factor": float(factor),
            }
        )
    return sorted(candidates, key=lambda item: item["frozen_pupil_defect_scaled"])


def best_fill_volumes(cfg, targets, radii):
    """Per-face displaced volume whose acoustics-off meniscus best fits the pupil.

    Minimizes the maximum pupil height error of the capillary-gravity meniscus
    at fixed pinned rims; each face is fitted with the other at zero offset,
    which is exact because the two acoustics-off interfaces do not interact.
    """
    from scipy.optimize import minimize_scalar

    volumes, errors = [], []
    for j, target in enumerate(targets):
        heights = target.evaluate(radii)

        def error(volume, j=j, heights=heights):
            offsets = [0.0, 0.0]
            offsets[j] = volume
            space = DualSurface(replace(cfg, volume_offsets_m3=tuple(offsets)), True)
            q = space.unforced_equilibrium()
            return float(np.max(abs(space.height(j, q[j], radii) - heights)))

        # The pupil max error is only piecewise smooth, so Brent's parabolic steps
        # stall; a bounded golden search around the target's own volume does not.
        guess = (
            target.volume_offset()
            if hasattr(target, "volume_offset")
            else np.pi * cfg.radius_m**2 / 2 * heights[0]
        )

        def safe(volume, error=error):
            try:
                return error(volume)
            except (ValueError, RuntimeError):
                return np.inf

        fit = minimize_scalar(
            safe,
            bounds=sorted((0.5 * guess, 1.5 * guess)),
            method="bounded",
            options={"xatol": 1e-6 * abs(guess)},
        )
        volumes.append(float(fit.x))
        errors.append(float(fit.fun))
    return volumes, errors
