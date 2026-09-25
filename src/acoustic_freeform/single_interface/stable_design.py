"""Search physical holding drives using optical and inertial stability objectives.

The force kernels are re-solved on every design surface. Stability sensitivities
are locally frozen during an inner design and must be independently recomputed
at the resulting equilibrium. An optimization penalty is not a certificate.
"""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig

from .config import LensConfig
from .design import design_stationary
from .dynamics import state_matrices
from .stationary import export_stationary
from .surface import SurfaceSpace


class StabilityObjective:
    """Penalize continuous growth rates above a requested negative margin.

    Eigenvalue derivatives use left/right eigenvectors. The expression is local
    to simple eigenvalues; independent finite differences and final spectral
    recomputation are required, especially near repeated eigenvalues.
    """

    def __init__(self, space, derivative, fluid, margin_s_inv=5.0, weight=10.0):
        self.kernel = np.einsum("ki,jkab->ijab", space.tangent, derivative["kernel_derivatives"])
        self.fluid = fluid
        self.stiffness = derivative["capillary_stiffness"]
        self.time_scale = space.config.capillary_time_s
        self.margin, self.weight = margin_s_inv, weight
        self.cached_drive = None

    def evaluate(self, drive):
        if self.cached_drive is not None and np.array_equal(drive, self.cached_drive):
            return self.cached
        shape = np.einsum("a,ijab,b->ij", drive.conj(), self.kernel, drive).real
        a, _ = state_matrices(
            self.fluid, self.stiffness - shape, np.zeros((len(shape), 0)), self.time_scale
        )
        values, left, right = eig(a, left=True, right=True)
        order = np.lexsort((values.imag, values.real))
        values, left, right = values[order], left[:, order], right[:, order]
        active = values.real + self.margin > 0
        residual = self.weight * np.maximum(values.real + self.margin, 0)
        product = np.einsum("ijab,b->ija", self.kernel, drive)
        dj = 2 * np.concatenate([product.real, product.imag], axis=2)
        ns = len(shape)
        g = self.fluid["coupling"].T / (self.fluid["inertia"] * self.time_scale)
        jac = np.zeros((len(values), 2 * len(drive)))
        for k in np.flatnonzero(active):
            numerator = np.einsum("i,ijc,j->c", left[ns:, k].conj() @ g, dj, right[:ns, k])
            jac[k] = self.weight * np.real(numerator / np.vdot(left[:, k], right[:, k]))
        self.cached_drive = drive.copy()
        self.cached = residual, jac, values
        return self.cached

    def residual(self, drive):
        return self.evaluate(drive)[0]

    def jacobian(self, drive):
        return self.evaluate(drive)[1]


def run_stable_design(source, stability_directory, destination, margin_s_inv=5.0):
    source, stability, out = Path(source), Path(stability_directory), Path(destination)
    if (out / "report.json").exists():
        raise FileExistsError(f"Completed design exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    cfg = LensConfig(**json.loads((source / "configuration.json").read_text()))
    data = np.load(source / "stationary.npz")
    derivative = dict(np.load(stability / "linearization.npz"))
    fluid = dict(np.load(stability / "fluid-reduction.npz"))
    if not np.array_equal(data["coefficients"], derivative["coefficients"]):
        raise ValueError("Stability sensitivities must belong to the input equilibrium.")
    space = SurfaceSpace(cfg)
    objective = StabilityObjective(space, derivative, fluid, margin_s_inv)
    # Directional check of the active spectral penalty before using its gradient.
    drive = data["drive_m_s"]
    rng = np.random.default_rng(9062026)
    direction = rng.normal(size=2 * len(drive))
    direction /= np.linalg.norm(direction)
    complex_direction = direction[: len(drive)] + 1j * direction[len(drive) :]
    eps = 1e-5
    fd = (
        np.dot(
            objective.residual(drive + eps * complex_direction),
            objective.residual(drive + eps * complex_direction),
        )
        - np.dot(
            objective.residual(drive - eps * complex_direction),
            objective.residual(drive - eps * complex_direction),
        )
    ) / (2 * eps)
    analytic = 2 * objective.residual(drive) @ objective.jacobian(drive) @ direction
    gradient_error = float(abs(fd - analytic) / max(1.0, abs(fd), abs(analytic)))
    if gradient_error > 1e-3:
        raise RuntimeError(
            f"Spectral objective gradient failed directional check: {gradient_error}"
        )

    def checkpoint(c, drive, history):
        np.savez_compressed(out / "design-checkpoint.npz", coefficients=c, drive_m_s=drive)
        (out / "design-progress.json").write_text(json.dumps(history, indent=2) + "\n")

    # This is an optimization stopping tolerance, not force-balance accuracy.
    # export_stationary independently re-solves the final fixed drive to 0.1 nm.
    c, drive, _, history = design_stationary(
        space,
        starts=1,
        initial_drive=drive,
        initial_coefficients=data["coefficients"],
        extra_objective=objective,
        tolerance_m=1e-8,
        checkpoint=checkpoint,
    )
    report = export_stationary(cfg, out, c, drive, history)
    followup = {
        "scope": "Holding drive optimized against a locally frozen inertial spectrum. This is a design diagnostic; independently recompute stability at the new equilibrium.",
        "sensitivity_source": str(stability.resolve()),
        "requested_decay_margin_s_inv": margin_s_inv,
        "spectral_objective_gradient_relative_error": gradient_error,
        "frozen_sensitivity_largest_real_growth_rate_s_inv": float(
            objective.evaluate(drive)[2].real.max()
        ),
    }
    (out / "stability-design.json").write_text(json.dumps(followup, indent=2) + "\n")
    print(followup, flush=True)
    return report
