"""Carrier-aware frozen-target inverse screening; never a coupled accuracy pass.

Run: python -m acoustic_freeform.inverse.screening CONFIG --out NEW_DIRECTORY
Reuse a declared frozen operator using --cache-from PREVIOUS_DIRECTORY.
"""

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import minimize

from ..mechanics.surface import CartesianPatch
from ..optics.raytrace import fixed_detector_spot_jacobian


def complex_linear_jacobian(matrix):
    return np.block([[matrix.real, -matrix.imag], [matrix.imag, matrix.real]])


def add_pupil_slope_observations(cache, space, stiffness, slope_scale_m):
    """Append scaled slope errors, keeping height rows explicitly distinguishable.

    This is a regularity proxy, not an optical spot certificate. A scale of
    1 mm makes a 10 microradian slope residual equal to a 10 nm height residual.
    """
    if not np.isfinite(slope_scale_m) or slope_scale_m <= 0:
        raise ValueError("Slope scale must be finite and positive")
    radius = space.config.clear_radius_m
    radii = np.unique(np.r_[0, space.r[space.r <= radius], radius])
    extra = slope_scale_m * block_diag(space.basis(radii, 1), space.basis(radii, 1))
    existing = len(cache["observation_radii_m"])
    cache["compliance_observation_m_n"] = np.vstack(
        [cache["compliance_observation_m_n"], extra @ np.linalg.inv(stiffness)]
    )
    cache["observation_radii_m"] = np.r_[cache["observation_radii_m"], np.tile(radii, 2)]
    cache["observation_faces"] = np.r_[cache["observation_faces"], np.repeat([0, 1], len(radii))]
    cache["observation_weights"] = np.r_[cache["observation_weights"], np.ones(2 * len(radii))]
    cache["observation_kind"] = np.r_[
        np.zeros(existing, dtype=int), np.ones(2 * len(radii), dtype=int)
    ]


def add_joint_spot_observations(cache, space, stiffness, faces, tolerance_m, rays=501):
    """Append the local two-interface fixed-detector optical map, not a slope proxy."""
    if not np.isfinite(tolerance_m) or tolerance_m <= 0 or rays < 3:
        raise ValueError("Positive spot tolerance and at least three rays are required")
    cfg = space.config
    vertices = np.array(cfg.levels_m[1:3]) + [f["vertex_displacement_m"] for f in faces]
    z0 = vertices[0] + faces[0]["optical"]["z_o_m"]
    z2 = vertices[1] + faces[1]["optical"]["z_i_m"]
    if (
        abs(vertices[0] + faces[0]["optical"]["z_i_m"] - vertices[1] - faces[1]["optical"]["z_o_m"])
        > 1e-14
    ):
        raise ValueError("Optical target must share a laboratory intermediate conjugate")
    indices = [faces[0]["optical"]["n_o"], faces[0]["optical"]["n_i"], faces[1]["optical"]["n_i"]]
    if faces[1]["optical"]["n_o"] != indices[1]:
        raise ValueError("Inconsistent middle optical medium")
    r = np.linspace(0, cfg.clear_radius_m, rays)
    target = CartesianPatch(cfg, 0, **faces[0])
    spot, derivative = fixed_detector_spot_jacobian(
        space,
        cache["target_coefficients_m"],
        indices,
        z0,
        z2,
        r,
        cfg.levels_m[1] + target.evaluate(r),
    )
    scale = 1e-8 / tolerance_m
    existing = len(cache["observation_radii_m"])
    cache["compliance_observation_m_n"] = np.vstack(
        [cache["compliance_observation_m_n"], scale * derivative @ np.linalg.inv(stiffness)]
    )
    cache["observation_offset_m"] = np.r_[np.zeros(existing), scale * spot]
    cache["observation_radii_m"] = np.r_[cache["observation_radii_m"], r]
    # Assign the joint optical constraint to the first epigraph group; it still
    # differentiates BOTH faces. Kind 2 prevents misreporting it as front height.
    cache["observation_faces"] = np.r_[cache["observation_faces"], np.zeros(rays, dtype=int)]
    cache["observation_weights"] = np.r_[cache["observation_weights"], np.ones(rays)]
    cache["observation_kind"] = np.r_[
        cache.get("observation_kind", np.zeros(existing, dtype=int)), np.full(rays, 2, dtype=int)
    ]
    cache["observation_optical_scale"] = np.array(scale)


class PrecisionObjective:
    """Dimensionless residual with exact source derivatives at frozen geometry."""

    def __init__(
        self,
        cache,
        carrier_weight,
        source_limit,
        effort_weight=1e-5,
        peak_power=None,
        clear_radius_m=None,
        include_carrier_in_peak=True,
    ):
        self.kernels = cache["force_kernels"]
        self.required = cache["required_force_n"]
        self.observe = cache["compliance_observation_m_n"] / 1e-8
        self.offset = cache.get("observation_offset_m", np.zeros(len(self.observe))) / 1e-8
        self.carrier = cache["carrier_height_response_s"] / 1e-8
        self.channels = self.kernels.shape[-1]
        self.shape_weight = cache["observation_weights"] / np.sqrt(len(self.observe))
        self.carrier_weight = carrier_weight / np.sqrt(len(self.carrier))
        self.source_limit = source_limit
        self.effort = effort_weight
        self.linear_jac = complex_linear_jacobian(self.carrier) * self.carrier_weight
        self.peak_power = peak_power
        self.include_carrier_in_peak = include_carrier_in_peak
        if peak_power is not None:
            if peak_power < 2 or not np.isfinite(peak_power):
                raise ValueError("Peak power must be finite and at least two.")
            pupil = cache["observation_radii_m"] <= clear_radius_m
            self.mean_masks = [pupil & (cache["observation_faces"] == j) for j in range(2)]
            self.carrier_masks = [cache["carrier_faces"] == j for j in range(2)]
            self.annulus = ~pupil

    def peak_norm(self, values):
        """Unnormalized Lp majorant of sampled maximum, with exact derivative."""
        magnitude = abs(values)
        scale = np.max(magnitude)
        if scale == 0:
            return 0.0, np.zeros_like(magnitude)
        scaled = magnitude / scale
        norm = np.sum(scaled**self.peak_power) ** (1 / self.peak_power)
        gradient = (scaled / norm) ** (self.peak_power - 1)
        return scale * norm, gradient

    def fit_residual(self, height, carrier):
        if self.peak_power is None:
            return np.r_[
                self.shape_weight * height,
                self.carrier_weight * carrier.real,
                self.carrier_weight * carrier.imag,
            ]
        peaks = [
            self.peak_norm(height[m])[0]
            + (self.peak_norm(carrier[b])[0] if self.include_carrier_in_peak else 0)
            for m, b in zip(self.mean_masks, self.carrier_masks)
        ]
        return np.r_[peaks, (self.shape_weight * height)[self.annulus]]

    def fit_jacobian(self, height, carrier, height_jac, carrier_matrix):
        if self.peak_power is None:
            return np.vstack(
                [
                    self.shape_weight[:, None] * height_jac,
                    self.carrier_weight * complex_linear_jacobian(carrier_matrix),
                ]
            )
        rows = []
        for m, b in zip(self.mean_masks, self.carrier_masks):
            _, dh = self.peak_norm(height[m])
            if not self.include_carrier_in_peak:
                rows.append((dh * np.sign(height[m])) @ height_jac[m])
                continue
            _, db = self.peak_norm(carrier[b])
            direction = np.divide(
                carrier[b].conj(),
                abs(carrier[b]),
                out=np.zeros_like(carrier[b]),
                where=abs(carrier[b]) > 0,
            )
            dc = direction[:, None] * carrier_matrix[b]
            rows.append((dh * np.sign(height[m])) @ height_jac[m] + db @ np.c_[dc.real, -dc.imag])
        return np.vstack([rows, (self.shape_weight[:, None] * height_jac)[self.annulus]])

    def unpack(self, x):
        return x[: self.channels] + 1j * x[self.channels :]

    def metrics(self, x):
        a = self.unpack(x)
        force = np.einsum("a,iab,b->i", a.conj(), self.kernels, a).real
        height = self.observe @ (force - self.required) + self.offset
        carrier = self.carrier @ a
        return height, carrier

    def residual(self, x):
        height, carrier = self.metrics(x)
        return np.r_[
            self.fit_residual(height, carrier),
            self.effort * x / self.source_limit,
        ]

    def jacobian(self, x):
        ka = np.einsum("iab,b->ia", self.kernels, self.unpack(x))
        derivative = self.observe @ (2 * np.c_[ka.real, ka.imag])
        height, carrier = self.metrics(x)
        return np.vstack(
            [
                self.fit_jacobian(height, carrier, derivative, self.carrier),
                self.effort * np.eye(2 * self.channels) / self.source_limit,
            ]
        )


class WhitenedObjective:
    """Observable source coordinates and a penalized physical amplitude disk.

    Eigenvalue truncation is a numerical restriction, not a reachability claim.
    A final uniform rescaling enforces the physical command bound exactly.
    """

    def __init__(
        self, base, cutoff=1e-12, bound_weight=1e4, carrier_ceiling_m=None, spectral_method="gram"
    ):
        self.base = base
        self.limit = base.source_limit * np.sqrt(2)
        gram = np.einsum("iab,iac->bc", base.kernels.conj(), base.kernels, optimize=True)
        projection = np.eye(base.channels, dtype=complex)
        self.carrier_subspace_bound_m = None
        if carrier_ceiling_m is not None:
            velocity_gram = base.carrier.conj().T @ base.carrier
            values, vectors = np.linalg.eigh(velocity_gram)
            threshold = carrier_ceiling_m / 1e-8 / (np.sqrt(base.channels) * self.limit)
            projection = vectors[:, values <= threshold**2]
            if projection.shape[1] == 0:
                raise ValueError(
                    "No retained source directions satisfy the chosen carrier ceiling."
                )
            self.carrier_subspace_bound_m = float(
                np.linalg.norm(base.carrier @ projection, ord=2)
                * 1e-8
                * np.sqrt(base.channels)
                * self.limit
            )
            if self.carrier_subspace_bound_m > carrier_ceiling_m * (1 + 1e-6):
                raise ValueError("The source subspace failed its carrier operator norm check.")
            gram = projection.conj().T @ gram @ projection
        if spectral_method == "svd":
            # Avoid squaring the condition number before resolving weak source
            # directions. Gram roundoff near epsilon can create false modes.
            operator = base.kernels.reshape(-1, base.channels) @ projection
            _, singular, right = np.linalg.svd(operator, full_matrices=False)
            eigenvalues = singular[::-1] ** 2
            vectors = right.conj().T[:, ::-1]
        elif spectral_method == "gram":
            eigenvalues, vectors = np.linalg.eigh(gram)
        else:
            raise ValueError("Unknown source spectral method")
        keep = eigenvalues > cutoff * eigenvalues[-1]
        self.transform = (projection @ vectors[:, keep]) * (
            eigenvalues[-1] / eigenvalues[keep]
        ) ** 0.25
        self.real_transform = complex_linear_jacobian(self.transform)
        self.channels = self.transform.shape[1]
        self.bound_weight = bound_weight
        self.eigenvalues = eigenvalues
        self.reduced_kernels = np.einsum(
            "ap,iab,bq->ipq", self.transform.conj(), base.kernels, self.transform, optimize=True
        )
        self.reduced_carrier = base.carrier @ self.transform
        self.reduced_linear_jac = (
            complex_linear_jacobian(self.reduced_carrier) * base.carrier_weight
        )

    def encode(self, source):
        z = np.linalg.lstsq(self.transform, source, rcond=None)[0]
        return np.r_[z.real, z.imag]

    def unpack(self, x):
        return self.transform @ (x[: self.channels] + 1j * x[self.channels :])

    def metrics(self, x):
        """Same physical quadratic metrics, evaluated in retained coordinates."""
        z = x[: self.channels] + 1j * x[self.channels :]
        ka = np.einsum("iab,b->ia", self.reduced_kernels, z)
        force = np.einsum("a,ia->i", z.conj(), ka).real
        return (
            self.base.observe @ (force - self.base.required) + self.base.offset,
            self.reduced_carrier @ z,
        )

    def observation_jacobian(self, x):
        z = x[: self.channels] + 1j * x[self.channels :]
        ka = np.einsum("iab,b->ia", self.reduced_kernels, z)
        return self.base.observe @ (2 * np.c_[ka.real, ka.imag])

    def residual(self, x):
        full = self.real_transform @ x
        z = x[: self.channels] + 1j * x[self.channels :]
        force = np.einsum("a,iab,b->i", z.conj(), self.reduced_kernels, z).real
        h = self.base.observe @ (force - self.base.required) + self.base.offset
        b = self.reduced_carrier @ z
        violation = np.maximum(abs(self.unpack(x)) / self.limit - 1, 0)
        return np.r_[
            self.base.fit_residual(h, b),
            self.base.effort * full / self.base.source_limit,
            self.bound_weight * violation,
        ]

    def jacobian(self, x):
        a = self.unpack(x)
        derivative = a.conj()[:, None] * self.transform / np.maximum(abs(a[:, None]), 1e-30)
        derivative = np.c_[derivative.real, -derivative.imag]
        active = abs(a) > self.limit
        z = x[: self.channels] + 1j * x[self.channels :]
        ka = np.einsum("iab,b->ia", self.reduced_kernels, z)
        shape_jac = self.base.observe @ (2 * np.c_[ka.real, ka.imag])
        force = np.einsum("a,ia->i", z.conj(), ka).real
        h = self.base.observe @ (force - self.base.required) + self.base.offset
        b = self.reduced_carrier @ z
        return np.vstack(
            [
                self.base.fit_jacobian(h, b, shape_jac, self.reduced_carrier),
                self.base.effort * self.real_transform / self.base.source_limit,
                self.bound_weight / self.limit * active[:, None] * derivative,
            ]
        )

    def source_slack(self, x):
        return 1 - abs(self.unpack(x)) ** 2 / self.limit**2

    def source_slack_jacobian(self, x):
        derivative = self.unpack(x).conj()[:, None] * self.transform
        return -2 * np.c_[derivative.real, -derivative.imag] / self.limit**2


def constrained_optimize(working, seed, iterations, objective_scale=1.0):
    """Exact amplitude-disk constraints avoid large penalty-induced stiffness."""
    if not np.isfinite(objective_scale) or objective_scale <= 0:
        raise ValueError("Objective scale must be finite and positive")
    # Least-squares projection into retained source coordinates need not preserve
    # the physical amplitude disks. Start feasible without changing that space.
    seed_peak = np.max(abs(working.unpack(seed)))
    seed = seed * min(1.0, 0.99 * working.limit / max(seed_peak, 1e-30))

    def value_and_gradient(x):
        residual = working.residual(x)[: -working.base.channels]
        derivative = working.jacobian(x)[: -working.base.channels]
        return (
            objective_scale * 0.5 * np.dot(residual, residual),
            objective_scale * derivative.T @ residual,
        )

    result = minimize(
        value_and_gradient,
        seed,
        jac=True,
        method="SLSQP",
        constraints={
            "type": "ineq",
            "fun": working.source_slack,
            "jac": working.source_slack_jacobian,
        },
        options={"maxiter": iterations, "ftol": 1e-12},
    )
    # Report objective and KKT residual in the original declared units.
    result.fun /= objective_scale
    result.jac /= objective_scale
    if getattr(result, "multipliers", None) is not None:
        result.multipliers /= objective_scale
    result.cost = float(result.fun)
    result.objective_scale = objective_scale
    result.initial_projected_source_peak_m_s = float(seed_peak)
    gradient = result.jac
    if getattr(result, "multipliers", None) is not None:
        gradient = gradient - working.source_slack_jacobian(result.x).T @ result.multipliers
    result.optimality = float(np.linalg.norm(gradient, ord=np.inf))
    return result
