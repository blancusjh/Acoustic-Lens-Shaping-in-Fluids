"""Carrier-aware frozen-target inverse screening; never a coupled accuracy pass.

Run: python -m acoustic_freeform.inverse.screening CONFIG --out NEW_DIRECTORY
Reuse a declared frozen operator using --cache-from PREVIOUS_DIRECTORY.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import least_squares, minimize

from ..acoustics.helmholtz import DualAcoustics
from ..apparatus.config import DualConfig
from ..apparatus.sources import export_sources
from ..core.paths import data_path
from ..core.provenance import capture_execution
from ..mechanics.surface import CartesianPatch, DualSurface
from ..optics.raytrace import fixed_detector_spot_jacobian
from ..verify.analytic import verify_capillary, verify_slab
from ..verify.metrics import maximum_error


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


def build_cache(inputs, output):
    cfg = DualConfig(**inputs["apparatus"])
    space = DualSurface(cfg)
    targets = [CartesianPatch(cfg, j, **f) for j, f in enumerate(inputs["case"]["faces"])]
    q = space.project(targets)
    validation = {
        "three_layer_slab": verify_slab(cfg, levels=(8, 16, 32)),
        "bessel_compliance": verify_capillary(cfg),
        "target_projection": [maximum_error(space, v, t) for v, t in zip(q, targets)],
        "scope": "Analytic limits and geometry projection, not coupled accuracy.",
    }
    print(f"Building frozen operator for {cfg.channels} shared sources", flush=True)
    response = DualAcoustics(cfg, space).solve(q, trace_only=True)
    required, stiffness, _ = space.mechanics(q)
    r = np.unique(
        np.r_[
            np.linspace(0, cfg.clear_radius_m, 121),
            np.linspace(cfg.clear_radius_m, cfg.radius_m, 61),
        ]
    )
    weights = np.where(r <= cfg.clear_radius_m, 1.0, inputs.get("annulus_weight", 0.2))
    observe = block_diag(space.basis(r), space.basis(r)) @ np.linalg.inv(stiffness)
    matrices = []
    face_ids = []
    for j in range(2):
        mask = response.radial_m[j] <= cfg.clear_radius_m
        matrices.append(
            response.velocity[j][mask]
            / (2 * np.pi * cfg.frequency_hz * abs(response.normals_z[j][mask, None]))
        )
        face_ids.extend([j] * np.count_nonzero(mask))
    cache = {
        "force_kernels": response.force_kernels,
        "required_force_n": required,
        "compliance_observation_m_n": observe,
        "observation_weights": np.tile(weights, 2),
        "observation_radii_m": np.tile(r, 2),
        "observation_faces": np.repeat([0, 1], len(r)),
        "carrier_height_response_s": np.vstack(matrices),
        "carrier_faces": np.array(face_ids),
        "target_coefficients_m": q,
    }
    np.savez_compressed(output / "frozen-operator.npz", **cache)
    validation["wave_linear_residual"] = response.residual
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    return cache


def run(inputs_path, output, cache_from=None):
    inputs = json.loads(inputs_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    capture_execution(output)
    (output / "config.json").write_text(json.dumps(inputs, indent=2) + "\n")
    cfg = DualConfig(**inputs["apparatus"])
    cfg.validate()
    if (
        inputs.get("pupil_slope_scale_m") is not None
        or inputs.get("joint_spot_tolerance_m") is not None
    ) and inputs.get("observation_sampling") != "spline_quadrature":
        raise ValueError("Slope proxy requires dense observations")
    export_sources(output, cfg)
    (output / "resolved-config.json").write_text(json.dumps(cfg.as_dict(), indent=2) + "\n")
    if cache_from is None:
        cache = build_cache(inputs, output)
    else:
        prior = json.loads((cache_from / "config.json").read_text())
        prior_cfg = json.loads((cache_from / "resolved-config.json").read_text())
        if (
            prior_cfg != json.loads(json.dumps(cfg.as_dict()))
            or prior["case"]["faces"] != inputs["case"]["faces"]
        ):
            raise ValueError("Cached geometry/apparatus differs from the requested problem.")
        if (
            prior.get("annulus_weight", 0.2) != inputs.get("annulus_weight", 0.2)
            and inputs.get("observation_sampling") != "spline_quadrature"
        ):
            raise ValueError("Cached observation weights differ.")
        source = cache_from / "frozen-operator.npz"
        shutil.copy2(source, output / source.name)
        shutil.copy2(cache_from / "validation.json", output / "validation.json")
        (output / "cache-provenance.json").write_text(
            json.dumps(
                {
                    "source": str(cache_from.resolve()),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "scope": "Reused frozen field operator; original execution provenance at source.",
                },
                indent=2,
            )
            + "\n"
        )
        cache = dict(np.load(source))
    initial_file = data_path(inputs["initial_drive_file"])
    if inputs.get("observation_sampling") == "spline_quadrature":
        space = DualSurface(cfg)
        required, stiffness, _ = space.mechanics(cache["target_coefficients_m"])
        np.testing.assert_allclose(required, cache["required_force_n"], rtol=1e-10, atol=1e-20)
        radii = np.unique(np.r_[0, space.r, cfg.clear_radius_m, cfg.radius_m])
        cache["compliance_observation_m_n"] = block_diag(
            space.basis(radii), space.basis(radii)
        ) @ np.linalg.inv(stiffness)
        cache["observation_radii_m"] = np.tile(radii, 2)
        cache["observation_faces"] = np.repeat([0, 1], len(radii))
        cache["observation_weights"] = np.tile(
            np.where(radii <= cfg.clear_radius_m, 1, inputs.get("annulus_weight", 0.2)), 2
        )
        if inputs.get("pupil_slope_scale_m") is not None:
            add_pupil_slope_observations(cache, space, stiffness, inputs["pupil_slope_scale_m"])
        if inputs.get("joint_spot_tolerance_m") is not None:
            add_joint_spot_observations(
                cache,
                space,
                stiffness,
                inputs["case"]["faces"],
                inputs["joint_spot_tolerance_m"],
                inputs.get("optical_observation_rays", 501),
            )
        np.savez_compressed(
            output / "observation-operator.npz",
            **{
                k: v
                for k, v in cache.items()
                if k.startswith("observation_") or k == "compliance_observation_m_n"
            },
        )
    if inputs.get("observation_only", False):
        if inputs.get("observation_sampling") != "spline_quadrature":
            raise ValueError("Observation-only mode requires dense observations")
        (output / "observation-validation.json").write_text(
            json.dumps(
                {
                    "observations": len(cache["observation_radii_m"]),
                    "joint_optical_observations": int(
                        np.sum(cache.get("observation_kind", []) == 2)
                    ),
                    "source_optimization_executed": False,
                    "coupled_equilibrium_executed": False,
                    "scope": "Local observation operator at the declared target only",
                },
                indent=2,
            )
            + "\n"
        )
        (output / "report.md").write_text(
            "# Target observation operator\n\nHeight and optional slope/optical derivatives. "
            "No source optimization, coupled equilibrium or physical trajectory executed.\n"
        )
        return
    shutil.copy2(initial_file, output / "initial-source.npz")
    initial = np.load(initial_file)["source_velocity_m_s"]
    initial = np.repeat(initial, inputs.get("initial_source_refinement_factor", 1))
    if len(initial) != cfg.channels:
        raise ValueError("Initial source vector does not match source supports.")
    limit = cfg.max_source_speed_m_s / np.sqrt(2)
    x0 = np.clip(np.r_[initial.real, initial.imag], -limit, limit)
    records = []
    for weight in inputs["carrier_weights"]:
        objective = PrecisionObjective(
            cache,
            weight,
            limit,
            inputs.get("effort_weight", 1e-5),
            peak_power=inputs.get("peak_power"),
            clear_radius_m=cfg.clear_radius_m,
            include_carrier_in_peak=inputs.get("include_carrier_in_peak", True),
        )
        working, seed, bounds = objective, x0, (-limit, limit)
        if inputs.get("coordinates") == "radiation_whitened":
            working = WhitenedObjective(
                objective,
                inputs.get("eigenvalue_cutoff", 1e-12),
                carrier_ceiling_m=inputs.get("carrier_subspace_ceiling_m"),
            )
            seed, bounds = working.encode(initial), (-np.inf, np.inf)
            print(f"Observable complex source coordinates: {working.channels}", flush=True)
        print(
            f"Carrier weight {weight}; initial residual {np.linalg.norm(working.residual(seed))}",
            flush=True,
        )
        if inputs.get("optimizer") == "slsqp":
            if not isinstance(working, WhitenedObjective):
                raise ValueError("SLSQP currently requires radiation_whitened coordinates.")
            result = constrained_optimize(
                working,
                seed,
                inputs.get("max_evaluations", 2000),
                objective_scale=inputs.get("constrained_objective_scale", 1.0),
            )
        else:
            result = least_squares(
                working.residual,
                seed,
                jac=working.jacobian,
                bounds=bounds,
                max_nfev=inputs.get("max_evaluations", 2000),
                ftol=1e-11,
                xtol=1e-11,
                gtol=1e-9,
                verbose=0,
            )
        source = working.unpack(result.x)
        scale = min(1.0, cfg.max_source_speed_m_s / max(np.max(abs(source)), 1e-30))
        source *= scale
        h, b = objective.metrics(np.r_[source.real, source.imag])
        audit_space = DualSurface(cfg)
        _, audit_stiffness, _ = audit_space.mechanics(cache["target_coefficients_m"])
        force_error = (
            np.einsum("a,iab,b->i", source.conj(), cache["force_kernels"], source).real
            - cache["required_force_n"]
        )
        perturbation = np.linalg.solve(audit_stiffness, force_error).reshape(2, -1)
        faces = cache["observation_faces"]
        pupil = cache["observation_radii_m"] <= cfg.clear_radius_m
        pupil &= cache.get("observation_kind", np.zeros(len(pupil), dtype=int)) == 0
        record = {
            "carrier_weight": weight,
            "peak_power": inputs.get("peak_power"),
            "include_carrier_in_peak": inputs.get("include_carrier_in_peak", True),
            "declared_accuracy_target": inputs.get("accuracy_target", "instantaneous"),
            "observation_sampling": inputs.get("observation_sampling", "legacy_uniform"),
            "pupil_slope_scale_m": inputs.get("pupil_slope_scale_m"),
            "joint_spot_tolerance_m": inputs.get("joint_spot_tolerance_m"),
            "frozen_linearized_spot_radius_m": (
                float(
                    np.max(abs(h[cache["observation_kind"] == 2]))
                    * 1e-8
                    / cache["observation_optical_scale"]
                )
                if "observation_optical_scale" in cache
                else None
            ),
            "frozen_sampled_pupil_slope_error_rad": [
                float(
                    np.max(
                        abs(audit_space.evaluate(q, np.linspace(0, cfg.clear_radius_m, 4001), 1))
                    )
                )
                for q in perturbation
            ],
            "frozen_polynomial_pupil_audit": [
                audit_space.polynomial_maximum(q, upper_m=cfg.clear_radius_m) for q in perturbation
            ],
            "frozen_polynomial_annulus_audit": [
                audit_space.polynomial_maximum(q, lower_m=cfg.clear_radius_m) for q in perturbation
            ],
            "optimizer_success": bool(result.success),
            "first_order_stationarity_test": bool(result.optimality <= 1e-6),
            "message": result.message,
            "evaluations": result.nfev,
            "cost": float(result.cost),
            "source_coordinates": inputs.get("coordinates", "physical_components"),
            "optimizer": inputs.get("optimizer", "least_squares"),
            "constrained_objective_scale": getattr(result, "objective_scale", None),
            "initial_projected_source_peak_m_s": getattr(
                result, "initial_projected_source_peak_m_s", None
            ),
            "carrier_subspace_ceiling_m": inputs.get("carrier_subspace_ceiling_m"),
            "carrier_subspace_operator_bound_m": getattr(working, "carrier_subspace_bound_m", None),
            "amplitude_feasibility_rescaling": float(scale),
            "optimality": float(result.optimality),
            "frozen_compliance_pupil_max_m": [
                float(np.max(abs(h[pupil & (faces == j)])) * 1e-8) for j in range(2)
            ],
            "frozen_carrier_pupil_max_m": [
                float(np.max(abs(b[cache["carrier_faces"] == j])) * 1e-8) for j in range(2)
            ],
            "source_component_peak_m_s": float(np.max(abs(source))),
            "sampled_conservative_combined_proxy_m": [
                float(
                    (
                        np.max(abs(h[pupil & (faces == j)]))
                        + np.max(abs(b[cache["carrier_faces"] == j]))
                    )
                    * 1e-8
                )
                for j in range(2)
            ],
            "coupled_accuracy_established": False,
        }
        name = f"weight-{weight:g}"
        np.savez_compressed(
            output / f"{name}.npz",
            source_velocity_m_s=source,
            target_coefficients_m=cache["target_coefficients_m"],
        )
        records.append(record)
        (output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
        print(record, flush=True)
    lines = [
        "# Carrier-aware frozen-target screening",
        "",
        "Not a coupled forward result, formation trajectory, or 10 nm certification.",
        "",
        "| Carrier weight | Linearized front/back (nm) | Carrier front/back (nm) | Solver termination flag |",
        "|---:|---:|---:|---|",
    ]
    for r in records:
        h, b = [
            np.array(r[k]) * 1e9
            for k in ("frozen_compliance_pupil_max_m", "frozen_carrier_pupil_max_m")
        ]
        lines.append(
            f"| {r['carrier_weight']} | {h[0]:.3f} / {h[1]:.3f} | {b[0]:.3f} / {b[1]:.3f} | {r['optimizer_success']} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-from", type=Path)
    args = parser.parse_args()
    run(args.config, args.out, args.cache_from)
