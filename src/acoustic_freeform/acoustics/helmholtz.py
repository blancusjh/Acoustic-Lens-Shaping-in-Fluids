"""Shared three-layer axisymmetric Helmholtz field on exact spline graphs.

Inviscid lossless fluids, ideal pressure/normal-velocity transmission, passive
source ports V.n=Y*P+g, peak phasors exp(-i*omega*t). The mesh coordinates are
scaled by R internally; pressures, source velocities, forces and saved data
are SI. Both traces of normal velocity are retained as a discretization check.
"""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import splu
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP2,
    ElementTriP3,
    ElementTriP4,
    FacetBasis,
    InteriorFacetBasis,
    LinearForm,
    MeshTri,
    asm,
)
from skfem.helpers import dot, grad
from skfem.mapping import MappingAffine

from .elements import ElementTriLagrange
from .linear import CondensedFactor, ReusedCondensedFactor


def weighted_source_grams(weights, shape_basis, values):
    """Exact quadrature contraction via matrix products, without a 4-D tensor.

    Generalized source kernel i is V† diag(w B_i) V. Explicit matrix products
    avoid NumPy's slow three-operand complex contraction for large source sets.
    No quadrature, source, or surface modes are dropped.
    """
    weights = np.asarray(weights)
    shape_basis, values = np.asarray(shape_basis), np.asarray(values)
    if (
        weights.ndim != 1
        or shape_basis.ndim != 2
        or values.ndim != 2
        or shape_basis.shape[0] != len(weights)
        or values.shape[0] != len(weights)
    ):
        raise ValueError("Quadrature weights, basis and source traces must share sample rows")
    if values.shape[1] == 1:
        return (shape_basis.T @ (weights * abs(values[:, 0]) ** 2))[:, None, None].astype(complex)
    result = np.empty((shape_basis.shape[1], values.shape[1], values.shape[1]), complex)
    adjoint = np.ascontiguousarray(values.conj().T)
    for i in range(shape_basis.shape[1]):
        result[i] = adjoint @ ((weights * shape_basis[:, i])[:, None] * values)
    return result


class LayerMapping(MappingAffine):
    def __init__(self, flat, surface, coefficients):
        super().__init__(flat)
        self.surface, self.q = surface, np.array(coefficients).copy()
        self.cfg = surface.config
        self.levels = np.asarray(self.cfg.levels_m) / self.cfg.radius_m
        self.regions = np.searchsorted(self.levels[1:-1], flat.p[1, flat.t].mean(axis=0))
        self.cache = {}

    def geometry(self, points, regions=None):
        r, z = points
        if regions is None:
            regions = np.searchsorted(self.levels[1:-1], z)
        elif np.ndim(regions) == 1:
            regions = regions[:, None]
        lower, upper = self.levels[regions], self.levels[regions + 1]
        t = (z - lower) / (upper - lower)
        h = (
            [np.zeros_like(r)]
            + [self.surface.evaluate(q, r * self.cfg.radius_m) / self.cfg.radius_m for q in self.q]
            + [np.zeros_like(r)]
        )
        hp = (
            [np.zeros_like(r)]
            + [self.surface.evaluate(q, r * self.cfg.radius_m, 1) for q in self.q]
            + [np.zeros_like(r)]
        )
        lo = np.choose(regions, h[:-1])
        hi = np.choose(regions, h[1:])
        shear = (1 - t) * np.choose(regions, hp[:-1]) + t * np.choose(regions, hp[1:])
        stretch = 1 + (hi - lo) / (upper - lower)
        if np.any(stretch <= 0):
            raise ValueError("A fluid layer has nonpositive thickness.")
        mapped = points.copy()
        mapped[1] += (1 - t) * lo + t * hi
        return mapped, stretch, shear

    def metrics(self, X, tind=None):
        key = (X.shape, X.tobytes(), None if tind is None else np.asarray(tind).tobytes())
        if key not in self.cache:
            if len(self.cache) > 20:
                self.cache.clear()
            regions = self.regions if tind is None else self.regions[tind]
            self.cache[key] = self.geometry(super().F(X, tind), regions)
        return self.cache[key]

    def F(self, X, tind=None):
        return self.metrics(X, tind)[0]

    def invF(self, points, tind=None):
        r, z = points
        if tind is None:
            moved = [np.full_like(r, self.levels[0])]
            moved += [
                self.levels[j + 1]
                + self.surface.evaluate(q, r * self.cfg.radius_m) / self.cfg.radius_m
                for j, q in enumerate(self.q)
            ]
            regions = (z >= moved[1]).astype(int) + (z >= moved[2]).astype(int)
        else:
            regions = self.regions[tind][:, None]
        lower, upper = self.levels[regions], self.levels[regions + 1]
        h = (
            [np.zeros_like(r)]
            + [self.surface.evaluate(q, r * self.cfg.radius_m) / self.cfg.radius_m for q in self.q]
            + [np.zeros_like(r)]
        )
        lo, hi = np.choose(regions, h[:-1]), np.choose(regions, h[1:])
        unwarped = points.copy()
        unwarped[1] = lower + (z - lower - lo) / (1 + (hi - lo) / (upper - lower))
        return super().invF(unwarped, tind)

    def DF(self, X, tind=None):
        _, stretch, shear = self.metrics(X, tind)
        base = super().DF(X, tind)
        result = base.copy()
        result[1] = shear[None] * base[0] + stretch[None] * base[1]
        return result

    def detDF(self, X, tind=None):
        return super().detDF(X, tind) * self.metrics(X, tind)[1]

    def invDF(self, X, tind=None):
        _, stretch, shear = self.metrics(X, tind)
        base = super().invDF(X, tind)
        result = base.copy()
        result[:, 0] -= base[:, 1] * shear[None] / stretch[None]
        result[:, 1] /= stretch[None]
        return result

    def G(self, X, find=None):
        return self.geometry(super().G(X, find))[0]

    def detDG(self, X, find=None):
        _, stretch, shear = self.geometry(super().G(X, find))
        tangent = self.B[:, 0] if find is None else self.B[:, 0, find]
        dr = np.broadcast_to(tangent[0, :, None], stretch.shape)
        dz = shear * dr + stretch * tangent[1, :, None]
        return np.sqrt(dr**2 + dz**2)


@dataclass
class WaveResponse:
    force_kernels: np.ndarray
    pressure: list
    velocity: list
    tangent_gradient: list
    radial_m: list
    weights_m2: list
    normals_z: list
    flux_jump: list
    solution: np.ndarray
    basis: Basis  # DOF/mapping metadata; volume integrals use volume_integration_order.
    boundary_matrix: object
    source_rhs: np.ndarray
    residual: float
    config: object
    applied_drive: object = None
    linear_solver: str = "full"
    factorized_dofs: int | None = None
    krylov_iterations: int = 0
    preconditioner_refreshes: int = 0
    volume_assembly_chunk_size: int | None = None
    volume_integration_order: int | None = None

    def force(self, drive):
        return np.einsum("a,iab,b->i", drive.conj(), self.force_kernels, drive).real

    def force_jacobian(self, drive):
        ka = np.einsum("iab,b->ia", self.force_kernels, drive)
        return 2 * np.c_[ka.real, ka.imag]

    def diagnostics(self, drive):
        if self.solution.shape[0] == 0:
            raise ValueError(
                "Trace-only response: use a fresh fixed-command solve for diagnostics."
            )
        omega = 2 * np.pi * self.config.frequency_hz
        p = self.solution @ drive
        # Wave assembly cancels the common 2*pi; physical watts restore it.
        absorption = np.pi * np.vdot(p, self.boundary_matrix @ p).real
        source = -np.pi * np.real(np.vdot(p, self.source_rhs @ drive / (1j * omega)))
        carrier, flux = [], []
        for r, vel, nz, jump in zip(self.radial_m, self.velocity, self.normals_z, self.flux_jump):
            mask = r <= self.config.clear_radius_m
            carrier.append(float(np.max(np.abs(vel[mask] @ drive) / (omega * abs(nz[mask])))))
            flux.append(float(np.max(np.abs(jump @ drive))))
        return {
            "sampled_pressure_peak_pa": float(np.max(abs(p))),
            "boundary_absorption_w": float(absorption),
            "source_power_w": float(source),
            "power_relative_error": float(abs(source - absorption) / max(abs(source), 1e-30)),
            "pupil_sampled_carrier_height_peak_m": carrier,
            "normal_velocity_jump_peak_m_s": flux,
            "wave_linear_residual": self.residual,
            "source_component_peak_m_s": float(
                np.max(abs(drive if self.applied_drive is None else self.applied_drive))
            ),
            "pressure_dofs": int(self.basis.N),
            "wave_linear_solver": self.linear_solver,
            "factorized_pressure_dofs": int(
                self.basis.N if self.factorized_dofs is None else self.factorized_dofs
            ),
            "wave_krylov_iterations": self.krylov_iterations,
            "preconditioner_refreshes": self.preconditioner_refreshes,
            "volume_assembly_chunk_size": self.volume_assembly_chunk_size,
            "volume_integration_order": self.volume_integration_order,
        }


class DualAcoustics:
    def __init__(
        self,
        config,
        surface,
        linear_solver="full",
        local_flux_assembly=True,
        volume_assembly_chunk_size=4096,
    ):
        if linear_solver not in ("full", "static_condensed", "reused_condensed"):
            raise ValueError("Unknown wave linear solver.")
        self.linear_solver = linear_solver
        self._reused_factor = None
        self.local_flux_assembly = local_flux_assembly
        if volume_assembly_chunk_size is not None and volume_assembly_chunk_size < 1:
            raise ValueError("Volume assembly chunk size must be positive or None")
        self.volume_assembly_chunk_size = volume_assembly_chunk_size
        self.config, self.surface = config, surface
        config.validate()
        r = np.unique(
            np.r_[
                np.linspace(0, 1, config.radial_cells + 1),
                np.linspace(0, 1, config.radial_ports + 1),
            ]
        )
        levels = np.asarray(config.levels_m) / config.radius_m
        z = np.unique(
            np.concatenate(
                [
                    np.r_[
                        np.linspace(lo, hi, config.cells_per_layer + 1),
                        np.linspace(lo, hi, config.side_ports_per_layer + 1),
                    ]
                    for lo, hi in pairwise(levels)
                ]
            )
        )
        self.flat = MeshTri.init_tensor(r, z)
        self.element = (
            {2: ElementTriP2, 3: ElementTriP3, 4: ElementTriP4}[config.acoustic_order]()
            if config.acoustic_order <= 4
            else ElementTriLagrange(config.acoustic_order)
        )
        self.order = 2 * config.acoustic_order + 3
        self.regions = np.searchsorted(levels[1:-1], self.flat.p[1, self.flat.t].mean(axis=0))
        mid = self.flat.p[:, self.flat.facets].mean(axis=1)
        external = self.flat.boundary_facets()
        self.boundaries = {
            "bottom": external[np.isclose(mid[1, external], levels[0])],
            "top": external[np.isclose(mid[1, external], levels[-1])],
            "side": external[np.isclose(mid[0, external], 1)],
        }
        self.interfaces = [np.flatnonzero(np.isclose(mid[1], level)) for level in levels[1:-1]]

    def solve(
        self,
        coefficients,
        drive=None,
        trace_only=False,
        source_block_size=16,
        retain_linearization=False,
    ):
        cfg, space = self.config, self.surface
        if retain_linearization and (
            drive is None or trace_only or cfg.normal_velocity_trace != "weak_flux"
        ):
            raise ValueError("Shape linearization requires one driven weak-flux wave solve")
        if drive is not None:
            drive = np.asarray(drive, complex)
            if drive.shape != (cfg.channels,) or not np.all(np.isfinite(drive)):
                raise ValueError("A finite complex command is required for every source region.")
        radius, omega = cfg.radius_m, 2 * np.pi * cfg.frequency_hz
        rho, speed = np.asarray(cfg.density_kg_m3), np.asarray(cfg.sound_speed_m_s)
        mapping = LayerMapping(self.flat, space, coefficients)
        # Keep global DOF/mapping metadata without retaining every high-order
        # volume quadrature field during sparse factorization. All physical
        # integrals below still use self.order, explicitly, on element batches.
        basis = Basis(
            self.flat,
            self.element,
            mapping=mapping,
            intorder=self.order if self.volume_assembly_chunk_size is None else 1,
        )
        regions = self.regions[:, None]

        @BilinearForm(dtype=complex)
        def wave(u, v, w):
            return (
                w.x[0]
                * radius
                * w.phase_mask
                / w.density
                * (dot(grad(u), grad(v)) - w.wavenumber_squared * u * v)
            )

        @BilinearForm
        def absorption(u, v, w):
            return w.x[0] * radius**2 * w.admittance * u * v

        @LinearForm(dtype=complex)
        def source(v, w):
            return 1j * omega * radius**2 * w.x[0] * w.mask * v

        def assemble_wave(selected_basis, elements, mask):
            material = self.regions[elements, None]
            return asm(
                wave,
                selected_basis,
                phase_mask=mask,
                density=rho[material],
                wavenumber_squared=(omega * radius / speed[material]) ** 2,
            ).tocsc()

        if self.volume_assembly_chunk_size is None:
            matrix = assemble_wave(basis, np.arange(len(self.regions)), np.ones_like(regions))
        else:
            matrix = csr_matrix((basis.N, basis.N), dtype=complex)
            for start in range(0, len(self.regions), self.volume_assembly_chunk_size):
                elements = np.arange(
                    start, min(start + self.volume_assembly_chunk_size, len(self.regions))
                )
                selected_basis = Basis(
                    self.flat,
                    self.element,
                    mapping=mapping,
                    dofs=basis.dofs,
                    elements=elements,
                    intorder=self.order,
                )
                matrix += assemble_wave(selected_basis, elements, np.ones((len(elements), 1)))
                mapping.cache.clear()
            del selected_basis
            matrix = matrix.tocsc()
        boundary_matrix = csr_matrix(matrix.shape, dtype=float)
        partial_boundary = [csr_matrix(matrix.shape, dtype=float) for _ in range(2)]
        loads = []

        def append_load(value):
            loads.append(csr_matrix(value[:, None]))

        source_phases = []
        levels = np.asarray(cfg.levels_m) / radius
        for kind in ("bottom", "top", "side"):
            fb = FacetBasis(
                self.flat,
                self.element,
                mapping=mapping,
                facets=self.boundaries[kind],
                intorder=self.order,
            )
            xx = fb.global_coordinates()
            phase = np.searchsorted(levels[1:-1], xx[1])
            factor = cfg.side_admittance_factor if kind == "side" else cfg.port_admittance_factor
            admittance = factor / (rho[phase] * speed[phase])
            boundary_matrix += asm(absorption, fb, admittance=admittance)
            if cfg.normal_velocity_trace == "weak_flux":
                for interface in range(2):
                    partial_boundary[interface] += asm(
                        absorption, fb, admittance=admittance * (phase <= interface)
                    )
            if kind in ("bottom", "top"):
                indices = np.minimum((xx[0] * cfg.radial_ports).astype(int), cfg.radial_ports - 1)
                if drive is not None:
                    offset = 0 if kind == "bottom" else cfg.radial_ports
                    append_load(asm(source, fb, mask=drive[offset + indices]))
                    source_phases.append(0 if kind == "bottom" else 2)
                else:
                    for j in range(cfg.radial_ports):
                        append_load(asm(source, fb, mask=(indices == j)))
                        source_phases.append(0 if kind == "bottom" else 2)
            else:
                for layer, (lo, hi) in enumerate(pairwise(levels)):
                    indices = np.minimum(
                        ((xx[1] - lo) / (hi - lo) * cfg.side_ports_per_layer).astype(int),
                        cfg.side_ports_per_layer - 1,
                    )
                    if drive is not None:
                        offset = 2 * cfg.radial_ports + layer * cfg.side_ports_per_layer
                        selected = np.clip(indices, 0, cfg.side_ports_per_layer - 1)
                        append_load(
                            asm(source, fb, mask=(phase == layer) * drive[offset + selected])
                        )
                        source_phases.append(layer)
                    else:
                        for j in range(cfg.side_ports_per_layer):
                            append_load(asm(source, fb, mask=((phase == layer) & (indices == j))))
                            source_phases.append(layer)
        matrix -= 1j * omega * boundary_matrix

        def flux_rows(interface):
            ids = basis.get_dofs(facets=self.interfaces[interface]).all()
            if self.local_flux_assembly:
                touching = np.any(np.isin(basis.element_dofs, ids), axis=0)
                elements = np.flatnonzero(touching & (self.regions <= interface))
            else:
                elements = np.arange(len(self.regions))
            selected_basis = Basis(
                self.flat,
                self.element,
                mapping=mapping,
                dofs=basis.dofs,
                elements=elements,
                intorder=self.order,
            )
            partial = assemble_wave(
                selected_basis, elements, self.regions[elements, None] <= interface
            )
            return partial[ids] - 1j * omega * partial_boundary[interface][ids]

        rhs = hstack(loads, format="csr")
        loads.clear()
        if self.linear_solver == "reused_condensed":
            if self._reused_factor is None:
                self._reused_factor = ReusedCondensedFactor(matrix, basis.dofs.interior_dofs)
            else:
                self._reused_factor.update(matrix)
            lu = self._reused_factor
        elif self.linear_solver == "static_condensed":
            lu = CondensedFactor(matrix, basis.dofs.interior_dofs)
        else:
            lu = splu(matrix.tocsc(), permc_spec="MMD_AT_PLUS_A")
        if trace_only:
            if drive is not None or source_block_size < 1:
                raise ValueError(
                    "Trace-only source blocks require a basis solve and positive block size."
                )
            partials = []
            for j in range(2):
                partials.append(flux_rows(j))
            return self._block_traces(
                coefficients,
                matrix,
                lu,
                rhs,
                basis,
                mapping,
                partials,
                np.asarray(source_phases),
                boundary_matrix,
                source_block_size,
            )
        selected_rhs = rhs.toarray() if drive is None else np.asarray(rhs.sum(axis=1))
        solution = lu.solve(selected_rhs)
        residual = np.linalg.norm(matrix @ solution - selected_rhs) / max(
            np.linalg.norm(selected_rhs), 1e-30
        )
        kernels, pressures, velocities, tangents, radii, weights, normal_z, jumps = (
            [] for _ in range(8)
        )
        linearization_faces = []
        for j, facets in enumerate(self.interfaces):
            traces = [
                InteriorFacetBasis(
                    self.flat,
                    self.element,
                    mapping=mapping,
                    facets=facets,
                    side=side,
                    intorder=self.order,
                )
                for side in (0, 1)
            ]
            p_values, grad_values, phase_values = [], [], []
            coords = traces[0].global_coordinates()
            rr = coords[0].ravel() * radius
            slope = space.evaluate(coefficients[j], rr, 1)
            nz = 1 / np.sqrt(1 + slope**2)
            normal = np.stack([-slope * nz, nz], axis=1)
            tangent = np.stack([nz, slope * nz], axis=1)
            for trace in traces:
                fields = [trace.interpolate(column) for column in solution.T]
                p_values.append(np.stack([np.asarray(field).ravel() for field in fields], axis=-1))
                grad_values.append(
                    np.stack([field.grad.reshape(2, -1).T / radius for field in fields], axis=-1)
                )
                phase_values.append(np.repeat(self.regions[trace.tind], trace.dx.shape[1]))
            p = (p_values[0] + p_values[1]) / 2
            # Use both density-weighted derivative traces; monitor their mismatch.
            vtrace = [
                np.einsum("sd,sdc->sc", normal, dg) / (1j * omega * rho[phase, None])
                for dg, phase in zip(grad_values, phase_values)
            ]
            vn = (vtrace[0] + vtrace[1]) / 2
            if cfg.normal_velocity_trace == "weak_flux":
                partial = flux_rows(j)
                interface_dofs = basis.get_dofs(facets=facets).all()
                sub_rhs = rhs[interface_dofs].toarray() * (np.asarray(source_phases)[None, :] <= j)
                if drive is not None:
                    sub_rhs = sub_rhs.sum(axis=1, keepdims=True)
                flux = (partial @ solution - sub_rhs) / (1j * omega)

                @BilinearForm
                def trace_mass(u, v, w):
                    return radius**2 * w.x[0] * u * v

                mass = asm(trace_mass, traces[0]).tocsc()[interface_dofs][:, interface_dofs]
                recovered = np.zeros_like(solution)
                recovered[interface_dofs] = solve(mass.toarray(), flux, assume_a="pos")
                vn = np.stack(
                    [np.asarray(traces[0].interpolate(column)).ravel() for column in recovered.T],
                    axis=-1,
                )
                if retain_linearization:
                    linearization_faces.append(
                        {
                            "trace": traces[0],
                            "ids": interface_dofs,
                            "mass": mass.toarray(),
                            "prefix": partial,
                            "normal_velocity_coefficients": recovered[:, 0].copy(),
                            "slope": slope,
                            "tangent": tangent,
                        }
                    )
            gt = np.einsum("sd,sdc->sc", tangent, (grad_values[0] + grad_values[1]) / 2)
            cp = 0.25 * (1 / (rho[j] * speed[j] ** 2) - 1 / (rho[j + 1] * speed[j + 1] ** 2))
            cv = (rho[j] - rho[j + 1]) / 4
            ct = cv / (omega**2 * rho[j] * rho[j + 1])
            w = traces[0].dx.ravel() * nz * rr * radius * 2 * np.pi
            b = space.basis(rr)
            kk = sum(
                factor * weighted_source_grams(w, b, values)
                for factor, values in ((cp, p), (cv, vn), (ct, gt))
            )
            kernels.append(kk)
            pressures.append(p)
            velocities.append(vn)
            tangents.append(gt)
            radii.append(rr)
            weights.append(w)
            normal_z.append(nz)
            jumps.append(vtrace[0] - vtrace[1])
        if retain_linearization:
            self._linearization = {
                "basis": basis,
                "mapping": mapping,
                "factor": lu,
                "faces": linearization_faces,
                "pressure": solution[:, 0],
            }
        return WaveResponse(
            np.concatenate(kernels),
            pressures,
            velocities,
            tangents,
            radii,
            weights,
            normal_z,
            jumps,
            solution,
            basis,
            boundary_matrix,
            csr_matrix(selected_rhs),
            float(residual),
            cfg,
            None if drive is None else np.array(drive).copy(),
            linear_solver=self.linear_solver,
            factorized_dofs=len(lu.boundary) if hasattr(lu, "boundary") else basis.N,
            krylov_iterations=getattr(lu, "iterations", 0),
            preconditioner_refreshes=getattr(lu, "refreshes", 0),
            volume_assembly_chunk_size=self.volume_assembly_chunk_size,
            volume_integration_order=self.order,
        )

    def shape_force_jacobian(
        self, coefficients, drive, directions=None, block_size=8, progress=None
    ):
        """Analytic discrete acoustic shape derivative at a fixed command.

        Differentiates the pulled-back volume form, the weak normal flux mass
        matrix, and both pressure/gradient traces. No finite differences or
        reduced mechanical feedback space are used. Columns correspond to
        physical coefficient directions; the full matrix has units N/m.
        This static derivative is not a hydrodynamic stability calculation.
        """
        if block_size < 1:
            raise ValueError("Positive tangent block size required")
        cfg, space = self.config, self.surface
        response = self.solve(coefficients, drive, retain_linearization=True)
        context = self._linearization
        basis, mapping = context["basis"], context["mapping"]
        pressure, factor = context["pressure"], context["factor"]
        size = 2 * space.count
        directions = np.eye(size) if directions is None else np.asarray(directions, float)
        if directions.ndim != 2 or directions.shape[0] != size:
            raise ValueError("Direction columns must span the two coefficient vectors")
        radius, omega = cfg.radius_m, 2 * np.pi * cfg.frequency_hz
        rho, speed = np.array(cfg.density_kg_m3), np.array(cfg.sound_speed_m_s)
        levels = np.array(cfg.levels_m) / radius
        answer = np.empty((size, directions.shape[1]))

        @LinearForm(dtype=complex)
        def volume_derivative(v, w):
            pr, pz = w.pressure.grad
            vr, vz = grad(v)
            return (
                w.x[0]
                * radius
                / w.density
                * (
                    w.vertical * (pr * vr - pz * vz - w.k2 * w.pressure * v)
                    - w.horizontal * (pz * vr + pr * vz)
                )
            )

        @LinearForm(dtype=complex)
        def mass_derivative(v, w):
            return radius**2 * w.x[0] * w.relative_stretch * w.velocity * v

        for start in range(0, directions.shape[1], block_size):
            selected = directions[:, start : start + block_size]
            count = selected.shape[1]
            load = np.zeros((basis.N, count), complex)
            prefix_load = [np.zeros((len(f["ids"]), count), complex) for f in context["faces"]]
            # Phase-pure batches allow the same assembled directional vector
            # to supply both the global and prefix-domain derivative rows.
            for phase in range(3):
                phase_elements = np.flatnonzero(self.regions == phase)
                chunk = self.volume_assembly_chunk_size or 4096
                for offset in range(0, len(phase_elements), chunk):
                    elements = phase_elements[offset : offset + chunk]
                    local = Basis(
                        self.flat,
                        self.element,
                        mapping=mapping,
                        dofs=basis.dofs,
                        elements=elements,
                        intorder=self.order,
                    )
                    field = local.interpolate(pressure)
                    reference = MappingAffine.F(mapping, local.X, elements)
                    radial = reference[0] * radius
                    fraction = (reference[1] - levels[phase]) / (levels[phase + 1] - levels[phase])
                    _, stretch, shear = mapping.metrics(local.X, elements)
                    for column, direction in enumerate(selected.T):
                        delta = direction.reshape(2, space.count)
                        lo = (
                            space.evaluate(delta[phase - 1], radial)
                            if phase > 0
                            else np.zeros_like(radial)
                        )
                        hi = (
                            space.evaluate(delta[phase], radial)
                            if phase < 2
                            else np.zeros_like(radial)
                        )
                        lo_prime = (
                            space.evaluate(delta[phase - 1], radial, 1)
                            if phase > 0
                            else np.zeros_like(radial)
                        )
                        hi_prime = (
                            space.evaluate(delta[phase], radial, 1)
                            if phase < 2
                            else np.zeros_like(radial)
                        )
                        vertical = (
                            (hi - lo) / (cfg.levels_m[phase + 1] - cfg.levels_m[phase]) / stretch
                        )
                        horizontal = (
                            (1 - fraction) * lo_prime + fraction * hi_prime - shear * vertical
                        )
                        vector = asm(
                            volume_derivative,
                            local,
                            pressure=field,
                            density=rho[phase],
                            k2=(omega * radius / speed[phase]) ** 2,
                            vertical=vertical,
                            horizontal=horizontal,
                        )
                        load[:, column] += vector
                        for j, face in enumerate(context["faces"]):
                            if phase <= j:
                                prefix_load[j][:, column] += vector[face["ids"]]
                    mapping.cache.clear()
                    del local, field
            pressure_delta = factor.solve(-load)
            del load
            for j, face in enumerate(context["faces"]):
                trace, ids = face["trace"], face["ids"]
                normal_field = trace.interpolate(face["normal_velocity_coefficients"])
                rr = response.radial_m[j]
                slope = face["slope"]
                right = (prefix_load[j] + face["prefix"] @ pressure_delta) / (1j * omega)
                slope_delta = []
                for column, direction in enumerate(selected.T):
                    variation = space.evaluate(direction.reshape(2, space.count)[j], rr, 1)
                    slope_delta.append(variation)
                    relative = (slope * variation / (1 + slope**2)).reshape(trace.dx.shape)
                    right[:, column] -= asm(
                        mass_derivative, trace, relative_stretch=relative, velocity=normal_field
                    )[ids]
                velocity_delta_coeff = solve(face["mass"], right, assume_a="pos")
                cp = 0.25 * (1 / (rho[j] * speed[j] ** 2) - 1 / (rho[j + 1] * speed[j + 1] ** 2))
                cv = (rho[j] - rho[j + 1]) / 4
                ct = cv / (omega**2 * rho[j] * rho[j + 1])
                observation = space.basis(rr).T * response.weights_m2[j]
                for column in range(count):
                    field = trace.interpolate(pressure_delta[:, column])
                    dp = np.asarray(field).ravel()
                    gradient = field.grad.reshape(2, -1).T / radius
                    dg = np.einsum("sd,sd->s", face["tangent"], gradient)
                    dg -= (
                        response.tangent_gradient[j][:, 0]
                        * slope
                        * slope_delta[column]
                        / (1 + slope**2)
                    )
                    recovered = np.zeros(basis.N, complex)
                    recovered[ids] = velocity_delta_coeff[:, column]
                    dv = np.asarray(trace.interpolate(recovered)).ravel()
                    traction = 2 * np.real(
                        cp * response.pressure[j][:, 0].conj() * dp
                        + cv * response.velocity[j][:, 0].conj() * dv
                        + ct * response.tangent_gradient[j][:, 0].conj() * dg
                    )
                    answer[j * space.count : (j + 1) * space.count, start + column] = (
                        observation @ traction
                    )
            if progress is not None:
                progress(start + count, directions.shape[1], answer[:, : start + count])
        del self._linearization
        return response, answer

    def _block_traces(
        self,
        coefficients,
        matrix,
        lu,
        rhs,
        basis,
        mapping,
        partials,
        phases,
        boundary_matrix,
        block_size,
    ):
        cfg, space = self.config, self.surface
        radius, omega = cfg.radius_m, 2 * np.pi * cfg.frequency_hz
        rho, speed = np.asarray(cfg.density_kg_m3), np.asarray(cfg.sound_speed_m_s)

        def probe(trace, derivative=None):
            values = np.stack(
                [
                    np.asarray(v[0]) if derivative is None else v[0].grad[derivative]
                    for v in trace.basis
                ]
            )
            columns = np.broadcast_to(trace.element_dofs[:, :, None], values.shape)
            rows = np.broadcast_to(
                np.arange(trace.dx.size).reshape((1,) + trace.dx.shape), values.shape
            )
            return csr_matrix(
                (values.ravel(), (rows.ravel(), columns.ravel())), shape=(trace.dx.size, basis.N)
            )

        records = []
        for j, facets in enumerate(self.interfaces):
            traces = [
                InteriorFacetBasis(
                    self.flat,
                    self.element,
                    mapping=mapping,
                    facets=facets,
                    side=side,
                    intorder=self.order,
                )
                for side in (0, 1)
            ]
            rr = traces[0].global_coordinates()[0].ravel() * radius
            slope = space.evaluate(coefficients[j], rr, 1)
            nz = 1 / np.sqrt(1 + slope**2)
            pp = [probe(t) for t in traces]
            gr = [probe(t, 0) / radius for t in traces]
            gz = [probe(t, 1) / radius for t in traces]
            normals = []
            for t, dr, dz in zip(traces, gr, gz):
                density = rho[np.repeat(self.regions[t.tind], t.dx.shape[1])]
                normals.append(
                    dr.multiply((-slope * nz / (1j * omega * density))[:, None])
                    + dz.multiply((nz / (1j * omega * density))[:, None])
                )
            tangent = (gr[0] + gr[1]).multiply((nz / 2)[:, None]) + (gz[0] + gz[1]).multiply(
                (slope * nz / 2)[:, None]
            )
            ids = basis.get_dofs(facets=facets).all()

            @BilinearForm
            def trace_mass(u, v, w):
                return radius**2 * w.x[0] * u * v

            mass = asm(trace_mass, traces[0]).tocsc()[ids][:, ids].toarray()
            shape = (len(rr), cfg.channels)
            records.append(
                {
                    "r": rr,
                    "nz": nz,
                    "weights": traces[0].dx.ravel() * nz * rr * radius * 2 * np.pi,
                    "p_op": (pp[0] + pp[1]) / 2,
                    "v_op": (normals[0] + normals[1]) / 2,
                    "t_op": tangent,
                    "jump_op": normals[0] - normals[1],
                    "flux_op": partials[j].tocsr() / (1j * omega),
                    "flux_rhs": rhs[ids].multiply((phases <= j)[None, :]).tocsr() / (1j * omega),
                    "mass_factor": cho_factor(mass),
                    "recovery": pp[0][:, ids],
                    **{key: np.zeros(shape, complex) for key in ("p", "v", "t", "jump")},
                }
            )
        partials.clear()
        residual2, rhs2 = 0.0, 0.0
        for start in range(0, cfg.channels, block_size):
            selected = slice(start, min(start + block_size, cfg.channels))
            load = rhs[:, selected].toarray()
            field = lu.solve(load)
            residual2 += np.linalg.norm(matrix @ field - load) ** 2
            rhs2 += np.linalg.norm(load) ** 2
            for record in records:
                for key in ("p", "t", "jump"):
                    record[key][:, selected] = record[f"{key}_op"] @ field
                if cfg.normal_velocity_trace == "weak_flux":
                    flux = record["flux_op"] @ field - record["flux_rhs"][:, selected].toarray()
                    record["v"][:, selected] = record["recovery"] @ cho_solve(
                        record["mass_factor"], flux
                    )
                else:
                    record["v"][:, selected] = record["v_op"] @ field
        kernels = []
        for j, record in enumerate(records):
            cp = 0.25 * (1 / (rho[j] * speed[j] ** 2) - 1 / (rho[j + 1] * speed[j + 1] ** 2))
            cv = (rho[j] - rho[j + 1]) / 4
            ct = cv / (omega**2 * rho[j] * rho[j + 1])
            b = space.basis(record["r"])
            kernels.append(
                sum(
                    factor * weighted_source_grams(record["weights"], b, record[key])
                    for factor, key in ((cp, "p"), (cv, "v"), (ct, "t"))
                )
            )
        return WaveResponse(
            np.concatenate(kernels),
            *[[r[key] for r in records] for key in ("p", "v", "t", "r", "weights", "nz", "jump")],
            np.empty((0, cfg.channels), complex),
            basis,
            boundary_matrix,
            rhs,
            float(np.sqrt(residual2 / max(rhs2, 1e-60))),
            cfg,
            linear_solver=self.linear_solver,
            factorized_dofs=len(lu.boundary) if hasattr(lu, "boundary") else basis.N,
            krylov_iterations=getattr(lu, "iterations", 0),
            preconditioner_refreshes=getattr(lu, "refreshes", 0),
            volume_assembly_chunk_size=self.volume_assembly_chunk_size,
            volume_integration_order=self.order,
        )
