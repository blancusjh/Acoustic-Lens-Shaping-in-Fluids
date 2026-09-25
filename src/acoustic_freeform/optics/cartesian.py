"""Vertex branch of a monochromatic Cartesian refracting diopter.

Silva-Lora & Torres, JOSA A 37, 1155 (2020), Eqs. (1), (3)-(11).
The paper's rho is sqrt(r**2 + z**2), not the cylindrical radius r.
Distances are signed from the vertex; propagation at the vertex is +z.
Positive indices describe refraction. Reflection and negative-index media
are deliberately outside this liquid-diopter implementation.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CartesianDiopter:
    n_o: float
    z_o_m: float
    n_i: float
    z_i_m: float

    def __post_init__(self):
        if not all(np.isfinite(n) and n > 0 for n in (self.n_o, self.n_i)):
            raise ValueError("Refractive indices must be finite and positive.")
        if self.n_o == self.n_i:
            raise ValueError("Equal indices do not define a unique refracting vertex branch.")
        if any(np.isnan(z) or z == 0 for z in (self.z_o_m, self.z_i_m)):
            raise ValueError("Conjugate distances must be nonzero, signed distances or infinity.")

    @property
    def vertex_curvature_m_inv(self):
        return (self.n_i / self.z_i_m - self.n_o / self.z_o_m) / (self.n_i - self.n_o)

    def form_parameters(self):
        """G, O, T, S in the paper, with inverse distances for infinity limits.

        Some regular surfaces have singular GOTS coordinates. Their physical
        branch is still evaluated from the unsquared Fermat equation by sag().
        """
        a, b = 1 / self.z_o_m, 1 / self.z_i_m
        no, ni = self.n_o, self.n_i
        d = ni * a - no * b
        if abs(d) < 1e-14 * max(abs(a), abs(b), 1):
            raise ValueError("Singular GOTS coordinates; use the signed Fermat branch.")
        o = self.vertex_curvature_m_inv
        og = (ni * ni * a - no * no * b) ** 2 / (ni * no * d * (ni - no))
        t = (ni - no) * (ni + no) ** 2 * a * a * b * b / (4 * ni * no * d)
        s = (ni + no) * (ni * ni * a - no * no * b) * a * b / (2 * ni * no * d)
        return {
            "G": og / o if o else None,
            "O_m_inv": o,
            "T_m_inv3": t,
            "S_m_inv2": s,
            "OG_m_inv": og,
        }

    def parametric(self, rho_m):
        """Paper Eqs. (10)-(11), the rationalized branch through the vertex."""
        rho = np.asarray(rho_m, dtype=float)
        p = self.form_parameters()
        o, t, s, og = (p[k] for k in ("O_m_inv", "T_m_inv3", "S_m_inv2", "OG_m_inv"))
        discriminant = 1 + (2 * s - o * og) * rho**2
        if np.any(discriminant < 0):
            raise ValueError("The requested polar parameter leaves the real vertex branch.")
        z = (o + t * rho**2) * rho**2 / (1 + s * rho**2 + np.sqrt(discriminant))
        radial_squared = rho**2 - z**2
        if np.any(radial_squared < -1e-24):
            raise ValueError("The requested polar parameter has no real cylindrical radius.")
        return np.sqrt(np.maximum(0, radial_squared)), z

    def fermat(self, r_m, z_m):
        """Signed optical-path difference relative to the axial vertex ray.

        Real object + real image gives no*Lo + ni*Li = constant. Signs
        extend this to virtual conjugates. Rationalized distance increments
        avoid subtracting almost equal long optical paths.
        """
        r, z = np.broadcast_arrays(np.asarray(r_m, float), np.asarray(z_m, float))
        result = np.zeros_like(r)
        for n, conjugate, side in ((self.n_o, self.z_o_m, -1), (self.n_i, self.z_i_m, 1)):
            if np.isinf(conjugate):
                result -= side * n * z
            else:
                length = np.hypot(r, z - conjugate)
                increment = (r * r + z * z - 2 * z * conjugate) / (length + abs(conjugate))
                result += side * n * np.sign(conjugate) * increment
        return result

    def fermat_gradient(self, r_m, z_m):
        r, z = np.broadcast_arrays(np.asarray(r_m, float), np.asarray(z_m, float))
        fr, fz = np.zeros_like(r), np.zeros_like(r)
        for n, conjugate, side in ((self.n_o, self.z_o_m, -1), (self.n_i, self.z_i_m, 1)):
            if np.isinf(conjugate):
                fz -= side * n
            else:
                length = np.hypot(r, z - conjugate)
                fr += side * n * np.sign(conjugate) * r / length
                fz += side * n * np.sign(conjugate) * (z - conjugate) / length
        return fr, fz

    def sag(self, r_m):
        """Solve the unsquared Fermat equation near its smooth vertex branch.

        A radial continuation guards against selecting an extraneous quartic
        branch. Single-valued upward-transmitting apertures are required.
        """
        r = np.asarray(r_m, dtype=float)
        if np.any(~np.isfinite(r)):
            raise ValueError("Aperture coordinates must be finite.")
        z = np.zeros_like(r)
        # Continue all radii outwards together; this also handles unsorted input.
        for fraction in np.linspace(0, 1, 33)[1:]:
            rr = abs(r) * fraction
            for _ in range(30):
                _, fz = self.fermat_gradient(rr, z)
                if np.any(abs(fz) < 1e-9 * abs(self.n_o - self.n_i)):
                    raise ValueError("The Cartesian branch reaches a vertical tangent.")
                delta = self.fermat(rr, z) / fz
                z -= delta
                if np.max(abs(delta), initial=0) < 2e-15:
                    break
            else:
                raise ValueError("Could not continue the Cartesian vertex branch to this aperture.")
        fr, fz = self.fermat_gradient(abs(r), z)
        slope = -fr / fz
        transmitted = self.refract(abs(r), z, slope)
        expected = self.image_direction(abs(r), z)
        if np.max(abs(transmitted - expected), initial=0) > 1e-8:
            raise ValueError("This aperture leaves the physical upward-transmitting vertex branch.")
        return z

    def slope(self, r_m, sag_m=None):
        r = np.asarray(r_m, float)
        z = self.sag(r) if sag_m is None else np.asarray(sag_m, float)
        fr, fz = self.fermat_gradient(r, z)
        return -fr / fz

    def incident_direction(self, r_m, z_m):
        r, z = np.broadcast_arrays(np.asarray(r_m, float), np.asarray(z_m, float))
        if np.isinf(self.z_o_m):
            return np.stack([np.zeros_like(r), np.ones_like(r)], axis=-1)
        v = np.stack([r, z - self.z_o_m], axis=-1)
        return -np.sign(self.z_o_m) * v / np.linalg.norm(v, axis=-1, keepdims=True)

    def image_direction(self, r_m, z_m):
        r, z = np.broadcast_arrays(np.asarray(r_m, float), np.asarray(z_m, float))
        if np.isinf(self.z_i_m):
            return np.stack([np.zeros_like(r), np.ones_like(r)], axis=-1)
        v = np.stack([-r, self.z_i_m - z], axis=-1)
        return np.sign(self.z_i_m) * v / np.linalg.norm(v, axis=-1, keepdims=True)

    def refract(self, r_m, z_m, slope):
        """Independent vector Snell law applied to any supplied surface."""
        incident = self.incident_direction(r_m, z_m)
        normal = np.stack([-np.asarray(slope), np.ones_like(slope)], axis=-1)
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
        cosine = np.sum(incident * normal, axis=-1)
        eta = self.n_o / self.n_i
        discriminant = 1 - eta**2 * (1 - cosine**2)
        if np.any(discriminant < -1e-13) or np.any(cosine <= 0):
            raise ValueError("Total internal reflection or back-facing diopter in the aperture.")
        return (
            eta * incident
            + (np.sqrt(np.maximum(0, discriminant)) - eta * cosine)[..., None] * normal
        )

    def as_dict(self):
        return {
            "n_o": self.n_o,
            "z_o_m": self.z_o_m if np.isfinite(self.z_o_m) else str(self.z_o_m),
            "n_i": self.n_i,
            "z_i_m": self.z_i_m if np.isfinite(self.z_i_m) else str(self.z_i_m),
            "coordinate_origin": "Target surface vertex; positive z follows axial light propagation.",
        }
