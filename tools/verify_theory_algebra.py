"""Exact symbolic checks of manuscript identities; no PDE solve or optimization.

Run with: uv run --with sympy==1.14.0 python tools/verify_theory_algebra.py
The checks do not establish model validity, functional regularity or stability.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import sympy as sp


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    checks: list[dict[str, str]] = []

    def zero(name: str, residual: sp.Expr, equation: str) -> None:
        reduced = sp.simplify(sp.expand_complex(residual))
        if reduced != 0:
            raise AssertionError(f"{name}: nonzero residual {reduced}")
        checks.append({"check": name, "equation_label": equation, "residual": "0"})

    def complex_symbol(name: str) -> sp.Expr:
        re, im = sp.symbols(f"{name}_re {name}_im", real=True)
        return re + sp.I * im

    omega, rho, sound = sp.symbols("omega rho sound", positive=True)
    pressure, dilation = complex_symbol("P"), complex_symbol("q")
    div_force = complex_symbol("div_alpha_F")
    # Eliminate velocity divergence from the two first-order equations.
    div_velocity = sp.I * omega * pressure / (rho * sound**2) + dilation
    div_pressure_gradient = sp.I * omega * div_velocity + div_force
    zero(
        "Elimination of velocity into forced Helmholtz equation",
        -div_pressure_gradient - omega**2 * pressure / (rho * sound**2)
        + div_force + sp.I * omega * dilation,
        "eq:forced-helmholtz",
    )

    adjoint = complex_symbol("Z")
    delta_p, delta_v = complex_symbol("delta_P"), complex_symbol("delta_v")
    a_obs, b_obs = complex_symbol("A_obs"), complex_symbol("B_obs")
    delta_g, delta_q = complex_symbol("delta_g"), complex_symbol("delta_q")
    admittance = complex_symbol("Y")
    jump_z = sp.I * b_obs / omega
    jump_flux_z = -a_obs
    interface_term = (
        -delta_p * sp.conjugate(jump_flux_z)
        + sp.I * omega * delta_v * sp.conjugate(jump_z)
    )
    zero(
        "Transmission Green interface term recovers both observation covectors",
        interface_term - sp.conjugate(a_obs) * delta_p
        - sp.conjugate(b_obs) * delta_v,
        "eq:source-adjoint-jumps",
    )
    adjoint_outer_flux = -sp.I * omega * sp.conjugate(admittance) * adjoint
    forward_outer_flux = sp.I * omega * (admittance * delta_p + delta_g)
    zero(
        "Adjoint admittance cancels the homogeneous outer Green term",
        delta_p * sp.conjugate(adjoint_outer_flux)
        - forward_outer_flux * sp.conjugate(adjoint)
        + sp.I * omega * delta_g * sp.conjugate(adjoint),
        "eq:source-adjoint-wall",
    )
    release_z = -sp.I * b_obs / omega
    zero(
        "Pressure-release adjoint preserves velocity observation",
        -sp.I * omega * delta_v * sp.conjugate(release_z)
        - sp.conjugate(b_obs) * delta_v,
        "eq:pressure-release-adjoint",
    )
    zero(
        "Boundary and dilation gradients use the declared real pairing",
        sp.re(sp.conjugate(-sp.I * omega * adjoint) * delta_g
              + sp.conjugate(sp.I * omega * adjoint) * delta_q)
        - sp.re(sp.I * omega * delta_g * sp.conjugate(adjoint)
                - sp.I * omega * delta_q * sp.conjugate(adjoint)),
        "eq:distributed-gradients",
    )

    # Derive real acoustic-energy flux divergence from the first-order fields.
    force = [complex_symbol(f"F{j}") for j in range(3)]
    velocity = [complex_symbol(f"V{j}") for j in range(3)]
    grad_p = [f + sp.I * omega * rho * v for f, v in zip(force, velocity, strict=True)]
    flux_divergence = sum(gp * sp.conjugate(v) for gp, v in zip(grad_p, velocity, strict=True))
    flux_divergence += pressure * sp.conjugate(div_velocity)
    source_work = sum(f * sp.conjugate(v) for f, v in zip(force, velocity, strict=True))
    source_work += pressure * sp.conjugate(dilation)
    zero("Local real energy-flux balance", sp.re(flux_divergence - source_work), "eq:source-power")

    rho_m, rho_p, c_m, c_p = sp.symbols("rho_minus rho_plus c_minus c_plus", positive=True)
    p2, v2, tangential_grad2 = sp.symbols("pressure_squared velocity_squared gradient_squared", nonnegative=True)

    def tensor_normal(rho_phase: sp.Expr, c_phase: sp.Expr) -> sp.Expr:
        velocity_norm2 = v2 + tangential_grad2 / (omega**2 * rho_phase**2)
        return p2 / (4 * rho_phase * c_phase**2) - rho_phase * velocity_norm2 / 4 + rho_phase * v2 / 2

    reduced_traction = p2 * (1 / (rho_m * c_m**2) - 1 / (rho_p * c_p**2)) / 4
    reduced_traction += (rho_m - rho_p) * v2 / 4
    reduced_traction -= (1 / rho_m - 1 / rho_p) * tangential_grad2 / (4 * omega**2)
    zero(
        "Full radiation-tensor contraction equals transmission trace formula",
        tensor_normal(rho_m, c_m) - tensor_normal(rho_p, c_p) - reduced_traction,
        "eq:traction-trace",
    )

    amplitude_squared = sp.symbols("x", nonnegative=True)
    penalty = sp.symbols("gamma", positive=True)
    target = sp.symbols("b0:3", real=True)
    load = sp.symbols("f0:3", real=True)
    projected_cost = sum((amplitude_squared * f - b)**2 / 2 for f, b in zip(load, target, strict=True))
    projected_cost += penalty * amplitude_squared / 2
    zero_cost = sum(b**2 / 2 for b in target)
    a_ray = sum(b * f for b, f in zip(target, load, strict=True)) - penalty / 2
    b_ray = sum(f**2 for f in load)
    zero(
        "Exact projected-load amplitude polynomial",
        projected_cost - zero_cost + a_ray * amplitude_squared - b_ray * amplitude_squared**2 / 2,
        "eq:source-ray-amplitude",
    )
    zero(
        "Interior amplitude stationary condition",
        sp.diff(projected_cost, amplitude_squared).subs(amplitude_squared, a_ray / b_ray),
        "eq:source-ray-amplitude",
    )

    # Graph Euler derivatives: upward displacement and outward traction have
    # opposite signs on the front face. The connected lens has one multiplier.
    r = sp.symbols("r", positive=True)
    hf, hb = sp.Function("hf")(r), sp.Function("hb")(r)
    sf, sb, drho, gravity, lam = sp.symbols("sf sb drho gravity lambda", real=True)
    energy = r*(sf*sp.sqrt(1+sp.diff(hf,r)**2)+sb*sp.sqrt(1+sp.diff(hb,r)**2)
                +drho*gravity*(hb**2-hf**2)/2-lam*(hb-hf))
    euler_f = (sp.diff(energy,hf)-sp.diff(sp.diff(energy,sp.diff(hf,r)),r))/r
    euler_b = (sp.diff(energy,hb)-sp.diff(sp.diff(energy,sp.diff(hb,r)),r))/r
    outward_f = sf*sp.diff(r*sp.diff(hf,r)/sp.sqrt(1+sp.diff(hf,r)**2),r)/r+drho*gravity*hf-lam
    outward_b = -sb*sp.diff(r*sp.diff(hb,r)/sp.sqrt(1+sp.diff(hb,r)**2),r)/r+drho*gravity*hb-lam
    # Cancel before expansion into real and imaginary parts of symbolic functions.
    zero("Front-face Euler derivative reverses outward work", sp.simplify(euler_f+outward_f),
         "eq:two-face-loads")
    zero("Back-face Euler derivative equals outward work", sp.simplify(euler_b-outward_b),
         "eq:two-face-loads")

    # The cutoff construction satisfies the two forced acoustic balances with q=0.
    phi_gradient, lap_gradient, lap_phi = (complex_symbol(name) for name in
                                          ("phi_gradient", "lap_gradient", "lap_phi"))
    p_gradient = rho*sound**2*lap_gradient/(sp.I*omega)
    force_cutoff = rho*sound**2*(lap_gradient+omega**2/sound**2*phi_gradient)/(sp.I*omega)
    zero("Cutoff volume force satisfies acoustic momentum",
         -sp.I*omega*rho*phi_gradient+p_gradient-force_cutoff, "eq:cutoff-volume-force")
    zero("Cutoff pressure satisfies acoustic compression",
         -sp.I*omega*(rho*sound**2*lap_phi/(sp.I*omega))+rho*sound**2*lap_phi,
         "eq:cutoff-volume-force")

    output = root / "artifacts/theory/review/algebra.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    source_files = sorted((root / "docs/theory").rglob("*.tex")) + [Path(__file__).resolve()]
    record = {
        "verified_utc": datetime.now(UTC).isoformat(),
        "method": "Exact symbolic identities using real and imaginary components",
        "sympy_version": sp.__version__,
        "physics_simulations_executed": False,
        "source_optimization_executed": False,
        "limitations": "Selected algebra only; no proof of model validity, trace regularity, full coupled adjoint or stability.",
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "checks": checks,
    }
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{len(checks)} exact symbolic checks passed; {output}")


if __name__ == "__main__":
    main()
