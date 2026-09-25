# Real-fluid feasibility screen

23 September 2026. User direction: choose whatever materials and operating
conditions are realistically reachable. The objective remains simultaneous
10 nm maximum cycle-mean error on both clear apertures, not instantaneous
carrier suppression. No material system is yet qualified for that objective.

## Experimental starting point, not a precision claim

The current material-referenced lens case is documented in
[resin design reference](resin-design-reference.md): uncured NOA 61 at 25 C,
with water on both sides and explicitly recorded unknown/assumed interfacial
and acoustic properties. Its full-domain static pressure audit is separate
from the earlier equal-5-Pa-s viscous trajectory. A real resin viscosity is
not a justification for assigning the same viscosity to surrounding liquids.

Koyama, Isago and Nakamura demonstrated a variable-focus acoustic lens using
degassed water and silicone oil in a 6 mm diameter, 3 mm high cell with an
annular piezoelectric transducer. Their abstract reports a 6.7 ms fastest
response with 100 cSt oil. This supports a material family and an actuation
mechanism, not arbitrary two-face shaping or 10 nm accuracy.
[Primary research, Optics Express 18, 25158–25169 (2010)](https://doi.org/10.1364/OE.18.025158).

As a procurement-screen example only, Dow lists PMX-200 100 cSt silicone
fluid with specific gravity 0.964 at 25 °C, refractive index 1.403 and
kinematic viscosity 100 cSt (1e-4 m²/s). These typical product properties
are not lot-specific measurements. Its listed surface tension must NOT be
substituted for oil–water interfacial tension.
[Manufacturer product data](https://www.dow.com/en-us/pdp.xiameter-pmx-200-silicone-fluid-100-cst.01013190z.html).
This product has not been identified as the exact oil used in the experiment.

Use 298.15 K as a provisional characterization temperature, not as a claim
that any particular temperature tolerance is sufficient for nanometre shaping.
Water/silicone oil is the first family to investigate, not a committed apparatus.

## Two-face apparatus constraints

The present hypothetical configuration has density decreasing upward:
1200 / 1100 / 1000 kg/m³. A water–oil–water stack would not reproduce this:
the upper water–oil interface places the denser fluid above the lighter fluid.
Consequently, this substitution needs a capillary, gravity and full
three-dimensional stability calculation; it cannot inherit the current
stratification assumptions. A third immiscible liquid might permit a different
density ordering, but no compatible triplet has been qualified. Container
orientation or confinement changes must be declared as new apparatus cases.

The current optical indices 1.33 / 1.50 / 1.33 likewise cannot be assigned to
water/PMX-200 by name. Preserve the present geometric target as a numerical
benchmark. If actual indices lead to a new Cartesian optical target, record
that as a separate physical case, not an improvement on the unchanged target.
Do not reduce aperture or sag silently.

## Data required before a physical candidate run

- Exact grades, compositions, temperature, optical wavelength, density and
  refractive index, including temperature derivatives and uncertainty.
- Sound speed and attenuation at the operating frequency, shear/bulk
  viscosity, thermal conductivity, heat capacity and nonlinear response.
- Pairwise immiscibility, interfacial tension and its temperature/composition
  dependence; wall wetting, pinning and contamination controls.
- Static pressure, dissolved-gas preparation and an experimentally supported
  operating envelope. A generic cavitation threshold is not an adequate bound.
- Real transducer geometry, bandwidth, displacement/voltage calibration,
  losses and heat removal. Ideal coherent boundary ports are not a fabricated
  448-channel array; computed acoustic source work is not electrical input.

Unknown values stay unknown. No plausible-looking defaults may be labelled
measured properties. Property screening precedes purchase recommendations.

## Current evidence and next gates

The earlier independent-target hypothetical 448-port candidate requires approximately 28 MPa sampled
peak pressure and 34 W modeled acoustic source work. Neither establishes a
realistic linear, isothermal operating regime. The fixed-command surface
refinement also fails: at 104 elements, mean errors are approximately 10.44 µm
front and 4.81 µm back. See [numerical evidence](precision-results.md).

The newer shared-conjugate 7.2 MHz candidate in
`artifacts/dual-stigmatic-2026-09-23/kernel-audit-direct-448` reaches approximately
53.3 MPa sampled pressure and 47.66 W modeled source work. Its independently
solved refined equilibrium still misses the target by 370/741 nm. This is not
a qualified physical operating point; see the [current correction study](joint-stigmatic-correction.md).

First resolve numerical traction/surface/acoustic convergence. Then qualify
a real material system and optimize with pressure, dissipation, temperature
and calibrated-source limits, retaining the cycle-mean criterion. Thermal
feedback, streaming, formation and three-dimensional stability must be tested
before calling any stationary solution a maintained lens. Allocate numerical,
material/calibration and measurement uncertainty within the 1e-8 m total
budget; assigning that budget alone does not demonstrate compliance.

More sources are permitted, but are not the next justification for success:
they must improve a converged, physically constrained forward model.
