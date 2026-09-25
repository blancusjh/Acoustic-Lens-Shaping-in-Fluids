# Material-referenced nominal design

## Decision, 23 September 2026

Use an axisymmetric nominal cylinder and annular source design. A full 3D
transient is not a prerequisite for nominal source optimization. Symmetry of
the apparatus does not exclude non-axisymmetric instabilities; retain that
as a robustness question, not an assertion that such modes do not exist.

Use **uncured Norland NOA 61 at 25 C** as the resin reference, with water on
both sides for the first material-referenced pressure calculation. This is a
new declared material case, not a relabeling of the old heavy-below stack.
The lower interface now has a destabilizing gravity contribution and the
upper a restoring one; gravity is retained with those signs. No assumption
that both density jumps are positive is inherited from the earlier proof.

## Manufacturer data and unit conversions

The [Norland technical sheet](https://norlandproducts.com/wp-content/uploads/2025/02/Norland-Products-NOA-61-TDS.pdf)
reports liquid viscosity 300 cP at 25 C (0.30 Pa s), liquid density 1.231 g/cm3
(1231 kg/m3), liquid refractive index 1.52 and cured refractive index 1.56.
Its surface-tension value 40 dyn/cm (0.040 N/m) is not a resin-water
interfacial-tension measurement. Its typical linear shrinkage is 1.5 percent;
the cured lens must therefore be treated separately from the driven liquid.
Do not equate that bulk shrinkage number to an exact surface-displacement map.

The previous 5 Pa s scenario was not taken from a chosen product. Nevertheless,
5 Pa s is within documented optical-adhesive values: the
[Dymax OP-52 sheet, revised 2023](https://dymax.com/content/download/4402/file_archived/OP-52%20PDS.pdf)
lists nominal viscosity 5000 cP. Assigning it to both surrounding liquids as
well as the resin was the unsupported part of the previous material scenario.
The newer OP-52 sheet reports cured index 1.53 and shrinkage 0.63 percent;
older catalog values differ. Do not mix these revisions or use a cured index
as the liquid index. NOA 61 is the first reference because it supplies explicit
liquid density and index as well as viscosity.

## Explicit engineering assumptions for the pressure calculation

The case is `configs/dual/noa61-water-pressure-reference.json`. Keep the
existing 4 mm cylinder radius, 2 mm clear radius, 6 mm cell height, vertex
displacements and shared laboratory conjugates. Reconstruct both Cartesian
surfaces for the new liquid indices; do not reuse a cached acoustic operator.
Use rounded water properties at 25 C, clean immiscible interfaces and pinned
volume-conserving rims. Resin-water interfacial tension is explicitly assumed
to be 0.025 N/m on both faces for this reference calculation. The source sheet
does not determine it. This is an engineering input, not a measured property.

The recorded 1500 m/s resin sound speed is an engineering placeholder, not
used by the static pressure solver. It must be varied or characterized in
acoustic design, not presented as manufacturer data. The prior acoustic
commands do not belong to this case. A static pressure audit does not use
viscosity and supplies no new settling time.

## Fixed modeling choices for the next source calculation

The separate [pressure audit](../artifacts/studies/S03-stigmatic-target-ideal-load/ideal-load-2026-09-23/noa61-water-reference/report.md)
was executed at five surface resolutions and two load-quadrature orders.
For the assumed 0.025 N/m interfacial tensions, the gauge-fixed complete-face
traction ranges are approximately -6.75 to 27.59 Pa (lower) and -10.46 to
23.69 Pa (upper). The coarsest independently recovered equilibrium has pupil
errors 0.142/0.111 nm and maximum geometric spot radius 0.0204 micrometres,
with all 4001 rays transmitted; errors decrease with refinement. This is a
static ideal-load check under the declared material assumptions, not a
new formation trajectory or an acoustic command. No old 20 s settling time
is assigned to the water/NOA 61/water configuration.

Water reference formulations for subsequent refinement are
[IAPWS liquid-water properties](https://www.iapws.org/relguide/LiquidWater.html)
and the [NIST refractive-index reference](https://www.nist.gov/publications/index-refraction-liquid-water).
The rounded engineering inputs are not claimed to be a fresh evaluation of
those formulations or nanometre-accurate material characterization.

The target remains the maximum cycle-mean liquid-surface error on both pupils
and the maximum geometric spot at the fixed detector. Hold UV off during
formation and optical verification; curing is a separate stage. Use
axisymmetric liquid momentum with inertia and viscosity for this much less
viscous outer fluid. Streaming denotes the slow mean flow driven by ultrasound,
and belongs in that same mean-momentum problem when its forcing is included.
Do not model the formation time by scaling the old equal-viscosity trajectory.

Represent sources initially by annular end-cap regions and cylindrical wall
bands. The eventual design output must give physical support, frequency,
complex normal-velocity commands, and fresh achieved full-interface traction,
surface and optical checks. Voltage requires a transducer calibration; ideal
normal-velocity commands must not be labeled hardware voltages. Unknown
properties are explicit parameters to a sensitivity/robustness calculation,
not reasons to fabricate a unique material-qualified command.
