# Research decisions — 5 September 2026

The objective is a credible forward investigation of acoustically shaped, eventually
cured freeform optics. The project is separate from benchmark packaging. There are
no predictions of model scores here. All current apparatus values are inspectable.

## First milestone

Build a non-axisymmetric coherent array reference whose numerical limits can be
checked independently, and make each physical field visible. Keep acoustic particle
motion, radiation traction, slow liquid motion and optical shape distinct.

The 256 pressure patches replace the old elementary forcing description. The
two-focus pulse example is an initial excitation, not an optimized lens. Arbitrary
element weights are available, but diffraction, limited aperture, traction sign,
capillarity, volume and stability constrain reachable surfaces.

The present small-deformation equations are intentionally simpler than a mature
forward simulator. Their purpose is to provide a trustworthy limit for testing later
moving-boundary CFD. Solving this linear reference is not itself the intended
scientific challenge or a reason to predict benchmark difficulty.

## What time contributes now

The interface has a state. In the wave limit that state includes height and velocity;
in the Stokes limit it includes height. Identical instantaneous drives can therefore
produce different surfaces after different pulse histories. The notebook compares
10 ms and 40 ms at the same drive amplitude and elapsed time within each pulse.

This is not yet rheological or chemical memory. Linear hydrodynamic memory can be
handled by ordinary convolution/state-space methods. More difficult behavior should
come from demonstrated physical interactions, such as an evolving interface changing
the acoustic field, finite-volume/contact-line dynamics, or measured cure-dependent
material properties. We will not add undocumented coefficients to force complexity.

## Evidence discovered in the first checks

1. The radiation tensor must include acoustic velocity. The pressure-release limit
   has almost zero total interface pressure while retaining nonzero radiation force.
2. Coherent cross terms materially affect the example; adding beam intensities loses
   a real interaction. A direct combined acoustic solve checks the quadratic basis.
3. The 28 mm box was too influential over the water–air observation window. The
   recorded 56-to-84 mm central shape difference is 0.478% through 60 ms.
4. The equal-density Stokes case is still domain sensitive: 8.85% in the same central
   norm through 300 ms. The absence of gravitational restoration in this idealized
   pair, long-wavelength response, and volume constraint deserve explicit study.
   The current comparison does not isolate those mechanisms individually.
5. Small source-pressure examples produce sub-micrometre surfaces. Their visual
   exaggeration is a display choice, not a claim of large physical deformation.

There is also a physical regime check to carry forward. For a Stokes mode, the ratio
of the omitted inertial term to viscous resistance on its relaxation time is
`ε = (ρ₁+ρ₂) S(k) / [4(μ₁+μ₂)² k³]`. In the 56 mm equal-density example it is
about 0.28 at the fundamental wavenumber, versus 0.016 at `k = 2000 m⁻¹`.
Thus the nominal material numbers do not justify neglecting inertia for every
long-wavelength mode of a real experiment. The example verifies the stipulated
Stokes equations; a physical resin prediction also needs the regime check.

## Next forward-model milestones, in dependency order

**Define a physically closed apparatus.** Specify chamber depth, lateral extent,
supporting frame, resin volume, both interfaces if relevant, wetting and contact-line
condition, array coupling layer, and treatment of returning sound. Compare finite-depth
mode response or a finite-domain solver with the deep reference. Decide which boundaries
are physical and which are numerical. Current periodic-cell results cannot make that decision.

**Couple wave propagation to the actual interface.** Start with prescribed non-flat
shapes and verify scattering, pressure/normal-velocity continuity, and radiation-force
sign. Then couple to interface evolution with nonlinear curvature. Benchmark against
the small-amplitude limit and the relevant cases in Chesneau et al. before interpreting
large shapes. Their axisymmetric BEM work is a physical anchor, not ready-made proof of
a general non-axisymmetric implementation.

**Choose the rheological regime from measurements.** Water–air weak damping and a
zero-inertia viscous pair are separate limits. For a candidate resin/bath, assemble
density, sound speed, attenuation versus frequency, viscosity versus temperature,
surface tension and wetting data. Check inertial, capillary and viscous time scales.
If unsteady viscous effects matter, use a justified transient two-fluid model rather
than extending the weak damping formula outside its range.

**Assess competing acoustic effects.** Compare attenuation-driven streaming,
boundary streaming and thermal/Marangoni effects with radiation-pressure forcing at
the intended drive levels. Add only mechanisms that the physical scales or experiments
support. A beautiful radiation-pressure-only shape does not prove that streaming is absent.

**Validate a forward experiment.** Reproduce a published transient measurement with
its documented geometry and uncertainties, or obtain a small set of bench measurements.
Identify discrepancies without compensating for them through undisclosed fitting.

Only after these forward steps are credible: add resin cure conversion and shrinkage,
measure final optical figure/index, and investigate inverse design/control. The relevant
question is which useful surface family can be created and retained with tolerable
error—not whether every mathematically drawn shape is achievable.

## Future experiments worth pursuing

- Opposing or surrounding arrays to widen signed-traction control, with a realistic
  model of cavity reflections and mechanical access for curing/metrology.
- Non-axisymmetric targets with saddle/astigmatic terms, to expose aperture and
  surface-tension limits beyond axisymmetric lenses.
- Pulse-history experiments separating wave interference from thermal or rheological
  memory; vary pulse separation while preserving the instantaneous input.
- Robustness to measured phase/amplitude calibration errors and fill-volume changes.
- Hold-and-cure experiments where final solid figure, not a transient liquid snapshot,
  is the meaningful optical output.

These are research directions, not implemented capabilities. The supporting paper
map is in `literature/reading-map.md`.
