# From optical conjugates to acoustic excitation

The research objective is the map

\[
(n_o,z_o,n_i,z_i;\ \text{apparatus, material, frequency})
\longrightarrow h_\star(r)\longrightarrow\Pi_\star(r)\longrightarrow\{w_j(t)\}.
\]

The four optical parameters determine the local stigmatic vertex branch. They do
not determine an electrical drive independently of the apparatus. A finite aperture,
an admissible branch and mounting geometry must also be specified. Several acoustic
excitations can produce similar traction; a common phase rotation always leaves the
cycle-averaged force unchanged.

## Paper and coordinates

The primary source is Silva-Lora and Torres, *Superconical aplanatic ovoid singlet
lenses*, JOSA A **37**, 1155–1165 (2020),
[DOI 10.1364/JOSAA.392795](https://doi.org/10.1364/JOSAA.392795).
The user's PDF is in the private library, with its hash in `references/manifest.json`.
Equations (1), (3)–(11) define the single surface used here; the paper's later
two-surface singlet design is a different optical system.

The subscript is the letter **o** for object. The origin is the **target vertex**,
with positive z following axial light propagation.

| Parameter | Meaning |
|---|---|
| `n_o` | Incident-medium refractive index |
| `z_o_m` | Signed object coordinate; negative for a real object, positive for a virtual object |
| `n_i` | Transmitted-medium refractive index |
| `z_i_m` | Signed image coordinate; positive for a real image, negative for a virtual image |

For a real object and real image, Fermat's condition is

\[
n_o\sqrt{r^2+(z-z_o)^2}+n_i\sqrt{r^2+(z_i-z)^2}
=n_o|z_o|+n_i|z_i|.
\]

The implementation continues the **unsquared** equation from the vertex. Signed
paths extend it to virtual conjugates; plane-wave limits handle infinite conjugates.
A real root of the quartic alone is insufficient because squaring can retain
branches that fail the intended Snell refraction.

Define \(a=1/z_o\), \(b=1/z_i\), and \(D=n_i a-n_o b\). The paper's form
parameters, expressed without infinite products, are

\[
O=\frac{n_i b-n_o a}{n_i-n_o},\qquad
G=\frac{(n_i^2a-n_o^2b)^2}{n_in_oD(n_i b-n_o a)},
\]
\[
T=\frac{(n_i-n_o)(n_i+n_o)^2a^2b^2}{4n_in_oD},\qquad
S=\frac{(n_i+n_o)(n_i^2a-n_o^2b)ab}{2n_in_oD}.
\]

The rationalized parameterization is

\[
z(\rho)=\frac{(O+T\rho^2)\rho^2}
{1+S\rho^2+\sqrt{1+(2S-O^2G)\rho^2}},\qquad
r(\rho)=\sqrt{\rho^2-z(\rho)^2}.
\]

Here **\(\rho=\sqrt{r^2+z^2}\), not cylindrical radius r**. GOTS coordinates can
be singular even when the Fermat branch is regular; the code reports that distinction.
The standalone optical class supports positive indices and signed real/virtual
conjugates. It rejects inaccessible graph branches and nonphysical transmission.
Reflection, negative-index media and overhanging interfaces are outside its scope.

As \(z_o\to-\infty\), \(T=S=0\) and \(G=K=-(n_o/n_i)^2\). Our first
asphere was therefore already a degenerate Cartesian oval: a hyperboloidal
liquid-to-air diopter with collimated input. A finite-conjugate oval is generally
not a conic; its G must not be reported as a fitted conic constant K.

## Mounting and illumination

The fluid solver currently supports a real object or collimated input, a real image,
and the existing liquid–air acoustic boundary. `CartesianDiopter` separately supports
other positive optical indices and virtual conjugates. Setting `n_i` to an immersion
liquid would require a two-medium acoustic and flow model, so the fluid configuration
rejects that substitution.

The exact oval is translated to meet the fixed chamber rim:

\[
h_\star(r)=z(r)-z(4\,\mathrm{mm}).
\]

This fixes the vertex height and required resin volume. Each optical parameter set
therefore defines a fill volume. Comparing separately filled experiments is not
continuous tuning at one conserved fill. A future fixed-fill family could use a
non-optical annulus or movable mount for the additional degree of freedom. Within
each run the chosen fill is conserved throughout the physical trajectory.

For finite conjugates the spherical incident wavefront is **prescribed inside the
resin**. An external point source generally acquires aberration at the flat window;
external corrective illumination has not been designed here. The viewer draws
finite-conjugate incident rays only inside the resin. The specified object coordinate
is the wavefront center in the homogeneous incident-medium optical model, not a
depicted physical source outside the chamber.

## Required perturbation and force

For the chosen fill, solve the unforced nonlinear equilibrium \(h_0(r)\). The
required perturbation is \(\delta h=h_\star-h_0\). It is an optical target, not a
prescribed displacement in the forward solver.

With outward normal into air, curvature is positive on a convex liquid cap:

\[
\kappa=-\left[\frac{h''}{(1+h'^2)^{3/2}}+
\frac{h'}{r\sqrt{1+h'^2}}\right].
\]

The normal balance is

\[
\Pi_\star(r)=\sigma\kappa(r)+\rho g h(r)-\lambda,
\]

where \(\lambda\) is the constant liquid-pressure/volume multiplier. The shape is
sensitive to the spatially varying traction. Our acoustic model gives

\[
\Pi(r;w,h)=\frac{\rho}{4}\left|\sum_jw_jV_{n,j}(r;h)\right|^2.
\]

The force contains coherent cross terms and depends on the deformed cavity. A force
fit on the target alone therefore needs a coupled forward check. The wave/interface
feedback is motivated by
[Chesneau et al., PRE 106, 065104 (2022)](https://doi.org/10.1103/PhysRevE.106.065104),
within our explicitly different finite-element, one-liquid approximation.

## Reproduce and read the excitation

The first configuration is
[`noa61_cartesian_50_20.toml`](../configs/lenses/noa61_cartesian_50_20.toml):
\(n_o=1.52,z_o=-50\,\mathrm{mm},n_i=1,z_i=20\,\mathrm{mm}\).
Its `[diopter]` section contains exactly those four optical parameters.

```bash
uv run lenslab simulate configs/lenses/noa61_cartesian_50_20.toml --out artifacts/my_cartesian
uv run lenslab excitation artifacts/my_cartesian
uv run lenslab render artifacts/my_cartesian
uv run lenslab verify artifacts/my_cartesian --spatial
```

`excitation/hold-drive.csv` gives the final amplitude and phase for each row, numbered
from the bottom. All sectors of a row share that drive. `drive-program.csv` contains
the complete sample-and-hold schedule. `surface-and-load.csv` contains the unforced,
target and computed profiles, required perturbation and capillary/gravitational load.
`traction-balance.csv` compares shape load with solved acoustic traction after a
constant pressure gauge is removed.

The physical convention is

\[
v_{n,j}(z,t)=e_j(z)\operatorname{Re}\{w_je^{-i2\pi ft}\}
=e_j(z)|w_j|\cos(2\pi ft-\arg w_j),
\]

where \(e_j\) is the declared cosine taper and the normal points out of the liquid.
Amplitudes are **peak wall velocity**, not voltage or pressure. Peak displacement is
\(|w_j|/(2\pi f)\). Voltages require a measured loaded transducer response. The phase
origin is arbitrary; relative phases control interference. The constrained design
includes a pressure penalty and is not claimed to be unique or globally optimal.

## Verification and interpretation

Tests compare the paper's four real/virtual examples against its parameterization,
an independently transcribed quartic and vector Snell tracing. Additional checks cover
reciprocity, both collimation limits, the original hyperboloid and fixed conjugate
positions during fluid evolution. Computed fluid surfaces undergo separate optical
evaluation and fixed-drive spatial refinement.

Stigmatism concerns **one axial pair at one wavelength**. It does not alone imply
off-axis coma correction, achromatism or an aplanatic singlet. A test checks the
nonconstant single-diopter sine ratio rather than asserting the Abbe sine condition
from a good axial focus. The paper's two surfaces provide additional design freedom.

The material, flow, heating, streaming and curing limitations in
[`model.md`](model.md) still apply. Numerical agreement does not calibrate a physical
fabrication process.

## Joint stationary design and stability

`lenslab stationary` provides the direct conjugates-to-holding-excitation route:

```bash
uv run lenslab stationary configs/lenses/noa61_cartesian_50_20_stationary.toml --out artifacts/my_stationary --stability
uv run lenslab render artifacts/my_stationary
```

The algorithm alternates array fitting on the current cavity and a nonlinear
fixed-volume capillary solve. Its optical and capillary linearization is evaluated
at the current surface, including the current ray error. This avoids treating a
force fit around the ideal target as the achieved optical surface. A separate
fixed-drive coupled solve checks the final state. Design iterations are explicitly
not fluid times; stationary outputs are stored in `stationary.npz`, and no approach
trajectory is invented for their visualization.

For the documented optimization branch, add
`--seed configs/initialization/cartesian_50_20.json --starts 1`. This is a recorded
initial guess; the optimizer still solves for the final drive. Run
`uv run lenslab verify artifacts/my_stationary` before rendering to include the
fixed-drive refinement comparison. A declared 0.1 nm full-aperture area-RMS
capillary-compliance correction controls the stationary algebraic solve. Meeting
this tolerance does not establish spatial convergence or dynamical stability.

The 1.35 MHz, 0.16 m/s design envelope used in the first transient did not give a
stigmatic result for the −50/+20 mm pair: the observed final ray RMS was 48.4 µm.
Many drives saturated their conservative real/imaginary component bounds. This is
an observed outcome of that search, not a proof that all drives within a circular
amplitude envelope are infeasible.

The separate stationary experiment uses 1.8 MHz and a provisional 1.28 m/s wall-speed
envelope. This is a changed design assumption, not a measured transducer capability.
It retains the same sixteen coherent rows, chamber, liquid and conjugates.
Its holding amplitudes/phases are in
[`hold-drive.csv`](../artifacts/studies/S01-single-interface/cartesian-50-20-stationary-p4/hold-drive.csv), with exact
conventions in the adjacent `excitation-definition.json`. The report contains
optical performance, pressure, power and the stability result; the first transient
and the stationary result must not be confused.

The nominal P4 design gives 0.389 µm geometric ray RMS, but independently re-solving
the same drive on a finer mesh with more surface modes gives **8.036 µm**. This
invalidates a submicrometre accuracy claim. The nominal stability calculation also
finds one growing mode. The excitation is therefore a **preliminary stationary
candidate**, not a validated stable stigmatic diopter. See
[`cartesian-results.md`](cartesian-results.md) for the complete evidence and open
questions. Neither acoustic feasibility for all conjugates nor a unique drive is
implied by the optical construction.

For a stationary surface, continuous fixed-drive linearization gives

\[
\dot{\delta\xi}=\frac{1}{\tau_c}M
\left[\frac{\partial(Q^T F_{\rm ac})}{\partial\xi}-Q^T H_E Q\right]\delta\xi.
\]

The acoustic shape derivative is calculated by central finite differences of full
curved-cavity solves. Positive real eigenvalues identify local growth with constant
drives, even if the equilibrium focuses well. This test includes acoustic feedback
through geometry; it does not freeze the radiation force during a hypothetical time
step. It is a local creeping-flow prediction, not a stability guarantee for a real
device. Large growth rates especially require checking fluid inertia, real sensing
and control bandwidth. An unstable stationary solution is an optical/control target,
not a lens demonstrated to hold its shape unattended.
