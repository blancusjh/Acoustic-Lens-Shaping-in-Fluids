# %% [markdown]
# # Acoustic freeform lab — first forward reference
#
# **Research endpoint:** shape a liquid resin interface using a coherent array,
# eventually preserve it by curing, and assess the resulting optic. This notebook
# implements and interrogates the first small-deformation forward reference.
# It does not model resin chemistry, a finite lens, or optical performance yet.
#
# The two examples are a water–air wave check and an idealized viscous liquid pair.
# Neither is a fitted reproduction of a published experiment. Their purpose is
# to establish transparent equations, independent checks, and useful visual tools.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from acoustic_freeform.config import load_experiment
from acoustic_freeform.simulation import load_result, run

ROOT = Path.cwd() if (Path.cwd()/"src").is_dir() else Path.cwd().parent
config = load_experiment(ROOT/"examples/water_air.toml")
run_dir = ROOT/"runs/water_air_v1"
result = load_result(run_dir) if (run_dir/"result.npz").exists() else run(config)
print(config.description)
print(config.material_status)
print(f"Array: {config.array.nx} × {config.array.ny} individually weighted pressure patches")
print(f"Box: {config.grid.lx_m*1000:g} mm; observation: {config.timing.end_s*1000:g} ms")

# %% [markdown]
# ## 1. What the apparatus means here
#
# A 16 × 16 array lies at $z=-H$, beneath a nominal interface at $z=0$.
# Each square patch has a complex outgoing pressure weight $w_j$.
# The example derives these weights from two displaced foci; arbitrary complex
# element weights can also be supplied to `run(config, element_weights=...)`.
# This exposes all 256 controls without claiming arbitrary shapes are reachable.
#
# **Source convention:** pressure phasors are **peak amplitudes**, with physical
# pressure $p'(\mathbf{x},t)=\Re[P(\mathbf{x})e^{-i\omega t}]$.
# Electrical power, piezo displacement, and total face velocity are different
# inputs and require a transducer model. The source plane absorbs returning sound.
# It is not a no-slip lower wall in the hydrodynamic model: both slow fluids are
# deep half-spaces. Thus $H$ is the acoustic gap, not the liquid-layer thickness.
#
# For source area $A=L_xL_y$ and patch width $d$, exact Fourier coefficients are
#
# $$\widehat P_s(\mathbf{k})=\frac{p_0 d^2}{A}
# \operatorname{sinc}(k_xd/2)\operatorname{sinc}(k_yd/2)
# \sum_j w_j e^{-i\mathbf{k}\cdot\mathbf{x}_j},$$
#
# where $\operatorname{sinc}(u)=\sin(u)/u$. No pixelated aperture mask is used.

# %%
weights = result.element_weights[0]
fig, ax = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
im = ax[0].imshow(np.angle(weights), origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
fig.colorbar(im, ax=ax[0], label="phase [rad]")
im = ax[1].imshow(np.abs(weights), origin="lower", cmap="viridis", vmin=0, vmax=1)
fig.colorbar(im, ax=ax[1], label="pressure amplitude / p₀")
for a, title in zip(ax, ["Element phases", "Element amplitudes"], strict=True):
    a.set(xlabel="column", ylabel="row", title=title)
plt.show()

# %% [markdown]
# ## 2. Acoustic propagation and reflection
#
# Let $k_{zi}=\sqrt{(\omega/c_i)^2-k_x^2-k_y^2}$ with nonnegative imaginary part.
# For normal velocity admittance $Y_i=k_{zi}/(\omega\rho_i)$,
#
# $$R=\frac{Y_1-Y_2}{Y_1+Y_2},\qquad T=1+R,\qquad
# \widehat P_I=\widehat P_s e^{ik_{z1}H}.$$
#
# Pressure and normal acoustic velocity are continuous at the interface.
# For the lower fluid, $P=P_I+P_R$ and
# $V_z=k_{z1}(P_I-P_R)/(\omega\rho_1)$; the reflected velocity has the opposite
# normal sign. Transverse components follow the pressure gradient.
#
# The acoustic interface stays flat in this reference. The default retains only
# propagating incident modes; evanescent source terms decay strongly over this gap.
# The independent Rayleigh-integral comparison in notebook 02 measures the combined
# effect of this omission and periodic replicas at prespecified pressure probes.
#
# ## 3. Radiation traction is not just surface pressure squared
#
# For a flat normal $\mathbf{n}=\mathbf{e}_z$, define the normal momentum flux
#
# $$M_{nn,i}=\frac{|P_i|^2}{4\rho_i c_i^2}
# +\frac{\rho_i}{4}\left(|V_{zi}|^2-|V_{xi}|^2-|V_{yi}|^2\right),\qquad
# \Pi=M_{nn,1}-M_{nn,2}.$$
#
# Positive $\Pi$ drives upward displacement. This is the peak-phasor form of the
# radiation tensor in [Chesneau et al. (2022), Eqs. 9–14](https://doi.org/10.1103/PhysRevE.106.065104).
# For a pressure-release limit, the total pressure at the surface approaches zero,
# but the acoustic velocity and radiation force do not. A pressure-only shortcut
# therefore fails even in a simple limiting case.
#
# Coherent beams satisfy $P=\sum_b a_b(t)P_b$, so
# $\Pi=\sum_{bc}a_b(t)a_c(t)K_{bc}$. The cross terms are retained. These envelopes
# are real; all fixed relative acoustic phases are in the complex beam weights.

# %%
peak_drive = int(np.argmax(np.sum(result.amplitudes**2, axis=1)))
solution = result.acoustic_model.solve(
    result.pressure_array.spectrum(result.combined_weights(peak_drive))
)
powers = solution.powers_w
print("Integrated acoustic powers [W]:", powers)
print("Relative flux balance:", abs(powers["incident"]-powers["reflected"]-powers["transmitted"])
      / powers["incident"])
g = result.grid
crop = np.flatnonzero(np.abs(g.x) <= .006)
fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
im = ax.pcolormesh(g.x[crop]*1000, g.y[crop]*1000,
                   result.pressure(peak_drive)[np.ix_(crop, crop)], shading="auto", cmap="magma")
fig.colorbar(im, ax=ax, label="upward radiation traction [Pa]")
ax.set(xlabel="x [mm]", ylabel="y [mm]", aspect="equal", title="Non-axisymmetric interface forcing")
plt.show()

# %% [markdown]
# ## 4. Two explicit hydrodynamic limits
#
# For each nonzero transverse wavenumber $k=|\mathbf{k}|$, the linear restoring
# stiffness per area is $S_k=\sigma k^2+(\rho_1-\rho_2)g$.
# The Fourier zero mode is held at zero: the periodic cell conserves liquid volume.
# A uniform pressure reaction balances the mean acoustic traction.
#
# **Weakly damped liquid under a light gas:**
#
# $$m_k\ddot{\hat h}+2\gamma_km_k\dot{\hat h}+S_k\hat h=\hat\Pi,
# \qquad m_k=\frac{\rho_1+\rho_2}{k},\quad\gamma_k=2\nu_1k^2.$$
#
# This has the inviscid capillary–gravity dispersion
# $\omega_k^2=[(\rho_1-\rho_2)gk+\sigma k^3]/(\rho_1+\rho_2)$.
# The damping is a weak-viscosity approximation; gas drag and the full transient
# viscous boundary layer are omitted. It must not be reused as a general viscous
# two-liquid model. [Denner (2016)](https://doi.org/10.1103/PhysRevE.94.023110)
# examines the limitations of customary damping descriptions when viscosity matters.
#
# **Deep two-fluid Stokes limit:**
#
# $$2(\mu_1+\mu_2)k\dot{\hat h}+S_k\hat h=\hat\Pi.$$
#
# One way to derive its mobility is to solve the biharmonic Stokes problem with
# normal interface speed $W=\dot{\hat h}$. The decaying solutions are
# $w_1=(1-kz)e^{kz}W$ below and $w_2=(1+kz)e^{-kz}W$ above. Incompressibility
# gives $\mathbf{u}_{\parallel,i}=-i\mathbf{k}z e^{-k|z|}W$.
# The normal viscous traction jump is $2(\mu_1+\mu_2)kW$, giving the equation above.
# This is the zero-inertia limit, not a fitted interpolation into the wave model.
#
# Constant forcing is integrated exactly mode by mode. Smooth pulse envelopes are
# sampled at each step midpoint, so forcing integration is second order.
# The saved height/normal-speed arrays are float32; integration and diagnostics use
# float64. This distinction is recorded in each report.

# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
t_ms = result.time_s*1000
axes[0].plot(t_ms, result.amplitudes[:, 0], color="#5e3c99")
axes[0].set_ylabel("Drive amplitude / p₀")
axes[1].plot(t_ms, result.metrics["max_height_m"]*1e6, label="highest point")
axes[1].plot(t_ms, result.metrics["min_height_m"]*1e6, label="lowest point")
axes[1].set(xlabel="time [ms]", ylabel="surface height [µm]")
axes[1].legend()
plt.show()

# %% [markdown]
# ## 5. Same instantaneous input, different surface
#
# At 10 ms and 40 ms the array has the same weights and full pulse amplitude.
# Both times are 8 ms after a pulse began. The second surface still contains
# motion excited by the first pulse. This is hydrodynamic state dependence.
# In this *linear* reference it can be represented by a convolution or state-space
# model; this observation alone does not establish a hard inverse problem.
# No chemical memory, viscoelastic memory, thermal history, or nonlinear hysteresis
# has been implemented. Those require separate physical models and evidence.

# %%
i1 = int(np.argmin(abs(result.time_s-.010)))
i2 = int(np.argmin(abs(result.time_s-.040)))
assert np.array_equal(result.amplitudes[i1], result.amplitudes[i2])
assert np.allclose(result.pressure(i1), result.pressure(i2))
h1, h2 = [result.height_m[i][np.ix_(crop, crop)]*1e6 for i in (i1, i2)]
print(f"Same drive amplitudes: {result.amplitudes[i1]}")
print(f"Difference in central surface: RMS={np.sqrt(np.mean((h2-h1)**2)):.5f} µm")
fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
bound = max(np.max(np.abs(h1)), np.max(np.abs(h2)), np.max(np.abs(h2-h1)))
for ax, h, title in zip(axes, [h1, h2, h2-h1], ["10 ms", "40 ms", "40 ms minus 10 ms"], strict=True):
    im = ax.pcolormesh(g.x[crop]*1000, g.y[crop]*1000, h, cmap="coolwarm",
                       vmin=-bound, vmax=bound, shading="auto")
    ax.set(xlabel="x [mm]", ylabel="y [mm]", aspect="equal", title=title)
fig.colorbar(im, ax=axes, label="height [µm]")
plt.show()

# %% [markdown]
# ## 6. PyVista surface and physical field exports
#
# Below is a newly rendered surface, not a sketch. Its horizontal geometry is in
# millimetres, height color in micrometres, and vertical geometry exaggerated by
# 1800 for legibility. The VTK files exported by `lenslab render` instead contain
# **metres, pascals and metres/second, with no geometric exaggeration**.
#
# For the live time slider, run `uv run lenslab view runs/water_air_v1` in the repo.
# `visuals/explore.html` offers camera interaction for a static time snapshot;
# `visuals/evolution.mp4` shows time evolution. The pressure-array panel in the
# dashboard is a fixed peak-drive reference, while the other panels evolve.

# %%
import pyvista as pv
from IPython.display import display
from PIL import Image

frame = int(np.argmax(result.metrics["rms_height_m"]))
h = result.height_m[frame][np.ix_(crop, crop)]
xx, yy = np.meshgrid(g.x[crop]*1000, g.y[crop]*1000)
surface = pv.StructuredGrid(xx.T, yy.T, (h*1000*1800).T)
surface["height [um]"] = h.ravel()*1e6
plotter = pv.Plotter(off_screen=True, window_size=(1000, 650))
plotter.set_background("#101e30")
plotter.add_mesh(surface, scalars="height [um]", cmap="coolwarm",
                 scalar_bar_args={"color": "white"})
plotter.add_text("Simulated surface | vertical geometry x1800", color="white", font_size=13)
plotter.camera_position = [(20, -28, 24), (0, 0, 0), (0, 0, 1)]
plotter.show(auto_close=False, interactive=False)
preview = plotter.screenshot(return_img=True)
plotter.close()
display(Image.fromarray(preview))

# %% [markdown]
# ## 7. What still separates this reference from a lens experiment
#
# The calculation supports arbitrary *inputs* and non-axisymmetric surfaces.
# It does not guarantee arbitrary *target shapes*. Finite aperture, diffraction,
# traction sign, capillary smoothing, fluid volume and stability restrict reachability.
#
# The first decisive extensions are finite fluid geometry and wave scattering
# from the evolving interface. [Bertin et al. (2012)](https://doi.org/10.1103/PhysRevLett.109.244304)
# and [Chesneau et al. (2022)](https://doi.org/10.1103/PhysRevE.106.065104)
# motivate the latter for larger deformations. Our tiny values of slope and $k h$
# only support using this as an initial small-deformation check; they are not an
# experimental error bound.
#
# [Sisombat et al. (2023)](https://doi.org/10.1038/s41598-023-39464-0) is a useful
# experimental anchor for pulse-driven interface deformation, with a different rig.
# [Gu et al. (2020)](https://doi.org/10.1021/acsnano.0c03754) motivates spatial
# patterning and attention to streaming, but does not establish an optical resin process.
# [Elgarisi et al. (2021)](https://doi.org/10.1364/OPTICA.438763) and
# [Na et al. (2024)](https://doi.org/10.1145/3680528.3687584) connect fluidic shaping
# to cured optics. Their methods do not validate this acoustic simulator.
#
# Before adding curing, choose a resin/bath pair with measured viscosity,
# density, acoustic speed/attenuation and surface tension. Then determine which
# omitted mechanisms materially affect the experimental times and length scales.
