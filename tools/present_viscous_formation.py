"""Build a source notebook for saved, forward-computed ideal-load trajectories."""

from pathlib import Path

import nbformat as nbf

root = Path(__file__).resolve().parents[1]
target = root / "notebooks/08_prescribed_load_formation.ipynb"
if target.exists():
    raise FileExistsError("Archive the existing notebook before replacing it")
notebook = nbf.v4.new_notebook()
notebook.cells = [
    nbf.v4.new_markdown_cell("""# Formation under a prescribed full-interface mean load

This isolates the **fluid mechanics before emitter synthesis**. The pressure
map is calculated from continuous target curvature and gravity, then held fixed
while a three-fluid Stokes solver evolves both interfaces from flat.

**Scope:** axisymmetric creeping flow, moving faceted fluid mesh, nonlinear
capillarity, gravity, phasewise incompressibility, continuous velocity and
pressure jumps, no-slip walls and pinned volume-conserving surface splines.
All three viscosities are a declared **hypothetical 5 Pa s**, not measured
properties of a qualified optical-fluid triplet. No inertia, acoustic field,
streaming, heating or 3D disturbance simulation is included. Startup is the
Stokes limit, not a resolved inertial switch-on.

The animation uses saved numerical states. It is **not target interpolation**.
The depicted cylinder and its two interfaces are the ideal mechanical test;
there are no emitter components in this test. This is not an acoustic success
or an experimental 10 nm claim.
"""),
    nbf.v4.new_code_cell("""from pathlib import Path
import json
import numpy as np
from html import escape
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display, Image
from acoustic_freeform.apparatus.config import DualConfig
from acoustic_freeform.mechanics.surface import DualSurface
from acoustic_freeform.optics.stigmatic import stigmatic_pair
from acoustic_freeform.optics.raytrace import trace_pair

ROOT = Path.cwd()
RUN = ROOT / 'artifacts/studies/S03-stigmatic-target-ideal-load/ideal-load-2026-09-23/viscous-fluid-refined'
inputs = json.loads((RUN / 'config.json').read_text())
cfg = DualConfig(**inputs['apparatus'])
space = DualSurface(cfg)
targets = stigmatic_pair(cfg, inputs['indices'], inputs['stigmatic_z_m'], inputs['vertex_displacement_m'])
saved = np.load(RUN / 'trajectory.npz')
report = json.loads((RUN / 'validation.json').read_text())
assert report['completed']
assert np.isclose(saved['time_s'][-1], inputs['duration_s'])
def table(rows):
    header = '<tr>'+''.join('<th>'+escape(str(k))+'</th>' for k in rows[0])+'</tr>'
    body = ''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row.values())+'</tr>' for row in rows)
    return HTML('<table>'+header+body+'</table>')
display(table([{'quantity': k, 'value': report[k]} for k in (
    'maximum_reynolds_radius_estimate', 'viscous_diffusion_radius_time_s',
    'final_max_geometric_spot_radius_m', 'final_all_rays_transmitted')]))
display(Image(filename=str(ROOT / 'artifacts/studies/S03-stigmatic-target-ideal-load/ideal-load-2026-09-23/stigmatic-c2/required-load.png')))
"""),
    nbf.v4.new_markdown_cell("""## Fixed-load convergence

Each run uses the same continuous load and material scenario. Differences in
formation time must be distinguished from equilibrium errors. Maximum spot
radius is measured about the prescribed axis at the fixed detector; a frame
with any lost rays fails even if the surviving spot is small. Heights are
sampled rather than interval-certified.
"""),
    nbf.v4.new_code_cell("""comparison = []
for name in ('viscous-base', 'viscous-time-refined', 'viscous-fluid-refined'):
    directory = RUN.parent / name
    result = json.loads((directory / 'validation.json').read_text())
    setting = json.loads((directory / 'config.json').read_text())
    last = result['observations'][-1]
    comparison.append({'run': name, 'dt [s]': setting['step_s'],
        'radial cells': setting['fluid_radial_cells'],
        'front max [nm]': last['sampled_pupil_error_m'][0]*1e9,
        'back max [nm]': last['sampled_pupil_error_m'][1]*1e9,
        'max spot [µm]': result['final_max_geometric_spot_radius_m']*1e6,
        'all final rays': result['final_all_rays_transmitted']})
display(table(comparison))
"""),
    nbf.v4.new_code_cell("""# Retrace each displayed SAVED state; do not aim rays at its moving surface.
indices = np.unique(np.linspace(0, len(saved['time_s'])-1, 81).astype(int))
times = saved['time_s'][indices]
states = saved['coefficients_m'][indices]
r = np.linspace(0, cfg.radius_m, 501)
pupil = r <= cfg.clear_radius_m
target_h = np.array([t.evaluate(r) for t in targets])
h = np.array([[space.evaluate(face, r) for face in q] for q in states])
errors = np.max(abs(h[:,:,pupil]-target_h[:,pupil]), axis=2)
ray_r = np.linspace(0, cfg.clear_radius_m, 401)
launch = cfg.levels_m[1]+targets[0].evaluate(ray_r)
traces = [trace_pair(space, q, inputs['indices'], inputs['stigmatic_z_m'][0],
    inputs['stigmatic_z_m'][2], launch_radius_m=ray_r, launch_height_m=launch) for q in states]
spots = np.array([t['max_radius_m'] for t in traces])

fig = plt.figure(figsize=(14, 8), constrained_layout=True)
grid = fig.add_gridspec(2, 3)
setup = fig.add_subplot(grid[:,0], projection='3d')
profiles = fig.add_subplot(grid[0,1])
spot = fig.add_subplot(grid[0,2])
err = fig.add_subplot(grid[1,1])
spot_history = fig.add_subplot(grid[1,2])
theta = np.linspace(0, 2*np.pi, 41)
rr = np.linspace(0, cfg.radius_m, 19)
X, Y = rr[:,None]*np.cos(theta)*1e3, rr[:,None]*np.sin(theta)*1e3
for z in (cfg.levels_m[0], cfg.levels_m[-1]):
    setup.plot(cfg.radius_m*1e3*np.cos(theta), cfg.radius_m*1e3*np.sin(theta),
               np.full_like(theta,z*1e3), color='gray', alpha=.5)
for angle in theta[::5]:
    setup.plot([cfg.radius_m*1e3*np.cos(angle)]*2,
               [cfg.radius_m*1e3*np.sin(angle)]*2,
               np.array(cfg.levels_m)[[0,-1]]*1e3, color='gray', alpha=.3)
setup.set(xlim=(-4,4), ylim=(-4,4), zlim=(-3,3), xlabel='x [mm]', ylabel='y [mm]',
          zlabel='z [mm]', title='Ideal-load cylinder; no emitters')
setup.set_box_aspect((8,8,6))
colors = ['tab:blue', 'tab:orange']
lines = []
for j in range(2):
    profiles.plot(r*1e3,(cfg.levels_m[j+1]+target_h[j])*1e3, '--', color=colors[j])
    lines.append(profiles.plot([],[], color=colors[j], label=['Front / lower','Back / upper'][j])[0])
profiles.set(xlim=(0,4), ylim=(-1.25,1.2), xlabel='Radius [mm]', ylabel='z [mm]',
             title='Both full profiles: dashed = target')
profiles.legend()
for j in range(2):
    err.semilogy(times, np.maximum(errors[:,j]*1e9, 1e-6), color=colors[j])
err.axhline(10, color='green', linestyle='--')
err.set(xlabel='Physical time [s]', ylabel='Max pupil error [nm]', title='Sampled height error; log scale')
spot_history.semilogy(times, np.maximum(spots*1e6, 1e-6))
spot_history.axhline(1, color='green', linestyle='--')
spot_history.set(xlabel='Physical time [s]', ylabel='Max radius [µm]', title='Surviving rays; losses separately flagged')
cursors = [ax.axvline(0, color='black', linestyle=':') for ax in (err,spot_history)]
points = spot.scatter([],[],s=6)
spot.add_patch(plt.Circle((0,0),1,fill=False,color='green',linestyle='--'))
spot.set(xlabel='Detector x [µm]', ylabel='Detector y [µm]', aspect='equal')
wireframes = []
label = fig.suptitle('')

def update(k):
    for artist in wireframes:
        artist.remove()
    wireframes.clear()
    for j in range(2):
        heights = cfg.levels_m[j+1]+space.evaluate(states[k,j],rr)
        wireframes.append(setup.plot_wireframe(X,Y,np.broadcast_to(heights[:,None]*1e3,X.shape),
                                              color=colors[j], linewidth=.5))
        lines[j].set_data(r*1e3,(cfg.levels_m[j+1]+h[k,j])*1e3)
    points.set_offsets(traces[k]['spots_m']*1e6)
    extent = max(1.2, float(spots[k]*1e6)*1.1)
    spot.set_xlim(-extent,extent)
    spot.set_ylim(-extent,extent)
    lost = len(ray_r)-int(np.sum(traces[k]['transmitted']))
    spot.set_title(f'Spot: ADAPTIVE display scale; lost rays: {lost}')
    for cursor in cursors:
        cursor.set_xdata([times[k],times[k]])
    passed = max(errors[k]) <= 1e-8 and spots[k] <= 1e-6 and lost == 0
    label.set_text(f'Physical time {times[k]:.3f} s — ideal Stokes model — sampled gate: '
                   + ('PASS' if passed else 'NOT REACHED'))
    return []

animation = FuncAnimation(fig, update, frames=len(times), interval=100, blit=False)
plt.rcParams['animation.embed_limit'] = 60
embedded = animation.to_jshtml()
plt.close(fig)
display(HTML(embedded))
"""),
    nbf.v4.new_markdown_cell("""## Interpretation

Playback is accelerated: displayed physical time comes from saved time steps.
The setup has equal physical scale; profile axes have unequal scales. The spot
panel changes its **display extent**, not detector position, illumination or
acceptance radius. Intermediate lost rays remain failures.

Reaching the sampled gate here means formation in a declared ideal-load Stokes
discretization. It does not show that emitters can generate this load or its
stabilizing response, or that real materials stay within 10 nm. Full inertia,
non-axisymmetric disturbances, material characterization and acoustic-induced
streaming/heating remain separate checks. See `docs/ideal-load-first.md` and
the canonical manuscript's viscous-load variational section.
"""),
]
nbf.write(notebook, target)
print(target)
