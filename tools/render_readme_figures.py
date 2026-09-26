"""Render the README setup figures and formation animation from saved local states.

Every drawn part is a declared apparatus component or a saved numerical state:
no target interpolation, no fitted shapes. Geometry is at equal scale unless a
panel says otherwise. Inputs are hashed into figures/provenance.json.

    uv run python tools/render_readme_figures.py [--only single|pair|formation]
"""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib import colors as mcolors
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
S01 = ROOT / "artifacts/studies/S01-single-interface"
S04 = ROOT / "artifacts/studies/S04-emitter-synthesis/noa61-emitter-2026-09-23"
FORMATION_RUN = S01 / "formation-notebook-2026-09-23/dt000625"
FORMATION_ANALYSIS = S01 / "notebooks/05_formation_and_spots/formation-analysis-si.npz"
PAIR_RUN = S04 / "verify-448-c4-clear-spot/design104"
PAIR_SYNTHESIS = S04 / "force-448-c4-clear-spot"

RESIN = "#e7a83a"
WATER = "#6fb3e0"
GLASS = "#c9d6de"
WALL = "#9aa7b0"
RAY = "#c0392b"
INK = "#1f2a33"
MUTED = "#5d6b75"
SOURCE_MAP = "plasma"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    }
)

inputs_used = {}


def read(path):
    path = Path(path)
    inputs_used[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def refract(d, n, n1, n2):
    """Vector Snell law; n is the unit normal on the incident side's exit direction."""
    cos_i = -np.sum(d * n, axis=-1, keepdims=True)
    flip = cos_i < 0
    n = np.where(flip, -n, n)
    cos_i = np.abs(cos_i)
    eta = n1 / n2
    k = 1 - eta**2 * (1 - cos_i**2)
    out = eta * d + (eta * cos_i - np.sqrt(np.maximum(k, 0))) * n
    return out / np.linalg.norm(out, axis=-1, keepdims=True), k[..., 0] >= 0


def meridional_rays(radius, height, z_object, launch_r, n1, n2, z_detector):
    """Trace meridional rays from an axial object through one saved graph surface.

    Each ray aims at the saved surface point above its launch radius and is
    refined to the actual intersection; this is a display trace, checked
    against the saved analysis spot size in the caller.
    """
    slope = np.gradient(height, radius)
    h = lambda r: np.interp(np.abs(r), radius, height)
    s = lambda r: np.sign(r) * np.interp(np.abs(r), radius, slope)
    r_hit = launch_r.copy()
    for _ in range(60):
        d = np.c_[r_hit, h(r_hit) - z_object]
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        t = (h(r_hit) - z_object) / d[:, 1]
        r_hit = t * d[:, 0]
    hit = np.c_[r_hit, h(r_hit)]
    normal = np.c_[-s(r_hit), np.ones_like(r_hit)]
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    out, ok = refract(d, normal, n1, n2)
    t_det = (z_detector - hit[:, 1]) / out[:, 1]
    detector = hit + t_det[:, None] * out
    return d, hit, detector, ok


# --------------------------------------------------------------------------- #
# Single-interface cell
# --------------------------------------------------------------------------- #


def single_interface_state():
    from acoustic_freeform.single_interface.config import LensConfig
    from acoustic_freeform.single_interface.surface import SurfaceSpace

    raw = json.loads(read(FORMATION_RUN / "configuration.json").read_text())
    names = {f.name for f in dataclasses.fields(LensConfig)}
    cfg = LensConfig(**{k: v for k, v in raw.items() if k in names})
    trajectory = np.load(read(FORMATION_RUN / "trajectory.npz"))
    analysis = np.load(read(FORMATION_ANALYSIS))
    report = json.loads(read(FORMATION_RUN / "report.json").read_text())
    return cfg, SurfaceSpace(cfg), trajectory, analysis, report


def autocrop(image, margin=12):
    """Trim uniform white borders of a rendered RGB(A) image."""
    ink = np.where(image[..., :3].min(axis=-1) < 0.985)
    top, bottom = max(ink[0].min() - margin, 0), ink[0].max() + margin
    left, right = max(ink[1].min() - margin, 0), ink[1].max() + margin
    return image[top:bottom, left:right]


def exaggeration(ax):
    """Vertical stretch of an axis relative to equal scale, from its drawn extent."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    box = ax.get_window_extent()
    return ((box.height / (y1 - y0)) / (box.width / (x1 - x0)))


def cut_front_quadrant(mesh):
    """Remove the x > 0, y < 0 quadrant so the interior faces the camera."""
    return mesh.clip_box((0, 1e3, -1e3, 0, -1e3, 1e3), invert=True)


def ray_tube(points_mm, radius=0.03):
    return pv.lines_from_points(np.asarray(points_mm)).tube(radius=radius, n_sides=12)


def render_single_3d(cfg, space, coefficients, drive, path, size=(1500, 1500)):
    from acoustic_freeform.single_interface.visualization import apparatus

    meshes, _, row_ids = apparatus(cfg, space, coefficients)
    p = pv.Plotter(off_screen=True, window_size=size)
    p.set_background("white")
    p.enable_anti_aliasing("ssaa")
    p.enable_lightkit()
    glass = {"smooth_shading": True, "specular": 0.7, "specular_power": 50}
    p.add_mesh(cut_front_quadrant(meshes["cylinder"]), color=GLASS, opacity=0.45, **glass)
    p.add_mesh(cut_front_quadrant(meshes["base"]), color="#b9d7ea", opacity=0.7, **glass)
    p.add_mesh(cut_front_quadrant(meshes["liquid"]), color=RESIN, opacity=0.30,
               smooth_shading=True)
    p.add_mesh(cut_front_quadrant(meshes["surface"]), color="#e39a2d", opacity=0.97,
               smooth_shading=True, specular=1.0, specular_power=80)
    p.add_mesh(cut_front_quadrant(meshes["rim"]), color="#4d565d", smooth_shading=True)
    array = meshes["array"]
    array.cell_data["amplitude"] = np.abs(drive)[row_ids]
    p.add_mesh(cut_front_quadrant(array), scalars="amplitude", cmap=SOURCE_MAP,
               clim=(0, np.abs(drive).max()), show_scalar_bar=False, specular=0.4)
    radius = np.linspace(0, cfg.radius_m, 801)
    height = cfg.radius_m * space.evaluate(coefficients, radius / cfg.radius_m)
    vertex = height[0]
    z_det = vertex + cfg.focal_distance_m
    launch = np.linspace(-cfg.clear_radius_m, cfg.clear_radius_m, 9)
    d, hit, det, _ = meridional_rays(radius, height, vertex + cfg.object_distance_m, launch,
                                     cfg.refractive_index, cfg.image_refractive_index, z_det)
    top = vertex + 0.006  # display the refracted rays for 6 mm, then stop
    for k in range(len(launch)):
        a = hit[k] + (-cfg.depth_m - hit[k, 1]) / d[k, 1] * d[k]
        out = det[k] - hit[k]
        b = hit[k] + (top - hit[k, 1]) / out[1] * out
        pts = np.array([a, hit[k], b]) * 1e3
        p.add_mesh(ray_tube(np.c_[pts[:, 0], np.zeros(3), pts[:, 1]]), color="#e0312b",
                   smooth_shading=True, ambient=0.55, diffuse=0.6)
    p.camera_position = [(21, -27, 12), (0, 0, -1.4), (0, 0, 1)]
    p.camera.view_angle = 24
    p.screenshot(path)
    p.close()
    return z_det


def section_frame(ax, cfg, drive, norm, cmap, mm=1e3):
    R, depth = cfg.radius_m * mm, cfg.depth_m * mm
    wall, base = cfg.wall_thickness_m * mm, cfg.base_thickness_m * mm
    for x in (-R - wall, R):
        ax.add_patch(Rectangle((x, -depth), wall, depth, color=GLASS, ec=WALL, lw=0.8))
    ax.add_patch(Rectangle((-R - wall, -depth - base), 2 * (R + wall), base, color="#cfe4f1",
                           ec=WALL, lw=0.8))
    pitch = depth / cfg.array_rows
    for row, amp in enumerate(np.abs(drive)):
        low = -depth + (row + 0.5 - 0.5 * cfg.element_fill) * pitch
        for x in (R + wall + 0.09 - 0.8 + 0.8, -R - wall - 0.035 - 0.25):
            ax.add_patch(Rectangle((x, low), 0.25, cfg.element_fill * pitch,
                                   color=cmap(norm(amp)), lw=0))
    return R, depth, wall, base


def single_interface_figure():
    cfg, space, trajectory, analysis, _report = single_interface_state()
    final = trajectory["coefficients"][-1]
    drive = trajectory["drive_m_s"][-1]
    render = FIGURES / "_single_render.png"
    z_det = render_single_3d(cfg, space, final, drive, render)
    mm = 1e3
    radius = analysis["radius_m"]
    h0, h1 = analysis["surface_height_m"][0], analysis["surface_height_m"][-1]
    target = analysis["target_height_m"]
    vertex = h1[0]
    z_obj = vertex + cfg.object_distance_m
    n1, n2 = cfg.refractive_index, cfg.image_refractive_index
    launch = np.linspace(-cfg.clear_radius_m, cfg.clear_radius_m, 13)
    d, hit, det, _ = meridional_rays(radius, h1, z_obj, launch, n1, n2, z_det)
    fine = np.linspace(-cfg.clear_radius_m, cfg.clear_radius_m, 401)
    *_, fine_det, _ = meridional_rays(radius, h1, z_obj, fine, n1, n2, z_det)
    display_spot = float(np.sqrt(np.mean(fine_det[:, 0] ** 2)))
    saved_spot = float(analysis["spot_rms_m"][-1])
    assert display_spot < 5e-6, "display trace disagrees with saved optics"
    norm = mcolors.Normalize(0, np.abs(drive).max())
    cmap = plt.get_cmap(SOURCE_MAP)

    fig = plt.figure(figsize=(17, 8.6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.0, 0.42], wspace=0.08)
    ax3 = fig.add_subplot(grid[0])
    ax3.imshow(autocrop(plt.imread(render)))
    ax3.set_axis_off()
    ax3.set_title("(a) Modeled cell · front quadrant removed · equal scale", loc="left",
                  color=INK, fontsize=12)
    note = {"fontsize": 9, "color": INK}
    # (b) section of the cell.
    ax = fig.add_subplot(grid[1])
    R, depth, wall, base = section_frame(ax, cfg, drive, norm, cmap)
    signed = np.r_[-radius[:0:-1], radius] * mm
    mirror = lambda h: np.r_[h[:0:-1], h] * mm
    ax.add_patch(Polygon(np.c_[np.r_[signed, R, -R], np.r_[mirror(h1), -depth, -depth]],
                         color=RESIN, alpha=0.30, lw=0))
    for k in range(len(launch)):
        a = hit[k] + (-cfg.depth_m - hit[k, 1]) / d[k, 1] * d[k]
        out = det[k] - hit[k]
        b = hit[k] + (vertex + 0.0014 - hit[k, 1]) / out[1] * out
        ax.plot(np.array([a[0], hit[k, 0], b[0]]) * mm, np.array([a[1], hit[k, 1], b[1]]) * mm,
                color=RAY, lw=0.7, alpha=0.8)
    ax.plot(signed, mirror(h1), color="#b86f0c", lw=2.2)
    ax.axvline(0, color=MUTED, lw=0.5, ls=(0, (6, 4)))
    ax.text(0, -3.3, f"NOA 61 resin, uncured\nn = {n1}, ρ = {cfg.density_kg_m3:.0f} kg/m³\n"
            f"μ = {cfg.viscosity_pa_s} Pa·s, σ = {cfg.surface_tension_n_m} N/m",
            ha="center", va="center", **note)
    ax.text(R - 0.3, vertex * mm + 0.9, "air, n = 1", ha="right", fontsize=9, color=MUTED)
    ax.text(-R - wall - 0.45, -depth / 2, f"{cfg.array_rows} coherent rows of ideal\n"
            f"wall-velocity sources\n({cfg.array_rows} × {cfg.array_sectors} sectors drawn)\n"
            f"{cfg.frequency_hz / 1e6:.1f} MHz, held command", ha="right", va="center", **note)
    ax.text(-R - wall - 0.45, -depth - base / 2, "base window", ha="right", va="center",
            fontsize=8.5, color=MUTED)
    ax.annotate("contact line\npinned at the rim", (-R, 0), (-R - wall - 0.45, 1.0), fontsize=8.5,
                color=MUTED, ha="right", va="center",
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6})
    arrow = {"arrowstyle": "<->", "color": MUTED, "lw": 0.8, "shrinkA": 0, "shrinkB": 0}
    y = -depth - base - 0.55
    ax.add_patch(FancyArrowPatch((-R, y), (R, y), **arrow))
    ax.text(0, y - 0.2, f"liquid diameter {2 * R:.0f} mm", ha="center", va="top", fontsize=8.5,
            color=MUTED)
    yc = vertex * mm + 1.75
    ax.add_patch(FancyArrowPatch((-cfg.clear_radius_m * mm, yc), (cfg.clear_radius_m * mm, yc),
                                 **arrow))
    ax.text(0, yc + 0.12, f"clear aperture {2 * cfg.clear_radius_m * mm:.0f} mm", ha="center",
            va="bottom", fontsize=8.5, color=MUTED)
    ax.add_patch(FancyArrowPatch((R + wall + 0.75, -depth), (R + wall + 0.75, 0), **arrow))
    ax.text(R + wall + 0.9, -depth / 2, f"{depth:.0f} mm", rotation=90, va="center",
            fontsize=8.5, color=MUTED)
    ax.set(xlim=(-R - 6.2, R + 2.0), ylim=(-depth - base - 1.5, vertex * mm + 5.6),
           aspect="equal", xlabel="x (mm)", ylabel="z above the rim (mm)")
    ax.set_title("(b) Meridional section at equal scale", loc="left", color=INK, fontsize=12)
    ax.set_yticks(range(-8, 3, 2))
    bar = fig.colorbar(plt.cm.ScalarMappable(norm, cmap), ax=ax, orientation="horizontal",
                       fraction=0.03, pad=0.09, aspect=40, shrink=0.7)
    bar.set_label("held wall-speed amplitude of each row (m/s)", color=INK, fontsize=9)
    # Vertex inset: the lift the sound produces.
    inset = ax.inset_axes([0.07, 0.75, 0.36, 0.15])
    window = radius <= 1.2e-3
    rr = np.r_[-radius[window][:0:-1], radius[window]] * mm
    cut = lambda h: np.r_[h[window][:0:-1], h[window]] * mm
    inset.plot(rr, cut(h0), color=MUTED, ls="--", lw=1.2, label="unforced, t = 0")
    inset.plot(rr, cut(target), color=INK, ls=":", lw=1.2,
               label="target (overlaps driven)")
    inset.plot(rr, cut(h1), color="#b86f0c", lw=2, label="driven, t = 0.2 s")
    inset.annotate("", (0, h1[0] * mm), (0, h0[0] * mm),
                   arrowprops={"arrowstyle": "->", "color": RAY, "lw": 1.2})
    inset.text(0.08, (h0[0] + h1[0]) / 2 * mm, f"+{(h1[0] - h0[0]) * 1e6:.0f} µm",
               color=RAY, fontsize=9, va="center")
    inset.tick_params(labelsize=7)
    inset.set_xlim(rr[0], rr[-1])
    inset.set_ylim(h0[0] * mm - 0.03, h1[0] * mm + 0.02)
    inset.legend(fontsize=6.5, frameon=False, loc="lower center", ncol=3,
                 bbox_to_anchor=(0.5, -0.62))
    fig.canvas.draw()
    inset.set_title(f"near the vertex · vertical ×{exaggeration(inset):.0f}", fontsize=8.5,
                    color=INK)
    # (c) the optical function.
    axo = fig.add_subplot(grid[2])
    section_frame(axo, cfg, drive, norm, cmap)
    axo.add_patch(Polygon(np.c_[np.r_[signed, R, -R], np.r_[mirror(h1), -depth, -depth]],
                          color=RESIN, alpha=0.30, lw=0))
    for k in range(len(launch)):
        a = hit[k] + (-cfg.depth_m - hit[k, 1]) / d[k, 1] * d[k]
        axo.plot(np.array([a[0], hit[k, 0], det[k, 0]]) * mm,
                 np.array([a[1], hit[k, 1], det[k, 1]]) * mm, color=RAY, lw=0.6, alpha=0.85)
    axo.plot(signed, mirror(h1), color="#b86f0c", lw=1.5)
    axo.plot(0, z_det * mm, "o", color=RAY, ms=4)
    axo.text(0, z_det * mm + 0.8, f"image point\n{cfg.focal_distance_m * mm:.0f} mm above vertex",
             ha="center", va="bottom", fontsize=8.5, color=INK)
    axo.text(0, (z_det + vertex) / 2 * mm, f"saved RMS\nspot\n{saved_spot * 1e6:.3f} µm",
             ha="center", va="center", fontsize=8.5, color=INK,
             bbox={"fc": "white", "ec": "none", "alpha": 0.85})
    axo.set(xlim=(-R - 1.5, R + 1.5), ylim=(-depth - base - 1.5, z_det * mm + 3.5),
            aspect="equal", xlabel="x (mm)")
    axo.set_title("(c) Focusing", loc="left", color=INK, fontsize=12)
    fig.text(0.01, 0.015, "Saved numerical state at t = 0.2 s of a fixed-command formation run "
             "(axisymmetric, isothermal; inertia, bulk absorption and viscous wall loss). "
             f"Rays: geometric, meridional, from an object {-cfg.object_distance_m * mm:.0f} mm "
             "below the vertex, prescribed inside the resin. Sources are ideal velocity boundaries, not calibrated "
             "transducers. Not an experiment.", fontsize=8.5, color=MUTED)
    out = FIGURES / "single-interface-cell.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    render.unlink()
    return {"figure": out.name, "display_trace_rms_spot_m": display_spot,
            "saved_rms_spot_m": saved_spot}


# --------------------------------------------------------------------------- #
# Two-face NOA 61 / water cell
# --------------------------------------------------------------------------- #


def pair_state():
    from acoustic_freeform.acoustics.helmholtz import DualAcoustics
    from acoustic_freeform.apparatus.config import DualConfig
    from acoustic_freeform.mechanics.surface import CartesianPatch, DualSurface
    from acoustic_freeform.optics.raytrace import trace_pair

    cfg = DualConfig(**json.loads(read(PAIR_RUN / "config.json").read_text()))
    settings = json.loads(read(PAIR_SYNTHESIS / "config.json").read_text())
    material = json.loads(read(ROOT / settings["material_reference"]).read_text())
    report = json.loads(read(PAIR_RUN / "validation.json").read_text())
    regions = json.loads(read(PAIR_RUN / "source-regions.json").read_text())["regions"]
    state = np.load(read(PAIR_RUN / "state.npz"))
    q, drive = state["coefficients_m"], state["source_velocity_m_s"]
    space = DualSurface(cfg)
    targets = [CartesianPatch(cfg, j, **face) for j, face in enumerate(settings["case"]["faces"])]
    z0, _, z2 = material["stigmatic_z_m"]
    launch = np.linspace(0, cfg.clear_radius_m, 4001)
    trace = trace_pair(space, q, material["indices"], z0, z2, launch_radius_m=launch,
                       launch_height_m=cfg.levels_m[1] + targets[0].evaluate(launch))
    np.testing.assert_allclose(trace["spots_m"], state["spots_m"], atol=1e-12, rtol=1e-7,
                               equal_nan=True)
    wave = DualAcoustics(cfg, space, linear_solver="static_condensed")
    response = wave.solve(q, drive)
    power = response.diagnostics(np.ones(1, complex))["source_power_w"]
    np.testing.assert_allclose(power, report["wave_diagnostics"]["source_power_w"], rtol=1e-6)
    xy = response.basis.mapping.geometry(wave.flat.p)[0] * cfg.radius_m
    field = {"xy": xy, "triangles": wave.flat.t.T,
             "pressure": response.solution[response.basis.nodal_dofs[0], 0], "power_w": power}
    return cfg, space, q, drive, regions, targets, material, report, trace, field


def revolve(radius, z, first=0.0, last=1.5 * np.pi, sectors=120):
    """Surface of revolution of a meridional polyline (radius, z), in the given units."""
    theta = np.linspace(first, last, sectors)
    x = np.outer(radius, np.cos(theta))
    y = np.outer(radius, np.sin(theta))
    return pv.StructuredGrid(x, y, np.repeat(np.asarray(z)[:, None], len(theta), 1))


def pair_ray_paths(cfg, space, q, material, targets, launch):
    """Object-side entry at the lower cap, both saved interfaces, then the image side."""
    from acoustic_freeform.optics.raytrace import trace_pair

    z0, _, z2 = material["stigmatic_z_m"]
    paths = []
    for rho in launch:
        rho = np.array([rho])
        tr = trace_pair(space, q, material["indices"], z0, z2, launch_radius_m=rho,
                        launch_height_m=cfg.levels_m[1] + targets[0].evaluate(rho))
        hits = tr["intersections_m"][:, 0, :]
        d = hits[0] - np.array([0.0, z0])
        entry = np.array([0.0, z0]) + (cfg.levels_m[0] - z0) / d[1] * d
        detector = np.array([tr["spots_m"][0, 0], z2])
        paths.append(np.vstack([entry, hits[0], hits[1], detector]))
    return np.array(paths)


def clip_path(path, z_top):
    """Cut the last (image-side) segment of a path at height z_top."""
    a, b = path[-2], path[-1]
    s = (z_top - a[1]) / (b[1] - a[1])
    return np.vstack([path[:-1], a + s * (b - a)])


def render_pair_3d(cfg, space, q, drive, regions, paths, path, size=(1500, 1400)):
    mm = 1e3
    _first, _last = 0.0, 1.5 * np.pi
    r = np.linspace(0, cfg.radius_m, 220)
    levels = np.asarray(cfg.levels_m)
    faces = [levels[j + 1] + space.evaluate(q[j], r) for j in range(2)]
    p = pv.Plotter(off_screen=True, window_size=size)
    p.set_background("white")
    p.enable_anti_aliasing("ssaa")
    p.enable_lightkit()
    turn = lambda m: m
    amp = np.abs(drive)
    cmap = plt.get_cmap(SOURCE_MAP)
    for reg in regions:
        a = amp[reg["channel"]]
        active = a > 1e-14
        colour = cmap(a / amp.max())[:3] if active else (0.80, 0.84, 0.87)
        if reg["boundary"] == "side":
            rr = np.full(2, cfg.radius_m * mm)
            zz = np.array([reg["z_min_m"], reg["z_max_m"]]) * mm
        else:
            rr = np.array([reg["r_min_m"], reg["r_max_m"]]) * mm
            zz = np.full(2, reg["z_min_m"] * mm)
        opacity = 1.0 if active else 0.25
        if reg["boundary"] == "top":
            opacity = min(opacity, 0.55)
        p.add_mesh(turn(revolve(rr, zz)), color=colour, opacity=opacity, smooth_shading=True,
                   specular=0.25)
    # Fluids: translucent volumes between the saved interfaces.
    bounds = [np.full_like(r, levels[0]), faces[0], faces[1], np.full_like(r, levels[3])]
    for j, colour in enumerate([WATER, RESIN, WATER]):
        low, high = bounds[j] * mm, bounds[j + 1] * mm
        rr = np.r_[r, r[::-1]] * mm
        zz = np.r_[high, low[::-1]]
        outline = revolve(rr, zz)
        p.add_mesh(turn(outline), color=colour, opacity=0.18 if j != 1 else 0.42,
                   smooth_shading=True)
        del outline
        # Cap both cut half-planes with this layer's meridional section.
        strip = np.stack([low, high])
        rho = np.broadcast_to(r * mm, strip.shape)
        for x, y in ((rho, np.zeros_like(rho)), (np.zeros_like(rho), -rho)):
            cap = pv.StructuredGrid(x[..., None].copy(), y[..., None].copy(),
                                    strip[..., None].copy())
            p.add_mesh(cap, color=colour, opacity=0.55 if j != 1 else 0.75,
                       smooth_shading=True, ambient=0.35)
    for j, z in enumerate(faces):
        p.add_mesh(turn(revolve(r * mm, z * mm, sectors=180)), color="#c9771a", opacity=0.6,
                   smooth_shading=True, specular=1.0, specular_power=90)
    # Rays lie in the two cut half-planes (theta = 0 and theta = 270 degrees).
    for ray in paths:
        clipped = clip_path(ray, levels[3] + 0.0035) * mm
        rho, z = np.abs(clipped[:, 0]), clipped[:, 1]
        for xyz in (np.c_[rho, np.zeros_like(rho), z], np.c_[np.zeros_like(rho), -rho, z]):
            xyz = xyz + np.array([0.02, -0.02, 0.0])  # just in front of the cap
            tube = pv.lines_from_points(xyz).tube(radius=0.016, n_sides=12)
            p.add_mesh(tube, color="#e0312b", ambient=0.55, diffuse=0.6, smooth_shading=True)
    p.camera_position = [(17, -17, 9.0), (0, 0, 0.2), (0, 0, 1)]
    p.camera.view_angle = 26
    p.screenshot(path)
    p.close()


def pair_figure():
    cfg, space, q, drive, regions, targets, material, report, trace, field = pair_state()
    mm = 1e3
    launch = np.linspace(-cfg.clear_radius_m, cfg.clear_radius_m, 11)
    paths = pair_ray_paths(cfg, space, q, material, targets, np.abs(launch))
    paths[:, :, 0] *= np.sign(launch + 1e-30)[:, None]
    render = FIGURES / "_pair_render.png"
    render_pair_3d(cfg, space, q, drive, regions, paths[launch > 0][::1], render)
    levels = np.asarray(cfg.levels_m) * mm
    R = cfg.radius_m * mm
    r = np.linspace(0, cfg.radius_m, 1601)
    faces = [cfg.levels_m[j + 1] + space.evaluate(q[j], r) for j in range(2)]
    goal = [cfg.levels_m[j + 1] + targets[j].evaluate(r) for j in range(2)]
    signed = np.r_[-r[:0:-1], r] * mm
    mirror = lambda h: np.r_[h[:0:-1], h] * mm
    amp = np.abs(drive)
    norm = mcolors.Normalize(0, amp.max())
    cmap = plt.get_cmap(SOURCE_MAP)
    n = material["indices"]
    rho = cfg.density_kg_m3

    fig = plt.figure(figsize=(17.5, 8.4))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 0.78], wspace=0.10)
    ax3 = fig.add_subplot(grid[0])
    ax3.imshow(autocrop(plt.imread(render)))
    ax3.set_axis_off()
    ax3.set_title("(a) Three-fluid cell · quadrant removed · equal scale",
                  loc="left", color=INK, fontsize=12)
    note = {"fontsize": 9, "color": INK}

    def boundary(ax, width=0.22):
        for reg in regions:
            a = amp[reg["channel"]]
            colour = cmap(norm(a)) if a > 1e-14 else "#c3ccd3"
            if reg["boundary"] == "side":
                z0, z1 = reg["z_min_m"] * mm, reg["z_max_m"] * mm
                for x in (R, -R - width):
                    ax.add_patch(Rectangle((x, z0), width, z1 - z0, color=colour, lw=0))
            else:
                z = reg["z_min_m"] * mm
                y = z if reg["boundary"] == "top" else z - width
                for x0, x1 in ((reg["r_min_m"], reg["r_max_m"]), (-reg["r_max_m"], -reg["r_min_m"])):
                    ax.add_patch(Rectangle((x0 * mm, y), (x1 - x0) * mm, width, color=colour,
                                           lw=0))

    # (b) section with fluids, sources and rays.
    ax = fig.add_subplot(grid[1])
    layers = [np.full_like(signed, levels[0]), mirror(faces[0]), mirror(faces[1]),
              np.full_like(signed, levels[3])]
    for j, colour in enumerate([WATER, RESIN, WATER]):
        ax.fill_between(signed, layers[j], layers[j + 1], color=colour,
                        alpha=0.22 if j != 1 else 0.35, lw=0)
    boundary(ax)
    for j in range(2):
        ax.plot(signed, mirror(faces[j]), color="#b86f0c", lw=2)
    for ray in paths:
        seg = clip_path(ray, cfg.levels_m[3] + 0.0012) * mm
        ax.plot(seg[:, 0], seg[:, 1], color=RAY, lw=0.7, alpha=0.85)
    ax.axvline(0, color=MUTED, lw=0.5, ls=(0, (6, 4)))
    ax.text(-R + 0.35, levels[0] + 0.55, f"water\nn = {n[0]}, ρ = {rho[0]:.0f} kg/m³",
            ha="left", **note)
    ax.text(-R + 0.35, 0.0, f"NOA 61 lens liquid\nn = {n[1]}, ρ = {rho[1]:.0f} kg/m³",
            ha="left", va="center", **note)
    ax.text(-R + 0.35, levels[3] - 0.55, f"water\nn = {n[2]}, ρ = {rho[2]:.0f} kg/m³",
            ha="left", va="top", **note)
    ax.annotate("front face", (2.6, faces[0][int(1600 * 2.6 / 4)] * mm), (2.9, -1.85),
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6}, **note)
    ax.annotate("back face", (2.6, faces[1][int(1600 * 2.6 / 4)] * mm), (2.9, 1.75),
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6}, **note)
    z0, z1, z2 = (np.array(material["stigmatic_z_m"]) * mm).round(0)
    ax.text(0, levels[0] - 0.75, f"from object on axis at z = {z0:.0f} mm", ha="center",
            va="top", fontsize=8.5, color=RAY)
    ax.text(0, levels[3] + 1.35, f"to image at z = {z2:.0f} mm (shared intermediate "
            f"conjugate {z1:.0f} mm)", ha="center", va="bottom", fontsize=8.5, color=RAY)
    arrow = {"arrowstyle": "<->", "color": MUTED, "lw": 0.8, "shrinkA": 0, "shrinkB": 0}
    ax.add_patch(FancyArrowPatch((-R, levels[0] - 1.5), (R, levels[0] - 1.5), **arrow))
    ax.text(0, levels[0] - 1.65, f"cell diameter {2 * R:.0f} mm; clear apertures "
            f"{2 * cfg.clear_radius_m * mm:.0f} mm", ha="center", va="top", fontsize=8.5,
            color=MUTED)
    ax.add_patch(FancyArrowPatch((R + 0.6, levels[0]), (R + 0.6, levels[3]), **arrow))
    ax.text(R + 0.75, 0, f"{levels[3] - levels[0]:.0f} mm", rotation=90, va="center",
            fontsize=8.5, color=MUTED)
    ax.set_yticks(range(-3, 4))
    ax.set(xlim=(-R - 0.6, R + 1.3), ylim=(levels[0] - 2.4, levels[3] + 5.6), aspect="equal",
           xlabel="x (mm)", ylabel="z (mm)")
    ax.set_title("(b) Meridional section at equal scale", loc="left", color=INK, fontsize=12)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm, cmap), ax=ax, orientation="horizontal",
                       fraction=0.03, pad=0.10, aspect=40, shrink=0.8)
    bar.set_label("commanded source amplitude (m/s); grey: inactive optical openings",
                  color=INK, fontsize=9)
    inset = ax.inset_axes([0.10, 0.80, 0.80, 0.16], zorder=5)
    inset.set_facecolor("white")
    window = r <= cfg.clear_radius_m
    for j in range(2):
        inset.plot(r[window] * mm, (faces[j][window] - cfg.levels_m[j + 1]) * 1e6,
                   color="#b86f0c", lw=1.6, label="driven" if j == 0 else None)
        inset.plot(r[window] * mm, (goal[j][window] - cfg.levels_m[j + 1]) * 1e6, color=INK,
                   ls=":", lw=1.1, label="Cartesian target" if j == 0 else None)
    inset.axhline(0, color=MUTED, lw=0.6, ls="--", label="unforced (flat)")
    inset.set_title("sag over the clear aperture (µm)", fontsize=8.5, color=INK)
    inset.tick_params(labelsize=7)
    inset.set_xlabel("radius (mm)", fontsize=7.5)
    inset.legend(fontsize=6.5, frameon=False, loc="center left")
    inset.text(0.55, (goal[1][0] - cfg.levels_m[2]) * 1e6 - 12, "back face", fontsize=7.5,
               va="top", color=INK)
    inset.text(0.25, (goal[0][0] - cfg.levels_m[1]) * 1e6 + 12, "front face", fontsize=7.5,
               va="bottom", color=INK)
    # (c) held acoustic field at one instant.
    import matplotlib.tri as mtri

    axf = fig.add_subplot(grid[2])
    xy = field["xy"] * mm
    tri = mtri.Triangulation(xy[0], xy[1], field["triangles"])
    pressure = field["pressure"].real / 1e6
    limit = float(np.percentile(np.abs(field["pressure"]), 99.5)) / 1e6
    image = axf.tripcolor(tri, pressure, shading="gouraud", cmap="RdBu_r", vmin=-limit,
                          vmax=limit, rasterized=True)
    for j in range(2):
        axf.plot(r * mm, faces[j] * mm, color="black", lw=1)
    for reg in regions:
        a = amp[reg["channel"]]
        colour = cmap(norm(a)) if a > 1e-14 else "#c3ccd3"
        axf.plot([reg["r_min_m"] * mm, reg["r_max_m"] * mm],
                 [reg["z_min_m"] * mm, reg["z_max_m"] * mm], color=colour, lw=4,
                 solid_capstyle="butt")
    axf.set(xlim=(-0.1, R + 0.1), ylim=(levels[0] - 0.1, levels[3] + 0.1), aspect="equal",
            xlabel="radius (mm)", ylabel="z (mm)")
    axf.set_title(f"(c) Harmonic pressure, one instant · {cfg.frequency_hz / 1e6:.1f} MHz",
                  loc="left", color=INK, fontsize=12)
    cb = fig.colorbar(image, ax=axf, orientation="horizontal", fraction=0.03, pad=0.10,
                      aspect=30, shrink=0.9, extend="both")
    peak = float(np.max(np.abs(field["pressure"]))) / 1e6
    cb.set_label(f"acoustic pressure (MPa); scale saturates, peak {peak:.0f} MPa", color=INK,
                 fontsize=9)
    errors = np.array(report["mean_pupil_max_error_m"]) * 1e9
    fig.text(0.01, 0.012,
             f"Saved stationary design-grid state (104 surface elements, {len(regions)} ideal "
             f"source regions, source power {field['power_w']:.1f} W, recomputed). Height errors "
             f"{errors[0]:.1f} / {errors[1]:.1f} nm; spot {trace['max_radius_m'] * 1e6:.3f} µm; "
             f"{int(trace['transmitted'].sum())}/{len(trace['transmitted'])} rays transmitted; "
             "fixed-command refinement unfinished. Lossless acoustics with passive port loss; "
             "not an experiment.", fontsize=8.5, color=MUTED)
    out = FIGURES / "two-face-cell.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    render.unlink()
    return {"figure": out.name, "source_power_w": float(field["power_w"]),
            "max_spot_m": float(trace["max_radius_m"])}


# --------------------------------------------------------------------------- #
# Formation animation (single interface, saved trajectory)
# --------------------------------------------------------------------------- #


def formation_animation(width_px=1150):
    from PIL import Image

    cfg, _space, trajectory, analysis, _report = single_interface_state()
    drive = trajectory["drive_m_s"]
    assert np.allclose(drive, drive[0]), "animation assumes one held command"
    mm = 1e3
    time = analysis["time_s"]
    radius, heights = analysis["radius_m"], analysis["surface_height_m"]
    pupil, errors = analysis["pupil_radius_m"], analysis["height_error_m"]
    max_error, spot_rms = analysis["maximum_height_error_m"], analysis["spot_rms_m"]
    spots = analysis["spot_xy_m"]
    z_det = float(analysis["detector_z_m"])
    target = analysis["target_height_m"]
    norm = mcolors.Normalize(0, np.abs(drive[0]).max())
    cmap = plt.get_cmap(SOURCE_MAP)
    launch = np.linspace(-cfg.clear_radius_m, cfg.clear_radius_m, 13)
    z_obj = target[0] + cfg.object_distance_m  # fixed laboratory object, not re-aimed per frame
    # Playback schedule: every saved state while the surface moves, sparser afterwards.
    early = np.flatnonzero(time <= 0.064)
    late = np.flatnonzero(time > 0.064)[3::4]
    if late[-1] != len(time) - 1:
        late = np.r_[late, len(time) - 1]
    schedule = [(k, 110) for k in early] + [(k, 90) for k in late]
    schedule[-1] = (schedule[-1][0], 2600)
    dt = float(time[1] - time[0])
    frames, durations = [], []
    fig = plt.figure(figsize=(11.5, 6.4), dpi=100, facecolor="white")
    for number, (k, duration) in enumerate(schedule):
        fig.clf()
        grid = fig.add_gridspec(3, 2, width_ratios=[1.05, 1], hspace=0.62, wspace=0.2,
                                left=0.06, right=0.98, top=0.86, bottom=0.08)
        ax = fig.add_subplot(grid[:, 0])
        R, depth, wall, base = section_frame(ax, cfg, drive[0], norm, cmap)
        signed = np.r_[-radius[:0:-1], radius] * mm
        mirror = lambda h: np.r_[h[:0:-1], h] * mm
        h = heights[k]
        ax.add_patch(Polygon(np.c_[np.r_[signed, R, -R], np.r_[mirror(h), -depth, -depth]],
                             color=RESIN, alpha=0.30, lw=0))
        ax.plot(signed, mirror(heights[0]), color=MUTED, lw=0.9, ls="--")
        d, hit, det, _ = meridional_rays(radius, h, z_obj, launch, cfg.refractive_index,
                                         cfg.image_refractive_index, z_det)
        for j in range(len(launch)):
            a = hit[j] + (-cfg.depth_m - hit[j, 1]) / d[j, 1] * d[j]
            out = det[j] - hit[j]
            b = hit[j] + (target[0] + 0.0045 - hit[j, 1]) / out[1] * out
            ax.plot(np.array([a[0], hit[j, 0], b[0]]) * mm, np.array([a[1], hit[j, 1], b[1]]) * mm,
                    color=RAY, lw=0.7, alpha=0.85)
        ax.plot(signed, mirror(h), color="#b86f0c", lw=2.2)
        ax.text(0, -3.2, "NOA 61 resin", ha="center", fontsize=9, color=INK)
        ax.text(-R - wall - 0.4, -depth / 2, "16 source rows\nheld command\nfrom t = 0",
                ha="right", va="center", fontsize=8.5, color=INK)
        ax.set(xlim=(-R - 3.6, R + 1.2), ylim=(-depth - base - 0.4, target[0] * mm + 4.6),
               aspect="equal", xlabel="x (mm)", ylabel="z above the rim (mm)")
        ax.set_title("Cell section, equal scale · dashed: unforced surface", loc="left",
                     fontsize=10, color=INK)
        # Height error across the pupil.
        ae = fig.add_subplot(grid[0, 1])
        ae.axhspan(-10, 10, color="#2e8b57", alpha=0.15, lw=0)
        ae.plot(pupil * mm, errors[k] * 1e9, color="#b86f0c", lw=1.6)
        ae.set_yscale("symlog", linthresh=10)
        ae.set_yticks([-1e5, -1e3, -10, 0, 10, 1e3, 1e5])
        ae.tick_params(labelsize=8)
        ae.set(ylim=(-3e5, 3e5), xlim=(0, cfg.clear_radius_m * mm), xlabel="pupil radius (mm)",
               ylabel="height − target (nm)")
        ae.set_title("Surface error (band: ±10 nm goal)", loc="left", fontsize=10, color=INK)
        # Spot diagram at the fixed detector.
        asp = fig.add_subplot(grid[1, 1])
        xy = spots[k] * 1e6
        scale = max(float(np.nanmax(np.abs(xy))), 0.05)
        scale = 10 ** np.ceil(np.log10(scale))
        asp.scatter(xy[:, 0], xy[:, 1], s=2, color=RAY, alpha=0.6, lw=0)
        asp.set(xlim=(-scale, scale), ylim=(-scale, scale), aspect="equal",
                xlabel="detector x (µm)", ylabel="y (µm)")
        asp.set_title(f"Spot, fixed image plane · RMS {spot_rms[k] * 1e6:.3g} µm",
                      loc="left", fontsize=10, color=INK)
        asp.tick_params(labelsize=8)
        # Histories.
        ah = fig.add_subplot(grid[2, 1])
        ah.semilogy(time * 1e3, max_error * 1e9, color="#b86f0c", lw=1.2, label="max height error (nm)")
        ah.semilogy(time * 1e3, spot_rms * 1e9, color=RAY, lw=1.2, label="spot RMS (nm)")
        ah.axhline(10, color="#2e8b57", lw=0.8, ls="--")
        ah.axvline(time[k] * 1e3, color=INK, lw=1)
        ah.set(xlabel="physical time (ms)", xlim=(0, time[-1] * 1e3))
        ah.legend(fontsize=7.5, frameon=False, loc="upper right")
        speed = dt / (duration / 1000) if number < len(schedule) - 1 else None
        fig.suptitle(f"Fixed-command formation of a Cartesian liquid surface · physical time "
                     f"t = {time[k] * 1e3:6.2f} ms", x=0.06, ha="left", fontsize=13, color=INK)
        fig.text(0.06, 0.905, "Saved ALE states (axisymmetric, isothermal numerical model); "
                 "geometric rays; not an experiment. "
                 + (f"Playback {1 / speed:.0f}× slower than real time." if speed
                    else "Final saved state."), fontsize=8.5, color=MUTED)
        fig.canvas.draw()
        image = Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                                 fig.canvas.buffer_rgba()).convert("RGB")
        scale_px = width_px / image.width
        frames.append(image.resize((width_px, int(image.height * scale_px)), Image.LANCZOS))
        durations.append(duration)
    plt.close(fig)
    palette = frames[len(frames) // 3].quantize(colors=160, method=Image.MEDIANCUT)
    quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    out = FIGURES / "interface-formation-and-spots.gif"
    quantized[0].save(out, save_all=True, append_images=quantized[1:], duration=durations,
                      loop=0, optimize=True)
    return {"figure": out.name, "frames": len(frames),
            "playback_s": round(sum(durations) / 1000, 1), "bytes": out.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("single", "pair", "formation"))
    args = parser.parse_args()
    results = {}
    if args.only in (None, "single"):
        results["single"] = single_interface_figure()
    if args.only in (None, "pair"):
        results["pair"] = pair_figure()
    if args.only in (None, "formation"):
        results["formation"] = formation_animation()
    record = FIGURES / "provenance.json"
    previous = json.loads(record.read_text()) if record.exists() else {}
    previous.setdefault("outputs", {}).update(results)
    previous.setdefault("inputs_sha256", {}).update(inputs_used)
    for name in ("single-interface-cell.png", "two-face-cell.png",
                 "interface-formation-and-spots.gif"):
        path = FIGURES / name
        if path.exists():
            previous.setdefault("outputs_sha256", {})[name] = hashlib.sha256(
                path.read_bytes()).hexdigest()
    previous["generator"] = "tools/render_readme_figures.py"
    record.write_text(json.dumps(previous, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2))
