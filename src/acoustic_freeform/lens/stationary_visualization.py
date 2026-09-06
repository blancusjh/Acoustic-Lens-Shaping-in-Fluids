"""Scientific figures and an explicitly stationary PyVista apparatus scene."""

import io
import json
import zipfile
from dataclasses import replace
from html import escape
from pathlib import Path

import numpy as np
import pyvista as pv

from .config import LensConfig
from .optics import trace_surface
from .surface import SurfaceSpace
from .visualization import apparatus


def export_scene_html(plotter, path, title, conjugates, qualification, refinement):
    """Preserve camera projection and visible scientific context in native VTK HTML.

    The installed trame serializer omits vtkCamera's parallel projection/scale
    and VTK corner annotations. Add the camera properties to the scene state,
    and provide the annotations as accessible HTML around the exported view.
    """
    from trame_vtk.tools.vtksz2html import write_html

    raw = plotter.export_vtksz(filename=None)
    with zipfile.ZipFile(io.BytesIO(raw)) as original:
        scene = json.loads(original.read("index.json"))

        def fix_camera(value):
            if isinstance(value, dict):
                if value.get("type") in ("vtkCamera", "vtkOpenGLCamera"):
                    value["properties"].update(
                        parallelProjection=int(plotter.camera.parallel_projection),
                        parallelScale=plotter.camera.parallel_scale,
                    )
                for child in value.values():
                    fix_camera(child)
            elif isinstance(value, list):
                for child in value:
                    fix_camera(child)

        fix_camera(scene)
        packed = io.BytesIO()
        with zipfile.ZipFile(packed, "w", zipfile.ZIP_DEFLATED) as output:
            for name in original.namelist():
                output.writestr(
                    name, json.dumps(scene) if name == "index.json" else original.read(name)
                )
    buffer = io.StringIO()
    write_html(packed.getvalue(), buffer)
    html = buffer.getvalue().replace(
        "<title>VTK.js | Example - OfflineLocalView</title>",
        f"<title>{escape(title)} · Acoustic Freeform Lab</title>",
    )
    context = f"""
<style>
body {{background:#111e2d;color:#e4eff5;font-family:system-ui,sans-serif;}}
#vtk-root {{position:fixed;inset:150px 0 82px;height:auto!important;}}
header,footer {{position:fixed;left:0;right:0;z-index:5;padding:16px 24px;}}
header {{top:0;background:#0c1723;border-bottom:1px solid #324859;}}
footer {{bottom:0;background:#0c1723;border-top:1px solid #324859;font-size:13px;}}
h1 {{font-size:25px;margin:6px 0;}} p {{margin:6px 0;font-size:14px;}}
small {{color:#75d8d1;letter-spacing:.08em;}}
a {{color:#75d8d1;margin-right:20px;}} .status {{color:#f0c78a;}}
</style>
<header><small>ACOUSTIC FREEFORM LAB / STATIONARY CANDIDATE</small>
<h1>{escape(title)}</h1><p>{escape(conjugates)}</p>
<p class="status">{escape(qualification)}. {escape(refinement)}</p></header>
<footer><p>Drag to rotate · Scroll to zoom · Shift-drag to pan.
Nominal computed surface, geometry in mm. No physical time evolution in this scene.</p>
<a href="figures/stationary-excitation.png">Surface, rays and refinement</a>
<a href="hold-drive.csv">Row excitations</a><a href="report.json">Numerical report</a></footer>
"""
    # Layout must exist before VTK sizes its canvas during initialization.
    styles, annotations = context.split("</style>", 1)
    html = html.replace("</head>", styles + "</style></head>")
    html = html.replace("</body>", annotations + "</body>")
    Path(path).write_text(html)


def render_stationary(result_directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = Path(result_directory)
    report = json.loads((result / "report.json").read_text())
    cfg = LensConfig(**report["configuration"])
    data = np.load(result / "stationary.npz")
    space = SurfaceSpace(cfg)
    c, target, initial, drive = (
        data[k] for k in ("coefficients", "target", "initial", "drive_m_s")
    )
    destination = result / "figures"
    destination.mkdir(exist_ok=True)
    r = np.linspace(0, cfg.radius_m, 1001)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for state, name, color in [
        (initial, "Unforced, same fill", "#6b7e92"),
        (target, "Exact optical target", "#dca038"),
        (c, "Nominal stationary candidate", "#00999b"),
    ]:
        axes[0, 0].plot(
            r * 1000,
            cfg.radius_m * space.evaluate(state, r / cfg.radius_m) * 1000,
            label=name,
            color=color,
        )
    axes[0, 0].set(
        xlabel="Radius (mm)",
        ylabel="Height above rim (mm)",
        title="Finite-conjugate Cartesian diopter",
    )
    axes[0, 0].legend(fontsize=8)
    ray = trace_surface(space, c, count=1001)
    axes[0, 1].plot(
        ray["pupil_r_m"] * 1000,
        ray["target_spot_r_m"] * 1e6,
        color="#00999b",
        label=f"Design mesh: {ray['rms_spot_at_target_m'] * 1e6:.3f} µm RMS",
    )
    refinement_note = "Spatial convergence has not been checked."
    convergence = result / "spatial-convergence.json"
    if convergence.exists():
        checks = json.loads(convergence.read_text())
        if checks:
            last = checks[-1]
            nr, nz = last["mesh"]
            modes = last["surface_modes"]
            refined = result / f"refined-P{last['order']}-{nr}-{nz}-{modes}.npz"
            if refined.exists():
                fine_space = SurfaceSpace(
                    replace(
                        cfg,
                        acoustic_order=last["order"],
                        mesh_radial=nr,
                        mesh_vertical=nz,
                        surface_modes=modes,
                    )
                )
                fine_ray = trace_surface(fine_space, np.load(refined)["coefficients"], count=1001)
                axes[0, 1].plot(
                    fine_ray["pupil_r_m"] * 1000,
                    fine_ray["target_spot_r_m"] * 1e6,
                    color="#b4593e",
                    label=f"Same drive, finer mesh: {fine_ray['rms_spot_at_target_m'] * 1e6:.3f} µm RMS",
                )
            refinement_note = (
                f"Ray RMS: {report['optics']['rms_spot_at_target_m'] * 1e6:.3f} µm on design mesh; "
                f"{last['ray_rms_at_target_m'] * 1e6:.3f} µm after refinement."
            )
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].set(
        xlabel="Pupil radius (mm)",
        ylabel="Signed ray intercept (µm)",
        title="Actual Snell rays at the specified image plane",
    )
    rows = np.arange(1, len(drive) + 1)
    axes[1, 0].bar(rows, abs(drive) * 1000, color="#00999b")
    axes[1, 0].set(
        xlabel="Array row from bottom",
        ylabel="Peak wall velocity (mm/s)",
        title=f"Holding amplitudes at {cfg.frequency_hz / 1e6:g} MHz",
    )
    axes[1, 1].plot(rows, np.degrees(np.angle(drive * drive[0].conjugate())), "o-", color="#ac7628")
    axes[1, 1].set(
        xlabel="Array row from bottom",
        ylabel="Phase relative to row 1 (degrees)",
        title="Coherent holding phases",
        ylim=(-190, 190),
    )
    for ax in axes.flat:
        ax.grid(alpha=0.18)
    stability = report.get("stability", {})
    unstable_modes = stability.get("unstable_modes", 0)
    mode_label = "mode" if unstable_modes == 1 else "modes"
    qualification = (
        f"{unstable_modes} growing {mode_label} with fixed drives in the nominal model"
        if unstable_modes
        else "Stationary solution; no approach trajectory in this figure"
    )
    object_label = "−∞" if cfg.object_distance_m is None else f"{cfg.object_distance_m * 1000:g}"
    conjugates = (
        f"nₒ={cfg.refractive_index:g}, zₒ={object_label} mm → "
        f"nᵢ={cfg.image_refractive_index:g}, zᵢ=+{cfg.focal_distance_m * 1000:g} mm"
    )
    fig.suptitle(
        f"{conjugates}\n{qualification}",
        fontsize=13,
    )
    fig.savefig(destination / "stationary-excitation.png", dpi=180)
    fig.savefig(destination / "stationary-excitation.pdf")
    plt.close(fig)

    meshes, _, row_ids = apparatus(cfg, space, c)
    geometry = result / "geometry"
    geometry.mkdir(exist_ok=True)
    plotter = pv.Plotter(off_screen=True, window_size=(1440, 1000))
    plotter.set_background("#111e2d")
    colors = {
        "liquid": "#36c4bf",
        "surface": "#73ddd2",
        "cylinder": "#9cadbe",
        "base": "#b1d6ef",
        "rim": "#dcb979",
        "array": "#b390d4",
    }
    meshes["array"].cell_data["drive_phase_deg"] = np.degrees(np.angle(drive))[row_ids]
    for key, mesh in meshes.items():
        physical = mesh.copy()
        physical.points *= 0.001
        physical.save(geometry / f"{key}.vtp")
        if key in ("array", "cylinder"):
            mesh = mesh.clip(normal=(0, 1, 0), origin=(0, 0, 0), invert=False)
        kwargs = (
            {
                "scalars": "drive_phase_deg",
                "cmap": "twilight",
                "clim": [-180, 180],
                "show_scalar_bar": False,
            }
            if key == "array"
            else {"color": colors[key], "scalars": None}
        )
        plotter.add_mesh(
            mesh,
            **kwargs,
            opacity=0.35 if key in ("liquid", "base", "cylinder") else 1,
            smooth_shading=True,
            specular=0.6,
            name=key,
        )
    plotter.add_text(
        "STATIONARY CARTESIAN-DIOPTER CANDIDATE\n"
        f"Chamber: {2 * cfg.radius_m * 1000:g} mm diameter; clear aperture: {2 * cfg.clear_radius_m * 1000:g} mm; uncured liquid\n"
        "Array colors show phase; amplitudes are in hold-drive.csv",
        font_size=12,
        color="white",
    )
    plotter.add_text(
        qualification
        + "\n"
        + refinement_note
        + "\nGeometry in mm, actual scale. Cutaway housing. Nominal acoustic material model.",
        position="lower_left",
        font_size=10,
        color="white",
    )
    plotter.camera_position = [(13, -19, 13), (0, 0, -2.3), (0, 0, 1)]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = 7.5
    plotter.screenshot(destination / "stationary-apparatus.png")
    export_scene_html(
        plotter,
        result / "stationary-viewer.html",
        "Cartesian diopter: apparatus",
        conjugates,
        qualification,
        refinement_note,
    )
    optical = trace_surface(space, c, count=18)
    for side in [-1, 1]:
        for j in range(18):
            radius, height = optical["pupil_r_m"][j], optical["surface_z_m"][j]
            incoming_slope = optical["incident_direction_r"][j] / optical["incident_direction_z"][j]
            base_radius = radius + (-cfg.depth_m - height) * incoming_slope
            points = (
                np.array(
                    [
                        [side * base_radius, 0, -cfg.depth_m],
                        [side * radius, 0, height],
                        [side * optical["target_spot_r_m"][j], 0, optical["target_plane_m"]],
                    ]
                )
                * 1000
            )
            plotter.add_mesh(pv.lines_from_points(points), color="#e5b35c", line_width=1.2)
    plotter.camera_position = [(24, -44, 14), (0, 0, 7), (0, 0, 1)]
    plotter.camera.parallel_scale = 17
    plotter.reset_camera_clipping_range()
    plotter.render()
    plotter.screenshot(destination / "stationary-optics.png")
    export_scene_html(
        plotter,
        result / "optics-viewer.html",
        "Cartesian diopter: optical rays",
        conjugates,
        qualification,
        refinement_note,
    )
    plotter.close()
    return result / "stationary-viewer.html"
