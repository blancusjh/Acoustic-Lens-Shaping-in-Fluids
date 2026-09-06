"""PyVista apparatus geometry, portable time viewer and scientific figure export."""

import base64
import json
import shutil
from pathlib import Path

import numpy as np
import pyvista as pv

from .config import LensConfig
from .geometry import chamber_mesh
from .optics import best_fit_sphere, trace_surface
from .surface import SurfaceSpace


def encoded(array, dtype="float32"):
    a = np.ascontiguousarray(array, dtype=dtype)
    return {
        "dtype": dtype,
        "shape": list(a.shape),
        "data": base64.b64encode(a.tobytes()).decode("ascii"),
    }


def mesh_record(mesh, name, color, opacity=1, **extra):
    return {
        "name": name,
        "points": encoded(mesh.points),
        "polys": encoded(mesh.faces, "uint32"),
        "lines": encoded(mesh.lines, "uint32"),
        "color": color,
        "opacity": opacity,
        **extra,
    }


def annular_sector(inner, outer, low, high, first=0, last=2 * np.pi, segments=128):
    """Closed curved solid, in display millimetres."""
    theta = np.linspace(first, last, segments + 1)
    rings = []
    for z, r in [(low, inner), (low, outer), (high, inner), (high, outer)]:
        rings.append(np.c_[r * np.cos(theta), r * np.sin(theta), np.full(len(theta), z)])
    n = len(theta)
    points = np.vstack(rings)
    faces = []
    for j in range(segments):
        for a, b in [(0, 1), (1, 3), (3, 2), (2, 0)]:
            faces.extend([4, a * n + j, a * n + j + 1, b * n + j + 1, b * n + j])
    faces.extend([4, 0, n, 3 * n, 2 * n])
    faces.extend([4, n - 1, 3 * n - 1, 4 * n - 1, 2 * n - 1])
    return pv.PolyData(points, np.array(faces))


def liquid_mesh(cfg, space, c, radial_count=81, sectors=128):
    radius = cfg.radius_m * 1000
    r = np.linspace(0, radius, radial_count)
    theta = np.arange(sectors) * 2 * np.pi / sectors
    points = [[0, 0, cfg.radius_m * space.evaluate(c, np.array([0]))[0] * 1000]]
    radial_ids = [0]
    for j in range(1, radial_count):
        height = radius * space.evaluate(c, np.array([r[j] / radius]))[0]
        points.extend(np.c_[r[j] * np.cos(theta), r[j] * np.sin(theta), np.full(sectors, height)])
        radial_ids.extend([j] * sectors)
    bottom = len(points)
    points.extend(
        np.c_[radius * np.cos(theta), radius * np.sin(theta), np.full(sectors, -cfg.depth_m * 1000)]
    )
    radial_ids.extend([-1] * sectors)
    bottom_center = len(points)
    points.append([0, 0, -cfg.depth_m * 1000])
    radial_ids.append(-1)
    faces = []
    for k in range(sectors):
        after = (k + 1) % sectors
        faces.extend([3, 0, 1 + k, 1 + after])
        for j in range(1, radial_count - 1):
            lower, upper = 1 + (j - 1) * sectors, 1 + j * sectors
            faces.extend([4, lower + k, upper + k, upper + after, lower + after])
        rim = 1 + (radial_count - 2) * sectors
        faces.extend([4, rim + k, bottom + k, bottom + after, rim + after])
        faces.extend([3, bottom_center, bottom + after, bottom + k])
    return pv.PolyData(np.array(points), np.array(faces)), np.array(radial_ids)


def apparatus(cfg, space, c):
    radius, depth = cfg.radius_m * 1000, cfg.depth_m * 1000
    outer = radius + cfg.wall_thickness_m * 1000
    liquid, radial_ids = liquid_mesh(cfg, space, c)
    top_cells = np.flatnonzero(liquid.cell_centers().points[:, 2] > -1e-8)
    surface = liquid.extract_cells(top_cells).extract_surface(algorithm="dataset_surface").clean()
    cylinder = annular_sector(radius, outer, -depth, 0)
    base = pv.Cylinder(
        center=(0, 0, -depth - cfg.base_thickness_m * 500),
        direction=(0, 0, 1),
        radius=outer,
        height=cfg.base_thickness_m * 1000,
        resolution=128,
    ).triangulate()
    rim = annular_sector(radius - 0.045, radius + 0.055, -0.055, 0.015)
    patches, row_ids = [], []
    pitch = depth / cfg.array_rows
    for row in range(cfg.array_rows):
        low = -depth + (row + 0.5 - 0.5 * cfg.element_fill) * pitch
        high = -depth + (row + 0.5 + 0.5 * cfg.element_fill) * pitch
        for sector in range(cfg.array_sectors):
            first = (sector + 0.035) * 2 * np.pi / cfg.array_sectors
            last = (sector + 0.965) * 2 * np.pi / cfg.array_sectors
            patch = annular_sector(radius + 0.09, outer + 0.035, low, high, first, last, 5)
            patches.append(patch)
            row_ids.extend([row] * patch.n_cells)
    # merge_points=False preserves the electrical sectors and cell order.
    array = pv.merge(patches, merge_points=False)
    return (
        {
            "liquid": liquid,
            "surface": surface,
            "cylinder": cylinder,
            "base": base,
            "rim": rim,
            "array": array,
        },
        radial_ids,
        np.array(row_ids),
    )


def export_viewer(result_directory):
    result = Path(result_directory).resolve()
    cfg = LensConfig(**json.loads((result / "configuration.json").read_text()))
    report = json.loads((result / "report.json").read_text())
    data = np.load(result / "trajectory.npz")
    space = SurfaceSpace(cfg)
    states = data["coefficients"]
    meshes, radial_ids, row_ids = apparatus(cfg, space, states[-1])
    colors = {
        "liquid": [0.12, 0.78, 0.76],
        "surface": [0.34, 0.83, 0.79],
        "cylinder": [0.43, 0.54, 0.64],
        "base": [0.61, 0.83, 0.96],
        "rim": [0.87, 0.73, 0.43],
        "array": [0.6, 0.5, 0.9],
    }
    opacities = {
        "liquid": 0.35,
        "surface": 1.0,
        "cylinder": 0.25,
        "base": 0.5,
        "rim": 1.0,
        "array": 1.0,
    }
    geometry = {
        key: mesh_record(mesh, key, colors[key], opacities[key]) for key, mesh in meshes.items()
    }
    mesh = chamber_mesh(cfg, space, np.zeros(space.count))
    # Acoustic and velocity nodal values correspond to the linear mesh vertices.
    vertex_count = mesh.t.max() + 1
    p = mesh.p[:, :vertex_count]
    slice_points = np.c_[
        p[0] * cfg.radius_m * 1000, np.zeros(vertex_count), p[1] * cfg.radius_m * 1000
    ]
    slice_faces = np.c_[np.full(mesh.t.shape[1], 3), mesh.t.T].ravel()
    right = pv.PolyData(slice_points, slice_faces)
    left = pv.PolyData(slice_points * np.array([-1, 1, 1]), slice_faces)
    cut = right.merge(left, merge_points=False)
    geometry["pressure"] = mesh_record(cut, "pressure", [1, 1, 1], 1.0)
    radial = np.linspace(0, cfg.radius_m, 81)
    h = np.array([cfg.radius_m * space.evaluate(c, radial / cfg.radius_m) for c in states])
    slopes = np.array([space.evaluate(c, radial / cfg.radius_m, 1) for c in states])
    # Use a common best-fit sphere for the departure plot; fitting is area weighted.
    sphere = best_fit_sphere(space, states[-1])
    rr = np.linspace(0, cfg.clear_radius_m, 121)
    spherical_h = sphere["vertex_m"] - rr**2 / (
        sphere["radius_m"] + np.sqrt(sphere["radius_m"] ** 2 - rr**2)
    )
    departure = np.array(
        [1e6 * (cfg.radius_m * space.evaluate(c, rr / cfg.radius_m) - spherical_h) for c in states]
    )
    rays = [trace_surface(space, c, count=24) for c in states]
    payload = {
        "config": cfg.as_dict(),
        "report": report,
        "geometry": geometry,
        "times": data["times_s"].tolist(),
        "radial_m": radial.tolist(),
        "height_m": h.tolist(),
        "slope": slopes.tolist(),
        "target_height_m": (
            cfg.radius_m * space.evaluate(data["target"], radial / cfg.radius_m)
        ).tolist(),
        "radial_ids": encoded(radial_ids, "int32"),
        "array_rows": encoded(row_ids, "int32"),
        "surface_ids": encoded(
            np.rint(
                np.linalg.norm(meshes["surface"].points[:, :2], axis=1) / (cfg.radius_m * 1000) * 80
            ),
            "int32",
        ),
        "drive_real": data["drive_m_s"].real.tolist(),
        "drive_imag": data["drive_m_s"].imag.tolist(),
        "slice_r": p[0].tolist(),
        "slice_eta": ((p[1] + cfg.depth_m / cfg.radius_m) / (cfg.depth_m / cfg.radius_m)).tolist(),
        "slice_vertex_count": int(vertex_count),
        "pressure_pa": encoded(data["acoustic_pressure_pa"]),
        "fluid_velocity": encoded(data["fluid_velocity_m_s"]),
        "radiation_pa": data["radiation_pressure_pa"].tolist(),
        "departure_r_m": rr.tolist(),
        "departure_um": departure.tolist(),
        "ray_r_m": [r["pupil_r_m"].tolist() for r in rays],
        "ray_z_m": [r["surface_z_m"].tolist() for r in rays],
        "ray_spot_m": [r["target_spot_r_m"].tolist() for r in rays],
        "display": {
            "geometry_scale": 1,
            "units": "mm",
            "velocity_arrow_travel_s": 1,
            "acoustic_quantity": "Peak pressure amplitude, acoustic cycle averaged envelope",
        },
    }
    destination = result / "viewer"
    destination.mkdir(exist_ok=True)
    web = Path(__file__).resolve().parents[3] / "web"
    for filename in ["index.html", "app.js", "style.css"]:
        shutil.copy2(web / filename, destination / filename)
    shutil.copytree(web / "static", destination / "static", dirs_exist_ok=True)
    (destination / "data.js").write_text(
        "window.LENS_DATA=" + json.dumps(payload, separators=(",", ":")) + ";\n"
    )
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    # Inspectable native geometry, in SI units; browser geometry alone uses mm.
    vtk_directory = result / "geometry"
    vtk_directory.mkdir(exist_ok=True)
    for key, value in meshes.items():
        physical = value.copy()
        physical.points *= 0.001
        physical.save(vtk_directory / f"{key}.vtp")
    return destination / "index.html"


def render_figures(result_directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = Path(result_directory)
    data = np.load(result / "trajectory.npz")
    report = json.loads((result / "report.json").read_text())
    cfg = LensConfig(**report["configuration"])
    space = SurfaceSpace(cfg)
    out = result / "figures"
    out.mkdir(exist_ok=True)
    r = np.linspace(-cfg.radius_m, cfg.radius_m, 601)
    initial, final, target = data["initial"], data["coefficients"][-1], data["target"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for c, label, color in [
        (initial, "Unforced liquid", "#657186"),
        (target, "Optical target", "#e19437"),
        (final, "Computed driven liquid", "#009c9c"),
    ]:
        axes[0, 0].plot(
            r * 1000,
            cfg.radius_m * space.evaluate(c, np.abs(r) / cfg.radius_m) * 1000,
            label=label,
            color=color,
            lw=2,
            ls="--" if label == "Optical target" else "-",
        )
    axes[0, 0].set(
        xlabel="Radius across lens (mm)",
        ylabel="Height above rim (mm)",
        title="The finite lens surface",
    )
    axes[0, 0].legend(fontsize=8)
    sphere = best_fit_sphere(space, final)
    rr = np.linspace(0, cfg.clear_radius_m, 301)
    hs = sphere["vertex_m"] - rr**2 / (
        sphere["radius_m"] + np.sqrt(sphere["radius_m"] ** 2 - rr**2)
    )
    axes[0, 1].plot(
        rr * 1000,
        1e6 * (cfg.radius_m * space.evaluate(final, rr / cfg.radius_m) - hs),
        color="#009c9c",
    )
    axes[0, 1].set(
        xlabel="Pupil radius (mm)",
        ylabel="Departure from best-fit sphere (µm)",
        title="Verified aspheric figure",
    )
    for c, label, color in [(initial, "Unforced", "#657186"), (final, "Driven", "#009c9c")]:
        ray = trace_surface(space, c, count=14)
        for sign in [-1, 1]:
            for j in range(14):
                z = np.array([ray["surface_z_m"][j], ray["target_plane_m"]])
                x = np.array([ray["pupil_r_m"][j], ray["target_spot_r_m"][j]]) * sign
                axes[1, 0].plot(z * 1000, x * 1000, color=color, alpha=0.65, lw=0.8)
        axes[1, 0].plot([], [], color=color, label=label)
    axes[1, 0].set(
        xlabel="Height from rim (mm)",
        ylabel="Ray radius (mm)",
        title="Actual Snell rays to the target plane",
    )
    axes[1, 0].legend(fontsize=8)
    t = data["times_s"]
    axes[1, 1].semilogy(
        t, [x["rms_spot_at_target_m"] * 1e6 for x in report["history"]], color="#009c9c"
    )
    axes[1, 1].set(
        xlabel="Physical fluid time (s)",
        ylabel="Geometric RMS ray spot (µm)",
        title="Computed time evolution",
    )
    for ax in axes.flat:
        ax.grid(alpha=0.18)
    fig.suptitle("6 mm clear-aperture liquid asphere · 20 mm target focus", fontsize=16)
    fig.savefig(out / "optical-validation.png", dpi=180)
    fig.savefig(out / "optical-validation.pdf")
    plt.close(fig)
    meshes, _, _ = apparatus(cfg, space, final)
    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1100))
    plotter.set_background("#101c2a")
    for key, mesh in meshes.items():
        if key in ("cylinder", "array"):
            mesh = mesh.clip(normal=(0, 1, 0), origin=(0, 0, 0), invert=False)
        color = {
            "liquid": "#39c7c3",
            "surface": "#66d4ca",
            "base": "#b5d8ef",
            "cylinder": "#a1afbd",
            "array": "#ba92e8",
            "rim": "#ddc081",
        }[key]
        plotter.add_mesh(
            mesh,
            color=color,
            opacity=0.35 if key in ("liquid", "base") else 1,
            smooth_shading=True,
            specular=0.65,
            specular_power=35,
        )
    plotter.add_text(
        "FINITE LIQUID ASPHERE\n8 mm chamber | 6 mm clear aperture\n256 array sectors, 16 coherent rows",
        color="white",
        font_size=14,
    )
    plotter.add_text(
        "Front half of housing/array cut away. Geometry at 1:1 scale.\nUncured liquid; nominal acoustic material model.",
        position="lower_left",
        color="white",
        font_size=11,
    )
    plotter.camera_position = [(13, -19, 12), (0, 0, -2.6), (0, 0, 1)]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = 7.6
    plotter.screenshot(out / "apparatus.png")
    plotter.close()
    return out


def serve_viewer(result_directory, port=8765):
    from flask import Flask, send_from_directory

    directory = Path(result_directory).resolve() / "viewer"
    if not (directory / "index.html").exists():
        export_viewer(result_directory)
    app = Flask(__name__, static_folder=None)

    @app.route("/")
    def index():
        return send_from_directory(directory, "index.html")

    @app.route("/<path:path>")
    def asset(path):
        return send_from_directory(directory, path)

    app.run(host="127.0.0.1", port=port, debug=False)
