"""Artist's impression of the proposed S09 machine, from its declared components.

Reads configs/S09-universal-source/machine-proposal.json and the design state it
names. Only the bore, the liquids and the lens shape are modelled (taken from
the formed design); every other part is a proposed, unsimulated component whose
dimensions are the placeholders declared there. Each drawn part maps to one
declared component id, and the label colour gives its status. Outputs a bench
overview, a true-scale cut-away of the cell module and a turntable video.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from vtkmodules.vtkRenderingCore import vtkCoordinate

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/S09-universal-source/machine-proposal.json"
MODELLED, PROPOSED = "#1f6fb2", "#d9480f"
inputs_used = {}


def read(path):
    path = Path(path)
    inputs_used[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def revolve_profile(r, z, first=0.0, last=1.5 * np.pi, sectors=160):
    """Surface of revolution of a closed (r, z) outline, plus the two cut faces."""
    theta = np.linspace(first, last, sectors)
    r, z = np.asarray(r, float), np.asarray(z, float)
    x, y = np.outer(r, np.cos(theta)), np.outer(r, np.sin(theta))
    shell = pv.StructuredGrid(x, y, np.repeat(z[:, None], len(theta), 1)).extract_surface(algorithm="dataset_surface")
    caps = []
    for angle in (first, last):
        points = np.c_[r * np.cos(angle), r * np.sin(angle), z]
        face = pv.PolyData(points, faces=np.r_[len(r), np.arange(len(r))])
        caps.append(face.triangulate())
    return shell.merge(caps)


def ring(radius_in, radius_out, z0, z1, **kw):
    return revolve_profile([radius_in, radius_out, radius_out, radius_in, radius_in], [z0, z0, z1, z1, z0], **kw)


def load(config_path):
    from acoustic_freeform.apparatus.config import DualConfig
    from acoustic_freeform.core.paths import data_path
    from acoustic_freeform.mechanics.surface import DualSurface

    proposal = json.loads(read(config_path).read_text())
    design = data_path(proposal["design_state"])
    cfg = DualConfig(**json.loads(read(design / "config.json").read_text()))
    space = DualSurface(cfg, allow_volume_offsets=True)
    q = np.load(read(design / "state.npz"))["coefficients_m"]
    components = {c["id"]: c for c in proposal["components"]}
    return proposal, components, cfg, space, q


def module(p, comps, cfg, space, q, cut=(0.0, 1.5 * np.pi), detail=True):
    """Cell module in mm; returns anchor points for labels."""
    mm = 1e3
    R, L = cfg.radius_m * mm, np.asarray(cfg.levels_m) * mm
    r = np.linspace(0, cfg.radius_m, 120)
    faces = [(cfg.levels_m[j + 1] + space.height(j, q[j], r)) * mm for j in range(2)]
    rr = r * mm
    kw = {"first": cut[0], "last": cut[1]}
    anchors = {}
    # Modelled liquids and the formed lens.
    lower = revolve_profile(np.r_[0, R, rr[::-1], 0], np.r_[L[0], L[0], faces[0][::-1], L[0]], **kw)
    resin = revolve_profile(np.r_[rr, rr[::-1], 0], np.r_[faces[0], faces[1][::-1], faces[0][0]], **kw)
    upper = revolve_profile(np.r_[0, rr, R, 0, 0], np.r_[faces[1][0], faces[1], L[3], L[3], faces[1][0]], **kw)
    p.add_mesh(lower, color="#9ecae1", opacity=0.35, smooth_shading=True, specular=0.6)
    p.add_mesh(upper, color="#9ecae1", opacity=0.35, smooth_shading=True, specular=0.6)
    p.add_mesh(resin, color="#f0a830", opacity=0.92, smooth_shading=True, specular=1.0, specular_power=60)
    anchors["liquids"] = (0.0, -0.6 * R, 0.5 * (faces[0][0] + faces[1][0]))
    anchors["bore"] = (-R * 0.7, -R * 0.7, L[3] - 0.3)
    # Pinning knife edges at the rim heights.
    pin = comps["pinning"]["render"]
    for z in L[1:3]:
        p.add_mesh(revolve_profile([R - pin["lip_mm"], R, R, R - pin["lip_mm"]],
                                   [z, z - pin["height_mm"], z + pin["height_mm"], z], **kw),
                   color="#c0c4c8", pbr=True, metallic=0.9, roughness=0.25)
    anchors["pinning"] = (R * np.cos(cut[1] - 0.2), R * np.sin(cut[1] - 0.2), L[1])
    # Segmented piezoceramic wall tube with 16 bands per layer (electrodes outside).
    tube = comps["wall_tube"]["render"]["outer_radius_mm"]
    p.add_mesh(ring(R, tube, L[0], L[3], **kw), color="#e8e2d6", smooth_shading=True, specular=0.2)
    if detail:
        for j in range(3):
            edges = np.linspace(L[j], L[j + 1], 17)
            for a, b in itertools.pairwise(edges):
                gap = 0.12 * (b - a)
                p.add_mesh(ring(tube, tube + 0.02, a + gap, b - gap, **kw), color="#d8d8d8",
                           pbr=True, metallic=1.0, roughness=0.3)
    anchors["wall_tube"] = (tube * np.cos(0.6), tube * np.sin(0.6), 0.5 * (L[0] + L[3]))
    # Transparent ring-electrode transducer windows.
    for key, z0, z1, face_z in (("window_lower", None, L[0], L[0]), ("window_upper", L[3], None, L[3])):
        spec = comps[key]["render"]
        a = face_z - spec["thickness_mm"] if z0 is None else z0
        b = face_z + spec["thickness_mm"] if z1 is None else z1
        p.add_mesh(ring(0.0, spec["radius_mm"], a, b, **kw), color="#cfe8f0", opacity=0.28,
                   smooth_shading=True, specular=1.0, specular_power=80)
        if detail:
            theta = np.linspace(cut[0], cut[1], 160)
            for radius in np.linspace(0, R, 33)[1:]:
                line = np.c_[radius * np.cos(theta), radius * np.sin(theta), np.full_like(theta, face_z)]
                p.add_mesh(pv.lines_from_points(line), color="#c9a227", line_width=1, opacity=0.8)
            for k, radius in enumerate(np.linspace(0, R, 33)[1:]):
                angle = cut[0] + 0.02 + 0.012 * k
                line = np.array([[radius * np.cos(angle), radius * np.sin(angle), face_z],
                                 [spec["radius_mm"] * np.cos(angle), spec["radius_mm"] * np.sin(angle), face_z]])
                p.add_mesh(pv.lines_from_points(line), color="#c9a227", line_width=1, opacity=0.6)
        anchors[key] = (spec["radius_mm"] * np.cos(cut[1] - 0.5), spec["radius_mm"] * np.sin(cut[1] - 0.5),
                        0.5 * (a + b))
    # Aluminium thermal housing (cut away) with two Peltier plates.
    house = comps["housing"]["render"]
    zc = 0.5 * (L[0] + L[3])
    w_top = comps["window_upper"]["render"]
    p.add_mesh(revolve_profile(
        [tube + 0.3, house["half_width_mm"], house["half_width_mm"], w_top["radius_mm"] + 0.2,
         w_top["radius_mm"] + 0.2, tube + 0.3, tube + 0.3],
        [zc - house["half_height_mm"], zc - house["half_height_mm"], zc + house["half_height_mm"],
         zc + house["half_height_mm"], L[3], L[3], zc - house["half_height_mm"]],
        first=0.0, last=np.pi),
        color="#c9cdd2", smooth_shading=True, specular=0.6, specular_power=25)
    anchors["housing"] = (house["half_width_mm"] * np.cos(1.2), house["half_width_mm"] * np.sin(1.2), zc + 5)
    for angle in (0.9, 2.6):
        centre = np.array([(house["half_width_mm"] + 0.6) * np.cos(angle), (house["half_width_mm"] + 0.6) * np.sin(angle), zc])
        plate = pv.Box(bounds=(-0.6, 0.6, -5, 5, -5, 5)).rotate_z(np.degrees(angle), inplace=False).translate(centre, inplace=False)
        p.add_mesh(plate, color="#f4f4f0", smooth_shading=False)
    # PTFE fill / vent capillaries, one per layer.
    ports = comps["ports"]["render"]["radius_mm"]
    port_z = [0.5 * (L[0] + faces[0][-1]), 0.5 * (faces[0][-1] + faces[1][-1]), 0.5 * (faces[1][-1] + L[3])]
    for z in port_z:
        tube_line = pv.Line((R * np.cos(0.25), R * np.sin(0.25), z),
                            (30 * np.cos(0.25), 30 * np.sin(0.25), z - 4))
        p.add_mesh(tube_line.tube(radius=ports), color="#f7f7f2", opacity=0.9, smooth_shading=True)
    anchors["ports"] = (22 * np.cos(0.25), 22 * np.sin(0.25), port_z[1] - 3)
    # Flex cables from both windows and the tube.
    for z, angle in ((L[0] - 1, 110), (L[3] + 1, 130), (zc, 150)):
        start = 6.2 if z != zc else tube
        flex = pv.Box(bounds=(start, 28, -2.5, 2.5, z - 0.08, z + 0.08)).rotate_z(angle, inplace=False)
        p.add_mesh(flex, color="#d98c1a", opacity=0.9)
    anchors["flex"] = (22 * np.cos(np.radians(130)), 22 * np.sin(np.radians(130)), L[3] + 1)
    return anchors


def bench(p, comps, cfg, space, q):
    anchors = {}
    p.add_mesh(pv.Box(bounds=(-160, 160, -120, 120, -132, -120)), color="#2b2d30", pbr=True, metallic=0.6, roughness=0.6)
    for x, y in ((15, 15), (-15, 15), (15, -15), (-15, -15)):
        p.add_mesh(pv.Cylinder(center=(x, y, 15), direction=(0, 0, 1), radius=3, height=270),
                   color="#d0d3d6", pbr=True, metallic=1.0, roughness=0.2)
    anchors["cage"] = (15, -15, 110)
    for z in (-95, -60, -35, 40, 85):
        p.add_mesh(pv.Box(bounds=(-21, 21, -21, 21, z - 3, z + 3)), color="#1d1f22", pbr=True, metallic=0.4, roughness=0.5)
    # Light source: LED, pinhole, collimating lens.
    p.add_mesh(pv.Cylinder(center=(0, 0, -105), direction=(0, 0, 1), radius=10, height=18), color="#111111")
    p.add_mesh(pv.Disc(center=(0, 0, -95.5), inner=0, outer=3, normal=(0, 0, 1)), color="#fff7d6")
    p.add_mesh(pv.Disc(center=(0, 0, -56.5), inner=0.3, outer=12, normal=(0, 0, 1)), color="#888888", pbr=True, metallic=1)
    lens = pv.ParametricEllipsoid(12, 12, 3).translate((0, 0, -35), inplace=False)
    p.add_mesh(lens, color="#d6f0f7", opacity=0.45, specular=1.0)
    anchors["light_source"] = (0, -12, -70)
    # Cell module at true scale on the axis (simplified detail at this scale).
    module(p, comps, cfg, space, q, cut=(0.0, 1.5 * np.pi), detail=False)
    anchors["module"] = (0, -13, 0)
    # Objective, beam splitter, camera, wavefront sensor.
    p.add_mesh(pv.Cylinder(center=(0, 0, 26), direction=(0, 0, 1), radius=9, height=26), color="#2a2a2a", pbr=True, metallic=0.5)
    p.add_mesh(ring(0, 6, 12.8, 13.4, first=0, last=2 * np.pi), color="#c0e0ee", opacity=0.6, specular=1)
    p.add_mesh(pv.Cylinder(center=(0, 0, 36), direction=(0, 0, 1), radius=9.4, height=2), color="#c9a227", pbr=True, metallic=1)
    anchors["objective"] = (-9, -9, 26)
    p.add_mesh(pv.Box(bounds=(-12, 12, -12, 12, 55, 79)), color="#bfe6f2", opacity=0.45, specular=1)
    p.add_mesh(pv.Box(bounds=(-16, 16, -16, 16, 95, 135)), color="#24262a", pbr=True, metallic=0.3, roughness=0.5)
    p.add_mesh(pv.Box(bounds=(14, 50, -16, 16, 51, 83)), color="#24262a", pbr=True, metallic=0.3, roughness=0.5)
    anchors["camera"] = (0, -16, 120)
    # Optional UV curing ring around the module.
    p.add_mesh(pv.ParametricTorus(ringradius=34, crosssectionradius=2.5).translate((0, 0, 9), inplace=False),
               color="#5b2a86", pbr=True, metallic=0.3, roughness=0.4)
    for a in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        p.add_mesh(pv.Sphere(radius=1.4, center=(31 * np.cos(a), 31 * np.sin(a), 9)), color="#b36bff")
    anchors["uv"] = (-34, 0, 9)
    # Syringe pumps with PTFE lines to the module.
    for k, y in enumerate((-55, -25, 5)):
        p.add_mesh(pv.Box(bounds=(-150, -95, y - 11, y + 11, -120, -95)), color="#e9ecef", pbr=True, metallic=0.1, roughness=0.6)
        p.add_mesh(pv.Cylinder(center=(-110, y, -88), direction=(1, 0, 0), radius=5, height=30), color="#e6f2f7", opacity=0.6)
        path = np.array([[-95, y, -88], [-60, y * 0.5, -40], [-35, 0, -2 + 1.0 * k], [-4, 0, 0.3 + 1.0 * k]])
        p.add_mesh(pv.Spline(path, 60).tube(radius=0.7), color="#f7f7f2")
    anchors["pumps"] = (-122, -55, -95)
    # Transmitter and control computer.
    p.add_mesh(pv.Box(bounds=(90, 155, -60, 60, -120, -70)), color="#3a3f44", pbr=True, metallic=0.6, roughness=0.4)
    for k in range(14):
        p.add_mesh(pv.Box(bounds=(90 - 0.5, 90, -55 + 8 * k, -51 + 8 * k, -85, -80)), color="#2ec27e")
    path = np.array([[90, 0, -90], [60, 10, -40], [30, -15, -5], [8, -12, 0]])
    p.add_mesh(pv.Spline(path, 60).tube(radius=2.0), color="#d98c1a")
    anchors["transmitter"] = (120, -60, -75)
    p.add_mesh(pv.Box(bounds=(95, 150, 70, 100, -120, -116)), color="#222222")
    p.add_mesh(pv.Box(bounds=(100, 145, 92, 96, -116, -84)), color="#111111")
    p.add_mesh(pv.Box(bounds=(101, 144, 91.9, 92, -114, -86)), color="#2b6cb0")
    anchors["control"] = (122, 92, -88)
    return anchors


def project(p, points):
    coord = vtkCoordinate()
    coord.SetCoordinateSystemToWorld()
    width, height = p.window_size
    out = {}
    for key, xyz in points.items():
        coord.SetValue(*xyz)
        x, y = coord.GetComputedDoubleDisplayValue(p.renderer)
        out[key] = (x / width, y / height)
    return out


def annotate(image, screen, comps, path, title, footnote, side_split=0.5):
    fig = plt.figure(figsize=(16, 16 * image.shape[0] / image.shape[1] + 1.2))
    ax = fig.add_axes([0.0, 0.06, 1.0, 0.88])
    ax.imshow(image)
    ax.axis("off")
    h, w = image.shape[:2]
    left = sorted([(k, v) for k, v in screen.items() if v[0] < side_split], key=lambda kv: -kv[1][1])
    right = sorted([(k, v) for k, v in screen.items() if v[0] >= side_split], key=lambda kv: -kv[1][1])
    for column, x_text, ha in ((left, 0.01, "left"), (right, 0.99, "right")):
        for n, (key, (x, y)) in enumerate(column):
            comp = comps.get(key)
            label = comp["label"] if comp else "Cell module (bore, liquids and lens modelled)"
            status = comp["status"] if comp else "modelled"
            colour = MODELLED if status == "modelled" else PROPOSED
            y_text = 0.95 - n * (0.9 / max(len(column), 1))
            ax.annotate(f"{label}\n({status})", (x * w, (1 - y) * h), (x_text * w, (1 - y_text) * h), fontsize=9,
                        color=colour, ha=ha, va="center",
                        arrowprops={"arrowstyle": "-", "color": colour, "lw": 0.8},
                        bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1.5})
    fig.text(0.01, 0.975, title, fontsize=15, va="top")
    fig.text(0.01, 0.015, footnote, fontsize=9, color="#555")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/studies/S09-universal-source/figures/machine")
    parser.add_argument("--frames", type=int, default=90)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    proposal, comps, cfg, space, q = load(args.config.resolve())
    footnote = ("Artist's impression of a proposal: only the bore, the three liquids and the lens shape are in the S09 "
                "numerical model (blue). Orange parts are unbuilt and unsimulated; their dimensions are placeholders.")
    # 1. True-scale cut-away of the cell module.
    p = pv.Plotter(off_screen=True, window_size=(1800, 1500))
    p.set_background("#f6f7f9", top="#ffffff")
    p.enable_anti_aliasing("ssaa")
    p.enable_lightkit()
    anchors = module(p, comps, cfg, space, q)
    p.camera_position = [(52, -46, 30), (0, 0, 0.0), (0, 0, 1)]
    p.camera.view_angle = 26
    image = p.screenshot(return_img=True)
    screen = project(p, anchors)
    p.close()
    annotate(image, screen, comps, args.out / "s09-machine-module.png",
             "Cell module, cut away, true scale (bore Ø 4.04 mm)", footnote)
    # 1b. Close-up into the bore: the formed lens between the two windows.
    p = pv.Plotter(off_screen=True, window_size=(1800, 1300))
    p.set_background("#f6f7f9", top="#ffffff")
    p.enable_anti_aliasing("ssaa")
    p.enable_lightkit()
    anchors = module(p, comps, cfg, space, q)
    p.camera_position = [(9.5, -8.5, 5.0), (0, 0, 0.55), (0, 0, 1)]
    p.camera.view_angle = 30
    image = p.screenshot(return_img=True)
    keep = ("liquids", "pinning", "wall_tube", "window_lower", "window_upper", "bore")
    screen = {k: v for k, v in project(p, anchors).items() if k in keep}
    p.close()
    annotate(image, screen, comps, args.out / "s09-machine-bore.png",
             "Inside the bore: the resin held in the Fig. 9b shape between the two transducer windows", footnote)
    # 2. Bench overview.
    p = pv.Plotter(off_screen=True, window_size=(1800, 1500))
    p.set_background("#eef0f3", top="#ffffff")
    p.enable_anti_aliasing("ssaa")
    p.enable_lightkit()
    anchors = bench(p, comps, cfg, space, q)
    p.camera_position = [(400, -460, 240), (0, 0, -5), (0, 0, 1)]
    p.camera.view_angle = 30
    image = p.screenshot(return_img=True)
    screen = project(p, anchors)
    p.close()
    annotate(image, screen, comps, args.out / "s09-machine-bench.png",
             "Proposed bench: vertical optical axis through the shaping cell", footnote)
    # 3. Turntable of the module.
    frames = []
    p = pv.Plotter(off_screen=True, window_size=(1200, 1000))
    p.set_background("#f6f7f9", top="#ffffff")
    p.enable_anti_aliasing("ssaa")
    p.enable_lightkit()
    module(p, comps, cfg, space, q)
    for k in range(args.frames):
        angle = 2 * np.pi * k / args.frames
        p.camera_position = [(66 * np.cos(angle - 0.72), 66 * np.sin(angle - 0.72), 30), (0, 0, 0.0), (0, 0, 1)]
        p.camera.view_angle = 26
        frames.append(p.screenshot(return_img=True))
    p.close()
    imageio.mimsave(args.out / "s09-machine-module-turntable.mp4", frames, fps=24, quality=8, macro_block_size=1)
    (args.out / "machine-provenance.json").write_text(json.dumps({
        "inputs_sha256": inputs_used,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "components": {k: v["status"] for k, v in comps.items()},
        "scope": proposal["scope"],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
