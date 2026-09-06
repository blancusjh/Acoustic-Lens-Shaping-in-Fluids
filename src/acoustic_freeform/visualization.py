"""PyVista views of distinct physical fields, with explicit display units.

VTK exports retain metres, pascals and metres/second. Only presentation views
convert to millimetres/micrometres and exaggerate the surface's vertical axis.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyvista as pv

from .simulation import Result


def surface_mesh(result: Result, frame: int) -> pv.StructuredGrid:
    g = result.grid
    mesh = pv.StructuredGrid(g.xx.T, g.yy.T, result.height_m[frame].T)
    mesh["height_m"] = result.height_m[frame].ravel()
    mesh["normal_velocity_m_s"] = result.velocity_m_s[frame].ravel()
    mesh["radiation_pressure_pa"] = result.pressure(frame).ravel()
    return mesh


def export_vtk(result: Result, directory: Path, frame: int) -> None:
    """Surface and acoustic slice at the specified frame, entirely in SI."""
    directory.mkdir(parents=True, exist_ok=True)
    surface_mesh(result, frame).save(directory / "surface.vts")
    g, ac = result.grid, result.acoustic_model
    solution = ac.solve(result.pressure_array.spectrum(result.combined_weights(frame)))
    z = np.linspace(-result.experiment.array.source_distance_m, 0, 81)
    xg, zg = np.meshgrid(g.x, z, indexing="ij")
    field = pv.StructuredGrid(xg, np.zeros_like(xg), zg)
    pressure, velocity = [], []
    for level in z:
        sample = ac.sample(solution, level, medium="lower")
        pressure.append(sample.pressure[len(g.y) // 2])
        velocity.append(sample.velocity[:, len(g.y) // 2, :].T)
    p = np.asarray(pressure).ravel()
    field["pressure_real_pa"] = p.real
    field["pressure_imag_pa"] = p.imag
    field["pressure_peak_amplitude_pa"] = np.abs(p)
    v = np.asarray(velocity).reshape(-1, 3)
    field["acoustic_velocity_real_m_s"] = v.real
    field["acoustic_velocity_imag_m_s"] = v.imag
    field.save(directory / "acoustic_slice.vts")
    (directory / "README.txt").write_text(
        f"Frame {frame}; time {result.time_s[frame]:.9g} s. All fields and coordinates SI.\n"
        "Surface z is actual height; no vertical exaggeration.\n"
        "Acoustic slice is y=0 below the flat reference interface.\n"
        "Complex acoustic fields are peak phasors, with exp(-i omega t).\n"
    )


class Dashboard:
    def __init__(
        self,
        result: Result,
        off_screen: bool = True,
        window_size: tuple[int, int] = (1600, 1120),
        exaggeration: float = 1800,
    ):
        self.result, self.exaggeration = result, exaggeration
        self.plotter = pv.Plotter(
            shape=(2, 2), off_screen=off_screen, window_size=window_size, border=False
        )
        self.plotter.set_background("#101e30", all_renderers=True)
        self.color = "#eaf2f6"
        self.bar = {
            "color": self.color,
            "title_font_size": 14,
            "label_font_size": 12,
            "vertical": False,
            "position_x": 0.14,
            "position_y": 0.045,
            "width": 0.72,
            "height": 0.055,
        }
        g = result.grid
        self.ix = np.flatnonzero(np.abs(g.x) <= 0.0085)
        self.iy = np.flatnonzero(np.abs(g.y) <= 0.0085)
        self.xx, self.yy = np.meshgrid(g.x[self.ix] * 1000, g.y[self.iy] * 1000)
        self.hlimit = max(float(np.max(np.abs(result.height_m))) * 1e6, 1e-8)
        self.plimit = max(
            max(float(np.max(np.abs(result.pressure(i)))) for i in range(len(result.time_s))), 1e-8
        )
        self.vlimit = max(float(np.max(np.abs(result.velocity_m_s))) * 1000, 1e-8)
        self.peak_drive = int(np.argmax(np.sum(result.amplitudes**2, axis=1)))
        self.peak_shape = int(np.argmax(result.metrics["rms_height_m"]))
        driven = np.max(np.abs(result.amplitudes), axis=1)
        active = np.flatnonzero(driven >= 0.9 * max(float(driven.max()), 1e-30))
        self.peak_driven_shape = (
            int(active[np.argmax(result.metrics["rms_height_m"][active])])
            if len(active)
            else self.peak_shape
        )
        last_pulse = max((pulse.stop_s for pulse in result.experiment.pulses), default=0)
        after = np.flatnonzero((result.time_s >= last_pulse) & (driven < 1e-8))
        self.after_pulse = (
            int(after[np.argmax(result.metrics["rms_height_m"][after])])
            if len(after)
            else len(result.time_s) - 1
        )
        self._apparatus()
        self._traction()
        self._surface()
        self._flow()
        self.update(self.peak_shape)

    def _text(self, text: str, **kwargs):
        return self.plotter.add_text(text, font_size=13, color=self.color, **kwargs)

    def _crop(self, values):
        return values[np.ix_(self.iy, self.ix)]

    def _apparatus(self):
        p, r = self.plotter, self.result
        p.subplot(0, 0)
        self._text(
            "01 / ARRAY + FAST ACOUSTIC FIELD\n"
            f"{r.experiment.array.nx} x {r.experiment.array.ny} elements | "
            f"{r.experiment.array.frequency_hz / 1e6:g} MHz\n"
            "Fixed peak-drive view; physical geometry scale",
            position="upper_left",
        )
        a = r.pressure_array
        weights = r.combined_weights(self.peak_drive)
        width = r.experiment.array.element_width_m * 1000
        z0 = -r.experiment.array.source_distance_m * 1000
        points, faces, phases = [], [], []
        for y in range(a.yy.shape[0]):
            for x in range(a.xx.shape[1]):
                cx, cy = a.xx[y, x] * 1000, a.yy[y, x] * 1000
                n = len(points)
                points.extend(
                    [
                        (cx - width / 2, cy - width / 2, z0),
                        (cx + width / 2, cy - width / 2, z0),
                        (cx + width / 2, cy + width / 2, z0),
                        (cx - width / 2, cy + width / 2, z0),
                    ]
                )
                faces.extend([4, n, n + 1, n + 2, n + 3])
                phases.append(np.angle(weights[y, x]))
        tiles = pv.PolyData(np.asarray(points), faces)
        tiles.cell_data["phase [rad]"] = phases
        bar1 = {**self.bar, "position_x": 0.05, "width": 0.4}
        p.add_mesh(
            tiles,
            scalars="phase [rad]",
            cmap="twilight",
            clim=(-np.pi, np.pi),
            show_edges=True,
            edge_color="#142337",
            scalar_bar_args=bar1,
        )
        solution = r.acoustic_model.solve(a.spectrum(weights))
        z = np.linspace(z0 / 1000, 0, 81)
        pressure = np.stack(
            [
                np.abs(
                    r.acoustic_model.sample(solution, zz, "lower").pressure[
                        len(r.grid.y) // 2, self.ix
                    ]
                )
                / 1000
                for zz in z
            ]
        )
        xx, zz = np.meshgrid(r.grid.x[self.ix] * 1000, z * 1000, indexing="ij")
        slice_mesh = pv.StructuredGrid(xx, np.zeros_like(xx), zz)
        slice_mesh["|p| [kPa]"] = pressure.ravel()
        bar2 = {**self.bar, "position_x": 0.55, "width": 0.4}
        p.add_mesh(
            slice_mesh,
            scalars="|p| [kPa]",
            cmap="magma",
            clim=(0, max(float(np.max(pressure)), 1e-8)),
            scalar_bar_args=bar2,
        )
        rim = pv.Rectangle([(-8.5, -8.5, 0), (8.5, -8.5, 0), (8.5, 8.5, 0)])
        p.add_mesh(rim, color="#d8e9ee", opacity=0.08, show_edges=True)
        p.add_point_labels(
            [[8.5, 0, 0]],
            ["reference interface z=0"],
            font_size=12,
            text_color=self.color,
            show_points=False,
            always_visible=True,
            shape=None,
        )
        p.camera_position = [(27, -37, 29), (0, 0, -2.2), (0, 0, 1)]
        p.camera.zoom(1.13)
        p.add_axes(color=self.color, line_width=2, xlabel="x", ylabel="y", zlabel="z")

    def _traction(self):
        p = self.plotter
        p.subplot(0, 1)
        self.traction = pv.StructuredGrid(self.xx.T, self.yy.T, np.zeros_like(self.xx.T))
        self.traction["traction [Pa]"] = np.zeros(self.xx.size)
        p.add_mesh(
            self.traction,
            scalars="traction [Pa]",
            clim=(-self.plimit, self.plimit),
            cmap="coolwarm",
            scalar_bar_args=self.bar,
            lighting=False,
        )
        self._text(
            "02 / RADIATION TRACTION\n"
            "Momentum-flux jump | positive pushes upward\nx, y span +/-8.5 mm",
            position="upper_left",
        )
        self.traction_clock = self._text("", position=(20, 80), name="traction_clock")
        p.view_xy()
        p.camera.zoom(0.78)
        p.add_axes(color=self.color)

    def _surface(self):
        p = self.plotter
        p.subplot(1, 0)
        self.surface = pv.StructuredGrid(self.xx.T, self.yy.T, np.zeros_like(self.xx.T))
        self.surface["height [um]"] = np.zeros(self.xx.size)
        p.add_mesh(
            self.surface,
            scalars="height [um]",
            cmap="coolwarm",
            clim=(-self.hlimit, self.hlimit),
            scalar_bar_args=self.bar,
            smooth_shading=False,
            specular=0.15,
        )
        self._text(
            "03 / TRANSIENT SURFACE\n"
            f"x,y: mm | height colors: micrometres\nVertical geometry exaggerated x{self.exaggeration:g}",
            position="upper_left",
        )
        self.surface_clock = self._text("", position=(20, 80), name="surface_clock")
        p.camera_position = [(29, -35, 30), (0, 0, 0), (0, 0, 1)]
        p.camera.zoom(1.05)
        p.add_axes(color=self.color, xlabel="x", ylabel="y", zlabel="h (scaled)")

    def _flow(self):
        p, r = self.plotter, self.result
        p.subplot(1, 1)
        self._text(
            "04 / SLOW FLUID MOTION\n"
            f"{r.experiment.interface.model.replace('_', ' ')} reconstruction | y=0\n"
            "Arrow length fixed; colors show speed | no streaming",
            position="upper_left",
        )
        self.flow_z = np.linspace(-0.003, -0.00005, 25)
        self.flow_x = self.ix[::3]
        xx, zz = np.meshgrid(r.grid.x[self.flow_x] * 1000, self.flow_z * 1000, indexing="ij")
        self.flow_mesh = pv.StructuredGrid(xx, np.zeros_like(xx), zz)
        self.flow_mesh["speed [mm/s]"] = np.zeros(xx.size)
        p.add_mesh(
            self.flow_mesh,
            scalars="speed [mm/s]",
            clim=(0, self.vlimit),
            cmap="viridis",
            scalar_bar_args=self.bar,
            lighting=False,
        )
        seeds = self.flow_mesh.points[::4]
        self.seeds = pv.PolyData(seeds)
        self.seeds["vectors"] = np.zeros_like(seeds)
        self.seeds["speed"] = np.zeros(len(seeds))
        self.arrow_actor = None
        p.add_mesh(pv.Line((-8.5, 0, 0), (8.5, 0, 0)), color=self.color, line_width=2)
        p.camera_position = [(0, -28, -1.5), (0, 0, -1.5), (0, 0, 1)]
        p.camera.parallel_projection = True
        p.camera.parallel_scale = 6.4
        p.add_axes(color=self.color)

    def update(self, frame: float):
        r, p = self.result, self.plotter
        frame = int(np.clip(round(frame), 0, len(r.time_s) - 1))
        self.traction["traction [Pa]"] = self._crop(r.pressure(frame)).ravel()
        h = self._crop(r.height_m[frame])
        points = self.surface.points.copy()
        points[:, 2] = h.ravel() * 1000 * self.exaggeration
        self.surface.points = points
        self.surface["height [um]"] = h.ravel() * 1e6
        driven = bool(np.max(np.abs(r.amplitudes[frame])) > 1e-8)
        phase = "DRIVE ON" if driven else "DRIVE OFF / free evolution"
        clock = f"t = {r.time_s[frame] * 1000:6.2f} ms  |  {phase}"
        self.traction_clock.SetInput(clock)
        self.surface_clock.SetInput(clock)
        vhat = r.grid.forward(r.velocity_m_s[frame])
        flow = (
            np.stack(
                [
                    r.dynamics.flow_plane(vhat, z)[:, len(r.grid.y) // 2, self.flow_x].T
                    for z in self.flow_z
                ]
            ).reshape(-1, 3)
            * 1000
        )
        self.flow_mesh["speed [mm/s]"] = np.linalg.norm(flow, axis=1)
        self.seeds["vectors"] = flow[::4]
        self.seeds["speed"] = np.linalg.norm(flow[::4], axis=1)
        visible = np.flatnonzero(self.seeds["speed"] > self.vlimit * 1e-4)
        if len(visible):
            arrows = pv.PolyData(self.seeds.points[visible])
            arrows["vectors"] = flow[::4][visible]
            glyphs = arrows.glyph(orient="vectors", scale=False, factor=0.45)
        else:
            glyphs = pv.PolyData()
        p.subplot(1, 1)
        if self.arrow_actor is None:
            if glyphs.n_points:
                self.arrow_actor = p.add_mesh(glyphs, color="#ebf2f4", show_scalar_bar=False)
        else:
            self.arrow_actor.mapper.dataset = glyphs

    def save(self, directory: Path, movie: bool = True):
        directory.mkdir(parents=True, exist_ok=True)
        self.plotter.show(auto_close=False, interactive=False)
        self.update(self.after_pulse)
        self.plotter.render()
        self.plotter.screenshot(directory / "after_pulse.png")
        self.update(self.peak_driven_shape)
        self.plotter.render()
        self.plotter.screenshot(directory / "overview.png")
        self.plotter.export_html(directory / "explore.html")
        if movie:
            self.plotter.open_movie(str(directory / "evolution.mp4"), framerate=20, quality=8)
            frames = np.unique(
                np.linspace(
                    0, len(self.result.time_s) - 1, min(121, len(self.result.time_s))
                ).astype(int)
            )
            for frame in frames:
                self.update(int(frame))
                self.plotter.write_frame()
        self.plotter.close()

    def show(self):
        self.plotter.subplot(1, 0)
        self.plotter.add_slider_widget(
            self.update,
            (0, len(self.result.time_s) - 1),
            value=self.peak_shape,
            title="Time frame",
            pointa=(0.12, 0.20),
            pointb=(0.88, 0.20),
            color=self.color,
            fmt="%0.0f",
            interaction_event="always",
        )
        self.plotter.show()


def plot_diagnostics(result: Result, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = result
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, layout="constrained")
    t = r.time_s * 1000
    for i, b in enumerate(r.experiment.beams):
        axes[0].plot(t, r.amplitudes[:, i], label=b.name)
    axes[0].set_ylabel("Pressure envelope\n(relative amplitude)")
    axes[0].legend()
    axes[1].plot(t, r.metrics["max_height_m"] * 1e6, label="highest point")
    axes[1].plot(t, r.metrics["min_height_m"] * 1e6, label="lowest point")
    axes[1].plot(t, r.metrics["rms_height_m"] * 1e6, label="domain RMS", linestyle="--")
    axes[1].set_ylabel("Surface height [µm]")
    axes[1].legend(ncol=3)
    axes[2].plot(t, r.metrics["potential_energy_j"] * 1e12, label="capillary + gravity")
    axes[2].plot(t, r.metrics["kinetic_energy_j"] * 1e12, label="kinetic (zero in Stokes limit)")
    axes[2].set_ylabel("Energy [pJ]")
    axes[2].set_xlabel("Time [ms]")
    axes[2].legend()
    fig.suptitle(
        r.experiment.name + "\nLinear forward reference; no experimental calibration", fontsize=14
    )
    fig.savefig(path, dpi=170)
    plt.close(fig)


def render(result: Result, directory: Path, movie: bool = True):
    directory.mkdir(parents=True, exist_ok=True)
    dashboard = Dashboard(result)
    peak_drive = dashboard.peak_drive
    peak_shape = dashboard.peak_shape
    dashboard.save(directory, movie=movie)
    plot_diagnostics(result, directory / "diagnostics.png")
    export_vtk(result, directory / "vtk_peak_shape", peak_shape)
    export_vtk(result, directory / "vtk_peak_drive", peak_drive)
    metadata = {
        "input_result_sha256": result.report.get("result_sha256"),
        "visualization_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "overview_time_s": float(result.time_s[dashboard.peak_driven_shape]),
        "after_pulse_time_s": float(result.time_s[dashboard.after_pulse]),
        "fixed_acoustic_panel_time_s": float(result.time_s[peak_drive]),
        "surface_vertical_exaggeration": dashboard.exaggeration,
        "vtk_units": "SI, physical coordinates, no exaggeration",
        "movie_written": movie,
    }
    (directory / "render_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
