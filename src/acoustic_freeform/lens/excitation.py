"""Export physical array excitations and the force needed to hold a diopter."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .acoustics import CavityAcoustics
from .config import LensConfig
from .surface import SurfaceSpace


def normal_shape_load(space, coefficients, radial_m):
    """sigma*kappa + rho*g*h, before the constant pressure/volume gauge.

    Curvature is positive on a convex liquid cap, normal pointing into air.
    The radius-zero expression is the smooth axisymmetric limit.
    """
    cfg = space.config
    r = np.asarray(radial_m, float)
    x = r / cfg.radius_m
    h = cfg.radius_m * space.evaluate(coefficients, x)
    slope = space.evaluate(coefficients, x, 1)
    second = space.evaluate(coefficients, x, 2) / cfg.radius_m
    azimuthal = np.divide(slope, r, out=second.copy(), where=r != 0)
    curvature = -(second / (1 + slope**2) ** 1.5 + azimuthal / np.sqrt(1 + slope**2))
    return cfg.surface_tension_n_m * curvature + cfg.density_kg_m3 * cfg.gravity_m_s2 * h


def export_excitation(result_directory):
    result = Path(result_directory)
    cfg = LensConfig(**json.loads((result / "configuration.json").read_text()))
    trajectory = np.load(result / "trajectory.npz")
    report = json.loads((result / "report.json").read_text())
    space = SurfaceSpace(cfg)
    target, final = trajectory["target"], trajectory["coefficients"][-1]
    drives, times = trajectory["drive_m_s"], trajectory["times_s"]
    frequency, omega = cfg.frequency_hz, 2 * np.pi * cfg.frequency_hz
    destination = result / "excitation"
    destination.mkdir(exist_ok=True)
    names = [
        "time_s",
        "row_from_bottom_1based",
        "row_center_z_m",
        "frequency_hz",
        "velocity_real_m_s",
        "velocity_imag_m_s",
        "velocity_peak_m_s",
        "phase_deg",
        "phase_relative_to_row1_deg",
        "displacement_peak_m",
    ]

    def records(indices):
        for k in indices:
            for row, w in enumerate(drives[k]):
                relative = np.angle(w * drives[k, 0].conjugate())
                yield [
                    times[k],
                    row + 1,
                    -cfg.depth_m + (row + 0.5) * cfg.depth_m / cfg.array_rows,
                    frequency,
                    w.real,
                    w.imag,
                    abs(w),
                    np.degrees(np.angle(w)),
                    np.degrees(relative),
                    abs(w) / omega,
                ]

    for filename, indices in (
        ("hold-drive.csv", [len(times) - 1]),
        ("drive-program.csv", range(len(times))),
    ):
        with (destination / filename).open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(names)
            writer.writerows(records(indices))

    # Compare continuum shape loads with the independently solved acoustic traction.
    field = CavityAcoustics(cfg, space).solve_basis(final)
    radial = cfg.radius_m * field.radial_samples
    weights = field.quadrature_weights
    radiation = field.surface_pressure(drives[-1], cfg.density_kg_m3)
    load = normal_shape_load(space, final, radial)
    target_load = normal_shape_load(space, target, radial)
    gauge = float(np.average(load - radiation, weights=weights))
    mismatch = load - radiation - gauge
    sample = np.linspace(0, cfg.radius_m, 501)
    h = lambda c: cfg.radius_m * space.evaluate(c, sample / cfg.radius_m)
    columns = np.c_[
        sample,
        h(trajectory["initial"]),
        h(target),
        h(final),
        h(target) - h(trajectory["initial"]),
        normal_shape_load(space, target, sample),
    ]
    np.savetxt(
        destination / "surface-and-load.csv",
        columns,
        delimiter=",",
        header="radius_m,initial_height_m,target_height_m,computed_height_m,target_perturbation_m,target_sigma_kappa_plus_rho_g_h_pa",
        comments="",
    )
    order = np.argsort(radial)
    np.savetxt(
        destination / "traction-balance.csv",
        np.c_[radial, radiation, load, target_load, mismatch][order],
        delimiter=",",
        header="radius_m,acoustic_traction_pa,computed_shape_load_pa,target_shape_load_pa,computed_balance_residual_after_constant_gauge_pa",
        comments="",
    )
    metadata = {
        "diopter": cfg.diopter.as_dict(),
        "target_vertex_above_rim_m": space.optical_vertex_m,
        "liquid_volume_m3": report["liquid_volume_m3"],
        "input_wavefront": report.get("illumination", "Collimated inside the resin."),
        "frequency_hz": frequency,
        "phasor_convention": "v_normal(z,t)=taper(z)*Re[w_row*exp(-i*2*pi*f*t)]; positive normal points radially out of the liquid.",
        "amplitude_convention": "Peak normal wall velocity in m/s at the center of the axial cosine taper; not RMS, pressure, or electrical voltage.",
        "spatial_taper": "0.5*(1+cos(pi*(z-z_center)/half_width)) for |z-z_center|<half_width; zero elsewhere. half_width=element_fill*depth/(2*array_rows).",
        "azimuthal_control": f"All {cfg.array_sectors} sectors of a row share one excitation; {cfg.array_rows} independent coherent rows.",
        "program_convention": "Zero-order hold: each recorded complex drive is applied until the next timestamp. The last row-drive set is the holding excitation.",
        "phase_nonuniqueness": "One common phase rotation leaves the cycle-averaged radiation force unchanged. Relative phases matter; no uniqueness or global optimum is asserted.",
        "electrical_calibration": "Voltages require the complex loaded transducer transfer function, which is not known here.",
        "constant_pressure_gauge_pa": gauge,
        "continuum_balance_residual_rms_pa": float(
            np.sqrt(np.average(mismatch**2, weights=weights))
        ),
        "continuum_balance_residual_max_pa": float(np.max(abs(mismatch))),
        "continuum_balance_note": "Strong-form residual includes finite surface/acoustic discretization error. Modal force balance is the equation integrated by the solver.",
        "max_wall_velocity_peak_m_s": float(np.max(abs(drives[-1]))),
        "max_wall_displacement_peak_m": float(np.max(abs(drives[-1])) / omega),
        "source_trajectory_sha256": hashlib.sha256(
            (result / "trajectory.npz").read_bytes()
        ).hexdigest(),
    }
    (destination / "definition.json").write_text(json.dumps(metadata, indent=2) + "\n")
    render_excitation(result)
    return metadata


def render_excitation(result_directory):
    """Standalone scientific figure from exported physical data, no redesign."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = Path(result_directory)
    source = result / "excitation"
    metadata = json.loads((source / "definition.json").read_text())
    load = np.genfromtxt(source / "surface-and-load.csv", delimiter=",", names=True)
    traction = np.genfromtxt(source / "traction-balance.csv", delimiter=",", names=True)
    rows = np.genfromtxt(source / "hold-drive.csv", delimiter=",", names=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for key, label, color in [
        ("initial_height_m", "Unforced liquid", "#6b7e92"),
        ("target_height_m", "Cartesian target", "#d79d36"),
        ("computed_height_m", "Computed driven liquid", "#008f98"),
    ]:
        axes[0, 0].plot(load["radius_m"] * 1000, load[key] * 1000, label=label, color=color)
    axes[0, 0].set(
        xlabel="Radius (mm)", ylabel="Height above rim (mm)", title="Mounted optical surface"
    )
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(load["radius_m"] * 1000, load["target_perturbation_m"] * 1e6, color="#008f98")
    axes[0, 1].set(
        xlabel="Radius (mm)",
        ylabel="Target − unforced height (µm)",
        title="Required perturbation at conserved fill",
    )
    gauge = metadata["constant_pressure_gauge_pa"]
    axes[1, 0].plot(
        traction["radius_m"] * 1000,
        traction["target_shape_load_pa"] - gauge,
        color="#d79d36",
        label="Target shape load − fitted pressure constant",
    )
    axes[1, 0].plot(
        traction["radius_m"] * 1000,
        traction["acoustic_traction_pa"],
        color="#008f98",
        lw=1,
        label="Solved acoustic radiation traction",
    )
    axes[1, 0].set(
        xlabel="Radius (mm)",
        ylabel="Normal traction (Pa)",
        title="Radiation force on the actual cavity",
    )
    axes[1, 0].legend(fontsize=7)
    row = rows["row_from_bottom_1based"]
    axes[1, 1].bar(row, rows["velocity_peak_m_s"] * 1000, color="#008f98", alpha=0.75)
    phase = axes[1, 1].twinx()
    phase.plot(row, rows["phase_relative_to_row1_deg"], "o-", color="#b47a22", lw=1)
    phase.set(ylabel="Phase relative to row 1 (degrees)", ylim=(-190, 190))
    axes[1, 1].set(
        xlabel="Array row from bottom",
        ylabel="Peak wall velocity (mm/s)",
        title="Holding excitation: amplitude and phase",
    )
    for ax in axes.flat:
        ax.grid(alpha=0.15)
    d = metadata["diopter"]
    zo = d["z_o_m"]
    object_label = "collimated" if isinstance(zo, str) else f"zₒ={zo * 1000:g} mm"
    fig.suptitle(
        f"Cartesian diopter · nₒ={d['n_o']:g}, {object_label}, nᵢ={d['n_i']:g}, zᵢ=+{d['z_i_m'] * 1000:g} mm\n{metadata['frequency_hz'] / 1e6:g} MHz · prescribed incident wavefront in resin · nominal acoustic material model",
        fontsize=13,
    )
    out = result / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "cartesian-excitation.png", dpi=180)
    fig.savefig(out / "cartesian-excitation.pdf")
    plt.close(fig)
