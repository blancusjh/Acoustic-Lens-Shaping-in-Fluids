"""Common-source synthesis from the convex capability program (theory section 10f).

For a prescribed singlet in a declared cell, solves the multi-band capability
program at the target (optionally with the fold-margin inequality), refines it
to exactly one tone per band within the power and pressure caps, then follows
the nonlinear quasi-static start-up branch from the acoustics-off meniscus at
the target fill volumes with all bands applied together. The design is written
in the held-command layout (``design/state.npz`` with a (bands, channels) drive
and ``band_frequencies_hz``) so that ``force-convergence``, ``coupled-stability``
and ``form`` apply unchanged. Acceptance remains those independent checks.
"""

import json
from dataclasses import replace

import numpy as np

from ..acoustics.multiband import MultiBandAcoustics
from ..apparatus.config import DualConfig
from ..apparatus.sources import export_sources, modal_source_matrix
from ..core.paths import data_path
from ..core.provenance import capture_execution
from ..forward.branch import startup_branch
from ..inverse.capability import Tone, best_tolerance, tone_per_band_refinement
from ..mechanics.demand import demanded_force
from ..mechanics.surface import DualSurface, singlet_geometry, targets_from_record


def run(configuration, output):
    inputs = json.loads(configuration.read_text())
    material = json.loads(data_path(inputs["material_reference"]).read_text())
    singlet = inputs["singlet"]
    geometry = singlet_geometry(singlet)
    bands = [float(f) for f in inputs["bands_hz"]]
    apparatus = {
        **material["apparatus"],
        **inputs["apparatus"],
        "radius_m": singlet["pin_radius_m"],
        "clear_radius_m": singlet["clear_radius_m"],
        "levels_m": geometry["levels_m"],
        # The config frequency is the highest band (meshes are sized for it).
        "frequency_hz": max(bands),
    }
    cfg = DualConfig(**apparatus)
    targets = targets_from_record(cfg, inputs)
    cfg = replace(cfg, volume_offsets_m3=tuple(t.volume_offset() for t in targets))
    targets = targets_from_record(cfg, inputs)
    space = DualSurface(cfg, allow_volume_offsets=True)
    q, demand, stiffness = demanded_force(space, targets)
    pupil = np.linspace(0, cfg.clear_radius_m, inputs.get("pupil_samples", 41))
    observe = np.kron(np.eye(2), space.basis(pupil)) @ np.linalg.inv(stiffness)
    modes, labels = modal_source_matrix(
        cfg, inputs["modes"]["window_modes"], inputs["modes"]["wall_modes"]
    )
    limits = inputs["limits"]
    margin = inputs.get("stability_margin")
    output.mkdir(parents=True, exist_ok=False)
    capture_execution(output)
    tones = []
    for f in bands:
        tone = Tone(replace(cfg, frequency_hz=f), space, q, modes)
        if margin is not None:
            tone.shape_derivatives()
        tones.append(tone)
    stability = None if margin is None else (stiffness, margin)
    convex = best_tolerance(
        tones, observe, demand, limits["source_power_w"], stability=stability
    )
    starts = [t["amplitudes"][:, 0] for t in convex["tones"]]
    refined = tone_per_band_refinement(
        tones,
        observe,
        demand,
        starts,
        limits["source_power_w"],
        limits["peak_pressure_pa"],
        stability=stability,
    )
    coefficients = refined["amplitudes"]
    drive = np.array([modes @ c for c in coefficients])
    wave = MultiBandAcoustics(cfg, space, bands, linear_solver="static_condensed")
    start = space.unforced_equilibrium()
    branch = startup_branch(
        wave, space, drive, inputs["sigmas"], initial=start, tolerance_m=inputs.get("tolerance_m", 1e-13)
    )
    end = branch["points"][-1]["coefficients_m"]
    fine = np.linspace(0, cfg.clear_radius_m, 401)
    errors = [float(np.max(abs(space.height(j, end[j], fine) - targets[j].evaluate(fine)))) for j in range(2)]
    start_errors = [
        float(np.max(abs(space.height(j, start[j], fine) - targets[j].evaluate(fine)))) for j in range(2)
    ]
    diagnostics = wave.solve(end, drive).diagnostics(np.ones(1, complex))
    summary = {
        "bands_hz": bands,
        "modes_per_band": modes.shape[1],
        "convex_pupil_error_m": convex["pupil_max_error_m"],
        "convex_ranks": [t["rank"] for t in convex["tones"]],
        "refined_frozen_pupil_error_m": refined["pupil_max_error_m"],
        "refined_fold_margin": refined["fold_margin"],
        "acoustics_off_pupil_error_m": start_errors,
        "branch_reached_end": branch["reached_end"],
        "branch_stop_reason": branch["stop_reason"],
        "branch_min_static": min(p["static_min_real"] for p in branch["points"]),
        "pupil_max_error_m": errors,
        "source_power_w": diagnostics["source_power_w"],
        "band_source_power_w": [b["source_power_w"] for b in diagnostics["bands"]],
        "band_peak_pressure_pa": diagnostics["band_sampled_pressure_peak_pa"],
        "worst_case_peak_pressure_pa": diagnostics["sampled_pressure_peak_pa"],
    }
    optics = {
        "indices": singlet["indices"],
        "stigmatic_z_m": [z if np.isfinite(z) else str(z) for z in geometry["conjugates_m"]],
        "thickness_m": geometry["thickness_m"],
    }
    record = {**inputs, "apparatus": cfg.as_dict(), "optics": optics, "case": {"faces": None}}
    record["source_file"] = str(output / "best-source.npz")
    (output / "config.json").write_text(json.dumps(record, indent=2) + "\n")
    (output / "modes.json").write_text(json.dumps(labels, indent=2) + "\n")
    np.savez_compressed(
        output / "best-source.npz",
        source_velocity_m_s=drive,
        modal_coefficients_m_s=np.array(coefficients),
        band_frequencies_hz=np.array(bands),
    )
    design = output / "design"
    design.mkdir()
    np.savez_compressed(
        design / "state.npz",
        coefficients_m=end,
        source_velocity_m_s=drive,
        band_frequencies_hz=np.array(bands),
    )
    (design / "config.json").write_text(json.dumps(cfg.as_dict(), indent=2) + "\n")
    (design / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    export_sources(design, cfg, drive)
    (output / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "report.md").write_text(
        "# Capability synthesis\n\nConvex start, one tone per band, nonlinear start-up "
        "branch. Acceptance needs force-convergence, coupled-stability and form.\n\n"
        "```json\n" + json.dumps(summary, indent=2) + "\n```\n"
    )
    print(json.dumps(summary), flush=True)
