"""Faithful two-face synthesis with a low-order modal source (theory section 10e).

Fits the modal coefficients so that the end of the quasi-static start-up
branch from the unforced state, not the target geometry itself, matches both
pupils; declared parameter tolerances enter as first-order equilibrium
shifts; source power and sampled peak pressure are hard caps. The design is
written in the held-command layout (``design/state.npz``) so that
``force-convergence``, ``coupled-stability``, ``startup-branch`` and ``form``
apply to it unchanged. Acceptance remains those independent checks.
"""

import json
from dataclasses import replace

import numpy as np

from ..apparatus.config import DualConfig
from ..apparatus.sources import export_sources, modal_source_matrix
from ..core.paths import data_path
from ..core.provenance import capture_execution
from ..inverse.faithful import FaithfulProblem, best_fill_volumes, frozen_start
from ..mechanics.surface import CartesianPatch, singlet_geometry, targets_from_record
from ..mechanics.viscous import assemble_mobility
from ..optics.stigmatic import pair_prescription


def run(configuration, output):
    inputs = json.loads(configuration.read_text())
    material = json.loads(data_path(inputs["material_reference"]).read_text())
    apparatus = dict(inputs["apparatus"])
    fill = apparatus.pop("volume_offsets_m3", None)
    if "singlet" in inputs:
        # A prescribed singlet fixes the lens; the cell (pinning ring and levels)
        # is derived from it.
        geometry = singlet_geometry(inputs["singlet"])
        apparatus["radius_m"] = inputs["singlet"]["pin_radius_m"]
        apparatus["clear_radius_m"] = inputs["singlet"]["clear_radius_m"]
        apparatus["levels_m"] = geometry["levels_m"]
        cfg = DualConfig(**{**material["apparatus"], **apparatus})
        targets = targets_from_record(cfg, inputs)
        optics = {
            "indices": inputs["singlet"]["indices"],
            # Strict JSON: an object at infinity is stored as "-inf" (read with float()).
            "stigmatic_z_m": [z if np.isfinite(z) else str(z) for z in geometry["conjugates_m"]],
            "thickness_m": geometry["thickness_m"],
        }
        faces = None
    else:
        cfg = DualConfig(**{**material["apparatus"], **apparatus})
        optics = {k: material[k] for k in ("indices", "stigmatic_z_m", "vertex_displacement_m")}
        optics.update(inputs.get("optics", {}))
        faces = pair_prescription(
            cfg, optics["indices"], optics["stigmatic_z_m"], optics["vertex_displacement_m"]
        )
        for face in faces:
            face["annulus_match_order"] = inputs.get("annulus_match_order", 4)
        targets = [CartesianPatch(cfg, j, **f) for j, f in enumerate(faces)]
    pupil = np.linspace(0, cfg.clear_radius_m, inputs.get("pupil_samples", 41))
    meniscus_errors = None
    if fill == "best_fit":
        fill, meniscus_errors = best_fill_volumes(cfg, targets, pupil)
    elif fill == "target" and "singlet" in inputs:
        # The lens's own displaced volumes: the pumps fill each layer to the target.
        fill = [t.volume_offset() for t in targets]
    if fill is not None:
        cfg = replace(cfg, volume_offsets_m3=tuple(fill))
    modes, labels = modal_source_matrix(
        cfg, inputs["modes"]["window_modes"], inputs["modes"]["wall_modes"]
    )
    output.mkdir(parents=True, exist_ok=False)
    capture_execution(output)
    record = {
        **inputs,
        "apparatus": cfg.as_dict(),
        "case": {"faces": faces},
        "acoustics_off_meniscus_pupil_error_m": meniscus_errors,
        "optics": optics,
        "source_file": str(output / "best-source.npz"),
    }
    (output / "config.json").write_text(json.dumps(record, indent=2) + "\n")
    (output / "modes.json").write_text(json.dumps(labels, indent=2) + "\n")
    export_sources(output, cfg)
    fluid = inputs.get("fluid_mesh")
    viscosity = np.array(material["viscosities_pa_s"])
    space_holder = {}

    def mobility(q):
        return assemble_mobility(
            space_holder["space"], q, viscosity, fluid["radial_cells"], fluid["cells_per_layer"]
        ).mobility

    problem = FaithfulProblem(
        cfg,
        modes,
        targets,
        pupil,
        inputs["height_goal_m"],
        inputs["sigmas"],
        perturbations=inputs.get("perturbations", []),
        mobility=mobility if fluid else None,
    )
    space_holder["space"] = problem.space
    limits = inputs["limits"]
    rng = np.random.default_rng(inputs.get("seed", 0))
    if inputs.get("initial_command", "frozen") == "convex":
        starts = convex_start(problem, modes, limits, inputs.get("stability_margin"))
    else:
        starts = frozen_start(problem, limits, inputs.get("seeds", 4), rng)
    (output / "starts.json").write_text(
        json.dumps([{k: v for k, v in s.items() if k != "coefficients"} for s in starts], indent=2)
        + "\n"
    )
    history = []
    current = None
    for start in starts:
        branch = problem.reachable(start["coefficients"])
        entry = {
            "start": True,
            "reached_end": branch["reached_end"],
            "reason": branch["stop_reason"],
        }
        history.append(entry)
        if branch["reached_end"]:
            current = (start["coefficients"], branch)
            break
    if current is None:
        raise RuntimeError("No frozen start reaches sigma = 1 along its start-up branch")
    c, branch = current
    q, operators, blocks = problem.linearize(c, branch)
    radius = inputs.get("initial_radius", 0.3)
    fixed = [modes.shape[1]]  # imaginary part of the first mode: global phase
    best = None

    def summary(c, q, operators, blocks, branch):
        residual = {name: float(np.max(abs(b[0])) * problem.scale) for name, b in blocks.items()}
        return {
            "pupil_max_error_m": residual["shape"],
            "perturbation_max_shift_m": {k: v for k, v in residual.items() if k != "shape"},
            "merit": float(sum(b[0] @ b[0] for b in blocks.values())),
            "source_power_w": operators.source_power(c),
            "peak_pressure_pa": operators.peak_pressure(c),
            "branch_min_static": min(p["static_min_real"] for p in branch["points"]),
            "branch_max_growth_rate_s": max(
                (p["max_growth_rate_s"] for p in branch["points"] if "max_growth_rate_s" in p),
                default=None,
            ),
            "coefficient_peak_m_s": float(np.max(abs(c))),
        }

    state = summary(c, q, operators, blocks, branch)
    for iteration in range(inputs.get("max_iterations", 20)):
        state["iteration"] = iteration
        history.append(state)
        print(json.dumps(state), flush=True)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if best is None or state["merit"] < best[0]["merit"]:
            best = (state, c, q, branch)
            save(output, cfg, modes, c, q, state)
        if state["pupil_max_error_m"] <= inputs["height_goal_m"] and all(
            v <= inputs["height_goal_m"] for v in state["perturbation_max_shift_m"].values()
        ):
            break
        proposal, predicted, _ = problem.step(c, operators, blocks, radius, limits, fixed)
        trial_branch = problem.reachable(proposal)
        if not trial_branch["reached_end"]:
            radius *= 0.5
            history.append({"rejected": "branch did not reach sigma = 1", "radius": radius})
            continue
        trial_q, trial_operators, trial_blocks = problem.linearize(proposal, trial_branch)
        trial_state = summary(proposal, trial_q, trial_operators, trial_blocks, trial_branch)
        actual = 0.5 * trial_state["merit"]
        current_merit = 0.5 * state["merit"]
        ratio = (current_merit - actual) / max(current_merit - predicted, 1e-300)
        if ratio > 0:
            c, q, operators, blocks, branch, state = (
                proposal,
                trial_q,
                trial_operators,
                trial_blocks,
                trial_branch,
                trial_state,
            )
            radius = radius * 2 if ratio > 0.75 else radius
        else:
            radius *= 0.25
            history.append({"rejected": "merit increased", "ratio": ratio, "radius": radius})
        if radius < 1e-6:
            history.append({"stop": "trust radius below 1e-6"})
            break
    if best is None:
        history.append(state)
        best = (state, c, q, branch)
        save(output, cfg, modes, c, q, state)
    state, c, q, branch = best
    report = {
        "best": state,
        "modes": len(labels),
        "limits": limits,
        "height_goal_m": inputs["height_goal_m"],
        "acceptance": "Requires force-convergence, coupled-stability, startup-branch and form checks",
        "model": "Axisymmetric lossless longitudinal acoustics with passive ports; creeping flow; no streaming or heating",
    }
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "report.md").write_text(
        "# Faithful modal synthesis\n\nDesign-grid values; acceptance needs the independent "
        "checks.\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n"
    )


def convex_start(problem, modes, limits, stability_margin=None):
    """Initial command from the convex multi-tone capability program at the target.

    The semidefinite relaxation is global, so it avoids the local minima of a
    frozen fit from random seeds; its leading eigenvector is refined to a single
    tone and scaled into the caps. With ``stability_margin`` mu, the program
    and the refinement require sym(K - A) >= mu K, a static fold margin of mu
    (theory section 10f). Acceptance is still the nonlinear branch.
    """
    from ..inverse.capability import Tone, best_tolerance, single_tone_refinement
    from ..mechanics.demand import demanded_force

    q, demand, stiffness = demanded_force(problem.space, problem.targets)
    observe = np.kron(np.eye(2), problem.pupil_basis) @ np.linalg.inv(stiffness)
    tone = Tone(problem.cfg, problem.space, q, modes)
    stability = None
    if stability_margin is not None:
        tone.shape_derivatives()
        stability = (stiffness, stability_margin)
    convex = best_tolerance([tone], observe, demand, limits["source_power_w"], stability=stability)
    lead = convex["tones"][0]["amplitudes"][:, 0]
    single = single_tone_refinement(
        tone,
        observe,
        demand,
        lead,
        limits["source_power_w"],
        pressure_cap_pa=limits["peak_pressure_pa"],
        stability=stability,
    )
    c = single["amplitudes"]
    factor = min(
        1.0,
        np.sqrt(limits["source_power_w"] / max(single["power_w"], 1e-30)),
        limits["peak_pressure_pa"] / max(single["peak_pressure_pa"], 1e-30),
    )
    return [
        {
            "coefficients": c * factor,
            "convex_multitone_pupil_error_m": convex["pupil_max_error_m"],
            "convex_rank": convex["tones"][0]["rank"],
            "single_tone_frozen_pupil_error_m": single["pupil_max_error_m"],
            "single_tone_power_w": single["power_w"],
            "single_tone_peak_pressure_pa": single["peak_pressure_pa"],
            "single_tone_fold_margin": single["fold_margin"],
            "required_fold_margin": stability_margin,
            "cap_scale_factor": float(factor),
        }
    ]


def save(output, cfg, modes, c, q, state):
    design = output / "design"
    design.mkdir(exist_ok=True)
    drive = modes @ c
    np.savez_compressed(
        output / "best-source.npz", source_velocity_m_s=drive, modal_coefficients_m_s=c
    )
    np.savez_compressed(design / "state.npz", coefficients_m=q, source_velocity_m_s=drive)
    (design / "config.json").write_text(json.dumps(cfg.as_dict(), indent=2) + "\n")
    (design / "summary.json").write_text(json.dumps(state, indent=2) + "\n")
    export_sources(design, cfg, drive)
