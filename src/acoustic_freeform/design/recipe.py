"""From a desired stigmatic singlet to the machine settings that form it.

``design`` places the lens in the fixed machine, checks that it fits, computes
the pin heights and fill volumes, finds a drive command (one tone at 1.5 MHz
first, one tone per band at 0.75 and 1.5 MHz if needed) and follows the
nonlinear start-up ramp from the liquid at rest. The returned ``Recipe`` says
what was established and what was not. Acceptance of a physical design still
needs the independent checks (force-convergence, coupled-stability, form);
see docs/design-a-lens.md.
"""

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.provenance import capture_execution
from ..steps import capability_synthesis as synthesis
from .machine import Machine, Singlet

STATUS = {
    "formed": "Formed from rest within the goal in the start-up ramp. Not yet independently verified.",
    "command-only": "A drive command holds the lens in the frozen model; the start-up ramp was not run.",
    "start-up-fails": "The command holds the lens, but the ramp from rest ends on another equilibrium.",
    "needs-more-source": "Reachable in principle, but no drive option of this machine reaches the goal.",
    "not-reachable": "The convex bound shows no command of this machine holds the lens within the goal.",
    "does-not-fit": "The lens does not fit the cell of this machine.",
}
SCOPE = (
    "Numerical model: axisymmetric, quiescent, creeping flow, lossless acoustics, ideal velocity "
    "sources. Not an experiment. Streaming, heating, non-axisymmetric disturbances, transducer "
    "realization and curing are not assessed."
)


@dataclass
class Recipe:
    status: str
    lens: dict
    machine: str
    settings: dict = field(default_factory=dict)
    prediction: dict = field(default_factory=dict)
    attempts: list = field(default_factory=list)
    message: str = ""
    seconds: float = 0.0
    _problem: dict | None = field(default=None, repr=False)
    _result: dict | None = field(default=None, repr=False)

    def as_dict(self):
        return {
            "status": self.status,
            "meaning": STATUS[self.status],
            "message": self.message,
            "lens": self.lens,
            "machine": self.machine,
            "settings": self.settings,
            "prediction": self.prediction,
            "attempts": self.attempts,
            "verification": "No independent checks run. Run force-convergence, coupled-stability and "
            "form on the saved design for acceptance (docs/design-a-lens.md).",
            "scope": SCOPE,
            "seconds": round(self.seconds, 1),
        }

    def save(self, output):
        """Write recipe.json and, when a command exists, the held-command design layout."""
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        capture_execution(output)
        if self._problem is not None and self._result is not None:
            synthesis.write(output, self._problem, self._result)
        (output / "recipe.json").write_text(json.dumps(self.as_dict(), indent=2, default=_plain) + "\n")
        return output


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def fit_problems(problem, acceptance):
    """Reasons the lens does not fit the cell (empty when it fits)."""
    cfg, targets = problem["cfg"], problem["targets"]
    levels, clearance = cfg.levels_m, acceptance["window_clearance_m"]
    r = np.linspace(0, cfg.radius_m, 801)
    h = [levels[j + 1] + targets[j].evaluate(r) for j in range(2)]
    problems = []
    if not levels[0] + clearance < levels[1] < levels[2] < levels[3] - clearance:
        problems.append("pin heights are out of order or too close to a window")
    if np.min(h[1] - h[0]) < acceptance["minimum_lens_thickness_m"]:
        problems.append("the faces come closer than the minimum lens thickness")
    if np.min(h[0]) < levels[0] + clearance:
        problems.append("the front face comes too close to the bottom window")
    if np.max(h[1]) > levels[3] - clearance:
        problems.append("the back face comes too close to the top window")
    return problems


def fill_volumes(cfg):
    """Liquid volume of each layer (m^3) for the pinned target faces."""
    area = math.pi * cfg.radius_m**2
    levels, offsets = cfg.levels_m, cfg.volume_offsets_m3
    return [
        area * (levels[1] - levels[0]) + offsets[0],
        area * (levels[2] - levels[1]) - offsets[0] + offsets[1],
        area * (levels[3] - levels[2]) - offsets[1],
    ]


def spot(problem, q):
    """Geometric spot on the design image plane over the full clear aperture."""
    from ..optics.raytrace import trace_pair

    cfg, targets, geometry = problem["cfg"], problem["targets"], problem["geometry"]
    indices = problem["inputs"]["singlet"]["indices"]
    z0, _, z2 = geometry["conjugates_m"]
    r = np.linspace(0, cfg.clear_radius_m, 401)
    trace = trace_pair(problem["space"], q, indices, z0, z2, launch_radius_m=r,
                       launch_height_m=cfg.levels_m[1] + targets[0].evaluate(r))
    f_number = (z2 - cfg.levels_m[2]) / (2 * cfg.clear_radius_m)
    return {
        "spot_max_radius_m": trace["max_radius_m"],
        "spot_rms_radius_m": trace["rms_radius_m"],
        "airy_radius_m_at_587_6_nm": 1.22 * 587.6e-9 * f_number,
        "working_f_number": f_number,
    }


def _settings(problem, result, material):
    cfg = problem["cfg"]
    summary = result["summary"]
    names = material.get("material_identity", {})
    volumes = fill_volumes(cfg)
    return {
        "pin_heights_above_bottom_window_m": [cfg.levels_m[1] - cfg.levels_m[0], cfg.levels_m[2] - cfg.levels_m[0]],
        "pin_heights_z_m": [cfg.levels_m[1], cfg.levels_m[2]],
        "fill_volumes_m3": {"lower layer": volumes[0], "lens layer": volumes[1], "upper layer": volumes[2]},
        "liquids": names or material.get("scope", ""),
        "drive": {
            "bands_hz": summary["bands_hz"],
            "channels": cfg.channels,
            "channel_commands_file": "design/source-commands.csv",
            "source_regions_file": "design/source-regions.json",
            "source_power_w": summary["source_power_w"],
            "band_peak_pressure_pa": summary["band_peak_pressure_pa"],
            "ramp": "all bands together, amplitude from 0 to 1 over the start-up (sin^2 over 1 s in form)",
        },
    }


def design(lens, machine=None, *, startup=True, progress=print):
    """Machine settings that form ``lens`` (a ``Singlet`` or its dict) on ``machine``."""
    started = time.time()
    lens = lens if isinstance(lens, Singlet) else Singlet.from_dict(lens)
    if not isinstance(machine, Machine):
        machine = Machine.load(machine) if machine else Machine.load()
    material = json.loads(Path(_material_path(machine)).read_text())
    attempts, best = [], None
    for bands in machine.drive_options_hz:
        progress(f"drive option {[f / 1e6 for f in bands]} MHz: building the cell")
        try:
            problem = synthesis.build(machine.synthesis_inputs(lens, bands))
        except (ValueError, np.linalg.LinAlgError) as error:
            return _recipe("does-not-fit", lens, machine, started, message=str(error))
        problems = fit_problems(problem, machine.acceptance)
        if problems:
            return _recipe("does-not-fit", lens, machine, started, message="; ".join(problems))
        progress("  convex bound and one tone per band")
        commanded = synthesis.command(problem)
        summary = commanded["summary"]
        attempt = {
            "bands_hz": bands,
            "convex_bound_m": summary["convex_pupil_error_m"],
            "frozen_error_m": summary["refined_frozen_pupil_error_m"],
        }
        attempts.append(attempt)
        if summary["convex_pupil_error_m"] > machine.acceptance["formed_goal_m"]:
            attempt["outcome"] = "convex bound above the goal"
            continue
        if summary["refined_frozen_pupil_error_m"] > machine.acceptance["accept_command_m"]:
            attempt["outcome"] = "one tone per band above the acceptance threshold"
            best = best or (problem, synthesis.settle(problem, commanded, startup=False))
            continue
        if not startup:
            attempt["outcome"] = "accepted; start-up ramp not run"
            return _finish("command-only", lens, machine, started, problem,
                           synthesis.settle(problem, commanded, startup=False), material, attempts)
        progress("  start-up ramp from the liquid at rest (minutes)")
        result = synthesis.settle(problem, commanded, startup=True)
        formed = result["summary"]["pupil_max_error_m"]
        attempt["start_up_error_m"] = formed
        attempt["start_up_reached_end"] = result["summary"]["branch_reached_end"]
        if result["summary"]["branch_reached_end"] and max(formed) <= machine.acceptance["formed_goal_m"]:
            attempt["outcome"] = "formed"
            return _finish("formed", lens, machine, started, problem, result, material, attempts)
        attempt["outcome"] = "start-up ramp did not end on the target"
        best = (problem, result)
    if best is None:
        return _recipe("not-reachable", lens, machine, started, attempts=attempts,
                       message="no drive option has a convex bound within the goal")
    status = "start-up-fails" if any("start_up_error_m" in a for a in attempts) else "needs-more-source"
    return _finish(status, lens, machine, started, *best, material, attempts)


def _material_path(machine):
    from ..core.paths import data_path

    return data_path(Path(machine.material_reference))


def _recipe(status, lens, machine, started, attempts=(), message=""):
    return Recipe(status=status, lens=lens.as_dict(), machine=machine.name, attempts=list(attempts),
                  message=message or STATUS[status], seconds=time.time() - started)


def _finish(status, lens, machine, started, problem, result, material, attempts):
    summary = result["summary"]
    prediction = {
        "formed_pupil_error_m": summary["pupil_max_error_m"],
        "acoustics_off_pupil_error_m": summary["acoustics_off_pupil_error_m"],
        "frozen_pupil_error_m": summary["refined_frozen_pupil_error_m"],
        "convex_bound_m": summary["convex_pupil_error_m"],
        "fold_margin_min": summary.get("branch_min_static"),
        "start_up_reached_end": summary.get("branch_reached_end"),
    }
    if result["startup"]:
        prediction.update(spot(problem, result["end"]))
    return Recipe(status=status, lens=lens.as_dict(), machine=machine.name,
                  settings=_settings(problem, result, material), prediction=prediction,
                  attempts=attempts, message=STATUS[status], seconds=time.time() - started,
                  _problem=problem, _result=result)
