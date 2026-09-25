"""Audit the available single-interface data associated with the shared chat.

Independently evaluate its Cartesian error and recompute its final acoustic
field using the inspected archived solver. This does not rerun the missing
formation/control driver or validate physical material properties.
"""

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from acoustic_freeform.dual.surface import cartesian_derivatives
from acoustic_freeform.lens.cartesian import CartesianDiopter
from acoustic_freeform.provenance import capture_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    capture_execution(args.out)
    shutil.copy2(__file__, args.out / "provenance/audit-driver.py")
    files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.source.iterdir())
        if path.is_file()
    }
    (args.out / "acquisition.json").write_text(
        json.dumps(
            {
                "source_directory": str(args.source.resolve()),
                "original_directory": "/Users/blancus/Downloads/acoustic_blender_model/source_data",
                "acquired_utc": datetime.now(UTC).isoformat(),
                "sha256": files,
                "role": "Imported discussion evidence, not a new physical experiment.",
            },
            indent=2,
        )
        + "\n"
    )
    data = np.load(args.source / "etd2_spatial.npz")
    report = json.loads((args.source / "etd2_spatial.json").read_text())
    radius, aperture = 0.004, 0.002
    length = radius - aperture
    diopter = CartesianDiopter(1.4, -0.1, 1.5, 0.2)
    h, hp, hpp = cartesian_derivatives(diopter, np.array([aperture]))

    def annulus(shift):
        coeff = np.zeros(6)
        coeff[:3] = h[0] + shift, hp[0] * length, 0.5 * hpp[0] * length**2
        coeff[3:] = np.linalg.solve(
            [[1, 1, 1], [3, 4, 5], [6, 12, 20]],
            [-coeff[:3].sum(), -coeff[1] - 2 * coeff[2], -2 * coeff[2]],
        )
        return np.polynomial.Polynomial(coeff)

    x, w = leggauss(120)
    ri = aperture * (x + 1) / 2
    ro = aperture + length * (x + 1) / 2
    zero, one = annulus(0), annulus(1)
    volume0 = np.dot(w * ri, diopter.sag(ri)) * aperture / 2
    volume0 += np.dot(w * ro, zero((ro - aperture) / length)) * length / 2
    volume1 = aperture**2 / 2 + np.dot(w * ro, (one - zero)((ro - aperture) / length)) * length / 2
    shift = -volume0 / volume1
    r = np.linspace(0, aperture, 4001)
    target = diopter.sag(r) + shift
    radial_nodes = data["radial_nodes"]
    nr, degree = int(data["nr"]), 3
    element = np.minimum((r / radius * nr).astype(int), nr - 1)
    xi = r / radius * nr - element
    local_nodes = np.linspace(0, 1, degree + 1)
    basis = np.stack(
        [
            np.polynomial.Polynomial.fromroots(np.delete(local_nodes, j))(xi)
            / np.prod(local_nodes[j] - np.delete(local_nodes, j))
            for j in range(degree + 1)
        ],
        axis=1,
    )
    indices = element[:, None] * degree + np.arange(degree + 1)
    times = data["history"][:, 0]
    if np.any(np.diff(times) <= 0) or abs(times[-1] - report["duration"]) > 1e-9:
        raise ValueError("Unrecognized saved physical time coordinate.")
    heights = np.einsum("rj,trj->tr", basis, data["heights"][:, indices])
    maximum = np.max(abs(heights - target), axis=1)
    holding = times >= report["feedback_start"] - 1e-12
    # The entire archived module was inspected; its body defines numerical
    # classes/functions and sets only two BLAS threading defaults.
    spec = importlib.util.spec_from_file_location("imported_chat_solver", args.source / "solver.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cfg = module.Setup()
    graph = module.FEGraph(radius, nr, degree, data["heights"][-1])
    wave = module.AcousticFEM(cfg, nr=nr, nz=int(data["nz"]), p=degree)
    fields = wave.solve(graph, w=data["commands"][-1])
    pupil = wave.rq <= aperture
    carrier = np.abs(fields["v"][pupil]) * np.sqrt(1 + graph(wave.rq[pupil], 1) ** 2) / cfg.omega
    source_power, absorbed_power = wave.power(fields["Pall"], data["commands"][-1])
    result = {
        "fresh_formation_integrated": False,
        "fresh_final_acoustic_solve": True,
        "surfaces_controlled": 1,
        "radial_nodes": len(radial_nodes),
        "saved_frames": len(times),
        "target_vertex_m": float(shift),
        "initial_height_max_m": float(np.max(abs(data["heights"][0]))),
        "initial_source_max_m_s": float(np.max(abs(data["commands"][0]))),
        "saved_frame_holding_max_error_m": float(np.max(maximum[holding])),
        "saved_frame_after_switch_max_error_m": float(
            np.max(maximum[times > report["feedback_start"] + 1e-12])
        ),
        "saved_final_max_error_m": float(maximum[-1]),
        "reported_holding_max_error_m": report["max_holding_nm"] * 1e-9,
        "recomputed_final_carrier_height_peak_m": float(np.max(carrier)),
        "recomputed_final_source_work_w": float(source_power),
        "recomputed_final_absorption_w": float(absorbed_power),
        "sampling_scope": "4001 radii at saved slow-time frames; not a continuous-time certificate.",
        "conclusion": (
            "Saved mean-shape frames are below 10 nm, but the reported 6.14 nm holding "
            "maximum is not reproduced. Carrier motion exceeds 10 nm. "
            "This is not a two-face result."
        ),
    }
    (args.out / "config.json").write_text(
        json.dumps(
            {
                "source": str(args.source),
                "n_o": 1.4,
                "z_o_m": -0.1,
                "n_i": 1.5,
                "z_i_m": 0.2,
                "radius_m": radius,
                "clear_radius_m": aperture,
                "interpretation": "Audit saved single-interface frames and recompute final acoustics.",
            },
            indent=2,
        )
        + "\n"
    )
    (args.out / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        args.out / "audited-errors.npz",
        times_s=times,
        maximum_height_error_m=maximum,
        radius_m=r,
        exact_target_height_m=target,
    )
    (args.out / "report.md").write_text(
        "# Audit of the imported single-interface discussion data\n\n"
        f"Recomputed maximum error over saved holding frames: {np.max(maximum[holding]) * 1e9:.4f} nm.\n"
        f"Recomputed final carrier height amplitude: {np.max(carrier) * 1e9:.4f} nm.\n\n"
        f"The imported holding maximum {report['max_holding_nm']:.4f} nm is not "
        "reproduced over these saved frames, even after excluding the switch frame. "
        "The unavailable driver prevents resolving its holding-window convention.\n\n"
        "This supports the discussion's distinction between mean and instantaneous geometry. "
        "Only one surface is represented. The formation/control driver was not available "
        "and the trajectory was not freshly integrated; the final acoustic field was freshly solved. "
        "Saved-frame and radial sampling do not certify errors between samples.\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
