"""Sequential convex source steps with true quadratic merit checks.

The target, frequency and apparatus remain fixed throughout optimization.
Iterations are not continuation of targets or physical time evolution.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..provenance import capture_execution
from .precision import PrecisionObjective, WhitenedObjective, complex_linear_jacobian


def optimize(
    cache,
    source,
    limit,
    steps=100,
    cutoff=1e-12,
    annulus_weight=0.05,
    checkpoint=None,
    precondition=False,
    include_carrier=True,
    clear_radius_m=None,
    front_rim_upper_bound_m=None,
    source_spectral_method="gram",
    second_order_corrections=0,
    nonlinear_rim_projection=False,
):
    import cvxpy as cp

    base = PrecisionObjective(cache, 1, limit / np.sqrt(2), 0)
    coordinates = WhitenedObjective(base, cutoff, spectral_method=source_spectral_method)
    transform = coordinates.real_transform
    dimension = transform.shape[1]
    z = coordinates.encode(source)
    step_map = np.eye(dimension)
    peak = max(abs(coordinates.unpack(z)))
    z *= min(1, limit / max(peak, 1e-30))
    if precondition:
        hj = coordinates.observation_jacobian(z)
        bj = complex_linear_jacobian(base.carrier) @ transform
        metric = np.vstack(
            [hj / np.sqrt(len(hj)), 0.001 * transform / limit]
            + ([bj / np.sqrt(len(bj))] if include_carrier else [])
        )
        _, singular, vt = np.linalg.svd(metric, full_matrices=False)
        step_map = vt.T / np.maximum(singular, singular[0] * 1e-14)
        transform = transform @ step_map
        z = np.linalg.solve(step_map, z)

    def unpack(point):
        full = transform @ point
        return full[: base.channels] + 1j * full[base.channels :]

    faces = cache["observation_faces"]
    # Observation weights are one precisely on the fixed pupil in these caches.
    pupil = (
        cache["observation_weights"] == 1
        if clear_radius_m is None
        else cache["observation_radii_m"] <= clear_radius_m
    )
    annulus = ~pupil
    carrier_matrix = complex_linear_jacobian(base.carrier) @ transform
    carrier_n = len(base.carrier)
    physical_n = base.channels
    delta = cp.Variable(dimension)
    center = cp.Parameter(dimension)
    jac = cp.Parameter((len(base.observe), dimension))
    height = cp.Parameter(len(base.observe))
    radius = cp.Parameter(nonneg=True)
    mean_bound = cp.Variable(2, nonneg=True)
    carrier_bound = cp.Variable(2, nonneg=True)
    annulus_bound = cp.Variable(nonneg=True)
    worst = cp.Variable(nonneg=True)
    full = transform @ (center + delta)
    carrier_vector = carrier_matrix @ (center + delta)
    carrier = carrier_vector[:carrier_n] + 1j * carrier_vector[carrier_n:]
    linear_height = height + jac @ delta
    constraints = [
        cp.norm(transform @ delta) <= radius,
        cp.norm(cp.vstack([full[:physical_n], full[physical_n:]]), axis=0) <= limit,
        mean_bound + (carrier_bound if include_carrier else 0) <= worst,
    ]
    for j in range(2):
        constraints += [cp.abs(linear_height[pupil & (faces == j)]) <= mean_bound[j]]
        if include_carrier:
            constraints += [cp.abs(carrier[cache["carrier_faces"] == j]) <= carrier_bound[j]]
    rim = None
    if front_rim_upper_bound_m is not None:
        if clear_radius_m is None or not np.isfinite(front_rim_upper_bound_m):
            raise ValueError("A finite rim bound and a declared clear radius are required")
        rim = (
            (faces == 0)
            & (cache.get("observation_kind", np.zeros(len(faces), dtype=int)) == 0)
            & np.isclose(cache["observation_radii_m"], clear_radius_m, rtol=0, atol=1e-15)
        )
        if not np.any(rim):
            raise ValueError("Front rim height observation is required for the interception bound")
        constraints += [linear_height[rim] <= front_rim_upper_bound_m / 1e-8]
    if np.any(annulus):
        constraints += [cp.abs(linear_height[annulus]) <= annulus_bound]
    problem = cp.Problem(cp.Minimize(worst + annulus_weight * annulus_bound), constraints)

    def merit(point):
        h, b = coordinates.metrics(step_map @ point)
        mh = np.array([np.max(abs(h[pupil & (faces == j)])) for j in range(2)])
        mb = np.array([np.max(abs(b[cache["carrier_faces"] == j])) for j in range(2)])
        outer = np.max(abs(h[annulus])) if np.any(annulus) else 0
        violation = (
            max(0.0, float(np.max(h[rim])) - front_rim_upper_bound_m / 1e-8)
            if rim is not None
            else 0.0
        )
        return (
            float(
                np.max(mh + (mb if include_carrier else 0))
                + annulus_weight * outer
                + 100 * violation
            ),
            mh,
            mb,
        )

    step_radius = 0.01 * limit * np.sqrt(physical_n)
    history = []
    old, mh, mb = merit(z)
    for iteration in range(steps):
        center.value = z
        h, _ = coordinates.metrics(step_map @ z)
        jac.value = coordinates.observation_jacobian(step_map @ z) @ step_map
        height.value, radius.value = h, step_radius
        predicted = problem.solve(
            solver="CLARABEL",
            tol_gap_abs=1e-9,
            tol_feas=1e-9,
            tol_gap_rel=1e-9,
            max_iter=150,
            verbose=False,
            ignore_dpp=True,
        )
        if delta.value is None:
            history.append({"iteration": iteration, "subproblem_status": problem.status})
            break
        trial = z + delta.value
        amplitude = np.max(abs(unpack(trial)))
        trial *= min(1, limit / max(amplitude, 1e-30))
        candidate, htrial, btrial = merit(trial)
        corrected_steps = 0
        linear_target = h + jac.value @ delta.value
        for _ in range(second_order_corrections):
            # Restore the conic step's predicted force/optical observations
            # after its quadratic remainder, without changing the target.
            current, _ = coordinates.metrics(step_map @ trial)
            tangent = coordinates.observation_jacobian(step_map @ trial) @ step_map
            correction = np.linalg.lstsq(tangent, linear_target - current, rcond=1e-12)[0]
            size = np.linalg.norm(transform @ correction)
            correction *= min(1.0, 0.5 * step_radius / max(size, 1e-30))
            corrected = trial + correction
            corrected *= min(1.0, limit / max(np.max(abs(unpack(corrected))), 1e-30))
            corrected_merit, corrected_h, corrected_b = merit(corrected)
            if corrected_merit >= candidate:
                break
            trial, candidate = corrected, corrected_merit
            htrial, btrial = corrected_h, corrected_b
            corrected_steps += 1
        if nonlinear_rim_projection and rim is not None:
            projected = trial.copy()
            rim_indices = np.flatnonzero(rim)
            for _ in range(6):
                current, _ = coordinates.metrics(step_map @ projected)
                index = rim_indices[np.argmax(current[rim])]
                violation = current[index] - front_rim_upper_bound_m / 1e-8
                if violation <= -1e-7:
                    break
                gradient = coordinates.observation_jacobian(step_map @ projected)[index] @ step_map
                shift = -(violation + 2e-7) * gradient / max(gradient @ gradient, 1e-30)
                size = np.linalg.norm(transform @ shift)
                shift *= min(1.0, 0.5 * step_radius / max(size, 1e-30))
                projected += shift
                projected *= min(1.0, limit / max(np.max(abs(unpack(projected))), 1e-30))
            projected_merit, projected_h, projected_b = merit(projected)
            if projected_merit < candidate:
                trial, candidate = projected, projected_merit
                htrial, btrial = projected_h, projected_b
        improvement = old - candidate
        predicted_improvement = old - predicted
        ratio = improvement / max(predicted_improvement, 1e-15)
        accepted = improvement > 0
        if accepted:
            z, old, mh, mb = trial, candidate, htrial, btrial
        record = {
            "iteration": iteration,
            "physical_source_channels": physical_n,
            "retained_complex_source_coordinates": coordinates.channels,
            "source_gram_cutoff": cutoff,
            "response_model": str(cache.get("observation_response_kind", "mechanical_only")),
            "accepted": bool(accepted),
            "second_order_corrections_accepted": corrected_steps,
            "subproblem_status": problem.status,
            "merit": old,
            "include_carrier_in_objective": include_carrier,
            "step_radius_m_s": step_radius,
            "actual_to_predicted_improvement": float(ratio),
            "frozen_observation_peak_m": (mh * 1e-8).tolist(),
            "frozen_carrier_max_m": (mb * 1e-8).tolist(),
            "conservative_frozen_combined_max_m": float(np.max(mh + mb) * 1e-8),
        }
        current_height, _ = coordinates.metrics(step_map @ z)
        height_only = cache.get("observation_kind", np.zeros(len(pupil), dtype=int)) == 0
        record["frozen_compliance_max_m"] = [
            float(np.max(abs(current_height[pupil & height_only & (faces == j)])) * 1e-8)
            for j in range(2)
        ]
        record["source_peak_m_s"] = float(np.max(abs(unpack(z))))
        if rim is not None:
            record["front_rim_upper_bound_m"] = front_rim_upper_bound_m
            record["predicted_front_rim_height_m"] = float(np.max(current_height[rim]) * 1e-8)
            record["predicted_front_rim_bound_met"] = bool(
                record["predicted_front_rim_height_m"] <= front_rim_upper_bound_m + 1e-14
            )
        if "observation_optical_scale" in cache:
            mask = cache["observation_kind"] == 2
            record["frozen_linearized_spot_radius_m"] = float(
                np.max(abs(current_height[mask])) * 1e-8 / cache["observation_optical_scale"]
            )
        history.append(record)
        if checkpoint:
            checkpoint(unpack(z), history)
        if iteration % 5 == 0:
            print(record, flush=True)
        if ratio < 0.2:
            step_radius *= 0.5
        elif ratio > 0.75:
            step_radius = min(step_radius * 1.5, limit * np.sqrt(physical_n))
        if step_radius < 1e-8 * limit or (abs(improvement) < 1e-8 and accepted):
            break
    return unpack(z), history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    capture_execution(args.out)
    (args.out / "config.json").write_text(json.dumps(inputs, indent=2) + "\n")
    cache_dir = Path(inputs["cache_directory"])
    source_path = cache_dir / "frozen-operator.npz"
    cache = dict(np.load(source_path))
    observation_path = inputs.get("observation_operator_file")
    if observation_path:
        cache.update(dict(np.load(observation_path)))
    source = np.load(inputs["initial_drive_file"])["source_velocity_m_s"]
    cfg = json.loads((cache_dir / "resolved-config.json").read_text())
    (args.out / "validation.json").write_text(
        json.dumps(
            {
                "cache_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "observation_sha256": (
                    hashlib.sha256(Path(observation_path).read_bytes()).hexdigest()
                    if observation_path
                    else None
                ),
                "independent_forward_solve": False,
                "scope": "Frozen-target minimax screening only; refinement and stability untested.",
            },
            indent=2,
        )
        + "\n"
    )

    def checkpoint(command, history):
        np.savez_compressed(
            args.out / "candidate.npz",
            source_velocity_m_s=command,
            target_coefficients_m=cache["target_coefficients_m"],
        )
        (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")

    command, history = optimize(
        cache,
        source,
        cfg["max_source_speed_m_s"],
        inputs.get("steps", 100),
        inputs.get("cutoff", 1e-12),
        inputs.get("annulus_weight", 0.05),
        checkpoint,
        inputs.get("precondition", False),
        include_carrier=inputs.get("include_carrier", True),
        clear_radius_m=cfg.get("clear_radius_m"),
        front_rim_upper_bound_m=inputs.get("front_rim_upper_bound_m"),
    )
    checkpoint(command, history)
    (args.out / "report.md").write_text(
        "# Sequential convex frozen-target inverse\n\n"
        + json.dumps(history[-1], indent=2)
        + "\n\nNo coupled surface, physical trajectory or stability is certified. "
        "The reported combined metric is a conservative sum of separate sampled maxima.\n"
    )


if __name__ == "__main__":
    main()
