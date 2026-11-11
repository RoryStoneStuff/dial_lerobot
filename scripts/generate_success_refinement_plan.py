"""Generate a dense refinement plan near measured 100%-completion designs."""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np


LOWER = np.asarray((-75.0, -75.0, -50.0))
UPPER = np.asarray((100.0, 100.0, 100.0))
GRID_STEP = 6.25
MIN_PREDICTED_COMPLETION = 0.95
MAX_SUCCESS_DISTANCE = 0.16
TOTAL_DESIGNS = 82


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalized(values):
    return (np.asarray(values, dtype=float) - LOWER) / (UPPER - LOWER)


def theta_from_percent(nominal, levels):
    return [
        nominal[0] * (1 + levels[0] / 100),
        nominal[1] * (1 + levels[1] / 100),
        nominal[2],
        nominal[3] * (1 + levels[2] / 100),
    ]


def predict_completion(candidates, measured_points, completion, neighbors=10):
    distances = np.linalg.norm(
        candidates[:, None, :] - measured_points[None, :, :], axis=2
    )
    nearest = np.argpartition(distances, neighbors - 1, axis=1)[:, :neighbors]
    local_distances = np.take_along_axis(distances, nearest, axis=1)
    weights = 1.0 / np.maximum(local_distances, 1e-4) ** 2
    prediction = np.sum(weights * completion[nearest], axis=1) / np.sum(weights, axis=1)
    return prediction, distances


def main():
    args = parse_args()
    dataset = json.loads(args.dataset.read_text())
    nominal = np.asarray(dataset["nominal_theta"], dtype=float)
    rows = dataset["morphologies"]
    measured = np.asarray([row["percent_from_nominal"] for row in rows], dtype=float)
    completion = np.asarray([row["completion_fraction"] for row in rows], dtype=float)
    measured_points = np.asarray([normalized(point) for point in measured])
    existing = {tuple(np.round(point, 8)) for point in measured}

    thigh_shin = np.arange(-75.0, 100.0 + GRID_STEP / 2, GRID_STEP)
    hips = np.arange(-50.0, 100.0 + GRID_STEP / 2, GRID_STEP)
    candidates = np.asarray([
        point for point in product(thigh_shin, thigh_shin, hips)
        if tuple(np.round(point, 8)) not in existing
    ], dtype=float)
    candidate_points = np.asarray([normalized(point) for point in candidates])
    prediction, distances = predict_completion(
        candidate_points, measured_points, completion
    )
    full = completion == 1.0
    nearest_success = distances[:, full].min(axis=1)
    nearest_nonfull = distances[:, ~full].min(axis=1)
    eligible = ((prediction >= MIN_PREDICTED_COMPLETION)
                & (nearest_success < nearest_nonfull)
                & (nearest_success <= MAX_SUCCESS_DISTANCE))
    candidates = candidates[eligible]
    candidate_points = candidate_points[eligible]
    prediction = prediction[eligible]
    nearest_success = nearest_success[eligible]
    nearest_nonfull = nearest_nonfull[eligible]
    if len(candidates) < TOTAL_DESIGNS:
        raise RuntimeError(f"Only {len(candidates)} eligible candidates for {TOTAL_DESIGNS} slots")

    selected_points = measured_points.copy()
    entries = []
    for index in range(TOTAL_DESIGNS):
        coverage = np.min(
            np.linalg.norm(candidate_points[:, None, :] - selected_points[None, :, :], axis=2),
            axis=1,
        )
        coverage /= max(float(coverage.max()), 1e-12)
        confidence = ((prediction - MIN_PREDICTED_COMPLETION)
                      / (1.0 - MIN_PREDICTED_COMPLETION))
        score = 0.75 * coverage + 0.25 * confidence
        chosen = int(np.argmax(score))
        levels = candidates[chosen]
        entries.append({
            "name": f"success_refine_{index + 1:03d}",
            "theta": theta_from_percent(nominal, levels),
            "percent_from_nominal": levels.tolist(),
            "category": "local_100_percent_refinement",
            "predicted_completion": float(prediction[chosen]),
            "normalized_distance_to_nearest_full_success": float(nearest_success[chosen]),
            "normalized_distance_to_nearest_nonfull": float(nearest_nonfull[chosen]),
        })
        selected_points = np.vstack((selected_points, candidate_points[chosen]))
        candidates = np.delete(candidates, chosen, axis=0)
        candidate_points = np.delete(candidate_points, chosen, axis=0)
        prediction = np.delete(prediction, chosen)
        nearest_success = np.delete(nearest_success, chosen)
        nearest_nonfull = np.delete(nearest_nonfull, chosen)

    plan = {
        "description": "Dense local refinement of measured 100%-completion neighborhoods",
        "selection": (
            "6.25%-increment lattice; inverse-distance 10-neighbor completion prediction >= 95%; "
            "closer to a fully completing measurement than any non-full measurement; "
            "greedy 75% coverage and 25% confidence selection"
        ),
        "source_dataset": str(args.dataset.resolve()),
        "nominal_theta": nominal.tolist(),
        "grid_step_percent": GRID_STEP,
        "minimum_predicted_completion": MIN_PREDICTED_COMPLETION,
        "maximum_normalized_success_distance": MAX_SUCCESS_DISTANCE,
        "preserve_link_masses": True,
        "morphologies": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
