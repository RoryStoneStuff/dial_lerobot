"""Generate 75 intermediate morphologies that complement an existing benchmark."""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np


THIGH_SHIN_LEVELS = (-62.5, -37.5, -12.5, 12.5, 37.5, 62.5, 87.5)
HIP_LEVELS = (-37.5, -12.5, 12.5, 37.5, 62.5, 87.5)
LOWER = np.asarray((-75.0, -75.0, -50.0))
UPPER = np.asarray((100.0, 100.0, 100.0))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-plan", type=Path, required=True)
    parser.add_argument("--existing-benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalized(levels):
    return (np.asarray(levels, dtype=float) - LOWER) / (UPPER - LOWER)


def theta_from_percent(nominal, levels):
    thigh, shin, hip = levels
    return [
        nominal[0] * (1 + thigh / 100),
        nominal[1] * (1 + shin / 100),
        nominal[2],
        nominal[3] * (1 + hip / 100),
    ]


def level_label(value):
    sign = "m" if value < 0 else "p"
    whole = int(abs(value))
    suffix = "p5" if abs(value) % 1 else ""
    return f"{sign}{whole:02d}{suffix}"


def predictor(existing_points, existing_values, candidates, neighbors=8):
    candidate_points = np.asarray([normalized(candidate) for candidate in candidates])
    distances = np.linalg.norm(
        candidate_points[:, None, :] - existing_points[None, :, :], axis=2
    )
    nearest = np.argpartition(distances, neighbors - 1, axis=1)[:, :neighbors]
    near_distances = np.take_along_axis(distances, nearest, axis=1)
    weights = 1.0 / np.maximum(near_distances, 1e-6) ** 2
    values = existing_values[nearest]
    return np.sum(weights * values, axis=1) / np.sum(weights, axis=1)


def select(candidates, count, selected_points, existing_points, completion, costs):
    candidates = list(candidates)
    predicted_completion = predictor(existing_points, completion, candidates)
    predicted_cost = predictor(existing_points, np.log1p(costs), candidates)
    cost_range = np.ptp(predicted_cost)
    if cost_range > 0:
        predicted_cost = (predicted_cost - predicted_cost.min()) / cost_range
    else:
        predicted_cost = np.zeros_like(predicted_cost)
    chosen = []
    for _ in range(count):
        candidate_points = np.asarray([normalized(candidate) for candidate in candidates])
        distances = np.min(
            np.linalg.norm(candidate_points[:, None, :] - selected_points[None, :, :], axis=2),
            axis=1,
        )
        coverage = distances / max(float(distances.max()), 1e-12)
        boundary = np.clip(1.0 - 2.0 * np.abs(predicted_completion - 0.5), 0.0, 1.0)
        quality = predicted_completion * (1.0 - predicted_cost)
        score = 0.50 * coverage + 0.35 * boundary + 0.15 * quality
        index = int(np.argmax(score))
        levels = candidates.pop(index)
        chosen.append(levels)
        selected_points = np.vstack((selected_points, normalized(levels)))
        predicted_completion = np.delete(predicted_completion, index)
        predicted_cost = np.delete(predicted_cost, index)
    return chosen, selected_points


def main():
    args = parse_args()
    old_plan = json.loads(args.existing_plan.read_text())
    benchmark = json.loads(args.existing_benchmark.read_text())
    nominal = np.asarray(old_plan["nominal_theta"], dtype=float)

    existing_levels = []
    completion = []
    costs = []
    plan_entries = {entry["name"]: entry for entry in old_plan["morphologies"]}
    for condition_name, condition in benchmark["conditions"].items():
        if condition.get("status") != "complete":
            continue
        morphology = condition_name.split("/", 1)[1]
        entry = plan_entries[morphology]
        existing_levels.append(entry["percent_from_nominal"])
        completion.append(condition["objective"]["completion_fraction"])
        costs.append(condition["objective"]["mean_cost_per_second"])
    existing_points = np.asarray([normalized(levels) for levels in existing_levels])
    completion = np.asarray(completion, dtype=float)
    costs = np.asarray(costs, dtype=float)
    selected_points = existing_points.copy()

    axes = (
        [(level, 0.0, 0.0) for level in THIGH_SHIN_LEVELS]
        + [(0.0, level, 0.0) for level in THIGH_SHIN_LEVELS]
        + [(0.0, 0.0, level) for level in HIP_LEVELS]
    )
    selected_points = np.vstack((selected_points, [normalized(levels) for levels in axes]))

    pair_groups = (
        list(product(THIGH_SHIN_LEVELS, THIGH_SHIN_LEVELS, (0.0,))),
        list(product(THIGH_SHIN_LEVELS, (0.0,), HIP_LEVELS)),
        list(product((0.0,), THIGH_SHIN_LEVELS, HIP_LEVELS)),
    )
    pairs = []
    for candidates in pair_groups:
        chosen, selected_points = select(
            candidates, 8, selected_points, existing_points, completion, costs
        )
        pairs.extend(chosen)

    interior_candidates = list(product(THIGH_SHIN_LEVELS, THIGH_SHIN_LEVELS, HIP_LEVELS))
    interiors, selected_points = select(
        interior_candidates, 31, selected_points, existing_points, completion, costs
    )

    def entry(name, levels, category):
        return {
            "name": name,
            "theta": theta_from_percent(nominal, levels),
            "percent_from_nominal": list(levels),
            "category": category,
        }

    entries = []
    for index, levels in enumerate(axes, start=1):
        entries.append(entry(f"mid_axis_{index:02d}", levels, "intermediate_axis"))
    for index, levels in enumerate(pairs, start=1):
        label = "_".join(level_label(value) for value in levels)
        entries.append(entry(f"mid_pair_{index:02d}_{label}", levels,
                             "intermediate_pair"))
    for index, levels in enumerate(interiors, start=1):
        label = "_".join(level_label(value) for value in levels)
        entries.append(entry(f"mid_interior_{index:02d}_{label}", levels,
                             "intermediate_interior"))
    if len(entries) != 75:
        raise RuntimeError(f"Expected 75 morphologies, generated {len(entries)}")

    plan = {
        "description": "Intermediate fixed-mass 3D morphology refinement",
        "selection": (
            "20 midpoint axes, 24 balanced midpoint pairs, and 31 midpoint interiors; "
            "greedy score combines coverage, current completion-boundary proximity, "
            "and predicted successful low-cost performance"
        ),
        "existing_plan": str(args.existing_plan.resolve()),
        "existing_benchmark": str(args.existing_benchmark.resolve()),
        "nominal_theta": nominal.tolist(),
        "thigh_percent_levels": list(THIGH_SHIN_LEVELS),
        "shin_percent_levels": list(THIGH_SHIN_LEVELS),
        "hip_percent_levels": list(HIP_LEVELS),
        "preserve_link_masses": old_plan["preserve_link_masses"],
        "morphologies": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
