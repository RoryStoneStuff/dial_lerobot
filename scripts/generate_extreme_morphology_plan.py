"""Generate a deterministic 75-design, three-dimensional morphology plan."""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from dial_lerobot.assets.model_builder import nominal_design
from dial_lerobot.config import load_config


THIGH_SHIN_LEVELS = (-75, -50, -25, 0, 25, 50, 75, 100)
HIP_LEVELS = (-50, -25, 0, 25, 50, 75, 100)
TOTAL_DESIGNS = 75


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--winner", type=Path, action="append", default=[])
    parser.add_argument("--winner-label", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def level_label(value):
    sign = "m" if value < 0 else "p"
    return f"{sign}{abs(value):03d}"


def theta_from_percent(nominal, levels):
    thigh, shin, hip = levels
    return [
        nominal[0] * (1 + thigh / 100),
        nominal[1] * (1 + shin / 100),
        nominal[2],
        nominal[3] * (1 + hip / 100),
    ]


def normalized(levels):
    lower = np.asarray((-75, -75, -50), dtype=float)
    upper = np.asarray((100, 100, 100), dtype=float)
    return (np.asarray(levels, dtype=float) - lower) / (upper - lower)


def main():
    args = parse_args()
    if len(args.winner) != len(args.winner_label):
        raise ValueError("Provide one --winner-label for each --winner")
    cfg = load_config(args.config)
    nominal = nominal_design(cfg.robot, include_hip_width=True)
    density = cfg.design.fixed_density_proxy
    if density is not None:
        nominal[2] = density

    forced = [("nominal", (0, 0, 0), "axis")]
    forced.extend((f"thigh_{level_label(level)}", (level, 0, 0), "axis")
                  for level in THIGH_SHIN_LEVELS if level)
    forced.extend((f"shin_{level_label(level)}", (0, level, 0), "axis")
                  for level in THIGH_SHIN_LEVELS if level)
    forced.extend((f"hips_{level_label(level)}", (0, 0, level), "axis")
                  for level in HIP_LEVELS if level)
    forced.extend(
        (f"corner_t_{level_label(thigh)}_s_{level_label(shin)}_h_{level_label(hip)}",
         (thigh, shin, hip), "corner")
        for thigh, shin, hip in product((-75, 100), (-75, 100), (-50, 100))
    )

    winner_entries = []
    winner_coordinates = []
    for path, label in zip(args.winner, args.winner_label):
        theta = np.asarray(json.loads(path.read_text())["theta"], dtype=float)
        if theta.shape != (4,):
            raise ValueError(f"Winner {path} does not have four design variables")
        levels = (
            100 * (theta[0] / nominal[0] - 1),
            100 * (theta[1] / nominal[1] - 1),
            100 * (theta[3] / nominal[3] - 1),
        )
        winner_entries.append({
            "name": f"winner_{label}",
            "theta": theta.tolist(),
            "percent_from_nominal": list(levels),
            "category": "optimized_winner",
            "source": str(path.resolve()),
        })
        winner_coordinates.append(normalized(levels))

    forced_coordinates = {levels for _, levels, _ in forced}
    candidates = [levels for levels in product(THIGH_SHIN_LEVELS,
                                                THIGH_SHIN_LEVELS,
                                                HIP_LEVELS)
                  if levels not in forced_coordinates]
    selected_coordinates = [normalized(levels) for _, levels, _ in forced]
    selected_coordinates.extend(winner_coordinates)
    interaction_count = TOTAL_DESIGNS - len(forced) - len(winner_entries)
    interactions = []
    for _ in range(interaction_count):
        selected = np.asarray(selected_coordinates)
        minimum_distances = [
            float(np.min(np.linalg.norm(selected - normalized(candidate), axis=1)))
            for candidate in candidates
        ]
        index = int(np.argmax(minimum_distances))
        levels = candidates.pop(index)
        interactions.append(levels)
        selected_coordinates.append(normalized(levels))

    entries = [{
        "name": name,
        "theta": theta_from_percent(nominal, levels),
        "percent_from_nominal": list(levels),
        "category": category,
    } for name, levels, category in forced]
    entries[1:1] = winner_entries
    entries.extend({
        "name": (f"interaction_{index:03d}_t_{level_label(levels[0])}_"
                 f"s_{level_label(levels[1])}_h_{level_label(levels[2])}"),
        "theta": theta_from_percent(nominal, levels),
        "percent_from_nominal": list(levels),
        "category": "space_filling_interaction",
    } for index, levels in enumerate(interactions, start=1))
    if len(entries) != TOTAL_DESIGNS:
        raise RuntimeError(f"Expected {TOTAL_DESIGNS} designs, generated {len(entries)}")

    plan = {
        "description": "Extreme fixed-mass 3D fractional morphology grid",
        "selection": "forced axes and corners plus greedy maximin lattice coverage",
        "nominal_theta": nominal.tolist(),
        "thigh_percent_levels": list(THIGH_SHIN_LEVELS),
        "shin_percent_levels": list(THIGH_SHIN_LEVELS),
        "hip_percent_levels": list(HIP_LEVELS),
        "preserve_link_masses": cfg.robot.preserve_link_masses,
        "morphologies": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
