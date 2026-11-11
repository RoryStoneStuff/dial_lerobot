"""Plot candidate mechanical cost of transport across CMA-ES generations."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("optimization", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-highest", type=int, default=0)
    parser.add_argument("--current-generation", type=Path)
    args = parser.parse_args()

    data = json.loads(args.optimization.read_text())
    points = []
    for generation in data["generations"]:
        for candidate, evaluation in enumerate(generation["evaluations"]):
            cot = evaluation["objective"].get("mean_positive_mechanical_cot")
            if cot is None:
                continue
            points.append({
                "generation": int(generation["generation"]) + 1,
                "candidate": candidate + 1,
                "cot": float(cot),
                "completion": float(evaluation["objective"]["completion_fraction"]),
            })
    if args.current_generation and args.current_generation.exists():
        partial = json.loads(args.current_generation.read_text())
        for candidate, evaluation in enumerate(partial.get("evaluations", [])):
            objective = evaluation.get("objective", {})
            cot = objective.get("mean_positive_mechanical_cot")
            if cot is None:
                continue
            points.append({
                "generation": int(partial["generation"]) + 1,
                "candidate": candidate + 1,
                "cot": float(cot),
                "completion": float(objective["completion_fraction"]),
            })
    if not points:
        raise ValueError("Optimization file contains no candidate evaluations")
    if args.exclude_highest:
        if args.exclude_highest >= len(points):
            raise ValueError("Cannot exclude every candidate")
        excluded = set(sorted(range(len(points)), key=lambda index: points[index]["cot"], reverse=True)[:args.exclude_highest])
        points = [point for index, point in enumerate(points) if index not in excluded]

    generations = sorted({point["generation"] for point in points})
    grouped = {
        generation: [point for point in points if point["generation"] == generation]
        for generation in generations
    }
    means = [np.mean([point["cot"] for point in grouped[generation]]) for generation in generations]
    minima = [min(point["cot"] for point in grouped[generation]) for generation in generations]
    running_best = np.minimum.accumulate(minima)

    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for point in points:
        axis.scatter(
            point["generation"], point["cot"],
            color=plt.cm.viridis((point["generation"] - 1) / max(1, len(generations) - 1)),
            edgecolor="white", linewidth=0.7, s=90, zorder=3,
        )
        axis.annotate(
            f"{point['candidate']}",
            (point["generation"], point["cot"]),
            xytext=(5, 4), textcoords="offset points", fontsize=8,
        )
    axis.plot(generations, means, "o--", color="0.35", label="Generation mean", zorder=2)
    axis.plot(generations, running_best, "s-", color="#d62728", linewidth=2,
              label="Best CoT so far", zorder=4)
    axis.set_xlabel("CMA-ES generation")
    axis.set_ylabel("Positive mechanical cost of transport")
    axis.set_title(f"Calibrated DRAFT optimization: {len(points)} candidates")
    axis.set_xticks(generations)
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.savefig(args.output, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
