"""Plot multiple CMA-ES runs as one continuous CoT history."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_points(path: Path, offset: int, current_generation: Path | None = None):
    data = json.loads(path.read_text())
    points = []
    for generation in data.get("generations", []):
        for candidate, evaluation in enumerate(generation.get("evaluations", [])):
            objective = evaluation.get("objective", {})
            cot = objective.get("mean_positive_mechanical_cot")
            if cot is not None:
                points.append((offset + int(generation["generation"]) + 1, candidate + 1, float(cot)))
    if current_generation and current_generation.exists():
        partial = json.loads(current_generation.read_text())
        generation_number = offset + int(partial["generation"]) + 1
        for candidate, evaluation in enumerate(partial.get("evaluations", [])):
            objective = evaluation.get("objective", {})
            cot = objective.get("mean_positive_mechanical_cot")
            if cot is not None:
                points.append((generation_number, candidate + 1, float(cot)))
    return points, max((int(g["generation"]) for g in data.get("generations", [])), default=-1) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("optimizations", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--current-generation", type=Path)
    args = parser.parse_args()

    points = []
    offset = 0
    for index, path in enumerate(args.optimizations):
        partial = args.current_generation if index == 0 else None
        run_points, generation_count = read_points(path, offset, partial)
        points.extend(run_points)
        offset += generation_count
    if not points:
        raise ValueError("No CoT evaluations found")

    generations = sorted({point[0] for point in points})
    grouped = {generation: [p[2] for p in points if p[0] == generation] for generation in generations}
    means = [np.mean(grouped[generation]) for generation in generations]
    minima = [min(grouped[generation]) for generation in generations]
    running_best = np.minimum.accumulate(minima)

    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    colors = plt.cm.viridis((np.asarray(generations) - 1) / max(1, len(generations) - 1))
    color_by_generation = dict(zip(generations, colors))
    for generation, candidate, cot in points:
        axis.scatter(generation, cot, color=color_by_generation[generation], edgecolor="white",
                     linewidth=0.7, s=90, zorder=3)
        axis.annotate(f"{candidate}", (generation, cot), xytext=(5, 4),
                      textcoords="offset points", fontsize=8)
    axis.plot(generations, means, "o--", color="0.35", label="Generation mean", zorder=2)
    axis.plot(generations, running_best, "s-", color="#d62728", linewidth=2,
              label="Best CoT so far", zorder=4)
    axis.set_xlabel("Optimization generation")
    axis.set_ylabel("Positive mechanical cost of transport")
    axis.set_title(f"DRAFT optimization CoT history ({len(points)} candidates)")
    axis.set_xticks(generations)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    fig.savefig(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
