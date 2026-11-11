#!/usr/bin/env python3
"""Plot completion-threshold design volumes colored by running cost."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull


LEVELS = (0.25, 0.50, 0.75, 1.00)


def load_results(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    benchmark = json.loads(path.read_text())
    nominal = np.asarray(benchmark["conditions"]["vx_0p85/nominal"]["theta"], dtype=float)
    points, completion, costs = [], [], []
    for condition in benchmark["conditions"].values():
        if condition.get("status") != "complete" or "objective" not in condition:
            continue
        theta = np.asarray(condition["theta"], dtype=float)
        points.append(100.0 * (theta[[0, 1, 3]] / nominal[[0, 1, 3]] - 1.0))
        completion.append(float(condition["objective"]["completion_fraction"]))
        costs.append(float(condition["objective"]["mean_cost_per_second"]))
    return np.asarray(points), np.asarray(completion), np.asarray(costs)


def make_figure(benchmark: Path, output_dir: Path) -> None:
    points, completion, costs = load_results(benchmark)
    norm = colors.LogNorm(vmin=0.25, vmax=20.0)
    cmap = plt.get_cmap("magma")
    fig = plt.figure(figsize=(32, 8.6))

    for panel, level in enumerate(LEVELS, start=1):
        axis = fig.add_subplot(1, 4, panel, projection="3d")
        selected = completion >= level - 1e-9
        volume_points = points[selected]
        volume_costs = costs[selected]
        hull = ConvexHull(volume_points)
        triangles = volume_points[hull.simplices]
        face_costs = volume_costs[hull.simplices].mean(axis=1)
        volume = Poly3DCollection(
            triangles,
            facecolors=cmap(norm(np.clip(face_costs, norm.vmin, norm.vmax))),
            edgecolors=(0.20, 0.12, 0.18, 0.42),
            linewidths=0.55,
            alpha=0.28,
        )
        axis.add_collection3d(volume)

        axis.scatter(
            points[~selected, 0], points[~selected, 1], points[~selected, 2],
            c="0.72", s=12, alpha=0.18, edgecolors="none", depthshade=False,
        )
        axis.scatter(
            volume_points[:, 0], volume_points[:, 1], volume_points[:, 2],
            c=volume_costs, cmap=cmap, norm=norm,
            s=25 + 30 * completion[selected], edgecolors="0.18", linewidths=0.45,
            alpha=0.98, depthshade=False,
        )
        axis.scatter(
            [0], [0], [0], marker="*", s=180, c="cyan", edgecolors="black",
            linewidths=0.8, depthshade=False,
        )
        axis.set(
            xlim=(-80, 105), ylim=(-80, 105), zlim=(-55, 105),
            xlabel="Thigh change (%)", ylabel="Shin change (%)", zlabel="Hip change (%)",
            title=(f"≥{int(level * 100)}% completion volume\n"
                   f"{selected.sum()} measured designs"),
        )
        axis.view_init(elev=24, azim=-58)
        axis.grid(alpha=0.35)

    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    colorbar_axis = fig.add_axes((0.945, 0.16, 0.012, 0.67))
    colorbar = fig.colorbar(scalar_map, cax=colorbar_axis)
    colorbar.set_label("Running cost per simulated second")
    colorbar.set_ticks([0.3, 1, 5, 20])
    colorbar.set_ticklabels(["0.3", "1", "5", "20"])
    fig.suptitle(
        "Measured morphology volumes meeting each completion threshold\n"
        "Transparent hulls are convex envelopes, not unmeasured performance claims; cyan star is nominal",
        fontsize=18, fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(left=0.015, right=0.91, bottom=0.03, top=0.84, wspace=0.03)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "05_completion_cost_volumes"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    make_figure(args.benchmark, args.output_dir)


if __name__ == "__main__":
    main()
