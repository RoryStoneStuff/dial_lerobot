#!/usr/bin/env python3
"""Plot morphology completion isosurfaces colored by running cost."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay


PLOT_LEVELS = ((0.25, 0.25), (0.50, 0.50), (0.75, 0.75), (1.00, 0.999))
EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


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


def polygon_order(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    centered = points - center
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    plane = centered @ basis[:2].T
    angles = np.arctan2(plane[:, 1], plane[:, 0])
    return np.argsort(angles)


def isosurface(
    points: np.ndarray,
    values: np.ndarray,
    costs: np.ndarray,
    simplices: np.ndarray,
    level: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    triangles: list[np.ndarray] = []
    triangle_costs: list[float] = []
    for simplex in simplices:
        xyz = points[simplex]
        vals = values[simplex]
        local_costs = costs[simplex]
        intersections: list[np.ndarray] = []
        intersection_costs: list[float] = []
        for first, second in EDGES:
            low, high = vals[first], vals[second]
            if (low - level) * (high - level) >= 0.0:
                continue
            fraction = (level - low) / (high - low)
            intersections.append(xyz[first] + fraction * (xyz[second] - xyz[first]))
            intersection_costs.append(
                local_costs[first] + fraction * (local_costs[second] - local_costs[first])
            )
        if len(intersections) < 3:
            continue
        polygon = np.asarray(intersections)
        polygon_costs = np.asarray(intersection_costs)
        order = polygon_order(polygon)
        polygon, polygon_costs = polygon[order], polygon_costs[order]
        for index in range(1, len(polygon) - 1):
            triangles.append(polygon[[0, index, index + 1]])
            triangle_costs.append(float(polygon_costs[[0, index, index + 1]].mean()))
    return triangles, np.asarray(triangle_costs)


def make_figure(benchmark: Path, output_dir: Path) -> None:
    points, completion, costs = load_results(benchmark)
    simplices = Delaunay(points).simplices
    norm = colors.LogNorm(vmin=0.25, vmax=20.0)
    cmap = plt.get_cmap("magma")
    fig = plt.figure(figsize=(32, 8.6))

    for panel, (label_level, surface_level) in enumerate(PLOT_LEVELS, start=1):
        axis = fig.add_subplot(1, 4, panel, projection="3d")
        triangles, triangle_costs = isosurface(
            points, completion, costs, simplices, surface_level
        )
        if triangles:
            surface = Poly3DCollection(
                triangles,
                facecolors=cmap(norm(np.clip(triangle_costs, norm.vmin, norm.vmax))),
                edgecolors=(0.30, 0.16, 0.24, 0.20),
                linewidths=0.35,
                alpha=0.72,
            )
            axis.add_collection3d(surface)
        axis.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=costs, cmap=cmap, norm=norm,
            s=12 + 52 * completion, edgecolors="0.25", linewidths=0.35,
            alpha=0.92, depthshade=False,
        )
        axis.set(
            xlim=(-80, 105), ylim=(-80, 105), zlim=(-55, 105),
            xlabel="Thigh change (%)", ylabel="Shin change (%)", zlabel="Hip change (%)",
            title=f"{int(label_level * 100)}% completion isosurface",
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
        "Completion contours colored by running cost\n"
        "Surfaces use piecewise-linear interpolation; markers are all 75 measured designs",
        fontsize=18, fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(left=0.015, right=0.91, bottom=0.03, top=0.86, wspace=0.03)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "04_completion_cost_isosurfaces"
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
