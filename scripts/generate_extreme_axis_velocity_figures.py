"""Plot episode velocity traces for the extreme one-variable morphology sweeps."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial import Delaunay


AXES = (
    (0, "Thigh-length variation"),
    (1, "Shin-length variation"),
    (2, "Hip-width variation"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def save_figure(figure, output, name):
    figure.savefig(output / f"{name}.png", dpi=240, bbox_inches="tight")
    figure.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def axis_entries(plan, parameter_index):
    entries = []
    for entry in plan["morphologies"]:
        percentages = np.asarray(entry["percent_from_nominal"], dtype=float)
        other_indices = [index for index in range(3) if index != parameter_index]
        if entry["category"] == "axis" and np.allclose(percentages[other_indices], 0.0):
            entries.append(entry)
    return sorted(entries, key=lambda entry: entry["percent_from_nominal"][parameter_index])


def condition_folder(benchmark, name):
    matches = [value for key, value in benchmark["conditions"].items()
               if key.split("/")[-1] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one benchmark condition for {name!r}, found {len(matches)}")
    return Path(matches[0]["output"])


def plot_entry_episodes(axis, benchmark, entry, parameter_index, color, alpha=0.38):
    folder = condition_folder(benchmark, entry["name"])
    for episode_path in sorted(folder.glob("episode_*.npz")):
        arrays = np.load(episode_path)
        axis.plot(arrays["time"], arrays["body_velocity"][:, 0],
                  color=color, alpha=alpha, linewidth=0.9)


def command_trace(benchmark):
    folder = condition_folder(benchmark, "nominal")
    arrays = np.load(folder / "episode_000.npz")
    return arrays["time"], arrays["target_velocity"][:, 0]


def plot_stacked(benchmark, plan, output, colormap, normalization):
    figure, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, sharey=True,
                                constrained_layout=True)
    for axis, (parameter_index, title) in zip(axes, AXES):
        entries = axis_entries(plan, parameter_index)
        for entry in entries:
            percentage = entry["percent_from_nominal"][parameter_index]
            plot_entry_episodes(axis, benchmark, entry, parameter_index,
                                colormap(normalization(percentage)))
        time, command = command_trace(benchmark)
        axis.plot(time, command, color="black", linestyle=":", linewidth=2.5,
                  label="Commanded velocity")
        axis.set_title(title, fontweight="bold")
        axis.set_ylabel("Forward velocity (m/s)")
        axis.grid(alpha=0.2)
    axes[0].legend(loc="lower right")
    axes[-1].set_xlabel("Simulation time (s)")
    scalar_map = plt.cm.ScalarMappable(norm=normalization, cmap=colormap)
    colorbar = figure.colorbar(scalar_map, ax=axes, location="right", pad=0.025, shrink=0.92)
    colorbar.set_label("Morphology change from nominal (%)")
    colorbar.set_ticks((-75, -50, -25, 0, 25, 50, 75, 100))
    figure.suptitle("Velocity response to independent morphology variations",
                    fontsize=16, fontweight="bold")
    save_figure(figure, output, "01_axis_sweep_velocity_stacked")


def plot_combined(benchmark, plan, output, colormap, normalization):
    figure, axis = plt.subplots(figsize=(14, 7.5), constrained_layout=True)
    entries = [entry for entry in plan["morphologies"] if entry["category"] == "axis"]
    for entry in entries:
        percentages = np.asarray(entry["percent_from_nominal"], dtype=float)
        nonzero = np.flatnonzero(~np.isclose(percentages, 0.0))
        percentage = 0.0 if len(nonzero) == 0 else percentages[nonzero[0]]
        plot_entry_episodes(axis, benchmark, entry, int(nonzero[0]) if len(nonzero) else 0,
                            colormap(normalization(percentage)), alpha=0.26)
    time, command = command_trace(benchmark)
    axis.plot(time, command, color="black", linestyle=":", linewidth=2.8)
    axis.set_xlabel("Simulation time (s)")
    axis.set_ylabel("Forward velocity (m/s)")
    axis.set_title("All independent morphology sweeps", fontweight="bold")
    axis.grid(alpha=0.2)
    scalar_map = plt.cm.ScalarMappable(norm=normalization, cmap=colormap)
    colorbar = figure.colorbar(scalar_map, ax=axis, location="right", pad=0.02)
    colorbar.set_label("Morphology change from nominal (%)")
    colorbar.set_ticks((-75, -50, -25, 0, 25, 50, 75, 100))
    save_figure(figure, output, "02_axis_sweep_velocity_combined_no_key")


def cost_marker_size(cost):
    lower_cost = 0.25
    upper_cost = 20.0
    clipped = np.clip(cost, lower_cost, upper_cost)
    score = ((np.log(upper_cost) - np.log(clipped)) /
             (np.log(upper_cost) - np.log(lower_cost)))
    return 25 + 180 * score


def plot_morphology_cube(benchmark, plan, output):
    plan_entries = {entry["name"]: entry for entry in plan["morphologies"]}
    points = []
    for key, condition in benchmark["conditions"].items():
        name = key.split("/")[-1]
        entry = plan_entries[name]
        objective = condition["objective"]
        points.append({
            "name": name,
            "percentages": np.asarray(entry["percent_from_nominal"], dtype=float),
            "survival": objective["survival_fraction"],
            "cost": objective["mean_cost_per_second"],
        })

    percentages = np.asarray([point["percentages"] for point in points])
    survival = np.asarray([point["survival"] for point in points])
    costs = np.asarray([point["cost"] for point in points])
    sizes = cost_marker_size(costs)
    figure = plt.figure(figsize=(13, 10), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    scatter = axis.scatter(
        percentages[:, 0], percentages[:, 1], percentages[:, 2],
        c=survival, s=sizes, cmap="viridis", vmin=0, vmax=1,
        edgecolors="#202020", linewidths=0.45, alpha=0.88, depthshade=True,
    )
    labels = {
        "nominal": "Nominal",
        "winner_seed42": "Winner 42",
        "winner_seed142": "Winner 142",
        "hips_m050": "Best cost",
        "corner_t_p100_s_p100_h_m050": "Stable extreme",
    }
    for point in points:
        if point["name"] not in labels:
            continue
        thigh, shin, hip = point["percentages"]
        axis.text(thigh + 2, shin + 2, hip + 2, labels[point["name"]], fontsize=8)
    axis.set_xlabel("Thigh length change (%)", labelpad=10)
    axis.set_ylabel("Shin length change (%)", labelpad=10)
    axis.set_zlabel("Hip width change (%)", labelpad=10)
    axis.set_xlim(-82, 107)
    axis.set_ylim(-82, 107)
    axis.set_zlim(-57, 107)
    axis.view_init(elev=24, azim=-56)
    axis.set_title("Morphology performance cube", fontsize=16, fontweight="bold", pad=18)
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.08, shrink=0.72)
    colorbar.set_label("Mean survival fraction")
    legend_costs = (0.3, 1.0, 5.0, 20.0)
    handles = [
        Line2D([], [], linestyle="", marker="o", markerfacecolor="#777777",
               markeredgecolor="#202020", markersize=np.sqrt(cost_marker_size(cost)),
               label=f"Cost/s = {cost:g}")
        for cost in legend_costs
    ]
    axis.legend(handles=handles, title="Larger marker = lower cost",
                loc="upper left", bbox_to_anchor=(0.02, 0.96), frameon=True)
    save_figure(figure, output, "03_morphology_performance_cube")


def completion_isosurface(points, completion, costs, level):
    tetrahedra = Delaunay(points).simplices
    edges = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    triangles = []
    triangle_log_costs = []
    surface_level = level - 1e-6
    log_costs = np.log(np.maximum(costs, 1e-8))
    for simplex in tetrahedra:
        vertices = points[simplex]
        values = completion[simplex]
        vertex_log_costs = log_costs[simplex]
        intersections = {}
        for first, second in edges:
            first_value = values[first]
            second_value = values[second]
            if ((first_value < surface_level <= second_value) or
                    (second_value < surface_level <= first_value)):
                fraction = ((surface_level - first_value) /
                            (second_value - first_value))
                position = vertices[first] + fraction * (vertices[second] - vertices[first])
                log_cost = (vertex_log_costs[first] + fraction *
                            (vertex_log_costs[second] - vertex_log_costs[first]))
                intersections[tuple(np.round(position, 10))] = (position, log_cost)
        if len(intersections) < 3:
            continue
        positions = np.asarray([value[0] for value in intersections.values()])
        costs_at_positions = np.asarray([value[1] for value in intersections.values()])
        center = np.mean(positions, axis=0)
        centered = positions - center
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
        first_axis = vectors[0]
        second_axis = vectors[1]
        angles = np.arctan2(centered @ second_axis, centered @ first_axis)
        order = np.argsort(angles)
        positions = positions[order]
        costs_at_positions = costs_at_positions[order]
        for index in range(1, len(positions) - 1):
            triangle = positions[[0, index, index + 1]]
            triangles.append(triangle)
            triangle_log_costs.append(float(np.mean(costs_at_positions[[0, index, index + 1]])))
    return triangles, np.exp(triangle_log_costs)


def plot_completion_cost_isosurfaces(benchmark, plan, output):
    plan_entries = {entry["name"]: entry for entry in plan["morphologies"]}
    points = []
    completion = []
    costs = []
    for key, condition in benchmark["conditions"].items():
        name = key.split("/")[-1]
        points.append(plan_entries[name]["percent_from_nominal"])
        completion.append(condition["objective"]["completion_fraction"])
        costs.append(condition["objective"]["mean_cost_per_second"])
    points = np.asarray(points, dtype=float)
    completion = np.asarray(completion, dtype=float)
    costs = np.asarray(costs, dtype=float)
    cost_normalization = LogNorm(vmin=0.25, vmax=20.0, clip=True)
    cost_colormap = plt.get_cmap("magma")
    figure = plt.figure(figsize=(21, 7.5), constrained_layout=True)
    axes = [figure.add_subplot(1, 3, index, projection="3d") for index in range(1, 4)]
    levels = (0.25, 0.50, 0.75)
    for axis, level in zip(axes, levels):
        triangles, triangle_costs = completion_isosurface(points, completion, costs, level)
        face_colors = cost_colormap(cost_normalization(triangle_costs))
        collection = Poly3DCollection(
            triangles, facecolors=face_colors, edgecolors=(0, 0, 0, 0.12),
            linewidths=0.18, alpha=0.78,
        )
        axis.add_collection3d(collection)
        axis.scatter(
            points[:, 0], points[:, 1], points[:, 2], c=costs,
            s=12 + 36 * completion, cmap=cost_colormap, norm=cost_normalization,
            edgecolors="#202020", linewidths=0.3, alpha=0.9, depthshade=False,
        )
        axis.set_xlabel("Thigh change (%)", labelpad=8)
        axis.set_ylabel("Shin change (%)", labelpad=8)
        axis.set_zlabel("Hip change (%)", labelpad=8)
        axis.set_xlim(-82, 107)
        axis.set_ylim(-82, 107)
        axis.set_zlim(-57, 107)
        axis.view_init(elev=24, azim=-56)
        axis.set_title(f"{level:.0%} completion isosurface", fontweight="bold")
    scalar_map = plt.cm.ScalarMappable(norm=cost_normalization, cmap=cost_colormap)
    colorbar = figure.colorbar(scalar_map, ax=axes, location="right", pad=0.02, shrink=0.78)
    colorbar.set_label("Running cost per simulated second")
    colorbar.set_ticks((0.3, 1.0, 5.0, 20.0))
    colorbar.set_ticklabels(("0.3", "1", "5", "20"))
    figure.suptitle(
        "Completion contours colored by running cost\n"
        "Surfaces use piecewise-linear interpolation; markers are all 75 measured designs",
        fontsize=15, fontweight="bold",
    )
    save_figure(figure, output, "04_completion_cost_isosurfaces")


def main():
    args = parse_args()
    benchmark = json.loads(args.benchmark.read_text())
    plan = json.loads(args.plan.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    colormap = plt.get_cmap("coolwarm")
    normalization = Normalize(vmin=-75, vmax=100)
    plot_stacked(benchmark, plan, args.output, colormap, normalization)
    plot_combined(benchmark, plan, args.output, colormap, normalization)
    plot_morphology_cube(benchmark, plan, args.output)
    plot_completion_cost_isosurfaces(benchmark, plan, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
