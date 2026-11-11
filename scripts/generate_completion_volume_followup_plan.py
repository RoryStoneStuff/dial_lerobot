"""Propose and preview points that densify and expand the 4/4-success volume."""

from __future__ import annotations

import argparse
import csv
from itertools import product
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, Delaunay


LOWER = np.asarray((-75.0, -75.0, -50.0))
UPPER = np.asarray((100.0, 100.0, 100.0))
GRID_STEP = 6.25
BOUNDARY_DESIGNS = 32
UPPER_SURFACE_DESIGNS = 18
UPPER_BRIDGE_LEVELS = (12.5, 25.0, 37.5, 50.0, 62.5)
UPPER_BRIDGE_PER_LEVEL = 4
INTERIOR_DESIGNS = 12
MIN_BOUNDARY_PREDICTION = 0.95
MIN_UPPER_SURFACE_PREDICTION = 0.90
MIN_UPPER_BRIDGE_PREDICTION = 0.95
MIN_INTERIOR_PREDICTION = 0.98
MIN_SURFACE_OFFSET = -0.03
MAX_SURFACE_OFFSET = 0.03
MIN_SURFACE_RADIUS = 0.85
MAX_SURFACE_RADIUS = 1.15
MIN_INTERIOR_RADIUS = 0.45
MAX_INTERIOR_RADIUS = 0.82


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalized(values):
    return (np.asarray(values, dtype=float) - LOWER) / (UPPER - LOWER)


def theta_from_percent(nominal, levels):
    return [
        nominal[0] * (1.0 + levels[0] / 100.0),
        nominal[1] * (1.0 + levels[1] / 100.0),
        nominal[2],
        nominal[3] * (1.0 + levels[2] / 100.0),
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


def scaled(values):
    values = np.asarray(values, dtype=float)
    span = float(np.ptp(values))
    if span <= 1e-12:
        return np.ones_like(values)
    return (values - values.min()) / span


def greedy_select(candidate_points, base_points, count, score_terms):
    selected_points = np.empty((0, 3), dtype=float)
    for _ in range(count):
        references = np.vstack((base_points, selected_points))
        coverage = np.linalg.norm(
            candidate_points[:, None, :] - references[None, :, :], axis=2
        ).min(axis=1)
        score = 0.55 * scaled(coverage)
        for weight, values in score_terms:
            score += weight * scaled(values)
        chosen = int(np.argmax(score))
        selected_points = np.vstack((selected_points, candidate_points[chosen]))
        candidate_points = np.delete(candidate_points, chosen, axis=0)
        score_terms = [
            (weight, np.delete(values, chosen)) for weight, values in score_terms
        ]
    return selected_points


def radial_fraction(points, hull, center):
    offsets = np.asarray(points, dtype=float) - center
    radii = np.linalg.norm(offsets, axis=1)
    directions = offsets / np.maximum(radii[:, None], 1e-12)
    denominators = directions @ hull.equations[:, :3].T
    numerators = -(center @ hull.equations[:, :3].T + hull.equations[:, 3])
    intersections = np.full_like(denominators, np.inf)
    np.divide(
        numerators, denominators, out=intersections,
        where=denominators > 1e-10,
    )
    return radii / intersections.min(axis=1), directions


def greedy_surface_select(candidate_points, base_points, prediction, priority,
                          center, count):
    offsets = candidate_points - center
    directions = offsets / np.maximum(
        np.linalg.norm(offsets, axis=1)[:, None], 1e-12
    )
    selected_points = np.empty((0, 3), dtype=float)
    selected_directions = np.empty((0, 3), dtype=float)
    for _ in range(count):
        references = np.vstack((base_points, selected_points))
        spatial_coverage = np.linalg.norm(
            candidate_points[:, None, :] - references[None, :, :], axis=2
        ).min(axis=1)
        if len(selected_directions):
            direction_coverage = np.linalg.norm(
                directions[:, None, :] - selected_directions[None, :, :], axis=2
            ).min(axis=1)
        else:
            direction_coverage = np.ones(len(candidate_points))
        score = (
            0.50 * scaled(direction_coverage)
            + 0.20 * scaled(spatial_coverage)
            + 0.15 * scaled(prediction)
            + 0.15 * scaled(priority)
        )
        chosen = int(np.argmax(score))
        selected_points = np.vstack((selected_points, candidate_points[chosen]))
        selected_directions = np.vstack((selected_directions, directions[chosen]))
        candidate_points = np.delete(candidate_points, chosen, axis=0)
        directions = np.delete(directions, chosen, axis=0)
        prediction = np.delete(prediction, chosen)
        priority = np.delete(priority, chosen)
    return selected_points


def hull_collection(points, facecolor, edgecolor, alpha, linewidth=0.65):
    hull = ConvexHull(points)
    return Poly3DCollection(
        points[hull.simplices], facecolors=facecolor, edgecolors=edgecolor,
        linewidths=linewidth, alpha=alpha,
    )


def configure_3d_axis(axis, title):
    axis.set(
        xlim=(-80, 105), ylim=(-80, 105), zlim=(-55, 105),
        xlabel="Thigh change (%)", ylabel="Shin change (%)", zlabel="Hip change (%)",
        title=title,
    )
    axis.view_init(elev=24, azim=-58)
    axis.grid(alpha=0.35)


def plot_volume(measured, completion, boundary, upper_surface, upper_bridge,
                interior, output):
    current = measured[completion == 1.0]
    proposed = np.vstack((boundary, upper_surface, upper_bridge, interior))
    hypothetical = np.vstack((current, proposed))
    current_volume = ConvexHull(normalized(current)).volume
    hypothetical_volume = ConvexHull(normalized(hypothetical)).volume
    increase = 100.0 * (hypothetical_volume / current_volume - 1.0)

    figure = plt.figure(figsize=(19, 8.5))
    current_axis = figure.add_subplot(1, 2, 1, projection="3d")
    current_axis.add_collection3d(hull_collection(
        current, "#6f58a8", "#30254d", 0.24,
    ))
    current_axis.scatter(
        current[:, 0], current[:, 1], current[:, 2], c="#34205c", s=28,
        edgecolors="white", linewidths=0.25, depthshade=False,
    )
    configure_3d_axis(
        current_axis,
        f"Current measured 4/4 volume\n{len(current)} successful morphologies",
    )

    proposed_axis = figure.add_subplot(1, 2, 2, projection="3d")
    proposed_axis.add_collection3d(hull_collection(
        current, "none", "#665b78", 0.24, linewidth=0.8,
    ))
    proposed_axis.add_collection3d(hull_collection(
        hypothetical, "#48a56a", "#174f2a", 0.20,
    ))
    proposed_axis.scatter(
        current[:, 0], current[:, 1], current[:, 2], c="#817b89", s=16,
        alpha=0.55, edgecolors="none", depthshade=False, label="Existing 4/4",
    )
    proposed_axis.scatter(
        boundary[:, 0], boundary[:, 1], boundary[:, 2], c="#ef8a24", marker="^",
        s=62, edgecolors="#5f3006", linewidths=0.6, depthshade=False,
        label=f"Balanced boundary ({len(boundary)})",
    )
    proposed_axis.scatter(
        upper_surface[:, 0], upper_surface[:, 1], upper_surface[:, 2], c="#d53e9f",
        marker="^", s=72, edgecolors="#59113f", linewidths=0.7, depthshade=False,
        label=f"Positive-hip cap ({len(upper_surface)})",
    )
    proposed_axis.scatter(
        upper_bridge[:, 0], upper_bridge[:, 1], upper_bridge[:, 2], c="#53c653",
        marker="s", s=58, edgecolors="#154f15", linewidths=0.6, depthshade=False,
        label=f"Nominal-to-cap bridge ({len(upper_bridge)})",
    )
    proposed_axis.scatter(
        interior[:, 0], interior[:, 1], interior[:, 2], c="#20b7cf", marker="o",
        s=52, edgecolors="#074956", linewidths=0.6, depthshade=False,
        label=f"Interior fill ({len(interior)})",
    )
    proposed_axis.scatter(
        [0], [0], [0], marker="*", s=190, c="yellow", edgecolors="black",
        linewidths=0.8, depthshade=False, label="Nominal",
    )
    configure_3d_axis(
        proposed_axis,
        f"Hypothetical volume if all 82 proposals complete 4/4\n"
        f"Convex-hull volume increase: {increase:.1f}%",
    )
    proposed_axis.legend(loc="upper left", fontsize=9)
    figure.suptitle(
        "Proposed 4/4 Completion-Volume Follow-Up",
        fontsize=19, fontweight="bold", y=0.97,
    )
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.88, wspace=0.02)
    stem = output / "01_hypothetical_4of4_volume"
    figure.savefig(stem.with_suffix(".png"), dpi=190, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return current_volume, hypothetical_volume, increase


def plot_projections(measured, completion, boundary, upper_surface, upper_bridge,
                     interior, output):
    full = measured[completion == 1.0]
    nonfull = measured[completion < 1.0]
    projections = (
        (0, 1, "Thigh change (%)", "Shin change (%)"),
        (0, 2, "Thigh change (%)", "Hip change (%)"),
        (1, 2, "Shin change (%)", "Hip change (%)"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(20, 6.6))
    for axis, (x_index, y_index, x_label, y_label) in zip(axes, projections):
        axis.scatter(
            nonfull[:, x_index], nonfull[:, y_index], c="#d5d2d8", marker="x",
            s=18, alpha=0.45, linewidths=0.7, label="Existing below 4/4",
        )
        axis.scatter(
            full[:, x_index], full[:, y_index], c="#817b89", s=22,
            alpha=0.65, edgecolors="none", label="Existing 4/4",
        )
        axis.scatter(
            boundary[:, x_index], boundary[:, y_index], c="#ef8a24", marker="^",
            s=58, edgecolors="#5f3006", linewidths=0.5,
            label="Balanced boundary",
        )
        axis.scatter(
            upper_surface[:, x_index], upper_surface[:, y_index], c="#d53e9f",
            marker="^", s=68, edgecolors="#59113f", linewidths=0.6,
            label="Positive-hip cap",
        )
        axis.scatter(
            upper_bridge[:, x_index], upper_bridge[:, y_index], c="#53c653",
            marker="s", s=52, edgecolors="#154f15", linewidths=0.5,
            label="Nominal-to-cap bridge",
        )
        axis.scatter(
            interior[:, x_index], interior[:, y_index], c="#20b7cf", marker="o",
            s=48, edgecolors="#074956", linewidths=0.5, label="Interior fill",
        )
        axis.scatter(
            [0], [0], marker="*", s=160, c="yellow", edgecolors="black",
            linewidths=0.7, label="Nominal",
        )
        axis.set(xlabel=x_label, ylabel=y_label)
        axis.grid(alpha=0.25)
        axis.set_aspect("equal", adjustable="box")
    axes[0].legend(loc="best", fontsize=8)
    figure.suptitle(
        "Proposed 82-Point Distribution Across All Morphology Projections\n"
        "Magenta samples the upper cap; green bridges nominal to that surface",
        fontsize=17, fontweight="bold",
    )
    figure.tight_layout()
    stem = output / "02_proposed_point_distribution"
    figure.savefig(stem.with_suffix(".png"), dpi=190, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    dataset = json.loads(args.dataset.read_text())
    nominal = np.asarray(dataset["nominal_theta"], dtype=float)
    rows = dataset["morphologies"]
    measured = np.asarray([row["percent_from_nominal"] for row in rows], dtype=float)
    completion = np.asarray([row["completion_fraction"] for row in rows], dtype=float)
    measured_points = normalized(measured)
    full_points = measured_points[completion == 1.0]
    nonfull_points = measured_points[completion < 1.0]
    hull = ConvexHull(full_points)
    tessellation = Delaunay(full_points[hull.vertices])
    hull_center = full_points.mean(axis=0)

    existing = {tuple(np.round(point, 8)) for point in measured}
    candidates = np.asarray([
        point for point in product(
            np.arange(LOWER[0], UPPER[0] + GRID_STEP / 2.0, GRID_STEP),
            np.arange(LOWER[1], UPPER[1] + GRID_STEP / 2.0, GRID_STEP),
            np.arange(LOWER[2], UPPER[2] + GRID_STEP / 2.0, GRID_STEP),
        ) if tuple(np.round(point, 8)) not in existing
    ], dtype=float)
    candidate_points = normalized(candidates)
    prediction, _ = predict_completion(
        candidate_points, measured_points, completion
    )
    nearest_success = np.linalg.norm(
        candidate_points[:, None, :] - full_points[None, :, :], axis=2
    ).min(axis=1)
    nearest_nonfull = np.linalg.norm(
        candidate_points[:, None, :] - nonfull_points[None, :, :], axis=2
    ).min(axis=1)
    hull_values = (
        candidate_points @ hull.equations[:, :3].T + hull.equations[:, 3]
    )
    hull_violation = hull_values.max(axis=1)
    inside = tessellation.find_simplex(candidate_points) >= 0
    radius_fraction, _ = radial_fraction(candidate_points, hull, hull_center)
    box_margin = np.minimum(candidate_points, 1.0 - candidate_points).min(axis=1)

    upper_surface_mask = (
        (candidates[:, 2] >= 62.5)
        & (hull_violation >= -0.04)
        & (hull_violation <= 0.04)
        & (radius_fraction >= MIN_SURFACE_RADIUS)
        & (radius_fraction <= 1.25)
        & (prediction >= MIN_UPPER_SURFACE_PREDICTION)
        & (nearest_success < nearest_nonfull)
    )
    upper_surface_pool = candidates[upper_surface_mask]
    if len(upper_surface_pool) < UPPER_SURFACE_DESIGNS:
        raise RuntimeError(
            f"Only {len(upper_surface_pool)} upper-surface candidates are eligible"
        )
    upper_surface_points = greedy_surface_select(
        candidate_points[upper_surface_mask],
        measured_points,
        prediction[upper_surface_mask],
        candidates[upper_surface_mask, 2],
        hull_center,
        UPPER_SURFACE_DESIGNS,
    )
    upper_surface = LOWER + upper_surface_points * (UPPER - LOWER)
    selected_coordinates = {
        tuple(np.round(point, 8)) for point in upper_surface
    }

    boundary_mask = (
        (hull_violation >= MIN_SURFACE_OFFSET)
        & (hull_violation <= MAX_SURFACE_OFFSET)
        & (radius_fraction >= MIN_SURFACE_RADIUS)
        & (radius_fraction <= MAX_SURFACE_RADIUS)
        & (prediction >= MIN_BOUNDARY_PREDICTION)
        & (nearest_success < nearest_nonfull)
    )
    boundary_mask &= np.asarray([
        tuple(np.round(point, 8)) not in selected_coordinates for point in candidates
    ])
    boundary_pool = candidates[boundary_mask]
    boundary_pool_points = candidate_points[boundary_mask]
    boundary_prediction = prediction[boundary_mask]
    boundary_margin = box_margin[boundary_mask]
    if len(boundary_pool) < BOUNDARY_DESIGNS:
        raise RuntimeError(f"Only {len(boundary_pool)} boundary candidates are eligible")
    boundary_points = greedy_surface_select(
        boundary_pool_points,
        measured_points,
        boundary_prediction,
        boundary_margin,
        hull_center,
        BOUNDARY_DESIGNS,
    )
    boundary = LOWER + boundary_points * (UPPER - LOWER)
    selected_coordinates.update(tuple(np.round(point, 8)) for point in boundary)

    bridge_groups = []
    bridge_references = np.vstack((full_points, upper_surface_points, boundary_points))
    for hip_level in UPPER_BRIDGE_LEVELS:
        bridge_mask = (
            (candidates[:, 2] == hip_level)
            & (candidates[:, 0] >= -25.0)
            & (candidates[:, 0] <= 37.5)
            & (candidates[:, 1] >= -18.75)
            & (candidates[:, 1] <= 56.25)
            & inside
            & (prediction >= MIN_UPPER_BRIDGE_PREDICTION)
            & (nearest_success < nearest_nonfull)
        )
        bridge_mask &= np.asarray([
            tuple(np.round(point, 8)) not in selected_coordinates for point in candidates
        ])
        bridge_pool = candidates[bridge_mask]
        if len(bridge_pool) < UPPER_BRIDGE_PER_LEVEL:
            raise RuntimeError(
                f"Only {len(bridge_pool)} bridge candidates are eligible at hip {hip_level}%"
            )
        level_points = greedy_select(
            candidate_points[bridge_mask],
            bridge_references,
            UPPER_BRIDGE_PER_LEVEL,
            ((0.45, prediction[bridge_mask]),),
        )
        bridge_groups.append(level_points)
        bridge_references = np.vstack((bridge_references, level_points))
        level_values = LOWER + level_points * (UPPER - LOWER)
        selected_coordinates.update(tuple(np.round(point, 8)) for point in level_values)
    upper_bridge_points = np.vstack(bridge_groups)
    upper_bridge = LOWER + upper_bridge_points * (UPPER - LOWER)

    interior_mask = (
        inside
        & (radius_fraction >= MIN_INTERIOR_RADIUS)
        & (radius_fraction <= MAX_INTERIOR_RADIUS)
        & (prediction >= MIN_INTERIOR_PREDICTION)
        & (nearest_success < nearest_nonfull)
    )
    interior_mask &= np.asarray([
        tuple(np.round(point, 8)) not in selected_coordinates for point in candidates
    ])
    interior_pool = candidates[interior_mask]
    interior_pool_points = candidate_points[interior_mask]
    interior_prediction = prediction[interior_mask]
    interior_radius = radius_fraction[interior_mask]
    middle_shell_score = -np.abs(interior_radius - 0.66)
    if len(interior_pool) < INTERIOR_DESIGNS:
        raise RuntimeError(f"Only {len(interior_pool)} interior candidates are eligible")
    interior_points = greedy_select(
        interior_pool_points,
        np.vstack((full_points, boundary_points, upper_surface_points, upper_bridge_points)),
        INTERIOR_DESIGNS,
        ((0.25, middle_shell_score), (0.20, interior_prediction)),
    )
    interior = LOWER + interior_points * (UPPER - LOWER)

    all_selected = np.vstack((boundary, upper_surface, upper_bridge, interior))
    all_selected_points = normalized(all_selected)
    all_prediction, _ = predict_completion(
        all_selected_points, measured_points, completion
    )
    all_nearest_success = np.linalg.norm(
        all_selected_points[:, None, :] - full_points[None, :, :], axis=2
    ).min(axis=1)
    all_nearest_nonfull = np.linalg.norm(
        all_selected_points[:, None, :] - nonfull_points[None, :, :], axis=2
    ).min(axis=1)
    all_hull_values = all_selected_points @ hull.equations[:, :3].T + hull.equations[:, 3]
    all_violation = all_hull_values.max(axis=1)

    categories = (
        ["balanced_boundary"] * len(boundary)
        + ["upper_hip_surface"] * len(upper_surface)
        + ["nominal_to_upper_bridge"] * len(upper_bridge)
        + ["interior_fill"] * len(interior)
    )
    entries = []
    for index, (levels, category) in enumerate(zip(all_selected, categories)):
        entries.append({
            "name": f"volume_followup_{index + 1:03d}",
            "theta": theta_from_percent(nominal, levels),
            "percent_from_nominal": levels.tolist(),
            "category": category,
            "predicted_completion": float(all_prediction[index]),
            "normalized_distance_to_nearest_full_success": float(all_nearest_success[index]),
            "normalized_distance_to_nearest_nonfull": float(all_nearest_nonfull[index]),
            "signed_normalized_hull_offset": float(all_violation[index]),
        })

    args.output.mkdir(parents=True, exist_ok=True)
    plan = {
        "description": "Balanced hull coverage with positive-hip cap and nominal-to-cap bridge",
        "selection": (
            "6.25%-increment lattice; 32 direction-balanced hull points; 18 exploratory "
            "positive-hip cap points; 20 bridge points across five hip levels from nominal "
            "toward the cap; 12 general middle-shell points"
        ),
        "source_dataset": str(args.dataset.resolve()),
        "nominal_theta": nominal.tolist(),
        "grid_step_percent": GRID_STEP,
        "preserve_link_masses": True,
        "hypothetical_completion_assumption": "All proposed points complete all four episodes",
        "morphologies": entries,
    }
    plan_path = args.output / "volume_followup_plan_82.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")

    with (args.output / "volume_followup_points.csv").open("w", newline="") as stream:
        fields = (
            "name", "category", "thigh_change_percent", "shin_change_percent",
            "hip_change_percent", "predicted_completion",
            "normalized_distance_to_nearest_full_success",
            "normalized_distance_to_nearest_nonfull", "signed_normalized_hull_offset",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            writer.writerow(dict(zip(fields, (
                entry["name"], entry["category"], *entry["percent_from_nominal"],
                entry["predicted_completion"],
                entry["normalized_distance_to_nearest_full_success"],
                entry["normalized_distance_to_nearest_nonfull"],
                entry["signed_normalized_hull_offset"],
            ))))

    current_volume, hypothetical_volume, increase = plot_volume(
        measured, completion, boundary, upper_surface, upper_bridge, interior, args.output
    )
    plot_projections(
        measured, completion, boundary, upper_surface, upper_bridge, interior, args.output
    )
    readme = [
        "# Proposed 4/4 Volume Follow-Up", "",
        "This is a preview only. No MPC episodes were run, and every proposed point is shown",
        "as completing 4/4 solely to visualize the potential geometry.", "",
        "## Distribution", "",
        f"- Balanced boundary points: {BOUNDARY_DESIGNS}",
        f"- Positive-hip cap points: {UPPER_SURFACE_DESIGNS}",
        f"- Nominal-to-cap bridge points: {len(UPPER_BRIDGE_LEVELS) * UPPER_BRIDGE_PER_LEVEL}",
        f"- Interior gap-fill points: {INTERIOR_DESIGNS}",
        f"- Total proposed points: {len(entries)}", "",
        "## Hypothetical Effect", "",
        f"- Current normalized 4/4 convex-hull volume: `{current_volume:.6f}`",
        f"- Hypothetical normalized 4/4 convex-hull volume: `{hypothetical_volume:.6f}`",
        f"- Hypothetical convex-hull increase: `{increase:.2f}%`", "",
        "Balanced boundary points require at least 95% local predicted completion. The upper",
        "cap uses a 90% threshold because no untested candidates above +62.5% hip meet 95%.",
        "The bridge uses five explicit hip levels from +12.5% through +62.5%, each at 95% or",
        "higher predicted completion. Predictions are selection heuristics,",
        "not guarantees of successful completion.",
    ]
    (args.output / "README.md").write_text("\n".join(readme) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
