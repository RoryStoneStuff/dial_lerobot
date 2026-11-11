"""Densely sample the successful hull and pair each surface point toward nominal."""

from __future__ import annotations

import argparse
import csv
from itertools import product
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull

from generate_completion_volume_followup_plan import (
    GRID_STEP,
    LOWER,
    UPPER,
    hull_collection,
    normalized,
    predict_completion,
    radial_fraction,
    theta_from_percent,
)


SURFACE_POINTS = 96
FOCUS_SURFACE_POINTS = 48
OUTER_FRACTION = 2.0 / 3.0
MIN_SURFACE_PREDICTION = 0.80
MIN_RADIAL_PREDICTION = 0.75


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def unit_directions(points, origin):
    offsets = np.asarray(points, dtype=float) - origin
    radii = np.linalg.norm(offsets, axis=1)
    return offsets / np.maximum(radii[:, None], 1e-12), radii


def select_surface_points(pool, hull_vertices, count):
    selected = []
    available = np.ones(len(pool), dtype=bool)
    for vertex in hull_vertices:
        distances = np.linalg.norm(pool - vertex, axis=1)
        distances[~available] = np.inf
        chosen = int(np.argmin(distances))
        selected.append(chosen)
        available[chosen] = False

    for axis_index in range(3):
        for sign in (-1, 1):
            priority = sign * pool[:, axis_index]
            priority[~available] = -np.inf
            chosen = int(np.argmax(priority))
            selected.append(chosen)
            available[chosen] = False

    nearest = np.linalg.norm(
        pool[:, None, :] - pool[np.asarray(selected)][None, :, :], axis=2
    ).min(axis=1)
    nearest[~available] = -np.inf
    while len(selected) < count:
        chosen = int(np.argmax(nearest))
        selected.append(chosen)
        available[chosen] = False
        distance = np.linalg.norm(pool - pool[chosen], axis=1)
        nearest = np.minimum(nearest, distance)
        nearest[~available] = -np.inf
    return np.asarray(selected, dtype=int)


def select_focus_points(pool, existing_surface, count):
    selected = []
    available = np.ones(len(pool), dtype=bool)
    nearest = np.linalg.norm(
        pool[:, None, :] - existing_surface[None, :, :], axis=2
    ).min(axis=1)
    for _ in range(count):
        nearest[~available] = -np.inf
        chosen = int(np.argmax(nearest))
        selected.append(chosen)
        available[chosen] = False
        distance = np.linalg.norm(pool - pool[chosen], axis=1)
        nearest = np.minimum(nearest, distance)
    return np.asarray(selected, dtype=int)


def assign_radial_pairs(surface_points, candidate_points, candidate_prediction,
                        hull_offset, unavailable, origin, target_fraction,
                        minimum_ratio, maximum_ratio):
    surface_directions, surface_radii = unit_directions(surface_points, origin)
    candidate_directions, candidate_radii = unit_directions(candidate_points, origin)
    cosine = surface_directions @ candidate_directions.T
    ratio = candidate_radii[None, :] / surface_radii[:, None]
    targets = origin + target_fraction * (surface_points - origin)
    distance = np.linalg.norm(
        targets[:, None, :] - candidate_points[None, :, :], axis=2
    )
    valid_candidates = (
        (hull_offset <= 1e-9)
        & (candidate_prediction >= MIN_RADIAL_PREDICTION)
        & ~unavailable
    )
    valid = (
        valid_candidates[None, :]
        & (ratio >= minimum_ratio)
        & (ratio <= maximum_ratio)
        & (cosine >= 0.92)
    )
    cost = distance + 1.00 * (1.0 - cosine) + 0.15 * np.abs(ratio - target_fraction)
    cost[~valid] = 1e6
    rows, columns = linear_sum_assignment(cost)
    if len(rows) != len(surface_points) or np.any(cost[rows, columns] >= 1e5):
        raise RuntimeError(f"Could not assign all radial pairs at fraction {target_fraction}")
    assignment = np.empty(len(surface_points), dtype=int)
    assignment[rows] = columns
    return assignment, ratio[np.arange(len(surface_points)), assignment], cosine[
        np.arange(len(surface_points)), assignment
    ]


def add_hull(axis, points, facecolor, edgecolor, alpha):
    axis.add_collection3d(hull_collection(points, facecolor, edgecolor, alpha))


def configure_axis(axis, title, azimuth=-58):
    axis.set(
        xlim=(-80, 105), ylim=(-80, 105), zlim=(-55, 105),
        xlabel="Thigh change (%)", ylabel="Shin change (%)", zlabel="Hip change (%)",
        title=title,
    )
    axis.view_init(elev=24, azim=azimuth)
    axis.grid(alpha=0.3)


def add_radial_design(axis, surface, focus_surface, outer, focus_outer,
                      show_labels=False):
    for surface_point, outer_point in zip(
        np.vstack((surface, focus_surface)), np.vstack((outer, focus_outer))
    ):
        path = np.vstack((np.zeros(3), outer_point, surface_point))
        axis.plot(
            path[:, 0], path[:, 1], path[:, 2], color="#536471",
            linewidth=0.45, alpha=0.24,
        )
    axis.scatter(
        surface[:, 0], surface[:, 1], surface[:, 2], c="#e34a33", marker="^",
        s=48, edgecolors="#5c160d", linewidths=0.45, depthshade=False,
        label=(f"Surface anchors ({len(surface)})" if show_labels else None),
    )
    axis.scatter(
        focus_surface[:, 0], focus_surface[:, 1], focus_surface[:, 2],
        c="#d53e9f", marker="^", s=58, edgecolors="#59113f", linewidths=0.5,
        depthshade=False,
        label=(f"Focused face anchors ({len(focus_surface)})" if show_labels else None),
    )
    axis.scatter(
        outer[:, 0], outer[:, 1], outer[:, 2], c="#35a853", marker="s",
        s=34, edgecolors="#10491e", linewidths=0.4, depthshade=False,
        label=(f"Two-thirds radial ({len(outer)})" if show_labels else None),
    )
    axis.scatter(
        focus_outer[:, 0], focus_outer[:, 1], focus_outer[:, 2], c="#8dd35f",
        marker="s", s=40, edgecolors="#315d18", linewidths=0.45, depthshade=False,
        label=(f"Focused two-thirds ({len(focus_outer)})" if show_labels else None),
    )
    axis.scatter(
        [0], [0], [0], marker="*", s=170, c="yellow", edgecolors="black",
        linewidths=0.7, depthshade=False, label=("Nominal" if show_labels else None),
    )


def plot_comparison(measured, completion, surface, focus_surface, outer,
                    focus_outer, output):
    current = measured[completion == 1.0]
    hypothetical = np.vstack((current, surface, focus_surface))
    current_volume = ConvexHull(normalized(current)).volume
    hypothetical_volume = ConvexHull(normalized(hypothetical)).volume
    increase = 100.0 * (hypothetical_volume / current_volume - 1.0)

    figure = plt.figure(figsize=(19, 8.5))
    current_axis = figure.add_subplot(1, 2, 1, projection="3d")
    add_hull(current_axis, current, "#6f58a8", "#30254d", 0.24)
    current_axis.scatter(
        current[:, 0], current[:, 1], current[:, 2], c="#34205c", s=25,
        edgecolors="white", linewidths=0.2, depthshade=False,
    )
    configure_axis(current_axis, f"Current measured hull\n{len(current)} 4/4 morphologies")

    preview_axis = figure.add_subplot(1, 2, 2, projection="3d")
    add_hull(preview_axis, hypothetical, "#48a56a", "#174f2a", 0.16)
    preview_axis.scatter(
        current[:, 0], current[:, 1], current[:, 2], c="#817b89", s=12,
        alpha=0.35, edgecolors="none", depthshade=False,
    )
    add_radial_design(
        preview_axis, surface, focus_surface, outer, focus_outer, show_labels=True
    )
    configure_axis(
        preview_axis,
        f"Hypothetical dense surface validation\nHull increase: {increase:.1f}%",
    )
    preview_axis.legend(loc="upper left", fontsize=8)
    figure.suptitle(
        "Dense Whole-Surface Coverage With Extra -Shin/+Hip Face Sampling",
        fontsize=18, fontweight="bold", y=0.97,
    )
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.88, wspace=0.02)
    stem = output / "01_dense_surface_hypothetical_volume"
    figure.savefig(stem.with_suffix(".png"), dpi=190, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return current_volume, hypothetical_volume, increase


def plot_four_views(measured, completion, surface, focus_surface, outer,
                    focus_outer, output):
    current = measured[completion == 1.0]
    hypothetical = np.vstack((current, surface, focus_surface))
    figure = plt.figure(figsize=(17, 15))
    for panel, azimuth in enumerate((-55, 35, 125, 215), start=1):
        axis = figure.add_subplot(2, 2, panel, projection="3d")
        add_hull(axis, hypothetical, "#48a56a", "#174f2a", 0.10)
        add_radial_design(axis, surface, focus_surface, outer, focus_outer)
        configure_axis(axis, f"Azimuth {azimuth}°", azimuth=azimuth)
    figure.suptitle(
        "Four Views of Dense Surface and Near-Radial Coverage",
        fontsize=19, fontweight="bold", y=0.97,
    )
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.92, wspace=0.02, hspace=0.08)
    stem = output / "02_dense_surface_four_views"
    figure.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_projections(measured, completion, surface, focus_surface, outer,
                     focus_outer, output):
    full = measured[completion == 1.0]
    projections = (
        (0, 1, "Thigh change (%)", "Shin change (%)"),
        (0, 2, "Thigh change (%)", "Hip change (%)"),
        (1, 2, "Shin change (%)", "Hip change (%)"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(20, 6.6))
    for axis, (x_index, y_index, x_label, y_label) in zip(axes, projections):
        axis.scatter(
            full[:, x_index], full[:, y_index], c="#817b89", s=18,
            alpha=0.45, edgecolors="none", label="Existing 4/4",
        )
        for surface_point, outer_point in zip(
            np.vstack((surface, focus_surface)), np.vstack((outer, focus_outer))
        ):
            path = np.vstack((np.zeros(3), outer_point, surface_point))
            axis.plot(
                path[:, x_index], path[:, y_index], color="#536471",
                linewidth=0.4, alpha=0.20,
            )
        axis.scatter(
            surface[:, x_index], surface[:, y_index], c="#e34a33", marker="^",
            s=42, edgecolors="#5c160d", linewidths=0.4, label="Surface anchors",
        )
        axis.scatter(
            focus_surface[:, x_index], focus_surface[:, y_index], c="#d53e9f",
            marker="^", s=50, edgecolors="#59113f", linewidths=0.45,
            label="Focused face anchors",
        )
        axis.scatter(
            outer[:, x_index], outer[:, y_index], c="#35a853", marker="s",
            s=31, edgecolors="#10491e", linewidths=0.35, label="Two-thirds radial",
        )
        axis.scatter(
            focus_outer[:, x_index], focus_outer[:, y_index], c="#8dd35f",
            marker="s", s=35, edgecolors="#315d18", linewidths=0.4,
            label="Focused two-thirds",
        )
        axis.scatter(
            [0], [0], marker="*", s=150, c="yellow", edgecolors="black",
            linewidths=0.7, label="Nominal",
        )
        axis.set(xlabel=x_label, ylabel=y_label)
        axis.grid(alpha=0.25)
        axis.set_aspect("equal", adjustable="box")
    axes[0].legend(loc="best", fontsize=8)
    figure.suptitle(
        "144 Surface Anchors With Two-Thirds Companions; Extra Density on -Shin/+Hip Face",
        fontsize=17, fontweight="bold",
    )
    figure.tight_layout()
    stem = output / "03_dense_surface_radial_projections"
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
    nominal_point = normalized(np.zeros(3))

    existing = {tuple(np.round(point, 8)) for point in measured}
    candidates = np.asarray([
        point for point in product(
            np.arange(LOWER[0], UPPER[0] + GRID_STEP / 2.0, GRID_STEP),
            np.arange(LOWER[1], UPPER[1] + GRID_STEP / 2.0, GRID_STEP),
            np.arange(LOWER[2], UPPER[2] + GRID_STEP / 2.0, GRID_STEP),
        ) if tuple(np.round(point, 8)) not in existing
    ], dtype=float)
    candidate_points = normalized(candidates)
    prediction, _ = predict_completion(candidate_points, measured_points, completion)
    hull_offset = (
        candidate_points @ hull.equations[:, :3].T + hull.equations[:, 3]
    ).max(axis=1)
    radius, _ = radial_fraction(candidate_points, hull, nominal_point)

    surface_mask = (
        (hull_offset >= -0.035)
        & (hull_offset <= 0.035)
        & (radius >= 0.85)
        & (radius <= 1.20)
        & (prediction >= MIN_SURFACE_PREDICTION)
    )
    surface_pool = candidate_points[surface_mask]
    surface_pool_indices = np.flatnonzero(surface_mask)
    surface_indices_in_pool = select_surface_points(
        surface_pool, full_points[hull.vertices], SURFACE_POINTS
    )
    surface_indices = surface_pool_indices[surface_indices_in_pool]
    surface_points = candidate_points[surface_indices]

    focus_mask = (
        (hull_offset >= -0.04)
        & (hull_offset <= 0.04)
        & (radius >= 0.82)
        & (radius <= 1.25)
        & (prediction >= 0.75)
        & (candidates[:, 1] <= -6.25)
        & (candidates[:, 2] >= 6.25)
    )
    focus_mask[surface_indices] = False
    focus_pool = candidate_points[focus_mask]
    focus_pool_indices = np.flatnonzero(focus_mask)
    focus_indices_in_pool = select_focus_points(
        focus_pool, surface_points, FOCUS_SURFACE_POINTS
    )
    focus_indices = focus_pool_indices[focus_indices_in_pool]
    focus_surface_points = candidate_points[focus_indices]
    all_surface_points = np.vstack((surface_points, focus_surface_points))

    unavailable = np.zeros(len(candidates), dtype=bool)
    unavailable[surface_indices] = True
    unavailable[focus_indices] = True
    outer_indices, outer_ratio, outer_cosine = assign_radial_pairs(
        all_surface_points, candidate_points, prediction, hull_offset, unavailable,
        nominal_point, OUTER_FRACTION, 0.50, 0.84,
    )
    surface = candidates[surface_indices]
    focus_surface = candidates[focus_indices]
    outer = candidates[outer_indices[:SURFACE_POINTS]]
    focus_outer = candidates[outer_indices[SURFACE_POINTS:]]

    nearest_success = np.linalg.norm(
        candidate_points[:, None, :] - full_points[None, :, :], axis=2
    ).min(axis=1)
    nearest_nonfull = np.linalg.norm(
        candidate_points[:, None, :] - nonfull_points[None, :, :], axis=2
    ).min(axis=1)
    entries = []
    all_surface_indices = np.concatenate((surface_indices, focus_indices))
    for pair_index in range(len(all_surface_indices)):
        focused = pair_index >= SURFACE_POINTS
        surface_category = "focus_surface_anchor" if focused else "surface_anchor"
        outer_category = "focus_outer_radial" if focused else "outer_radial"
        for category, candidate_index, target_fraction, actual_ratio, cosine in (
            (surface_category, all_surface_indices[pair_index], 1.0,
             radius[all_surface_indices[pair_index]], 1.0),
            (outer_category, outer_indices[pair_index], OUTER_FRACTION,
             outer_ratio[pair_index], outer_cosine[pair_index]),
        ):
            levels = candidates[candidate_index]
            entries.append({
                "name": f"dense_surface_{pair_index + 1:03d}_{category}",
                "theta": theta_from_percent(nominal, levels),
                "percent_from_nominal": levels.tolist(),
                "category": category,
                "surface_pair_id": pair_index + 1,
                "target_nominal_to_surface_fraction": target_fraction,
                "actual_nominal_to_surface_radius_ratio": float(actual_ratio),
                "direction_alignment_cosine": float(cosine),
                "direction_error_degrees": float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))),
                "predicted_completion": float(prediction[candidate_index]),
                "normalized_distance_to_nearest_full_success": float(nearest_success[candidate_index]),
                "normalized_distance_to_nearest_nonfull": float(nearest_nonfull[candidate_index]),
                "signed_normalized_hull_offset": float(hull_offset[candidate_index]),
            })

    args.output.mkdir(parents=True, exist_ok=True)
    plan = {
        "description": "Dense whole-surface sampling focused on the negative-shin positive-hip face",
        "selection": (
            "96 surface anchors seeded around every measured hull vertex and all six axis "
            "extremes, plus 48 extra anchors on the shin <= -6.25%, hip >= +6.25% face; "
            "each anchor has one unique near-line companion around two-thirds from nominal"
        ),
        "source_dataset": str(args.dataset.resolve()),
        "nominal_theta": nominal.tolist(),
        "grid_step_percent": GRID_STEP,
        "preserve_link_masses": True,
        "hypothetical_completion_assumption": "All proposed points complete all four episodes",
        "surface_anchor_count": len(all_surface_indices),
        "focus_surface_anchor_count": FOCUS_SURFACE_POINTS,
        "points_per_anchor": 2,
        "morphologies": entries,
    }
    (args.output / "dense_surface_focus_plan_288.json").write_text(
        json.dumps(plan, indent=2) + "\n"
    )

    with (args.output / "dense_surface_focus_points.csv").open("w", newline="") as stream:
        fields = (
            "name", "category", "surface_pair_id", "thigh_change_percent",
            "shin_change_percent", "hip_change_percent", "predicted_completion",
            "target_nominal_to_surface_fraction", "actual_nominal_to_surface_radius_ratio",
            "direction_error_degrees", "signed_normalized_hull_offset",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            writer.writerow(dict(zip(fields, (
                entry["name"], entry["category"], entry["surface_pair_id"],
                *entry["percent_from_nominal"], entry["predicted_completion"],
                entry["target_nominal_to_surface_fraction"],
                entry["actual_nominal_to_surface_radius_ratio"],
                entry["direction_error_degrees"], entry["signed_normalized_hull_offset"],
            ))))

    current_volume, hypothetical_volume, increase = plot_comparison(
        measured, completion, surface, focus_surface, outer, focus_outer, args.output
    )
    plot_four_views(
        measured, completion, surface, focus_surface, outer, focus_outer, args.output
    )
    plot_projections(
        measured, completion, surface, focus_surface, outer, focus_outer, args.output
    )
    max_outer_angle = float(np.degrees(np.arccos(np.clip(outer_cosine, -1.0, 1.0))).max())
    lines = [
        "# Dense Whole-Surface Radial Follow-Up", "",
        "This is a preview only. No MPC episodes were run; all proposed points are shown",
        "as hypothetical 4/4 successes.", "",
        "## Distribution", "",
        f"- Whole-surface anchors: {len(surface)}",
        f"- Extra negative-shin/positive-hip face anchors: {len(focus_surface)}",
        f"- Two-thirds radial companions: {len(outer) + len(focus_outer)}",
        "- One-third radial companions: 0",
        f"- Total proposed morphologies: {len(entries)}", "",
        "Every surface anchor has one unique two-thirds companion. The extra magenta anchors",
        "and light-green companions concentrate coverage on the negative-shin, positive-hip",
        "face while the original whole-surface distribution is retained.", "",
        "## Pair Alignment", "",
        f"- Maximum two-thirds direction error: `{max_outer_angle:.2f} degrees`", "",
        "## Hypothetical Effect", "",
        f"- Current normalized hull volume: `{current_volume:.6f}`",
        f"- Hypothetical normalized hull volume: `{hypothetical_volume:.6f}`",
        f"- Hypothetical hull increase: `{increase:.2f}%`", "",
        "Whole-surface anchors use an 80% local predicted-completion floor; focused anchors",
        "and radial companions use 75%. These are sampling heuristics, not guarantees.",
    ]
    (args.output / "README.md").write_text("\n".join(lines) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
