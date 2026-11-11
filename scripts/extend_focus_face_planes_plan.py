"""Extend a focused surface plan toward every boundary plane of its target face."""

from __future__ import annotations

import argparse
import csv
from itertools import product
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
from generate_dense_surface_radial_plan import assign_radial_pairs


ADDED_SURFACE_POINTS = 72
PLANE_SEEDS = 4
OUTER_FRACTION = 2.0 / 3.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def farthest_indices(candidate_points, references, count, unavailable=None):
    unavailable = (
        np.zeros(len(candidate_points), dtype=bool)
        if unavailable is None else unavailable.copy()
    )
    selected = []
    nearest = np.linalg.norm(
        candidate_points[:, None, :] - references[None, :, :], axis=2
    ).min(axis=1)
    for _ in range(count):
        nearest[unavailable] = -np.inf
        chosen = int(np.argmax(nearest))
        if not np.isfinite(nearest[chosen]):
            break
        selected.append(chosen)
        unavailable[chosen] = True
        nearest = np.minimum(
            nearest, np.linalg.norm(candidate_points - candidate_points[chosen], axis=1)
        )
    return selected, unavailable


def configure_axis(axis, title, azimuth=-58):
    axis.set(
        xlim=(-80, 105), ylim=(-80, 105), zlim=(-55, 105),
        xlabel="Thigh change (%)", ylabel="Shin change (%)", zlabel="Hip change (%)",
        title=title,
    )
    axis.view_init(elev=24, azim=azimuth)
    axis.grid(alpha=0.3)


def add_pairs(axis, surface, outer, color, outer_color, label=None, alpha=1.0):
    for surface_point, outer_point in zip(surface, outer):
        path = np.vstack((np.zeros(3), outer_point, surface_point))
        axis.plot(
            path[:, 0], path[:, 1], path[:, 2], color=color,
            linewidth=0.45, alpha=0.20 * alpha,
        )
    axis.scatter(
        surface[:, 0], surface[:, 1], surface[:, 2], c=color, marker="^",
        s=54, alpha=alpha, edgecolors="#262126", linewidths=0.4,
        depthshade=False, label=label,
    )
    axis.scatter(
        outer[:, 0], outer[:, 1], outer[:, 2], c=outer_color, marker="s",
        s=38, alpha=alpha, edgecolors="#263326", linewidths=0.35,
        depthshade=False,
    )


def plot_comparison(full, base_surface, base_outer, added_surface, added_outer, output):
    before = np.vstack((full, base_surface))
    after = np.vstack((before, added_surface))
    before_volume = ConvexHull(normalized(before)).volume
    after_volume = ConvexHull(normalized(after)).volume
    increase = 100.0 * (after_volume / before_volume - 1.0)

    figure = plt.figure(figsize=(19, 8.5))
    before_axis = figure.add_subplot(1, 2, 1, projection="3d")
    before_axis.add_collection3d(hull_collection(before, "#5b9c74", "#174f2a", 0.14))
    add_pairs(before_axis, base_surface, base_outer, "#b35439", "#48a968",
              label="Retained 288-point plan", alpha=0.75)
    before_axis.scatter([0], [0], [0], marker="*", s=165, c="yellow",
                        edgecolors="black", linewidths=0.7, depthshade=False)
    configure_axis(before_axis, "Retained focused plan\n144 anchor-pair sets")

    after_axis = figure.add_subplot(1, 2, 2, projection="3d")
    after_axis.add_collection3d(hull_collection(after, "#5b9c74", "#174f2a", 0.13))
    add_pairs(after_axis, base_surface, base_outer, "#b7a9a4", "#aabbb0",
              label="Retained anchors", alpha=0.42)
    add_pairs(after_axis, added_surface, added_outer, "#7b2cbf", "#23b5d3",
              label=f"Added plane-spread pairs ({len(added_surface)})")
    after_axis.scatter([0], [0], [0], marker="*", s=165, c="yellow",
                       edgecolors="black", linewidths=0.7, depthshade=False,
                       label="Nominal")
    configure_axis(
        after_axis,
        f"Extended toward every target-face plane\nAdditional hull growth: {increase:.1f}%",
    )
    after_axis.legend(loc="upper left", fontsize=8)
    figure.suptitle(
        "Negative-Shin / Positive-Hip Face Extension",
        fontsize=19, fontweight="bold", y=0.97,
    )
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.88, wspace=0.02)
    stem = output / "01_focus_face_plane_extension"
    figure.savefig(stem.with_suffix(".png"), dpi=190, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return before_volume, after_volume, increase


def plot_four_views(full, base_surface, base_outer, added_surface, added_outer, output):
    after = np.vstack((full, base_surface, added_surface))
    figure = plt.figure(figsize=(17, 15))
    for panel, azimuth in enumerate((-55, 35, 125, 215), start=1):
        axis = figure.add_subplot(2, 2, panel, projection="3d")
        axis.add_collection3d(hull_collection(after, "#5b9c74", "#174f2a", 0.08))
        add_pairs(axis, base_surface, base_outer, "#b7a9a4", "#aabbb0", alpha=0.28)
        add_pairs(axis, added_surface, added_outer, "#7b2cbf", "#23b5d3")
        axis.scatter([0], [0], [0], marker="*", s=140, c="yellow",
                     edgecolors="black", linewidths=0.6, depthshade=False)
        configure_axis(axis, f"Azimuth {azimuth}°", azimuth)
    figure.suptitle(
        "Added Surface-to-Plane Coverage From Four Views",
        fontsize=19, fontweight="bold", y=0.97,
    )
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.92,
                           wspace=0.02, hspace=0.08)
    stem = output / "02_focus_face_plane_extension_four_views"
    figure.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_projections(base_surface, added_surface, added_outer, output):
    projections = (
        (0, 1, "Thigh change (%)", "Shin change (%)"),
        (0, 2, "Thigh change (%)", "Hip change (%)"),
        (1, 2, "Shin change (%)", "Hip change (%)"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(20, 6.6))
    for axis, (x_index, y_index, x_label, y_label) in zip(axes, projections):
        axis.scatter(
            base_surface[:, x_index], base_surface[:, y_index], c="#b7a9a4",
            marker="^", s=27, alpha=0.45, edgecolors="none", label="Retained anchors",
        )
        for surface_point, outer_point in zip(added_surface, added_outer):
            path = np.vstack((np.zeros(3), outer_point, surface_point))
            axis.plot(path[:, x_index], path[:, y_index], color="#7b2cbf",
                      linewidth=0.45, alpha=0.20)
        axis.scatter(
            added_surface[:, x_index], added_surface[:, y_index], c="#7b2cbf",
            marker="^", s=52, edgecolors="#29133c", linewidths=0.4,
            label="Added surface-to-plane anchors",
        )
        axis.scatter(
            added_outer[:, x_index], added_outer[:, y_index], c="#23b5d3",
            marker="s", s=37, edgecolors="#0b4652", linewidths=0.35,
            label="Added two-thirds companions",
        )
        axis.scatter([0], [0], marker="*", s=145, c="yellow", edgecolors="black",
                     linewidths=0.7, label="Nominal")
        axis.set(xlabel=x_label, ylabel=y_label)
        axis.grid(alpha=0.25)
        axis.set_aspect("equal", adjustable="box")
    axes[0].legend(loc="best", fontsize=8)
    figure.suptitle(
        "72 Added Anchors Spread Toward All Boundary Planes of the Target Face",
        fontsize=17, fontweight="bold",
    )
    figure.tight_layout()
    stem = output / "03_focus_face_plane_extension_projections"
    figure.savefig(stem.with_suffix(".png"), dpi=190, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    dataset = json.loads(args.dataset.read_text())
    base_plan = json.loads(args.base_plan.read_text())
    nominal = np.asarray(dataset["nominal_theta"], dtype=float)
    rows = dataset["morphologies"]
    measured = np.asarray([row["percent_from_nominal"] for row in rows], dtype=float)
    completion = np.asarray([row["completion_fraction"] for row in rows], dtype=float)
    measured_points = normalized(measured)
    full_points = measured_points[completion == 1.0]
    hull = ConvexHull(full_points)
    nominal_point = normalized(np.zeros(3))

    base_entries = base_plan["morphologies"]
    base_surface_entries = [
        entry for entry in base_entries if "surface_anchor" in entry["category"]
    ]
    base_outer_entries = [
        entry for entry in base_entries if "outer_radial" in entry["category"]
    ]
    base_surface = np.asarray([
        entry["percent_from_nominal"] for entry in base_surface_entries
    ], dtype=float)
    base_outer = np.asarray([
        entry["percent_from_nominal"] for entry in base_outer_entries
    ], dtype=float)
    base_outer_by_pair = {
        entry["surface_pair_id"]: np.asarray(entry["percent_from_nominal"], dtype=float)
        for entry in base_outer_entries
    }
    base_outer = np.asarray([
        base_outer_by_pair[entry["surface_pair_id"]] for entry in base_surface_entries
    ])

    used = {tuple(np.round(point, 8)) for point in measured}
    used.update(
        tuple(np.round(entry["percent_from_nominal"], 8)) for entry in base_entries
    )
    candidates = np.asarray([
        point for point in product(
            np.arange(LOWER[0], UPPER[0] + GRID_STEP / 2.0, GRID_STEP),
            np.arange(LOWER[1], UPPER[1] + GRID_STEP / 2.0, GRID_STEP),
            np.arange(LOWER[2], UPPER[2] + GRID_STEP / 2.0, GRID_STEP),
        ) if tuple(np.round(point, 8)) not in used
    ], dtype=float)
    candidate_points = normalized(candidates)
    prediction, _ = predict_completion(candidate_points, measured_points, completion)
    hull_offset = (
        candidate_points @ hull.equations[:, :3].T + hull.equations[:, 3]
    ).max(axis=1)
    radius, _ = radial_fraction(candidate_points, hull, nominal_point)
    face_mask = (
        (hull_offset >= -0.05)
        & (hull_offset <= 0.05)
        & (radius >= 0.78)
        & (radius <= 1.30)
        & (prediction >= 0.65)
        & (candidates[:, 1] <= -6.25)
        & (candidates[:, 2] >= 6.25)
    )
    pool_indices = np.flatnonzero(face_mask)
    pool_points = candidate_points[face_mask]
    pool_values = candidates[face_mask]
    selected_local = []
    unavailable = np.zeros(len(pool_points), dtype=bool)
    references = normalized(base_surface)

    for axis_index in range(3):
        for use_maximum in (False, True):
            boundary = (
                pool_values[:, axis_index].max()
                if use_maximum else pool_values[:, axis_index].min()
            )
            plane = pool_values[:, axis_index] == boundary
            plane_unavailable = unavailable | ~plane
            chosen, _ = farthest_indices(
                pool_points, np.vstack((references, pool_points[selected_local]))
                if selected_local else references,
                min(PLANE_SEEDS, int(plane.sum())), plane_unavailable,
            )
            selected_local.extend(chosen)
            unavailable[chosen] = True

    remaining_count = ADDED_SURFACE_POINTS - len(selected_local)
    references_with_seeds = np.vstack((references, pool_points[selected_local]))
    chosen, unavailable = farthest_indices(
        pool_points, references_with_seeds, remaining_count, unavailable
    )
    selected_local.extend(chosen)
    if len(selected_local) != ADDED_SURFACE_POINTS:
        raise RuntimeError(f"Selected {len(selected_local)} of {ADDED_SURFACE_POINTS} anchors")
    added_indices = pool_indices[np.asarray(selected_local)]
    added_surface_points = candidate_points[added_indices]

    pair_unavailable = np.zeros(len(candidates), dtype=bool)
    pair_unavailable[added_indices] = True
    added_outer_indices, outer_ratio, outer_cosine = assign_radial_pairs(
        added_surface_points, candidate_points, prediction, hull_offset,
        pair_unavailable, nominal_point, OUTER_FRACTION, 0.50, 0.84,
    )
    added_surface = candidates[added_indices]
    added_outer = candidates[added_outer_indices]

    pair_start = max(entry["surface_pair_id"] for entry in base_entries) + 1
    added_entries = []
    for index in range(ADDED_SURFACE_POINTS):
        pair_id = pair_start + index
        for category, candidate_index, target, actual_ratio, cosine in (
            ("plane_spread_surface_anchor", added_indices[index], 1.0,
             radius[added_indices[index]], 1.0),
            ("plane_spread_outer_radial", added_outer_indices[index], OUTER_FRACTION,
             outer_ratio[index], outer_cosine[index]),
        ):
            levels = candidates[candidate_index]
            added_entries.append({
                "name": f"face_plane_spread_{index + 1:03d}_{category}",
                "theta": theta_from_percent(nominal, levels),
                "percent_from_nominal": levels.tolist(),
                "category": category,
                "surface_pair_id": pair_id,
                "target_nominal_to_surface_fraction": target,
                "actual_nominal_to_surface_radius_ratio": float(actual_ratio),
                "direction_alignment_cosine": float(cosine),
                "direction_error_degrees": float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))),
                "predicted_completion": float(prediction[candidate_index]),
                "signed_normalized_hull_offset": float(hull_offset[candidate_index]),
            })

    combined_entries = base_entries + added_entries
    args.output.mkdir(parents=True, exist_ok=True)
    output_plan = dict(base_plan)
    output_plan.update({
        "description": "Retained focused plan extended toward every boundary plane of the negative-shin positive-hip face",
        "base_plan": str(args.base_plan.resolve()),
        "added_surface_anchor_count": ADDED_SURFACE_POINTS,
        "surface_anchor_count": 216,
        "points_per_anchor": 2,
        "morphologies": combined_entries,
    })
    (args.output / "focus_face_plane_extension_plan_432.json").write_text(
        json.dumps(output_plan, indent=2) + "\n"
    )

    with (args.output / "added_face_plane_points.csv").open("w", newline="") as stream:
        fields = (
            "name", "category", "surface_pair_id", "thigh_change_percent",
            "shin_change_percent", "hip_change_percent", "predicted_completion",
            "target_nominal_to_surface_fraction", "actual_nominal_to_surface_radius_ratio",
            "direction_error_degrees", "signed_normalized_hull_offset",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for entry in added_entries:
            writer.writerow(dict(zip(fields, (
                entry["name"], entry["category"], entry["surface_pair_id"],
                *entry["percent_from_nominal"], entry["predicted_completion"],
                entry["target_nominal_to_surface_fraction"],
                entry["actual_nominal_to_surface_radius_ratio"],
                entry["direction_error_degrees"], entry["signed_normalized_hull_offset"],
            ))))

    full = measured[completion == 1.0]
    before_volume, after_volume, increase = plot_comparison(
        full, base_surface, base_outer, added_surface, added_outer, args.output
    )
    plot_four_views(full, base_surface, base_outer, added_surface, added_outer, args.output)
    plot_projections(base_surface, added_surface, added_outer, args.output)
    max_angle = float(np.degrees(np.arccos(np.clip(outer_cosine, -1.0, 1.0))).max())
    lines = [
        "# Focus-Face Plane Extension", "",
        "This is a preview only. No MPC episodes were run.", "",
        "## Distribution", "",
        "- Retained morphologies: 288",
        f"- Added surface-to-plane anchors: {len(added_surface)}",
        f"- Added two-thirds companions: {len(added_outer)}",
        f"- Combined morphologies: {len(combined_entries)}",
        "- One-third companions: 0", "",
        "The added anchors begin with samples on the minimum and maximum thigh, shin, and",
        "hip planes available within the negative-shin/positive-hip face. Farthest-point",
        "sampling then fills the spaces between those plane seeds.", "",
        "## Added Coverage", "",
        f"- Added anchor thigh range: `{added_surface[:, 0].min():.2f}%` to `{added_surface[:, 0].max():.2f}%`",
        f"- Added anchor shin range: `{added_surface[:, 1].min():.2f}%` to `{added_surface[:, 1].max():.2f}%`",
        f"- Added anchor hip range: `{added_surface[:, 2].min():.2f}%` to `{added_surface[:, 2].max():.2f}%`",
        f"- Maximum added radial direction error: `{max_angle:.2f} degrees`", "",
        "## Hypothetical Effect", "",
        f"- Retained-plan normalized hull volume: `{before_volume:.6f}`",
        f"- Extended normalized hull volume: `{after_volume:.6f}`",
        f"- Additional hypothetical hull increase: `{increase:.2f}%`", "",
        "Added surface anchors use a 65% local predicted-completion floor. This is an",
        "exploratory sampling heuristic, not a guarantee of successful completion.",
    ]
    (args.output / "README.md").write_text("\n".join(lines) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
