"""Extend a retained face plan toward increasing thigh and increasing hip."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

from extend_focus_face_planes_plan import add_pairs, configure_axis, farthest_indices
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


ADDED_SURFACE_POINTS = 36
OUTER_FRACTION = 2.0 / 3.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def plot_comparison(full, base_surface, base_outer, added_surface, added_outer, output):
    before = np.vstack((full, base_surface))
    after = np.vstack((before, added_surface))
    before_volume = ConvexHull(normalized(before)).volume
    after_volume = ConvexHull(normalized(after)).volume
    increase = 100.0 * (after_volume / before_volume - 1.0)

    figure = plt.figure(figsize=(19, 8.5))
    before_axis = figure.add_subplot(1, 2, 1, projection="3d")
    before_axis.add_collection3d(hull_collection(before, "#5b9c74", "#174f2a", 0.12))
    add_pairs(before_axis, base_surface, base_outer, "#b7a9a4", "#aabbb0",
              label="Retained 432-point plan", alpha=0.45)
    before_axis.scatter([0], [0], [0], marker="*", s=165, c="yellow",
                        edgecolors="black", linewidths=0.7, depthshade=False)
    configure_axis(before_axis, "Retained face-extension plan\n216 anchor-pair sets")

    after_axis = figure.add_subplot(1, 2, 2, projection="3d")
    after_axis.add_collection3d(hull_collection(after, "#5b9c74", "#174f2a", 0.12))
    add_pairs(after_axis, base_surface, base_outer, "#c4bdb9", "#b8c5bc",
              label="Retained anchors", alpha=0.30)
    add_pairs(after_axis, added_surface, added_outer, "#ff7f0e", "#1f77b4",
              label=f"Added upper-right pairs ({len(added_surface)})")
    after_axis.scatter([0], [0], [0], marker="*", s=165, c="yellow",
                       edgecolors="black", linewidths=0.7, depthshade=False,
                       label="Nominal")
    configure_axis(
        after_axis,
        f"Higher thigh / higher hip extension\nAdditional hull growth: {increase:.1f}%",
    )
    after_axis.legend(loc="upper left", fontsize=8)
    figure.suptitle(
        "Upper-Right Continuation Along the Negative-Shin Face",
        fontsize=19, fontweight="bold", y=0.97,
    )
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.88, wspace=0.02)
    stem = output / "01_upper_right_face_extension"
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
        add_pairs(axis, base_surface, base_outer, "#c4bdb9", "#b8c5bc", alpha=0.22)
        add_pairs(axis, added_surface, added_outer, "#ff7f0e", "#1f77b4")
        axis.scatter([0], [0], [0], marker="*", s=140, c="yellow",
                     edgecolors="black", linewidths=0.6, depthshade=False)
        configure_axis(axis, f"Azimuth {azimuth}°", azimuth)
    figure.suptitle(
        "Higher-Thigh / Higher-Hip Surface Continuation From Four Views",
        fontsize=19, fontweight="bold", y=0.97,
    )
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.92,
                           wspace=0.02, hspace=0.08)
    stem = output / "02_upper_right_face_extension_four_views"
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
            base_surface[:, x_index], base_surface[:, y_index], c="#c4bdb9",
            marker="^", s=25, alpha=0.35, edgecolors="none", label="Retained anchors",
        )
        for surface_point, outer_point in zip(added_surface, added_outer):
            path = np.vstack((np.zeros(3), outer_point, surface_point))
            axis.plot(path[:, x_index], path[:, y_index], color="#ff7f0e",
                      linewidth=0.5, alpha=0.24)
        axis.scatter(
            added_surface[:, x_index], added_surface[:, y_index], c="#ff7f0e",
            marker="^", s=58, edgecolors="#602800", linewidths=0.45,
            label="Added surface anchors",
        )
        axis.scatter(
            added_outer[:, x_index], added_outer[:, y_index], c="#1f77b4",
            marker="s", s=40, edgecolors="#08334f", linewidths=0.4,
            label="Added two-thirds companions",
        )
        axis.scatter([0], [0], marker="*", s=145, c="yellow", edgecolors="black",
                     linewidths=0.7, label="Nominal")
        axis.set(xlabel=x_label, ylabel=y_label)
        axis.grid(alpha=0.25)
        axis.set_aspect("equal", adjustable="box")
    axes[0].legend(loc="best", fontsize=8)
    figure.suptitle(
        "36 Added Anchors Continue Toward Increasing Thigh and Increasing Hip",
        fontsize=17, fontweight="bold",
    )
    figure.tight_layout()
    stem = output / "03_upper_right_face_extension_projections"
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
    outer_by_pair = {
        entry["surface_pair_id"]: np.asarray(entry["percent_from_nominal"], dtype=float)
        for entry in base_outer_entries
    }
    base_surface = np.asarray([
        entry["percent_from_nominal"] for entry in base_surface_entries
    ], dtype=float)
    base_outer = np.asarray([
        outer_by_pair[entry["surface_pair_id"]] for entry in base_surface_entries
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
    wedge_mask = (
        (hull_offset >= -0.07)
        & (hull_offset <= 0.07)
        & (radius >= 0.72)
        & (radius <= 1.45)
        & (prediction >= 0.55)
        & (candidates[:, 1] <= -6.25)
        & (candidates[:, 0] >= 25.0)
        & (candidates[:, 2] >= 25.0)
    )
    pool_indices = np.flatnonzero(wedge_mask)
    pool_points = candidate_points[wedge_mask]
    if len(pool_points) < ADDED_SURFACE_POINTS:
        raise RuntimeError(f"Only {len(pool_points)} wedge candidates available")
    selected_local, _ = farthest_indices(
        pool_points, normalized(base_surface), ADDED_SURFACE_POINTS
    )
    added_indices = pool_indices[np.asarray(selected_local)]
    added_surface_points = candidate_points[added_indices]

    unavailable = np.zeros(len(candidates), dtype=bool)
    unavailable[added_indices] = True
    added_outer_indices, outer_ratio, outer_cosine = assign_radial_pairs(
        added_surface_points, candidate_points, prediction, hull_offset,
        unavailable, nominal_point, OUTER_FRACTION, 0.50, 0.84,
    )
    added_surface = candidates[added_indices]
    added_outer = candidates[added_outer_indices]

    pair_start = max(entry["surface_pair_id"] for entry in base_entries) + 1
    added_entries = []
    for index in range(ADDED_SURFACE_POINTS):
        pair_id = pair_start + index
        for category, candidate_index, target, actual_ratio, cosine in (
            ("upper_right_surface_anchor", added_indices[index], 1.0,
             radius[added_indices[index]], 1.0),
            ("upper_right_outer_radial", added_outer_indices[index], OUTER_FRACTION,
             outer_ratio[index], outer_cosine[index]),
        ):
            levels = candidates[candidate_index]
            added_entries.append({
                "name": f"upper_right_{index + 1:03d}_{category}",
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
        "description": "Retained face plan extended toward increasing thigh and increasing hip",
        "base_plan": str(args.base_plan.resolve()),
        "added_upper_right_surface_anchor_count": ADDED_SURFACE_POINTS,
        "surface_anchor_count": 252,
        "points_per_anchor": 2,
        "morphologies": combined_entries,
    })
    (args.output / "upper_right_face_extension_plan_504.json").write_text(
        json.dumps(output_plan, indent=2) + "\n"
    )

    full = measured[completion == 1.0]
    before_volume, after_volume, increase = plot_comparison(
        full, base_surface, base_outer, added_surface, added_outer, args.output
    )
    plot_four_views(full, base_surface, base_outer, added_surface, added_outer, args.output)
    plot_projections(base_surface, added_surface, added_outer, args.output)
    max_angle = float(np.degrees(np.arccos(np.clip(outer_cosine, -1.0, 1.0))).max())
    lines = [
        "# Upper-Right Face Extension", "",
        "This is a preview only. No MPC episodes were run.", "",
        "## Distribution", "",
        "- Retained morphologies: 432",
        f"- Added higher-thigh/higher-hip anchors: {len(added_surface)}",
        f"- Added two-thirds companions: {len(added_outer)}",
        f"- Combined morphologies: {len(combined_entries)}",
        "- One-third companions: 0", "",
        "## Added Coverage", "",
        f"- Thigh range: `{added_surface[:, 0].min():.2f}%` to `{added_surface[:, 0].max():.2f}%`",
        f"- Shin range: `{added_surface[:, 1].min():.2f}%` to `{added_surface[:, 1].max():.2f}%`",
        f"- Hip range: `{added_surface[:, 2].min():.2f}%` to `{added_surface[:, 2].max():.2f}%`",
        f"- Maximum radial direction error: `{max_angle:.2f} degrees`", "",
        "## Hypothetical Effect", "",
        f"- Retained-plan normalized hull volume: `{before_volume:.6f}`",
        f"- Extended normalized hull volume: `{after_volume:.6f}`",
        f"- Additional hypothetical hull increase: `{increase:.2f}%`", "",
        "Added anchors use a 55% local predicted-completion floor because this deliberately",
        "pushes beyond the densely successful region. This is not a success guarantee.",
    ]
    (args.output / "README.md").write_text("\n".join(lines) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
