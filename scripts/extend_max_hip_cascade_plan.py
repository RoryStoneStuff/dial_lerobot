"""Extend a retained morphology plan from the maximum-hip cap down all outer faces."""

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


TOP_CAP_POINTS = 48
DESCENDING_HIP_LEVELS = (87.5, 75.0, 62.5, 50.0, 37.5, 25.0, 12.5, 0.0, -12.5, -25.0, -37.5, -50.0)
POINTS_PER_DESCENDING_LEVEL = 8
OUTER_FRACTION = 2.0 / 3.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def split_base_pairs(entries):
    surface_entries = [entry for entry in entries if "surface_anchor" in entry["category"]]
    outer_entries = [entry for entry in entries if "outer_radial" in entry["category"]]
    outer_by_pair = {
        entry["surface_pair_id"]: np.asarray(entry["percent_from_nominal"], dtype=float)
        for entry in outer_entries
    }
    surface = np.asarray([entry["percent_from_nominal"] for entry in surface_entries], dtype=float)
    outer = np.asarray([outer_by_pair[entry["surface_pair_id"]] for entry in surface_entries])
    return surface, outer


def choose_farthest(candidate_points, mask, references, count):
    pool_indices = np.flatnonzero(mask)
    if len(pool_indices) < count:
        raise RuntimeError(f"Only {len(pool_indices)} candidates available for {count} selections")
    selected_local, _ = farthest_indices(candidate_points[mask], references, count)
    return pool_indices[np.asarray(selected_local)]


def add_colored_pairs(axis, surface, outer, label=None):
    colors = plt.colormaps["plasma"]((surface[:, 2] - LOWER[2]) / (UPPER[2] - LOWER[2]))
    for surface_point, outer_point, color in zip(surface, outer, colors):
        path = np.vstack((np.zeros(3), outer_point, surface_point))
        axis.plot(path[:, 0], path[:, 1], path[:, 2], color=color, linewidth=0.55, alpha=0.30)
    axis.scatter(
        surface[:, 0], surface[:, 1], surface[:, 2], c=surface[:, 2], cmap="plasma",
        vmin=LOWER[2], vmax=UPPER[2], marker="^", s=57, edgecolors="#251333",
        linewidths=0.4, depthshade=False, label=label,
    )
    axis.scatter(
        outer[:, 0], outer[:, 1], outer[:, 2], c=surface[:, 2], cmap="plasma",
        vmin=LOWER[2], vmax=UPPER[2], marker="s", s=34, edgecolors="#251333",
        linewidths=0.3, depthshade=False,
    )


def plot_comparison(full, base_surface, base_outer, added_surface, added_outer, output):
    before = np.vstack((full, base_surface))
    after = np.vstack((before, added_surface))
    before_volume = ConvexHull(normalized(before)).volume
    after_volume = ConvexHull(normalized(after)).volume
    increase = 100.0 * (after_volume / before_volume - 1.0)

    figure = plt.figure(figsize=(19, 8.5))
    before_axis = figure.add_subplot(1, 2, 1, projection="3d")
    before_axis.add_collection3d(hull_collection(before, "#5b9c74", "#174f2a", 0.12))
    add_pairs(before_axis, base_surface, base_outer, "#b7a9a4", "#aabbb0", alpha=0.42)
    before_axis.scatter([0], [0], [0], marker="*", s=165, c="yellow", edgecolors="black")
    configure_axis(before_axis, "Retained 432-point plan")

    after_axis = figure.add_subplot(1, 2, 2, projection="3d")
    after_axis.add_collection3d(hull_collection(after, "#5b9c74", "#174f2a", 0.10))
    add_pairs(after_axis, base_surface, base_outer, "#c4bdb9", "#b8c5bc", alpha=0.20)
    add_colored_pairs(after_axis, added_surface, added_outer, "Added max-hip cascade")
    after_axis.scatter([0], [0], [0], marker="*", s=165, c="yellow", edgecolors="black", label="Nominal")
    configure_axis(after_axis, f"Maximum-hip cap cascading down all faces\nHull growth: {increase:.1f}%")
    after_axis.legend(loc="upper left", fontsize=8)
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(LOWER[2], UPPER[2]), cmap="plasma"),
        ax=after_axis, fraction=0.035, pad=0.08,
    )
    colorbar.set_label("Hip change of surface anchor (%)")
    figure.suptitle("Dense Coverage From the Highest Hip Extreme Down the Outer Faces", fontsize=19, fontweight="bold")
    figure.subplots_adjust(left=0.02, right=0.95, bottom=0.03, top=0.90, wspace=0.02)
    stem = output / "01_max_hip_cascade"
    figure.savefig(stem.with_suffix(".png"), dpi=190, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return before_volume, after_volume, increase


def plot_views(full, base_surface, base_outer, added_surface, added_outer, output):
    after = np.vstack((full, base_surface, added_surface))
    figure = plt.figure(figsize=(17, 15))
    for panel, azimuth in enumerate((-55, 35, 125, 215), start=1):
        axis = figure.add_subplot(2, 2, panel, projection="3d")
        axis.add_collection3d(hull_collection(after, "#5b9c74", "#174f2a", 0.07))
        add_pairs(axis, base_surface, base_outer, "#c4bdb9", "#b8c5bc", alpha=0.15)
        add_colored_pairs(axis, added_surface, added_outer)
        axis.scatter([0], [0], [0], marker="*", s=140, c="yellow", edgecolors="black")
        configure_axis(axis, f"Azimuth {azimuth}°", azimuth)
    figure.suptitle("Maximum-Hip Cap and Descending Outer-Face Bands", fontsize=19, fontweight="bold")
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.92, wspace=0.02, hspace=0.08)
    stem = output / "02_max_hip_cascade_four_views"
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
        axis.scatter(base_surface[:, x_index], base_surface[:, y_index], c="#c4bdb9", s=20, alpha=0.25)
        for surface_point, outer_point in zip(added_surface, added_outer):
            path = np.vstack((np.zeros(3), outer_point, surface_point))
            axis.plot(path[:, x_index], path[:, y_index], color="#70438f", linewidth=0.4, alpha=0.17)
        points = axis.scatter(
            added_surface[:, x_index], added_surface[:, y_index], c=added_surface[:, 2],
            cmap="plasma", vmin=LOWER[2], vmax=UPPER[2], marker="^", s=48,
            edgecolors="#251333", linewidths=0.35,
        )
        axis.scatter(added_outer[:, x_index], added_outer[:, y_index], c=added_surface[:, 2], cmap="plasma", vmin=LOWER[2], vmax=UPPER[2], marker="s", s=27)
        axis.scatter([0], [0], marker="*", s=135, c="yellow", edgecolors="black")
        axis.set(xlabel=x_label, ylabel=y_label)
        axis.grid(alpha=0.25)
        axis.set_aspect("equal", adjustable="box")
    colorbar = figure.colorbar(points, ax=axes, fraction=0.018, pad=0.025)
    colorbar.set_label("Hip change of surface anchor (%)")
    figure.suptitle("Top-Cap Coverage and Descent Across Every Outer Face", fontsize=17, fontweight="bold")
    figure.subplots_adjust(left=0.05, right=0.91, bottom=0.11, top=0.88, wspace=0.24)
    stem = output / "03_max_hip_cascade_projections"
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
    base_surface, base_outer = split_base_pairs(base_entries)
    used = {tuple(np.round(point, 8)) for point in measured}
    used.update(tuple(np.round(entry["percent_from_nominal"], 8)) for entry in base_entries)
    candidates = np.asarray([
        point for point in product(
            np.arange(LOWER[0], UPPER[0] + GRID_STEP / 2.0, GRID_STEP),
            np.arange(LOWER[1], UPPER[1] + GRID_STEP / 2.0, GRID_STEP),
            np.arange(LOWER[2], UPPER[2] + GRID_STEP / 2.0, GRID_STEP),
        ) if tuple(np.round(point, 8)) not in used
    ], dtype=float)
    candidate_points = normalized(candidates)
    prediction, _ = predict_completion(candidate_points, measured_points, completion)
    hull_offset = (candidate_points @ hull.equations[:, :3].T + hull.equations[:, 3]).max(axis=1)
    radius, _ = radial_fraction(candidate_points, hull, nominal_point)
    shell = (hull_offset >= -0.10) & (hull_offset <= 0.12) & (radius >= 0.65) & (radius <= 1.55)

    references = normalized(base_surface)
    top_mask = shell & (prediction >= 0.40) & np.isclose(candidates[:, 2], UPPER[2])
    top_indices = choose_farthest(candidate_points, top_mask, references, TOP_CAP_POINTS)
    selected_indices = list(top_indices)
    references = np.vstack((references, candidate_points[top_indices]))
    for hip_level in DESCENDING_HIP_LEVELS:
        level_mask = shell & (prediction >= 0.45) & np.isclose(candidates[:, 2], hip_level)
        level_indices = choose_farthest(
            candidate_points, level_mask, references, POINTS_PER_DESCENDING_LEVEL
        )
        selected_indices.extend(level_indices)
        references = np.vstack((references, candidate_points[level_indices]))
    added_indices = np.asarray(selected_indices, dtype=int)
    added_surface_points = candidate_points[added_indices]

    unavailable = np.zeros(len(candidates), dtype=bool)
    unavailable[added_indices] = True
    added_outer_indices, outer_ratio, outer_cosine = assign_radial_pairs(
        added_surface_points, candidate_points, prediction, hull_offset,
        unavailable, nominal_point, OUTER_FRACTION, 0.48, 0.86,
    )
    added_surface = candidates[added_indices]
    added_outer = candidates[added_outer_indices]

    pair_start = max(entry["surface_pair_id"] for entry in base_entries) + 1
    added_entries = []
    for index, surface_index in enumerate(added_indices):
        pair_id = pair_start + index
        surface_category = "max_hip_surface_anchor" if np.isclose(candidates[surface_index, 2], UPPER[2]) else "descending_face_surface_anchor"
        for category, candidate_index, target, actual_ratio, cosine in (
            (surface_category, surface_index, 1.0, radius[surface_index], 1.0),
            (surface_category.replace("surface_anchor", "outer_radial"), added_outer_indices[index], OUTER_FRACTION, outer_ratio[index], outer_cosine[index]),
        ):
            levels = candidates[candidate_index]
            added_entries.append({
                "name": f"max_hip_cascade_{index + 1:03d}_{category}",
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
        "description": "Retained plan extended across the maximum-hip cap and down every outer face",
        "base_plan": str(args.base_plan.resolve()),
        "added_max_hip_surface_anchor_count": TOP_CAP_POINTS,
        "added_descending_face_surface_anchor_count": len(DESCENDING_HIP_LEVELS) * POINTS_PER_DESCENDING_LEVEL,
        "surface_anchor_count": len(combined_entries) // 2,
        "points_per_anchor": 2,
        "morphologies": combined_entries,
    })
    plan_path = args.output / "max_hip_cascade_plan_720.json"
    plan_path.write_text(json.dumps(output_plan, indent=2) + "\n")

    full = measured[completion == 1.0]
    before_volume, after_volume, increase = plot_comparison(
        full, base_surface, base_outer, added_surface, added_outer, args.output
    )
    plot_views(full, base_surface, base_outer, added_surface, added_outer, args.output)
    plot_projections(base_surface, added_surface, added_outer, args.output)
    max_angle = float(np.degrees(np.arccos(np.clip(outer_cosine, -1.0, 1.0))).max())
    lines = [
        "# Maximum-Hip Cascade Extension", "",
        "This is a preview only. No MPC episodes were run.", "",
        "## Distribution", "",
        f"- Retained morphologies: {len(base_entries)}",
        f"- Added anchors on the hip +100% cap: {TOP_CAP_POINTS}",
        f"- Added anchors descending over all outer faces: {len(DESCENDING_HIP_LEVELS) * POINTS_PER_DESCENDING_LEVEL}",
        f"- Added two-thirds companions: {len(added_outer)}",
        f"- Combined morphologies: {len(combined_entries)}",
        "- One-third companions: 0", "",
        "## Descending Bands", "",
        f"- Hip levels: `{', '.join(f'{level:g}%' for level in DESCENDING_HIP_LEVELS)}`",
        f"- Points per descending level: {POINTS_PER_DESCENDING_LEVEL}",
        f"- Added thigh range: `{added_surface[:, 0].min():.2f}%` to `{added_surface[:, 0].max():.2f}%`",
        f"- Added shin range: `{added_surface[:, 1].min():.2f}%` to `{added_surface[:, 1].max():.2f}%`",
        f"- Maximum radial direction error: `{max_angle:.2f} degrees`", "",
        "## Hypothetical Effect", "",
        f"- Retained-plan normalized hull volume: `{before_volume:.6f}`",
        f"- Extended normalized hull volume: `{after_volume:.6f}`",
        f"- Additional hypothetical hull increase: `{increase:.2f}%`", "",
        "The highest hip face is sampled broadly first. Additional bands then descend from",
        "that cap across the complete outer shell rather than targeting a single corner.",
    ]
    (args.output / "README.md").write_text("\n".join(lines) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
