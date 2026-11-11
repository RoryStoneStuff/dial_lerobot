"""Generate an all-around hull and radial-intermediate morphology preview."""

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
    greedy_select,
    greedy_surface_select,
    hull_collection,
    normalized,
    predict_completion,
    radial_fraction,
    theta_from_percent,
)


SECTORS = (
    (0, 1, "+thigh"),
    (0, -1, "-thigh"),
    (1, 1, "+shin"),
    (1, -1, "-shin"),
    (2, 1, "+hip"),
    (2, -1, "-hip"),
)
SURFACE_PER_SECTOR = 6
OUTER_INTERMEDIATE_PER_SECTOR = 4
INNER_INTERMEDIATE_PER_SECTOR = 4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def direction_sectors(points, origin):
    offsets = np.asarray(points, dtype=float) - origin
    directions = offsets / np.maximum(
        np.linalg.norm(offsets, axis=1)[:, None], 1e-12
    )
    dominant_axes = np.argmax(np.abs(directions), axis=1)
    dominant_signs = np.where(
        directions[np.arange(len(directions)), dominant_axes] >= 0.0, 1, -1
    )
    return dominant_axes, dominant_signs


def select_layer(candidate_points, prediction, radius, mask, references, count,
                 target_radius):
    pool = candidate_points[mask]
    pool_prediction = prediction[mask]
    pool_radius = radius[mask]
    if len(pool) < count:
        raise RuntimeError(f"Only {len(pool)} candidates available for {count} slots")
    return greedy_select(
        pool,
        references,
        count,
        (
            (0.25, pool_prediction),
            (0.20, -np.abs(pool_radius - target_radius)),
        ),
    )


def configure_axis(axis, title):
    axis.set(
        xlim=(-80, 105), ylim=(-80, 105), zlim=(-55, 105),
        xlabel="Thigh change (%)", ylabel="Shin change (%)", zlabel="Hip change (%)",
        title=title,
    )
    axis.view_init(elev=24, azim=-58)
    axis.grid(alpha=0.35)


def plot_volume(measured, completion, surface, outer, inner, output):
    current = measured[completion == 1.0]
    proposed = np.vstack((surface, outer, inner))
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
    configure_axis(
        current_axis,
        f"Current measured 4/4 volume\n{len(current)} successful morphologies",
    )

    preview_axis = figure.add_subplot(1, 2, 2, projection="3d")
    preview_axis.add_collection3d(hull_collection(
        hypothetical, "#48a56a", "#174f2a", 0.20,
    ))
    preview_axis.scatter(
        current[:, 0], current[:, 1], current[:, 2], c="#817b89", s=15,
        alpha=0.45, edgecolors="none", depthshade=False, label="Existing 4/4",
    )
    preview_axis.scatter(
        surface[:, 0], surface[:, 1], surface[:, 2], c="#e34a33", marker="^",
        s=68, edgecolors="#5c160d", linewidths=0.6, depthshade=False,
        label=f"Surface extremes ({len(surface)})",
    )
    preview_axis.scatter(
        outer[:, 0], outer[:, 1], outer[:, 2], c="#35a853", marker="s",
        s=55, edgecolors="#10491e", linewidths=0.6, depthshade=False,
        label=f"Outer intermediates ({len(outer)})",
    )
    preview_axis.scatter(
        inner[:, 0], inner[:, 1], inner[:, 2], c="#2b8cbe", marker="D",
        s=48, edgecolors="#0a3d58", linewidths=0.6, depthshade=False,
        label=f"Inner intermediates ({len(inner)})",
    )
    preview_axis.scatter(
        [0], [0], [0], marker="*", s=190, c="yellow", edgecolors="black",
        linewidths=0.8, depthshade=False, label="Nominal",
    )
    configure_axis(
        preview_axis,
        f"Hypothetical volume if all 84 proposals complete 4/4\n"
        f"Convex-hull volume increase: {increase:.1f}%",
    )
    preview_axis.legend(loc="upper left", fontsize=9)
    figure.suptitle(
        "All-Around Surface and Radial-Intermediate Follow-Up",
        fontsize=19, fontweight="bold", y=0.97,
    )
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.88, wspace=0.02)
    stem = output / "01_hypothetical_all_around_4of4_volume"
    figure.savefig(stem.with_suffix(".png"), dpi=190, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return current_volume, hypothetical_volume, increase


def plot_projections(measured, completion, surface, outer, inner, output):
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
            s=18, alpha=0.4, linewidths=0.7, label="Existing below 4/4",
        )
        axis.scatter(
            full[:, x_index], full[:, y_index], c="#817b89", s=20,
            alpha=0.55, edgecolors="none", label="Existing 4/4",
        )
        axis.scatter(
            surface[:, x_index], surface[:, y_index], c="#e34a33", marker="^",
            s=62, edgecolors="#5c160d", linewidths=0.5, label="Surface extremes",
        )
        axis.scatter(
            outer[:, x_index], outer[:, y_index], c="#35a853", marker="s",
            s=50, edgecolors="#10491e", linewidths=0.5,
            label="Outer intermediates",
        )
        axis.scatter(
            inner[:, x_index], inner[:, y_index], c="#2b8cbe", marker="D",
            s=43, edgecolors="#0a3d58", linewidths=0.5,
            label="Inner intermediates",
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
        "84-Point Distribution: Equal Coverage of Six Directional Extremes\n"
        "Each direction has 6 surface, 4 outer-intermediate, and 4 inner-intermediate points",
        fontsize=17, fontweight="bold",
    )
    figure.tight_layout()
    stem = output / "02_all_around_point_distribution"
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
    nearest_success = np.linalg.norm(
        candidate_points[:, None, :] - full_points[None, :, :], axis=2
    ).min(axis=1)
    nearest_nonfull = np.linalg.norm(
        candidate_points[:, None, :] - nonfull_points[None, :, :], axis=2
    ).min(axis=1)
    hull_offset = (
        candidate_points @ hull.equations[:, :3].T + hull.equations[:, 3]
    ).max(axis=1)
    radius, _ = radial_fraction(candidate_points, hull, nominal_point)
    dominant_axes, dominant_signs = direction_sectors(candidate_points, nominal_point)

    surface_groups = []
    outer_groups = []
    inner_groups = []
    records = []
    references = measured_points.copy()
    for axis_index, sign, sector_name in SECTORS:
        sector = (dominant_axes == axis_index) & (dominant_signs == sign)
        surface_mask = (
            sector
            & (hull_offset >= -0.04)
            & (hull_offset <= 0.04)
            & (radius >= 0.82)
            & (radius <= 1.25)
            & (prediction >= 0.90)
            & (nearest_success < nearest_nonfull)
        )
        surface_pool = candidate_points[surface_mask]
        surface_prediction = prediction[surface_mask]
        surface_priority = sign * surface_pool[:, axis_index]
        extreme_index = int(np.argmax(surface_priority))
        extreme_point = surface_pool[extreme_index:extreme_index + 1]
        remaining = np.arange(len(surface_pool)) != extreme_index
        spread_points = greedy_surface_select(
            surface_pool[remaining],
            np.vstack((references, extreme_point)),
            surface_prediction[remaining],
            surface_priority[remaining],
            nominal_point,
            SURFACE_PER_SECTOR - 1,
        )
        surface_points = np.vstack((extreme_point, spread_points))
        surface_groups.append(surface_points)
        references = np.vstack((references, surface_points))

        outer_mask = (
            sector
            & (radius >= 0.55)
            & (radius < 0.82)
            & (hull_offset <= 0.0)
            & (prediction >= 0.95)
            & (nearest_success < nearest_nonfull)
        )
        outer_points = select_layer(
            candidate_points, prediction, radius, outer_mask, references,
            OUTER_INTERMEDIATE_PER_SECTOR, 0.69,
        )
        outer_groups.append(outer_points)
        references = np.vstack((references, outer_points))

        inner_mask = (
            sector
            & (radius >= 0.20)
            & (radius < 0.55)
            & (hull_offset <= 0.0)
            & (prediction >= 0.925)
            & (nearest_success < nearest_nonfull)
        )
        inner_points = select_layer(
            candidate_points, prediction, radius, inner_mask, references,
            INNER_INTERMEDIATE_PER_SECTOR, 0.40,
        )
        inner_groups.append(inner_points)
        references = np.vstack((references, inner_points))

        for layer, points in (
            ("surface_extreme", surface_points),
            ("outer_intermediate", outer_points),
            ("inner_intermediate", inner_points),
        ):
            for point in points:
                records.append((layer, sector_name, point))

    surface_points = np.vstack(surface_groups)
    outer_points = np.vstack(outer_groups)
    inner_points = np.vstack(inner_groups)
    surface = LOWER + surface_points * (UPPER - LOWER)
    outer = LOWER + outer_points * (UPPER - LOWER)
    inner = LOWER + inner_points * (UPPER - LOWER)

    selected_points = np.asarray([record[2] for record in records])
    selected = LOWER + selected_points * (UPPER - LOWER)
    selected_prediction, _ = predict_completion(
        selected_points, measured_points, completion
    )
    selected_success_distance = np.linalg.norm(
        selected_points[:, None, :] - full_points[None, :, :], axis=2
    ).min(axis=1)
    selected_nonfull_distance = np.linalg.norm(
        selected_points[:, None, :] - nonfull_points[None, :, :], axis=2
    ).min(axis=1)
    selected_hull_offset = (
        selected_points @ hull.equations[:, :3].T + hull.equations[:, 3]
    ).max(axis=1)
    selected_radius, _ = radial_fraction(selected_points, hull, nominal_point)

    entries = []
    for index, ((layer, sector, _), levels) in enumerate(zip(records, selected)):
        entries.append({
            "name": f"all_around_{index + 1:03d}",
            "theta": theta_from_percent(nominal, levels),
            "percent_from_nominal": levels.tolist(),
            "category": layer,
            "directional_sector": sector,
            "predicted_completion": float(selected_prediction[index]),
            "nominal_to_hull_radius_fraction": float(selected_radius[index]),
            "normalized_distance_to_nearest_full_success": float(
                selected_success_distance[index]
            ),
            "normalized_distance_to_nearest_nonfull": float(
                selected_nonfull_distance[index]
            ),
            "signed_normalized_hull_offset": float(selected_hull_offset[index]),
        })

    args.output.mkdir(parents=True, exist_ok=True)
    plan = {
        "description": "Equal all-around surface and radial-intermediate coverage",
        "selection": (
            "Six dominant directional sectors (+/- thigh, +/- shin, +/- hip); each has "
            "6 surface-extreme, 4 outer-intermediate, and 4 inner-intermediate points"
        ),
        "source_dataset": str(args.dataset.resolve()),
        "nominal_theta": nominal.tolist(),
        "grid_step_percent": GRID_STEP,
        "preserve_link_masses": True,
        "hypothetical_completion_assumption": "All proposed points complete all four episodes",
        "morphologies": entries,
    }
    (args.output / "all_around_followup_plan_84.json").write_text(
        json.dumps(plan, indent=2) + "\n"
    )

    with (args.output / "all_around_followup_points.csv").open("w", newline="") as stream:
        fields = (
            "name", "category", "directional_sector", "thigh_change_percent",
            "shin_change_percent", "hip_change_percent", "predicted_completion",
            "nominal_to_hull_radius_fraction", "signed_normalized_hull_offset",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            writer.writerow(dict(zip(fields, (
                entry["name"], entry["category"], entry["directional_sector"],
                *entry["percent_from_nominal"], entry["predicted_completion"],
                entry["nominal_to_hull_radius_fraction"],
                entry["signed_normalized_hull_offset"],
            ))))

    current_volume, hypothetical_volume, increase = plot_volume(
        measured, completion, surface, outer, inner, args.output
    )
    plot_projections(measured, completion, surface, outer, inner, args.output)
    lines = [
        "# All-Around 4/4 Volume Follow-Up", "",
        "This is a preview only. No MPC episodes were run; all proposed points are shown",
        "as hypothetical 4/4 successes.", "",
        "## Distribution", "",
        "Each of `+thigh`, `-thigh`, `+shin`, `-shin`, `+hip`, and `-hip` receives:", "",
        f"- Surface-extreme points: {SURFACE_PER_SECTOR}",
        f"- Outer-intermediate points: {OUTER_INTERMEDIATE_PER_SECTOR}",
        f"- Inner-intermediate points: {INNER_INTERMEDIATE_PER_SECTOR}",
        f"- Total per direction: {SURFACE_PER_SECTOR + OUTER_INTERMEDIATE_PER_SECTOR + INNER_INTERMEDIATE_PER_SECTOR}",
        f"- Overall total: {len(entries)}", "",
        "## Hypothetical Effect", "",
        f"- Current normalized hull volume: `{current_volume:.6f}`",
        f"- Hypothetical normalized hull volume: `{hypothetical_volume:.6f}`",
        f"- Hypothetical hull increase: `{increase:.2f}%`", "",
        "Surface points use a 90% local predicted-completion threshold, outer intermediates",
        "use 95%, and inner intermediates use 92.5%. These predictions guide sampling and",
        "do not guarantee successful completion.",
    ]
    (args.output / "README.md").write_text("\n".join(lines) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
