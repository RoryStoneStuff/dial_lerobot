"""Combine morphology benchmarks and plot color-coded completion volumes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import mujoco
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull


LEVELS = (0.25, 0.50, 0.75, 1.00)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--color-metric", choices=("cot", "running_cost"), default="cot")
    return parser.parse_args()


def episode_cot(model, episode_path, distance):
    with np.load(episode_path) as arrays:
        time = np.asarray(arrays["time"], dtype=float)
        if not len(time) or distance <= 1e-6:
            return None, 0.0
        timestep = (float(np.median(np.diff(time))) if len(time) > 1
                    else float(model.opt.timestep))
        actuator_dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
        torque = np.asarray(arrays["actuator_force"], dtype=float)
        velocity = np.asarray(arrays["qvel"][:, actuator_dofs], dtype=float)
        work = float(np.maximum(torque * velocity, 0.0).sum() * timestep)
    mass = float(model.body_mass.sum())
    return work / (mass * 9.81 * distance), work


def load_rows(benchmarks):
    documents = [json.loads(path.read_text()) for path in benchmarks]
    nominal = None
    for document in documents:
        for item in document["matrix"]:
            if item["morphology"] == "nominal":
                nominal = np.asarray(item["theta"], dtype=float)
                break
    if nominal is None:
        raise ValueError("At least one benchmark must contain the nominal morphology")

    rows = []
    seen = set()
    for benchmark_path, document in zip(benchmarks, documents):
        for item in document["matrix"]:
            condition = document["conditions"].get(item["name"])
            if not condition or condition.get("status") != "complete":
                raise ValueError(f"Incomplete condition in {benchmark_path}: {item['name']}")
            theta = np.asarray(item["theta"], dtype=float)
            coordinate = tuple(float(value) for value in np.round(
                100.0 * (theta[[0, 1, 3]] / nominal[[0, 1, 3]] - 1.0), 8
            ))
            if coordinate in seen:
                raise ValueError(f"Duplicate morphology coordinate: {coordinate}")
            seen.add(coordinate)
            folder = Path(condition["output"])
            model = mujoco.MjModel.from_xml_path(str(folder / "model.xml"))
            cots, work = [], []
            episode_paths = sorted(folder.glob("episode_*.npz"))
            if len(episode_paths) != len(condition["episodes"]):
                raise ValueError(f"Episode-file mismatch in {folder}")
            for summary, episode_path in zip(condition["episodes"], episode_paths):
                cot, positive_work = episode_cot(
                    model, episode_path, float(summary["distance_x_m"])
                )
                cots.append(cot)
                work.append(positive_work)
            finite_cots = [value for value in cots if value is not None]
            rows.append({
                "morphology": item["morphology"],
                "source_benchmark": str(benchmark_path.resolve()),
                "theta": theta.tolist(),
                "percent_from_nominal": list(coordinate),
                "completion_fraction": condition["objective"]["completion_fraction"],
                "survival_fraction": condition["objective"]["survival_fraction"],
                "mean_cost_per_second": condition["objective"]["mean_cost_per_second"],
                "mean_positive_mechanical_cot": (
                    float(np.mean(finite_cots)) if len(finite_cots) == len(cots) else None
                ),
                "episode_positive_mechanical_cot": cots,
                "episode_positive_mechanical_work_j": work,
                "mean_vx_m_s": float(np.mean([
                    episode["mean_vx_m_s"] for episode in condition["episodes"]
                ])),
                "episodes": len(condition["episodes"]),
                "output": str(folder.resolve()),
            })
    return nominal, rows


def metric_spec(color_metric):
    if color_metric == "running_cost":
        return "mean_cost_per_second", "Mean running cost per simulated second", "running_cost", "magma"
    return "mean_positive_mechanical_cot", "Mean positive mechanical cost of transport", "cot", "viridis"


def color_scale(rows, metric_key):
    values = np.asarray([
        row[metric_key] for row in rows if row[metric_key] is not None
    ])
    lower, upper = np.percentile(values, (5, 95))
    return colors.LogNorm(vmin=max(float(lower), 1e-4), vmax=float(upper))


def plot_volumes(rows, output, color_metric):
    metric_key, metric_label, metric_slug, cmap_name = metric_spec(color_metric)
    points = np.asarray([row["percent_from_nominal"] for row in rows])
    completion = np.asarray([row["completion_fraction"] for row in rows])
    metric = np.asarray([
        row[metric_key] if row[metric_key] is not None else np.nan
        for row in rows
    ])
    norm = color_scale(rows, metric_key)
    cmap = plt.get_cmap(cmap_name)
    fig = plt.figure(figsize=(32, 8.6))
    for panel, level in enumerate(LEVELS, start=1):
        axis = fig.add_subplot(1, 4, panel, projection="3d")
        selected = completion >= level - 1e-9
        volume_points = points[selected]
        volume_metric = metric[selected]
        hull = ConvexHull(volume_points)
        triangles = volume_points[hull.simplices]
        display_metric = np.where(np.isfinite(volume_metric), volume_metric, norm.vmax)
        face_metric = display_metric[hull.simplices].mean(axis=1)
        volume = Poly3DCollection(
            triangles,
            facecolors=cmap(norm(np.clip(face_metric, norm.vmin, norm.vmax))),
            edgecolors=(0.08, 0.15, 0.18, 0.45), linewidths=0.5, alpha=0.27,
        )
        axis.add_collection3d(volume)
        axis.scatter(points[~selected, 0], points[~selected, 1], points[~selected, 2],
                     c="0.72", s=10, alpha=0.16, edgecolors="none", depthshade=False)
        finite = selected & np.isfinite(metric)
        axis.scatter(points[finite, 0], points[finite, 1], points[finite, 2],
                     c=metric[finite], cmap=cmap, norm=norm, s=45,
                     edgecolors="0.12", linewidths=0.4, depthshade=False)
        undefined = selected & ~np.isfinite(metric)
        axis.scatter(points[undefined, 0], points[undefined, 1], points[undefined, 2],
                     c="red", marker="x", s=50, linewidths=1.0, depthshade=False)
        axis.scatter([0], [0], [0], marker="*", s=180, c="cyan",
                     edgecolors="black", linewidths=0.8, depthshade=False)
        axis.set(
            xlim=(-80, 105), ylim=(-80, 105), zlim=(-55, 105),
            xlabel="Thigh change (%)", ylabel="Shin change (%)", zlabel="Hip change (%)",
            title=f"≥{int(level * 100)}% completion volume\n{selected.sum()} measured morphologies",
        )
        axis.view_init(elev=24, azim=-58)
        axis.grid(alpha=0.35)
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    colorbar_axis = fig.add_axes((0.945, 0.16, 0.012, 0.67))
    colorbar = fig.colorbar(scalar_map, cax=colorbar_axis)
    colorbar.set_label(metric_label)
    fig.suptitle(
        f"Completion volumes across {len(rows)} measured morphologies, colored by {metric_label.lower()}\n"
        "Transparent hulls are convex envelopes, not unmeasured performance claims; cyan star is nominal",
        fontsize=18, fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(left=0.015, right=0.91, bottom=0.03, top=0.84, wspace=0.03)
    stem = output / f"01_completion_volumes_colored_by_{metric_slug}"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_scatter(rows, output, color_metric):
    metric_key, metric_label, metric_slug, cmap_name = metric_spec(color_metric)
    points = np.asarray([row["percent_from_nominal"] for row in rows])
    completion = np.asarray([row["completion_fraction"] for row in rows])
    metric = np.asarray([
        row[metric_key] if row[metric_key] is not None else np.nan for row in rows
    ], dtype=float)
    norm = color_scale(rows, metric_key)
    fig = plt.figure(figsize=(12, 9))
    axis = fig.add_subplot(111, projection="3d")
    finite = np.isfinite(metric)
    image = axis.scatter(points[finite, 0], points[finite, 1], points[finite, 2], c=metric[finite],
                         cmap=cmap_name, norm=norm, s=22 + 90 * completion[finite],
                         edgecolors="0.15", linewidths=0.45, depthshade=False)
    undefined = ~finite
    if undefined.any():
        axis.scatter(points[undefined, 0], points[undefined, 1], points[undefined, 2],
                     c="red", marker="x", s=35, linewidths=0.9, depthshade=False,
                     label=f"Undefined {metric_label}")
    axis.scatter([0], [0], [0], marker="*", s=220, c="cyan",
                 edgecolors="black", linewidths=0.9, depthshade=False)
    axis.set(xlabel="Thigh change (%)", ylabel="Shin change (%)", zlabel="Hip change (%)",
             title=f"All {len(rows)} measured morphologies\nMarker size indicates episode completion")
    axis.view_init(elev=24, azim=-58)
    if undefined.any():
        axis.legend(loc="upper left")
    fig.colorbar(image, ax=axis, fraction=0.025, pad=0.08,
                 label=metric_label)
    fig.tight_layout()
    stem = output / f"02_all_morphologies_{metric_slug}_scatter"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_data(nominal, rows, benchmarks, output, color_metric):
    metric_key, metric_label, _, _ = metric_spec(color_metric)
    payload = {
        "description": (f"Combined {len(rows)}-morphology benchmark with running cost "
                        "and recomputed positive mechanical CoT"),
        "sources": [str(path.resolve()) for path in benchmarks],
        "nominal_theta": nominal.tolist(),
        "morphologies": rows,
    }
    (output / "combined_morphology_dataset.json").write_text(json.dumps(payload, indent=2) + "\n")
    fields = ("morphology", "thigh_change_percent", "shin_change_percent",
              "hip_change_percent", "completion_fraction", "survival_fraction",
              "mean_positive_mechanical_cot", "mean_cost_per_second", "mean_vx_m_s")
    with (output / "combined_morphology_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(zip(fields, (
                row["morphology"], *row["percent_from_nominal"],
                row["completion_fraction"], row["survival_fraction"],
                row["mean_positive_mechanical_cot"], row["mean_cost_per_second"],
                row["mean_vx_m_s"],
            ))))

    counts = Counter(row["completion_fraction"] for row in rows)
    fully_complete = [row for row in rows if row["completion_fraction"] == 1.0]
    best = min(fully_complete, key=lambda row: row[metric_key])
    nominal_row = next(row for row in rows if row["morphology"] == "nominal")
    improvement = 100.0 * (1.0 - best[metric_key] / nominal_row[metric_key])
    lines = [
        f"# Combined {len(rows)}-Morphology Dataset", "",
        f"{len(benchmarks)} non-overlapping morphology benchmarks are combined here. Positive",
        "mechanical cost of transport was recomputed from saved actuator torque and joint velocity using",
        "`sum(max(tau*qdot, 0))*dt/(mass*g*forward_distance)`.", "",
        "## Completion", "",
        f"- 100% completion: {counts[1.0]}",
        f"- 75% completion: {counts[0.75]}",
        f"- 50% completion: {counts[0.5]}",
        f"- 25% completion: {counts[0.25]}",
        f"- 0% completion: {counts[0.0]}", "",
        f"## Best Fully Completing Design by {metric_label}", "",
        f"- Morphology: `{best['morphology']}`",
        f"- Changes [thigh, shin, hip]: `{best['percent_from_nominal']}`%",
        f"- {metric_label}: `{best[metric_key]:.6f}`",
        f"- Mean forward velocity: `{best['mean_vx_m_s']:.4f} m/s`", "",
        f"Nominal {metric_label.lower()} is `{nominal_row[metric_key]:.6f}`;",
        f"the best fully completing measured design is `{improvement:.2f}%` lower.", "",
        "The transparent volumes are convex envelopes of measured successful points. They do",
        "not claim that every unmeasured morphology inside an envelope will succeed.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    nominal, rows = load_rows(args.benchmark)
    expected = sum(len(json.loads(path.read_text())["matrix"]) for path in args.benchmark)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} combined morphologies, found {len(rows)}")
    write_data(nominal, rows, args.benchmark, args.output, args.color_metric)
    plot_volumes(rows, args.output, args.color_metric)
    plot_scatter(rows, args.output, args.color_metric)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
