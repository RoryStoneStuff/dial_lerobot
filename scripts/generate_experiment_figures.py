"""Generate publication-style figures from morphology benchmarks and searches."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import mujoco
import numpy as np


MORPHOLOGY_ORDER = (
    "nominal",
    "winner",
    "short_legs",
    "long_legs",
    "narrow_hips",
    "wide_hips",
    "short_thigh",
    "long_thigh",
    "short_shin",
    "long_shin",
    "extra_wide_hips",
)
MORPHOLOGY_LABELS = {
    "nominal": "Nominal",
    "winner": "Optimized winner",
    "short_legs": "Short legs",
    "long_legs": "Long legs",
    "narrow_hips": "Narrow hips",
    "wide_hips": "Wide hips",
    "short_thigh": "Short thigh only",
    "long_thigh": "Long thigh only",
    "short_shin": "Short shin only",
    "long_shin": "Long shin only",
    "extra_wide_hips": "Extra-wide hips",
}
COLORS = {
    "nominal": "#4C78A8",
    "winner": "#F58518",
    "short_legs": "#54A24B",
    "long_legs": "#E45756",
    "narrow_hips": "#B279A2",
    "wide_hips": "#FF9DA6",
    "short_thigh": "#9D755D",
    "long_thigh": "#BAB0AC",
    "short_shin": "#72B7B2",
    "long_shin": "#B6992D",
    "extra_wide_hips": "#D37295",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--optimization", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path):
    return json.loads(path.read_text())


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def condition_rows(benchmark):
    rows = []
    for key, condition in benchmark["conditions"].items():
        morphology = key.split("/")[-1]
        speed = float(condition["speed_m_s"])
        episodes = condition["episodes"]
        folder = Path(condition["output"])
        squared_errors = []
        for index in range(len(episodes)):
            arrays = np.load(folder / f"episode_{index:03d}.npz")
            squared_errors.extend(
                ((arrays["body_velocity"][:, 0] - arrays["target_velocity"][:, 0]) ** 2).tolist()
            )
        objective = condition["objective"]
        rows.append({
            "key": key,
            "morphology": morphology,
            "speed": speed,
            "completed": sum(not episode["fell"] for episode in episodes),
            "episodes": len(episodes),
            "completion_fraction": objective["completion_fraction"],
            "survival_fraction": objective["survival_fraction"],
            "cost_per_second": objective["mean_cost_per_second"],
            "mean_vx": float(np.mean([episode["mean_vx_m_s"] for episode in episodes])),
            "tracking_rmse": float(np.sqrt(np.mean(squared_errors))),
            "mean_survival_seconds": float(np.mean([episode["sim_seconds"] for episode in episodes])),
            "survival_seconds": [episode["sim_seconds"] for episode in episodes],
            "folder": folder,
        })
    return rows


def plot_overview(rows, output):
    morphology_order = [name for name in MORPHOLOGY_ORDER
                        if any(row["morphology"] == name for row in rows)]
    speeds = sorted({row["speed"] for row in rows})
    row_map = {(row["morphology"], row["speed"]): row for row in rows}
    metrics = (
        ("completion_fraction", "Completion rate", "YlGn", 0.0, 1.0, lambda value: f"{value:.0%}"),
        ("survival_fraction", "Mean survival fraction", "YlGn", 0.0, 1.0, lambda value: f"{value:.0%}"),
        ("tracking_rmse", "Velocity tracking RMSE (m/s)", "YlOrRd", 0.0, None, lambda value: f"{value:.2f}"),
        ("cost_per_second", "Running cost per second", "YlOrRd", 0.0, None, lambda value: f"{value:.2f}"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, max(8.5, len(morphology_order) * 0.72)),
                             constrained_layout=True)
    for axis, (field, title, cmap, minimum, maximum, formatter) in zip(axes.flat, metrics):
        values = np.asarray([
            [row_map[(morphology, speed)][field] for speed in speeds]
            for morphology in morphology_order
        ])
        image = axis.imshow(values, cmap=cmap, vmin=minimum, vmax=maximum, aspect="auto")
        axis.set_title(title, fontweight="bold")
        axis.set_xticks(range(len(speeds)), [f"{speed:.2f}" for speed in speeds])
        axis.set_yticks(range(len(morphology_order)),
                        [MORPHOLOGY_LABELS[name] for name in morphology_order])
        axis.set_xlabel("Commanded forward speed (m/s)")
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                axis.text(column_index, row_index, formatter(value), ha="center", va="center", fontsize=9)
        fig.colorbar(image, ax=axis, shrink=0.82)
    fig.suptitle("DIAL MPC performance across commands and morphologies", fontsize=15, fontweight="bold")
    save_figure(fig, output, "01_command_morphology_overview")


def plot_survival(rows, output):
    morphology_order = [name for name in MORPHOLOGY_ORDER
                        if any(row["morphology"] == name for row in rows)]
    speeds = sorted({row["speed"] for row in rows})
    row_map = {(row["morphology"], row["speed"]): row for row in rows}
    fig, axis = plt.subplots(figsize=(15, 6.2), constrained_layout=True)
    group_width = 0.74
    box_width = group_width / len(speeds)
    legend_handles = []
    for speed_index, speed in enumerate(speeds):
        positions = np.arange(len(morphology_order)) - group_width / 2 + box_width * (speed_index + 0.5)
        values = [row_map[(name, speed)]["survival_seconds"] for name in morphology_order]
        boxes = axis.boxplot(values, positions=positions, widths=box_width * 0.8,
                             patch_artist=True, showmeans=True, manage_ticks=False,
                             meanprops={"marker": "o", "markerfacecolor": "white",
                                        "markeredgecolor": "black", "markersize": 4})
        color = plt.cm.viridis(speed_index / max(len(speeds) - 1, 1))
        for patch in boxes["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        for element in ("whiskers", "caps", "medians"):
            for item in boxes[element]:
                item.set_color("#333333")
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.8,
                                            label=f"{speed:.2f} m/s"))
    axis.axhline(20.0, color="#333333", linestyle="--", linewidth=1, label="Full episode")
    axis.set_xticks(range(len(morphology_order)),
                    [MORPHOLOGY_LABELS[name] for name in morphology_order], rotation=24, ha="right")
    axis.set_ylabel("Survival time (s)")
    axis.set_ylim(0, 21)
    episode_counts = sorted({row["episodes"] for row in rows})
    count_label = str(episode_counts[0]) if len(episode_counts) == 1 else "multiple"
    axis.set_title(f"Episode survival across {count_label} held-out seeds", fontweight="bold")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(handles=legend_handles, title="Command", ncol=3, loc="lower left")
    save_figure(fig, output, "02_survival_distributions")


def plot_trajectories(rows, output, speed=0.85):
    selected = {row["morphology"]: row for row in rows if np.isclose(row["speed"], speed)}
    morphology_order = [name for name in MORPHOLOGY_ORDER if name in selected]
    columns = 3
    rows_count = math.ceil(len(morphology_order) / columns)
    fig, axes = plt.subplots(rows_count, columns, figsize=(14, 3.1 * rows_count),
                             sharex=True, sharey=True, constrained_layout=True, squeeze=False)
    for axis, morphology in zip(axes.flat, morphology_order):
        row = selected[morphology]
        for episode_path in sorted(row["folder"].glob("episode_*.npz")):
            arrays = np.load(episode_path)
            axis.plot(arrays["time"], arrays["body_velocity"][:, 0],
                      color=COLORS[morphology], alpha=0.34, linewidth=0.9)
        first = np.load(sorted(row["folder"].glob("episode_*.npz"))[0])
        axis.plot(first["time"], first["target_velocity"][:, 0], color="black",
                  linestyle="--", linewidth=1.5, label="Command")
        axis.axhline(0, color="#888888", linewidth=0.6)
        axis.set_title(f"{MORPHOLOGY_LABELS[morphology]} "
                       f"({row['completed']}/{row['episodes']} complete)",
                       fontweight="bold")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(morphology_order):]:
        axis.set_visible(False)
    for axis in axes.flat:
        if axis.get_visible():
            axis.set_xlabel("Simulation time (s)")
            axis.set_ylabel("Forward velocity (m/s)")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle(f"All held-out velocity trajectories at {speed:.2f} m/s command",
                 fontsize=15, fontweight="bold")
    save_figure(fig, output, "03_velocity_trajectories_vx085")


def plot_combined_trajectories(rows, output, speed=0.85):
    selected = {
        row["morphology"]: row for row in rows
        if np.isclose(row["speed"], speed) and row["completed"] == row["episodes"]
    }
    morphology_order = [name for name in MORPHOLOGY_ORDER if name in selected]
    fig, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
    command_plotted = False
    for morphology in morphology_order:
        row = selected[morphology]
        episodes = [np.load(path) for path in sorted(row["folder"].glob("episode_*.npz"))]
        sample_count = min(len(arrays["time"]) for arrays in episodes)
        time = episodes[0]["time"][:sample_count]
        velocities = np.stack([
            arrays["body_velocity"][:sample_count, 0] for arrays in episodes
        ])
        mean_velocity = np.mean(velocities, axis=0)
        standard_deviation = np.std(velocities, axis=0)
        color = COLORS[morphology]
        axis.plot(time, mean_velocity, color=color, linewidth=1.6,
                  label=MORPHOLOGY_LABELS[morphology])
        axis.fill_between(time, mean_velocity - standard_deviation,
                          mean_velocity + standard_deviation, color=color, alpha=0.08)
        if not command_plotted:
            axis.plot(time, episodes[0]["target_velocity"][:sample_count, 0],
                      color="black", linestyle="--", linewidth=2.2,
                      label="Commanded velocity")
            command_plotted = True
    axis.set_xlabel("Simulation time (s)")
    axis.set_ylabel("Forward velocity (m/s)")
    axis.set_title("Commanded and achieved velocity across stable morphologies",
                   fontweight="bold")
    axis.grid(alpha=0.22)
    axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    save_figure(fig, output, "03b_combined_velocity_trajectories_vx085")


def plot_heights(rows, output, speed=0.85):
    selected = {row["morphology"]: row for row in rows if np.isclose(row["speed"], speed)}
    morphology_order = [name for name in MORPHOLOGY_ORDER if name in selected]
    columns = 3
    rows_count = math.ceil(len(morphology_order) / columns)
    fig, axes = plt.subplots(rows_count, columns, figsize=(14, 3.1 * rows_count),
                             sharex=True, constrained_layout=True, squeeze=False)
    for axis, morphology in zip(axes.flat, morphology_order):
        row = selected[morphology]
        for episode_path in sorted(row["folder"].glob("episode_*.npz")):
            arrays = np.load(episode_path)
            axis.plot(arrays["time"], arrays["qpos"][:, 2],
                      color=COLORS[morphology], alpha=0.34, linewidth=0.9)
        axis.set_title(f"{MORPHOLOGY_LABELS[morphology]} "
                       f"({row['completed']}/{row['episodes']} complete)",
                       fontweight="bold")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(morphology_order):]:
        axis.set_visible(False)
    for axis in axes.flat:
        if axis.get_visible():
            axis.set_xlabel("Simulation time (s)")
            axis.set_ylabel("Torso height (m)")
    fig.suptitle(f"Torso-height trajectories at {speed:.2f} m/s command",
                 fontsize=15, fontweight="bold")
    save_figure(fig, output, "04_height_trajectories_vx085")


def plot_transient_cost_of_transport(rows, output, speed=0.85, window_seconds=1.0):
    selected = {row["morphology"]: row for row in rows if np.isclose(row["speed"], speed)}
    morphology_order = [name for name in MORPHOLOGY_ORDER if name in selected]
    fig, axis = plt.subplots(figsize=(14, 7.5), constrained_layout=True)
    for morphology in morphology_order:
        row = selected[morphology]
        model = mujoco.MjModel.from_xml_path(str(row["folder"] / "model.xml"))
        actuator_dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
        mass = float(np.sum(model.body_mass))
        episode_series = []
        for episode_path in sorted(row["folder"].glob("episode_*.npz")):
            arrays = np.load(episode_path)
            time = arrays["time"].astype(float)
            timestep = float(np.median(np.diff(time)))
            window_steps = max(1, round(window_seconds / timestep))
            torque = arrays["actuator_force"].astype(float)
            joint_velocity = arrays["qvel"][:, actuator_dofs].astype(float)
            positive_power = np.maximum(torque * joint_velocity, 0.0).sum(axis=1)
            cumulative_work = np.concatenate(([0.0], np.cumsum(positive_power) * timestep))
            x_position = arrays["qpos"][:, 0].astype(float)
            transient_cot = np.full(len(time), np.nan)
            window_work = cumulative_work[window_steps + 1:] - cumulative_work[1:-window_steps]
            window_distance = x_position[window_steps:] - x_position[:-window_steps]
            valid = window_distance > 0.05
            values = np.full(len(window_distance), np.nan)
            values[valid] = window_work[valid] / (mass * 9.81 * window_distance[valid])
            transient_cot[window_steps:] = values
            episode_series.append(transient_cot)
        sample_count = max(len(series) for series in episode_series)
        stacked = np.full((len(episode_series), sample_count), np.nan)
        for index, series in enumerate(episode_series):
            stacked[index, :len(series)] = series
        count = np.sum(np.isfinite(stacked), axis=0)
        mean = np.divide(np.nansum(stacked, axis=0), count,
                         out=np.full(sample_count, np.nan), where=count > 0)
        centered = np.where(np.isfinite(stacked), stacked - mean, np.nan)
        deviation = np.sqrt(np.divide(np.nansum(centered ** 2, axis=0), count,
                                      out=np.full(sample_count, np.nan), where=count > 0))
        plot_time = np.arange(sample_count) * 0.01
        color = COLORS[morphology]
        linestyle = "-" if row["completed"] == row["episodes"] else "--"
        label = (f"{MORPHOLOGY_LABELS[morphology]} "
                 f"({row['completed']}/{row['episodes']} complete)")
        axis.plot(plot_time, mean, color=color, linestyle=linestyle, linewidth=1.7, label=label)
        axis.fill_between(plot_time, mean - deviation, mean + deviation, color=color, alpha=0.06)
    axis.set_xlabel("Simulation time (s)")
    axis.set_ylabel("1-second rolling positive mechanical CoT")
    axis.set_ylim(bottom=0)
    axis.set_title("Transient mechanical cost of transport across morphologies",
                   fontweight="bold")
    axis.grid(alpha=0.22)
    axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    save_figure(fig, output, "08_transient_cost_of_transport_vx085")


def incumbent_series(optimization):
    baseline = optimization["baseline"]["objective"]
    generations = [0]
    completion = [baseline["completion_fraction"]]
    survival = [baseline["survival_fraction"]]
    transport = [baseline.get("mean_positive_mechanical_cot", np.nan)]
    cost = [baseline["mean_cost_per_second"]]
    for index, generation in enumerate(optimization["generations"], start=1):
        objective = generation["best_objective_so_far"]
        generations.append(index)
        completion.append(objective["completion_fraction"])
        survival.append(objective["survival_fraction"])
        transport.append(objective.get("mean_positive_mechanical_cot", np.nan))
        cost.append(objective["mean_cost_per_second"])
    return np.asarray(generations), (completion, survival, transport, cost)


def plot_convergence(optimizations, labels, output):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), constrained_layout=True)
    titles = ("Best completion fraction", "Best survival fraction",
              "Best positive mechanical CoT", "Best cost per second")
    for optimization, label in zip(optimizations, labels):
        generations, series = incumbent_series(optimization)
        for axis, values in zip(axes, series):
            axis.step(generations, values, where="post", marker="o", markersize=3, label=label)
    for axis, title in zip(axes, titles):
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Completed CMA-ES generation")
        axis.grid(alpha=0.25)
        axis.set_xlim(0, max(len(run["generations"]) for run in optimizations))
    axes[0].set_ylabel("Fraction")
    axes[1].set_ylabel("Fraction")
    axes[2].set_ylabel("Dimensionless CoT")
    axes[3].set_ylabel("Integrated running cost / s")
    axes[0].set_ylim(-0.03, 1.03)
    axes[1].set_ylim(-0.03, 1.03)
    axes[0].legend()
    fig.suptitle("Independent morphology-optimization convergence", fontsize=15, fontweight="bold")
    save_figure(fig, output, "05_optimization_convergence")


def plot_design_space(optimizations, labels, output):
    fig, axes = plt.subplots(2, len(optimizations), figsize=(6.2 * len(optimizations), 9),
                             constrained_layout=True, squeeze=False)
    normalization = Normalize(0, 1)
    for column, (optimization, label) in enumerate(zip(optimizations, labels)):
        evaluations = [evaluation for generation in optimization["generations"]
                       for evaluation in generation["evaluations"] if "objective" in evaluation]
        theta = np.asarray([evaluation["theta"] for evaluation in evaluations])
        survival = np.asarray([evaluation["objective"]["survival_fraction"] for evaluation in evaluations])
        completion = np.asarray([evaluation["objective"]["completion_fraction"] for evaluation in evaluations])
        generations = np.repeat(np.arange(1, len(optimization["generations"]) + 1),
                                [len(generation["evaluations"]) for generation in optimization["generations"]])
        scatter = axes[0, column].scatter(theta[:, 0], theta[:, 1], c=survival,
                                          cmap="viridis", norm=normalization,
                                          s=35 + 70 * completion, alpha=0.8,
                                          edgecolors="#333333", linewidths=0.25)
        nominal = np.asarray(optimization["theta0"])
        winner = np.asarray(optimization["best_theta"])
        axes[0, column].scatter(nominal[0], nominal[1], marker="X", s=150,
                                color="white", edgecolors="black", label="Nominal")
        axes[0, column].scatter(winner[0], winner[1], marker="*", s=220,
                                color="#E45756", edgecolors="black", label="Winner")
        axes[0, column].set_xlabel("Thigh length (m)")
        axes[0, column].set_ylabel("Shin length (m)")
        axes[0, column].set_title(f"{label}: length search", fontweight="bold")
        axes[0, column].grid(alpha=0.2)
        axes[0, column].legend()
        axes[1, column].scatter(generations, theta[:, 3], c=completion,
                                cmap="plasma", norm=normalization, s=45, alpha=0.8,
                                edgecolors="#333333", linewidths=0.25)
        axes[1, column].axhline(nominal[3], color="black", linestyle="--", label="Nominal")
        axes[1, column].axhline(winner[3], color="#E45756", linestyle="-", label="Winner")
        axes[1, column].set_xlabel("Generation")
        axes[1, column].set_ylabel("Hip width (m)")
        axes[1, column].set_title(f"{label}: hip-width exploration", fontweight="bold")
        axes[1, column].grid(alpha=0.2)
        axes[1, column].legend()
    fig.colorbar(scatter, ax=list(axes[0]), label="Survival fraction", shrink=0.78)
    completion_map = plt.cm.ScalarMappable(norm=normalization, cmap="plasma")
    fig.colorbar(completion_map, ax=list(axes[1]), label="Completion fraction", shrink=0.78)
    fig.suptitle("Morphology candidates evaluated by independent CMA-ES runs",
                 fontsize=15, fontweight="bold")
    save_figure(fig, output, "06_design_search_space")


def plot_winner_dimensions(optimizations, labels, output):
    nominal = np.asarray(optimizations[0]["theta0"])
    designs = [nominal] + [np.asarray(run["best_theta"]) for run in optimizations]
    names = ["Nominal"] + labels
    dimensions = ((0, "Thigh length (m)"), (1, "Shin length (m)"), (3, "Hip width (m)"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), constrained_layout=True)
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for axis, (index, title) in zip(axes, dimensions):
        values = [design[index] for design in designs]
        bars = axis.bar(names, values, color=colors[:len(values)])
        axis.set_title(title, fontweight="bold")
        axis.tick_params(axis="x", rotation=18)
        axis.set_ylim(min(values) * 0.88, max(values) * 1.08)
        axis.grid(axis="y", alpha=0.2)
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3)
    fig.suptitle("Nominal geometry and independently optimized winners",
                 fontsize=15, fontweight="bold")
    save_figure(fig, output, "07_winner_dimensions")


def write_summary(rows, optimizations, labels, output):
    with (output / "benchmark_summary.csv").open("w", newline="") as file:
        fields = ("morphology", "speed_m_s", "completed", "episodes", "completion_fraction",
                  "survival_fraction", "mean_survival_seconds", "mean_vx_m_s",
                  "tracking_rmse_m_s", "cost_per_second")
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "morphology": row["morphology"],
                "speed_m_s": row["speed"],
                "completed": row["completed"],
                "episodes": row["episodes"],
                "completion_fraction": row["completion_fraction"],
                "survival_fraction": row["survival_fraction"],
                "mean_survival_seconds": row["mean_survival_seconds"],
                "mean_vx_m_s": row["mean_vx"],
                "tracking_rmse_m_s": row["tracking_rmse"],
                "cost_per_second": row["cost_per_second"],
            })
    episode_counts = sorted({row["episodes"] for row in rows})
    episode_label = str(episode_counts[0]) if len(episode_counts) == 1 else "a variable number of"
    lines = [
        "# MPC morphology experiment figures",
        "",
        f"All benchmark panels aggregate {episode_label} held-out episodes per morphology-command condition.",
        "Trajectory panels show every episode rather than a selected successful rollout.",
        "",
        "## Optimization winners",
        "",
        "| Run | Thigh (m) | Shin (m) | Hip width (m) | Completion | Survival | CoT | Cost/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for optimization, label in zip(optimizations, labels):
        theta = optimization["best_theta"]
        objective = optimization["best_objective"]
        transport = objective.get("mean_positive_mechanical_cot")
        transport_label = f"{transport:.4f}" if transport is not None else "n/a"
        lines.append(
            f"| {label} | {theta[0]:.4f} | {theta[1]:.4f} | {theta[3]:.4f} | "
            f"{objective['completion_fraction']:.0%} | {objective['survival_fraction']:.0%} | "
            f"{transport_label} | {objective['mean_cost_per_second']:.4f} |"
        )
    lines.extend((
        "",
        "The optimization objectives use the four common search episodes and are not held-out validation.",
        "The multi-morphology benchmark uses evaluation seed 1045 and 20-second episodes.",
    ))
    (output / "README.md").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if len(args.optimization) != len(args.label):
        raise ValueError("Provide one --label for each --optimization")
    args.output.mkdir(parents=True, exist_ok=True)
    benchmark = load_json(args.benchmark)
    optimizations = [load_json(path) for path in args.optimization]
    rows = condition_rows(benchmark)
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    plot_overview(rows, args.output)
    plot_survival(rows, args.output)
    plot_trajectories(rows, args.output)
    plot_combined_trajectories(rows, args.output)
    plot_heights(rows, args.output)
    plot_convergence(optimizations, args.label, args.output)
    plot_design_space(optimizations, args.label, args.output)
    plot_winner_dimensions(optimizations, args.label, args.output)
    plot_transient_cost_of_transport(rows, args.output)
    write_summary(rows, optimizations, args.label, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
