"""Generate interpolation-order and knot-count comparison figures."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mujoco
import numpy as np


ROOT = Path("/home/rory/mpc_results")
ORDER_ROOT = ROOT / "interpolation_sweep_512_5_2_20261101"
KNOT_ROOT = ROOT / "interpolation_qc_knot_sweep_vx085_512_5_2_20261102"
SIMPLE_KNOT_ROOT = ROOT / "interpolation_knot_sweep_vx085_512_5_2_20261102"
OUTPUT = ROOT / "interpolation_figure_bundle_20261102"
WINDOW_SECONDS = 0.5
COLORS = {
    "zero": "#4C78A8",
    "linear": "#F58518",
    "quadratic_dial": "#54A24B",
    "cubic_dial": "#E45756",
}
ORDER_LABELS = {
    "zero": "Zero-order",
    "linear": "Linear",
    "quadratic_dial": "Quadratic",
    "cubic_dial": "Cubic",
}


def episode_paths(folder):
    return sorted(folder.glob("episode_*.npz"))


def load_episode(path):
    arrays = np.load(path)
    time = np.asarray(arrays["time"], dtype=float)
    velocity = np.asarray(arrays["body_velocity"], dtype=float)[:, 0]
    target = np.asarray(arrays["target_velocity"], dtype=float)[:, 0]
    model = mujoco.MjModel.from_xml_path(str(path.parent / "model.xml"))
    actuator_dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    torque = np.asarray(arrays["actuator_force"], dtype=float)
    joint_velocity = np.asarray(arrays["qvel"], dtype=float)[:, actuator_dofs]
    positive_power = np.maximum(torque * joint_velocity, 0.0).sum(axis=1)
    return time, velocity, target, positive_power, float(model.body_mass.sum())


def transient_cot(path):
    time, _, _, power, mass = load_episode(path)
    if len(time) < 2:
        return np.array([]), np.array([])
    dt = float(np.median(np.diff(time)))
    edges = np.arange(0.0, time[-1] + WINDOW_SECONDS, WINDOW_SECONDS)
    values = []
    centers = []
    data = np.load(path)
    forward_velocity = np.asarray(data["body_velocity"], dtype=float)[:, 0]
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (time >= start) & (time < end)
        if not np.any(mask):
            continue
        work = float(np.sum(power[mask]) * dt)
        distance = float(np.sum(np.maximum(forward_velocity[mask], 0.0)) * dt)
        values.append(work / (mass * 9.81 * distance) if distance > 1e-5 else np.nan)
        centers.append(float((start + end) / 2.0))
    return np.asarray(centers), np.asarray(values)


def condition(root, name):
    folder = root / name
    if not (folder / "evaluation.json").exists():
        raise FileNotFoundError(folder)
    return folder


def plot_order_figure(field, output_path):
    speeds = (0.50, 0.85, 1.25)
    orders = ("zero", "linear", "quadratic_dial", "cubic_dial")
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=False)
    for axis, speed in zip(axes, speeds):
        speed_label = f"{speed:.2f}".replace(".", "p")
        for order in orders:
            folder = condition(ORDER_ROOT, f"{order}_vx_{speed_label}")
            for episode in episode_paths(folder):
                if field == "cot":
                    x, y = transient_cot(episode)
                else:
                    data = np.load(episode)
                    x = np.asarray(data["time"])
                    y = np.asarray(data["body_velocity"])[:, 0]
                axis.plot(x, y, color=COLORS[order], alpha=0.55, linewidth=1.1)
        if field == "velocity":
            sample = np.load(episode_paths(condition(ORDER_ROOT, f"quadratic_dial_vx_{speed_label}"))[0])
            axis.plot(np.asarray(sample["time"]), np.asarray(sample["target_velocity"])[:, 0],
                      "k:", linewidth=1.8, label="Commanded velocity")
            axis.set_ylabel("Velocity (m/s)")
        else:
            axis.set_ylabel("Windowed CoT")
        axis.set_title(f"Commanded speed: {speed:.2f} m/s")
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    handles = [Line2D([0], [0], color=COLORS[key], linewidth=2, label=ORDER_LABELS[key])
               for key in orders]
    if field == "velocity":
        handles.append(Line2D([0], [0], color="black", linestyle=":", linewidth=2,
                              label="Commanded velocity"))
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.84, 0.5), frameon=True)
    fig.subplots_adjust(right=0.82, hspace=0.30)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_simple_knot_figure(orders, field, output_path, root):
    knots = tuple(knot for knot in (4, 6, 8, 10, 16, 24)
                  if all((root / f"{order}_{knot}knots_vx_0p85" / "evaluation.json").exists()
                         for order in orders))
    knot_colors = dict(zip(knots, ("#4C78A8", "#F58518", "#54A24B", "#E45756")))
    fig, axes = plt.subplots(len(orders), 1, figsize=(12, 8), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, order in zip(axes, orders):
        for knot in knots:
            folder = condition(root, f"{order}_{knot}knots_vx_0p85")
            for episode in episode_paths(folder):
                data = np.load(episode)
                if field == "velocity":
                    x = np.asarray(data["time"])
                    y = np.asarray(data["body_velocity"])[:, 0]
                else:
                    x, y = transient_cot(episode)
                axis.plot(x, y, color=knot_colors[knot], alpha=0.55, linewidth=1.1)
        if field == "velocity":
            sample = np.load(episode_paths(condition(root, f"{order}_{knots[0]}knots_vx_0p85"))[0])
            axis.plot(np.asarray(sample["time"]), np.asarray(sample["target_velocity"])[:, 0],
                      "k:", linewidth=1.8, label="Commanded velocity")
            axis.set_ylabel("Velocity (m/s)")
        else:
            axis.set_ylabel("Windowed CoT")
        axis.set_title(f"{ORDER_LABELS[order]} interpolation")
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    handles = [Line2D([0], [0], color=knot_colors[knot], linewidth=2, label=f"{knot} knots")
               for knot in knots]
    if field == "velocity":
        handles.append(Line2D([0], [0], color="black", linestyle=":", linewidth=2,
                              label="Commanded velocity"))
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.84, 0.5), frameon=True)
    fig.subplots_adjust(right=0.82, hspace=0.30)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plot_order_figure("cot", OUTPUT / "interpolation_orders_cot_by_speed.png")
    plot_order_figure("velocity", OUTPUT / "interpolation_orders_velocity_by_speed.png")
    plot_simple_knot_figure(("zero", "linear"), "velocity",
                            OUTPUT / "zero_linear_velocity_by_knots.png", SIMPLE_KNOT_ROOT)
    plot_simple_knot_figure(("quadratic_dial", "cubic_dial"), "cot",
                            OUTPUT / "quadratic_cubic_cot_by_knots.png", KNOT_ROOT)
    plot_simple_knot_figure(("quadratic_dial", "cubic_dial"), "velocity",
                            OUTPUT / "quadratic_cubic_velocity_by_knots.png", KNOT_ROOT)
    manifest = {"output": str(OUTPUT), "window_seconds": WINDOW_SECONDS,
                "order_root": str(ORDER_ROOT), "knot_root": str(KNOT_ROOT)}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated figures in {OUTPUT}")


if __name__ == "__main__":
    main()
