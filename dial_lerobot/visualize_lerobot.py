"""Live DIAL, saved trajectory replay, static rendering and diagnostic plots."""

import json
from pathlib import Path
import time

import mujoco
import numpy as np

from dial_lerobot.assets.model_builder import build_model, initial_data
from dial_lerobot.config import from_dict


def _camera(viewer, model):
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = model.body("torso_subassembly").id
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.4, 125, -15


def _key_handler(flags):
    def key(code):
        if code == 32:
            flags["paused"] = not flags["paused"]
        elif code in (ord("R"), ord("r")):
            flags["reset"] = True
    return key


def _set_frame(model, data, row):
    data.qpos[:], data.qvel[:], data.ctrl[:] = row["qpos"], row["qvel"], row["ctrl"]
    data.time = float(row["time"])
    mujoco.mj_forward(model, data)


def live_view(evaluator, seconds=10.0, rate=1.0):
    import jax
    import mujoco.viewer

    print("Compiling DIAL before opening the viewer...", flush=True)
    print(f"Warmup: {evaluator.warmup():.2f}s", flush=True)
    state, params = evaluator.reset(evaluator.cfg.evaluation.seed, height_offset=0)
    data = initial_data(evaluator.model)
    flags = {"paused": False, "reset": False}
    print("Space: pause/resume; R: reset; close window: exit. Velocity targets come from --vx/--vy or TOML.", flush=True)
    dt = evaluator.cfg.robot.timestep * evaluator.cfg.mpc.control_decimation
    with mujoco.viewer.launch_passive(evaluator.model, data, key_callback=_key_handler(flags)) as view:
        _camera(view, evaluator.model)
        last_report = 0.0
        while view.is_running():
            tick = time.perf_counter()
            if flags["reset"]:
                state, params = evaluator.reset(evaluator.cfg.evaluation.seed, height_offset=0)
                flags.update(reset=False, paused=False)
            if not flags["paused"]:
                state, params, row = evaluator.step(state, params)
                row = jax.device_get(row)
                if not all(np.isfinite(v).all() for v in row.values()):
                    raise FloatingPointError("Non-finite live MPC state")
                _set_frame(evaluator.model, data, row)
                if time.perf_counter() - last_report > 2:
                    print(f"sim {data.time:.2f}s | vx {row['body_velocity'][0]:.3f} "
                          f"target {row['target_velocity'][0]:.3f} m/s | z {data.qpos[2]:.3f}m", flush=True)
                    last_report = time.perf_counter()
                if seconds > 0 and data.time >= seconds:
                    flags["paused"] = True
                    print("Duration reached. R restarts; close the window to exit.", flush=True)
            view.sync()
            time.sleep(max(0.001, (0.01 if flags["paused"] else dt / rate) - (time.perf_counter() - tick)))


def load_trajectory(path):
    path = Path(path)
    meta_path = path.parent / "metadata.json"
    if not meta_path.exists():
        meta_path = path.parent / "evaluation.json"
    meta = json.loads(meta_path.read_text())
    xml_path = path.parent / "model.xml"
    model = (mujoco.MjModel.from_xml_path(str(xml_path)) if xml_path.is_file()
             else build_model(meta["model"]["theta"], from_dict(meta["config"]).robot))
    with np.load(path, allow_pickle=False) as archive:
        trajectory = {k: archive[k] for k in ("time", "qpos", "qvel", "ctrl")}
    return model, trajectory


def replay(path, rate=1.0, loop=False):
    import mujoco.viewer

    model, trajectory = load_trajectory(path)
    data = initial_data(model)
    flags = {"paused": False, "reset": False}
    dt = float(trajectory["time"][1] - trajectory["time"][0]) if len(trajectory["time"]) > 1 else 0.02
    print("Replay: Space pauses/resumes; R restarts; close window exits.", flush=True)
    with mujoco.viewer.launch_passive(model, data, key_callback=_key_handler(flags)) as view:
        _camera(view, model)
        index = 0
        while view.is_running():
            tick = time.perf_counter()
            if flags["reset"]:
                index = 0
                flags.update(reset=False, paused=False)
            if not flags["paused"]:
                _set_frame(model, data, {k: v[index] for k, v in trajectory.items()})
                index += 1
                if index == len(trajectory["time"]):
                    index = 0 if loop else index - 1
                    flags["paused"] = not loop
            view.sync()
            time.sleep(max(0.001, dt / rate - (time.perf_counter() - tick)))


def snapshot(model, output):
    from PIL import Image

    data = initial_data(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, data.qpos[2] / 2]
    camera.distance, camera.azimuth, camera.elevation = 1.9, 125, -12
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 720
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


def plot_results(path, output=None):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from dial_lerobot.tasks.lerobot_task import COST_NAMES

    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            a = dict(archive)
        t = a["time"]
        fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
        axes[0, 0].plot(t, a["body_velocity"][:, 0], label="actual vx")
        axes[0, 0].plot(t, a["target_velocity"][:, 0], "--", label="target vx")
        axes[0, 0].set(ylabel="m/s", title="Forward velocity in body frame")
        axes[0, 1].plot(t, a["qpos"][:, 2], label="base height")
        axes[0, 1].set(ylabel="m", title="Base height")
        for i, side in enumerate(("left", "right")):
            axes[1, 0].plot(t, a["feet"][:, i, 2], label=side)
            axes[1, 0].plot(t, a["foot_target_z"][:, i], "--", label=f"{side} target")
        axes[1, 0].set(ylabel="m", title="Sole height above flat ground")
        for i, name in enumerate(COST_NAMES):
            axes[1, 1].plot(t, a["cost_terms"][:, i], label=name)
        axes[1, 1].set(title="Weighted cost per MPC update")
        axes[2, 0].plot(a["qpos"][:, 0], a["qpos"][:, 1], label="root path")
        axes[2, 0].set(xlabel="world x (m)", ylabel="world y (m)", title="Path")
        axes[2, 0].axis("equal")
        axes[2, 1].plot(t, a["ctrl"])
        axes[2, 1].set(title="12 position targets", ylabel="rad")
        for ax in axes.flat:
            ax.grid(alpha=0.3)
            if ax is not axes[2, 0]:
                ax.set_xlabel("simulation time (s)")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
    else:
        log = json.loads(path.read_text())
        if "rows" in log:
            rows = log["rows"]
            fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
            scatter = ax.scatter([r["theta"][0] for r in rows], [r["theta"][1] for r in rows],
                                 c=[r["fitness"] for r in rows], marker="s", s=200)
            fig.colorbar(scatter, ax=ax, label="fitness (larger is better)")
            ax.set(xlabel="thigh anchor distance (m)", ylabel="shin anchor distance (m)", title="Design sweep")
        else:
            generations = log["generations"]
            fig, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)
            axes[0].plot([g["generation"] + 1 for g in generations], [g["best_so_far"] for g in generations])
            axes[0].set(ylabel="best fitness", xlabel="generation")
            for i, name in enumerate(("thigh (m)", "shin (m)", "density proxy")):
                best = [max(g["evaluations"], key=lambda r: r["fitness"])["theta"][i] for g in generations]
                axes[1].plot(range(1, len(best) + 1), best, label=name)
            axes[1].set(xlabel="generation", ylabel="generation-best parameters")
            axes[1].legend()
    output = Path(output) if output else path.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
