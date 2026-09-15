"""Headless closed-loop DIAL evaluation; also used by the live viewer."""

from importlib.metadata import version
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
from hydrax.algs import DIAL
from mujoco import mjx
import numpy as np

from dial_lerobot.assets.model_builder import build_model, describe_model, initial_data, model_xml, nominal_design
from dial_lerobot.config import Config
from dial_lerobot.tasks.lerobot_task import COST_NAMES, LeRobotLocomotionTask


def write_json(path, value):
    """Atomic progress file: interrupted experiments leave the last complete data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def runtime_versions():
    return {"packages": {p: version(p) for p in ("mujoco", "mujoco-mjx", "jax", "jaxlib", "hydrax", "cma")},
            "devices": [str(d) for d in jax.devices()],
            "jax_cache_dir": jax.config.jax_compilation_cache_dir}


class Evaluator:
    def __init__(self, cfg: Config = Config(), theta=None):
        self.cfg = cfg.validate()
        self.theta = nominal_design(cfg.robot) if theta is None else np.asarray(theta, dtype=float)
        self.model = build_model(self.theta, cfg.robot)
        self.task = LeRobotLocomotionTask(self.model, cfg.cost)
        m = cfg.mpc
        self.controller = DIAL(
            self.task, num_samples=m.num_samples, noise_level=m.noise_level,
            beta_opt_iter=m.beta_opt_iter, beta_horizon=m.beta_horizon,
            temperature=m.temperature, plan_horizon=m.plan_horizon,
            num_knots=m.num_knots, iterations=m.iterations, spline_type=m.spline_type,
        )
        self.step = jax.jit(self._step)

    def reset(self, seed=0, height_offset=None):
        if height_offset is None:
            height_offset = np.random.default_rng(seed).uniform(
                -self.cfg.evaluation.height_jitter, self.cfg.evaluation.height_jitter)
        data = mjx.put_data(self.model, initial_data(self.model, height_offset))
        params = self.controller.init_params(
            initial_knots=jnp.zeros((self.cfg.mpc.num_knots, self.model.nu)), seed=seed)
        return data, params

    def _step(self, state, params):
        params, rollouts = self.controller.optimize(state, params)
        action = jnp.clip(self.controller.get_action(params, state.time), self.task.u_min, self.task.u_max)

        def physics_step(data, _):
            data = mjx.step(self.task.model, data.replace(ctrl=action))
            # Refresh kinematics so feet and qpos describe the same instant.
            data = mjx.forward(self.task.model, data)
            return data, self.task.cost_terms(data, action)

        state, costs = jax.lax.scan(physics_step, state, None, length=self.cfg.mpc.control_decimation)
        feet = state.site_xpos[self.task.foot_sites]
        row = {
            "time": state.time, "qpos": state.qpos, "qvel": state.qvel,
            "ctrl": action, "actuator_force": state.actuator_force,
            "cost_terms": jnp.sum(costs, axis=0), "body_velocity": self.task.body_velocity(state),
            "target_velocity": self.task.target_velocity(state.time), "feet": feet,
            "foot_target_z": self.task.gait_reference(state.time)[0],
            "healthy": self.task.healthy(state),
            "plan_cost_min": jnp.min(jnp.sum(rollouts.costs, axis=-1)),
        }
        return state, params, row

    def warmup(self):
        """Compile without consuming the actual episode state or random sequence."""
        start = time.perf_counter()
        state, params = self.reset(height_offset=0.0)
        state, params, row = self.step(state, params)
        state, _, row = self.step(state, params)
        jax.block_until_ready(state.qpos)
        if not np.isfinite(np.asarray(row["plan_cost_min"])):
            raise FloatingPointError("Non-finite planning cost during compilation/warmup")
        return time.perf_counter() - start

    def metadata(self):
        return {"config": self.cfg.to_dict(), "model": describe_model(self.model, self.theta, self.cfg.robot),
                "task": self.task.describe(), "runtime": runtime_versions(), "controller": "DIAL"}

    def episode(self, seed=0, progress=False):
        state, params = self.reset(seed)
        initial_x = float(state.qpos[0])
        log = {}
        start = time.perf_counter()
        for step in range(self.cfg.evaluation.episode_steps):
            tick = time.perf_counter()
            state, params, row = self.step(state, params)
            host = jax.device_get(row)
            if not all(np.isfinite(v).all() for v in host.values()):
                raise FloatingPointError(f"Non-finite DIAL rollout at update {step}")
            for key, value in host.items():
                log.setdefault(key, []).append(value)
            log.setdefault("wall_step_seconds", []).append(time.perf_counter() - tick)
            if progress and (step + 1) % 25 == 0:
                print(f"  update {step + 1}/{self.cfg.evaluation.episode_steps}, "
                      f"sim {float(host['time']):.2f}s, vx {float(host['body_velocity'][0]):.3f} m/s, "
                      f"z {float(host['qpos'][2]):.3f}m", flush=True)
        arrays = {key: np.asarray(value) for key, value in log.items()}
        distance = float(arrays["qpos"][-1, 0] - initial_x)
        total_cost = float(arrays["cost_terms"].sum())
        summary = {"seed": int(seed), "fitness": distance - total_cost,
                   "distance_x_m": distance, "total_running_cost": total_cost,
                   "sim_seconds": float(arrays["time"][-1]), "wall_seconds": time.perf_counter() - start,
                   "mean_vx_m_s": float(arrays["body_velocity"][:, 0].mean()),
                   "final_height_m": float(arrays["qpos"][-1, 2]),
                   "healthy_fraction": float(arrays["healthy"].mean()),
                   "cost_totals": dict(zip(COST_NAMES, arrays["cost_terms"].sum(axis=0).tolist()))}
        return summary, arrays

    def evaluate(self, seed=None, output=None, progress=False):
        seed = self.cfg.evaluation.seed if seed is None else seed
        if output:
            folder = Path(output)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "model.xml").write_text(model_xml(self.theta, self.cfg.robot))
            write_json(folder / "metadata.json", self.metadata())
        seeds = np.random.SeedSequence(seed).generate_state(self.cfg.evaluation.num_episodes)
        summaries = []
        for i, episode_seed in enumerate(seeds):
            if progress:
                print(f"Episode {i + 1}/{len(seeds)}", flush=True)
            summary, arrays = self.episode(int(episode_seed), progress)
            summaries.append(summary)
            if output:
                folder = Path(output)
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / f"episode_{i:03d}.npz", **arrays)
                columns = np.column_stack((arrays["time"], arrays["qpos"][:, 2],
                                           arrays["body_velocity"][:, :2], arrays["target_velocity"],
                                           arrays["feet"][:, :, 2], arrays["healthy"], arrays["cost_terms"]))
                np.savetxt(folder / f"episode_{i:03d}.csv", columns, delimiter=",", comments="",
                           header="time,height,vx,vy,target_vx,target_vy,left_foot_z,right_foot_z,healthy," + ",".join(COST_NAMES))
        result = self.metadata() | {"fitness": float(np.mean([s["fitness"] for s in summaries])),
                                    "episodes": summaries}
        if output:
            write_json(Path(output) / "evaluation.json", result)
        return result


def evaluate_design(theta, seed=0, cfg=Config()):
    """Compatibility with the source co-design fitness: final x - sum(cost)."""
    evaluator = Evaluator(cfg, theta)
    return evaluator.evaluate(seed)["fitness"]
