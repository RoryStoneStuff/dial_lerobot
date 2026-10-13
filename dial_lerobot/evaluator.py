"""Headless closed-loop DIAL evaluation; also used by the live viewer."""

from importlib.metadata import version
from copy import copy
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
from dial_lerobot.controllers import LeRobotDIAL
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


class MorphologyKernel:
    """Shared JIT function for fixed topology and changing physical arrays."""
    attributes = ("rest_height", "healthy_z_min", "foot_geom_sizes", "foot_height_offset")

    def __init__(self, template):
        self.template = template
        self.config = (template.cfg.robot, template.cfg.cost, template.cfg.mpc)
        self.structure = jax.tree.structure(self.inputs(template.task))
        self.step = jax.jit(self._step)

    def inputs(self, task):
        return (task.model, tuple(jnp.asarray(getattr(task, name), dtype=jnp.float32)
                                 for name in self.attributes))

    def _step(self, inputs, state, params):
        # Copies exist only during tracing; no runtime tracers escape into
        # the persistent task/controller or another morphology evaluator.
        evaluator = copy(self.template)
        evaluator.task = copy(self.template.task)
        evaluator.task.model, values = inputs
        for name, value in zip(self.attributes, values):
            setattr(evaluator.task, name, value)
        evaluator.controller = copy(self.template.controller)
        evaluator.controller.task = evaluator.task
        evaluator.controller.model = evaluator.task.model
        return evaluator._step(state, params)


class Evaluator:
    def __init__(self, cfg: Config = Config(), theta=None, *, shared_kernel=None, reuse_compilation=False):
        self.cfg = cfg.validate()
        self.theta = nominal_design(cfg.robot) if theta is None else np.asarray(theta, dtype=float)
        self.model = build_model(self.theta, cfg.robot)
        self.physics_model = (build_model(self.theta, cfg.robot, include_visual=False)
                              if reuse_compilation or shared_kernel is not None else self.model)
        self.task = LeRobotLocomotionTask(self.physics_model, cfg.cost)
        if reuse_compilation or shared_kernel is not None:
            # This static bound is consulted only for height-field collisions.
            # Flat-ground designs can share it without freezing physical bounds.
            if self.physics_model.nhfield:
                raise ValueError("Shared morphology compilation requires flat ground")
            impl = self.task.model._impl
            self.task.model = self.task.model.replace(_impl=impl.replace(
                geom_rbound_hfield=np.zeros_like(impl.geom_rbound_hfield)))
        m = cfg.mpc
        self.controller = LeRobotDIAL(
            self.task, num_samples=m.num_samples, noise_level=m.noise_level,
            beta_opt_iter=m.beta_opt_iter, beta_horizon=m.beta_horizon,
            temperature=m.temperature, plan_horizon=m.plan_horizon,
            num_knots=m.num_knots, iterations=m.iterations, spline_type=m.spline_type,
            initial_diffusion_iterations=m.initial_diffusion_iterations,
            trajectory_diffuse_factor=m.trajectory_diffuse_factor,
            normalize_costs_for_temperature=m.normalize_costs_for_temperature,
            append_mean_sample=m.append_mean_sample, fix_initial_knot=m.fix_initial_knot,
            shift_reconstruction=m.shift_reconstruction, shift_steps=m.control_decimation,
        )
        self.shared_kernel = shared_kernel
        if reuse_compilation and shared_kernel is None:
            self.shared_kernel = MorphologyKernel(self)
        if self.shared_kernel is None:
            self.step = jax.jit(self._step)
        else:
            kernel = self.shared_kernel
            if kernel.config != (cfg.robot, cfg.cost, cfg.mpc):
                raise ValueError("Shared kernel requires the same robot, cost, and MPC configuration")
            inputs = kernel.inputs(self.task)
            if jax.tree.structure(inputs) != kernel.structure:
                raise ValueError("Morphology changes static MJX topology; cannot reuse executable")
            self.step = lambda state, params: kernel.step(inputs, state, params)

    def reset(self, seed=0, height_offset=None):
        if height_offset is None:
            height_offset = np.random.default_rng(seed).uniform(
                -self.cfg.evaluation.height_jitter, self.cfg.evaluation.height_jitter)
        data = mjx.put_data(self.physics_model, initial_data(self.physics_model, height_offset))
        params = self.controller.init_params(
            initial_knots=jnp.zeros((self.cfg.mpc.num_knots, self.model.nu)), seed=seed)
        # Give initial scalar leaves the same explicit dtypes returned by JIT,
        # avoiding a separate startup executable for Python weak scalars.
        params = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.asarray(x).dtype), params)
        return data, params

    def _step(self, state, params):
        params, rollouts = self.controller.optimize(state, params)
        action = jnp.clip(self.controller.get_action(params, state.time), self.task.u_min, self.task.u_max)
        control = self.task.action_to_control(state, action)

        def physics_step(data, _):
            control_now = self.task.action_to_control(data, action)
            data = mjx.step(self.task.model, data.replace(ctrl=control_now))
            # Refresh kinematics so feet and qpos describe the same instant.
            data = self.task.refresh_kinematics(self.task.model, data)
            return data, self.task.cost_terms(data, action)

        state, costs = jax.lax.scan(physics_step, state, None, length=self.cfg.mpc.control_decimation)
        feet = state.site_xpos[self.task.foot_sites]
        plan_costs = jnp.sum(rollouts.costs, axis=-1)
        plan_cost_finite = jnp.isfinite(plan_costs)
        plan_cost_finite_count = jnp.sum(plan_cost_finite)
        finite_plan_costs = jnp.where(plan_cost_finite, plan_costs, 0.0)
        plan_weights = self.controller.cost_weights(plan_costs)
        row = {
            "time": state.time, "qpos": state.qpos, "qvel": state.qvel,
            "ctrl": control, "action": action, "actuator_force": state.actuator_force,
            "cost_terms": jnp.sum(costs, axis=0), "body_velocity": self.task.body_velocity(state),
            "target_velocity": self.task.target_velocity(state.time), "feet": feet,
            "foot_target_z": self.task.gait_reference(state.time)[0] + self.task.foot_height_offset,
            "foot_bottoms": self.task.foot_bottoms(state),
            "foot_velocity": self.task.foot_velocity(state),
            "stance_mask": self.task.stance_mask(state.time),
            "healthy": self.task.healthy(state),
            "plan_cost_min": jnp.min(plan_costs),
            "plan_cost_scaled_std": self.controller.normalized_cost_spread(plan_costs),
            "plan_cost_finite_count": plan_cost_finite_count,
            "plan_cost_nan_count": jnp.sum(jnp.isnan(plan_costs)),
            "plan_cost_inf_count": jnp.sum(jnp.isinf(plan_costs)),
            "plan_cost_finite_mean": jnp.sum(finite_plan_costs) / jnp.maximum(plan_cost_finite_count, 1),
            "plan_cost_finite_max": jnp.max(finite_plan_costs),
            "plan_weight_ess": 1.0 / jnp.sum(plan_weights ** 2),
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
        initial_height = float(state.qpos[2])
        failure_time = None
        log = {}
        start = time.perf_counter()
        failure_reason = None
        failure_diagnostic = None
        for step in range(self.cfg.evaluation.episode_steps):
            tick = time.perf_counter()
            previous_state = (state.time, state.qpos, state.qvel)
            state, params, row = self.step(state, params)
            host = jax.device_get(row)
            if not all(np.isfinite(v).all() for v in host.values()):
                failure_reason = "nonfinite"
                failure_time = (float(host["time"]) if np.isfinite(host["time"])
                                else (step + 1) * self.cfg.robot.timestep * self.cfg.mpc.control_decimation)
                bad_fields = {}
                for key, value in host.items():
                    value = np.asarray(value)
                    bad = ~np.isfinite(value)
                    if bad.any():
                        bad_fields[key] = {
                            "shape": list(value.shape),
                            "nonfinite_count": int(bad.sum()),
                            "indices": np.argwhere(bad)[:32].tolist(),
                        }
                prev_time, prev_qpos, prev_qvel = jax.device_get(previous_state)
                failure_diagnostic = {
                    "update": step + 1,
                    "previous_finite_state": {
                        "time": float(prev_time),
                        "qpos": np.asarray(prev_qpos).tolist(),
                        "qvel": np.asarray(prev_qvel).tolist(),
                    },
                    "failed_fields": bad_fields,
                    "action": (np.asarray(host["action"]).tolist()
                               if np.isfinite(host["action"]).all() else None),
                    "control": (np.asarray(host["ctrl"]).tolist()
                                if np.isfinite(host["ctrl"]).all() else None),
                    "planning": {
                        key: float(np.asarray(host[key]))
                        for key in (
                            "plan_cost_min", "plan_cost_scaled_std", "plan_cost_finite_count",
                            "plan_cost_nan_count", "plan_cost_inf_count",
                            "plan_cost_finite_mean", "plan_cost_finite_max", "plan_weight_ess",
                        )
                        if key in host and np.isfinite(host[key]).all()
                    },
                }
                if progress:
                    print(
                        f"  update {step + 1}: non-finite output in "
                        f"{', '.join(bad_fields)}; stopping and saving diagnostics",
                        flush=True,
                    )
                break
            for key, value in host.items():
                log.setdefault(key, []).append(value)
            log.setdefault("wall_step_seconds", []).append(time.perf_counter() - tick)
            healthy = bool(np.asarray(host["healthy"]))
            if progress and ((step + 1) % 25 == 0 or not healthy):
                print(f"  update {step + 1}/{self.cfg.evaluation.episode_steps}, "
                      f"sim {float(host['time']):.2f}s, vx {float(host['body_velocity'][0]):.3f} m/s, "
                      f"z {float(host['qpos'][2]):.3f}m, healthy={healthy}", flush=True)
            if self.cfg.evaluation.terminate_on_fall and not healthy:
                break
        arrays = {key: np.asarray(value) for key, value in log.items()}
        if not log:
            # Preserve an empty, correctly shaped prefix if the very first
            # update fails, rather than raising KeyError and losing the result.
            arrays = {key: np.empty((0,) + np.shape(value), dtype=np.asarray(value).dtype)
                      for key, value in host.items()}
            arrays["wall_step_seconds"] = np.empty(0)
        count = len(arrays["time"])
        distance = float(arrays["qpos"][-1, 0] - initial_x) if count else 0.0
        total_cost = float(arrays["cost_terms"].sum())
        fell = failure_reason is not None or (count > 0 and not bool(arrays["healthy"][-1]))
        if failure_reason is None and fell:
            failure_reason = "fall"
            failure_time = float(arrays["time"][-1])
        summary = {"seed": int(seed), "fitness": distance - total_cost,
                   "distance_x_m": distance, "total_running_cost": total_cost,
                   "integrated_running_cost": total_cost * self.cfg.robot.timestep,
                   "sim_seconds": float(arrays["time"][-1]) if count else 0.0, "wall_seconds": time.perf_counter() - start,
                   "updates": int(len(arrays["time"])), "fell": fell,
                   "failure_time_s": failure_time,
                   "failure_reason": failure_reason,
                   "failure_diagnostic": failure_diagnostic,
                   "mean_vx_m_s": float(arrays["body_velocity"][:, 0].mean()) if count else None,
                   "final_height_m": float(arrays["qpos"][-1, 2]) if count else initial_height,
                   "healthy_fraction": float(arrays["healthy"].mean()) if count else None,
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
