"""The co_design humanoid cost expressed in LeRobot's joint coordinates."""

import jax.numpy as jnp
import numpy as np
from hydrax.task_base import Task

from dial_lerobot.assets.model_builder import FOOT_SITES, initial_data
from dial_lerobot.config import CostConfig
from dial_lerobot.runtime import configure_jax_cache

COST_NAMES = ("gait", "joint_reference", "velocity", "upright", "height", "hip_lateral", "unhealthy")


def quat_rotate(q, v):
    uv = jnp.cross(q[1:], v)
    return v + 2 * (q[0] * uv + jnp.cross(q[1:], uv))


class LeRobotLocomotionTask(Task):
    def __init__(self, mj_model, cfg: CostConfig = CostConfig()):
        # HydraX has already imported and set its global cache default. Override
        # that before Task.__init__ constructs MJX models/JAX arrays.
        self.cache_dir = configure_jax_cache()
        super().__init__(mj_model, trace_sites=FOOT_SITES)
        self.cfg = cfg
        self.rest_height = (float(initial_data(mj_model).qpos[2])
                            if cfg.rest_height is None else cfg.rest_height)
        self.healthy_z_min = (self.rest_height * cfg.source_healthy_z_min / cfg.source_rest_height
                              if cfg.healthy_z_min is None else cfg.healthy_z_min)
        self.foot_sites = jnp.array([mj_model.site(s).id for s in FOOT_SITES])
        # Resolve names, since the zoo has right-first qpos and grouped actuators.
        self.hip_qadr = self._qadr(("hipy_left", "hipy_right"))
        self.knee_qadr = self._qadr(("knee_left", "knee_right"))
        self.lateral_qadr = self._qadr(("hipx_left", "hipz_left", "hipx_right", "hipz_right"))
        self.hip_bounds = self._bounds(("hipy_left", "hipy_right"))
        self.knee_bounds = self._bounds(("knee_left", "knee_right"))
        # Relative to Berkeley pitch coordinates: left hip flips, both knees flip.
        self.hip_sign = jnp.array([-1.0, 1.0])
        self.knee_sign = jnp.array([-1.0, -1.0])

    def _qadr(self, names):
        return jnp.array([self.mj_model.jnt_qposadr[self.mj_model.joint(n).id] for n in names])

    def _bounds(self, names):
        return jnp.array([self.mj_model.actuator_ctrlrange[self.mj_model.actuator(n).id] for n in names])

    def gait_reference(self, time):
        c = self.cfg
        phase = jnp.mod(time * c.cadence + jnp.array([0.0, 0.5]), 1.0)
        stance = (phase < c.duty).astype(jnp.float32)
        stance_frac = jnp.clip(phase / c.duty, 0.0, 1.0)
        swing_frac = jnp.clip((phase - c.duty) / (1 - c.duty), 0.0, 1.0)
        swing = (1 - stance) * jnp.sin(jnp.pi * swing_frac)
        foot_z = c.swing_height * swing
        hip_berkeley = c.qstand_hip + jnp.where(
            stance > 0, -c.hip_amplitude * jnp.cos(jnp.pi * stance_frac),
            c.hip_amplitude * jnp.cos(jnp.pi * swing_frac))
        knee_berkeley = c.qstand_knee + c.knee_amplitude * swing
        # Source stand posture is the origin of the transform: zoo CAD zero
        # already bends the legs. Do not command Berkeley's +0.4 knee angle.
        hip = self.hip_sign * (hip_berkeley - (-0.20))
        knee = self.knee_sign * (knee_berkeley - 0.40)
        return (foot_z, jnp.clip(hip, self.hip_bounds[:, 0], self.hip_bounds[:, 1]),
                jnp.clip(knee, self.knee_bounds[:, 0], self.knee_bounds[:, 1]))

    def body_velocity(self, state):
        inverse = state.qpos[3:7] * jnp.array([1.0, -1.0, -1.0, -1.0])
        return quat_rotate(inverse, state.qvel[:3])

    def target_velocity(self, time):
        ramp = jnp.clip(time / max(self.cfg.ramp_up_time, 1e-6), 0, 1)
        return ramp * jnp.array([self.cfg.target_vx, self.cfg.target_vy])

    def up_vector(self, state):
        return quat_rotate(state.qpos[3:7], jnp.array([0.0, 0.0, 1.0]))

    def healthy(self, state):
        return (self.up_vector(state)[2] > self.cfg.healthy_up_dot) & (state.qpos[2] > self.healthy_z_min)

    def cost_terms(self, state, control):
        del control  # The source task has no effort/control penalty.
        c = self.cfg
        foot_ref, hip_ref, knee_ref = self.gait_reference(state.time)
        # Flat-ground clearance from the existing sole sites stays meaningful
        # during swing; contact.dist is absent once a foot leaves the floor.
        gait_error = jnp.sum((state.site_xpos[self.foot_sites, 2] - foot_ref)**2)
        ref_error = jnp.sum((state.qpos[self.hip_qadr] - hip_ref)**2
                            + (state.qpos[self.knee_qadr] - knee_ref)**2)
        velocity_error = jnp.sum((self.body_velocity(state)[:2] - self.target_velocity(state.time))**2)
        upright_error = jnp.sum((self.up_vector(state) - jnp.array([0.0, 0.0, 1.0]))**2)
        return jnp.stack((
            c.gait_weight * gait_error,
            c.ref_tracking_weight * ref_error,
            c.velocity_weight * velocity_error,
            c.upright_weight * upright_error,
            c.height_weight * (state.qpos[2] - self.rest_height)**2,
            c.hip_lateral_weight * jnp.sum(state.qpos[self.lateral_qadr]**2),
            jnp.where(self.healthy(state), 0.0, c.unhealthy_penalty),
        ))

    def running_cost(self, state, control):
        return jnp.sum(self.cost_terms(state, control))

    def terminal_cost(self, state):
        c = self.cfg
        upright_error = jnp.sum((self.up_vector(state) - jnp.array([0.0, 0.0, 1.0]))**2)
        # Preserve the source's un-ramped WORLD-x terminal velocity target.
        return (c.terminal_height_weight * (state.qpos[2] - self.rest_height)**2
                + c.terminal_velocity_weight * (state.qvel[0] - c.target_vx)**2
                + c.terminal_upright_weight * upright_error)

    def describe(self):
        return {"rest_height_m": self.rest_height, "healthy_z_min_m": self.healthy_z_min,
                "phase_order": ["left", "right"], "hip_sign": np.asarray(self.hip_sign).tolist(),
                "knee_sign": np.asarray(self.knee_sign).tolist(), "cost_terms": list(COST_NAMES)}
