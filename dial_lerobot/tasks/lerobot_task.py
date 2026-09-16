"""The co_design humanoid cost expressed in LeRobot's joint coordinates."""

import jax.numpy as jnp
import numpy as np
from hydrax.task_base import Task

from dial_lerobot.assets.model_builder import FOOT_SITES, JOINT_NAMES, initial_data
from dial_lerobot.config import CostConfig
from dial_lerobot.runtime import configure_jax_cache

COST_NAMES = ("gait", "joint_reference", "velocity", "upright", "yaw", "angular_velocity", "height", "hip_lateral", "contact", "energy", "unhealthy")


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
        # These sites are frame markers above the sole collision geometry.
        # Express gait height relative to their nominal neutral height so a
        # stance target of zero means "on the floor", not "inside the floor".
        self.foot_height_offset = jnp.asarray(initial_data(mj_model).site_xpos[self.foot_sites, 2])
        # Resolve names, since the zoo has right-first qpos and grouped actuators.
        self.hip_qadr = self._qadr(("hipy_left", "hipy_right"))
        self.knee_qadr = self._qadr(("knee_left", "knee_right"))
        self.lateral_qadr = self._qadr(("hipx_left", "hipz_left", "hipx_right", "hipz_right"))
        self.hip_bounds = self._bounds(("hipy_left", "hipy_right"))
        self.knee_bounds = self._bounds(("knee_left", "knee_right"))
        self.qpos_indices = jnp.array([self.mj_model.jnt_qposadr[self.mj_model.joint(n).id]
                                        for n in JOINT_NAMES])
        self.dof_indices = jnp.array([self.mj_model.jnt_dofadr[self.mj_model.joint(n).id]
                                      for n in JOINT_NAMES])
        # Official DIAL H1 torque mode: normalized action -> joint target -> PD torque.
        self.kp = jnp.array([200.0 if not n.startswith("ankle") else 60.0 for n in JOINT_NAMES])
        self.kd = jnp.array([5.0 if not n.startswith("ankle") else 1.5 for n in JOINT_NAMES])
        self.action_joint_bounds = self._bounds(JOINT_NAMES)
        self.u_min = -jnp.ones(len(JOINT_NAMES))
        self.u_max = jnp.ones(len(JOINT_NAMES))
        # Relative to Berkeley pitch coordinates: left hip flips, both knees flip.
        self.hip_sign = jnp.array([-1.0, 1.0])
        self.knee_sign = jnp.array([-1.0, -1.0])

    def _qadr(self, names):
        return jnp.array([self.mj_model.jnt_qposadr[self.mj_model.joint(n).id] for n in names])

    def _bounds(self, names):
        # Torque actuators expose Nm in actuator_ctrlrange, so gait references
        # must be clipped against the corresponding joint angle limits.
        bounds = []
        for name in names:
            jid = self.mj_model.joint(name).id
            limits = self.mj_model.jnt_range[jid]
            midpoint = limits.mean()
            bounds.append(midpoint + 0.9 * (limits - midpoint))
        return jnp.array(bounds)

    def action_to_control(self, state, action):
        """Map a normalized DIAL action to an H1-style clipped PD torque."""
        limits = self.action_joint_bounds
        # The zoo XML neutral pose is q=0, while knee/ankle physical ranges
        # are asymmetric. Centering the action on the range midpoint would
        # apply a constant torque at action=0, unlike the source H1 home pose.
        q_neutral = jnp.zeros_like(action)
        target = q_neutral + jnp.where(
            action >= 0.0,
            action * (limits[:, 1] - q_neutral),
            action * (q_neutral - limits[:, 0]),
        )
        q = state.qpos[self.qpos_indices]
        qd = state.qvel[self.dof_indices]
        torque = self.kp * (target - q) - self.kd * qd
        force_range = jnp.asarray(self.mj_model.actuator_forcerange)
        return jnp.clip(torque, force_range[:, 0], force_range[:, 1])

    def gait_reference(self, time):
        c = self.cfg
        phase = jnp.mod(time * c.cadence + jnp.array([0.0, 0.5]), 1.0)
        stance = (phase < c.duty).astype(jnp.float32)
        stance_frac = jnp.clip(phase / c.duty, 0.0, 1.0)
        swing_frac = jnp.clip((phase - c.duty) / (1 - c.duty), 0.0, 1.0)
        swing = (1 - stance) * jnp.sin(jnp.pi * swing_frac)
        foot_z = c.swing_height * swing
        # The LeRobot XML neutral pose is q=0 for every lower-body joint.
        # Use a zero-centered sagittal swing so the first MPC update does not
        # impose the source H1's nonzero home-pose offset.
        hip = self.hip_sign * c.hip_amplitude * jnp.sin(2 * jnp.pi * phase)
        knee = self.knee_sign * c.knee_amplitude * swing
        return (foot_z, jnp.clip(hip, self.hip_bounds[:, 0], self.hip_bounds[:, 1]),
                jnp.clip(knee, self.knee_bounds[:, 0], self.knee_bounds[:, 1]))

    def body_velocity(self, state):
        inverse = state.qpos[3:7] * jnp.array([1.0, -1.0, -1.0, -1.0])
        return quat_rotate(inverse, state.qvel[:3])

    def target_velocity(self, time):
        ramp = jnp.clip(time / max(self.cfg.ramp_up_time, 1e-6), 0, 1)
        return ramp * jnp.array([self.cfg.target_vx, self.cfg.target_vy])

    def body_angular_velocity(self, state):
        inverse = state.qpos[3:7] * jnp.array([1.0, -1.0, -1.0, -1.0])
        return quat_rotate(inverse, state.qvel[3:6])

    def yaw(self, state):
        w, x, y, z = state.qpos[3:7]
        return jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def up_vector(self, state):
        return quat_rotate(state.qpos[3:7], jnp.array([0.0, 0.0, 1.0]))

    def healthy(self, state):
        return (self.up_vector(state)[2] > self.cfg.healthy_up_dot) & (state.qpos[2] > self.healthy_z_min)

    def cost_terms(self, state, control):
        c = self.cfg
        foot_ref, hip_ref, knee_ref = self.gait_reference(state.time)
        foot_z = state.site_xpos[self.foot_sites, 2] - self.foot_height_offset
        phase = jnp.mod(state.time * c.cadence + jnp.array([0.0, 0.5]), 1.0)
        stance = phase < c.duty
        # The official H1 task uses contact distances. The LeRobot model has
        # reliable sole sites but no H1 contact layout, so use sole height for
        # an equivalent stance/swing contact consistency term.
        stance_contact_error = jnp.where(stance, foot_z**2, jnp.maximum(0.01 - foot_z, 0.0)**2)
        gait_error = jnp.sum((foot_z - foot_ref)**2)
        ref_error = jnp.sum((state.qpos[self.hip_qadr] - hip_ref)**2
                            + (state.qpos[self.knee_qadr] - knee_ref)**2)
        velocity_error = jnp.sum((self.body_velocity(state)[:2] - self.target_velocity(state.time))**2)
        upright_error = jnp.sum((self.up_vector(state) - jnp.array([0.0, 0.0, 1.0]))**2)
        yaw_error = self.yaw(state) ** 2
        angular_velocity_error = self.body_angular_velocity(state)[2] ** 2
        height_error = (state.qpos[2] - self.rest_height)**2
        lateral_error = jnp.sum(state.qpos[self.lateral_qadr]**2)
        # Penalize normalized command magnitude, as in the DIAL action space.
        energy_error = jnp.sum(control ** 2)
        return jnp.stack((
            c.gait_weight * gait_error,
            c.ref_tracking_weight * ref_error,
            c.velocity_weight * velocity_error,
            c.upright_weight * upright_error,
            c.yaw_weight * yaw_error,
            c.angular_velocity_weight * angular_velocity_error,
            c.height_weight * height_error,
            c.hip_lateral_weight * lateral_error,
            c.contact_weight * jnp.sum(stance_contact_error),
            c.energy_weight * energy_error,
            jnp.where(self.healthy(state), 0.0, c.unhealthy_penalty),
        ))

    def running_cost(self, state, control):
        return jnp.sum(self.cost_terms(state, control))

    def terminal_cost(self, state):
        c = self.cfg
        upright_error = jnp.sum((self.up_vector(state) - jnp.array([0.0, 0.0, 1.0]))**2)
        # Preserve the source's un-ramped WORLD-x terminal velocity target.
        return (c.terminal_height_weight * (state.qpos[2] - self.rest_height)**2
                + c.terminal_velocity_weight * (self.body_velocity(state)[0] - c.target_vx)**2
                + c.terminal_upright_weight * upright_error)

    def describe(self):
        return {"rest_height_m": self.rest_height, "healthy_z_min_m": self.healthy_z_min,
                "phase_order": ["left", "right"],
                "foot_height_offset_m": np.asarray(self.foot_height_offset).tolist(),
                "action_joint_bounds": np.asarray(self.action_joint_bounds).tolist(),
                "action_neutral_q": [0.0] * len(JOINT_NAMES),
                "hip_sign": np.asarray(self.hip_sign).tolist(),
                "knee_sign": np.asarray(self.knee_sign).tolist(), "cost_terms": list(COST_NAMES)}
