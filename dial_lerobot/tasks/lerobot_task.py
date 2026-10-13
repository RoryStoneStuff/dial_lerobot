"""The co_design humanoid cost expressed in LeRobot's joint coordinates."""

import jax.numpy as jnp
import numpy as np
from mujoco import mjx
from dial_lerobot.assets.foot_geometry import foot_capsules, capsule_bottoms
from hydrax.task_base import Task

from dial_lerobot.assets.model_builder import FOOT_SITES, JOINT_NAMES, initial_data
from dial_lerobot.config import CostConfig
from dial_lerobot.runtime import configure_jax_cache

COST_NAMES = ("gait", "joint_reference", "velocity", "upright", "yaw", "angular_velocity", "yaw_rate", "height", "foot_level", "foot_placement", "position", "hip_lateral", "contact", "contact_event", "air_time", "energy", "stance_slip", "alive", "com_support", "unhealthy")


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
        neutral = initial_data(mj_model)
        self.foot_geom_ids = foot_capsules(mj_model)
        self.foot_geom_sizes = jnp.asarray(mj_model.geom_size)[self.foot_geom_ids]
        neutral_bottoms = capsule_bottoms(mj_model, neutral, self.foot_geom_ids)
        self.rest_height = (float(neutral.qpos[2] - neutral_bottoms.min())
                            if cfg.rest_height is None else cfg.rest_height)
        self.healthy_z_min = (self.rest_height * cfg.source_healthy_z_min / cfg.source_rest_height
                              if cfg.healthy_z_min is None else cfg.healthy_z_min)
        self.foot_sites = jnp.array([mj_model.site(s).id for s in FOOT_SITES])
        # Marker-to-collision offset excludes any configured starting clearance.
        self.foot_height_offset = jnp.asarray(
            neutral.site_xpos[np.asarray(self.foot_sites), 2] - neutral_bottoms.min(axis=1))
        self.foot_bodies = mj_model.site_bodyid[np.asarray(self.foot_sites)]
        self.foot_roots = mj_model.body_rootid[self.foot_bodies]
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
        self.torque_limits = jnp.asarray(mj_model.actuator_forcerange[:, 1])
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
        # apply a constant torque at action=0, so zero now means the LeRobot neutral pose.
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

    def gait_phase(self, time):
        return jnp.mod(jnp.maximum(time - self.cfg.gait_start_delay, 0.0)
                       * self.cfg.cadence + jnp.array([0.0, 0.5]), 1.0)

    def stance_mask(self, time):
        c = self.cfg
        standing = c.swing_height == 0 and c.hip_amplitude == 0 and c.knee_amplitude == 0
        # H1's get_foot_step places the swing window symmetrically around
        # phase 0.5. Stance therefore wraps across phase 0 instead of being
        # the initial [0, duty) interval. Keep scheduled stance semantics in
        # sync with the foot-height target and auxiliary terms.
        phase = self.gait_phase(time)
        swing_start = 0.5 * c.duty
        swing_end = 1.0 - swing_start
        swing = (phase > swing_start) & (phase < swing_end)
        return (~swing) | (time < c.gait_start_delay) | standing

    def gait_reference(self, time):
        c = self.cfg
        phase = self.gait_phase(time)
        stance = self.stance_mask(time).astype(jnp.float32)
        swing_frac = jnp.clip(
            (phase - 0.5 * c.duty) / max(1.0 - c.duty, 1e-6), 0.0, 1.0
        )
        # H1 get_foot_step's cosine arch, expressed over LeRobot's gait phase.
        swing = (1 - stance) * jnp.sin(jnp.pi * swing_frac)
        foot_z = c.swing_height * swing
        # qstand values use the source/Berkeley convention; translate their
        # signs into the zoo CAD coordinates. H1's pose reward is disabled,
        # and LeRobot's joint-reference weight is zero in the H1 baseline,
        # but this target should be valid if tracking is enabled later.
        hip_stand = self.hip_sign * c.qstand_hip
        knee_stand = self.knee_sign * c.qstand_knee
        hip = hip_stand + jnp.where(
            time < c.gait_start_delay,
            0.0,
            self.hip_sign * c.hip_amplitude * jnp.sin(2 * jnp.pi * phase),
        )
        knee = knee_stand + self.knee_sign * c.knee_amplitude * swing
        return (foot_z, jnp.clip(hip, self.hip_bounds[:, 0], self.hip_bounds[:, 1]),
                jnp.clip(knee, self.knee_bounds[:, 0], self.knee_bounds[:, 1]))

    def body_velocity(self, state):
        inverse = state.qpos[3:7] * jnp.array([1.0, -1.0, -1.0, -1.0])
        return quat_rotate(inverse, state.qvel[:3])

    def target_velocity(self, time):
        ramp = jnp.clip(time / max(self.cfg.ramp_up_time, 1e-6), 0, 1)
        return ramp * jnp.array([self.cfg.target_vx, self.cfg.target_vy])

    def target_yaw_rate(self, time):
        ramp = jnp.clip(time / max(self.cfg.ramp_up_time, 1e-6), 0, 1)
        return ramp * self.cfg.target_wz

    def target_displacement(self, time):
        """Integral of the ramped command, in the world XY plane.

        The source task tracks a commanded root position in addition to
        velocity, but gives that term zero weight. This analytic integral
        avoids storing mutable task state inside vmap'd MJX rollouts.
        """
        ramp = max(self.cfg.ramp_up_time, 1e-6)
        t = jnp.maximum(time, 0.0)
        before = t * t / (2.0 * ramp)
        after = t - 0.5 * ramp
        elapsed = jnp.where(t < ramp, before, after)
        return elapsed * jnp.array([self.cfg.target_vx, self.cfg.target_vy])

    def body_angular_velocity(self, state):
        # MuJoCo free-joint qvel stores angular velocity in body coordinates.
        # Do not apply a second quaternion rotation here.
        return state.qvel[3:6]

    def yaw(self, state):
        w, x, y, z = state.qpos[3:7]
        return jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def up_vector(self, state):
        return quat_rotate(state.qpos[3:7], jnp.array([0.0, 0.0, 1.0]))

    def healthy(self, state):
        return (self.up_vector(state)[2] > self.cfg.healthy_up_dot) & (state.qpos[2] > self.healthy_z_min)

    @staticmethod
    def refresh_kinematics(model, state):
        # mjx.step integrates qpos/qvel after computing derived fields. Refresh
        # only position/velocity kinematics for post-step costs, without a
        # second contact solve. Use the SAME convention in planning/execution.
        return mjx.com_vel(model, mjx.com_pos(model, mjx.kinematics(model, state)))

    def foot_bottoms(self, state):
        return capsule_bottoms(self.model, state, self.foot_geom_ids, xp=jnp)

    def foot_velocity(self, state):
        # cvel is world-oriented, centered at the free body's subtree COM.
        spatial = jnp.asarray(state.cvel)[self.foot_bodies]
        lever = (jnp.asarray(state.site_xpos)[self.foot_sites]
                 - jnp.asarray(state.subtree_com)[self.foot_roots])
        return spatial[:, 3:] + jnp.cross(spatial[:, :3], lever)

    def com_support_error(self, state):
        # Squared horizontal COM excursion beyond actual foot collision bounds.
        # MuJoCo capsules extend along local z; project each capsule exactly.
        geom_mats = jnp.asarray(state.geom_xmat).reshape((-1, 3, 3))[self.foot_geom_ids]
        geom_xy = jnp.asarray(state.geom_xpos)[self.foot_geom_ids][..., :2]
        axis = geom_mats[..., :, 2]
        extents = (self.foot_geom_sizes[..., 0, None]
                   + self.foot_geom_sizes[..., 1, None] * jnp.abs(axis[..., :2]))
        lower = jnp.min(geom_xy - extents, axis=(0, 1)) - self.cfg.com_support_margin
        upper = jnp.max(geom_xy + extents, axis=(0, 1)) + self.cfg.com_support_margin
        com_xy = jnp.asarray(state.subtree_com)[0, :2]
        outside = jnp.maximum(lower - com_xy, 0.0) + jnp.maximum(com_xy - upper, 0.0)
        scale = jnp.maximum(self.cfg.com_support_scale, 1e-6)
        return jnp.sum((outside / scale) ** 2)

    def cost_terms(self, state, control):
        c = self.cfg
        foot_ref, hip_ref, knee_ref = self.gait_reference(state.time)
        bottoms = self.foot_bottoms(state)
        foot_z = jnp.min(bottoms, axis=1)
        stance = self.stance_mask(state.time)
        # Stance: reward a supported, approximately flat sole, with 2 mm of
        # tolerance for soft contacts. Swing: smoothly demand clearance only
        # as the reference lifts, so lift-off/touchdown do not have a 1 cm jump.
        stance_error = jnp.mean((jnp.maximum(jnp.abs(bottoms) - c.contact_tolerance, 0.0)
                                 / c.contact_height_scale) ** 2, axis=1)
        swing_error = (jnp.maximum(jnp.minimum(0.01, foot_ref) - foot_z, 0.0)
                       / c.contact_height_scale) ** 2
        stance_contact_error = jnp.where(stance, stance_error, swing_error)
        near_ground = foot_z <= c.contact_tolerance
        slip_error = jnp.sum(jnp.where(stance & near_ground,
            jnp.sum((self.foot_velocity(state)[:, :2] / c.stance_slip_scale) ** 2, axis=1), 0.0))
        gait_error = jnp.sum((foot_z - foot_ref)**2)
        ref_error = jnp.sum((state.qpos[self.hip_qadr] - hip_ref)**2
                            + (state.qpos[self.knee_qadr] - knee_ref)**2)
        velocity_error = jnp.sum((self.body_velocity(state)[:2] - self.target_velocity(state.time))**2)
        upright_error = jnp.sum((self.up_vector(state) - jnp.array([0.0, 0.0, 1.0]))**2)
        yaw_error = self.yaw(state) ** 2
        # Penalize torso pitch/roll rates as well as yaw. This damps the
        # forward-pitch failure mode of the lower-body LeRobot model.
        angular_velocity_error = jnp.sum(self.body_angular_velocity(state) ** 2)
        # Optional commanded body yaw rate. The existing angular-velocity
        # term remains, preserving the baseline when yaw_rate_weight is zero.
        yaw_rate_error = (self.body_angular_velocity(state)[2]
                          - self.target_yaw_rate(state.time)) ** 2
        height_error = (state.qpos[2] - self.rest_height)**2
        # Keep both soles level in world coordinates, matching the source H1
        # locomotion variant. The LeRobot task has only lower-body joints, so
        # this term is especially useful while the torso is being stabilized.
        foot_mats = jnp.reshape(jnp.asarray(state.site_xmat)[self.foot_sites], (-1, 3, 3))
        foot_up = foot_mats @ jnp.array([0.0, 0.0, 1.0])
        foot_level_error = jnp.sum((foot_up - jnp.array([0.0, 0.0, 1.0])) ** 2)
        # Cartesian forward placement target. During stance each foot is held
        # behind the body centre; during swing it moves smoothly to the front.
        # This supplies the missing horizontal stepping objective without
        # relying on a potentially mismatched joint-coordinate reference.
        inverse = state.qpos[3:7] * jnp.array([1.0, -1.0, -1.0, -1.0])
        foot_rel = jnp.asarray(state.site_xpos)[self.foot_sites] - state.qpos[:3]
        foot_forward = quat_rotate(inverse, foot_rel)[:, 0]
        step_length = jnp.abs(c.target_vx) / c.cadence
        phase = self.gait_phase(state.time)
        swing_fraction = jnp.clip(
            (phase - 0.5 * c.duty) / max(1.0 - c.duty, 1e-6), 0.0, 1.0
        )
        # Stance wraps through phase 0. Progress from swing touchdown at
        # phase 1-duty/2 through the wrap to swing lift-off at duty/2.
        stance_fraction = jnp.clip(
            jnp.mod(phase + 0.5 * c.duty, 1.0) / c.duty, 0.0, 1.0
        )
        # A planted foot moves from ahead of the body to behind it as the
        # body advances. Swing then brings it smoothly back to the front.
        # This makes the target continuous at both stance/swing boundaries.
        stance_target = 0.5 * step_length - step_length * stance_fraction
        swing_target = -0.5 * step_length + step_length * swing_fraction
        foot_forward_target = jnp.where(stance > 0, stance_target, swing_target)
        foot_placement_error = jnp.sum((foot_forward - foot_forward_target) ** 2)
        lateral_error = jnp.sum(state.qpos[self.lateral_qadr]**2)
        # Disabled root-position approximation and additional contact mismatch.
        # The ordinary source H1 walking reward has no contact-event penalty.
        # Position is measured in world XY; velocity is in the body frame.
        position_error = jnp.sum((state.qpos[:2] - self.target_displacement(state.time)) ** 2)
        near_ground = foot_z <= c.contact_tolerance
        contact_event_error = jnp.sum((near_ground != stance).astype(jnp.float32))
        phase = self.gait_phase(state.time)
        touchdown_phase = 1.0 - 0.5 * c.duty
        touchdown = stance & (jnp.abs(phase - touchdown_phase) < self.dt * c.cadence)
        scheduled_air_time = (1.0 - c.duty) / c.cadence
        # Placeholder only: this scheduled event is action-independent and is
        # NOT the source's measured flight-time reward. Keep its weight zero;
        # a faithful implementation needs per-foot history in the rollout carry.
        air_time_error = jnp.sum(jnp.where(
            touchdown, 0.1 - scheduled_air_time, 0.0))
        # Match UnitreeH1LocoEnv's normalized mechanical-power proxy:
        # (torque / torque_limit * joint_speed / 160)^2. HydraX stores the
        # clipped physical PD torque in ctrl; map joint speeds by name so the
        # zoo's actuator/qpos ordering cannot silently misalign this term.
        energy_error = jnp.sum(
            (state.ctrl / self.torque_limits * state.qvel[self.dof_indices] / 160.0) ** 2
        )
        alive_error = jnp.where(self.healthy(state), -1.0, 0.0)
        com_support_error = self.com_support_error(state)
        return jnp.stack((
            c.gait_weight * gait_error,
            c.ref_tracking_weight * ref_error,
            c.velocity_weight * velocity_error,
            c.upright_weight * upright_error,
            c.yaw_weight * yaw_error,
            c.angular_velocity_weight * angular_velocity_error,
            c.yaw_rate_weight * yaw_rate_error,
            c.height_weight * height_error,
            c.foot_level_weight * foot_level_error,
            c.foot_placement_weight * foot_placement_error,
            c.position_weight * position_error,
            c.hip_lateral_weight * lateral_error,
            c.contact_weight * jnp.sum(stance_contact_error),
            c.contact_event_weight * contact_event_error,
            c.air_time_weight * air_time_error,
            c.energy_weight * energy_error,
            c.stance_slip_weight * slip_error,
            c.alive_weight * alive_error,
            c.com_support_weight * com_support_error,
            jnp.where(self.healthy(state), 0.0, c.unhealthy_penalty),
        ))

    def running_cost(self, state, control):
        return jnp.sum(self.cost_terms(state, control))

    def terminal_cost(self, state):
        c = self.cfg
        upright_error = jnp.sum((self.up_vector(state) - jnp.array([0.0, 0.0, 1.0]))**2)
        # Un-ramped body-forward terminal velocity target.
        return (c.terminal_height_weight * (state.qpos[2] - self.rest_height)**2
                + c.terminal_velocity_weight * (self.body_velocity(state)[0] - c.target_vx)**2
                + c.terminal_upright_weight * upright_error)

    def describe(self):
        return {"rest_height_m": self.rest_height, "healthy_z_min_m": self.healthy_z_min,
                "phase_order": ["left", "right"],
                "contact_measurement": "capsule surfaces to world z=0 plane; geometric proximity, not force",
                "paper_terms": {"gait": "implemented H1-centered swing phase and lift curve", "air_time": "scheduled phase placeholder, NOT measured flight time; disabled at weight 0", "position": "implemented world-XY ramp integral; default weight 0", "upright": "implemented", "yaw": "implemented", "velocity": "implemented body frame", "angular_velocity": "implemented all axes", "height": "implemented", "foot_level": "implemented source-H1 flat-foot term", "foot_placement": "implemented body-frame periodic Cartesian target", "energy": "implemented H1-style normalized mechanical power", "contact": "additional LeRobot sole geometry penalty; absent from source H1 walking reward", "alive": "implemented; default weight 0", "com_support": "whole-body COM outside projected foot collision bounds; standing-only by default"},
                "foot_collision_geoms": [[self.mj_model.geom(int(g)).name for g in foot] for foot in self.foot_geom_ids],
                "foot_height_offset_m": np.asarray(self.foot_height_offset).tolist(),
                "action_joint_bounds": np.asarray(self.action_joint_bounds).tolist(),
                "action_neutral_q": [0.0] * len(JOINT_NAMES),
                "hip_sign": np.asarray(self.hip_sign).tolist(),
                "knee_sign": np.asarray(self.knee_sign).tolist(), "cost_terms": list(COST_NAMES)}
